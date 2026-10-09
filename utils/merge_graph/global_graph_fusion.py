import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from llms import LLMClient
from utils.utils import parse_json_response

from utils.merge_graph.global_graph_builder import (
    GENERIC_SERVICE_MENTIONS,
    add_edge,
    edge_key,
    global_endpoint,
    normalize_text,
    occurrence_endpoint,
    singularize_simple,
)

logger = logging.getLogger(__name__)


FUSION_SYSTEM_PROMPT = """
You decide whether two already-built global graph candidates represent the SAME
canonical entity, concept, or action.

Important:
- SAME means semantic identity / true synonymy at the same abstraction level.
- Do NOT mark a subtype as SAME with its parent. "Personal User" is not the same
  canonical concept as "User". "ERNIE Bot service" is not the same canonical
  entity as generic "Service".
- Do NOT merge different named entities or different named services.
- Modifiers that narrow a category normally indicate a subtype, not identity.
- Surface variants that genuinely denote the same category may be SAME, e.g.
  "others" and "other users" in a user-role context.
- RelationSignature and Modifiers are semantic constraints, not decorative text.
- Return false when uncertain.

Return JSON only.
""".strip()


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> bool:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True


def _usage_snapshot(llm_client: Any) -> Any:
    getter = getattr(llm_client, "get_usage", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


def _usage_since(llm_client: Any, before: Any) -> Dict[str, int]:
    getter = getattr(llm_client, "get_usage_since", None)
    if before is not None and callable(getter):
        try:
            usage = getter(before)
            if isinstance(usage, dict):
                return usage
        except Exception:
            pass
    return {}


def _normalized_modifiers(record: Dict) -> Tuple[str, ...]:
    values = []
    for modifier in record.get("Modifiers", []):
        normalized = normalize_text(modifier)
        if normalized:
            values.append(normalized)
    return tuple(sorted(set(values)))


def _person_alias_key(record: Dict) -> Optional[str]:
    text = normalize_text(record.get("Text"))
    head = singularize_simple(record.get("Head"))
    modifiers = set(_normalized_modifiers(record))

    if text in {"others", "other users", "other user", "other people", "other persons"}:
        return "other user"
    if head in {"user", "person", "other user"} and "other" in modifiers:
        return "other user"
    if text in {"user", "users"} or head == "user" and not modifiers:
        return "user"
    return None


def _deterministic_identity_key(record: Dict) -> Optional[Tuple]:
    if not record.get("CanonicalEligible"):
        return None
    fusion_kind = record.get("FusionKind")
    if fusion_kind not in {"Entity", "Concept", "Action"}:
        return None

    text = normalize_text(record.get("Text"))
    head = singularize_simple(record.get("Head"))
    modifiers = _normalized_modifiers(record)

    if fusion_kind == "Entity":
        entity_type = record.get("EntityType") or "Entity"
        if entity_type == "Organization":
            if not text:
                return None
            return ("Entity", "Organization", text)
        if entity_type == "Service":
            if not text:
                return None
            service_key = re.sub(r"^(?:the|a|an)\s+", "", text).strip()
            # Service identity is issuer-scoped. Article-only surface variation is
            # deterministic identity, but different named services never merge.
            return (
                "Entity",
                "Service",
                normalize_text(record.get("Organization")),
                service_key,
            )
        if text:
            return ("Entity", entity_type, text)
        return None

    if fusion_kind == "Action":
        action = normalize_text(record.get("Head")) or text
        return ("Action", action) if action else None

    person_alias = _person_alias_key(record)
    if person_alias:
        return ("Concept", person_alias)

    if not text:
        return None
    if record.get("HasRelations"):
        return ("ConceptText", text)

    if head:
        return ("ConceptPhrase", head, modifiers)
    return ("ConceptText", text)


def _eligible_occurrences(graph: Dict) -> Dict[str, Dict]:
    return {
        occurrence_id: record
        for occurrence_id, record in graph.get("Occurrences", {}).items()
        if isinstance(record, dict)
        and record.get("CanonicalEligible")
        and record.get("FusionKind") in {"Entity", "Concept", "Action"}
    }


def _deterministic_union(records: Dict[str, Dict], uf: _UnionFind) -> int:
    groups: Dict[Tuple, List[str]] = defaultdict(list)
    for occurrence_id, record in records.items():
        key = _deterministic_identity_key(record)
        if key is not None:
            groups[key].append(occurrence_id)

    merged = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        first = members[0]
        for member in members[1:]:
            if uf.union(first, member):
                merged += 1
    return merged


def _cluster_members(records: Dict[str, Dict], uf: _UnionFind) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for occurrence_id in records:
        groups[uf.find(occurrence_id)].append(occurrence_id)
    for members in groups.values():
        members.sort()
    return dict(groups)


def _choose_text(member_records: List[Dict]) -> str:
    texts = [
        str(record.get("Text") or "").strip()
        for record in member_records
        if str(record.get("Text") or "").strip()
    ]
    if not texts:
        return ""
    counts = Counter(normalize_text(text) for text in texts)

    def score(text: str) -> Tuple[int, int, int, str]:
        normalized = normalize_text(text)
        tokens = normalized.split()
        generic_penalty = (
            1
            if normalized in GENERIC_SERVICE_MENTIONS
            or normalized in {"this", "that", "it"}
            else 0
        )
        return (
            -generic_penalty,
            min(len(tokens), 6),
            counts.get(normalized, 0),
            -len(text),
        )

    return max(texts, key=score)


def _cluster_summary(records: Dict[str, Dict], members: List[str]) -> Dict:
    member_records = [records[item] for item in members]
    fusion_kinds = {record.get("FusionKind") for record in member_records}
    entity_types = {
        record.get("EntityType")
        for record in member_records
        if record.get("EntityType")
    }
    kind = next(iter(fusion_kinds)) if len(fusion_kinds) == 1 else "Mixed"
    entity_type = next(iter(entity_types)) if len(entity_types) == 1 else None
    text = _choose_text(member_records)

    heads = [
        record.get("Head")
        for record in member_records
        if normalize_text(record.get("Head"))
    ]
    head = (
        Counter(normalize_text(value) for value in heads).most_common(1)[0][0]
        if heads
        else ""
    )
    modifiers = sorted(
        {
            normalize_text(modifier)
            for record in member_records
            for modifier in record.get("Modifiers", [])
            if normalize_text(modifier)
        }
    )
    person_keys = {
        key for key in (_person_alias_key(record) for record in member_records) if key
    }
    if person_keys == {"other user"}:
        head = "user"
        if "other" not in modifiers:
            modifiers.append("other")
    elif person_keys == {"user"}:
        head = "user"

    aliases = sorted(
        {
            str(record.get("Text") or "").strip()
            for record in member_records
            if str(record.get("Text") or "").strip()
        }
    )
    if kind == "Action":
        # Canonical action is the base action concept. Prefer an actual member
        # surface that exactly equals the action head; otherwise use the head.
        exact = [
            alias for alias in aliases if normalize_text(alias) == normalize_text(head)
        ]
        text = exact[0] if exact else (head or text)
    # Concept / Entity display text always comes from a real member surface.
    # Normalization is for matching only; do not manufacture labels such as "heroe".

    organizations = sorted(
        {
            str(record.get("Organization") or "")
            for record in member_records
            if str(record.get("Organization") or "")
        }
    )
    source_ids = sorted(
        {
            str(record.get("SourceID") or "")
            for record in member_records
            if str(record.get("SourceID") or "")
        }
    )
    has_relations = any(bool(record.get("HasRelations")) for record in member_records)
    relation_signature = sorted(
        {
            signature
            for record in member_records
            for signature in record.get("_RelationSignature", [])
            if isinstance(signature, str) and signature
        }
    )

    return {
        "Members": members,
        "Kind": kind,
        "EntityType": entity_type,
        "Text": text,
        "Head": head,
        "Modifiers": modifiers,
        "Aliases": aliases,
        "Organizations": organizations,
        "SourceIDs": source_ids,
        "HasRelations": has_relations,
        "_RelationSignature": relation_signature,
    }


def _token_set(text: Any) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if token not in {"the", "a", "an"}
    }


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_MODIFIER_NOISE = {
    "the",
    "a",
    "an",
    "any",
    "some",
    "such",
    "this",
    "that",
    "our",
    "your",
    "their",
    "its",
    "current",
    "same",
}


def _modifier_atoms(node: Dict) -> Set[str]:
    atoms: Set[str] = set()
    for modifier in node.get("Modifiers", []):
        for token in re.findall(r"[a-z0-9]+", normalize_text(modifier)):
            if token not in _MODIFIER_NOISE:
                atoms.add(singularize_simple(token))
    return atoms


def _relation_set(node: Dict) -> Set[str]:
    return {
        str(value)
        for value in node.get("_RelationSignature", [])
        if isinstance(value, str) and value
    }


def _normalized_word_set(text: Any) -> Set[str]:
    return {
        singularize_simple(token)
        for token in _token_set(text)
        if singularize_simple(token)
    }


def _is_simple_parent(node: Dict) -> bool:
    if _relation_set(node):
        return False
    head = singularize_simple(node.get("Head") or node.get("Text"))
    if not head:
        return False
    text_tokens = _normalized_word_set(node.get("Text"))
    allowed = _normalized_word_set(head)
    allowed |= _modifier_atoms(node)
    return bool(text_tokens) and text_tokens <= allowed


def _deterministic_subtype_direction(left: Dict, right: Dict) -> Optional[str]:
    """Return 'left' when left<:right, 'right' when right<:left, else None."""
    if left.get("Kind") != "Concept" or right.get("Kind") != "Concept":
        return None
    left_head = singularize_simple(left.get("Head") or left.get("Text"))
    right_head = singularize_simple(right.get("Head") or right.get("Text"))
    if not left_head or left_head != right_head:
        return None

    left_mods = _modifier_atoms(left)
    right_mods = _modifier_atoms(right)
    left_rel = _relation_set(left)
    right_rel = _relation_set(right)

    # Modifier narrowing: Personal User <: User, Other User <: User.
    if left_mods > right_mods and _is_simple_parent(right) and right_rel <= left_rel:
        return "left"
    if right_mods > left_mods and _is_simple_parent(left) and left_rel <= right_rel:
        return "right"

    # A simple generic concept may be narrowed by an explicit relation.
    if left_mods == right_mods:
        if left_rel and not right_rel and _is_simple_parent(right):
            return "left"
        if right_rel and not left_rel and _is_simple_parent(left):
            return "right"
    return None


def _relations_clearly_conflict(left: Dict, right: Dict) -> bool:
    left_rel = _relation_set(left)
    right_rel = _relation_set(right)
    if not left_rel and not right_rel:
        return False
    if bool(left_rel) != bool(right_rel):
        return True
    if left_rel == right_rel:
        return False
    # Different relation structures are not candidates for SAME merely because
    # they share a head such as "content".
    return _jaccard(set(left_rel), set(right_rel)) < 0.75


def _modifiers_clearly_conflict(left: Dict, right: Dict) -> bool:
    left_mods = _modifier_atoms(left)
    right_mods = _modifier_atoms(right)
    if not left_mods or not right_mods:
        return False
    # Explicit "other" role vs a differently qualified role is normally a
    # sibling distinction, not synonymy (e.g. Other User vs Personal User).
    if ("other" in left_mods) != ("other" in right_mods):
        return True
    return False


def _semantically_close_concept(left: Dict, right: Dict) -> bool:
    left_head = singularize_simple(left.get("Head") or left.get("Text"))
    right_head = singularize_simple(right.get("Head") or right.get("Text"))
    if not left_head or left_head != right_head:
        return False

    if _relations_clearly_conflict(left, right) or _modifiers_clearly_conflict(
        left, right
    ):
        return False

    left_mods = _modifier_atoms(left)
    right_mods = _modifier_atoms(right)
    relation_equal = _relation_set(left) == _relation_set(right)
    similarity = _jaccard(_token_set(left.get("Text")), _token_set(right.get("Text")))

    # Same structural signature but surface wording remains ambiguous.
    if left_mods == right_mods:
        return relation_equal and similarity >= 0.60

    # Neither side is a deterministic subtype. Small competing modifier sets on
    # the same head are the main LLM case: personal information vs private information.
    if (
        left_mods
        and right_mods
        and not (left_mods < right_mods or right_mods < left_mods)
    ):
        if len(left_mods) <= 2 and len(right_mods) <= 2 and relation_equal:
            return similarity >= (1.0 / 3.0)

    # Jaccard is only a final auxiliary recall signal, never a primary rule.
    return relation_equal and similarity >= 0.75


def _candidate_pair(left: Dict, right: Dict) -> bool:
    # 1) Type compatibility.
    if left.get("Kind") != right.get("Kind"):
        return False

    # Entity and Action identity are deterministic in v1. LLM canonical fusion is
    # reserved for genuinely ambiguous Concept pairs. Reference/coreference resolution
    # is completed earlier by enrich_policy_graph and is never redone globally.
    if left.get("Kind") != "Concept":
        return False

    # 2) Deterministic SAME has already been unioned before this function.
    # 3) Deterministic SUBTYPE must not be sent to SAME/NOT-SAME LLM judgment.
    if _deterministic_subtype_direction(left, right) is not None:
        return False

    # 4) Clear structural/modifier conflicts are rejected, and only the remaining
    # very-close ambiguous Concept pairs become LLM candidates.
    return _semantically_close_concept(left, right)


def _fusion_tasks(records: Dict[str, Dict], uf: _UnionFind) -> List[Dict]:
    clusters = _cluster_members(records, uf)
    summaries = [
        (root, _cluster_summary(records, members))
        for root, members in sorted(clusters.items())
    ]
    tasks: List[Dict] = []
    counter = 0
    for index, (left_root, left) in enumerate(summaries):
        for right_root, right in summaries[index + 1 :]:
            if not _candidate_pair(left, right):
                continue
            tasks.append(
                {
                    "id": f"fusion_{counter}",
                    "left_root": left_root,
                    "right_root": right_root,
                    "Left": {
                        **{
                            key: value
                            for key, value in left.items()
                            if key not in {"Members", "_RelationSignature"}
                        },
                        "RelationSignature": left.get("_RelationSignature", []),
                    },
                    "Right": {
                        **{
                            key: value
                            for key, value in right.items()
                            if key not in {"Members", "_RelationSignature"}
                        },
                        "RelationSignature": right.get("_RelationSignature", []),
                    },
                }
            )
            counter += 1
    return tasks


def _fusion_prompt(tasks: List[Dict]) -> str:
    payload = [
        {
            "id": task["id"],
            "Left": task["Left"],
            "Right": task["Right"],
        }
        for task in tasks
    ]
    return f"""
For every pair, decide whether Left and Right are the SAME canonical item.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return exactly:
[
  {{"id": "fusion_0", "Same": true}},
  {{"id": "fusion_1", "Same": false}}
]
""".strip()


def _parse_fusion_response(response: str) -> Dict[str, bool]:
    parsed = parse_json_response(response)
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        parsed = parsed["results"]
    if isinstance(parsed, dict) and "id" in parsed:
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Global fusion response must be a list.")
    out: Dict[str, bool] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        task_id = item.get("id")
        same = item.get("Same")
        if isinstance(task_id, str) and isinstance(same, bool):
            out[task_id] = same
    return out


def _resolve_fusion_tasks(
    llm_client: Optional[LLMClient],
    tasks: List[Dict],
    uf: _UnionFind,
    *,
    batch_size: int,
    max_retries: int,
) -> Tuple[int, int]:
    if llm_client is None or not tasks:
        return 0, len(tasks)

    merged = 0
    unresolved = 0
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    logger.info(
        "GlobalFusion | llm start | tasks=%d | batch_size=%d | batches=%d",
        len(tasks),
        batch_size,
        total_batches,
    )

    for start in range(0, len(tasks), batch_size):
        batch_index = start // batch_size + 1
        remaining = tasks[start : start + batch_size]
        logger.info(
            "GlobalFusion | batch start | batch=%d/%d | tasks=%d",
            batch_index,
            total_batches,
            len(remaining),
        )

        for attempt in range(1, max_retries + 1):
            if not remaining:
                break
            before_attempt = len(remaining)
            usage_before_call = _usage_snapshot(llm_client)
            try:
                response = llm_client.invoke(
                    _fusion_prompt(remaining),
                    system_prompt=FUSION_SYSTEM_PROMPT,
                )
                if response is None or (
                    isinstance(response, str) and not response.strip()
                ):
                    raise ValueError("Empty global fusion response.")
                parsed = _parse_fusion_response(response)
                retry: List[Dict] = []
                merged_this_attempt = 0
                for task in remaining:
                    if task["id"] not in parsed:
                        retry.append(task)
                        continue
                    if parsed[task["id"]]:
                        if uf.union(task["left_root"], task["right_root"]):
                            merged += 1
                            merged_this_attempt += 1
                remaining = retry
                usage = _usage_since(llm_client, usage_before_call)
                logger.info(
                    "GlobalFusion | batch done | batch=%d/%d | attempt=%d/%d | "
                    "submitted=%d | answered=%d | merged=%d | remaining=%d | "
                    "api_calls=%s | input_tokens=%s | output_tokens=%s | total_tokens=%s",
                    batch_index,
                    total_batches,
                    attempt,
                    max_retries,
                    before_attempt,
                    before_attempt - len(remaining),
                    merged_this_attempt,
                    len(remaining),
                    usage.get("api_calls", "?"),
                    usage.get("input_tokens", "?"),
                    usage.get("output_tokens", "?"),
                    usage.get("total_tokens", "?"),
                )
            except Exception as exc:
                usage = _usage_since(llm_client, usage_before_call)
                logger.warning(
                    "GlobalFusion | retry | batch=%d/%d | attempt=%d/%d | "
                    "tasks=%d | api_calls=%s | input_tokens=%s | output_tokens=%s | "
                    "total_tokens=%s | error=%s",
                    batch_index,
                    total_batches,
                    attempt,
                    max_retries,
                    before_attempt,
                    usage.get("api_calls", "?"),
                    usage.get("input_tokens", "?"),
                    usage.get("output_tokens", "?"),
                    usage.get("total_tokens", "?"),
                    exc,
                )

        unresolved += len(remaining)

    logger.info(
        "GlobalFusion | llm completed | tasks=%d | merged=%d | unresolved=%d",
        len(tasks),
        merged,
        unresolved,
    )
    return merged, unresolved


def _assign_canonical_nodes(
    graph: Dict, records: Dict[str, Dict], uf: _UnionFind
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    clusters = _cluster_members(records, uf)
    summaries = [_cluster_summary(records, members) for members in clusters.values()]
    summaries.sort(
        key=lambda item: (
            str(item.get("Kind") or ""),
            str(item.get("EntityType") or ""),
            normalize_text(item.get("Text")),
            item.get("Members", [""])[0],
        )
    )

    occurrence_to_canonical: Dict[str, str] = {}
    canonical_members: Dict[str, List[str]] = {}

    for index, summary in enumerate(summaries):
        global_id = f"g{index}"
        members = list(summary.pop("Members"))
        graph["CanonicalNodes"][global_id] = {
            "ID": global_id,
            **summary,
            "MemberCount": len(members),
        }
        canonical_members[global_id] = members
        for occurrence_id in members:
            occurrence_to_canonical[occurrence_id] = global_id
            add_edge(
                graph,
                {
                    "Type": "CANONICAL_OF",
                    "Source": occurrence_endpoint(occurrence_id),
                    "Target": global_endpoint(global_id),
                    "Origin": "global_fusion",
                },
            )

    graph["_occurrence_to_canonical"] = occurrence_to_canonical
    graph["_canonical_members"] = canonical_members
    return occurrence_to_canonical, canonical_members


def _assign_relation_action_canonicals(graph: Dict) -> int:
    """Attach Relation predicates to the same deterministic Action canonical space."""
    nodes = graph.get("CanonicalNodes", {})
    occurrences = graph.get("Occurrences", {})
    by_head = {
        normalize_text(node.get("Head") or node.get("Text")): global_id
        for global_id, node in nodes.items()
        if isinstance(node, dict)
        and node.get("Kind") == "Action"
        and normalize_text(node.get("Head") or node.get("Text"))
    }
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for edge in graph.get("Edges", []):
        if isinstance(edge, dict) and edge.get("Type") == "RELATION":
            head = normalize_text(edge.get("ActionHead"))
            if head:
                groups[head].append(edge)

    next_index = (
        max(
            (
                int(global_id[1:])
                for global_id in nodes
                if re.fullmatch(r"g\d+", str(global_id))
            ),
            default=-1,
        )
        + 1
    )
    created = 0
    for head in sorted(groups):
        edges = groups[head]
        global_id = by_head.get(head)
        if global_id is None:
            global_id = f"g{next_index}"
            next_index += 1
            by_head[head] = global_id
            owner_records = [
                occurrences.get(edge.get("Owner", {}).get("Occurrence"), {})
                for edge in edges
                if isinstance(edge.get("Owner"), dict)
            ]
            display_head = str(edges[0].get("ActionHead") or "").strip() or head
            nodes[global_id] = {
                "ID": global_id,
                "Kind": "Action",
                "EntityType": None,
                "Text": display_head,
                "Head": head,
                "Modifiers": [],
                "Aliases": sorted(
                    {str(edge.get("Predicate") or "").strip() for edge in edges}
                ),
                "Organizations": sorted(
                    {
                        str(r.get("Organization"))
                        for r in owner_records
                        if r.get("Organization")
                    }
                ),
                "SourceIDs": sorted(
                    {str(r.get("SourceID")) for r in owner_records if r.get("SourceID")}
                ),
                "HasRelations": False,
                "_RelationSignature": [],
                "MemberCount": 0,
            }
            created += 1
        for edge in edges:
            edge["ActionCanonical"] = global_id

    graph["_edge_keys"] = {
        edge_key(edge) for edge in graph.get("Edges", []) if isinstance(edge, dict)
    }
    return created


def _canonical_modifiers(node: Dict) -> Set[str]:
    return _modifier_atoms(node)


def _build_subtype_edges(graph: Dict) -> int:
    nodes = graph.get("CanonicalNodes", {})
    added = 0
    ids = sorted(nodes)

    for child_id in ids:
        child = nodes[child_id]
        if child.get("Kind") not in {"Concept", "Entity"}:
            continue

        for parent_id in ids:
            if child_id == parent_id:
                continue
            parent = nodes[parent_id]
            if parent.get("Kind") != "Concept":
                continue

            is_subtype = False
            reason = None

            if child.get("Kind") == "Entity":
                entity_type = child.get("EntityType")
                parent_head = singularize_simple(
                    parent.get("Head") or parent.get("Text")
                )
                parent_text = singularize_simple(parent.get("Text"))
                if (
                    entity_type == "Service"
                    and parent_head == "service"
                    and parent_text == "service"
                ):
                    is_subtype = True
                    reason = "named_service_to_service"
                elif (
                    entity_type == "Organization"
                    and parent_head == "organization"
                    and parent_text == "organization"
                ):
                    is_subtype = True
                    reason = "named_organization_to_organization"
            else:
                direction = _deterministic_subtype_direction(child, parent)
                if direction == "left":
                    is_subtype = True
                    child_mods = _modifier_atoms(child)
                    parent_mods = _modifier_atoms(parent)
                    child_rel = _relation_set(child)
                    parent_rel = _relation_set(parent)
                    if child_mods > parent_mods:
                        reason = "modifier_narrowing"
                    elif child_rel and not parent_rel:
                        reason = "relation_narrowing"
                    else:
                        reason = "deterministic_narrowing"

            if not is_subtype:
                continue
            if add_edge(
                graph,
                {
                    "Type": "SUBTYPE_OF",
                    "Source": global_endpoint(child_id),
                    "Target": global_endpoint(parent_id),
                    "Origin": "global_fusion",
                    "Method": "deterministic",
                    "Reason": reason,
                },
            ):
                added += 1

    return added


def apply_global_fusion(
    graph: Dict,
    llm_client: Optional[LLMClient],
    *,
    batch_size: int = 20,
    max_retries: int = 2,
) -> Dict:
    """Add only global canonical identity and subtype relations.

    REFERS_TO edges are owned by enrich_policy_graph. This pass never resolves or
    refines references, including generic service mentions. Local occurrences and
    local/enrichment edges are never deleted or rewritten.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")

    usage_before = _usage_snapshot(llm_client) if llm_client is not None else None
    records = _eligible_occurrences(graph)
    uf = _UnionFind(records.keys())

    deterministic_merges = _deterministic_union(records, uf)
    tasks = _fusion_tasks(records, uf)
    llm_merges, fusion_unresolved = _resolve_fusion_tasks(
        llm_client,
        tasks,
        uf,
        batch_size=batch_size,
        max_retries=max_retries,
    )

    _assign_canonical_nodes(graph, records, uf)
    relation_actions_created = _assign_relation_action_canonicals(graph)
    subtype_edges = _build_subtype_edges(graph)

    # Keep runtime stats private for logging/diagnostics. strip_internal_fields()
    # removes every private key, so _FusionStats is never exported.
    usage = _usage_since(llm_client, usage_before) if llm_client is not None else {}
    graph["_FusionStats"] = {
        "EligibleOccurrences": len(records),
        "DeterministicMerges": deterministic_merges,
        "LLMFusionTasks": len(tasks),
        "LLMMerges": llm_merges,
        "FusionUnresolved": fusion_unresolved,
        "CanonicalNodes": len(graph.get("CanonicalNodes", {})),
        "RelationActionCanonicalsCreated": relation_actions_created,
        "SubtypeEdges": subtype_edges,
        # Kept as runtime-only zeros for compatibility with existing log format.
        # Global fusion no longer performs any service-reference resolution.
        "ServiceReferencesDeterministic": 0,
        "ServiceReferencesLLM": 0,
        "ServiceReferencesUnresolved": 0,
        "APICalls": usage.get("api_calls"),
        "InputTokens": usage.get("input_tokens"),
        "OutputTokens": usage.get("output_tokens"),
        "TotalTokens": usage.get("total_tokens"),
    }
    return graph
