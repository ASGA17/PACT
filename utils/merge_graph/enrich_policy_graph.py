import copy
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from llms import LLMClient
from utils.utils import parse_json_response

logger = logging.getLogger(__name__)

EDGE_TYPES = {"EXAMPLE_OF", "REFERS_TO", "DISTINCT_FROM", "SAME_REFERENT"}
SEMANTIC_KINDS = {"Subject", "Object", "Condition"}
REFERENCE_TARGET_KINDS = {"Subject", "Object", "Condition", "Frame"}
EXAMPLE_MATCH_IGNORED = {"a", "an", "any", "more", "one", "or", "the"}

ENRICH_SYSTEM_PROMPT = """
You resolve semantic links inside an already-built policy graph.

The local Subject, Predicate, Object, Condition, and Frame graphs are immutable.
You must not create nodes, rewrite text, change Frames, merge nodes, or invent ids.
For every task, choose exactly one Target from that task's Candidates, or null when
no candidate is clearly correct.

Task types:
- EXAMPLE_OF: choose the existing main-scope node or Path-qualified semantic facet that the example instantiates.
- REFERS_TO: choose the existing node, Path-qualified semantic facet, or Context target that the mention refers to.

Important rules:
- A Target must be copied exactly from one supplied Candidate.Target, including an optional Path.
- Path identifies the semantic facet inside an existing node; it is not a new node.
- Use null if the relation is genuinely ambiguous or unsupported.
- Generic service mentions such as "this service", "our service", and "the service"
  may resolve to an explicitly named service only when that candidate occurs in the SAME
  sentence/scope. Do not use a named service from another sentence, example, or policy.
  If the current sentence/scope has no clear named-service antecedent, use
  Context.Organization as the issuer-relative fallback.
- An explicitly named service mention such as "the ERNIE Bot service" is already a
  concrete entity mention; do not equate it with Context.Organization.
- A larger coordinated or relational phrase merely containing words such as "this service"
  is not itself a service reference. Resolve only the standalone semantic mention.
- For discourse nouns such as "such a situation", "this situation", "that state",
  "such an outcome", or "such a result", prefer an existing semantic state/result node
  (especially a Condition relation target) over a whole Frame when one is available.
- Abstract mentions such as "such restriction", "this rule", or "that requirement" may
  legitimately refer to a Frame representing the whole constrained behavior.
- Keep all original node text unchanged.
- Do not create SAME_REFERENT or DISTINCT_FROM here; those are handled elsewhere.

Return JSON only.
""".strip()


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


def _nonempty_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _normalize_text(value: Any) -> str:
    text = _nonempty_text(value)
    if text is None:
        return ""

    text = text.replace("’", "'").replace("`", "'").strip().casefold()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(' \t\r\n.,;:!?()[]{}"')

    # Normalize ordinary English possessives so "Baidu's" can match "Baidu" and
    # "others'" can match "others". This is only used for matching; source text
    # is never rewritten.
    if text.endswith("'s"):
        text = text[:-2].rstrip()
    elif text.endswith("'"):
        text = text[:-1].rstrip()

    return text


def _example_match_tokens(value: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", _normalize_text(value))
        if token not in EXAMPLE_MATCH_IGNORED
    ]


def _contains_token_sequence(container: List[str], target: List[str]) -> bool:
    if not target or len(target) > len(container):
        return False
    width = len(target)
    return any(
        container[index : index + width] == target
        for index in range(len(container) - width + 1)
    )


def _description_modifiers(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return []
    groups = value.get("DescriptionGroups")
    if not isinstance(groups, list):
        return []

    out: List[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        modifier = _nonempty_text(group.get("Modifier"))
        if modifier is not None:
            out.append(modifier)
    return out


def _condition_entity(node: Any) -> Optional[Dict]:
    if not isinstance(node, dict):
        return None
    body = node.get("Body")
    if not isinstance(body, dict):
        return None
    entity = body.get("Entity")
    return entity if isinstance(entity, dict) else None


def _condition_event(node: Any) -> Optional[Dict]:
    if not isinstance(node, dict):
        return None
    body = node.get("Body")
    if not isinstance(body, dict):
        return None
    event = body.get("Event")
    return event if isinstance(event, dict) else None


def _node_text(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    return _nonempty_text(node.get("Text")) or ""


def _node_head(node: Any, kind: Optional[str] = None) -> str:
    if not isinstance(node, dict):
        return ""

    head = _nonempty_text(node.get("Head"))
    if head is not None:
        return head

    if kind == "Predicate":
        action = _nonempty_text(node.get("Action"))
        if action is not None:
            return action

    entity = _condition_entity(node)
    if entity is not None:
        head = _nonempty_text(entity.get("Head"))
        if head is not None:
            return head

    return ""


def _ref_endpoint(scope: str, ref: str, path: Optional[str] = None) -> Dict[str, str]:
    out = {"Scope": scope, "Ref": ref}
    if path:
        out["Path"] = path
    return out


def _context_endpoint(name: str) -> Dict[str, str]:
    return {"Context": name}


def _endpoint_key(endpoint: Any) -> Tuple:
    if not isinstance(endpoint, dict):
        return ("invalid",)
    if set(endpoint) == {"Context"} and isinstance(endpoint.get("Context"), str):
        return ("context", endpoint["Context"])
    return (
        "local",
        endpoint.get("Scope"),
        endpoint.get("Ref"),
        endpoint.get("Path"),
    )


def _edge_key(edge: Any) -> Tuple:
    if not isinstance(edge, dict):
        return ("invalid",)
    return (
        edge.get("Type"),
        _endpoint_key(edge.get("Source")),
        _endpoint_key(edge.get("Target")),
    )


def _graph_nodes(element: Any, field: str) -> Dict[str, Dict]:
    if not isinstance(element, dict):
        return {}
    graph = element.get(field)
    if not isinstance(graph, dict):
        return {}
    nodes = graph.get("Nodes")
    return nodes if isinstance(nodes, dict) else {}


def _graph_root(element: Any, field: str) -> Optional[str]:
    if not isinstance(element, dict):
        return None
    graph = element.get(field)
    if not isinstance(graph, dict):
        return None
    root = graph.get("Root")
    return root if isinstance(root, str) and root else None


def _scope_records(rule: Dict) -> List[Tuple[str, Dict]]:
    out: List[Tuple[str, Dict]] = []

    main = rule.get("Main_Clause_Elements")
    if isinstance(main, dict):
        out.append(("main", main))

    examples = rule.get("Example_Sentences_Elements", [])
    if isinstance(examples, list):
        for index, element in enumerate(examples):
            if isinstance(element, dict):
                out.append((f"example:{index}", element))

    return out


def _build_inventory(rule: Dict) -> Dict[str, Dict]:
    inventory: Dict[str, Dict] = {}

    field_specs = (
        ("Subject_Elements", "Subject"),
        ("Predicate_Elements", "Predicate"),
        ("Object_Elements", "Object"),
        ("Condition_Elements", "Condition"),
        ("Frame_Elements", "Frame"),
    )

    for scope, element in _scope_records(rule):
        nodes: Dict[str, Dict] = {}
        kinds: Dict[str, str] = {}

        for field, kind in field_specs:
            for ref, node in _graph_nodes(element, field).items():
                if not isinstance(ref, str) or not isinstance(node, dict):
                    continue
                nodes[ref] = node
                kinds[ref] = kind

        inventory[scope] = {
            "element": element,
            "nodes": nodes,
            "kinds": kinds,
            "subject_root": _graph_root(element, "Subject_Elements"),
            "frame_root": _graph_root(element, "Frame_Elements"),
            "sentence": _nonempty_text(element.get("Sentence")) or "",
        }

    return inventory


def _planned_example_records(rule: Dict) -> Tuple[Dict[str, Dict], Dict[int, str]]:
    """Assign stable ordinary-example refs without mutating the input yet."""
    examples = rule.get("Examples_Elements", [])
    if not isinstance(examples, list):
        return {}, {}

    by_ref: Dict[str, Dict] = {}
    by_index: Dict[int, str] = {}
    used: Set[str] = set()
    next_counter = 0

    for index, item in enumerate(examples):
        if not isinstance(item, dict):
            continue

        existing_ref = item.get("Ref")
        if (
            isinstance(existing_ref, str)
            and re.fullmatch(r"e\d+", existing_ref)
            and existing_ref not in used
        ):
            ref = existing_ref
        else:
            preferred = f"e{index}"
            if preferred not in used:
                ref = preferred
            else:
                while f"e{next_counter}" in used:
                    next_counter += 1
                ref = f"e{next_counter}"
                next_counter += 1

        used.add(ref)
        text = (
            _nonempty_text(item.get("Text"))
            or _nonempty_text(item.get("Example"))
            or ""
        )
        by_index[index] = ref
        by_ref[ref] = {"index": index, "item": item, "text": text}

    return by_ref, by_index


def _walk_path(value: Any, path: Optional[str]) -> Any:
    if not path:
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def _local_endpoint_exists(
    inventory: Dict[str, Dict], endpoint: Dict, *, require_path: bool = True
) -> bool:
    scope = endpoint.get("Scope")
    ref = endpoint.get("Ref")
    if not isinstance(scope, str) or not isinstance(ref, str):
        return False

    record = inventory.get(scope)
    if not isinstance(record, dict):
        return False
    node = record.get("nodes", {}).get(ref)
    if not isinstance(node, dict):
        return False

    path = endpoint.get("Path")
    if path is None:
        return True
    if not require_path or not isinstance(path, str) or not path:
        return False
    return _walk_path(node, path) is not None


def _example_endpoint_exists(example_records: Dict[str, Dict], endpoint: Dict) -> bool:
    if endpoint.get("Scope") != "examples":
        return False
    ref = endpoint.get("Ref")
    return isinstance(ref, str) and ref in example_records and "Path" not in endpoint


def _context_endpoint_exists(context: Dict, endpoint: Dict) -> bool:
    if set(endpoint) != {"Context"}:
        return False
    name = endpoint.get("Context")
    if name != "Organization":
        return False
    return _nonempty_text(context.get(name)) is not None


def _validate_edge(
    edge: Any,
    inventory: Dict[str, Dict],
    example_records: Dict[str, Dict],
    context: Dict,
) -> bool:
    if not isinstance(edge, dict) or set(edge) != {"Type", "Source", "Target"}:
        return False

    edge_type = edge.get("Type")
    source = edge.get("Source")
    target = edge.get("Target")
    if (
        edge_type not in EDGE_TYPES
        or not isinstance(source, dict)
        or not isinstance(target, dict)
    ):
        return False

    source_is_local = _local_endpoint_exists(inventory, source)
    source_is_example = _example_endpoint_exists(example_records, source)
    target_is_local = _local_endpoint_exists(inventory, target)
    target_is_context = _context_endpoint_exists(context, target)

    if edge_type == "EXAMPLE_OF":
        # Example anchors may point either to a whole existing node or to a semantic
        # facet inside that node (for example c0.Body.Entity). The facet is addressed
        # by Path and does not create a virtual node.
        return source_is_example and target_is_local and target.get("Scope") == "main"

    if edge_type == "REFERS_TO":
        # A referent may also be an existing semantic facet. Context targets never have Path.
        return source_is_local and (target_is_local or target_is_context)

    if edge_type == "DISTINCT_FROM":
        if not source_is_local or not target_is_local or "Path" in target:
            return False
        return source.get("Scope") == target.get("Scope")

    if edge_type == "SAME_REFERENT":
        return source_is_local and target_is_local and "Path" not in target

    return False


def _ref_sort_key(value: Any) -> Tuple[str, int, str]:
    if not isinstance(value, str):
        return ("", -1, "")
    match = re.fullmatch(r"([A-Za-z_:-]*?)(\d+)", value)
    if match:
        return (match.group(1), int(match.group(2)), value)
    return (value, -1, value)


def _scope_sort_key(value: Any) -> Tuple[int, int, str]:
    if value == "main":
        return (0, 0, "main")
    if value == "examples":
        return (1, 0, "examples")
    if isinstance(value, str):
        match = re.fullmatch(r"example:(\d+)", value)
        if match:
            return (2, int(match.group(1)), value)
    return (3, 0, str(value or ""))


def _edge_sort_key(edge: Dict) -> Tuple:
    order = {
        "EXAMPLE_OF": 0,
        "REFERS_TO": 1,
        "DISTINCT_FROM": 2,
        "SAME_REFERENT": 3,
    }
    source = edge.get("Source", {})
    target = edge.get("Target", {})
    return (
        order.get(edge.get("Type"), 99),
        _scope_sort_key(source.get("Scope")),
        _ref_sort_key(source.get("Ref")),
        source.get("Path", ""),
        target.get("Context", ""),
        _scope_sort_key(target.get("Scope")),
        _ref_sort_key(target.get("Ref")),
    )


def _add_edge(state: Dict, edge_type: str, source: Dict, target: Dict) -> bool:
    edge = {
        "Type": edge_type,
        "Source": copy.deepcopy(source),
        "Target": copy.deepcopy(target),
    }
    key = _edge_key(edge)
    if key in state["edge_keys"]:
        return False
    state["edges"].append(edge)
    state["edge_keys"].add(key)
    return True


def _has_edge_from(state: Dict, edge_type: str, source: Dict) -> bool:
    source_key = _endpoint_key(source)
    for edge in state["edges"]:
        if (
            edge.get("Type") == edge_type
            and _endpoint_key(edge.get("Source")) == source_key
        ):
            return True
    return False


def _targets_from_source(state: Dict, edge_type: str, source: Dict) -> List[Dict]:
    source_key = _endpoint_key(source)
    out = []
    for edge in state["edges"]:
        if edge.get("Type") != edge_type:
            continue
        if _endpoint_key(edge.get("Source")) != source_key:
            continue
        target = edge.get("Target")
        if isinstance(target, dict):
            out.append(target)
    return out


def _organization_aliases(organization: str) -> Set[str]:
    normalized = _normalize_text(organization)
    return {normalized} if normalized else set()


def _is_generic_service_mention(text: Any) -> bool:
    """Issuer-relative service mention whose surface text must stay unchanged."""
    return _normalize_text(text) in {"this service", "our service", "the service"}


GENERIC_SERVICE_MODIFIERS = {
    "this",
    "that",
    "the",
    "our",
    "your",
    "their",
    "its",
    "a",
    "an",
    "any",
    "some",
    "such",
    "current",
    "same",
    "other",
    "another",
}


def _node_modifiers(node: Any, kind: Optional[str] = None) -> List[str]:
    if not isinstance(node, dict):
        return []
    if kind == "Condition":
        entity = _condition_entity(node)
        if entity is not None:
            return _description_modifiers(entity)
    return _description_modifiers(node)


def _surface_service_name(text: Any) -> str:
    """Return a name-like prefix from '<name> service', or ''.

    This is a conservative fallback for cases where the local graph did not put the
    service name in DescriptionGroups. It is used only to recognize an already-built
    named service node; it never rewrites source text or creates a node.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    # Normalize a possessive immediately before the noun: "Baidu's service".
    normalized = re.sub(r"'s(?=\s+service$)", "", normalized)
    if normalized.startswith("the "):
        normalized = normalized[4:].strip()
    if not normalized.endswith(" service"):
        return ""

    prefix = normalized[: -len(" service")].strip()
    if not prefix:
        return ""

    words = [word for word in prefix.split() if word]
    if not words or all(word in GENERIC_SERVICE_MODIFIERS for word in words):
        return ""
    return prefix


def _named_service_name(node: Any, kind: Optional[str]) -> str:
    """Return a stable name signal iff this node is an explicitly named service."""
    if not isinstance(node, dict):
        return ""

    head = _normalize_text(_node_head(node, kind))
    if head != "service":
        # Deliberately exclude plural generic category nodes such as "services".
        return ""

    text = _node_text(node)
    if _is_generic_service_mention(text):
        return ""

    modifiers = []
    for modifier in _node_modifiers(node, kind):
        normalized = _normalize_text(modifier)
        if normalized and normalized not in GENERIC_SERVICE_MODIFIERS:
            modifiers.append(modifier.strip())

    if modifiers:
        return " ".join(modifiers)

    return _surface_service_name(text)


def _named_service_candidates_in_scope(
    state: Dict,
    scope: str,
    *,
    exclude_ref: Optional[str] = None,
) -> List[Dict]:
    record = state["inventory"].get(scope, {})
    nodes = record.get("nodes", {})
    kinds = record.get("kinds", {})

    out: List[Dict] = []
    for ref, node in nodes.items():
        if ref == exclude_ref:
            continue
        kind = kinds.get(ref)
        if kind not in SEMANTIC_KINDS:
            continue
        service_name = _named_service_name(node, kind)
        if not service_name:
            continue
        out.append(
            {
                "Target": _ref_endpoint(scope, ref),
                "Kind": kind,
                "Text": _node_text(node),
                "Head": _node_head(node, kind),
                "ServiceName": service_name,
            }
        )

    out.sort(
        key=lambda item: (
            _scope_sort_key(item["Target"].get("Scope")),
            _ref_sort_key(item["Target"].get("Ref")),
        )
    )
    return out


def _service_reference_resolution(
    state: Dict,
    scope: str,
    source_ref: str,
) -> Tuple[str, Optional[Dict], List[Dict]]:
    """Resolve a generic service mention inside the current sentence/scope only.

    Rules:
      1. Search explicitly named service nodes only in ``scope``.
      2. One candidate -> resolve to that existing local node.
      3. Multiple candidates -> let the LLM choose only among those same-scope nodes.
      4. No clear same-scope antecedent -> Context.Organization fallback.

    We intentionally never search main/other example scopes here. A service name that
    appears in another sentence is corpus context, not evidence that the current
    "this service" has that referent.
    """
    candidates = _named_service_candidates_in_scope(
        state,
        scope,
        exclude_ref=source_ref,
    )

    if len(candidates) == 1:
        return "resolved", candidates[0]["Target"], []
    if len(candidates) > 1:
        return "ambiguous", None, candidates

    if _nonempty_text(state["context"].get("Organization")) is not None:
        return "resolved", _context_endpoint("Organization"), []

    return "skip", None, []


def _source_mention_text(inventory: Dict[str, Dict], endpoint: Any) -> str:
    if not isinstance(endpoint, dict):
        return ""
    scope = endpoint.get("Scope")
    ref = endpoint.get("Ref")
    if not isinstance(scope, str) or not isinstance(ref, str):
        return ""
    node = inventory.get(scope, {}).get("nodes", {}).get(ref)
    if not isinstance(node, dict):
        return ""

    path = endpoint.get("Path")
    if isinstance(path, str) and path:
        value = _walk_path(node, path)
        if isinstance(value, dict):
            return _nonempty_text(value.get("Text")) or ""
        if isinstance(value, str):
            return value
        return ""

    return _node_text(node)


def _is_recomputable_service_fallback_edge(
    edge: Any, inventory: Dict[str, Dict]
) -> bool:
    """Drop cached generic-service fallbacks so current-scope resolution is rerun.

    Recomputing is safe and idempotent: a generic service may upgrade only to an
    explicitly named service in the SAME sentence/scope; otherwise it is written back
    to Context.Organization.
    """
    if not isinstance(edge, dict) or edge.get("Type") != "REFERS_TO":
        return False
    target = edge.get("Target")
    if target != {"Context": "Organization"}:
        return False
    mention = _source_mention_text(inventory, edge.get("Source"))
    return _is_generic_service_mention(mention)


def _is_second_person_reference(text: Any) -> bool:
    return _normalize_text(text) in {"you", "your", "yours"}


def _is_first_person_reference(text: Any) -> bool:
    return _normalize_text(text) in {"we", "our", "ours", "us"}


def _is_direct_discourse_mention(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    if normalized in {
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "this",
        "that",
        "these",
        "those",
    }:
        return True

    # "such as" is an illustrative marker, not a discourse reference.
    return (
        (normalized.startswith("such ") and not normalized.startswith("such as "))
        or normalized.startswith("this ")
        or normalized.startswith("that ")
    )


def _semantic_description_groups(node: Dict, kind: str) -> List[Dict]:
    if kind == "Condition":
        entity = _condition_entity(node)
        if entity is not None:
            groups = entity.get("DescriptionGroups")
            return groups if isinstance(groups, list) else []
    groups = node.get("DescriptionGroups")
    return groups if isinstance(groups, list) else []


def _node_is_pure_direct_discourse_source(node: Dict, kind: str) -> bool:
    """Reject larger phrases that merely begin with a discourse expression.

    Example: ID22's o0 = "this service and any content it contains" is a coordination
    container, not a referential mention. The standalone child o1 = "this service" is the
    actual service mention and is handled separately by the service resolver.
    """
    if not isinstance(node, dict):
        return False

    if kind == "Condition":
        entity = _condition_entity(node)
        coordination = entity.get("Coordination") if isinstance(entity, dict) else None
    else:
        coordination = node.get("Coordination")
    if isinstance(coordination, dict):
        return False

    for group in _semantic_description_groups(node, kind):
        if not isinstance(group, dict):
            continue
        if "Relation" in group or "Operator" in group:
            return False

    return True


SITUATION_DISCOURSE_HEADS = {
    "situation",
    "state",
    "circumstance",
    "condition",
    "outcome",
    "result",
}

STRUCTURAL_RELATION_PREDICATES = {
    "of",
    "on",
    "in",
    "at",
    "to",
    "for",
    "by",
    "with",
    "from",
    "towards",
    "toward",
    "against",
    "within",
    "into",
    "upon",
}


def _discourse_head(text: Any) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    words = normalized.split()
    return words[-1] if words else ""


def _is_situation_discourse_mention(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if not (
        normalized.startswith("such ")
        or normalized.startswith("this ")
        or normalized.startswith("that ")
    ):
        return False
    return _discourse_head(normalized) in SITUATION_DISCOURSE_HEADS


def _walk_relations(value: Any) -> Iterable[Tuple[str, str]]:
    """Yield (predicate, target_ref) from existing local Relation objects."""
    if isinstance(value, dict):
        relation = value.get("Relation")
        if isinstance(relation, dict):
            predicate = _nonempty_text(relation.get("Predicate")) or ""
            target = relation.get("Target")
            target_ref = target.get("Ref") if isinstance(target, dict) else None
            if predicate and isinstance(target_ref, str):
                yield predicate, target_ref
        for child in value.values():
            yield from _walk_relations(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_relations(child)


def _situation_reference_resolution(
    state: Dict,
    scope: str,
    source_ref: str,
) -> Tuple[str, Optional[Dict], List[Dict]]:
    """Prefer semantic state/result nodes over whole Frames for 'such a situation'.

    The strongest generic signal available in the existing graph is an internal Condition
    node that is the Target of a content-bearing Relation (for example c0 --imposes--> c1).
    Pure structural/prepositional relations such as 'on' or 'of' are intentionally ignored.
    """
    record = state["inventory"].get(scope, {})
    nodes = record.get("nodes", {})
    kinds = record.get("kinds", {})

    strong: List[Dict] = []
    seen: Set[Tuple] = set()
    for ref, node in nodes.items():
        if kinds.get(ref) != "Condition" or not isinstance(node, dict):
            continue
        for predicate, target_ref in _walk_relations(node):
            if target_ref == source_ref or kinds.get(target_ref) != "Condition":
                continue
            normalized_predicate = _normalize_text(predicate)
            if normalized_predicate in STRUCTURAL_RELATION_PREDICATES:
                continue
            target_node = nodes.get(target_ref)
            if not isinstance(target_node, dict):
                continue
            candidate = {
                "Target": _ref_endpoint(scope, target_ref),
                "Kind": "Condition",
                "Text": _node_text(target_node),
                "Head": _node_head(target_node, "Condition"),
                "IncomingPredicate": predicate,
            }
            key = _endpoint_key(candidate["Target"])
            if key not in seen:
                seen.add(key)
                strong.append(candidate)

    strong.sort(key=lambda item: _ref_sort_key(item.get("Target", {}).get("Ref")))
    if len(strong) == 1:
        return "resolved", strong[0]["Target"], []
    if len(strong) > 1:
        return "ambiguous", None, strong

    # If no strong relation target exists, still keep semantic nodes ahead of Frames.
    semantic = _kind_candidates(
        state,
        scope,
        kinds={"Condition", "Object"},
        exclude_ref=source_ref,
    )
    if semantic:
        return "ambiguous", None, semantic

    fallback = _reference_candidates(state, scope, source_ref)
    return ("ambiguous", None, fallback) if fallback else ("skip", None, [])


def _reference_paths(node: Dict) -> List[Tuple[str, Dict]]:
    out: List[Tuple[str, Dict]] = []

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key == "Reference" and isinstance(child, dict):
                    text = _nonempty_text(child.get("Text"))
                    if text is not None:
                        out.append((path, child))
                    continue
                walk(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                path = f"{prefix}.{index}" if prefix else str(index)
                walk(child, path)

    walk(node)
    return out


def _kind_candidates(
    state: Dict,
    scope: str,
    *,
    kinds: Set[str],
    exclude_ref: Optional[str] = None,
) -> List[Dict]:
    record = state["inventory"].get(scope, {})
    nodes = record.get("nodes", {})
    node_kinds = record.get("kinds", {})

    out: List[Dict] = []
    for ref, node in nodes.items():
        if ref == exclude_ref:
            continue
        kind = node_kinds.get(ref)
        if kind not in kinds:
            continue
        out.append(
            {
                "Target": _ref_endpoint(scope, ref),
                "Kind": kind,
                "Text": _node_text(node),
                "Head": _node_head(node, kind),
            }
        )
    return out


def _exact_local_matches(
    state: Dict, scope: str, text: str, source_ref: str
) -> List[Dict]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    candidates = _kind_candidates(
        state,
        scope,
        kinds=SEMANTIC_KINDS,
        exclude_ref=source_ref,
    )

    text_matches = [
        candidate
        for candidate in candidates
        if _normalize_text(candidate.get("Text")) == normalized
    ]
    if text_matches:
        return text_matches

    return [
        candidate
        for candidate in candidates
        if _normalize_text(candidate.get("Head")) == normalized
    ]


def _frame_summary(state: Dict, scope: str, ref: str, node: Dict) -> str:
    record = state["inventory"].get(scope, {})
    nodes = record.get("nodes", {})

    behavior = node.get("Behavior") if isinstance(node, dict) else None
    if isinstance(behavior, dict):
        parts = []
        for role in ("Subject", "Predicate", "Object"):
            endpoint = behavior.get(role)
            role_ref = endpoint.get("Ref") if isinstance(endpoint, dict) else None
            role_node = nodes.get(role_ref) if isinstance(role_ref, str) else None
            if isinstance(role_node, dict):
                parts.append(f"{role}={_node_text(role_node) or role_ref}")
            elif role_ref is None:
                parts.append(f"{role}=null")
        conditions = behavior.get("Conditions")
        if isinstance(conditions, list):
            condition_texts = []
            for item in conditions:
                cond_ref = item.get("Ref") if isinstance(item, dict) else None
                cond_node = nodes.get(cond_ref) if isinstance(cond_ref, str) else None
                if isinstance(cond_node, dict):
                    condition_texts.append(_node_text(cond_node) or cond_ref)
            if condition_texts:
                parts.append("Conditions=" + " | ".join(condition_texts))
        return "; ".join(parts)

    coordination = node.get("Coordination") if isinstance(node, dict) else None
    if isinstance(coordination, dict):
        operator = coordination.get("Operator")
        items = coordination.get("Items")
        refs = []
        if isinstance(items, list):
            for item in items:
                item_ref = item.get("Ref") if isinstance(item, dict) else None
                if isinstance(item_ref, str):
                    refs.append(item_ref)
        return f"Frame coordination {operator}: {', '.join(refs)}"

    return ref


def _reference_candidates(state: Dict, scope: str, source_ref: str) -> List[Dict]:
    candidates = _kind_candidates(
        state,
        scope,
        kinds=REFERENCE_TARGET_KINDS,
        exclude_ref=source_ref,
    )

    # Example clauses may refer back to a participant established by the main
    # clause of the same policy (for example, "such rights").
    if scope.startswith("example:"):
        candidates.extend(
            _kind_candidates(
                state,
                "main",
                kinds=REFERENCE_TARGET_KINDS,
            )
        )

    # A later pronoun may refer to the possessor/coreferent represented by an
    # earlier node's Reference facet, rather than to the whole possessed entity.
    # Expose that existing semantic facet as a Path-qualified target.
    candidate_scopes = [scope]
    if scope.startswith("example:"):
        candidate_scopes.append("main")
    for candidate_scope in candidate_scopes:
        record = state["inventory"].get(candidate_scope, {})
        for ref, node in record.get("nodes", {}).items():
            if candidate_scope == scope and ref == source_ref:
                continue
            kind = record.get("kinds", {}).get(ref)
            if kind not in SEMANTIC_KINDS or not isinstance(node, dict):
                continue
            for path, reference in _reference_paths(node):
                reference_text = _nonempty_text(reference.get("Text"))
                if reference_text is None:
                    continue
                candidates.append(
                    {
                        "Target": _ref_endpoint(candidate_scope, ref, path),
                        "Kind": f"{kind}Reference",
                        "Text": reference_text,
                        "Head": reference_text,
                    }
                )

    for candidate in candidates:
        if candidate.get("Kind") != "Frame":
            continue
        target = candidate.get("Target", {})
        ref = target.get("Ref")
        node = state["inventory"].get(scope, {}).get("nodes", {}).get(ref)
        if isinstance(ref, str) and isinstance(node, dict):
            candidate["Summary"] = _frame_summary(state, scope, ref, node)

    context = state["context"]
    if _nonempty_text(context.get("Organization")) is not None:
        candidates.append(
            {
                "Target": _context_endpoint("Organization"),
                "Kind": "Context",
                "Text": context["Organization"],
                "Head": "Organization",
            }
        )
    return candidates


def _example_candidates(state: Dict, anchor_type: str) -> List[Dict]:
    """Return semantic anchor candidates for ordinary examples.

    Subject/Object/Predicate nodes are already entity/action nodes, so a Head match points
    to the whole node. Condition nodes are containers whose Body may hold the real semantic
    Entity/Event. For those nested facets we expose a Path-qualified endpoint instead of
    collapsing the match to the whole c* node.
    """
    target_kind = {
        "Subject": "Subject",
        "Predicate": "Predicate",
        "Object": "Object",
        "Condition": "Condition",
    }.get(anchor_type)
    if target_kind is None:
        return []

    if target_kind != "Condition":
        return _kind_candidates(state, "main", kinds={target_kind})

    record = state["inventory"].get("main", {})
    nodes = record.get("nodes", {})
    kinds = record.get("kinds", {})
    out: List[Dict] = []

    for ref, node in nodes.items():
        if kinds.get(ref) != "Condition" or not isinstance(node, dict):
            continue

        # Whole-node candidate: exact Text matches should still resolve to c*.
        out.append(
            {
                "Target": _ref_endpoint("main", ref),
                "Kind": "Condition",
                "Text": _node_text(node),
                # Do not expose the nested Entity.Head here. Otherwise a Head match like
                # "means" would incorrectly collapse to the whole "by any means" node.
                "Head": "",
            }
        )

        entity = _condition_entity(node)
        if entity is not None:
            entity_head = _nonempty_text(entity.get("Head")) or ""
            if entity_head:
                semantic_text = " ".join(
                    [*_description_modifiers(entity), entity_head]
                ).strip()
                out.append(
                    {
                        "Target": _ref_endpoint("main", ref, "Body.Entity"),
                        "Kind": "ConditionEntity",
                        "Text": "",
                        "SemanticText": semantic_text,
                        "Head": entity_head,
                        "ContainerText": _node_text(node),
                    }
                )

        event = _condition_event(node)
        if event is not None:
            action = _nonempty_text(event.get("Action")) or ""
            if action:
                out.append(
                    {
                        "Target": _ref_endpoint("main", ref, "Body.Event"),
                        "Kind": "ConditionEvent",
                        "Text": "",
                        "Head": action,
                        "ContainerText": _node_text(node),
                    }
                )

    out.sort(
        key=lambda item: (
            _ref_sort_key(item.get("Target", {}).get("Ref")),
            item.get("Target", {}).get("Path", ""),
        )
    )
    return out


def _resolve_example_deterministically(
    state: Dict,
    anchor_type: str,
    anchor_text: str,
) -> Tuple[Optional[Dict], List[Dict]]:
    candidates = _example_candidates(state, anchor_type)
    if not candidates:
        return None, []

    normalized = _normalize_text(anchor_text)
    text_matches = [
        candidate
        for candidate in candidates
        if _normalize_text(candidate.get("Text")) == normalized
    ]
    if len(text_matches) == 1:
        return text_matches[0]["Target"], candidates
    if len(text_matches) > 1:
        return None, text_matches

    head_matches = [
        candidate
        for candidate in candidates
        if _normalize_text(candidate.get("Head")) == normalized
    ]
    if len(head_matches) == 1:
        return head_matches[0]["Target"], candidates
    if len(head_matches) > 1:
        return None, head_matches

    # Upstream anchors intentionally omit articles, quantifiers, and Condition
    # markers. Choose the shortest unique semantic candidate that contains the
    # complete anchor token sequence (for example, "rights of others" within
    # "the rights of others", or "commercial purposes" within a Condition
    # entity under "for commercial purposes").
    anchor_tokens = _example_match_tokens(anchor_text)
    contained = []
    for candidate in candidates:
        surface = candidate.get("SemanticText") or candidate.get("Text")
        surface_tokens = _example_match_tokens(surface)
        if _contains_token_sequence(surface_tokens, anchor_tokens):
            contained.append((len(surface_tokens), candidate))
    if contained:
        minimum = min(length for length, _ in contained)
        best = [candidate for length, candidate in contained if length == minimum]
        if len(best) == 1:
            return best[0]["Target"], candidates
        return None, best

    if len(candidates) == 1:
        return candidates[0]["Target"], candidates

    return None, candidates


def _finalize_example_item(state: Dict, example_index: int, example_ref: str) -> None:
    examples = state["rule"].get("Examples_Elements")
    if not isinstance(examples, list) or not (0 <= example_index < len(examples)):
        return
    item = examples[example_index]
    if not isinstance(item, dict):
        return
    text = _nonempty_text(item.get("Text")) or _nonempty_text(item.get("Example")) or ""
    examples[example_index] = {"Ref": example_ref, "Text": text}


def _collect_example_work(state: Dict, tasks: List[Dict]) -> Tuple[int, int]:
    deterministic = 0
    ambiguous = 0

    examples = state["rule"].get("Examples_Elements", [])
    if not isinstance(examples, list):
        return deterministic, ambiguous

    for index, item in enumerate(examples):
        if not isinstance(item, dict):
            continue

        example_ref = state["example_ref_by_index"].get(index)
        if not isinstance(example_ref, str):
            continue
        source = _ref_endpoint("examples", example_ref)

        # If a valid EXAMPLE_OF edge already exists, it is now the canonical anchor.
        if _has_edge_from(state, "EXAMPLE_OF", source):
            _finalize_example_item(state, index, example_ref)
            continue

        # Already-final examples no longer contain anchor metadata. Without an
        # existing EXAMPLE_OF edge, the anchor cannot safely be reconstructed.
        if "AnchorType" not in item and "AnchorText" not in item:
            continue

        anchor_type = _nonempty_text(item.get("AnchorType"))
        anchor_text = _nonempty_text(item.get("AnchorText"))
        if anchor_type is None or anchor_text is None:
            continue

        target, candidates = _resolve_example_deterministically(
            state, anchor_type, anchor_text
        )
        if target is not None:
            _add_edge(state, "EXAMPLE_OF", source, target)
            _finalize_example_item(state, index, example_ref)
            deterministic += 1
            continue

        if not candidates:
            logger.warning(
                "PolicyEnrich | unresolved example anchor | policy=%s | example=%r | anchor_type=%s | anchor=%r",
                state["policy_id"],
                item.get("Example"),
                anchor_type,
                anchor_text,
            )
            continue

        task_id = f"policy{state['state_index']}_example{index}"
        tasks.append(
            {
                "id": task_id,
                "type": "EXAMPLE_OF",
                "state": state,
                "source": source,
                "example_index": index,
                "example_ref": example_ref,
                "mention": _nonempty_text(item.get("Example")) or "",
                "anchor_type": anchor_type,
                "anchor_text": anchor_text,
                "scope": "main",
                "sentence": _nonempty_text(state["rule"].get("Original_Policy")) or "",
                "candidates": candidates,
            }
        )
        ambiguous += 1

    return deterministic, ambiguous


def _deterministic_reference_target(
    state: Dict,
    scope: str,
    source_ref: str,
    reference_text: str,
) -> Tuple[str, Optional[Dict], List[Dict]]:
    """Return (status, target, candidates).

    status is one of: resolved, ambiguous, skip.
    "skip" means a deliberate no-resolution case with insufficient information.
    """
    normalized = _normalize_text(reference_text)
    if not normalized:
        return "skip", None, []

    subject_root = state["inventory"].get(scope, {}).get("subject_root")
    if _is_second_person_reference(reference_text) and isinstance(subject_root, str):
        return "resolved", _ref_endpoint(scope, subject_root), []

    if _is_first_person_reference(reference_text):
        if _nonempty_text(state["context"].get("Organization")) is not None:
            return "resolved", _context_endpoint("Organization"), []
        return "skip", None, []

    if normalized in _organization_aliases(state["context"].get("Organization") or ""):
        return "resolved", _context_endpoint("Organization"), []

    if _is_generic_service_mention(reference_text):
        return _service_reference_resolution(state, scope, source_ref)

    local_matches = _exact_local_matches(state, scope, reference_text, source_ref)
    if len(local_matches) == 1:
        return "resolved", local_matches[0]["Target"], []

    candidates = _reference_candidates(state, scope, source_ref)
    if len(local_matches) > 1:
        # Keep all exact duplicate mentions as candidates; Context remains useful
        # only if it was independently supplied.
        exact_keys = {_endpoint_key(candidate["Target"]) for candidate in local_matches}
        filtered = [
            candidate
            for candidate in candidates
            if _endpoint_key(candidate["Target"]) in exact_keys
        ]
        if filtered:
            return "ambiguous", None, filtered

    return "ambiguous", None, candidates


def _collect_reference_work(state: Dict, tasks: List[Dict]) -> Tuple[int, int]:
    deterministic = 0
    ambiguous = 0

    for scope, record in state["inventory"].items():
        nodes = record.get("nodes", {})
        kinds = record.get("kinds", {})

        for ref, node in nodes.items():
            kind = kinds.get(ref)
            if kind not in SEMANTIC_KINDS:
                continue

            node_text = _node_text(node)

            # Standalone first/second-person pronouns are deterministic even when
            # the splitter did not emit a structured Reference object.
            whole_source = _ref_endpoint(scope, ref)
            if _is_second_person_reference(node_text):
                subject_root = record.get("subject_root")
                if isinstance(subject_root, str) and not _has_edge_from(
                    state, "REFERS_TO", whole_source
                ):
                    _add_edge(
                        state,
                        "REFERS_TO",
                        whole_source,
                        _ref_endpoint(scope, subject_root),
                    )
                    deterministic += 1
                continue
            if _is_first_person_reference(node_text):
                if _nonempty_text(
                    state["context"].get("Organization")
                ) is not None and not _has_edge_from(state, "REFERS_TO", whole_source):
                    _add_edge(
                        state,
                        "REFERS_TO",
                        whole_source,
                        _context_endpoint("Organization"),
                    )
                    deterministic += 1
                continue

            # Generic service mentions may resolve only to a named service in this
            # same sentence/scope. Otherwise Context.Organization is the fallback.
            # The local node text is never rewritten.
            if _is_generic_service_mention(node_text):
                source = _ref_endpoint(scope, ref)
                if _has_edge_from(state, "REFERS_TO", source):
                    continue

                status, target, candidates = _service_reference_resolution(
                    state,
                    scope,
                    ref,
                )
                if status == "resolved" and target is not None:
                    _add_edge(state, "REFERS_TO", source, target)
                    deterministic += 1
                    continue
                if status == "ambiguous" and candidates:
                    task_id = f"policy{state['state_index']}_{scope.replace(':', '_')}_{ref}_service"
                    tasks.append(
                        {
                            "id": task_id,
                            "type": "REFERS_TO",
                            "state": state,
                            "source": source,
                            "scope": scope,
                            "mention": node_text,
                            "reference_type": "ServiceReference",
                            "node_text": node_text,
                            "sentence": record.get("sentence", ""),
                            "candidates": candidates,
                        }
                    )
                    ambiguous += 1
                continue

            # A standalone explicit organization mention can safely use Context.Organization.
            if _normalize_text(node_text) in _organization_aliases(
                state["context"].get("Organization") or ""
            ):
                source = _ref_endpoint(scope, ref)
                if not _has_edge_from(state, "REFERS_TO", source):
                    _add_edge(
                        state, "REFERS_TO", source, _context_endpoint("Organization")
                    )
                    deterministic += 1

            for path, reference in _reference_paths(node):
                source = _ref_endpoint(scope, ref, path)
                if _has_edge_from(state, "REFERS_TO", source):
                    continue

                reference_text = _nonempty_text(reference.get("Text"))
                if reference_text is None:
                    continue

                status, target, candidates = _deterministic_reference_target(
                    state,
                    scope,
                    ref,
                    reference_text,
                )
                if status == "resolved" and target is not None:
                    _add_edge(state, "REFERS_TO", source, target)
                    deterministic += 1
                    continue
                if status == "skip" or not candidates:
                    continue

                task_id = f"policy{state['state_index']}_{scope.replace(':', '_')}_{ref}_{path.replace('.', '_')}"
                tasks.append(
                    {
                        "id": task_id,
                        "type": "REFERS_TO",
                        "state": state,
                        "source": source,
                        "scope": scope,
                        "mention": reference_text,
                        "reference_type": (
                            "ServiceReference"
                            if _is_generic_service_mention(reference_text)
                            else (_nonempty_text(reference.get("Type")) or "")
                        ),
                        "node_text": node_text,
                        "sentence": record.get("sentence", ""),
                        "candidates": candidates,
                    }
                )
                ambiguous += 1

            # Direct discourse mentions which the local graph has not represented as
            # a structured Reference. Only resolve a pure mention node. A coordination or
            # relational container that merely contains/starts with "this ..." is not itself
            # a reference source (ID22's o0 is the canonical example).
            if _is_direct_discourse_mention(
                node_text
            ) and _node_is_pure_direct_discourse_source(node, kind):
                source = _ref_endpoint(scope, ref)
                if _has_edge_from(state, "REFERS_TO", source):
                    continue

                normalized_direct = _normalize_text(node_text)
                if scope.startswith("example:") and normalized_direct.startswith(
                    "such "
                ):
                    antecedent_text = normalized_direct[len("such ") :].strip()
                    matches = _exact_local_matches(
                        state,
                        "main",
                        antecedent_text,
                        source_ref="",
                    )
                    if len(matches) == 1:
                        _add_edge(state, "REFERS_TO", source, matches[0]["Target"])
                        deterministic += 1
                        continue

                if _is_situation_discourse_mention(node_text):
                    status, target, candidates = _situation_reference_resolution(
                        state,
                        scope,
                        ref,
                    )
                    if status == "resolved" and target is not None:
                        _add_edge(state, "REFERS_TO", source, target)
                        deterministic += 1
                        continue
                    if status == "skip" or not candidates:
                        continue
                else:
                    candidates = _reference_candidates(state, scope, ref)
                    if not candidates:
                        continue

                task_id = f"policy{state['state_index']}_{scope.replace(':', '_')}_{ref}_direct"
                tasks.append(
                    {
                        "id": task_id,
                        "type": "REFERS_TO",
                        "state": state,
                        "source": source,
                        "scope": scope,
                        "mention": node_text,
                        "reference_type": (
                            "DiscourseSituation"
                            if _is_situation_discourse_mention(node_text)
                            else "Discourse"
                        ),
                        "node_text": node_text,
                        "sentence": record.get("sentence", ""),
                        "candidates": candidates,
                    }
                )
                ambiguous += 1

    return deterministic, ambiguous


def _entity_is_other_person(entity: Any, text: str = "") -> bool:
    if not isinstance(entity, dict):
        return False

    head = _normalize_text(entity.get("Head"))
    modifiers = {_normalize_text(item) for item in _description_modifiers(entity)}
    normalized_text = _normalize_text(text)

    person_heads = {
        "user",
        "users",
        "person",
        "persons",
        "people",
        "individual",
        "individuals",
        "others",
    }

    if normalized_text in {
        "others",
        "other users",
        "another user",
        "other people",
        "another person",
    }:
        return True
    if head == "others":
        return True
    if head in person_heads and any(
        modifier in {"other", "another"} for modifier in modifiers
    ):
        return True
    return False


def _node_is_other_person(node: Dict, kind: str) -> bool:
    if kind == "Condition":
        entity = _condition_entity(node)
        if entity is not None:
            return _entity_is_other_person(entity, _node_text(node))

    pseudo_entity = {
        "Head": node.get("Head"),
        "DescriptionGroups": node.get("DescriptionGroups", []),
    }
    return _entity_is_other_person(pseudo_entity, _node_text(node))


def _reference_is_other_person(reference: Dict) -> bool:
    text = _normalize_text(reference.get("Text"))
    return text in {
        "others",
        "other users",
        "another user",
        "other people",
        "another person",
    }


def _collect_distinct_edges(state: Dict) -> int:
    added = 0

    for scope, record in state["inventory"].items():
        subject_root = record.get("subject_root")
        if not isinstance(subject_root, str):
            continue
        target = _ref_endpoint(scope, subject_root)
        nodes = record.get("nodes", {})
        kinds = record.get("kinds", {})

        for ref, node in nodes.items():
            kind = kinds.get(ref)
            if kind not in SEMANTIC_KINDS or ref == subject_root:
                continue

            if _node_is_other_person(node, kind):
                if kind == "Condition" and _condition_entity(node) is not None:
                    source = _ref_endpoint(scope, ref, "Body.Entity")
                else:
                    source = _ref_endpoint(scope, ref)
                if _add_edge(state, "DISTINCT_FROM", source, target):
                    added += 1

            for path, reference in _reference_paths(node):
                if not _reference_is_other_person(reference):
                    continue
                source = _ref_endpoint(scope, ref, path)
                if _add_edge(state, "DISTINCT_FROM", source, target):
                    added += 1

    return added


def _service_fallback_target(task: Dict) -> Optional[Dict]:
    """Return issuer fallback for an ambiguous generic-service task, if available."""
    if task.get("type") != "REFERS_TO":
        return None
    if task.get("reference_type") != "ServiceReference":
        return None
    state = task.get("state")
    if not isinstance(state, dict):
        return None
    context = state.get("context")
    if not isinstance(context, dict):
        return None
    if _nonempty_text(context.get("Organization")) is None:
        return None
    return _context_endpoint("Organization")


def _task_prompt_record(task: Dict) -> Dict:
    base = {
        "id": task["id"],
        "Type": task["type"],
        "Source": task["source"],
        "Mention": task.get("mention", ""),
        "Sentence": task.get("sentence", ""),
        "Candidates": task.get("candidates", []),
    }

    if task["type"] == "EXAMPLE_OF":
        base["AnchorType"] = task.get("anchor_type", "")
        base["AnchorText"] = task.get("anchor_text", "")
    else:
        base["ReferenceType"] = task.get("reference_type", "")
        base["NodeText"] = task.get("node_text", "")

    return base


def _build_enrich_prompt(tasks: List[Dict]) -> str:
    payload = [_task_prompt_record(task) for task in tasks]
    return f"""
Resolve each task by selecting one exact Candidate.Target or null.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return:
[
  {{
    "id": "...",
    "Target": {{"Scope": "...", "Ref": "...", "Path": "..."}} |
              {{"Scope": "...", "Ref": "..."}} |
              {{"Context": "Organization"}} | null
  }}
]
""".strip()


def _parse_llm_targets(response: str) -> Dict[str, Optional[Dict]]:
    parsed = parse_json_response(response)

    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        records = parsed["results"]
    elif isinstance(parsed, dict) and "id" in parsed:
        records = [parsed]
    else:
        raise ValueError("Enrichment response must be a list of results.")

    out: Dict[str, Optional[Dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        task_id = record.get("id")
        target = record.get("Target")
        if not isinstance(task_id, str):
            continue
        if target is None:
            out[task_id] = None
        elif isinstance(target, dict):
            out[task_id] = target
    return out


def _candidate_keys(task: Dict) -> Set[Tuple]:
    out: Set[Tuple] = set()
    for candidate in task.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        target = candidate.get("Target")
        if isinstance(target, dict):
            out.add(_endpoint_key(target))
    return out


def _apply_llm_target(task: Dict, target: Dict) -> None:
    state = task["state"]
    _add_edge(state, task["type"], task["source"], target)

    if task["type"] == "EXAMPLE_OF":
        _finalize_example_item(
            state,
            task["example_index"],
            task["example_ref"],
        )


def _resolve_ambiguous_tasks(
    llm_client: LLMClient,
    tasks: List[Dict],
    *,
    batch_size: int,
    max_retries: int,
) -> Tuple[int, int, int]:
    resolved = 0
    explicit_unresolved = 0
    failed = 0

    for start in range(0, len(tasks), batch_size):
        remaining = tasks[start : start + batch_size]

        for attempt in range(1, max_retries + 1):
            if not remaining:
                break

            try:
                response = llm_client.invoke(
                    _build_enrich_prompt(remaining),
                    system_prompt=ENRICH_SYSTEM_PROMPT,
                )
                if response is None or (
                    isinstance(response, str) and not response.strip()
                ):
                    raise ValueError("Empty enrichment response.")

                parsed = _parse_llm_targets(response)
                retry: List[Dict] = []

                for task in remaining:
                    if task["id"] not in parsed:
                        retry.append(task)
                        continue

                    target = parsed[task["id"]]
                    if target is None:
                        fallback = _service_fallback_target(task)
                        if fallback is not None:
                            _apply_llm_target(task, fallback)
                            resolved += 1
                            logger.info(
                                "PolicyEnrich | service fallback to organization | id=%s | mention=%r",
                                task["id"],
                                task.get("mention", ""),
                            )
                        else:
                            explicit_unresolved += 1
                            logger.info(
                                "PolicyEnrich | LLM left unresolved | id=%s | type=%s | mention=%r",
                                task["id"],
                                task["type"],
                                task.get("mention", ""),
                            )
                        continue

                    if _endpoint_key(target) not in _candidate_keys(task):
                        logger.warning(
                            "PolicyEnrich | invalid LLM target | id=%s | attempt=%d | target=%s",
                            task["id"],
                            attempt,
                            target,
                        )
                        retry.append(task)
                        continue

                    _apply_llm_target(task, target)
                    resolved += 1

                remaining = retry

            except Exception as exc:
                logger.warning(
                    "PolicyEnrich | batch retry | attempt=%d/%d | error=%s",
                    attempt,
                    max_retries,
                    exc,
                )

        for task in remaining:
            fallback = _service_fallback_target(task)
            if fallback is not None:
                _apply_llm_target(task, fallback)
                resolved += 1
                logger.warning(
                    "PolicyEnrich | service fallback after retries | id=%s | mention=%r",
                    task["id"],
                    task.get("mention", ""),
                )
                continue

            failed += 1
            logger.error(
                "PolicyEnrich | unresolved after retries | id=%s | type=%s | mention=%r",
                task["id"],
                task["type"],
                task.get("mention", ""),
            )

    return resolved, explicit_unresolved, failed


def _ensure_policy_graph(rule: Dict) -> Dict:
    policy_graph = rule.get("Policy_Graph")
    if not isinstance(policy_graph, dict):
        policy_graph = {"Root": None, "ExampleFrames": []}
        rule["Policy_Graph"] = policy_graph

    # Preserve the routing manifest produced by policy_graph.py.
    policy_graph.setdefault("Root", None)
    policy_graph.setdefault("ExampleFrames", [])
    return policy_graph


def _prepare_state(
    rule: Dict,
    *,
    state_index: int,
    organization: str,
) -> Dict:
    policy_graph = _ensure_policy_graph(rule)
    context = {
        "Organization": organization,
    }
    policy_graph["Context"] = context

    inventory = _build_inventory(rule)
    example_records, example_ref_by_index = _planned_example_records(rule)

    edges: List[Dict] = []
    edge_keys: Set[Tuple] = set()
    raw_edges = policy_graph.get("Edges", [])
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not _validate_edge(edge, inventory, example_records, context):
                continue
            # Recompute generic-service references on every run under the current
            # same-sentence/scope rule. Cross-sentence named services are ignored.
            if _is_recomputable_service_fallback_edge(edge, inventory):
                continue
            key = _edge_key(edge)
            if key in edge_keys:
                continue
            edges.append(copy.deepcopy(edge))
            edge_keys.add(key)

    return {
        "state_index": state_index,
        "policy_id": str(rule.get("ID", state_index)),
        "rule": rule,
        "policy_graph": policy_graph,
        "context": context,
        "inventory": inventory,
        "example_records": example_records,
        "example_ref_by_index": example_ref_by_index,
        "edges": edges,
        "edge_keys": edge_keys,
    }


def _assert_local_graphs_unchanged(rule: Dict, snapshot: Dict) -> None:
    current = {
        "Main_Clause_Elements": rule.get("Main_Clause_Elements"),
        "Example_Sentences_Elements": rule.get("Example_Sentences_Elements"),
    }
    if current != snapshot:
        raise AssertionError(
            f"enrich_policy_graph modified immutable local graphs for policy {rule.get('ID')!r}."
        )


def validate_enriched_policy_graph(rule: Dict) -> List[str]:
    """Validate Policy_Graph enrichment without changing the rule."""
    errors: List[str] = []
    if not isinstance(rule, dict):
        return ["rule:not_dict"]

    policy_graph = rule.get("Policy_Graph")
    if not isinstance(policy_graph, dict):
        return ["policy_graph:missing"]

    context = policy_graph.get("Context")
    if not isinstance(context, dict):
        errors.append("context:missing")
        context = {}
    else:
        if _nonempty_text(context.get("Organization")) is None:
            errors.append("context:organization_missing")

    inventory = _build_inventory(rule)
    example_records, _ = _planned_example_records(rule)

    edges = policy_graph.get("Edges")
    if not isinstance(edges, list):
        errors.append("edges:not_list")
        return errors

    seen: Set[Tuple] = set()
    for index, edge in enumerate(edges):
        if not _validate_edge(edge, inventory, example_records, context):
            errors.append(f"edge[{index}]:invalid")
            continue
        key = _edge_key(edge)
        if key in seen:
            errors.append(f"edge[{index}]:duplicate")
        seen.add(key)

    return errors


def enrich_policy_graph_batch(
    llm_client: LLMClient,
    rules: List[Dict],
    organization: str,
    batch_size: int = 10,
    max_retries: int = 2,
) -> List[Dict]:
    """Enrich existing Policy_Graphs with semantic edges.

    Responsibilities of this pass:
    - add Context.Organization,
    - attach ordinary examples with EXAMPLE_OF,
    - resolve remaining references with REFERS_TO,
    - add explicit entity inequality constraints with DISTINCT_FROM.

    This pass never changes Subject_Elements, Predicate_Elements, Object_Elements,
    Condition_Elements, or Frame_Elements. It mutates and returns ``rules`` in the
    same style as ``build_policy_graph_batch``.
    """
    if not isinstance(rules, list):
        raise TypeError("rules must be a list.")
    if _nonempty_text(organization) is None:
        raise ValueError("organization must be a non-empty string.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")

    usage_before = _usage_snapshot(llm_client)
    tasks: List[Dict] = []
    states: List[Dict] = []

    deterministic_examples = 0
    ambiguous_examples = 0
    deterministic_references = 0
    ambiguous_references = 0
    distinct_edges = 0

    snapshots: Dict[int, Dict] = {}

    for state_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue

        snapshots[state_index] = copy.deepcopy(
            {
                "Main_Clause_Elements": rule.get("Main_Clause_Elements"),
                "Example_Sentences_Elements": rule.get("Example_Sentences_Elements"),
            }
        )

        state = _prepare_state(
            rule,
            state_index=state_index,
            organization=organization.strip(),
        )
        states.append(state)

        det_ex, amb_ex = _collect_example_work(state, tasks)
        deterministic_examples += det_ex
        ambiguous_examples += amb_ex

        det_ref, amb_ref = _collect_reference_work(state, tasks)
        deterministic_references += det_ref
        ambiguous_references += amb_ref

        distinct_edges += _collect_distinct_edges(state)

    llm_resolved, llm_null, llm_failed = _resolve_ambiguous_tasks(
        llm_client,
        tasks,
        batch_size=batch_size,
        max_retries=max_retries,
    )

    invalid_rules = 0
    for state in states:
        rule = state["rule"]
        # Revalidate after any Example_Elements transformation and LLM additions.
        example_records, _ = _planned_example_records(rule)
        valid_edges = []
        seen: Set[Tuple] = set()
        for edge in state["edges"]:
            if not _validate_edge(
                edge, state["inventory"], example_records, state["context"]
            ):
                continue
            key = _edge_key(edge)
            if key in seen:
                continue
            seen.add(key)
            valid_edges.append(edge)

        valid_edges.sort(key=_edge_sort_key)
        state["policy_graph"]["Edges"] = valid_edges

        _assert_local_graphs_unchanged(rule, snapshots[state["state_index"]])

        errors = validate_enriched_policy_graph(rule)
        if errors:
            invalid_rules += 1
            logger.error(
                "PolicyEnrich | validation failed | policy=%s | errors=%s",
                state["policy_id"],
                "; ".join(errors),
            )

    if llm_failed or invalid_rules:
        raise ValueError(
            "Policy enrichment is incomplete; refusing to save lossy graphs: "
            f"llm_failed={llm_failed}, invalid_rules={invalid_rules}"
        )

    usage = _usage_since(llm_client, usage_before)
    logger.info(
        "PolicyEnrich | completed | policies=%d | deterministic_examples=%d | "
        "ambiguous_examples=%d | deterministic_references=%d | ambiguous_references=%d | "
        "distinct_edges=%d | llm_tasks=%d | llm_resolved=%d | llm_null=%d | "
        "llm_failed=%d | invalid_rules=%d | api_calls=%s | input_tokens=%s | "
        "output_tokens=%s | total_tokens=%s",
        len(states),
        deterministic_examples,
        ambiguous_examples,
        deterministic_references,
        ambiguous_references,
        distinct_edges,
        len(tasks),
        llm_resolved,
        llm_null,
        llm_failed,
        invalid_rules,
        usage.get("api_calls", "?"),
        usage.get("input_tokens", "?"),
        usage.get("output_tokens", "?"),
        usage.get("total_tokens", "?"),
    )

    return rules
