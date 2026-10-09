import copy
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

LOCAL_NODE_KINDS = {
    "Subject",
    "Predicate",
    "Object",
    "Condition",
    "Frame",
    "Example",
    "Context",
}
GLOBAL_KINDS = {"Entity", "Concept", "Action"}

GENERIC_SERVICE_MENTIONS = {"this service", "our service", "the service"}
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


def nonempty_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def normalize_text(value: Any) -> str:
    text = nonempty_text(value)
    if text is None:
        return ""
    text = text.replace("’", "'").replace("`", "'").strip().casefold()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(' \t\r\n.,;:!?()[]{}"')
    if text.endswith("'s"):
        text = text[:-2].rstrip()
    elif text.endswith("'"):
        text = text[:-1].rstrip()
    return text


def singularize_simple(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    irregular = {
        "people": "person",
        "persons": "person",
        "users": "user",
        "others": "other user",
        "services": "service",
        "policies": "policy",
        "activities": "activity",
        "systems": "system",
        "facilities": "facility",
        "methods": "method",
        "networks": "network",
        "accounts": "account",
        "servers": "server",
        "regulations": "regulation",
        "laws": "law",
    }
    if text in irregular:
        return irregular[text]
    if len(text) > 4 and text.endswith("ies"):
        return text[:-3] + "y"
    if len(text) > 4 and text.endswith("ses"):
        return text[:-2]
    if len(text) > 3 and text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text


def description_modifiers(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return []
    groups = value.get("DescriptionGroups")
    if not isinstance(groups, list):
        return []
    out: List[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        modifier = nonempty_text(group.get("Modifier"))
        if modifier is not None:
            out.append(modifier)
    return out


def action_modifiers(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return []
    raw = value.get("Modifiers")
    if not isinstance(raw, list):
        return []
    return [text for item in raw if (text := nonempty_text(item)) is not None]


def action_operators(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return []
    raw = value.get("Operators")
    if not isinstance(raw, list):
        return []
    return [text for item in raw if (text := nonempty_text(item)) is not None]


def _group_path_text(path: Tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in path) or "root"


def _description_group_id(owner_occurrence: str, path: Tuple[Any, ...]) -> str:
    return f"{owner_occurrence}#desc:{_group_path_text(path)}"


def _relation_group_id(owner_occurrence: str, path: Tuple[Any, ...]) -> str:
    return f"{owner_occurrence}#rel:{_group_path_text(path)}"


def _joined_path(prefix: str, path: Tuple[Any, ...]) -> str:
    suffix = ".".join(str(part) for part in path)
    return f"{prefix}.{suffix}" if suffix else prefix


def _structured_description_groups(
    owner_occurrence: str,
    value: Any,
    *,
    source_id: str,
    policy_id: str,
    scope: str,
    ref: str,
    path_prefix: str,
) -> List[Dict]:
    """Preserve explicit AND/OR DescriptionGroup branches without virtual nodes."""
    out: List[Dict] = []

    def visit(current: Any, path: Tuple[Any, ...] = ()) -> None:
        if isinstance(current, list):
            for index, child in enumerate(current):
                visit(child, path + (index,))
            return
        if not isinstance(current, dict):
            return

        operator = nonempty_text(current.get("Operator"))
        items = current.get("Items")
        if operator and isinstance(items, list):
            operator = operator.upper()
            group_id = _description_group_id(owner_occurrence, path)
            structured_items: List[Dict] = []
            for position, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Unsupported DescriptionGroup item at {group_id}[{position}]: expected dict."
                    )
                modifier = nonempty_text(item.get("Modifier"))
                relation = item.get("Relation")
                event = item.get("Event")
                nested_operator = nonempty_text(item.get("Operator"))
                nested_items = item.get("Items")
                branch_kinds = [
                    modifier is not None,
                    isinstance(relation, dict),
                    isinstance(event, dict),
                    bool(nested_operator and isinstance(nested_items, list)),
                ]
                if sum(branch_kinds) != 1:
                    raise ValueError(
                        f"Unsupported DescriptionGroup branch at {group_id}[{position}]: "
                        "expected exactly one of Modifier, Relation, Event, or nested Operator/Items."
                    )

                if modifier is not None:
                    structured_items.append(
                        {
                            "Position": position,
                            "Kind": "Modifier",
                            "Text": modifier,
                        }
                    )
                elif isinstance(relation, dict):
                    structured_items.append(
                        {
                            "Position": position,
                            "Kind": "Relation",
                            "RelationGroupID": _relation_group_id(
                                owner_occurrence, path
                            ),
                        }
                    )
                elif isinstance(event, dict):
                    event_path = _joined_path(
                        path_prefix,
                        path + ("Items", position, "Event"),
                    )
                    structured_items.append(
                        {
                            "Position": position,
                            "Kind": "Event",
                            "EventOccurrence": make_occurrence_id(
                                source_id, policy_id, scope, ref, event_path
                            ),
                        }
                    )
                else:
                    child_path = path + ("Items", position)
                    structured_items.append(
                        {
                            "Position": position,
                            "Kind": "Group",
                            "GroupID": _description_group_id(
                                owner_occurrence, child_path
                            ),
                        }
                    )

                visit(item, path + ("Items", position))

            if structured_items:
                out.append(
                    {
                        "GroupID": group_id,
                        "Operator": operator,
                        "Items": structured_items,
                    }
                )
            return

        for key, child in current.items():
            if key not in {"Relation", "Event"}:
                visit(child, path + (key,))

    visit(value)
    out.sort(key=lambda group: group["GroupID"])
    return out


def _relation_signature(value: Any) -> List[str]:
    """Return a ref-independent structural relation signature for fusion filtering."""
    signatures: Set[str] = set()

    def visit(current: Any) -> None:
        if isinstance(current, list):
            for item in current:
                visit(item)
            return
        if not isinstance(current, dict):
            return

        relation = current.get("Relation")
        if isinstance(relation, dict):
            predicate = normalize_text(relation.get("Predicate"))
            if predicate:
                has_source = isinstance(relation.get("Source"), dict)
                has_target = isinstance(relation.get("Target"), dict)
                direction = (
                    "in"
                    if has_source and not has_target
                    else "out" if has_target and not has_source else "linked"
                )
                signatures.add(f"rel:{direction}:{predicate}")

        event = current.get("Event")
        if isinstance(event, dict):
            action = normalize_text(event.get("Action"))
            operators = tuple(
                normalize_text(item)
                for item in event.get("Operators", [])
                if normalize_text(item)
            )
            modifiers = tuple(
                normalize_text(item)
                for item in event.get("Modifiers", [])
                if normalize_text(item)
            )
            if action:
                signatures.add(
                    "event:"
                    + action
                    + (":" + ",".join(operators) if operators else "")
                    + (":" + ",".join(modifiers) if modifiers else "")
                )

        arguments = current.get("Arguments")
        if isinstance(arguments, list):
            for argument in arguments:
                if not isinstance(argument, dict):
                    continue
                role = normalize_text(argument.get("Role"))
                marker = normalize_text(argument.get("Marker"))
                if role:
                    signatures.add(f"arg:{role}:{marker}" if marker else f"arg:{role}")

        for child in current.values():
            visit(child)

    visit(value)
    return sorted(signatures)


def condition_entity(node: Any) -> Optional[Dict]:
    if not isinstance(node, dict):
        return None
    body = node.get("Body")
    if not isinstance(body, dict):
        return None
    entity = body.get("Entity")
    return entity if isinstance(entity, dict) else None


def condition_event(node: Any) -> Optional[Dict]:
    if not isinstance(node, dict):
        return None
    body = node.get("Body")
    if not isinstance(body, dict):
        return None
    event = body.get("Event")
    return event if isinstance(event, dict) else None


def node_text(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    return nonempty_text(node.get("Text")) or ""


def node_head(node: Any, kind: Optional[str] = None) -> str:
    if not isinstance(node, dict):
        return ""
    head = nonempty_text(node.get("Head"))
    if head is not None:
        return head
    if kind == "Predicate":
        action = nonempty_text(node.get("Action"))
        if action is not None:
            return action
    entity = condition_entity(node)
    if entity is not None:
        head = nonempty_text(entity.get("Head"))
        if head is not None:
            return head
    return ""


def walk_path(value: Any, path: Optional[str]) -> Any:
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


def ref_endpoint(scope: str, ref: str, path: Optional[str] = None) -> Dict[str, str]:
    out = {"Scope": scope, "Ref": ref}
    if path:
        out["Path"] = path
    return out


def occurrence_endpoint(occurrence_id: str) -> Dict[str, str]:
    return {"Occurrence": occurrence_id}


def global_endpoint(global_id: str) -> Dict[str, str]:
    return {"Global": global_id}


def _safe_id_part(value: Any) -> str:
    text = nonempty_text(str(value)) or ""
    text = text.replace("|", "_")
    return text


def make_occurrence_id(
    source_id: str,
    policy_id: str,
    scope: str,
    ref: str,
    path: Optional[str] = None,
) -> str:
    parts = [
        "occ",
        _safe_id_part(source_id),
        _safe_id_part(policy_id),
        _safe_id_part(scope),
        _safe_id_part(ref),
    ]
    if path:
        parts.append(_safe_id_part(path))
    return "|".join(parts)


def make_context_occurrence_id(source_id: str, name: str = "Organization") -> str:
    return "|".join(["ctx", _safe_id_part(source_id), _safe_id_part(name)])


def make_policy_key(source_id: str, policy_id: str) -> str:
    return f"{source_id}:{policy_id}"


def edge_key(edge: Any) -> str:
    if not isinstance(edge, dict):
        return "invalid"
    return json.dumps(edge, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def add_edge(graph: Dict, edge: Dict) -> bool:
    if not isinstance(edge, dict):
        return False
    key = edge_key(edge)
    keys: Set[str] = graph.setdefault("_edge_keys", set())
    if key in keys:
        return False
    graph.setdefault("Edges", []).append(edge)
    keys.add(key)
    return True


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


def _graph_nodes(element: Any, field: str) -> Dict[str, Dict]:
    if not isinstance(element, dict):
        return {}
    graph = element.get(field)
    if not isinstance(graph, dict):
        return {}
    nodes = graph.get("Nodes")
    return nodes if isinstance(nodes, dict) else {}


def _description_has_relation(value: Any) -> bool:
    if isinstance(value, dict):
        if "Relation" in value and isinstance(value.get("Relation"), dict):
            return True
        if "Event" in value and isinstance(value.get("Event"), dict):
            return True
        return any(_description_has_relation(child) for child in value.values())
    if isinstance(value, list):
        return any(_description_has_relation(child) for child in value)
    return False


def _is_generic_service_mention(text: Any) -> bool:
    return normalize_text(text) in GENERIC_SERVICE_MENTIONS


def _surface_service_name(text: Any) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
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


def named_service_name(node: Any, kind: Optional[str] = None) -> str:
    if not isinstance(node, dict):
        return ""
    head = normalize_text(node_head(node, kind))
    if head != "service":
        return ""
    text = node_text(node)
    if _is_generic_service_mention(text):
        return ""
    modifiers = []
    source = condition_entity(node) if kind == "Condition" else node
    for modifier in description_modifiers(source):
        normalized = normalize_text(modifier)
        if normalized and normalized not in GENERIC_SERVICE_MODIFIERS:
            modifiers.append(modifier.strip())
    if modifiers:
        return " ".join(modifiers)
    return _surface_service_name(text)


def _entity_surface(entity: Dict, fallback_text: str = "") -> str:
    head = nonempty_text(entity.get("Head"))
    modifiers = description_modifiers(entity)
    if head:
        if modifiers:
            return " ".join(modifiers + [head]).strip()
        return head
    return fallback_text


def _condition_semantic_record(node: Dict) -> Tuple[str, str, List[str], bool]:
    entity = condition_entity(node)
    if entity is not None:
        text = _entity_surface(entity, node_text(node))
        head = nonempty_text(entity.get("Head")) or ""
        modifiers = description_modifiers(entity)
        return (
            text,
            head,
            modifiers,
            _description_has_relation(entity.get("DescriptionGroups", [])),
        )
    event = condition_event(node)
    if event is not None:
        action = nonempty_text(event.get("Action")) or ""
        return action or node_text(node), action, action_modifiers(event), True
    return node_text(node), "", [], False


def _classify_fusion(
    *,
    organization: str,
    node: Dict,
    local_kind: str,
    text: str,
    head: str,
    modifiers: List[str],
    path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], bool]:
    normalized_text = normalize_text(text)
    normalized_org = normalize_text(organization)

    if local_kind in {"Frame", "Example"}:
        return None, None, False

    if path and path.endswith("Reference"):
        return None, None, False

    if local_kind == "Predicate" or (path and path.split(".")[-1] == "Event"):
        action = normalize_text(head or text)
        if not action:
            return None, None, False
        coordination = node.get("Coordination") if isinstance(node, dict) else None
        if (
            local_kind == "Predicate"
            and isinstance(coordination, dict)
            and nonempty_text(node.get("Action")) is None
        ):
            return None, None, False
        return "Action", None, True

    if local_kind == "Context":
        return "Entity", "Organization", True

    if normalized_text and normalized_text == normalized_org:
        return "Entity", "Organization", True

    service_name = named_service_name(
        node, "Condition" if local_kind == "Condition" else local_kind
    )
    if service_name:
        return "Entity", "Service", True

    if _is_generic_service_mention(text):
        # Referential mention, not a generic Service concept.
        return None, None, False

    if local_kind == "Condition":
        marker = nonempty_text(node.get("Marker"))
        if marker is not None and path != "Body.Entity":
            return None, None, False
        if condition_entity(node) is None and not (
            path and path.endswith("Body.Entity")
        ):
            return None, None, False

    coordination = node.get("Coordination") if isinstance(node, dict) else None
    if isinstance(coordination, dict) and not normalize_text(head):
        return None, None, False

    if not normalized_text and not normalize_text(head):
        return None, None, False
    return "Concept", None, True


def _base_occurrence_record(
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    scope: str,
    ref: str,
    node: Dict,
    local_kind: str,
    path: Optional[str] = None,
    path_value: Any = None,
) -> Dict:
    # Whole-node occurrences preserve the original local node surface text.
    # Semantic facets (Body.Entity / Body.Event / Reference) get their own text.
    text = node_text(node)
    head = node_head(node, local_kind)
    modifiers = (
        action_modifiers(node)
        if local_kind == "Predicate"
        else description_modifiers(node)
    )
    operators = action_operators(node) if local_kind == "Predicate" else []
    description_source: Any = node.get("DescriptionGroups", [])
    description_path_prefix = "DescriptionGroups"
    has_relations = _description_has_relation(description_source)
    relation_signature = _relation_signature(description_source)
    marker = nonempty_text(node.get("Marker")) if local_kind == "Condition" else None

    if local_kind == "Condition":
        _, semantic_head, semantic_modifiers, semantic_has_relations = (
            _condition_semantic_record(node)
        )
        head = semantic_head
        modifiers = semantic_modifiers
        has_relations = semantic_has_relations
        entity = condition_entity(node)
        event = condition_event(node)
        if entity is not None:
            description_source = entity.get("DescriptionGroups", [])
            description_path_prefix = "Body.Entity.DescriptionGroups"
            relation_signature = _relation_signature(description_source)
            # With an outer condition Marker, Body.Entity is materialized as a
            # separate semantic-facet occurrence and owns these groups/edges.
            # Keeping a second copy on the surface condition occurrence creates
            # dangling RelationGroupID references during global validation.
            if marker is not None and path is None:
                description_source = []
                relation_signature = []
                modifiers = []
        elif event is not None:
            description_source = []
            relation_signature = _relation_signature({"Event": event})
            operators = action_operators(event)

    if path:
        value = path_value if path_value is not None else walk_path(node, path)
        marker = None
        if path.endswith("Body.Entity") and isinstance(value, dict):
            text = _entity_surface(value, node_text(node))
            head = nonempty_text(value.get("Head")) or ""
            modifiers = description_modifiers(value)
            operators = []
            description_source = value.get("DescriptionGroups", [])
            description_path_prefix = f"{path}.DescriptionGroups"
            has_relations = _description_has_relation(description_source)
            relation_signature = _relation_signature(description_source)
        elif path.split(".")[-1] == "Event" and isinstance(value, dict):
            description_source = []
            action = nonempty_text(value.get("Action")) or ""
            text = action or node_text(node)
            head = action
            modifiers = action_modifiers(value)
            operators = action_operators(value)
            has_relations = bool(value.get("Arguments")) or _description_has_relation(
                value
            )
            relation_signature = _relation_signature({"Event": value})
        elif path.endswith("Reference") and isinstance(value, dict):
            text = nonempty_text(value.get("Text")) or ""
            head = text
            modifiers = []
            operators = []
            has_relations = False
            relation_signature = []
        elif isinstance(value, dict):
            text = nonempty_text(value.get("Text")) or text
            head = nonempty_text(value.get("Head")) or head
            modifiers = description_modifiers(value) or modifiers
            operators = action_operators(value) or operators
            description_source = value.get("DescriptionGroups", [])
            description_path_prefix = f"{path}.DescriptionGroups"
            relation_signature = _relation_signature(value)
        elif isinstance(value, str):
            text = value
            head = value
            modifiers = []
            operators = []
            description_source = []
            relation_signature = []

    fusion_kind, entity_type, eligible = _classify_fusion(
        organization=organization,
        node=node,
        local_kind=local_kind,
        text=text,
        head=head,
        modifiers=modifiers,
        path=path,
    )

    occurrence_id = make_occurrence_id(source_id, policy_id, scope, ref, path)
    structured_description_groups = _structured_description_groups(
        occurrence_id,
        description_source,
        source_id=source_id,
        policy_id=policy_id,
        scope=scope,
        ref=ref,
        path_prefix=description_path_prefix,
    )
    record = {
        "ID": occurrence_id,
        "SourceID": source_id,
        "Organization": organization,
        "PolicyID": policy_id,
        "Scope": scope,
        "Ref": ref,
        **({"Path": path} if path else {}),
        "NodeKind": local_kind,
        "Text": text,
        "Head": head,
        "Modifiers": modifiers,
        "Operators": operators,
        **({"Marker": marker} if marker is not None else {}),
        **(
            {"DescriptionGroups": structured_description_groups}
            if structured_description_groups
            else {}
        ),
        "HasRelations": bool(has_relations),
        "FusionKind": fusion_kind,
        "EntityType": entity_type,
        "CanonicalEligible": bool(eligible),
        "_RelationSignature": relation_signature,
    }
    return record


def _validate_sources(sources: List[Dict]) -> List[Dict]:
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list.")

    normalized_sources: List[Dict] = []
    seen_source_ids: Set[str] = set()

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise TypeError(f"sources[{index}] must be a dict.")
        source_id = nonempty_text(source.get("source_id"))
        organization = nonempty_text(source.get("organization"))
        rules = source.get("rules")
        if source_id is None:
            raise ValueError(f"sources[{index}].source_id is required.")
        if "|" in source_id:
            raise ValueError(f"source_id must not contain '|': {source_id!r}")
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate source_id: {source_id!r}")
        if organization is None:
            raise ValueError(f"sources[{index}].organization is required.")
        if not isinstance(rules, list):
            raise TypeError(f"sources[{index}].rules must be a list.")

        seen_source_ids.add(source_id)
        normalized_sources.append(
            {
                "source_id": source_id,
                "organization": organization,
                "rules": rules,
                "path": nonempty_text(source.get("path")),
            }
        )

    return normalized_sources


def _new_graph() -> Dict:
    return {
        "Sources": {},
        "Policies": {},
        "Occurrences": {},
        "CanonicalNodes": {},
        "Edges": [],
        "_edge_keys": set(),
        "_local_lookup": {},
        "_source_rules": {},
    }


def _register_context(graph: Dict, source_id: str, organization: str) -> str:
    context_id = make_context_occurrence_id(source_id, "Organization")
    if context_id not in graph["Occurrences"]:
        graph["Occurrences"][context_id] = {
            "ID": context_id,
            "SourceID": source_id,
            "Organization": organization,
            "PolicyID": None,
            "Scope": "context",
            "Ref": "Organization",
            "NodeKind": "Context",
            "Text": organization,
            "Head": "Organization",
            "Modifiers": [],
            "Operators": [],
            "HasRelations": False,
            "FusionKind": "Entity",
            "EntityType": "Organization",
            "CanonicalEligible": True,
        }
    return context_id


def _register_occurrence(graph: Dict, record: Dict) -> str:
    occurrence_id = record["ID"]
    existing = graph["Occurrences"].get(occurrence_id)
    if existing is not None and existing != record:
        raise ValueError(f"Occurrence collision for {occurrence_id!r}")
    graph["Occurrences"][occurrence_id] = record
    lookup_key = (
        record.get("SourceID"),
        record.get("PolicyID"),
        record.get("Scope"),
        record.get("Ref"),
        record.get("Path"),
    )
    graph["_local_lookup"][lookup_key] = occurrence_id
    return occurrence_id


def _lookup_occurrence(
    graph: Dict,
    source_id: str,
    policy_id: str,
    scope: str,
    ref: str,
    path: Optional[str] = None,
) -> Optional[str]:
    return graph["_local_lookup"].get((source_id, policy_id, scope, ref, path))


def _local_node_info(rule: Dict, scope: str, ref: str) -> Optional[Tuple[Dict, str]]:
    fields = (
        ("Subject_Elements", "Subject", "Subject"),
        ("Predicate_Elements", "Predicate", "Predicate"),
        ("Object_Elements", "Object", "Object"),
        ("Condition_Elements", "Condition", None),
        ("Frame_Elements", "Frame", None),
    )
    element = None
    for candidate_scope, candidate_element in _scope_records(rule):
        if candidate_scope == scope:
            element = candidate_element
            break
    if not isinstance(element, dict):
        return None
    for field, kind, text_field in fields:
        if text_field is not None and nonempty_text(element.get(text_field)) is None:
            continue
        node = _graph_nodes(element, field).get(ref)
        if isinstance(node, dict):
            return node, kind
    return None


def _ensure_path_occurrence(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
    scope: str,
    ref: str,
    path: str,
) -> Optional[str]:
    existing = _lookup_occurrence(graph, source_id, policy_id, scope, ref, path)
    if existing is not None:
        return existing

    node_info = _local_node_info(rule, scope, ref)
    if node_info is None:
        return None
    node, local_kind = node_info
    value = walk_path(node, path)
    if value is None:
        return None

    record = _base_occurrence_record(
        source_id=source_id,
        organization=organization,
        policy_id=policy_id,
        scope=scope,
        ref=ref,
        node=node,
        local_kind=local_kind,
        path=path,
        path_value=value,
    )
    return _register_occurrence(graph, record)


def _register_scope_nodes(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
    scope: str,
    element: Dict,
) -> None:
    field_specs = (
        ("Subject_Elements", "Subject", "Subject"),
        ("Predicate_Elements", "Predicate", "Predicate"),
        ("Object_Elements", "Object", "Object"),
        ("Condition_Elements", "Condition", None),
        ("Frame_Elements", "Frame", None),
    )

    for field, local_kind, text_field in field_specs:
        if text_field is not None and nonempty_text(element.get(text_field)) is None:
            continue
        for ref, node in _graph_nodes(element, field).items():
            if not isinstance(ref, str) or not isinstance(node, dict):
                continue
            record = _base_occurrence_record(
                source_id=source_id,
                organization=organization,
                policy_id=policy_id,
                scope=scope,
                ref=ref,
                node=node,
                local_kind=local_kind,
            )
            _register_occurrence(graph, record)

            if local_kind in {"Subject", "Object"}:
                _register_description_event_occurrences(
                    graph,
                    source_id=source_id,
                    organization=organization,
                    policy_id=policy_id,
                    rule=rule,
                    scope=scope,
                    ref=ref,
                    value=node.get("DescriptionGroups", []),
                    path_prefix="DescriptionGroups",
                )

            if local_kind == "Condition":
                marker = nonempty_text(node.get("Marker"))
                entity = condition_entity(node)
                if entity is not None:
                    _register_description_event_occurrences(
                        graph,
                        source_id=source_id,
                        organization=organization,
                        policy_id=policy_id,
                        rule=rule,
                        scope=scope,
                        ref=ref,
                        value=entity.get("DescriptionGroups", []),
                        path_prefix="Body.Entity.DescriptionGroups",
                    )
                if entity is not None and marker is not None:
                    _ensure_path_occurrence(
                        graph,
                        source_id=source_id,
                        organization=organization,
                        policy_id=policy_id,
                        rule=rule,
                        scope=scope,
                        ref=ref,
                        path="Body.Entity",
                    )
                event = condition_event(node)
                if event is not None:
                    _ensure_path_occurrence(
                        graph,
                        source_id=source_id,
                        organization=organization,
                        policy_id=policy_id,
                        rule=rule,
                        scope=scope,
                        ref=ref,
                        path="Body.Event",
                    )


def _register_examples(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
) -> None:
    examples = rule.get("Examples_Elements", [])
    if not isinstance(examples, list):
        return
    for index, item in enumerate(examples):
        if not isinstance(item, dict):
            continue
        ref = nonempty_text(item.get("Ref")) or f"e{index}"
        text = (
            nonempty_text(item.get("Text")) or nonempty_text(item.get("Example")) or ""
        )
        occurrence_id = make_occurrence_id(source_id, policy_id, "examples", ref)
        _register_occurrence(
            graph,
            {
                "ID": occurrence_id,
                "SourceID": source_id,
                "Organization": organization,
                "PolicyID": policy_id,
                "Scope": "examples",
                "Ref": ref,
                "NodeKind": "Example",
                "Text": text,
                "Head": "",
                "Modifiers": [],
                "Operators": [],
                "HasRelations": False,
                "FusionKind": None,
                "EntityType": None,
                "CanonicalEligible": False,
            },
        )


def _register_policy(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
) -> None:
    policy_key = make_policy_key(source_id, policy_id)
    policy_graph = (
        rule.get("Policy_Graph") if isinstance(rule.get("Policy_Graph"), dict) else {}
    )

    context = policy_graph.get("Context") if isinstance(policy_graph, dict) else None
    if isinstance(context, dict):
        declared = nonempty_text(context.get("Organization"))
        if declared is not None and normalize_text(declared) != normalize_text(
            organization
        ):
            raise ValueError(
                f"Policy {policy_key} declares Context.Organization={declared!r}, "
                f"but source organization is {organization!r}."
            )

    root_id = None
    root = policy_graph.get("Root") if isinstance(policy_graph, dict) else None
    if isinstance(root, dict):
        scope = root.get("Scope")
        ref = root.get("Ref")
        if isinstance(scope, str) and isinstance(ref, str):
            root_id = _lookup_occurrence(graph, source_id, policy_id, scope, ref)

    example_frames: List[str] = []
    frames = (
        policy_graph.get("ExampleFrames", []) if isinstance(policy_graph, dict) else []
    )
    if isinstance(frames, list):
        for item in frames:
            if not isinstance(item, dict):
                continue
            scope = item.get("Scope")
            ref = item.get("Ref")
            if isinstance(scope, str) and isinstance(ref, str):
                occurrence_id = _lookup_occurrence(
                    graph, source_id, policy_id, scope, ref
                )
                if occurrence_id:
                    example_frames.append(occurrence_id)

    graph["Policies"][policy_key] = {
        "SourceID": source_id,
        "Organization": organization,
        "PolicyID": policy_id,
        "OriginalPolicy": nonempty_text(rule.get("Original_Policy")) or "",
        "Root": root_id,
        "ExampleFrames": example_frames,
    }


def _add_structural_edge(
    graph: Dict,
    *,
    edge_type: str,
    source_id: str,
    target_id: str,
    policy_key: str,
    **attributes: Any,
) -> None:
    edge = {
        "Type": edge_type,
        "Source": occurrence_endpoint(source_id),
        "Target": occurrence_endpoint(target_id),
        "Origin": "local_structure",
        "PolicyKey": policy_key,
    }
    for key, value in attributes.items():
        if value is not None:
            edge[key] = value
    add_edge(graph, edge)


def _flatten_coordination(
    graph: Dict,
    *,
    owner_occurrence: str,
    coordination: Any,
    source_id: str,
    policy_id: str,
    scope: str,
    policy_key: str,
) -> None:
    if not isinstance(coordination, dict):
        return
    operator = nonempty_text(coordination.get("Operator")) or ""
    items = coordination.get("Items")
    if not isinstance(items, list):
        return
    for position, item in enumerate(items):
        ref = item.get("Ref") if isinstance(item, dict) else None
        if not isinstance(ref, str):
            continue
        target = _lookup_occurrence(graph, source_id, policy_id, scope, ref)
        if target is None:
            continue
        _add_structural_edge(
            graph,
            edge_type="COORD_ITEM",
            source_id=owner_occurrence,
            target_id=target,
            policy_key=policy_key,
            Operator=operator,
            Position=position,
        )


_NON_ACTION_RELATION_PREDICATES = {
    "of",
    "on",
    "for",
    "against",
    "related to",
    "involved in",
    "indifferent to",
    "are in competition with",
}


def _relation_action_parts(predicate: Any) -> Tuple[str, List[str]]:
    text = nonempty_text(predicate)
    if text is None:
        return "", []
    normalized = normalize_text(text)
    if (
        normalized in _NON_ACTION_RELATION_PREDICATES
        or " or " in normalized
        or " and " in normalized
    ):
        return "", []
    for prefix in ("is not supposed to ", "designed to ", "may ", "to "):
        if normalized.startswith(prefix):
            head = text[len(prefix) :].strip()
            return (head, [text[: len(prefix)].strip()]) if head else ("", [])
    return text, []


def _iter_description_events(
    value: Any,
    path: Tuple[Any, ...] = (),
    group: Optional[Tuple[Tuple[Any, ...], str, int]] = None,
) -> Iterable[Tuple[Dict, Tuple[Any, ...], Optional[Tuple[Tuple[Any, ...], str, int]]]]:
    """Yield embedded DescriptionGroup Events with their local path and group scope."""
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_description_events(child, path + (index,), group)
        return
    if not isinstance(value, dict):
        return

    operator = nonempty_text(value.get("Operator"))
    items = value.get("Items")
    if operator and isinstance(items, list):
        for position, child in enumerate(items):
            yield from _iter_description_events(
                child,
                path + ("Items", position),
                (path, operator.upper(), position),
            )
        for key, child in value.items():
            if key not in {"Operator", "Items", "Relation", "Event"}:
                yield from _iter_description_events(child, path + (key,), group)
        return

    event = value.get("Event")
    if isinstance(event, dict):
        yield event, path + ("Event",), group
    for key, child in value.items():
        if key not in {"Relation", "Event"}:
            yield from _iter_description_events(child, path + (key,), group)


def _register_description_event_occurrences(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
    scope: str,
    ref: str,
    value: Any,
    path_prefix: str,
) -> None:
    for _event, event_path, _group in _iter_description_events(value):
        _ensure_path_occurrence(
            graph,
            source_id=source_id,
            organization=organization,
            policy_id=policy_id,
            rule=rule,
            scope=scope,
            ref=ref,
            path=_joined_path(path_prefix, event_path),
        )


def _iter_relations(
    value: Any,
    path: Tuple[Any, ...] = (),
    group: Optional[Tuple[Tuple[Any, ...], str, int]] = None,
) -> Iterable[Tuple[Dict, Optional[Tuple[Tuple[Any, ...], str, int]]]]:
    """Yield relations together with their nearest AND/OR description-group scope."""
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_relations(child, path + (index,), group)
        return
    if not isinstance(value, dict):
        return

    operator = nonempty_text(value.get("Operator"))
    items = value.get("Items")
    if operator and isinstance(items, list):
        for position, child in enumerate(items):
            yield from _iter_relations(
                child,
                path + ("Items", position),
                (path, operator.upper(), position),
            )
        for key, child in value.items():
            if key not in {"Operator", "Items", "Relation"}:
                yield from _iter_relations(child, path + (key,), group)
    else:
        relation = value.get("Relation")
        if isinstance(relation, dict):
            yield relation, group
        for key, child in value.items():
            if key != "Relation":
                yield from _iter_relations(child, path + (key,), group)


def _flatten_relations(
    graph: Dict,
    *,
    owner_occurrence: str,
    value: Any,
    source_id: str,
    policy_id: str,
    scope: str,
    policy_key: str,
) -> None:
    for relation, group in _iter_relations(value):
        predicate = nonempty_text(relation.get("Predicate"))
        if predicate is None:
            continue
        source_ref = None
        target_ref = None
        source = relation.get("Source")
        target = relation.get("Target")
        if isinstance(source, dict) and isinstance(source.get("Ref"), str):
            source_ref = source["Ref"]
        if isinstance(target, dict) and isinstance(target.get("Ref"), str):
            target_ref = target["Ref"]

        if source_ref is None and target_ref is None:
            continue
        source_occurrence = (
            owner_occurrence
            if source_ref is None
            else _lookup_occurrence(graph, source_id, policy_id, scope, source_ref)
        )
        target_occurrence = (
            owner_occurrence
            if target_ref is None
            else _lookup_occurrence(graph, source_id, policy_id, scope, target_ref)
        )
        if source_occurrence is None or target_occurrence is None:
            continue

        attributes: Dict[str, Any] = {
            "Owner": occurrence_endpoint(owner_occurrence),
            "Predicate": predicate,
        }
        action_head, action_operators = _relation_action_parts(predicate)
        if action_head:
            attributes["ActionHead"] = action_head
            attributes["ActionOperators"] = action_operators
        if group is not None:
            group_path, operator, position = group
            attributes.update(
                GroupID=_relation_group_id(owner_occurrence, group_path),
                Operator=operator,
                Position=position,
            )

        _add_structural_edge(
            graph,
            edge_type="RELATION",
            source_id=source_occurrence,
            target_id=target_occurrence,
            policy_key=policy_key,
            **attributes,
        )


def _flatten_event_arguments(
    graph: Dict,
    *,
    event_occurrence: str,
    owner_occurrence: str,
    event: Dict,
    source_id: str,
    policy_id: str,
    scope: str,
    policy_key: str,
) -> None:
    arguments = event.get("Arguments")
    if not isinstance(arguments, list):
        return
    for position, argument in enumerate(arguments):
        if not isinstance(argument, dict):
            continue
        value = argument.get("Value")
        target = None
        is_self = False
        if (
            isinstance(value, dict)
            and value.get("Self") is True
            and set(value) == {"Self"}
        ):
            target = owner_occurrence
            is_self = True
        else:
            arg_ref = value.get("Ref") if isinstance(value, dict) else None
            if isinstance(arg_ref, str):
                target = _lookup_occurrence(graph, source_id, policy_id, scope, arg_ref)
        if target is None:
            continue
        _add_structural_edge(
            graph,
            edge_type="ARGUMENT",
            source_id=event_occurrence,
            target_id=target,
            policy_key=policy_key,
            Role=nonempty_text(argument.get("Role")) or "",
            Marker=nonempty_text(argument.get("Marker")),
            Position=position,
            Self=True if is_self else None,
        )


def _flatten_description_events(
    graph: Dict,
    *,
    owner_occurrence: str,
    value: Any,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
    scope: str,
    ref: str,
    policy_key: str,
    path_prefix: str,
) -> None:
    for event, event_path, _group in _iter_description_events(value):
        full_path = _joined_path(path_prefix, event_path)
        event_occurrence = _ensure_path_occurrence(
            graph,
            source_id=source_id,
            organization=organization,
            policy_id=policy_id,
            rule=rule,
            scope=scope,
            ref=ref,
            path=full_path,
        )
        if event_occurrence is None:
            continue
        _add_structural_edge(
            graph,
            edge_type="HAS_SEMANTIC_BODY",
            source_id=owner_occurrence,
            target_id=event_occurrence,
            policy_key=policy_key,
            BodyKind="Event",
            DescriptionPath=full_path,
        )
        _flatten_event_arguments(
            graph,
            event_occurrence=event_occurrence,
            owner_occurrence=owner_occurrence,
            event=event,
            source_id=source_id,
            policy_id=policy_id,
            scope=scope,
            policy_key=policy_key,
        )


def _flatten_frame_node(
    graph: Dict,
    *,
    source_id: str,
    policy_id: str,
    scope: str,
    ref: str,
    node: Dict,
    policy_key: str,
) -> None:
    frame_occ = _lookup_occurrence(graph, source_id, policy_id, scope, ref)
    if frame_occ is None:
        return
    behavior = node.get("Behavior")
    if isinstance(behavior, dict):
        for role, edge_type in (
            ("Subject", "FRAME_SUBJECT"),
            ("Predicate", "FRAME_PREDICATE"),
            ("Object", "FRAME_OBJECT"),
        ):
            endpoint = behavior.get(role)
            role_ref = endpoint.get("Ref") if isinstance(endpoint, dict) else None
            if isinstance(role_ref, str):
                target = _lookup_occurrence(
                    graph, source_id, policy_id, scope, role_ref
                )
                if target:
                    _add_structural_edge(
                        graph,
                        edge_type=edge_type,
                        source_id=frame_occ,
                        target_id=target,
                        policy_key=policy_key,
                    )
        conditions = behavior.get("Conditions")
        if isinstance(conditions, list):
            for position, item in enumerate(conditions):
                cond_ref = item.get("Ref") if isinstance(item, dict) else None
                if not isinstance(cond_ref, str):
                    continue
                target = _lookup_occurrence(
                    graph, source_id, policy_id, scope, cond_ref
                )
                if target:
                    _add_structural_edge(
                        graph,
                        edge_type="FRAME_CONDITION",
                        source_id=frame_occ,
                        target_id=target,
                        policy_key=policy_key,
                        Position=position,
                    )
    _flatten_coordination(
        graph,
        owner_occurrence=frame_occ,
        coordination=node.get("Coordination"),
        source_id=source_id,
        policy_id=policy_id,
        scope=scope,
        policy_key=policy_key,
    )


def _flatten_condition_node(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    scope: str,
    ref: str,
    node: Dict,
    rule: Dict,
    policy_key: str,
) -> None:
    base_occ = _lookup_occurrence(graph, source_id, policy_id, scope, ref)
    if base_occ is None:
        return

    marker = nonempty_text(node.get("Marker"))
    entity = condition_entity(node)
    if entity is not None:
        if marker is not None:
            facet_occ = _ensure_path_occurrence(
                graph,
                source_id=source_id,
                organization=organization,
                policy_id=policy_id,
                rule=rule,
                scope=scope,
                ref=ref,
                path="Body.Entity",
            )
            relation_owner = facet_occ or base_occ
            if facet_occ:
                _add_structural_edge(
                    graph,
                    edge_type="HAS_SEMANTIC_BODY",
                    source_id=base_occ,
                    target_id=facet_occ,
                    policy_key=policy_key,
                    BodyKind="Entity",
                )
        else:
            relation_owner = base_occ

        _flatten_coordination(
            graph,
            owner_occurrence=relation_owner,
            coordination=entity.get("Coordination"),
            source_id=source_id,
            policy_id=policy_id,
            scope=scope,
            policy_key=policy_key,
        )
        _flatten_relations(
            graph,
            owner_occurrence=relation_owner,
            value=entity.get("DescriptionGroups", []),
            source_id=source_id,
            policy_id=policy_id,
            scope=scope,
            policy_key=policy_key,
        )
        _flatten_description_events(
            graph,
            owner_occurrence=relation_owner,
            value=entity.get("DescriptionGroups", []),
            source_id=source_id,
            organization=organization,
            policy_id=policy_id,
            rule=rule,
            scope=scope,
            ref=ref,
            policy_key=policy_key,
            path_prefix="Body.Entity.DescriptionGroups",
        )

    event = condition_event(node)
    if event is not None:
        event_occ = _ensure_path_occurrence(
            graph,
            source_id=source_id,
            organization=organization,
            policy_id=policy_id,
            rule=rule,
            scope=scope,
            ref=ref,
            path="Body.Event",
        )
        if event_occ:
            _add_structural_edge(
                graph,
                edge_type="HAS_SEMANTIC_BODY",
                source_id=base_occ,
                target_id=event_occ,
                policy_key=policy_key,
                BodyKind="Event",
            )
            _flatten_event_arguments(
                graph,
                event_occurrence=event_occ,
                owner_occurrence=base_occ,
                event=event,
                source_id=source_id,
                policy_id=policy_id,
                scope=scope,
                policy_key=policy_key,
            )


def _flatten_scope(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
    scope: str,
    element: Dict,
) -> None:
    policy_key = make_policy_key(source_id, policy_id)

    for ref, node in _graph_nodes(element, "Subject_Elements").items():
        if not isinstance(node, dict):
            continue
        owner = _lookup_occurrence(graph, source_id, policy_id, scope, ref)
        if owner:
            _flatten_coordination(
                graph,
                owner_occurrence=owner,
                coordination=node.get("Coordination"),
                source_id=source_id,
                policy_id=policy_id,
                scope=scope,
                policy_key=policy_key,
            )
            _flatten_relations(
                graph,
                owner_occurrence=owner,
                value=node.get("DescriptionGroups", []),
                source_id=source_id,
                policy_id=policy_id,
                scope=scope,
                policy_key=policy_key,
            )
            _flatten_description_events(
                graph,
                owner_occurrence=owner,
                value=node.get("DescriptionGroups", []),
                source_id=source_id,
                organization=organization,
                policy_id=policy_id,
                rule=rule,
                scope=scope,
                ref=ref,
                policy_key=policy_key,
                path_prefix="DescriptionGroups",
            )

    for ref, node in _graph_nodes(element, "Object_Elements").items():
        if not isinstance(node, dict):
            continue
        owner = _lookup_occurrence(graph, source_id, policy_id, scope, ref)
        if owner:
            _flatten_coordination(
                graph,
                owner_occurrence=owner,
                coordination=node.get("Coordination"),
                source_id=source_id,
                policy_id=policy_id,
                scope=scope,
                policy_key=policy_key,
            )
            _flatten_relations(
                graph,
                owner_occurrence=owner,
                value=node.get("DescriptionGroups", []),
                source_id=source_id,
                policy_id=policy_id,
                scope=scope,
                policy_key=policy_key,
            )
            _flatten_description_events(
                graph,
                owner_occurrence=owner,
                value=node.get("DescriptionGroups", []),
                source_id=source_id,
                organization=organization,
                policy_id=policy_id,
                rule=rule,
                scope=scope,
                ref=ref,
                policy_key=policy_key,
                path_prefix="DescriptionGroups",
            )

    for ref, node in _graph_nodes(element, "Predicate_Elements").items():
        if not isinstance(node, dict):
            continue
        owner = _lookup_occurrence(graph, source_id, policy_id, scope, ref)
        if owner:
            _flatten_coordination(
                graph,
                owner_occurrence=owner,
                coordination=node.get("Coordination"),
                source_id=source_id,
                policy_id=policy_id,
                scope=scope,
                policy_key=policy_key,
            )

    for ref, node in _graph_nodes(element, "Condition_Elements").items():
        if isinstance(node, dict):
            _flatten_condition_node(
                graph,
                source_id=source_id,
                organization=organization,
                policy_id=policy_id,
                scope=scope,
                ref=ref,
                node=node,
                rule=rule,
                policy_key=policy_key,
            )

    for ref, node in _graph_nodes(element, "Frame_Elements").items():
        if isinstance(node, dict):
            _flatten_frame_node(
                graph,
                source_id=source_id,
                policy_id=policy_id,
                scope=scope,
                ref=ref,
                node=node,
                policy_key=policy_key,
            )


def _resolve_enrichment_endpoint(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
    endpoint: Dict,
) -> Optional[str]:
    if set(endpoint) == {"Context"} and endpoint.get("Context") == "Organization":
        return make_context_occurrence_id(source_id, "Organization")

    scope = endpoint.get("Scope")
    ref = endpoint.get("Ref")
    path = endpoint.get("Path")
    if not isinstance(scope, str) or not isinstance(ref, str):
        return None

    if scope == "examples":
        return _lookup_occurrence(graph, source_id, policy_id, scope, ref)

    if path is None:
        return _lookup_occurrence(graph, source_id, policy_id, scope, ref)
    if not isinstance(path, str) or not path:
        return None
    return _ensure_path_occurrence(
        graph,
        source_id=source_id,
        organization=organization,
        policy_id=policy_id,
        rule=rule,
        scope=scope,
        ref=ref,
        path=path,
    )


def _import_enrichment_edges(
    graph: Dict,
    *,
    source_id: str,
    organization: str,
    policy_id: str,
    rule: Dict,
) -> None:
    policy_graph = rule.get("Policy_Graph")
    if not isinstance(policy_graph, dict):
        return
    edges = policy_graph.get("Edges")
    if not isinstance(edges, list):
        return
    policy_key = make_policy_key(source_id, policy_id)

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        edge_type = edge.get("Type")
        source = edge.get("Source")
        target = edge.get("Target")
        if (
            not isinstance(edge_type, str)
            or not isinstance(source, dict)
            or not isinstance(target, dict)
        ):
            continue
        source_occ = _resolve_enrichment_endpoint(
            graph,
            source_id=source_id,
            organization=organization,
            policy_id=policy_id,
            rule=rule,
            endpoint=source,
        )
        target_occ = _resolve_enrichment_endpoint(
            graph,
            source_id=source_id,
            organization=organization,
            policy_id=policy_id,
            rule=rule,
            endpoint=target,
        )
        if source_occ is None or target_occ is None:
            continue
        imported = {
            "Type": edge_type,
            "Source": occurrence_endpoint(source_occ),
            "Target": occurrence_endpoint(target_occ),
            "Origin": "enrichment",
            "PolicyKey": policy_key,
        }
        if edge_type == "REFERS_TO" and set(target) == {"Context"}:
            imported["ReferenceKind"] = "CONTEXT_ANCHOR"
        add_edge(graph, imported)


def build_occurrence_graph(sources: List[Dict]) -> Dict:
    """Register all local Policy Graph structures in one occurrence graph.

    This pass does not perform cross-policy fusion. Every original node remains an
    occurrence identified by SourceID + PolicyID + Scope + Ref + optional Path.
    """
    normalized_sources = _validate_sources(sources)
    graph = _new_graph()

    for source in normalized_sources:
        source_id = source["source_id"]
        organization = source["organization"]
        rules = source["rules"]
        graph["Sources"][source_id] = {
            "Organization": organization,
            "PolicyCount": len(rules),
            **({"Path": source["path"]} if source.get("path") else {}),
        }
        graph["_source_rules"][source_id] = rules
        _register_context(graph, source_id, organization)

        seen_policy_ids: Set[str] = set()
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            policy_id = nonempty_text(str(rule.get("ID", index))) or str(index)
            if policy_id in seen_policy_ids:
                raise ValueError(
                    f"Duplicate Policy ID within source {source_id!r}: {policy_id!r}"
                )
            seen_policy_ids.add(policy_id)

            for scope, element in _scope_records(rule):
                _register_scope_nodes(
                    graph,
                    source_id=source_id,
                    organization=organization,
                    policy_id=policy_id,
                    rule=rule,
                    scope=scope,
                    element=element,
                )
            _register_examples(
                graph,
                source_id=source_id,
                organization=organization,
                policy_id=policy_id,
                rule=rule,
            )
            _register_policy(
                graph,
                source_id=source_id,
                organization=organization,
                policy_id=policy_id,
                rule=rule,
            )

            for scope, element in _scope_records(rule):
                _flatten_scope(
                    graph,
                    source_id=source_id,
                    organization=organization,
                    policy_id=policy_id,
                    rule=rule,
                    scope=scope,
                    element=element,
                )
            _import_enrichment_edges(
                graph,
                source_id=source_id,
                organization=organization,
                policy_id=policy_id,
                rule=rule,
            )

    return graph


def strip_internal_fields(graph: Dict) -> Dict:
    out = copy.deepcopy(graph)

    def remove_private(value: Any) -> None:
        if isinstance(value, dict):
            for key in list(value):
                if isinstance(key, str) and key.startswith("_"):
                    value.pop(key, None)
                    continue
                remove_private(value[key])
        elif isinstance(value, list):
            for item in value:
                remove_private(item)

    remove_private(out)

    def endpoint_sort(endpoint: Any) -> Tuple[str, str]:
        if isinstance(endpoint, dict):
            if isinstance(endpoint.get("Occurrence"), str):
                return ("Occurrence", endpoint["Occurrence"])
            if isinstance(endpoint.get("Global"), str):
                return ("Global", endpoint["Global"])
        return ("", "")

    edges = out.get("Edges")
    if isinstance(edges, list):
        edges.sort(
            key=lambda edge: (
                str(edge.get("Type") or "") if isinstance(edge, dict) else "",
                endpoint_sort(edge.get("Source") if isinstance(edge, dict) else None),
                endpoint_sort(edge.get("Target") if isinstance(edge, dict) else None),
                str(edge.get("Predicate") or "") if isinstance(edge, dict) else "",
                str(edge.get("PolicyKey") or "") if isinstance(edge, dict) else "",
            )
        )
    return out
