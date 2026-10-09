from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.merge_graph.global_graph_builder import normalize_text

STRUCTURAL_EDGE_TYPES = {
    "FRAME_SUBJECT",
    "FRAME_PREDICATE",
    "FRAME_OBJECT",
    "FRAME_CONDITION",
    "COORD_ITEM",
    "RELATION",
    "ARGUMENT",
    "HAS_SEMANTIC_BODY",
}
SEMANTIC_EDGE_TYPES = {"EXAMPLE_OF", "REFERS_TO", "DISTINCT_FROM", "SAME_REFERENT"}
GLOBAL_EDGE_TYPES = {"CANONICAL_OF", "SUBTYPE_OF"}
ALLOWED_EDGE_TYPES = STRUCTURAL_EDGE_TYPES | SEMANTIC_EDGE_TYPES | GLOBAL_EDGE_TYPES


def _endpoint_kind(endpoint: Any) -> Optional[str]:
    if not isinstance(endpoint, dict) or len(endpoint) != 1:
        return None
    if isinstance(endpoint.get("Occurrence"), str):
        return "Occurrence"
    if isinstance(endpoint.get("Global"), str):
        return "Global"
    return None


def _endpoint_id(endpoint: Dict) -> Optional[str]:
    kind = _endpoint_kind(endpoint)
    if kind is None:
        return None
    return endpoint.get(kind)


def _canonical_map(graph: Dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for edge in graph.get("Edges", []):
        if not isinstance(edge, dict) or edge.get("Type") != "CANONICAL_OF":
            continue
        source = edge.get("Source")
        target = edge.get("Target")
        if _endpoint_kind(source) != "Occurrence" or _endpoint_kind(target) != "Global":
            continue
        occurrence_id = source["Occurrence"]
        global_id = target["Global"]
        if occurrence_id not in out:
            out[occurrence_id] = global_id
    return out


def _subtype_adjacency(graph: Dict) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for edge in graph.get("Edges", []):
        if not isinstance(edge, dict) or edge.get("Type") != "SUBTYPE_OF":
            continue
        source = edge.get("Source")
        target = edge.get("Target")
        if _endpoint_kind(source) == "Global" and _endpoint_kind(target) == "Global":
            out[source["Global"]].add(target["Global"])
    return out


def _find_cycle(adjacency: Dict[str, Set[str]]) -> Optional[List[str]]:
    visiting: Set[str] = set()
    visited: Set[str] = set()
    stack: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        if node in visiting:
            try:
                index = stack.index(node)
            except ValueError:
                index = 0
            return stack[index:] + [node]
        if node in visited:
            return None
        visiting.add(node)
        stack.append(node)
        for target in sorted(adjacency.get(node, set())):
            cycle = dfs(target)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in sorted(adjacency):
        cycle = dfs(node)
        if cycle:
            return cycle
    return None


def validate_global_graph(graph: Dict) -> List[str]:
    errors: List[str] = []
    if not isinstance(graph, dict):
        return ["graph:not_dict"]

    sources = graph.get("Sources")
    policies = graph.get("Policies")
    occurrences = graph.get("Occurrences")
    canonical_nodes = graph.get("CanonicalNodes")
    edges = graph.get("Edges")

    if not isinstance(sources, dict):
        errors.append("sources:not_dict")
        sources = {}
    if not isinstance(policies, dict):
        errors.append("policies:not_dict")
        policies = {}
    if not isinstance(occurrences, dict):
        errors.append("occurrences:not_dict")
        occurrences = {}
    if not isinstance(canonical_nodes, dict):
        errors.append("canonical_nodes:not_dict")
        canonical_nodes = {}
    if not isinstance(edges, list):
        errors.append("edges:not_list")
        return errors

    description_groups: Dict[str, Dict[str, Any]] = {}
    description_relation_refs: List[Tuple[str, str, str, int, str]] = []
    description_event_refs: List[Tuple[str, str, str, int, str]] = []
    description_child_refs: List[Tuple[str, str, str]] = []

    for source_id, source in sources.items():
        if not isinstance(source_id, str) or not source_id:
            errors.append("source:invalid_id")
            continue
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("Organization"), str)
            or not source.get("Organization")
        ):
            errors.append(f"source:{source_id}:organization_missing")

    for policy_key, policy in policies.items():
        if not isinstance(policy, dict):
            errors.append(f"policy:{policy_key}:not_dict")
            continue
        source_id = policy.get("SourceID")
        if source_id not in sources:
            errors.append(f"policy:{policy_key}:unknown_source")
        root = policy.get("Root")
        if root is not None and root not in occurrences:
            errors.append(f"policy:{policy_key}:root_missing")
        frames = policy.get("ExampleFrames", [])
        if not isinstance(frames, list):
            errors.append(f"policy:{policy_key}:example_frames_not_list")
        else:
            for frame_id in frames:
                if frame_id not in occurrences:
                    errors.append(
                        f"policy:{policy_key}:example_frame_missing:{frame_id}"
                    )

    for occurrence_id, occurrence in occurrences.items():
        if not isinstance(occurrence, dict):
            errors.append(f"occurrence:{occurrence_id}:not_dict")
            continue
        if occurrence.get("ID") != occurrence_id:
            errors.append(f"occurrence:{occurrence_id}:id_mismatch")
        source_id = occurrence.get("SourceID")
        if source_id not in sources:
            errors.append(f"occurrence:{occurrence_id}:unknown_source")
        elif normalize_text(occurrence.get("Organization")) != normalize_text(
            sources[source_id].get("Organization")
        ):
            errors.append(f"occurrence:{occurrence_id}:organization_mismatch")

        groups = occurrence.get("DescriptionGroups")
        if groups is not None:
            if not isinstance(groups, list):
                errors.append(f"occurrence:{occurrence_id}:description_groups_not_list")
            else:
                for group_index, group in enumerate(groups):
                    if not isinstance(group, dict):
                        errors.append(
                            f"occurrence:{occurrence_id}:description_group[{group_index}]:not_dict"
                        )
                        continue
                    group_id = group.get("GroupID")
                    operator = group.get("Operator")
                    items = group.get("Items")
                    if not isinstance(group_id, str) or not group_id:
                        errors.append(
                            f"occurrence:{occurrence_id}:description_group[{group_index}]:invalid_id"
                        )
                        continue
                    if group_id in description_groups:
                        errors.append(
                            f"occurrence:{occurrence_id}:description_group_duplicate:{group_id}"
                        )
                    else:
                        description_groups[group_id] = {
                            "Owner": occurrence_id,
                            "Operator": operator,
                        }
                    if operator not in {"AND", "OR"}:
                        errors.append(
                            f"occurrence:{occurrence_id}:description_group:{group_id}:invalid_operator"
                        )
                    if not isinstance(items, list) or not items:
                        errors.append(
                            f"occurrence:{occurrence_id}:description_group:{group_id}:items_invalid"
                        )
                        continue

                    seen_positions: Set[int] = set()
                    for item_index, item in enumerate(items):
                        if not isinstance(item, dict):
                            errors.append(
                                f"occurrence:{occurrence_id}:description_group:{group_id}:item[{item_index}]:not_dict"
                            )
                            continue
                        position = item.get("Position")
                        kind = item.get("Kind")
                        if (
                            not isinstance(position, int)
                            or isinstance(position, bool)
                            or position < 0
                        ):
                            errors.append(
                                f"occurrence:{occurrence_id}:description_group:{group_id}:item[{item_index}]:invalid_position"
                            )
                            continue
                        if position in seen_positions:
                            errors.append(
                                f"occurrence:{occurrence_id}:description_group:{group_id}:duplicate_position:{position}"
                            )
                        seen_positions.add(position)
                        if position != item_index:
                            errors.append(
                                f"occurrence:{occurrence_id}:description_group:{group_id}:position_order_mismatch:{position}"
                            )

                        if kind == "Modifier":
                            if (
                                not isinstance(item.get("Text"), str)
                                or not item["Text"].strip()
                            ):
                                errors.append(
                                    f"occurrence:{occurrence_id}:description_group:{group_id}:modifier_text_invalid:{position}"
                                )
                        elif kind == "Relation":
                            relation_group_id = item.get("RelationGroupID")
                            if (
                                not isinstance(relation_group_id, str)
                                or not relation_group_id
                            ):
                                errors.append(
                                    f"occurrence:{occurrence_id}:description_group:{group_id}:relation_group_invalid:{position}"
                                )
                            else:
                                description_relation_refs.append(
                                    (
                                        occurrence_id,
                                        group_id,
                                        operator,
                                        position,
                                        relation_group_id,
                                    )
                                )
                        elif kind == "Event":
                            event_occurrence = item.get("EventOccurrence")
                            if (
                                not isinstance(event_occurrence, str)
                                or not event_occurrence
                            ):
                                errors.append(
                                    f"occurrence:{occurrence_id}:description_group:{group_id}:event_occurrence_invalid:{position}"
                                )
                            else:
                                description_event_refs.append(
                                    (
                                        occurrence_id,
                                        group_id,
                                        operator,
                                        position,
                                        event_occurrence,
                                    )
                                )
                        elif kind == "Group":
                            child_group_id = item.get("GroupID")
                            if (
                                not isinstance(child_group_id, str)
                                or not child_group_id
                            ):
                                errors.append(
                                    f"occurrence:{occurrence_id}:description_group:{group_id}:child_group_invalid:{position}"
                                )
                            else:
                                description_child_refs.append(
                                    (occurrence_id, group_id, child_group_id)
                                )
                        else:
                            errors.append(
                                f"occurrence:{occurrence_id}:description_group:{group_id}:invalid_kind:{kind}"
                            )

    for global_id, node in canonical_nodes.items():
        if not isinstance(node, dict):
            errors.append(f"global:{global_id}:not_dict")
            continue
        if node.get("ID") != global_id:
            errors.append(f"global:{global_id}:id_mismatch")
        if node.get("Kind") not in {"Entity", "Concept", "Action"}:
            errors.append(f"global:{global_id}:invalid_kind")

    seen_edges: Set[str] = set()
    canonical_targets: Dict[str, Set[str]] = defaultdict(set)
    relation_groups: Dict[str, Dict[str, Any]] = {}
    event_body_owners: Dict[str, Set[str]] = defaultdict(set)
    event_description_paths: Dict[str, Set[str]] = defaultdict(set)
    event_argument_positions: Dict[str, Set[int]] = defaultdict(set)
    self_arguments: List[Tuple[int, str, str]] = []

    import json

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edge[{index}]:not_dict")
            continue
        edge_type = edge.get("Type")
        if edge_type not in ALLOWED_EDGE_TYPES:
            errors.append(f"edge[{index}]:unknown_type:{edge_type}")

        signature = json.dumps(
            edge, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if signature in seen_edges:
            errors.append(f"edge[{index}]:duplicate")
        seen_edges.add(signature)

        source = edge.get("Source")
        target = edge.get("Target")
        source_kind = _endpoint_kind(source)
        target_kind = _endpoint_kind(target)
        if source_kind is None or target_kind is None:
            errors.append(f"edge[{index}]:invalid_endpoint")
            continue
        source_id = _endpoint_id(source)
        target_id = _endpoint_id(target)
        if source_kind == "Occurrence" and source_id not in occurrences:
            errors.append(f"edge[{index}]:source_occurrence_missing:{source_id}")
        if target_kind == "Occurrence" and target_id not in occurrences:
            errors.append(f"edge[{index}]:target_occurrence_missing:{target_id}")
        if source_kind == "Global" and source_id not in canonical_nodes:
            errors.append(f"edge[{index}]:source_global_missing:{source_id}")
        if target_kind == "Global" and target_id not in canonical_nodes:
            errors.append(f"edge[{index}]:target_global_missing:{target_id}")

        if edge_type == "RELATION":
            owner = edge.get("Owner")
            if _endpoint_kind(owner) != "Occurrence":
                errors.append(f"edge[{index}]:relation_owner_invalid")
            else:
                owner_id = owner["Occurrence"]
                if owner_id not in occurrences:
                    errors.append(f"edge[{index}]:relation_owner_missing:{owner_id}")

            group_id = edge.get("GroupID")
            operator = edge.get("Operator")
            position = edge.get("Position")
            grouped = (
                group_id is not None or operator is not None or position is not None
            )
            if grouped:
                if not isinstance(group_id, str) or not group_id:
                    errors.append(f"edge[{index}]:relation_group_id_invalid")
                if operator not in {"AND", "OR"}:
                    errors.append(f"edge[{index}]:relation_operator_invalid")
                if (
                    not isinstance(position, int)
                    or isinstance(position, bool)
                    or position < 0
                ):
                    errors.append(f"edge[{index}]:relation_position_invalid")
                if (
                    isinstance(group_id, str)
                    and group_id
                    and operator in {"AND", "OR"}
                    and isinstance(position, int)
                    and not isinstance(position, bool)
                    and position >= 0
                    and _endpoint_kind(owner) == "Occurrence"
                ):
                    owner_id = owner["Occurrence"]
                    known = relation_groups.get(group_id)
                    if known is None:
                        relation_groups[group_id] = {
                            "Owner": owner_id,
                            "Operator": operator,
                            "Positions": {position},
                        }
                    else:
                        if known["Owner"] != owner_id:
                            errors.append(
                                f"edge[{index}]:relation_group_owner_mismatch:{group_id}"
                            )
                        if known["Operator"] != operator:
                            errors.append(
                                f"edge[{index}]:relation_group_operator_mismatch:{group_id}"
                            )
                        known["Positions"].add(position)

            action_head = edge.get("ActionHead")
            action_operators = edge.get("ActionOperators")
            action_canonical = edge.get("ActionCanonical")
            has_action_fields = (
                action_head is not None
                or action_operators is not None
                or action_canonical is not None
            )
            if has_action_fields:
                if not isinstance(action_head, str) or not normalize_text(action_head):
                    errors.append(f"edge[{index}]:relation_action_head_invalid")
                if not isinstance(action_operators, list) or any(
                    not isinstance(value, str) or not value.strip()
                    for value in action_operators
                ):
                    errors.append(f"edge[{index}]:relation_action_operators_invalid")
                if action_canonical is None:
                    if canonical_nodes:
                        errors.append(
                            f"edge[{index}]:relation_action_canonical_missing"
                        )
                elif (
                    not isinstance(action_canonical, str)
                    or action_canonical not in canonical_nodes
                ):
                    errors.append(f"edge[{index}]:relation_action_canonical_invalid")
                else:
                    action_node = canonical_nodes.get(action_canonical, {})
                    if action_node.get("Kind") != "Action":
                        errors.append(
                            f"edge[{index}]:relation_action_canonical_not_action"
                        )
                    elif normalize_text(
                        action_node.get("Head") or action_node.get("Text")
                    ) != normalize_text(action_head):
                        errors.append(
                            f"edge[{index}]:relation_action_canonical_head_mismatch"
                        )

        if edge_type == "HAS_SEMANTIC_BODY":
            if source_kind != "Occurrence" or target_kind != "Occurrence":
                errors.append(f"edge[{index}]:semantic_body_direction")
            else:
                body_kind = edge.get("BodyKind")
                if body_kind not in {"Entity", "Event"}:
                    errors.append(f"edge[{index}]:semantic_body_kind_invalid")
                if body_kind == "Event":
                    target_record = occurrences.get(target_id, {})
                    path = target_record.get("Path")
                    if target_record.get("FusionKind") != "Action" or not (
                        isinstance(path, str) and path.split(".")[-1] == "Event"
                    ):
                        errors.append(
                            f"edge[{index}]:semantic_body_event_target_invalid"
                        )
                    event_body_owners[target_id].add(source_id)
                    description_path = edge.get("DescriptionPath")
                    if description_path is not None:
                        if (
                            not isinstance(description_path, str)
                            or not description_path
                        ):
                            errors.append(
                                f"edge[{index}]:semantic_body_description_path_invalid"
                            )
                        elif target_record.get("Path") != description_path:
                            errors.append(
                                f"edge[{index}]:semantic_body_description_path_mismatch"
                            )
                        else:
                            event_description_paths[target_id].add(description_path)

        if edge_type == "ARGUMENT":
            if source_kind != "Occurrence" or target_kind != "Occurrence":
                errors.append(f"edge[{index}]:argument_direction")
            else:
                source_record = occurrences.get(source_id, {})
                source_path = source_record.get("Path")
                if source_record.get("FusionKind") != "Action" or not (
                    isinstance(source_path, str)
                    and source_path.split(".")[-1] == "Event"
                ):
                    errors.append(f"edge[{index}]:argument_source_not_event")
                role = edge.get("Role")
                if role not in {"Subject", "Object", "Complement"}:
                    errors.append(f"edge[{index}]:argument_role_invalid")
                marker = edge.get("Marker")
                if marker is not None and (
                    not isinstance(marker, str) or not marker.strip()
                ):
                    errors.append(f"edge[{index}]:argument_marker_invalid")
                position = edge.get("Position")
                if (
                    not isinstance(position, int)
                    or isinstance(position, bool)
                    or position < 0
                ):
                    errors.append(f"edge[{index}]:argument_position_invalid")
                else:
                    if position in event_argument_positions[source_id]:
                        errors.append(
                            f"edge[{index}]:argument_position_duplicate:{position}"
                        )
                    event_argument_positions[source_id].add(position)
                if "Self" in edge:
                    if edge.get("Self") is not True:
                        errors.append(f"edge[{index}]:argument_self_invalid")
                    else:
                        self_arguments.append((index, source_id, target_id))

        if edge_type == "CANONICAL_OF":
            if source_kind != "Occurrence" or target_kind != "Global":
                errors.append(f"edge[{index}]:canonical_direction")
                continue
            occurrence = occurrences.get(source_id, {})
            global_node = canonical_nodes.get(target_id, {})
            if occurrence.get("NodeKind") == "Frame":
                errors.append(f"edge[{index}]:frame_canonicalized")
            if not occurrence.get("CanonicalEligible"):
                errors.append(f"edge[{index}]:ineligible_canonical_source")
            if occurrence.get("FusionKind") != global_node.get("Kind"):
                errors.append(f"edge[{index}]:canonical_kind_mismatch")
            canonical_targets[source_id].add(target_id)

        if edge_type == "SUBTYPE_OF":
            if source_kind != "Global" or target_kind != "Global":
                errors.append(f"edge[{index}]:subtype_direction")
                continue
            if source_id == target_id:
                errors.append(f"edge[{index}]:subtype_self_loop")
            parent = canonical_nodes.get(target_id, {})
            if parent.get("Kind") != "Concept":
                errors.append(f"edge[{index}]:subtype_parent_not_concept")

        if edge_type == "REFERS_TO" and target_kind == "Global":
            errors.append(f"edge[{index}]:refers_to_global_not_allowed")

        if edge_type == "REFERS_TO" and edge.get("ReferenceKind") == "CONTEXT_ANCHOR":
            if source_kind != "Occurrence" or target_kind != "Occurrence":
                errors.append(f"edge[{index}]:context_anchor_direction")
                continue
            source_record = occurrences.get(source_id, {})
            target_record = occurrences.get(target_id, {})
            if (
                target_record.get("NodeKind") != "Context"
                or target_record.get("EntityType") != "Organization"
            ):
                errors.append(f"edge[{index}]:context_anchor_target_invalid")
            if normalize_text(source_record.get("Organization")) != normalize_text(
                target_record.get("Organization")
            ):
                errors.append(f"edge[{index}]:context_anchor_cross_org")

    for occurrence_id, targets in canonical_targets.items():
        if len(targets) > 1:
            errors.append(f"occurrence:{occurrence_id}:multiple_canonical_targets")

    described_relation_positions: Set[Tuple[str, str, str, int]] = set()
    for (
        occurrence_id,
        description_group_id,
        operator,
        position,
        relation_group_id,
    ) in description_relation_refs:
        relation_group = relation_groups.get(relation_group_id)
        if relation_group is None:
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{description_group_id}:relation_group_missing:{relation_group_id}"
            )
            continue
        if relation_group["Owner"] != occurrence_id:
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{description_group_id}:relation_owner_mismatch:{relation_group_id}"
            )
        if relation_group["Operator"] != operator:
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{description_group_id}:relation_operator_mismatch:{relation_group_id}"
            )
        if position not in relation_group["Positions"]:
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{description_group_id}:relation_position_missing:{position}"
            )
        described_relation_positions.add(
            (occurrence_id, relation_group_id, operator, position)
        )

    for occurrence_id, parent_group_id, child_group_id in description_child_refs:
        child = description_groups.get(child_group_id)
        if child is None:
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{parent_group_id}:child_group_missing:{child_group_id}"
            )
        elif child["Owner"] != occurrence_id:
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{parent_group_id}:child_group_owner_mismatch:{child_group_id}"
            )

    described_event_occurrences: Set[str] = set()
    semantic_event_links = {
        (owner, event_occurrence)
        for event_occurrence, owners in event_body_owners.items()
        for owner in owners
    }
    for (
        occurrence_id,
        description_group_id,
        operator,
        position,
        event_occurrence,
    ) in description_event_refs:
        event_record = occurrences.get(event_occurrence)
        if not isinstance(event_record, dict):
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{description_group_id}:event_missing:{event_occurrence}"
            )
            continue
        owner_record = occurrences.get(occurrence_id, {})
        for field in ("SourceID", "PolicyID", "Scope", "Ref"):
            if event_record.get(field) != owner_record.get(field):
                errors.append(
                    f"occurrence:{occurrence_id}:description_group:{description_group_id}:event_{field.lower()}_mismatch:{position}"
                )
        event_path = event_record.get("Path")
        if event_record.get("FusionKind") != "Action" or not (
            isinstance(event_path, str) and event_path.split(".")[-1] == "Event"
        ):
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{description_group_id}:event_not_action:{position}"
            )
        if (occurrence_id, event_occurrence) not in semantic_event_links:
            errors.append(
                f"occurrence:{occurrence_id}:description_group:{description_group_id}:event_body_edge_missing:{position}"
            )
        described_event_occurrences.add(event_occurrence)

    for event_occurrence, owners in event_body_owners.items():
        if len(owners) > 1:
            errors.append(f"event:{event_occurrence}:multiple_semantic_body_owners")

    for edge_index, event_occurrence, target_occurrence in self_arguments:
        owners = event_body_owners.get(event_occurrence, set())
        if len(owners) != 1 or target_occurrence not in owners:
            errors.append(f"edge[{edge_index}]:argument_self_target_mismatch")

    for occurrence_id, occurrence in occurrences.items():
        if not isinstance(occurrence, dict):
            continue
        path = occurrence.get("Path")
        if not (isinstance(path, str) and path.split(".")[-1] == "Event"):
            continue
        owners = event_body_owners.get(occurrence_id, set())
        if len(owners) != 1:
            errors.append(f"event:{occurrence_id}:semantic_body_owner_missing")
        if "DescriptionGroups" in path and ".Items." in path:
            if occurrence_id not in described_event_occurrences:
                errors.append(f"event:{occurrence_id}:group_branch_not_described")
        if "DescriptionGroups" in path:
            if path not in event_description_paths.get(occurrence_id, set()):
                errors.append(f"event:{occurrence_id}:description_path_edge_missing")

    for relation_group_id, relation_group in relation_groups.items():
        for position in relation_group["Positions"]:
            key = (
                relation_group["Owner"],
                relation_group_id,
                relation_group["Operator"],
                position,
            )
            if key not in described_relation_positions:
                errors.append(
                    f"relation_group:{relation_group_id}:position_not_described:{position}"
                )

    canonical_map = _canonical_map(graph)
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict) or edge.get("Type") != "DISTINCT_FROM":
            continue
        source = edge.get("Source", {})
        target = edge.get("Target", {})
        if (
            _endpoint_kind(source) != "Occurrence"
            or _endpoint_kind(target) != "Occurrence"
        ):
            continue
        left_global = canonical_map.get(source["Occurrence"])
        right_global = canonical_map.get(target["Occurrence"])
        if not left_global or left_global != right_global:
            continue
        global_node = canonical_nodes.get(left_global, {})
        if global_node.get("Kind") == "Entity":
            errors.append(f"edge[{index}]:distinct_same_entity:{left_global}")

    cycle = _find_cycle(_subtype_adjacency(graph))
    if cycle:
        errors.append("subtype:cycle:" + "->".join(cycle))

    return errors
