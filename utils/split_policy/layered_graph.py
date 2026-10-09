import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

NODE_KEYS = {
    "Text",
    "Head",
    "Coordination",
    "DescriptionGroups",
    "Reference",
}
ENTITY_BODY_KEYS = NODE_KEYS - {"Text"}
GRAPH_KEYS = {"Root", "Nodes"}
REF_KEYS = {"Ref"}
STUB_KEYS = {"Text"}
REFERENCE_TYPES = {"Coreference", "Possessor"}
OPERATORS = {"AND", "OR"}
MAX_DESCRIPTION_DEPTH = 16

ACTION_BODY_KEYS = {"Action", "Modifiers", "Operators"}
EVENT_FRAME_KEYS = {"Action", "Modifiers", "Operators", "Arguments"}
EVENT_ARGUMENT_KEYS = {"Role", "Marker", "Value"}
EVENT_ARGUMENT_ROLES = {"Subject", "Object", "Complement"}
SELF_KEYS = {"Self"}


@dataclass
class GraphState:
    candidate_id: str
    element: Dict
    root_text: str
    context: str
    prompt_context: Dict
    node_prefix: str = "n"

    nodes: Dict[str, Dict] = field(default_factory=dict)
    meta: Dict[str, Dict] = field(default_factory=dict)
    text_index: Dict[str, List[str]] = field(default_factory=dict)
    next_node_index: int = 0
    root_id: str = ""
    fallback_nodes: int = 0

    def new_node(self, text: str, depth: int, parent_hint: Dict) -> str:
        node_id = f"{self.node_prefix}{self.next_node_index}"
        self.next_node_index += 1

        self.nodes[node_id] = {"Text": text}
        self.meta[node_id] = {
            "status": "pending",
            "depth": depth,
            "parent_hint": parent_hint,
        }

        key = normalize_space(text).lower()
        self.text_index.setdefault(key, []).append(node_id)

        if not self.root_id:
            self.root_id = node_id

        return node_id


def flatten_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(flatten_strings(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(flatten_strings(item))
        return result
    return []


def build_context_text(element: Dict, keys: Optional[Tuple[str, ...]] = None) -> str:
    if keys is None:
        keys = ("Sentence", "Subject", "Predicate", "Object", "Condition")
    parts = []
    for key in keys:
        parts.extend(flatten_strings(element.get(key)))
    return " ".join(part.strip() for part in parts if part.strip())


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def canonicalize_possessor_reference_text(text: str) -> str:
    """Canonicalize possessor wording by removing only the possessive suffix."""
    if not isinstance(text, str):
        return text

    value = normalize_space(text)
    if re.search(r"['’]s$", value, flags=re.IGNORECASE):
        return value[:-2].rstrip()
    if value.endswith(("'", "’")):
        return value[:-1].rstrip()
    return value


def tokenize(text: str) -> List[str]:
    return re.findall(
        r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?",
        text.lower(),
    )


def is_subsequence(candidate: List[str], source: List[str]) -> bool:
    if not candidate:
        return False
    index = 0
    for token in candidate:
        while index < len(source) and source[index] != token:
            index += 1
        if index >= len(source):
            return False
        index += 1
    return True


def phrase_grounded(text: str, source: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    text_norm = normalize_space(text).lower()
    source_norm = normalize_space(source).lower()
    if text_norm in source_norm:
        return True
    return is_subsequence(tokenize(text), tokenize(source))


def exact_occurrence_count(text: str, source: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    text_norm = normalize_space(text).lower()
    source_norm = normalize_space(source).lower()
    return source_norm.count(text_norm)


def simple_token(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text.strip()))


def fallback_graph(root_text: str, node_prefix: str = "n") -> Dict:
    text = root_text.strip() if isinstance(root_text, str) else ""
    root_id = f"{node_prefix}0"
    return {
        "Root": root_id,
        "Nodes": {
            root_id: {
                "Text": text,
                "Head": text or None,
                "Coordination": None,
                "DescriptionGroups": [],
                "Reference": None,
            }
        },
    }


def deterministic_graph(
    root_text: str,
    node_prefix: str = "n",
) -> Optional[Dict]:
    text = root_text.strip()
    if not simple_token(text):
        return None
    return fallback_graph(text, node_prefix=node_prefix)


def make_task_id(candidate_id: str, node_id: str) -> str:
    return f"{candidate_id}:{node_id}"


def make_graph_state(candidate: Dict) -> GraphState:
    state = GraphState(
        candidate_id=candidate["id"],
        element=candidate["element"],
        root_text=candidate["RootText"],
        context=candidate["Context"],
        prompt_context=dict(candidate.get("PromptContext", {})),
        node_prefix=candidate.get("NodePrefix", "n"),
    )
    state.new_node(candidate["RootText"], 0, {"Role": "Root"})
    return state


def build_node_task(state: GraphState, node_id: str) -> Dict:
    meta = state.meta[node_id]
    return {
        "id": make_task_id(state.candidate_id, node_id),
        "candidate_id": state.candidate_id,
        "node_id": node_id,
        "state": state,
        "RootText": state.root_text,
        "CurrentText": state.nodes[node_id]["Text"],
        "ParentHint": meta.get("parent_hint", {"Role": "Root"}),
        "Context": state.context,
        "PromptContext": state.prompt_context,
    }


def pending_tasks(states: List[GraphState]) -> List[Dict]:
    tasks = []
    for state in states:
        for node_id, meta in state.meta.items():
            if meta.get("status") == "pending":
                tasks.append(build_node_task(state, node_id))
    return tasks


def unique_existing_node_for_text(
    state: GraphState,
    text: str,
    exclude_node_id: Optional[str] = None,
) -> Optional[str]:
    if exact_occurrence_count(text, state.root_text) != 1:
        return None
    key = normalize_space(text).lower()
    candidates = [
        node_id
        for node_id in state.text_index.get(key, [])
        if node_id != exclude_node_id
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def intern_child_node(
    state: GraphState,
    current_node_id: str,
    text: str,
    depth: int,
    parent_hint: Dict,
    local_cache: Dict[Tuple[str, str], str],
    role: str,
    max_nodes_per_graph: int,
    allow_unique_existing: bool,
) -> str:
    normalized = normalize_space(text).lower()
    local_key = (role, normalized)
    if local_key in local_cache:
        return local_cache[local_key]

    if allow_unique_existing:
        existing = unique_existing_node_for_text(
            state,
            text,
            exclude_node_id=current_node_id,
        )
        if existing is not None:
            local_cache[local_key] = existing
            return existing

    if len(state.nodes) >= max_nodes_per_graph:
        raise ValueError(
            f"max_nodes_per_graph exceeded for {state.candidate_id}: "
            f"{max_nodes_per_graph}"
        )

    child_id = state.new_node(text, depth, parent_hint)
    local_cache[local_key] = child_id
    return child_id


def fallback_pending_node(state: GraphState, node_id: str) -> None:
    text = state.nodes[node_id].get("Text", "")
    state.nodes[node_id] = {
        "Text": text,
        "Head": text or None,
        "Coordination": None,
        "DescriptionGroups": [],
        "Reference": None,
    }
    state.meta[node_id]["status"] = "expanded"
    state.meta[node_id]["fallback"] = True
    state.fallback_nodes += 1


def create_fallback_child(
    state: GraphState,
    text: str,
    depth: int,
    parent_hint: Dict,
    max_nodes_per_graph: int,
) -> str:
    if len(state.nodes) >= max_nodes_per_graph:
        raise ValueError(
            f"max_nodes_per_graph exceeded for {state.candidate_id}: "
            f"{max_nodes_per_graph}"
        )
    child_id = state.new_node(text, depth, parent_hint)
    fallback_pending_node(state, child_id)
    return child_id


def expand_deterministic_pending(state: GraphState, node_id: str) -> bool:
    text = state.nodes[node_id].get("Text", "")
    if not simple_token(text):
        return False
    state.nodes[node_id] = {
        "Text": text,
        "Head": text,
        "Coordination": None,
        "DescriptionGroups": [],
        "Reference": None,
    }
    state.meta[node_id]["status"] = "expanded"
    state.meta[node_id]["deterministic"] = True
    return True


def stub_text_reason(value: Any, source: str, path: str) -> str:
    if not isinstance(value, dict) or set(value) != STUB_KEYS:
        return f"{path}:invalid_stub"
    text = value.get("Text")
    if not isinstance(text, str) or not phrase_grounded(text, source):
        return f"{path}.Text:not_grounded({text!r})"
    return ""


def action_body_reason(
    value: Any,
    source: str,
    path: str = "action",
    *,
    require_action: bool = True,
) -> str:
    if not isinstance(value, dict) or set(value) != ACTION_BODY_KEYS:
        return f"{path}:invalid_schema"

    action = value.get("Action")
    if require_action:
        if not isinstance(action, str) or not phrase_grounded(action, source):
            return f"{path}.Action:not_grounded({action!r})"
    elif action is not None and (
        not isinstance(action, str) or not phrase_grounded(action, source)
    ):
        return f"{path}.Action:not_grounded({action!r})"

    for field_name in ("Modifiers", "Operators"):
        items = value.get(field_name)
        if not isinstance(items, list):
            return f"{path}.{field_name}:not_list"
        seen = set()
        for index, item in enumerate(items):
            if not isinstance(item, str) or not phrase_grounded(item, source):
                return f"{path}.{field_name}[{index}]:not_grounded({item!r})"
            key = normalize_space(item).lower()
            if key in seen:
                return f"{path}.{field_name}[{index}]:duplicate({item!r})"
            seen.add(key)

    return ""


def event_frame_plan_reason(
    value: Any,
    current_text: str,
    root_text: str,
    path: str = "event",
    *,
    allow_self: bool = True,
) -> str:
    if not isinstance(value, dict) or set(value) != EVENT_FRAME_KEYS:
        return f"{path}:invalid_schema"

    reason = action_body_reason(
        {
            "Action": value.get("Action"),
            "Modifiers": value.get("Modifiers"),
            "Operators": value.get("Operators"),
        },
        current_text,
        path,
        require_action=True,
    )
    if reason:
        return reason

    arguments = value.get("Arguments")
    if not isinstance(arguments, list):
        return f"{path}.Arguments:not_list"

    seen = set()
    for index, argument in enumerate(arguments):
        item_path = f"{path}.Arguments[{index}]"
        if not isinstance(argument, dict) or set(argument) != EVENT_ARGUMENT_KEYS:
            return f"{item_path}:invalid_schema"

        role = argument.get("Role")
        if role not in EVENT_ARGUMENT_ROLES:
            return f"{item_path}.Role:invalid({role!r})"

        marker = argument.get("Marker")
        if marker is not None and (
            not isinstance(marker, str) or not phrase_grounded(marker, current_text)
        ):
            return f"{item_path}.Marker:not_grounded({marker!r})"

        argument_value = argument.get("Value")
        if isinstance(argument_value, dict) and set(argument_value) == SELF_KEYS:
            if not allow_self or argument_value.get("Self") is not True:
                return f"{item_path}.Value:invalid_self"
        else:
            reason = stub_text_reason(
                argument_value,
                root_text,
                f"{item_path}.Value",
            )
            if reason:
                return reason

        signature = repr((role, marker, argument_value))
        if signature in seen:
            return f"{item_path}:duplicate_argument"
        seen.add(signature)

    return ""


def install_event_frame(
    task: Dict,
    value: Dict,
    max_depth: int,
    max_nodes_per_graph: int,
    *,
    allow_self: bool = True,
    fallback_node_handler: Optional[Callable[[GraphState, str], None]] = None,
) -> Dict:
    state: GraphState = task["state"]
    current_node_id = task["node_id"]
    depth = state.meta[current_node_id]["depth"]
    local_cache: Dict[Tuple[str, str], str] = {}

    arguments_out = []
    for argument in value.get("Arguments", []):
        argument_value = argument["Value"]
        if set(argument_value) == SELF_KEYS:
            if not allow_self or argument_value.get("Self") is not True:
                raise ValueError("Invalid Event Self argument.")
            value_out = {"Self": True}
        else:
            text = argument_value["Text"]
            parent_hint = {
                "Role": "EventArgument",
                "ArgumentRole": argument["Role"],
                "Marker": argument.get("Marker"),
                "Action": value["Action"],
            }
            if depth + 1 > max_depth:
                if len(state.nodes) >= max_nodes_per_graph:
                    raise ValueError(
                        f"max_nodes_per_graph exceeded for {state.candidate_id}: "
                        f"{max_nodes_per_graph}"
                    )
                child_id = state.new_node(text, depth + 1, parent_hint)
                if fallback_node_handler is None:
                    fallback_pending_node(state, child_id)
                else:
                    fallback_node_handler(state, child_id)
            else:
                child_id = intern_child_node(
                    state,
                    current_node_id,
                    text,
                    depth + 1,
                    parent_hint,
                    local_cache,
                    role=f"event_argument:{argument['Role']}",
                    max_nodes_per_graph=max_nodes_per_graph,
                    allow_unique_existing=True,
                )
            value_out = {"Ref": child_id}

        arguments_out.append(
            {
                "Role": argument["Role"],
                "Marker": argument.get("Marker"),
                "Value": value_out,
            }
        )

    return {
        "Action": value["Action"],
        "Modifiers": copy.deepcopy(value.get("Modifiers", [])),
        "Operators": copy.deepcopy(value.get("Operators", [])),
        "Arguments": arguments_out,
    }


def event_frame_graph_reason(
    value: Any,
    node_text: str,
    nodes: Dict[str, Dict],
    path: str = "event",
    *,
    allow_self: bool = True,
) -> str:
    if not isinstance(value, dict) or set(value) != EVENT_FRAME_KEYS:
        return f"{path}:invalid_schema"

    reason = action_body_reason(
        {
            "Action": value.get("Action"),
            "Modifiers": value.get("Modifiers"),
            "Operators": value.get("Operators"),
        },
        node_text,
        path,
        require_action=True,
    )
    if reason:
        return reason

    arguments = value.get("Arguments")
    if not isinstance(arguments, list):
        return f"{path}.Arguments:not_list"

    seen = set()
    for index, argument in enumerate(arguments):
        item_path = f"{path}.Arguments[{index}]"
        if not isinstance(argument, dict) or set(argument) != EVENT_ARGUMENT_KEYS:
            return f"{item_path}:invalid_schema"

        role = argument.get("Role")
        if role not in EVENT_ARGUMENT_ROLES:
            return f"{item_path}.Role:invalid({role!r})"

        marker = argument.get("Marker")
        if marker is not None and (
            not isinstance(marker, str) or not phrase_grounded(marker, node_text)
        ):
            return f"{item_path}.Marker:not_grounded({marker!r})"

        argument_value = argument.get("Value")
        if isinstance(argument_value, dict) and set(argument_value) == SELF_KEYS:
            if not allow_self or argument_value.get("Self") is not True:
                return f"{item_path}.Value:invalid_self"
        elif not isinstance(argument_value, dict) or set(argument_value) != REF_KEYS:
            return f"{item_path}.Value:invalid_ref"
        else:
            ref_id = argument_value.get("Ref")
            if not isinstance(ref_id, str) or ref_id not in nodes:
                return f"{item_path}.Value:missing_ref({ref_id!r})"

        signature = repr((role, marker, argument_value))
        if signature in seen:
            return f"{item_path}:duplicate_argument"
        seen.add(signature)

    return ""


def event_frame_edges(value: Dict) -> Set[str]:
    edges: Set[str] = set()
    for argument in value.get("Arguments", []):
        argument_value = argument.get("Value")
        if isinstance(argument_value, dict) and set(argument_value) == REF_KEYS:
            ref_id = argument_value.get("Ref")
            if isinstance(ref_id, str):
                edges.add(ref_id)
    return edges


def _graph_description_items_reason(
    items: Any,
    node_text: str,
    nodes: Dict[str, Dict],
    path: str,
    *,
    parent_operator: Optional[str] = None,
    top_level: bool = False,
    depth: int = 0,
) -> str:
    if depth > MAX_DESCRIPTION_DEPTH:
        return f"{path}:max_depth"
    if not isinstance(items, list):
        return f"{path}:not_list"

    seen = set()
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            return f"{item_path}:not_object"

        signature = _description_signature(item)
        if signature in seen:
            return f"{item_path}:duplicate_description"
        seen.add(signature)

        keys = set(item)
        if keys == {"Modifier"}:
            modifier = item.get("Modifier")
            if not isinstance(modifier, str) or not phrase_grounded(
                modifier, node_text
            ):
                return f"{item_path}.Modifier:not_grounded({modifier!r})"
            continue

        if keys == {"Relation"}:
            relation = item.get("Relation")
            if not isinstance(relation, dict):
                return f"{item_path}.Relation:not_object"
            relation_keys = set(relation)
            if relation_keys == {"Predicate", "Target"}:
                predicate = relation.get("Predicate")
                if not isinstance(predicate, str) or not phrase_grounded(
                    predicate, node_text
                ):
                    return f"{item_path}.Relation.Predicate:not_grounded({predicate!r})"
                target = relation.get("Target")
                if not isinstance(target, dict) or set(target) != REF_KEYS:
                    return f"{item_path}.Relation.Target:invalid_ref"
                ref_id = target.get("Ref")
                if not isinstance(ref_id, str) or ref_id not in nodes:
                    return f"{item_path}.Relation.Target:missing_ref({ref_id!r})"
                continue
            if relation_keys == {"Source", "Predicate"}:
                predicate = relation.get("Predicate")
                if not isinstance(predicate, str) or not phrase_grounded(
                    predicate, node_text
                ):
                    return f"{item_path}.Relation.Predicate:not_grounded({predicate!r})"
                source = relation.get("Source")
                if not isinstance(source, dict) or set(source) != REF_KEYS:
                    return f"{item_path}.Relation.Source:invalid_ref"
                ref_id = source.get("Ref")
                if not isinstance(ref_id, str) or ref_id not in nodes:
                    return f"{item_path}.Relation.Source:missing_ref({ref_id!r})"
                continue
            return f"{item_path}.Relation:invalid_schema(keys={sorted(relation_keys)})"

        if keys == {"Event"}:
            reason = event_frame_graph_reason(
                item.get("Event"),
                node_text,
                nodes,
                f"{item_path}.Event",
                allow_self=True,
            )
            if reason:
                return reason
            continue

        if keys != {"Operator", "Items"}:
            return f"{item_path}:invalid_schema(keys={sorted(keys)})"

        operator = item.get("Operator")
        if operator not in OPERATORS:
            return f"{item_path}:invalid_operator({operator!r})"
        if top_level and operator == "AND":
            return f"{item_path}:redundant_top_level_and"
        if parent_operator == operator:
            return f"{item_path}:redundant_same_operator_nesting({operator})"

        group_items = item.get("Items")
        if not isinstance(group_items, list) or len(group_items) < 2:
            return f"{item_path}.Items:need_at_least_2"

        reason = _graph_description_items_reason(
            group_items,
            node_text,
            nodes,
            f"{item_path}.Items",
            parent_operator=operator,
            top_level=False,
            depth=depth + 1,
        )
        if reason:
            return reason

    return ""


def entity_body_graph_reason(
    body: Any,
    node_text: str,
    nodes: Dict[str, Dict],
    context: str,
    path: str = "entity",
) -> str:
    if not isinstance(body, dict) or set(body) != ENTITY_BODY_KEYS:
        return f"{path}:invalid_schema"

    head = body.get("Head")
    if head is not None and (
        not isinstance(head, str) or not phrase_grounded(head, node_text)
    ):
        return f"{path}.Head:not_grounded({head!r})"

    coordination = body.get("Coordination")
    if coordination is not None:
        if not isinstance(coordination, dict) or set(coordination) != {
            "Operator",
            "Items",
        }:
            return f"{path}.Coordination:invalid_schema"
        if coordination.get("Operator") not in OPERATORS:
            return f"{path}.Coordination:invalid_operator"
        items = coordination.get("Items")
        if not isinstance(items, list) or len(items) < 2:
            return f"{path}.Coordination.Items:need_at_least_2"
        seen_refs = set()
        for index, item in enumerate(items):
            item_path = f"{path}.Coordination.Items[{index}]"
            if not isinstance(item, dict) or set(item) != REF_KEYS:
                return f"{item_path}:invalid_ref"
            ref_id = item.get("Ref")
            if not isinstance(ref_id, str) or ref_id not in nodes:
                return f"{item_path}:missing_ref({ref_id!r})"
            if ref_id in seen_refs:
                return f"{item_path}:duplicate_ref"
            seen_refs.add(ref_id)

    reason = _graph_description_items_reason(
        body.get("DescriptionGroups"),
        node_text,
        nodes,
        f"{path}.DescriptionGroups",
        top_level=True,
    )
    if reason:
        return reason

    reference = body.get("Reference")
    if reference is not None:
        if not isinstance(reference, dict) or set(reference) != {"Type", "Text"}:
            return f"{path}.Reference:invalid_schema"
        if reference.get("Type") not in REFERENCE_TYPES:
            return f"{path}.Reference:invalid_type"
        ref_text = reference.get("Text")
        if not isinstance(ref_text, str):
            return f"{path}.Reference.Text:not_string"

        if reference.get("Type") == "Possessor":
            canonical_ref_text = canonicalize_possessor_reference_text(ref_text)
            if not canonical_ref_text:
                return f"{path}.Reference.Text:empty_possessor"
            if canonical_ref_text != normalize_space(ref_text):
                return (
                    f"{path}.Reference.Text:noncanonical_possessor"
                    f"({ref_text!r}->{canonical_ref_text!r})"
                )
            ref_text = canonical_ref_text

        if not phrase_grounded(ref_text, context):
            return f"{path}.Reference.Text:not_grounded({ref_text!r})"

    if head is None and coordination is None and reference is None:
        return f"{path}:missing_head_coordination_reference"

    return ""


def entity_body_edges(body: Dict) -> Set[str]:
    edges: Set[str] = set()
    coordination = body.get("Coordination")
    if isinstance(coordination, dict):
        for item in coordination.get("Items", []):
            if isinstance(item, dict) and isinstance(item.get("Ref"), str):
                edges.add(item["Ref"])

    def add_description(item: Dict) -> None:
        keys = set(item)
        if keys == {"Relation"}:
            relation = item["Relation"]
            endpoint = (
                relation.get("Target")
                if "Target" in relation
                else relation.get("Source")
            )
            if isinstance(endpoint, dict) and isinstance(endpoint.get("Ref"), str):
                edges.add(endpoint["Ref"])
            return
        if keys == {"Event"}:
            edges.update(event_frame_edges(item["Event"]))
            return
        if keys == {"Operator", "Items"}:
            for child in item.get("Items", []):
                add_description(child)

    for item in body.get("DescriptionGroups", []):
        add_description(item)
    return edges


def materialize_event_frame(
    value: Dict,
    expand_ref: Callable[[Dict, Set[str]], Dict],
    stack: Set[str],
) -> Dict:
    arguments = []
    for argument in value.get("Arguments", []):
        argument_value = argument["Value"]
        if set(argument_value) == SELF_KEYS:
            value_out = {"Self": True}
        else:
            value_out = expand_ref(argument_value, stack)
        arguments.append(
            {
                "Role": argument["Role"],
                "Marker": argument.get("Marker"),
                "Value": value_out,
            }
        )

    return {
        "Action": value["Action"],
        "Modifiers": copy.deepcopy(value.get("Modifiers", [])),
        "Operators": copy.deepcopy(value.get("Operators", [])),
        "Arguments": arguments,
    }


def _description_signature(value: Dict) -> str:
    import json

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _plan_description_items_reason(
    items: Any,
    current_text: str,
    root_text: str,
    path: str,
    *,
    parent_operator: Optional[str] = None,
    top_level: bool = False,
    depth: int = 0,
) -> str:
    if depth > MAX_DESCRIPTION_DEPTH:
        return f"{path}:max_depth"
    if not isinstance(items, list):
        return f"{path}:not_list"

    seen = set()
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            return f"{item_path}:not_object"

        signature = _description_signature(item)
        if signature in seen:
            return f"{item_path}:duplicate_description"
        seen.add(signature)

        keys = set(item)

        if keys == {"Modifier"}:
            modifier = item.get("Modifier")
            if not isinstance(modifier, str) or not phrase_grounded(
                modifier,
                current_text,
            ):
                return f"{item_path}.Modifier:not_grounded({modifier!r})"
            continue

        if keys == {"Relation"}:
            relation = item.get("Relation")
            if not isinstance(relation, dict):
                return f"{item_path}.Relation:not_object"

            relation_keys = set(relation)
            if relation_keys == {"Predicate", "Target"}:
                predicate = relation.get("Predicate")
                if not isinstance(predicate, str) or not phrase_grounded(
                    predicate,
                    current_text,
                ):
                    return (
                        f"{item_path}.Relation.Predicate:"
                        f"not_grounded({predicate!r})"
                    )
                reason = stub_text_reason(
                    relation.get("Target"),
                    root_text,
                    f"{item_path}.Relation.Target",
                )
                if reason:
                    return reason
                continue

            if relation_keys == {"Source", "Predicate"}:
                predicate = relation.get("Predicate")
                if not isinstance(predicate, str) or not phrase_grounded(
                    predicate,
                    current_text,
                ):
                    return (
                        f"{item_path}.Relation.Predicate:"
                        f"not_grounded({predicate!r})"
                    )
                reason = stub_text_reason(
                    relation.get("Source"),
                    root_text,
                    f"{item_path}.Relation.Source",
                )
                if reason:
                    return reason
                continue

            return (
                f"{item_path}.Relation:" f"invalid_schema(keys={sorted(relation_keys)})"
            )

        if keys == {"Event"}:
            reason = event_frame_plan_reason(
                item.get("Event"),
                current_text,
                root_text,
                f"{item_path}.Event",
                allow_self=True,
            )
            if reason:
                return reason
            continue

        if keys != {"Operator", "Items"}:
            return f"{item_path}:invalid_schema(keys={sorted(keys)})"

        operator = item.get("Operator")
        if operator not in OPERATORS:
            return f"{item_path}:invalid_operator({operator!r})"

        # The top-level DescriptionGroups array itself already means cumulative AND.
        if top_level and operator == "AND":
            return f"{item_path}:redundant_top_level_and"

        # AND(AND(...), ...) and OR(OR(...), ...) have a unique flatter form.
        if parent_operator == operator:
            return f"{item_path}:redundant_same_operator_nesting({operator})"

        group_items = item.get("Items")
        if not isinstance(group_items, list) or len(group_items) < 2:
            return f"{item_path}.Items:need_at_least_2"

        reason = _plan_description_items_reason(
            group_items,
            current_text,
            root_text,
            f"{item_path}.Items",
            parent_operator=operator,
            top_level=False,
            depth=depth + 1,
        )
        if reason:
            return reason

    return ""


def entity_plan_reason(task: Dict, value: Any) -> str:
    current_text = task["CurrentText"]
    root_text = task["RootText"]
    context = task["Context"]

    if not isinstance(value, dict):
        return "node:not_object"
    if set(value) != NODE_KEYS:
        missing = sorted(NODE_KEYS - set(value))
        extra = sorted(set(value) - NODE_KEYS)
        return f"node:keys(missing={missing},extra={extra})"

    text_value = value.get("Text")
    if not isinstance(text_value, str):
        return "node.Text:not_string"

    head = value.get("Head")
    if head is not None and (
        not isinstance(head, str) or not phrase_grounded(head, current_text)
    ):
        return f"node.Head:not_grounded({head!r})"

    coordination = value.get("Coordination")
    if coordination is not None:
        if not isinstance(coordination, dict) or set(coordination) != {
            "Operator",
            "Items",
        }:
            return "node.Coordination:invalid_schema"
        if coordination.get("Operator") not in OPERATORS:
            return (
                "node.Coordination:invalid_operator"
                f"({coordination.get('Operator')!r})"
            )
        items = coordination.get("Items")
        if not isinstance(items, list) or len(items) < 2:
            return "node.Coordination.Items:need_at_least_2"
        for index, item in enumerate(items):
            reason = stub_text_reason(
                item,
                current_text,
                f"node.Coordination.Items[{index}]",
            )
            if reason:
                return reason

    reason = _plan_description_items_reason(
        value.get("DescriptionGroups"),
        current_text,
        root_text,
        "node.DescriptionGroups",
        top_level=True,
    )
    if reason:
        return reason

    reference = value.get("Reference")
    if reference is not None:
        if not isinstance(reference, dict) or set(reference) != {"Type", "Text"}:
            return "node.Reference:invalid_schema"
        if reference.get("Type") not in REFERENCE_TYPES:
            return "node.Reference:invalid_type" f"({reference.get('Type')!r})"
        ref_text = reference.get("Text")
        if not isinstance(ref_text, str):
            return "node.Reference.Text:not_string"

        if reference.get("Type") == "Possessor":
            ref_text = canonicalize_possessor_reference_text(ref_text)
            if not ref_text:
                return "node.Reference.Text:empty_possessor"

        if not phrase_grounded(ref_text, context):
            return f"node.Reference.Text:not_grounded({ref_text!r})"

    if head is None and coordination is None and reference is None:
        return "node:missing_head_coordination_reference"

    return ""


def install_entity_plan(
    task: Dict,
    plan: Dict,
    max_depth: int,
    max_nodes_per_graph: int,
) -> None:
    state: GraphState = task["state"]
    node_id = task["node_id"]
    depth = state.meta[node_id]["depth"]
    local_endpoint_cache: Dict[Tuple[str, str], str] = {}

    def install_child(
        text: str,
        parent_hint: Dict,
        role: str,
        allow_unique_existing: bool,
    ) -> str:
        if depth + 1 > max_depth:
            return create_fallback_child(
                state,
                text,
                depth + 1,
                parent_hint,
                max_nodes_per_graph,
            )

        return intern_child_node(
            state,
            node_id,
            text,
            depth + 1,
            parent_hint,
            local_endpoint_cache,
            role=role,
            max_nodes_per_graph=max_nodes_per_graph,
            allow_unique_existing=allow_unique_existing,
        )

    coordination_out = None
    coordination = plan.get("Coordination")
    if coordination is not None:
        coordination_out = {
            "Operator": coordination["Operator"],
            "Items": [],
        }
        for item in coordination["Items"]:
            child_id = install_child(
                item["Text"],
                {
                    "Role": "CoordinationItem",
                    "Operator": coordination["Operator"],
                },
                role=f"coordination:{coordination['Operator']}",
                allow_unique_existing=False,
            )
            coordination_out["Items"].append({"Ref": child_id})

    def install_relation(relation: Dict) -> Dict:
        keys = set(relation)

        if keys == {"Predicate", "Target"}:
            predicate = relation["Predicate"]
            child_id = install_child(
                relation["Target"]["Text"],
                {"Role": "Target", "Predicate": predicate},
                role="relation_target",
                allow_unique_existing=True,
            )
            return {
                "Predicate": predicate,
                "Target": {"Ref": child_id},
            }

        if keys == {"Source", "Predicate"}:
            predicate = relation["Predicate"]
            child_id = install_child(
                relation["Source"]["Text"],
                {"Role": "Source", "Predicate": predicate},
                role="relation_source",
                allow_unique_existing=True,
            )
            return {
                "Source": {"Ref": child_id},
                "Predicate": predicate,
            }

        raise ValueError(f"Invalid simple Relation: {relation!r}")

    def install_description(item: Dict) -> Dict:
        keys = set(item)

        if keys == {"Modifier"}:
            return {"Modifier": item["Modifier"]}

        if keys == {"Relation"}:
            return {"Relation": install_relation(item["Relation"])}

        if keys == {"Event"}:
            return {
                "Event": install_event_frame(
                    task,
                    item["Event"],
                    max_depth,
                    max_nodes_per_graph,
                    allow_self=True,
                )
            }

        if keys == {"Operator", "Items"}:
            return {
                "Operator": item["Operator"],
                "Items": [install_description(child) for child in item["Items"]],
            }

        raise ValueError(f"Invalid description item: {item!r}")

    reference_out = copy.deepcopy(plan.get("Reference"))
    if (
        isinstance(reference_out, dict)
        and reference_out.get("Type") == "Possessor"
        and isinstance(reference_out.get("Text"), str)
    ):
        reference_out["Text"] = canonicalize_possessor_reference_text(
            reference_out["Text"]
        )

    state.nodes[node_id] = {
        "Text": task["CurrentText"],
        "Head": plan.get("Head"),
        "Coordination": coordination_out,
        "DescriptionGroups": [
            install_description(item) for item in plan.get("DescriptionGroups", [])
        ],
        "Reference": reference_out,
    }
    state.meta[node_id]["status"] = "expanded"


def graph_validation_reason(root_text: str, graph: Any, context: str) -> str:
    if not isinstance(graph, dict) or set(graph) != GRAPH_KEYS:
        return "graph:invalid_schema"

    root = graph.get("Root")
    nodes = graph.get("Nodes")
    if not isinstance(root, str):
        return "graph.Root:not_string"
    if not isinstance(nodes, dict):
        return "graph.Nodes:not_object"
    if root not in nodes:
        return "graph.Root:missing_node"
    if not root_text.strip():
        return ""

    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}

    for node_id, node in nodes.items():
        path = f"Nodes[{node_id!r}]"
        if not isinstance(node_id, str):
            return "graph.Nodes:non_string_id"
        if not isinstance(node, dict) or set(node) != NODE_KEYS:
            return f"{path}:invalid_node_schema"

        text = node.get("Text")
        if not isinstance(text, str) or not phrase_grounded(text, root_text):
            return f"{path}.Text:not_grounded({text!r})"

        body = {key: node[key] for key in ENTITY_BODY_KEYS}
        reason = entity_body_graph_reason(
            body,
            text,
            nodes,
            context,
            path,
        )
        if reason:
            return reason
        adjacency[node_id].update(entity_body_edges(body))

    visited: Set[str] = set()
    visiting: Set[str] = set()

    def dfs(node_id: str) -> str:
        if node_id in visiting:
            return f"graph:cycle_at({node_id})"
        if node_id in visited:
            return ""
        visiting.add(node_id)
        for child_id in adjacency[node_id]:
            reason = dfs(child_id)
            if reason:
                return reason
        visiting.remove(node_id)
        visited.add(node_id)
        return ""

    cycle_reason = dfs(root)
    if cycle_reason:
        return cycle_reason

    unreachable = sorted(set(nodes) - visited)
    if unreachable:
        return f"graph:unreachable_nodes({unreachable})"

    return ""


def validate_graph(root_text: str, graph: Any, context: str) -> bool:
    return not graph_validation_reason(root_text, graph, context)


def materialize_graph(graph: Dict) -> Dict:
    if not isinstance(graph, dict) or set(graph) != GRAPH_KEYS:
        raise ValueError("Invalid layered graph.")

    root = graph["Root"]
    nodes = graph["Nodes"]

    def expand_ref(ref: Dict, stack: Set[str]) -> Dict:
        if not isinstance(ref, dict) or set(ref) != REF_KEYS:
            raise ValueError(f"Invalid Ref: {ref!r}")
        return expand_node(ref["Ref"], stack)

    def expand_relation(relation: Dict, stack: Set[str]) -> Dict:
        keys = set(relation)

        if keys == {"Predicate", "Target"}:
            return {
                "Predicate": relation["Predicate"],
                "Target": expand_ref(relation["Target"], stack),
            }

        if keys == {"Source", "Predicate"}:
            return {
                "Source": expand_ref(relation["Source"], stack),
                "Predicate": relation["Predicate"],
            }

        raise ValueError(f"Invalid simple Relation: {relation!r}")

    def expand_description(item: Dict, stack: Set[str]) -> Dict:
        keys = set(item)

        if keys == {"Modifier"}:
            return {"Modifier": item["Modifier"]}

        if keys == {"Relation"}:
            return {"Relation": expand_relation(item["Relation"], stack)}

        if keys == {"Event"}:
            return {
                "Event": materialize_event_frame(
                    item["Event"],
                    expand_ref,
                    stack,
                )
            }

        if keys == {"Operator", "Items"}:
            return {
                "Operator": item["Operator"],
                "Items": [expand_description(child, stack) for child in item["Items"]],
            }

        raise ValueError(f"Invalid description item: {item!r}")

    def expand_node(node_id: str, stack: Set[str]) -> Dict:
        if node_id not in nodes:
            raise ValueError(f"Unknown node ref: {node_id}")
        if node_id in stack:
            raise ValueError(f"Cycle detected while materializing at {node_id}")

        stack = set(stack)
        stack.add(node_id)
        node = nodes[node_id]

        coordination = node.get("Coordination")
        coordination_out = None
        if coordination is not None:
            coordination_out = {
                "Operator": coordination["Operator"],
                "Items": [expand_ref(item, stack) for item in coordination["Items"]],
            }

        return {
            "Text": node["Text"],
            "Head": node["Head"],
            "Coordination": coordination_out,
            "DescriptionGroups": [
                expand_description(item, stack)
                for item in node.get("DescriptionGroups", [])
            ],
            "Reference": copy.deepcopy(node["Reference"]),
        }

    return expand_node(root, set())


def build_entity_system_prompt(
    entity_name: str,
    external_fields: Tuple[str, ...],
) -> str:
    return f"""
You decompose policy {entity_name} spans into one-level semantic node plans.

Preserve source wording, complete semantic participant boundaries,
explicit logical scope, and source-supported attachment.

Do not infer unstated facts, paraphrase semantic content,
or generate policy logic.

Return JSON only.
""".strip()


def build_entity_prompt(
    tasks: List[Dict],
    organization: str,
    entity_name: str,
    context_fields: Tuple[str, ...],
) -> str:
    root_key = f"Root{entity_name}"
    records = []

    for task in tasks:
        record = {
            "id": task["id"],
            "Sentence": task["PromptContext"].get("Sentence", ""),
            root_key: task["RootText"],
        }
        for field_name in context_fields:
            default = [] if field_name == "Condition" else ""
            record[field_name] = task["PromptContext"].get(
                field_name,
                default,
            )
        record["CurrentText"] = task["CurrentText"]
        record["ParentHint"] = task["ParentHint"]
        records.append(record)

    import json

    input_text = json.dumps(
        records,
        ensure_ascii=False,
        indent=2,
    )
    external = ", ".join(context_fields)
    output_root = "s0" if entity_name == "Subject" else "o0"

    return f"""
Policy issuer: {organization}

Input:
{input_text}

Return one result per id.

Current node plan:
{{
  "Text": "CurrentText",
  "Head": "source wording" | null,
  "Coordination": null | {{
    "Operator": "AND" | "OR",
    "Items": [{{"Text": "child source span"}}, ...]
  }},
  "DescriptionGroups": [description_item, ...],
  "Reference": null | {{
    "Type": "Coreference" | "Possessor",
    "Text": "antecedent/possessor source wording"
  }}
}}

description_item is exactly one of:
- {{"Modifier": "source wording"}}
- {{"Relation": {{"Predicate": "source wording", "Target": {{"Text": "child source span"}}}}}}
- {{"Relation": {{"Source": {{"Text": "child source span"}}, "Predicate": "source wording"}}}}
- {{"Event": {{
    "Action": "source wording",
    "Modifiers": ["source wording"],
    "Operators": ["source wording"],
    "Arguments": [{{
      "Role": "Subject" | "Object" | "Complement",
      "Marker": "source wording" | null,
      "Value": {{"Text": "child source span"}} | {{"Self": true}}
    }}]
  }}}}
- {{"Operator": "AND" | "OR", "Items": [description_item, ...]}}

Rules:

1. Scope and context
   Analyze only ONE semantic layer of CurrentText. Root{entity_name}, Sentence, and
   {external} are context only for resolving scope, attachment, ellipsis, participant
   boundaries, and whether a reference is {entity_name}-internal. Do not import their
   participants or wording into this graph. ParentHint only disambiguates the parent
   role already established by Python; do not reproduce that parent-child link.

2. Head
   Head is the minimal lexical semantic head and preserves source wording and
   capitalization. Omit semantically empty articles "a", "an", and "the".
   Put substantive determiners, quantifiers, and same-referent descriptors such as
   "any", "all", "each", "every", "no", "other", "this", and "such" in
   DescriptionGroups as Modifiers when they describe the current participant.
   If the node is primarily a Coordination with no single governing head, use Head=null.

3. DescriptionGroups
   DescriptionGroups is the only container for descriptions attached to the current
   participant; its top-level array is cumulative AND.
   - Modifier = same-referent description.
   - Relation = explicit semantic link to another participant.
   - Event = embedded event that cannot be represented losslessly by one Relation.
   - AND/OR group = explicit logical grouping among descriptions.
   If wording introduces a distinct participant, use Relation rather than Modifier.
   Do not output top-level Modifiers/Relations outside DescriptionGroups, redundant
   top-level AND, same-operator nesting, or duplicate descriptions in the same scope.

4. Relation and Event
   For a Relation, identify the COMPLETE Source/Target participant first; only then
   use the remaining source-grounded wording as Predicate. Keep relation direction
   faithful to the source. Explicit "A of B" normally becomes A --"of"--> B.
   Do not split lexicalized expressions such as "because of", "instead of", or "out of".
   A Relation represents one Source/Target pairing. Do not distort participant
   boundaries merely to split coordinated action wording.
   Action-level coordination may remain temporarily inside one Relation.Predicate
   when all coordinated actions share the same complete Source/Target boundary;
   downstream action normalization may decompose it later.
   Prefer a simple Relation when the current entity can serve as implicit Source or
   Target without losing participants. Use Event only when one Relation cannot
   preserve the embedded event. In Event, {{"Self": true}} means the current entity.
   Relative introducers "that", "which", and "who" are not Action wording unless
   inseparable from the lexical expression.

   Examples:
   - "A in which B R C"
     → may use Event(Action=R, Subject=B, Object=C,
       Complement(marker="in", Self=true)) when a simple Relation is insufficient.
   - "A of a M1, M2, or R-to-X N"
     → A --"of"--> N; N keeps OR(M1, M2, R-to-X).
     Do not absorb M1, M2, or R-to-X into the "of" Predicate.
   - "R1, R2, or R3 the N of X"
     → when "the N of X" is genuinely the complete shared argument of all relations,
       each Relation uses that complete Target.
     Do not absorb N or "of X" into Predicate merely to force a smaller shared Target.
   - For "A R-ed by B", when A is the current participant and "R-ed by" expresses
    the source-grounded relation between A and B, preserve the passive wording
    and use B as the Target:
        A --"R-ed by"--> B
    Do NOT reverse the edge merely because B is the semantic agent.
    Examples:
    - "principles established by the Constitution"
        → principles --"established by"--> Constitution
    
5. Coordination and shared scope
   Coordination is participant-level AND/OR between distinct independently referable
   participants. Preserve an explicit or semantically clear AND/OR.
   For a flat enumeration, use OR only when syntax/context clearly presents alternatives;
   do not invent an operator merely to force Coordination.
   Shared material belongs on the coordination parent when its scope over multiple
   participants is syntactically or semantically supported. Do not attach shared
   material only to the first/last child merely because of linear adjacency.
   Use child-only attachment when the source clearly supports narrower scope.

   Leading shared description example:
   - "other computer code, files, or programs"
     → parent Modifier="other"
     → parent Coordination OR("computer code", "files", "programs").

   Trailing shared description example:
   - "computer code, files, or programs designed to interfere with X"
     → parent Coordination OR("computer code", "files", "programs")
     → parent Relation Predicate="designed to interfere with", Target="X".

   Mixed coordinated relation example:
   - "other computer code, files, or programs designed to interfere with, destroy,
      or limit the functionality of X"
     → parent Modifier="other"
     → parent Coordination OR("computer code", "files", "programs")
     → parent Relation Predicate="designed to interfere with, destroy, or limit"
       Target="the functionality of X".
     Preserve the complete participant boundary before action-level normalization;
     do not move "functionality" into Predicate.

   Only when shared versus local scope remains genuinely ambiguous after considering
   syntax, participant structure, and context should the narrowest explicitly supported
   attachment be preferred.

   Contrasting example:
   - "the artificial intelligence-generated watermarks or prominent labels for
      deep synthesis content involved in this service"
     → keep "for deep synthesis content involved in this service" with
       "prominent labels", not with the whole OR Coordination.
     Correct grouping:
       artificial intelligence-generated watermarks
       OR
       [prominent labels for deep synthesis content involved in this service]

6. Reference
   Reference is only for referring wording.
   - Coreference = same participant.
   - Possessor = possessive reference.
   For Possessor, Reference.Text identifies the possessor, removing only a trailing
   possessive suffix ('s, ’s, ' or ’) while preserving source wording/capitalization.
   Examples:
   - "Alice's consent" → Reference.Text="Alice"
   - "users' data" → Reference.Text="users"
   Reference.Text must be grounded inside Root{entity_name}. If the antecedent exists
   only in {external}, keep Reference=null and defer the cross-field link.
   Prefer explicit local Relation over Reference. If a Relation endpoint already
   unambiguously refers to an existing {entity_name}-internal participant, reuse it.
   Special case:
    - For a local pronoun inside the same Root{entity_name}, resolve it to an earlier
    explicit participant when the antecedent is unambiguous.

    Example:
        "this service and any content it contains"

    → "it" refers to "this service".

    Prefer:
        this service --"contains"--> content

    If direct reuse of the existing participant is not available at the current
    one-layer expansion step, keep the pronoun node but set:

        Reference = {{
        "Type": "Coreference",
        "Text": "this service"
        }}

    Do not leave "it" as an ordinary entity with Reference=null when the local
    antecedent is explicit and unambiguous.
  
7. Final representation and child stubs
   When multiple representations remain valid, choose the one preserving explicit
   logical scope, complete participant boundaries, source-supported attachment, and
   substantive source meaning with the least inference. Do not flatten structure when
   flattening would lose a Modifier, Relation, Event, logical operator, participant
   boundary, or attachment distinction.
   Every child participant stub must contain only {{"Text": "source span"}}.
   This applies to Coordination Items, Relation Source/Target, and Event argument Values.
   Event arguments may instead use exactly {{"Self": true}} when permitted.
   Do not recursively expand child stubs in the same response; Python expands them later.
   Always include DescriptionGroups, using [] when there are no descriptions.

Output:
[
  {{"id": "r0_main:{output_root}", "Node": node_plan}}
]
""".strip()


def _collect_entity_targets(
    rules: List[Dict],
    phase: str,
) -> List[Dict]:
    targets = []

    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue

        if phase == "main":
            element = rule.get("Main_Clause_Elements")
            if isinstance(element, dict):
                targets.append(
                    {
                        "id": f"r{rule_index}_main",
                        "element": element,
                    }
                )
            continue

        if phase == "example":
            elements = rule.get("Example_Sentences_Elements", [])
            if isinstance(elements, list):
                for example_index, element in enumerate(elements):
                    if isinstance(element, dict):
                        targets.append(
                            {
                                "id": (f"r{rule_index}_example" f"{example_index}"),
                                "element": element,
                            }
                        )
            continue

        raise ValueError("phase must be 'main' or 'example'.")

    return targets


def _parse_entity_response(
    response: str,
    tasks: List[Dict],
    parse_json: Callable[[str], Any],
    entity_name: str,
) -> Tuple[Dict, List[Dict], List[str]]:
    parsed = parse_json(response)
    if not isinstance(parsed, list):
        raise ValueError(f"Shallow {entity_name} decomposition output must be a list.")

    expected = {task["id"]: task for task in tasks}
    item_map = {}
    duplicates = set()

    for item in parsed:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id not in expected:
            continue
        if item_id in item_map:
            duplicates.add(item_id)
            continue
        item_map[item_id] = item

    result = {}
    failed = []
    failure_details = []

    for task in tasks:
        task_id = task["id"]

        if task_id in duplicates:
            failed.append(task)
            failure_details.append(f"{task_id}:duplicate_id")
            continue

        output_item = item_map.get(task_id)
        if not isinstance(output_item, dict):
            failed.append(task)
            failure_details.append(f"{task_id}:missing_id")
            continue

        if "Node" not in output_item:
            failed.append(task)
            failure_details.append(f"{task_id}:missing_Node")
            continue

        plan = output_item.get("Node")
        if isinstance(plan, dict):
            plan = dict(plan)
            plan["Text"] = task["CurrentText"]

        local_task = dict(task)
        local_task["Context"] = task["RootText"]

        reason = entity_plan_reason(local_task, plan)
        if reason:
            failed.append(task)
            failure_details.append(
                f"{task_id}:{reason};" f"CurrentText={task['CurrentText']!r}"
            )
            continue

        result[task_id] = plan

    return result, failed, failure_details


def entity_graph_validation_reason(
    root_text: str,
    graph: Any,
    _context: str,
    *,
    node_prefix: str,
    entity_name: str,
) -> str:
    if not isinstance(graph, dict) or set(graph) != GRAPH_KEYS:
        return "graph:invalid_schema"

    root = graph.get("Root")
    nodes = graph.get("Nodes")

    if not isinstance(root, str) or not root.startswith(node_prefix):
        return f"graph.Root:invalid_{entity_name.lower()}_prefix" f"({root!r})"

    if not isinstance(nodes, dict):
        return "graph.Nodes:not_object"

    for node_id, node in nodes.items():
        path = f"Nodes[{node_id!r}]"

        if not isinstance(node_id, str) or not node_id.startswith(node_prefix):
            return f"{path}:invalid_{entity_name.lower()}_prefix"

        if not isinstance(node, dict) or set(node) != NODE_KEYS:
            return f"{path}:invalid_node_schema"

    if entity_name == "Object" and len(nodes) == 1 and root in nodes:
        root_node = nodes[root]
        head = root_node.get("Head")
        unresolved_coordination = bool(
            isinstance(head, str)
            and normalize_space(head).casefold()
            == normalize_space(root_text).casefold()
            and len(tokenize(root_text)) >= 4
            and re.search(r"[,;]|\b(?:and|or)\b", root_text, re.I)
        )
        if unresolved_coordination:
            return "graph.Root:unresolved_complex_singleton"

    # Subject/Object references are local to their own root span.
    return graph_validation_reason(
        root_text,
        graph,
        root_text,
    )


def materialize_entity_graph(
    graph: Dict,
    *,
    node_prefix: str,
    entity_name: str,
) -> Dict:
    if not isinstance(graph, dict) or set(graph) != GRAPH_KEYS:
        raise ValueError(f"Invalid {entity_name} layered graph.")

    root = graph.get("Root")
    if not isinstance(root, str) or not root.startswith(node_prefix):
        raise ValueError(f"Invalid {entity_name} root prefix.")

    return materialize_graph(graph)


def _process_entity_targets(
    llm_client: Any,
    targets: List[Dict],
    organization: str,
    *,
    entity_name: str,
    element_field: str,
    output_field: str,
    node_prefix: str,
    context_fields: Tuple[str, ...],
    parse_json: Callable[[str], Any],
    batch_size: int,
    max_retries: int,
    phase: str,
    max_depth: int,
    max_nodes_per_graph: int,
) -> Dict[str, int]:
    candidates = []
    empty_count = 0
    existing_count = 0
    deterministic_count = 0

    system_prompt = build_entity_system_prompt(
        entity_name,
        context_fields,
    )

    def prompt_builder(
        tasks: List[Dict],
        issuer: str,
    ) -> str:
        return build_entity_prompt(
            tasks,
            issuer,
            entity_name,
            context_fields,
        )

    def response_parser(
        response: str,
        tasks: List[Dict],
    ):
        return _parse_entity_response(
            response,
            tasks,
            parse_json,
            entity_name,
        )

    def final_reason(
        root_text: str,
        graph: Any,
        context: str,
    ) -> str:
        return entity_graph_validation_reason(
            root_text,
            graph,
            context,
            node_prefix=node_prefix,
            entity_name=entity_name,
        )

    for target in targets:
        element = target["element"]
        entity_text = element.get(element_field, "")

        if not isinstance(entity_text, str) or not entity_text.strip():
            element[output_field] = fallback_graph(
                "",
                node_prefix=node_prefix,
            )
            empty_count += 1
            continue

        entity_text = entity_text.strip()
        context = build_context_text(element)
        existing = element.get(output_field)

        if not final_reason(entity_text, existing, context):
            existing_count += 1
            continue

        deterministic = deterministic_graph(
            entity_text,
            node_prefix=node_prefix,
        )
        if deterministic is not None:
            element[output_field] = deterministic
            deterministic_count += 1
            continue

        prompt_context = {
            "Sentence": element.get("Sentence", ""),
        }
        for field_name in context_fields:
            default = [] if field_name == "Condition" else ""
            prompt_context[field_name] = element.get(
                field_name,
                default,
            )

        candidates.append(
            {
                "id": target["id"],
                "element": element,
                "RootText": entity_text,
                "Context": context,
                "NodePrefix": node_prefix,
                "PromptContext": prompt_context,
            }
        )

    graphs, expansion_stats = expand_layered_graphs(
        llm_client=llm_client,
        candidates=candidates,
        organization=organization,
        system_prompt=system_prompt,
        build_prompt=prompt_builder,
        parse_response=response_parser,
        install_plan=install_entity_plan,
        phase=phase,
        log_label=f"Split{entity_name}Graph",
        batch_size=batch_size,
        max_retries=max_retries,
        max_depth=max_depth,
        max_nodes_per_graph=max_nodes_per_graph,
        final_graph_reason=final_reason,
    )

    for candidate in candidates:
        candidate["element"][output_field] = graphs[candidate["id"]]

    return {
        "targets": len(targets),
        "candidates": len(candidates),
        "empty": empty_count,
        "existing": existing_count,
        "deterministic": deterministic_count,
        "node_batches": expansion_stats["node_batches"],
        "llm_nodes": expansion_stats["llm_nodes"],
        "deterministic_nodes": expansion_stats["deterministic_nodes"],
        "retries": expansion_stats["retries"],
        "fallback_nodes": expansion_stats["fallback_nodes"],
        "invalid_graphs": expansion_stats["invalid_graphs"],
        "total_nodes": expansion_stats["total_nodes"],
    }


def split_entity_groups_batch(
    llm_client: Any,
    rules: List[Dict],
    organization: str,
    *,
    entity_name: str,
    element_field: str,
    output_field: str,
    node_prefix: str,
    context_fields: Tuple[str, ...],
    parse_json: Callable[[str], Any],
    batch_size: int = 10,
    max_retries: int = 2,
    max_depth: int = 12,
    max_nodes_per_graph: int = 256,
) -> List[Dict]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")
    if max_depth <= 0:
        raise ValueError("max_depth must be greater than 0.")
    if max_nodes_per_graph <= 0:
        raise ValueError("max_nodes_per_graph must be greater than 0.")

    main_targets = _collect_entity_targets(rules, "main")
    example_targets = _collect_entity_targets(rules, "example")
    usage_before = llm_client.get_usage()

    logger.info(
        "Split%sGraph | start | rules=%d | " "main_targets=%d | example_targets=%d",
        entity_name,
        len(rules),
        len(main_targets),
        len(example_targets),
    )

    common = {
        "entity_name": entity_name,
        "element_field": element_field,
        "output_field": output_field,
        "node_prefix": node_prefix,
        "context_fields": context_fields,
        "parse_json": parse_json,
        "batch_size": batch_size,
        "max_retries": max_retries,
        "max_depth": max_depth,
        "max_nodes_per_graph": max_nodes_per_graph,
    }

    main_stats = _process_entity_targets(
        llm_client,
        main_targets,
        organization,
        phase="main",
        **common,
    )
    example_stats = _process_entity_targets(
        llm_client,
        example_targets,
        organization,
        phase="example",
        **common,
    )

    usage = llm_client.get_usage_since(usage_before)

    logger.info(
        "Split%sGraph | completed | "
        "main_targets=%d | main_graphs=%d | "
        "example_targets=%d | example_graphs=%d | "
        "existing=%d | deterministic=%d | empty=%d | "
        "total_nodes=%d | llm_nodes=%d | "
        "deterministic_nodes=%d | node_batches=%d | retries=%d | "
        "fallback_nodes=%d | invalid_graphs=%d | llm_calls=%d | "
        "input_tokens=%d | output_tokens=%d | total_tokens=%d",
        entity_name,
        main_stats["targets"],
        main_stats["candidates"],
        example_stats["targets"],
        example_stats["candidates"],
        main_stats["existing"] + example_stats["existing"],
        main_stats["deterministic"] + example_stats["deterministic"],
        main_stats["empty"] + example_stats["empty"],
        main_stats["total_nodes"] + example_stats["total_nodes"],
        main_stats["llm_nodes"] + example_stats["llm_nodes"],
        (main_stats["deterministic_nodes"] + example_stats["deterministic_nodes"]),
        main_stats["node_batches"] + example_stats["node_batches"],
        main_stats["retries"] + example_stats["retries"],
        main_stats["fallback_nodes"] + example_stats["fallback_nodes"],
        main_stats["invalid_graphs"] + example_stats["invalid_graphs"],
        usage["api_calls"],
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
    )

    return rules


def expand_layered_graphs(
    llm_client: Any,
    candidates: List[Dict],
    organization: str,
    system_prompt: str,
    build_prompt: Callable[[List[Dict], str], str],
    parse_response: Callable[[str, List[Dict]], Tuple[Dict, List[Dict], List[str]]],
    install_plan: Callable[[Dict, Dict, int, int], None],
    phase: str,
    log_label: str,
    batch_size: int,
    max_retries: int,
    max_depth: int,
    max_nodes_per_graph: int,
    final_graph_reason: Callable[[str, Any, str], str] = graph_validation_reason,
    deterministic_expander: Callable[
        [GraphState, str], bool
    ] = expand_deterministic_pending,
    fallback_node_handler: Callable[[GraphState, str], None] = fallback_pending_node,
    fallback_graph_factory: Callable[[str, str], Dict] = fallback_graph,
) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    states = [make_graph_state(candidate) for candidate in candidates]
    retries = 0
    node_batches = 0
    llm_nodes = 0
    deterministic_nodes = 0
    fallback_nodes = 0
    round_index = 0

    while True:
        pending = pending_tasks(states)
        if not pending:
            break

        round_index += 1
        llm_pending = []
        for task in pending:
            state: GraphState = task["state"]
            node_id = task["node_id"]
            if deterministic_expander(state, node_id):
                deterministic_nodes += 1
            else:
                llm_pending.append(task)

        if not llm_pending:
            continue

        logger.info(
            "%s | %s | round=%d | pending_nodes=%d | llm_nodes=%d",
            log_label,
            phase,
            round_index,
            len(pending),
            len(llm_pending),
        )

        for start in range(0, len(llm_pending), batch_size):
            batch = llm_pending[start : start + batch_size]
            node_batches += 1
            remaining = batch
            usage_before = llm_client.get_usage()

            for attempt in range(1, max_retries + 1):
                if not remaining:
                    break

                try:
                    response = llm_client.invoke(
                        build_prompt(remaining, organization),
                        system_prompt=system_prompt,
                    )
                    if response is None or (
                        isinstance(response, str) and not response.strip()
                    ):
                        raise ValueError("Empty layered graph response.")

                    parsed, failed, failure_details = parse_response(
                        response,
                        remaining,
                    )
                    failed_ids = {task["id"] for task in failed}
                    installed_failures = []

                    for task in remaining:
                        if task["id"] in failed_ids:
                            continue
                        plan = parsed.get(task["id"])
                        if plan is None:
                            continue
                        try:
                            install_plan(
                                task,
                                plan,
                                max_depth,
                                max_nodes_per_graph,
                            )
                            llm_nodes += 1
                        except Exception as exc:
                            installed_failures.append(task)
                            failure_details.append(f"{task['id']}:install_error({exc})")

                    failed.extend(installed_failures)
                    if failed and attempt < max_retries:
                        retries += 1
                        logger.warning(
                            "%s | %s round=%d batch=%d | "
                            "attempt=%d/%d failed | retry_nodes=%d | reasons=%s",
                            log_label,
                            phase,
                            round_index,
                            node_batches,
                            attempt,
                            max_retries,
                            len(failed),
                            "; ".join(failure_details),
                        )
                    remaining = failed

                except Exception as exc:
                    if attempt < max_retries:
                        retries += 1
                    logger.warning(
                        "%s | %s round=%d batch=%d | "
                        "attempt=%d/%d failed | error=%s",
                        log_label,
                        phase,
                        round_index,
                        node_batches,
                        attempt,
                        max_retries,
                        exc,
                    )

            usage = llm_client.get_usage_since(usage_before)
            if remaining:
                for task in remaining:
                    state: GraphState = task["state"]
                    node_id = task["node_id"]
                    if state.meta[node_id].get("status") == "pending":
                        fallback_node_handler(state, node_id)
                        fallback_nodes += 1

                logger.error(
                    "%s | %s round=%d batch=%d failed | "
                    "fallback_nodes=%d | llm_calls=%d",
                    log_label,
                    phase,
                    round_index,
                    node_batches,
                    len(remaining),
                    usage["api_calls"],
                )
            else:
                logger.info(
                    "%s | %s round=%d batch=%d | nodes=%d | llm_calls=%d",
                    log_label,
                    phase,
                    round_index,
                    node_batches,
                    len(batch),
                    usage["api_calls"],
                )

    graphs = {}
    invalid_graphs = 0
    total_nodes = 0

    for state in states:
        graph = {
            "Root": state.root_id,
            "Nodes": state.nodes,
        }
        reason = final_graph_reason(state.root_text, graph, state.context)
        if reason:
            invalid_graphs += 1
            logger.error(
                "%s | %s | id=%s final_graph_invalid | reason=%s | fallback=root",
                log_label,
                phase,
                state.candidate_id,
                reason,
            )
            graph = fallback_graph_factory(
                state.root_text,
                state.node_prefix,
            )
        graphs[state.candidate_id] = graph
        total_nodes += len(graph["Nodes"])

    return graphs, {
        "candidates": len(candidates),
        "node_batches": node_batches,
        "llm_nodes": llm_nodes,
        "deterministic_nodes": deterministic_nodes,
        "retries": retries,
        "fallback_nodes": fallback_nodes,
        "invalid_graphs": invalid_graphs,
        "total_nodes": total_nodes,
    }
