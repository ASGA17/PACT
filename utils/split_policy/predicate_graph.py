import copy
import logging
import re
from typing import Any, Dict, List, Set

from llms import LLMClient
from utils.utils import parse_json_response
from utils.split_policy.action_span import (
    ACTION_SPAN_SYSTEM_PROMPT,
    action_span_plan_reason,
    build_action_span_prompt,
    parse_action_span_response,
)
from utils.split_policy.layered_graph import (
    GRAPH_KEYS,
    OPERATORS,
    GraphState,
    action_body_reason,
    build_context_text,
    expand_layered_graphs,
    intern_child_node,
    phrase_grounded,
    simple_token,
)

logger = logging.getLogger(__name__)


PREDICATE_NODE_KEYS = {
    "Text",
    "Action",
    "Modifiers",
    "Operators",
    "Coordination",
}

PREDICATE_COVERAGE_IGNORED = {
    "a",
    "an",
    "and",
    "as",
    "for",
    "in",
    "nor",
    "of",
    "or",
    "the",
    "to",
}


def _predicate_content_tokens(value: str) -> List[str]:
    return [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", value or "")
        if token.casefold() not in PREDICATE_COVERAGE_IGNORED
    ]


SYSTEM_PROMPT = ACTION_SPAN_SYSTEM_PROMPT


def _collect_targets(rules: List[Dict], phase: str) -> List[Dict]:
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
                                "id": f"r{rule_index}_example{example_index}",
                                "element": element,
                            }
                        )
            continue

        raise ValueError("phase must be 'main' or 'example'.")

    return targets


def _build_predicate_prompt(tasks: List[Dict], organization: str) -> str:
    return build_action_span_prompt(
        tasks,
        organization,
        role_name="Predicate",
    )


def _predicate_plan_reason(task: Dict, value: Any) -> str:
    return action_span_plan_reason(task, value)


def _parse_predicate_response(
    response: str,
    tasks: List[Dict],
):
    return parse_action_span_response(
        response,
        tasks,
        parse_json_response,
    )


def _fallback_predicate_graph(
    root_text: str,
    node_prefix: str = "p",
) -> Dict:
    text = root_text.strip() if isinstance(root_text, str) else ""
    root_id = f"{node_prefix}0"

    return {
        "Root": root_id,
        "Nodes": {
            root_id: {
                "Text": text,
                "Action": text or None,
                "Modifiers": [],
                "Operators": [],
                "Coordination": None,
            }
        },
    }


def _fallback_predicate_node(state: GraphState, node_id: str) -> None:
    text = state.nodes[node_id].get("Text", "")

    state.nodes[node_id] = {
        "Text": text,
        "Action": text or None,
        "Modifiers": [],
        "Operators": [],
        "Coordination": None,
    }
    state.meta[node_id]["status"] = "expanded"
    state.meta[node_id]["fallback"] = True
    state.fallback_nodes += 1


def _expand_predicate_deterministic(
    state: GraphState,
    node_id: str,
) -> bool:
    text = state.nodes[node_id].get("Text", "")

    if not simple_token(text):
        return False

    state.nodes[node_id] = {
        "Text": text,
        "Action": text,
        "Modifiers": [],
        "Operators": [],
        "Coordination": None,
    }
    state.meta[node_id]["status"] = "expanded"
    state.meta[node_id]["deterministic"] = True
    return True


def _install_predicate_plan(
    task: Dict,
    plan: Dict,
    max_depth: int,
    max_nodes_per_graph: int,
) -> None:
    state: GraphState = task["state"]
    node_id = task["node_id"]
    depth = state.meta[node_id]["depth"]

    coordination_out = None
    coordination = plan.get("Coordination")

    if coordination is not None:
        coordination_out = {
            "Operator": coordination["Operator"],
            "Items": [],
        }
        local_cache = {}

        for item in coordination["Items"]:
            text = item["Text"]
            parent_hint = {
                "Role": "PredicateCoordinationItem",
                "Operator": coordination["Operator"],
            }

            if depth + 1 > max_depth:
                if len(state.nodes) >= max_nodes_per_graph:
                    raise ValueError(
                        f"max_nodes_per_graph exceeded for "
                        f"{state.candidate_id}: {max_nodes_per_graph}"
                    )

                child_id = state.new_node(
                    text,
                    depth + 1,
                    parent_hint,
                )
                _fallback_predicate_node(state, child_id)
            else:
                child_id = intern_child_node(
                    state,
                    node_id,
                    text,
                    depth + 1,
                    parent_hint,
                    local_cache,
                    role=f"predicate_coordination:{coordination['Operator']}",
                    max_nodes_per_graph=max_nodes_per_graph,
                    allow_unique_existing=False,
                )

            coordination_out["Items"].append({"Ref": child_id})

    state.nodes[node_id] = {
        "Text": task["CurrentText"],
        "Action": plan.get("Action"),
        "Modifiers": copy.deepcopy(plan.get("Modifiers", [])),
        "Operators": copy.deepcopy(plan.get("Operators", [])),
        "Coordination": coordination_out,
    }
    state.meta[node_id]["status"] = "expanded"


def _predicate_graph_reason(
    root_text: str,
    graph: Any,
    _context: str,
) -> str:
    if not isinstance(graph, dict) or set(graph) != GRAPH_KEYS:
        return "graph:invalid_schema"

    root = graph.get("Root")
    nodes = graph.get("Nodes")

    if not isinstance(root, str) or not root.startswith("p"):
        return f"graph.Root:invalid_predicate_prefix({root!r})"

    if not isinstance(nodes, dict):
        return "graph.Nodes:not_object"

    if root not in nodes:
        return "graph.Root:missing_node"

    if not root_text.strip():
        return ""

    def ref_reason(value: Any, path: str) -> str:
        if not isinstance(value, dict) or set(value) != {"Ref"}:
            return f"{path}:invalid_ref"

        ref_id = value.get("Ref")
        if not isinstance(ref_id, str) or ref_id not in nodes:
            return f"{path}:missing_ref({ref_id!r})"

        return ""

    for node_id, node in nodes.items():
        path = f"Nodes[{node_id!r}]"

        if not isinstance(node_id, str) or not node_id.startswith("p"):
            return f"{path}:invalid_predicate_prefix"

        if not isinstance(node, dict) or set(node) != PREDICATE_NODE_KEYS:
            return f"{path}:invalid_node_schema"

        text = node.get("Text")
        if not isinstance(text, str) or not phrase_grounded(text, root_text):
            return f"{path}.Text:not_grounded({text!r})"

        action = node.get("Action")
        modifiers = node.get("Modifiers")
        operators = node.get("Operators")
        reason = action_body_reason(
            {
                "Action": action,
                "Modifiers": modifiers,
                "Operators": operators,
            },
            text,
            path,
            require_action=False,
        )
        if reason:
            return reason

        coordination = node.get("Coordination")
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
                reason = ref_reason(
                    item,
                    f"{path}.Coordination.Items[{index}]",
                )
                if reason:
                    return reason

                ref_id = item["Ref"]
                if ref_id in seen_refs:
                    return f"{path}.Coordination.Items[{index}]:duplicate_ref"
                seen_refs.add(ref_id)

        if action is None and coordination is None:
            return f"{path}:missing_action_or_coordination"

        if action is not None and coordination is not None:
            return f"{path}:action_and_coordination_both_present"

        if action is not None:
            unresolved_coordination = (
                len(nodes) == 1
                and action.strip().casefold() == text.strip().casefold()
                and bool(re.search(r"[,;]|\b(?:and|or|nor)\b", text, re.I))
            )
            if unresolved_coordination:
                return f"{path}:unresolved_complex_singleton"

            represented = " ".join([action, *modifiers, *operators])
            represented_tokens = set(_predicate_content_tokens(represented))
            missing = [
                token
                for token in _predicate_content_tokens(text)
                if token not in represented_tokens
            ]
            if missing:
                return f"{path}:unrepresented_content({missing})"

    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}

    for node_id, node in nodes.items():
        coordination = node.get("Coordination")
        if coordination is None:
            continue

        for item in coordination["Items"]:
            adjacency[node_id].add(item["Ref"])

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


def _validate_predicate_graph(
    root_text: str,
    graph: Any,
    context: str,
) -> bool:
    return not _predicate_graph_reason(
        root_text,
        graph,
        context,
    )


def materialize_predicate_graph(graph: Dict) -> Dict:
    if not isinstance(graph, dict) or set(graph) != GRAPH_KEYS:
        raise ValueError("Invalid Predicate_Elements graph.")

    root = graph["Root"]
    nodes = graph["Nodes"]

    def expand_node(node_id: str, stack: Set[str]) -> Dict:
        if node_id not in nodes:
            raise ValueError(f"Unknown predicate node ref: {node_id}")

        if node_id in stack:
            raise ValueError(
                f"Cycle detected while materializing predicate at {node_id}"
            )

        stack = set(stack)
        stack.add(node_id)
        node = nodes[node_id]

        coordination = node.get("Coordination")
        if coordination is None:
            coordination_out = None
        else:
            coordination_out = {
                "Operator": coordination["Operator"],
                "Items": [
                    expand_node(item["Ref"], stack) for item in coordination["Items"]
                ],
            }

        return {
            "Text": node["Text"],
            "Action": node["Action"],
            "Modifiers": copy.deepcopy(node["Modifiers"]),
            "Operators": copy.deepcopy(node["Operators"]),
            "Coordination": coordination_out,
        }

    return expand_node(root, set())


def _process_targets(
    llm_client: LLMClient,
    targets: List[Dict],
    organization: str,
    batch_size: int,
    max_retries: int,
    phase: str,
    max_depth: int,
    max_nodes_per_predicate: int,
) -> Dict[str, int]:
    candidates = []
    empty_count = 0
    existing_count = 0
    deterministic_count = 0

    for target in targets:
        element = target["element"]
        predicate_text = element.get("Predicate", "")

        if not isinstance(predicate_text, str) or not predicate_text.strip():
            element["Predicate_Elements"] = _fallback_predicate_graph(
                "",
                node_prefix="p",
            )
            empty_count += 1
            continue

        predicate_text = predicate_text.strip()
        context = build_context_text(element)
        existing = element.get("Predicate_Elements")

        if _validate_predicate_graph(predicate_text, existing, context):
            existing_count += 1
            continue

        if simple_token(predicate_text):
            element["Predicate_Elements"] = _fallback_predicate_graph(
                predicate_text,
                node_prefix="p",
            )
            deterministic_count += 1
            continue

        candidates.append(
            {
                "id": target["id"],
                "element": element,
                "RootText": predicate_text,
                "Context": context,
                "NodePrefix": "p",
                "PromptContext": {
                    "Sentence": element.get("Sentence", ""),
                    "Subject": element.get("Subject", ""),
                    "Object": element.get("Object", ""),
                    "Condition": element.get("Condition", []),
                },
            }
        )

    graphs, expansion_stats = expand_layered_graphs(
        llm_client=llm_client,
        candidates=candidates,
        organization=organization,
        system_prompt=SYSTEM_PROMPT,
        build_prompt=_build_predicate_prompt,
        parse_response=_parse_predicate_response,
        install_plan=_install_predicate_plan,
        phase=phase,
        log_label="SplitPredicateGraph",
        batch_size=batch_size,
        max_retries=max_retries,
        max_depth=max_depth,
        max_nodes_per_graph=max_nodes_per_predicate,
        final_graph_reason=_predicate_graph_reason,
        deterministic_expander=_expand_predicate_deterministic,
        fallback_node_handler=_fallback_predicate_node,
        fallback_graph_factory=_fallback_predicate_graph,
    )

    invalid_fallbacks = []
    for candidate in candidates:
        graph = graphs[candidate["id"]]
        reason = _predicate_graph_reason(
            candidate["RootText"],
            graph,
            candidate["Context"],
        )
        if reason:
            invalid_fallbacks.append(f"{candidate['id']}:{reason}")
            continue
        candidate["element"]["Predicate_Elements"] = graph

    if invalid_fallbacks:
        raise ValueError(
            "Predicate graph generation produced lossy fallback graphs: "
            + "; ".join(invalid_fallbacks)
        )

    return {
        "targets": len(targets),
        "candidates": len(candidates),
        "empty_predicates": empty_count,
        "existing": existing_count,
        "deterministic_predicates": deterministic_count,
        "node_batches": expansion_stats["node_batches"],
        "llm_nodes": expansion_stats["llm_nodes"],
        "deterministic_nodes": expansion_stats["deterministic_nodes"],
        "retries": expansion_stats["retries"],
        "fallback_nodes": expansion_stats["fallback_nodes"],
        "invalid_graphs": expansion_stats["invalid_graphs"],
        "total_nodes": expansion_stats["total_nodes"],
    }


def split_predicate_groups_batch(
    llm_client: LLMClient,
    rules: List[Dict],
    organization: str,
    batch_size: int = 10,
    max_retries: int = 2,
    max_depth: int = 12,
    max_nodes_per_predicate: int = 128,
) -> List[Dict]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")

    if max_depth <= 0:
        raise ValueError("max_depth must be greater than 0.")

    if max_nodes_per_predicate <= 0:
        raise ValueError("max_nodes_per_predicate must be greater than 0.")

    main_targets = _collect_targets(rules, "main")
    example_targets = _collect_targets(rules, "example")
    usage_before = llm_client.get_usage()

    logger.info(
        "SplitPredicateGraph | start | rules=%d | "
        "main_targets=%d | example_targets=%d",
        len(rules),
        len(main_targets),
        len(example_targets),
    )

    main_stats = _process_targets(
        llm_client,
        main_targets,
        organization,
        batch_size,
        max_retries,
        "main",
        max_depth,
        max_nodes_per_predicate,
    )

    example_stats = _process_targets(
        llm_client,
        example_targets,
        organization,
        batch_size,
        max_retries,
        "example",
        max_depth,
        max_nodes_per_predicate,
    )

    usage = llm_client.get_usage_since(usage_before)

    logger.info(
        "SplitPredicateGraph | completed | "
        "main_targets=%d | main_predicates=%d | "
        "example_targets=%d | example_predicates=%d | "
        "existing=%d | deterministic_predicates=%d | empty=%d | "
        "total_nodes=%d | llm_nodes=%d | deterministic_nodes=%d | "
        "node_batches=%d | retries=%d | fallback_nodes=%d | "
        "invalid_graphs=%d | llm_calls=%d | "
        "input_tokens=%d | output_tokens=%d | total_tokens=%d",
        main_stats["targets"],
        main_stats["candidates"],
        example_stats["targets"],
        example_stats["candidates"],
        main_stats["existing"] + example_stats["existing"],
        main_stats["deterministic_predicates"]
        + example_stats["deterministic_predicates"],
        main_stats["empty_predicates"] + example_stats["empty_predicates"],
        main_stats["total_nodes"] + example_stats["total_nodes"],
        main_stats["llm_nodes"] + example_stats["llm_nodes"],
        main_stats["deterministic_nodes"] + example_stats["deterministic_nodes"],
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
