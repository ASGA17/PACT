import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from utils.split_policy.action_span import (
    analyze_action_spans_batch,
    needs_action_decomposition,
)

logger = logging.getLogger(__name__)


def _usage_snapshot(llm_client: Any) -> Any:
    getter = getattr(llm_client, "get_usage", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _usage_since(llm_client: Any, before: Any) -> Dict[str, int]:
    getter = getattr(llm_client, "get_usage_since", None)
    if before is None or not callable(getter):
        return {}
    try:
        usage = getter(before)
    except Exception:
        return {}
    return usage if isinstance(usage, dict) else {}


def _iter_rule_elements(rules: List[Dict]):
    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        main = rule.get("Main_Clause_Elements")
        if isinstance(main, dict):
            yield rule_index, "main", main
        examples = rule.get("Example_Sentences_Elements", [])
        if isinstance(examples, list):
            for example_index, element in enumerate(examples):
                if isinstance(element, dict):
                    yield rule_index, f"example:{example_index}", element


def _collect_from_description_items(
    items: Any,
    *,
    owner_key: str,
    node_text: str,
    prompt_context: Dict,
    candidates: List[Dict],
    relation_ids: Dict[int, str],
) -> None:
    if not isinstance(items, list):
        return
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        keys = set(item)
        if keys == {"Relation"}:
            relation = item.get("Relation")
            if not isinstance(relation, dict):
                continue
            predicate = relation.get("Predicate")
            if not isinstance(predicate, str) or not needs_action_decomposition(
                predicate
            ):
                continue
            relation_obj_id = id(relation)
            if relation_obj_id in relation_ids:
                continue
            candidate_id = f"relact|{owner_key}|{len(candidates)}"
            relation_ids[relation_obj_id] = candidate_id
            candidates.append(
                {
                    "id": candidate_id,
                    "Text": predicate,
                    # Keep the original clause-role context unchanged.
                    # Subject/Object are therefore treated symmetrically; the
                    # embedded Relation.Predicate itself is carried by Text.
                    "PromptContext": dict(prompt_context),
                    "ParentHint": {
                        "Role": "RelationPredicate",
                        "Owner": owner_key,
                    },
                }
            )
            continue
        if keys == {"Operator", "Items"}:
            _collect_from_description_items(
                item.get("Items"),
                owner_key=f"{owner_key}.group{index}",
                node_text=node_text,
                prompt_context=prompt_context,
                candidates=candidates,
                relation_ids=relation_ids,
            )


def _shared_scope_parts(plan: Dict) -> List[str]:
    text = plan.get("Text", "")
    candidates = []
    for field in ("Operators", "Modifiers"):
        for item in plan.get(field, []) or []:
            if not isinstance(item, str) or not item.strip():
                continue
            pos = text.lower().find(item.lower()) if isinstance(text, str) else -1
            candidates.append((pos if pos >= 0 else 10**9, item.strip()))
    candidates.sort(key=lambda pair: pair[0])
    out = []
    seen = set()
    for _pos, item in candidates:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _apply_prefixes(text: str, prefixes: List[str]) -> str:
    result = text.strip()
    low = result.casefold()
    missing = []
    for prefix in prefixes:
        key = prefix.casefold()
        if key and key not in low:
            missing.append(prefix)
    if not missing:
        return result
    return " ".join(missing + [result]).strip()


def _action_tree_from_plan(
    plan: Dict,
    *,
    candidate_id: str,
    plan_map: Dict[str, Dict],
    request_map: Dict[str, Dict],
    depth: int = 0,
    max_depth: int = 8,
    inherited_prefixes: Optional[List[str]] = None,
) -> Dict:
    inherited_prefixes = list(inherited_prefixes or [])
    local_prefixes = _shared_scope_parts(plan)
    effective_prefixes = inherited_prefixes + [
        item
        for item in local_prefixes
        if item.casefold() not in {x.casefold() for x in inherited_prefixes}
    ]

    text = plan.get("Text", "")
    coordination = plan.get("Coordination")
    if depth >= max_depth or not isinstance(coordination, dict):
        return {
            "Text": _apply_prefixes(text, inherited_prefixes),
            "Coordination": None,
        }

    children = []
    for child_index, child in enumerate(coordination.get("Items", [])):
        child_text = child.get("Text", "") if isinstance(child, dict) else ""
        child_id = f"{candidate_id}/child{child_index}"
        child_plan = plan_map.get(child_id)
        if child_plan is None:
            children.append(
                {
                    "Text": _apply_prefixes(child_text, effective_prefixes),
                    "Coordination": None,
                }
            )
        else:
            children.append(
                _action_tree_from_plan(
                    child_plan,
                    candidate_id=child_id,
                    plan_map=plan_map,
                    request_map=request_map,
                    depth=depth + 1,
                    max_depth=max_depth,
                    inherited_prefixes=effective_prefixes,
                )
            )

    return {
        "Text": text,
        "Coordination": {
            "Operator": coordination.get("Operator"),
            "Items": children,
        },
    }


def _expand_requests_recursively(
    llm_client: Any,
    initial_requests: List[Dict],
    *,
    batch_size: int,
    max_retries: int,
    max_depth: int = 8,
) -> Dict[str, Dict]:
    plan_map: Dict[str, Dict] = {}
    request_map: Dict[str, Dict] = {item["id"]: item for item in initial_requests}
    pending = list(initial_requests)
    depth_by_id = {item["id"]: 0 for item in initial_requests}

    while pending:
        parsed = analyze_action_spans_batch(
            llm_client,
            pending,
            batch_size=batch_size,
            max_retries=max_retries,
        )
        next_pending: List[Dict] = []
        for request in pending:
            request_id = request["id"]
            plan = parsed.get(request_id)
            if plan is None:
                continue
            plan_map[request_id] = plan
            coordination = plan.get("Coordination")
            depth = depth_by_id.get(request_id, 0)
            if not isinstance(coordination, dict) or depth >= max_depth:
                continue
            for child_index, child in enumerate(coordination.get("Items", [])):
                if not isinstance(child, dict):
                    continue
                child_text = child.get("Text")
                if not isinstance(child_text, str) or not needs_action_decomposition(
                    child_text
                ):
                    continue
                child_id = f"{request_id}/child{child_index}"
                if child_id in request_map:
                    continue
                child_request = {
                    "id": child_id,
                    "Text": child_text,
                    "PromptContext": request.get("PromptContext", {}),
                    "ParentHint": {
                        "Role": "NestedRelationPredicate",
                        "Parent": request_id,
                    },
                }
                request_map[child_id] = child_request
                depth_by_id[child_id] = depth + 1
                next_pending.append(child_request)
        pending = next_pending

    return plan_map


def _relation_item_for_text(relation: Dict, predicate_text: str) -> Dict:
    new_relation = copy.deepcopy(relation)
    new_relation["Predicate"] = predicate_text
    return {"Relation": new_relation}


def _tree_to_description(relation: Dict, tree: Dict) -> Dict:
    coordination = tree.get("Coordination")
    if not isinstance(coordination, dict):
        return _relation_item_for_text(
            relation, tree.get("Text", relation.get("Predicate", ""))
        )

    operator = coordination.get("Operator")
    items = [
        _tree_to_description(relation, child) for child in coordination.get("Items", [])
    ]
    items = _flatten_same_operator(operator, items)
    if len(items) < 2:
        return _relation_item_for_text(relation, relation.get("Predicate", ""))
    return {"Operator": operator, "Items": items}


def _flatten_same_operator(operator: Optional[str], items: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for item in items:
        if (
            isinstance(item, dict)
            and set(item) == {"Operator", "Items"}
            and item.get("Operator") == operator
        ):
            nested = item.get("Items")
            if isinstance(nested, list):
                out.extend(nested)
                continue
        out.append(item)
    return out


def _rewrite_description_items(
    items: Any,
    *,
    relation_ids: Dict[int, str],
    plan_map: Dict[str, Dict],
) -> List[Dict]:
    if not isinstance(items, list):
        return []
    out: List[Dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        keys = set(item)
        if keys == {"Relation"}:
            relation = item.get("Relation")
            if not isinstance(relation, dict):
                out.append(copy.deepcopy(item))
                continue
            candidate_id = relation_ids.get(id(relation))
            plan = plan_map.get(candidate_id) if candidate_id else None
            if not isinstance(plan, dict) or not isinstance(
                plan.get("Coordination"), dict
            ):
                out.append(copy.deepcopy(item))
                continue
            tree = _action_tree_from_plan(
                plan,
                candidate_id=candidate_id,
                plan_map=plan_map,
                request_map={},
            )
            out.append(_tree_to_description(relation, tree))
            continue
        if keys == {"Operator", "Items"}:
            operator = item.get("Operator")
            children = _rewrite_description_items(
                item.get("Items"),
                relation_ids=relation_ids,
                plan_map=plan_map,
            )
            children = _flatten_same_operator(operator, children)
            if len(children) >= 2:
                out.append({"Operator": operator, "Items": children})
            else:
                out.extend(children)
            continue
        out.append(copy.deepcopy(item))
    return out


def _normalize_description_owners_batch(
    llm_client: Any,
    owners: List[Tuple[List[Dict], str, str, Dict]],
    *,
    batch_size: int,
    max_retries: int,
) -> int:
    candidates: List[Dict] = []
    relation_ids: Dict[int, str] = {}
    for description_groups, owner_key, node_text, context in owners:
        _collect_from_description_items(
            description_groups,
            owner_key=owner_key,
            node_text=node_text,
            prompt_context=context,
            candidates=candidates,
            relation_ids=relation_ids,
        )

    if not candidates:
        logger.info(
            "RelationPredicateNormalizer | candidates=0 | owners_changed=0 | "
            "llm_calls=0 | input_tokens=0 | output_tokens=0 | total_tokens=0"
        )
        return 0

    usage_before = _usage_snapshot(llm_client)

    plan_map = _expand_requests_recursively(
        llm_client,
        candidates,
        batch_size=batch_size,
        max_retries=max_retries,
    )

    usage = _usage_since(llm_client, usage_before)

    changed = 0
    for description_groups, _owner_key, _node_text, _context in owners:
        before = copy.deepcopy(description_groups)
        rewritten = _rewrite_description_items(
            description_groups,
            relation_ids=relation_ids,
            plan_map=plan_map,
        )
        description_groups[:] = rewritten
        if before != rewritten:
            changed += 1

    logger.info(
        "RelationPredicateNormalizer | candidates=%d | owners_changed=%d | "
        "llm_calls=%s | input_tokens=%s | output_tokens=%s | total_tokens=%s",
        len(candidates),
        changed,
        usage.get("api_calls", "?"),
        usage.get("input_tokens", "?"),
        usage.get("output_tokens", "?"),
        usage.get("total_tokens", "?"),
    )
    return changed


def normalize_entity_relation_predicates_batch(
    llm_client: Any,
    rules: List[Dict],
    *,
    graph_field: str,
    batch_size: int = 20,
    max_retries: int = 2,
) -> List[Dict]:
    owners: List[Tuple[List[Dict], str, str, Dict]] = []
    for rule_index, scope, element in _iter_rule_elements(rules):
        graph = element.get(graph_field)
        if not isinstance(graph, dict):
            continue
        nodes = graph.get("Nodes")
        if not isinstance(nodes, dict):
            continue
        context = {
            "Sentence": element.get("Sentence", ""),
            "Subject": element.get("Subject", ""),
            "Predicate": element.get("Predicate", ""),
            "Object": element.get("Object", ""),
            "Condition": element.get("Condition", []),
        }
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            groups = node.get("DescriptionGroups")
            if not isinstance(groups, list):
                continue
            owners.append(
                (
                    groups,
                    f"rule{rule_index}|{scope}|{graph_field}|{node_id}",
                    node.get("Text", ""),
                    context,
                )
            )

    _normalize_description_owners_batch(
        llm_client,
        owners,
        batch_size=batch_size,
        max_retries=max_retries,
    )
    return rules


def normalize_condition_relation_predicates_batch(
    llm_client: Any,
    rules: List[Dict],
    *,
    batch_size: int = 20,
    max_retries: int = 2,
) -> List[Dict]:
    owners: List[Tuple[List[Dict], str, str, Dict]] = []
    for rule_index, scope, element in _iter_rule_elements(rules):
        graph = element.get("Condition_Elements")
        if not isinstance(graph, dict):
            continue
        nodes = graph.get("Nodes")
        if not isinstance(nodes, dict):
            continue
        context = {
            "Sentence": element.get("Sentence", ""),
            "Subject": element.get("Subject", ""),
            "Predicate": element.get("Predicate", ""),
            "Object": element.get("Object", ""),
            "Condition": element.get("Condition", []),
        }
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            body = node.get("Body")
            if not isinstance(body, dict):
                continue
            entity = body.get("Entity")
            if not isinstance(entity, dict):
                continue
            groups = entity.get("DescriptionGroups")
            if not isinstance(groups, list):
                continue
            owners.append(
                (
                    groups,
                    f"rule{rule_index}|{scope}|Condition_Elements|{node_id}",
                    node.get("Text", ""),
                    context,
                )
            )

    _normalize_description_owners_batch(
        llm_client,
        owners,
        batch_size=batch_size,
        max_retries=max_retries,
    )
    return rules


__all__ = [
    "normalize_entity_relation_predicates_batch",
    "normalize_condition_relation_predicates_batch",
]
