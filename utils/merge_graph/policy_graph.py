import copy
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from llms import LLMClient
from utils.utils import parse_json_response

logger = logging.getLogger(__name__)

OPERATORS = {"AND", "OR"}

FRAME_SYSTEM_PROMPT = """
You bind already-built policy Subject, Predicate, Object, and Condition graphs
into a minimal FrameGraph.

Do not rewrite, paraphrase, add, delete, or merge local semantic nodes.
Use only existing node ids from the supplied local graphs.

A Behavior is one legal cross-role binding of Subject / Predicate / Object /
Condition. Preserve local graph coordination whenever all cross-role bindings
are shared. Expand a local coordination into multiple Behaviors only when this
is necessary to preserve Predicate-Object pairing or Condition scope.

Rules:
- Subject normally stays at the Subject root unless the sentence clearly binds
  different subject children to different behaviors.
- Predicate may stay at its root when its coordinated actions share the same
  Object and Conditions.
- Object may stay at its root when its internal alternatives all belong to the
  same Predicate binding. An individual Behavior may use Object=null when the
  clause Object belongs only to a different coordinated Predicate.
- Decide Object sharing from the original Sentence, not merely from the fact that
  there is one clause-level ObjectGraph. A trailing Object after the last
  coordinated predicate is shared with earlier predicates only when the source
  syntax supports that shared scope. If earlier predicates are complete without
  that Object, keep Object=null for those Behaviors.
- Conditions may reference only ConditionGraph roots, never internal c* nodes.
- Every source Condition root must appear in at least one Behavior.
- Each Behavior's Conditions list is its complete effective condition set.
- Do not infer unstated relations.
- Prefer the smallest number of Behaviors that preserves the source meaning.

Return JSON only.
""".strip()


def _empty_frame_graph() -> Dict:
    return {"Root": None, "Nodes": {}}


def _ref(node_id: Optional[str]) -> Optional[Dict[str, str]]:
    if node_id is None:
        return None
    return {"Ref": node_id}


def _walk_refs(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if set(value) == {"Ref"} and isinstance(value.get("Ref"), str):
            yield value["Ref"]
            return
        for child in value.values():
            yield from _walk_refs(child)
        return

    if isinstance(value, list):
        for child in value:
            yield from _walk_refs(child)


def _graph_root(graph: Any) -> Optional[str]:
    if not isinstance(graph, dict):
        return None
    root = graph.get("Root")
    return root if isinstance(root, str) and root else None


def _role_root(element: Dict, text_field: str, graph_field: str) -> Optional[str]:
    """Return a role root only when the corresponding surface role is non-empty.

    Upstream graph stages intentionally retain an empty singleton graph for an
    absent Object.  That placeholder is not a semantic participant and must not
    be bound into a FrameGraph.
    """
    text = element.get(text_field)
    if not isinstance(text, str) or not text.strip():
        return None
    return _graph_root(element.get(graph_field))


def _graph_nodes(graph: Any) -> Dict[str, Dict]:
    if not isinstance(graph, dict):
        return {}
    nodes = graph.get("Nodes")
    return nodes if isinstance(nodes, dict) else {}


def _condition_roots(graph: Any) -> List[str]:
    if not isinstance(graph, dict):
        return []
    roots = graph.get("Roots")
    if not isinstance(roots, list):
        return []

    out = []
    for item in roots:
        if isinstance(item, dict) and set(item) == {"Ref"}:
            ref = item.get("Ref")
            if isinstance(ref, str) and ref:
                out.append(ref)
    return out


def _coordination(graph: Any, node_id: Optional[str]) -> Optional[Dict]:
    if not node_id:
        return None
    node = _graph_nodes(graph).get(node_id)
    if not isinstance(node, dict):
        return None
    coordination = node.get("Coordination")
    return coordination if isinstance(coordination, dict) else None


def _coordination_item_refs(graph: Any, node_id: Optional[str]) -> List[str]:
    coordination = _coordination(graph, node_id)
    if not coordination:
        return []
    items = coordination.get("Items")
    if not isinstance(items, list):
        return []

    refs = []
    for item in items:
        if isinstance(item, dict) and set(item) == {"Ref"}:
            ref = item.get("Ref")
            if isinstance(ref, str):
                refs.append(ref)
    return refs


def _validate_single_graph(
    graph: Any,
    *,
    prefix: str,
    allow_empty: bool,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(graph, dict):
        return [f"{prefix}:graph_missing"]

    nodes = graph.get("Nodes")
    if not isinstance(nodes, dict):
        return [f"{prefix}:nodes_invalid"]

    root = graph.get("Root")
    if root is None:
        if allow_empty and not nodes:
            return []
        errors.append(f"{prefix}:root_missing")
    elif not isinstance(root, str) or not root.startswith(prefix):
        errors.append(f"{prefix}:root_invalid")
    elif root not in nodes:
        errors.append(f"{prefix}:root_unknown")

    for node_id, node in nodes.items():
        if not isinstance(node_id, str) or not node_id.startswith(prefix):
            errors.append(f"{prefix}:node_id_invalid({node_id!r})")
            continue
        if not isinstance(node, dict):
            errors.append(f"{prefix}:node_invalid({node_id})")
            continue
        for ref in _walk_refs(node):
            if not ref.startswith(prefix):
                errors.append(f"{prefix}:cross_graph_ref({node_id}->{ref})")
            elif ref not in nodes:
                errors.append(f"{prefix}:dangling_ref({node_id}->{ref})")

    return errors


def _validate_condition_graph(graph: Any, allow_empty: bool) -> List[str]:
    errors: List[str] = []
    if not isinstance(graph, dict):
        return ["c:graph_missing"]

    nodes = graph.get("Nodes")
    roots = graph.get("Roots")
    if not isinstance(nodes, dict):
        errors.append("c:nodes_invalid")
        return errors
    if not isinstance(roots, list):
        errors.append("c:roots_invalid")
        return errors

    root_refs = _condition_roots(graph)
    if not root_refs and nodes and not allow_empty:
        errors.append("c:roots_missing")

    if allow_empty and not roots and not nodes:
        return []

    if len(root_refs) != len(roots):
        errors.append("c:root_ref_invalid")

    for ref in root_refs:
        if not ref.startswith("c"):
            errors.append(f"c:root_prefix_invalid({ref})")
        elif ref not in nodes:
            errors.append(f"c:root_unknown({ref})")

    for node_id, node in nodes.items():
        if not isinstance(node_id, str) or not node_id.startswith("c"):
            errors.append(f"c:node_id_invalid({node_id!r})")
            continue
        if not isinstance(node, dict):
            errors.append(f"c:node_invalid({node_id})")
            continue
        for ref in _walk_refs(node):
            if not ref.startswith("c"):
                errors.append(f"c:cross_graph_ref({node_id}->{ref})")
            elif ref not in nodes:
                errors.append(f"c:dangling_ref({node_id}->{ref})")

    return errors


def validate_clause_graphs(element: Dict) -> List[str]:
    """Validate local graph consistency. Empty roles are legal when source role is empty."""
    if not isinstance(element, dict):
        return ["clause:not_dict"]

    errors: List[str] = []

    role_specs = (
        ("Subject", "Subject_Elements", "s"),
        ("Predicate", "Predicate_Elements", "p"),
        ("Object", "Object_Elements", "o"),
    )

    for text_field, graph_field, prefix in role_specs:
        text = element.get(text_field, "")
        text_nonempty = isinstance(text, str) and bool(text.strip())
        graph = element.get(graph_field)
        errors.extend(
            _validate_single_graph(
                graph,
                prefix=prefix,
                allow_empty=not text_nonempty,
            )
        )

    conditions = element.get("Condition", [])
    condition_nonempty = isinstance(conditions, list) and bool(conditions)
    errors.extend(
        _validate_condition_graph(
            element.get("Condition_Elements"),
            allow_empty=not condition_nonempty,
        )
    )

    return errors


def _clause_refs(element: Dict) -> Dict[str, Any]:
    condition_graph = element.get("Condition_Elements")

    return {
        "Subject": _role_root(element, "Subject", "Subject_Elements"),
        "Predicate": _role_root(element, "Predicate", "Predicate_Elements"),
        "Object": _role_root(element, "Object", "Object_Elements"),
        "Conditions": _condition_roots(condition_graph),
    }


def _needs_llm_binding(element: Dict) -> bool:
    """LLM is only needed when a coordinated predicate may require cross-role binding."""
    predicate_graph = element.get("Predicate_Elements")
    predicate_root = _role_root(element, "Predicate", "Predicate_Elements")
    predicate_coordination = _coordination(predicate_graph, predicate_root)
    if not predicate_coordination:
        return False

    condition_roots = _condition_roots(element.get("Condition_Elements"))
    object_graph = element.get("Object_Elements")
    object_root = _role_root(element, "Object", "Object_Elements")
    object_coordination = _coordination(object_graph, object_root)

    return bool(condition_roots or object_coordination)


def _deterministic_plan(element: Dict) -> Dict:
    refs = _clause_refs(element)
    if refs["Predicate"] is None:
        return {"Operator": None, "Behaviors": []}

    return {
        "Operator": None,
        "Behaviors": [
            {
                "Subject": refs["Subject"],
                "Predicate": refs["Predicate"],
                "Object": refs["Object"],
                "Conditions": refs["Conditions"],
            }
        ],
    }


def _frame_graph_from_plan(plan: Dict) -> Dict:
    behaviors = plan["Behaviors"]
    if not behaviors:
        return _empty_frame_graph()

    nodes: Dict[str, Dict] = {}

    if len(behaviors) == 1:
        behavior = behaviors[0]
        nodes["f0"] = {
            "Behavior": {
                "Subject": _ref(behavior["Subject"]),
                "Predicate": _ref(behavior["Predicate"]),
                "Object": _ref(behavior["Object"]),
                "Conditions": [_ref(ref) for ref in behavior["Conditions"]],
            }
        }
        return {"Root": "f0", "Nodes": nodes}

    item_refs = []
    for index, behavior in enumerate(behaviors, start=1):
        node_id = f"f{index}"
        item_refs.append({"Ref": node_id})
        nodes[node_id] = {
            "Behavior": {
                "Subject": _ref(behavior["Subject"]),
                "Predicate": _ref(behavior["Predicate"]),
                "Object": _ref(behavior["Object"]),
                "Conditions": [_ref(ref) for ref in behavior["Conditions"]],
            }
        }

    nodes["f0"] = {
        "Coordination": {
            "Operator": plan["Operator"],
            "Items": item_refs,
        }
    }
    return {"Root": "f0", "Nodes": nodes}


def _plan_reason(element: Dict, plan: Any) -> str:
    if not isinstance(plan, dict):
        return "plan:not_dict"

    if set(plan) != {"Operator", "Behaviors"}:
        return "plan:keys"

    behaviors = plan.get("Behaviors")
    if not isinstance(behaviors, list) or not behaviors:
        return "plan:behaviors"

    operator = plan.get("Operator")
    if len(behaviors) == 1:
        if operator is not None:
            return "plan:single_behavior_operator"
    elif operator not in OPERATORS:
        return "plan:operator"

    subject_graph = element.get("Subject_Elements")
    predicate_graph = element.get("Predicate_Elements")
    object_graph = element.get("Object_Elements")
    condition_graph = element.get("Condition_Elements")

    subject_nodes = set(_graph_nodes(subject_graph))
    predicate_nodes = set(_graph_nodes(predicate_graph))
    object_nodes = set(_graph_nodes(object_graph))
    condition_nodes = set(_graph_nodes(condition_graph))
    condition_roots = set(_condition_roots(condition_graph))

    subject_root = _role_root(element, "Subject", "Subject_Elements")
    predicate_root = _role_root(element, "Predicate", "Predicate_Elements")
    object_root = _role_root(element, "Object", "Object_Elements")

    predicate_coord = _coordination(predicate_graph, predicate_root)
    predicate_children = set(_coordination_item_refs(predicate_graph, predicate_root))
    predicate_operator = (
        predicate_coord.get("Operator") if isinstance(predicate_coord, dict) else None
    )

    if len(behaviors) > 1 and predicate_operator in OPERATORS:
        if operator != predicate_operator:
            return "plan:operator_mismatch"

    seen_behaviors: Set[Tuple] = set()
    used_conditions: Set[str] = set()
    used_predicates: Set[str] = set()
    used_objects: Set[str] = set()

    for index, behavior in enumerate(behaviors):
        if not isinstance(behavior, dict):
            return f"behavior[{index}]:not_dict"
        if set(behavior) != {"Subject", "Predicate", "Object", "Conditions"}:
            return f"behavior[{index}]:keys"

        subject = behavior.get("Subject")
        predicate = behavior.get("Predicate")
        obj = behavior.get("Object")
        conditions = behavior.get("Conditions")

        if subject_root is None:
            if subject is not None:
                return f"behavior[{index}]:subject_should_be_null"
        elif not isinstance(subject, str) or subject not in subject_nodes:
            return f"behavior[{index}]:subject_ref"

        if predicate_root is None:
            return f"behavior[{index}]:predicate_missing"
        if not isinstance(predicate, str) or predicate not in predicate_nodes:
            return f"behavior[{index}]:predicate_ref"

        if object_root is None:
            if obj is not None:
                return f"behavior[{index}]:object_should_be_null"
        elif obj is not None:
            if not isinstance(obj, str) or obj not in object_nodes:
                return f"behavior[{index}]:object_ref"
            used_objects.add(obj)

        if not isinstance(conditions, list):
            return f"behavior[{index}]:conditions"
        if len(conditions) != len(set(conditions)):
            return f"behavior[{index}]:duplicate_condition"
        for condition in conditions:
            if not isinstance(condition, str) or condition not in condition_nodes:
                return f"behavior[{index}]:condition_ref"
            if condition not in condition_roots:
                return f"behavior[{index}]:condition_not_root"
            used_conditions.add(condition)

        used_predicates.add(predicate)
        signature = (subject, predicate, obj, tuple(conditions))
        if signature in seen_behaviors:
            return f"behavior[{index}]:duplicate"
        seen_behaviors.add(signature)

    if used_conditions != condition_roots:
        return "plan:condition_coverage"

    # A clause may contain an ObjectGraph whose object belongs only to one of
    # several coordinated predicates. Individual Behaviors may therefore have
    # Object=null, but a non-empty top-level Object must still be represented by
    # at least one Behavior.
    if object_root is not None and not used_objects:
        return "plan:object_coverage"

    if predicate_children:
        if len(behaviors) == 1:
            if behaviors[0]["Predicate"] != predicate_root:
                return "plan:predicate_coordination_incomplete"
        else:
            if predicate_root in used_predicates:
                return "plan:mixed_predicate_root_and_children"
            if used_predicates != predicate_children:
                return "plan:predicate_child_coverage"

    return ""


def _normalize_llm_record(record: Any) -> Optional[Tuple[str, Dict]]:
    if not isinstance(record, dict):
        return None

    task_id = record.get("id")
    behaviors = record.get("Behaviors")
    operator = record.get("Operator")
    if not isinstance(task_id, str):
        return None

    if not isinstance(behaviors, list):
        return None

    normalized_behaviors = []
    for behavior in behaviors:
        if not isinstance(behavior, dict):
            return None
        normalized_behaviors.append(
            {
                "Subject": behavior.get("Subject"),
                "Predicate": behavior.get("Predicate"),
                "Object": behavior.get("Object"),
                "Conditions": behavior.get("Conditions", []),
            }
        )

    return task_id, {
        "Operator": operator,
        "Behaviors": normalized_behaviors,
    }


def _parse_llm_response(response: str) -> Dict[str, Dict]:
    parsed = parse_json_response(response)

    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        records = parsed["results"]
    elif isinstance(parsed, dict) and "id" in parsed:
        records = [parsed]
    else:
        raise ValueError("Frame response must be a list of results.")

    out: Dict[str, Dict] = {}
    for record in records:
        normalized = _normalize_llm_record(record)
        if normalized is None:
            continue
        task_id, plan = normalized
        out[task_id] = plan
    return out


def _prompt_graph(graph: Any) -> Dict:
    if not isinstance(graph, dict):
        return {}
    return copy.deepcopy(graph)


def _build_frame_prompt(tasks: List[Dict], organization: str) -> str:
    records = []
    for task in tasks:
        element = task["element"]
        records.append(
            {
                "id": task["id"],
                "Sentence": element.get("Sentence", ""),
                "Subject": element.get("Subject", ""),
                "Predicate": element.get("Predicate", ""),
                "Object": element.get("Object", ""),
                "Condition": element.get("Condition", []),
                "SubjectGraph": _prompt_graph(element.get("Subject_Elements")),
                "PredicateGraph": _prompt_graph(element.get("Predicate_Elements")),
                "ObjectGraph": _prompt_graph(element.get("Object_Elements")),
                "ConditionGraph": _prompt_graph(element.get("Condition_Elements")),
            }
        )

    payload = json.dumps(records, ensure_ascii=False, indent=2)
    return f"""
Policy issuer: {organization}

Bind each clause below using only existing node ids.

Input:
{payload}

Return one object per id:
[
  {{
    "id": "...",
    "Operator": null | "AND" | "OR",
    "Behaviors": [
      {{
        "Subject": "s*" | null,
        "Predicate": "p*",
        "Object": "o*" | null,
        "Conditions": ["c*", ...]
      }}
    ]
  }}
]

Canonical rules:
1. If one Behavior can losslessly reference coordinated local roots, use one
   Behavior and Operator=null.
2. Expand into several Behaviors only for cross-role pairing or different
   Condition scope.
3. If Predicate root is expanded, cover every immediate Predicate child exactly
   through the Behavior set; do not mix the Predicate root with its children.
4. If there are multiple Behaviors and the Predicate root has AND/OR
   Coordination, use that same Operator.
5. Object may be null for an individual Behavior even when the clause has an
   ObjectGraph, when the source object belongs only to another coordinated
   predicate. Do not broadcast a clause-level Object to sibling predicates merely
   because they share one ObjectGraph. Determine shared Object scope from the
   original Sentence. In particular, when earlier coordinated predicates are
   grammatically complete on their own and the Object is syntactically introduced
   only with a later predicate, use Object=null for those earlier Behaviors. Every
   non-empty clause Object must still be represented by at least one Behavior.
6. Every ConditionGraph root must be assigned to at least one Behavior.
7. Conditions list contains the complete conditions effective for that Behavior.
8. Do not use internal non-root Condition nodes as frame Conditions.
""".strip()


def _collect_clause_targets(rules: List[Dict]) -> List[Dict]:
    targets = []
    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue

        main = rule.get("Main_Clause_Elements")
        if isinstance(main, dict):
            targets.append(
                {
                    "id": f"r{rule_index}_main",
                    "rule_index": rule_index,
                    "scope": "main",
                    "element": main,
                }
            )

        examples = rule.get("Example_Sentences_Elements", [])
        if isinstance(examples, list):
            for example_index, element in enumerate(examples):
                if not isinstance(element, dict):
                    continue
                targets.append(
                    {
                        "id": f"r{rule_index}_example{example_index}",
                        "rule_index": rule_index,
                        "scope": f"example:{example_index}",
                        "element": element,
                    }
                )
    return targets


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


def _bind_ambiguous_targets(
    llm_client: LLMClient,
    targets: List[Dict],
    organization: str,
    *,
    batch_size: int,
    max_retries: int,
) -> Tuple[int, int]:
    resolved = 0
    unresolved = 0

    for start in range(0, len(targets), batch_size):
        remaining = targets[start : start + batch_size]

        for attempt in range(1, max_retries + 1):
            if not remaining:
                break

            try:
                response = llm_client.invoke(
                    _build_frame_prompt(remaining, organization),
                    system_prompt=FRAME_SYSTEM_PROMPT,
                )
                if response is None or (
                    isinstance(response, str) and not response.strip()
                ):
                    raise ValueError("Empty FrameGraph response.")

                parsed = _parse_llm_response(response)
                failed = []

                for task in remaining:
                    plan = parsed.get(task["id"])
                    reason = _plan_reason(task["element"], plan)
                    if reason:
                        logger.warning(
                            "PolicyFrame | invalid plan | id=%s | attempt=%d | %s",
                            task["id"],
                            attempt,
                            reason,
                        )
                        failed.append(task)
                        continue

                    task["element"]["Frame_Elements"] = _frame_graph_from_plan(plan)
                    resolved += 1

                remaining = failed

            except Exception as exc:
                logger.warning(
                    "PolicyFrame | batch retry | attempt=%d/%d | error=%s",
                    attempt,
                    max_retries,
                    exc,
                )

        for task in remaining:
            task["element"]["Frame_Elements"] = _empty_frame_graph()
            unresolved += 1
            logger.error(
                "PolicyFrame | unresolved binding | id=%s | sentence=%r",
                task["id"],
                task["element"].get("Sentence", ""),
            )

    return resolved, unresolved


def _drop_redundant_surface_fields(rule: Dict) -> None:
    """Drop redundant surface fields after graph assembly.

    ``Original_Policy`` remains the complete source provenance. Main clauses,
    example sentences, and ordinary examples are represented by their structured
    ``*_Elements`` fields, so the raw split fields are no longer needed here.
    """
    rule.pop("Main_Clause", None)
    rule.pop("Example_Sentences", None)
    rule.pop("Examples", None)


def compose_policy_graph(rule: Dict) -> Dict:
    """Create a lightweight routing manifest; local graphs stay in their clauses."""
    main = rule.get("Main_Clause_Elements")
    main_root = None
    if isinstance(main, dict):
        frame_graph = main.get("Frame_Elements")
        if isinstance(frame_graph, dict):
            main_root = frame_graph.get("Root")

    example_frames = []
    examples = rule.get("Example_Sentences_Elements", [])
    if isinstance(examples, list):
        for index, element in enumerate(examples):
            if not isinstance(element, dict):
                continue
            frame_graph = element.get("Frame_Elements")
            root = frame_graph.get("Root") if isinstance(frame_graph, dict) else None
            if isinstance(root, str):
                example_frames.append(
                    {
                        "Scope": f"example:{index}",
                        "Ref": root,
                    }
                )

    return {
        "Root": (
            {"Scope": "main", "Ref": main_root} if isinstance(main_root, str) else None
        ),
        "ExampleFrames": example_frames,
    }


def build_policy_graph_batch(
    llm_client: LLMClient,
    rules: List[Dict],
    organization: str,
    batch_size: int = 10,
    max_retries: int = 2,
) -> List[Dict]:
    """
    Build Frame_Elements for Main_Clause_Elements and every Example_Sentences_Elements.

    Simple clauses are bound deterministically. Only coordinated clauses with
    potential Predicate/Object pairing or Condition-scope ambiguity are sent to LLM.
    The function mutates and returns `rules`, matching the existing splitter style.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")

    targets = _collect_clause_targets(rules)
    ambiguous: List[Dict] = []
    invalid = 0
    deterministic = 0

    usage_before = _usage_snapshot(llm_client)

    for task in targets:
        element = task["element"]
        errors = validate_clause_graphs(element)
        if errors:
            element["Frame_Elements"] = _empty_frame_graph()
            invalid += 1
            logger.error(
                "PolicyFrame | invalid local graphs | id=%s | errors=%s",
                task["id"],
                "; ".join(errors),
            )
            continue

        if _needs_llm_binding(element):
            ambiguous.append(task)
            continue

        plan = _deterministic_plan(element)
        element["Frame_Elements"] = _frame_graph_from_plan(plan)
        deterministic += 1

    llm_resolved, unresolved = _bind_ambiguous_targets(
        llm_client,
        ambiguous,
        organization,
        batch_size=batch_size,
        max_retries=max_retries,
    )

    if invalid or unresolved:
        raise ValueError(
            "PolicyFrame generation is incomplete; refusing to save lossy graphs: "
            f"invalid={invalid}, unresolved={unresolved}"
        )

    for rule in rules:
        if isinstance(rule, dict):
            rule["Policy_Graph"] = compose_policy_graph(rule)
            _drop_redundant_surface_fields(rule)

    usage = _usage_since(llm_client, usage_before)
    logger.info(
        "PolicyFrame | completed | clauses=%d | deterministic=%d | "
        "llm_candidates=%d | llm_resolved=%d | unresolved=%d | invalid=%d | "
        "api_calls=%s | input_tokens=%s | output_tokens=%s | total_tokens=%s",
        len(targets),
        deterministic,
        len(ambiguous),
        llm_resolved,
        unresolved,
        invalid,
        usage.get("api_calls", "?"),
        usage.get("input_tokens", "?"),
        usage.get("output_tokens", "?"),
        usage.get("total_tokens", "?"),
    )

    return rules
