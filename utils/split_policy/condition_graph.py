import copy
import json
import logging
import re
from typing import Any, Dict, List, Set, Tuple

from llms import LLMClient
from utils.utils import parse_json_response
from utils.split_policy.relation_predicate_normalizer import (
    normalize_condition_relation_predicates_batch,
)
from utils.split_policy.layered_graph import (
    ENTITY_BODY_KEYS,
    GRAPH_KEYS,
    NODE_KEYS,
    OPERATORS,
    REF_KEYS,
    GraphState,
    build_context_text,
    entity_body_edges,
    entity_body_graph_reason,
    entity_plan_reason,
    event_frame_edges,
    event_frame_graph_reason,
    event_frame_plan_reason,
    expand_layered_graphs,
    install_entity_plan,
    install_event_frame,
    intern_child_node,
    materialize_event_frame,
    normalize_space,
    phrase_grounded,
    simple_token,
    stub_text_reason,
)

logger = logging.getLogger(__name__)

CONDITION_NODE_KEYS = {"Text", "Marker", "Body"}
CONDITION_BODY_TYPES = {"Atom", "Entity", "Event", "Coordination"}
CONDITION_GRAPH_KEYS = {"Roots", "Nodes"}
CONDITION_ROOT_KEYS = {"Ref"}
CONDITION_COVERAGE_IGNORED = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "being",
    "by",
    "either",
    "for",
    "i",
    "ii",
    "in",
    "is",
    "nor",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "to",
    "was",
    "who",
    "whom",
    "which",
}
POSSESSIVE_SURFACE_TOKENS = {
    "her",
    "his",
    "its",
    "my",
    "our",
    "their",
    "thereof",
    "whose",
    "your",
}


def _condition_content_tokens(value: str) -> List[str]:
    return [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", value or "")
        if token.casefold() not in CONDITION_COVERAGE_IGNORED
    ]


def _condition_node_coverage_reason(
    node: Dict,
    nodes: Dict[str, Dict],
    path: str,
) -> str:
    """Require every substantive node token to survive in its semantic body."""
    represented: List[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            represented.append(value)
            return
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if not isinstance(value, dict):
            return
        if set(value) == {"Ref"}:
            target = nodes.get(value.get("Ref"), {})
            text = target.get("Text") if isinstance(target, dict) else None
            if isinstance(text, str):
                represented.append(text)
            return
        for key, item in value.items():
            if key not in {"Self"}:
                collect(item)

    collect(node.get("Marker"))
    collect(node.get("Body"))
    represented_tokens = set(_condition_content_tokens(" ".join(represented)))
    missing = [
        token
        for token in _condition_content_tokens(node.get("Text", ""))
        if token not in represented_tokens
    ]
    body = node.get("Body", {})
    entity = body.get("Entity") if isinstance(body, dict) else None
    reference = entity.get("Reference") if isinstance(entity, dict) else None
    if isinstance(reference, dict):
        missing = [
            token
            for token in missing
            if token not in POSSESSIVE_SURFACE_TOKENS
            and not token.endswith(("'s", "’s"))
        ]
    if missing:
        return f"{path}:unrepresented_content({missing})"
    return ""


SYSTEM_PROMPT = """
You decompose policy Condition spans into recursive constraint graphs.

Each Condition node has exactly three fields: Text, Marker, and Body.
Body is exactly one of four semantic forms:
- Atom: an indivisible condition expression,
- Entity: a nominal/entity structure,
- Event: an event/state with semantic arguments,
- Coordination: AND/OR between complete condition expressions.

Reuse entity semantics for nominal material and action semantics for events.
Analyze only one semantic layer of CurrentText. Child spans must be returned only
as {"Text": "..."} stubs and are expanded by Python later.

Marker is only the wording that connects the whole CurrentText condition to its
outer constrained event. Internal prepositions/relations belong inside Entity or
Event, not in Marker.

Use Sentence, Subject, Predicate, and Object only for disambiguation. Do not import
their participants or wording into the Condition graph. Cross-field references are
deferred to the final merged policy graph.

Preserve source wording and explicit logical scope. Do not infer unstated facts,
paraphrase semantic content, or generate policy logic. Return JSON only.
""".strip()


def _collect_elements(rules: List[Dict], phase: str) -> List[Dict]:
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


def _condition_texts(element: Dict) -> List[str]:
    raw = element.get("Condition", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def _build_condition_prompt(tasks: List[Dict], _organization: str) -> str:
    input_text = json.dumps(
        [
            {
                "id": task["id"],
                "Sentence": task["PromptContext"].get("Sentence", ""),
                "Subject": task["PromptContext"].get("Subject", ""),
                "Predicate": task["PromptContext"].get("Predicate", ""),
                "Object": task["PromptContext"].get("Object", ""),
                "RootCondition": task["RootText"],
                "CurrentText": task["CurrentText"],
                "ParentHint": task["ParentHint"],
            }
            for task in tasks
        ],
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Input:
{input_text}

Return one result per id. Analyze only ONE semantic layer of CurrentText.

Current Condition node plan:
{{
  "Text": "CurrentText",
  "Marker": "source wording" | null,
  "Body":
      {{"Atom": "source wording"}}

    | {{"Entity": {{
        "Head": "source wording" | null,
        "Coordination": null | {{
          "Operator": "AND" | "OR",
          "Items": [{{"Text": "child entity span"}}, ...]
        }},
        "DescriptionGroups": [description_item, ...],
        "Reference": null | {{
          "Type": "Coreference" | "Possessor",
          "Text": "antecedent/possessor source wording"
        }}
      }}}}

    | {{"Event": {{
        "Action": "source wording",
        "Modifiers": ["source wording"],
        "Operators": ["source wording"],
        "Arguments": [
          {{
            "Role": "Subject" | "Object" | "Complement",
            "Marker": "source wording" | null,
            "Value": {{"Text": "child participant span"}}
          }}
        ]
      }}}}

    | {{"Coordination": {{
        "Operator": "AND" | "OR",
        "Items": [{{"Text": "child condition span"}}, ...]
      }}}}
}}

description_item is exactly one of:
- {{"Modifier": "source wording"}}
- {{"Relation": {{
    "Predicate": "source wording",
    "Target": {{"Text": "child entity span"}}
  }}}}
- {{"Relation": {{
    "Source": {{"Text": "child entity span"}},
    "Predicate": "source wording"
  }}}}
- {{"Event": {{
    "Action": "source wording",
    "Modifiers": ["source wording"],
    "Operators": ["source wording"],
    "Arguments": [
      {{
        "Role": "Subject" | "Object" | "Complement",
        "Marker": "source wording" | null,
        "Value": {{"Text": "child entity span"}} | {{"Self": true}}
      }}
    ]
  }}}}
- {{"Operator": "AND" | "OR", "Items": [description_item, ...]}}

Rules:

1. Body is EXACTLY one of Atom, Entity, Event, or Coordination.
   - Atom: simple non-nominal/non-event condition wording such as an adverb/state.
   - Entity: CurrentText is centered on a nominal participant or nominal phrase.
   - Event: CurrentText is centered on an action/state with its own participants.
   - Coordination: CurrentText coordinates complete condition expressions.

2. Marker belongs only to the OUTER condition-to-main-event connection of CurrentText.
   Typical Marker wording includes forms such as "if", "unless", "without", "by",
   "for", "in", "towards", "when", "after", "before", or infinitival "to" when it
   introduces the whole condition. Do not put internal entity Relations or Event
   argument markers in the node Marker.

   For internal child participants created by Entity/Relation/Event arguments,
   normally use Marker=null. A Condition Coordination child may have its own Marker.

3. Atom preserves the substantive local condition wording and is used only when
   Entity/Event analysis would add no meaningful structure.
   If removing the outer Marker leaves a nominal phrase with its own lexical head,
   use Entity, not Atom. In particular, a multiword phrase after a Marker should not
   be collapsed into Atom when it can be analyzed as an Entity.

   Example:
   "by any means"
   -> Marker="by"
   -> Body.Entity(Head="means", Modifier="any")

4. Entity uses the SAME semantics as Subject/Object entities:
   - Head is the minimal lexical nominal head.
   - Coordination is AND/OR between distinct independently referable participants.
   - For a flat participant enumeration with no clear operator, default to OR.
   - DescriptionGroups is the only description container; its top-level list is
     cumulative AND.
   - Use Modifier for same-referent descriptions.
   - Use Relation for an explicit link to another participant.
   - Explicit "A of B" normally becomes A --"of"--> B.
   - Keep complete Relation endpoints and source-faithful direction.
   - Relative introducers "that", "which", and "who" are not relation Predicate wording.
   - Reference is only Entity-internal Coreference/Possessor grounded in RootCondition.
     Cross-field references stay null.

5. Prefer a simple Entity Relation for an embedded relative-clause event whenever the
   current Entity can be represented losslessly as the implicit Source or Target.
   This preference is mandatory when the relative clause has one complete non-self
   participant. Use an Entity description Event only when one simple Relation cannot
   preserve the full participant structure. In such an Event, {{"Self": true}} means
   the current Entity.

   Keep a Relation endpoint semantically complete. Do not pull a prepositional phrase
   out of a nominal endpoint when that phrase completes the noun phrase itself.

   Examples:
   "a manner that imposes an unreasonable load on X"
   -> Entity Head="manner"
   -> Relation Predicate="imposes", Target="an unreasonable load on X"

   "A in which B R C"
   may use Event(Action=R, Subject=B, Object=C,
   Complement(marker="in", Self=true)) when the simple Relation forms are insufficient.

6. Event reuses EXACTLY the same lexical-action boundary principles as PredicateGraph:
   - Action is the minimal semantic action/state wording.
   - Preserve particles/prepositions required by the lexical action or its argument
     structure, such as "engage in", "tamper with", and "intrude into". Do not move
     such a required lexical preposition into an Argument Marker.
   - Modifiers describe how the action is performed.
   - Operators scope over the Action, such as "attempt to", "fail to", or "continue to".
   - Identify complete argument spans BEFORE splitting their internal Entity structure.
   - Arguments represent explicit event participants. Use Subject, Object, or Complement.
   - Argument Marker belongs to that argument relation, not the outer Condition Marker.
   - Child argument Value is only {{"Text": "..."}}; Python expands it later.
   - If several coordinated verbs share the same participant structure and keeping the
     coordinated wording in one Action loses no participant/scope information, it may
     remain one source-grounded Action. Use Condition Coordination for distinct complete
     condition events.

   Example:
   "to engage in illegal activities"
   -> Marker="to", Event.Action="engage in", Object="illegal activities"

7. Condition Coordination is only for AND/OR between COMPLETE condition expressions.
   Entity-level alternatives stay inside Entity.Coordination. Preserve explicit or
   semantically clear AND/OR; if a flat complete-condition enumeration has no clear
   operator, default to OR. Child Items contain only {{"Text": "..."}}.

8. Preserve scope. Shared material belongs on a parent only when clearly shared;
   otherwise keep the narrowest explicitly supported attachment. Do not flatten a
   subgroup when flattening would lose a local Modifier, Relation, Event, Marker, or
   logical operator.

9. Analyze only RootCondition. Sentence, Subject, Predicate, and Object are context for
   disambiguation only. Do not import their wording or participants. ParentHint only
   disambiguates the role already created by Python; do not reproduce that parent link.

10. Preserve source wording and substantive meaning. Do not infer, paraphrase, or add
    policy logic. All child stubs contain only Text, except Entity-description Event may
    use exactly {{"Self": true}}.

Output:
[
  {{"id": "r0_main_condition0:c0", "Node": condition_node_plan}}
]
""".strip()


def _condition_plan_reason(task: Dict, value: Any) -> str:
    current_text = task["CurrentText"]
    root_text = task["RootText"]

    if not isinstance(value, dict) or set(value) != CONDITION_NODE_KEYS:
        return "node:invalid_schema"
    if not isinstance(value.get("Text"), str):
        return "node.Text:not_string"

    marker = value.get("Marker")
    if marker is not None and (
        not isinstance(marker, str) or not phrase_grounded(marker, current_text)
    ):
        return f"node.Marker:not_grounded({marker!r})"

    parent_role = task.get("ParentHint", {}).get("Role")
    if marker is not None and parent_role not in {"Root", "ConditionCoordinationItem"}:
        return f"node.Marker:not_allowed_for_parent_role({parent_role!r})"

    body = value.get("Body")
    if not isinstance(body, dict) or len(body) != 1:
        return "node.Body:need_exactly_one_type"

    body_type = next(iter(body))
    if body_type not in CONDITION_BODY_TYPES:
        return f"node.Body:invalid_type({body_type!r})"

    # Entity Relation/Coordination children are semantic participants.
    if (
        parent_role in {"Target", "Source", "CoordinationItem"}
        and body_type != "Entity"
    ):
        return f"node.Body:{body_type}_not_allowed_for_entity_participant"

    if body_type == "Atom":
        atom = body.get("Atom")
        if not isinstance(atom, str) or not phrase_grounded(atom, current_text):
            return f"node.Body.Atom:not_grounded({atom!r})"
        # With an outer Marker, a multiword residual phrase is usually a structured
        # nominal/event body rather than an indivisible Atom. Retry instead of silently
        # collapsing analyzable structure (e.g. "by any means").
        if marker is not None and len(normalize_space(atom).split()) > 1:
            return "node.Body.Atom:multiword_after_marker_requires_structure"
        return ""

    if body_type == "Entity":
        entity = body.get("Entity")
        if not isinstance(entity, dict) or set(entity) != ENTITY_BODY_KEYS:
            return "node.Body.Entity:invalid_schema"
        full_plan = {"Text": current_text, **copy.deepcopy(entity)}
        local_task = dict(task)
        local_task["Context"] = root_text
        return entity_plan_reason(local_task, full_plan)

    if body_type == "Event":
        return event_frame_plan_reason(
            body.get("Event"),
            current_text,
            root_text,
            "node.Body.Event",
            allow_self=False,
        )

    coordination = body.get("Coordination")
    if not isinstance(coordination, dict) or set(coordination) != {"Operator", "Items"}:
        return "node.Body.Coordination:invalid_schema"
    operator = coordination.get("Operator")
    if operator not in OPERATORS:
        return f"node.Body.Coordination:invalid_operator({operator!r})"
    items = coordination.get("Items")
    if not isinstance(items, list) or len(items) < 2:
        return "node.Body.Coordination.Items:need_at_least_2"

    seen = set()
    for index, item in enumerate(items):
        reason = stub_text_reason(
            item,
            current_text,
            f"node.Body.Coordination.Items[{index}]",
        )
        if reason:
            return reason
        key = normalize_space(item["Text"]).lower()
        if key in seen:
            return f"node.Body.Coordination.Items[{index}]:duplicate_text"
        seen.add(key)

    return ""


def _parse_condition_response(response: str, tasks: List[Dict]):
    parsed = parse_json_response(response)
    if not isinstance(parsed, list):
        raise ValueError("Shallow Condition decomposition output must be a list.")

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
            plan = copy.deepcopy(plan)
            plan["Text"] = task["CurrentText"]

        reason = _condition_plan_reason(task, plan)
        if reason:
            failed.append(task)
            failure_details.append(
                f"{task_id}:{reason};CurrentText={task['CurrentText']!r}"
            )
            continue

        result[task_id] = plan

    return result, failed, failure_details


def _fallback_condition_graph(root_text: str, node_prefix: str = "c") -> Dict:
    text = root_text.strip() if isinstance(root_text, str) else ""
    root_id = f"{node_prefix}0"
    return {
        "Root": root_id,
        "Nodes": {
            root_id: {
                "Text": text,
                "Marker": None,
                "Body": {"Atom": text},
            }
        },
    }


def _simple_condition_body(state: GraphState, node_id: str, text: str) -> Dict:
    parent_hint = state.meta[node_id].get("parent_hint", {})
    role = parent_hint.get("Role")
    if role in {"Target", "Source", "CoordinationItem", "EventArgument"}:
        return {
            "Entity": {
                "Head": text or None,
                "Coordination": None,
                "DescriptionGroups": [],
                "Reference": None,
            }
        }
    return {"Atom": text}


def _fallback_condition_node(state: GraphState, node_id: str) -> None:
    text = state.nodes[node_id].get("Text", "")
    state.nodes[node_id] = {
        "Text": text,
        "Marker": None,
        "Body": _simple_condition_body(state, node_id, text),
    }
    state.meta[node_id]["status"] = "expanded"
    state.meta[node_id]["fallback"] = True
    state.fallback_nodes += 1


def _expand_condition_deterministic(state: GraphState, node_id: str) -> bool:
    text = state.nodes[node_id].get("Text", "")
    if not simple_token(text):
        return False
    state.nodes[node_id] = {
        "Text": text,
        "Marker": None,
        "Body": _simple_condition_body(state, node_id, text),
    }
    state.meta[node_id]["status"] = "expanded"
    state.meta[node_id]["deterministic"] = True
    return True


def _wrap_entity_node(node: Dict) -> Dict:
    if not isinstance(node, dict) or set(node) != NODE_KEYS:
        raise ValueError("Expected installed Entity node.")
    return {
        "Text": node["Text"],
        "Marker": None,
        "Body": {"Entity": {key: copy.deepcopy(node[key]) for key in ENTITY_BODY_KEYS}},
    }


def _convert_new_entity_fallback_nodes(
    state: GraphState,
    before_ids: Set[str],
) -> None:
    for node_id in set(state.nodes) - before_ids:
        node = state.nodes[node_id]
        if isinstance(node, dict) and set(node) == NODE_KEYS:
            state.nodes[node_id] = _wrap_entity_node(node)


def _install_condition_plan(
    task: Dict,
    plan: Dict,
    max_depth: int,
    max_nodes_per_graph: int,
) -> None:
    state: GraphState = task["state"]
    node_id = task["node_id"]
    depth = state.meta[node_id]["depth"]
    marker = plan.get("Marker")
    body = plan["Body"]
    body_type = next(iter(body))

    if body_type == "Atom":
        state.nodes[node_id] = {
            "Text": task["CurrentText"],
            "Marker": marker,
            "Body": {"Atom": body["Atom"]},
        }
        state.meta[node_id]["status"] = "expanded"
        return

    if body_type == "Entity":
        before_ids = set(state.nodes)
        entity_plan = {
            "Text": task["CurrentText"],
            **copy.deepcopy(body["Entity"]),
        }
        install_entity_plan(
            task,
            entity_plan,
            max_depth,
            max_nodes_per_graph,
        )
        installed = state.nodes[node_id]
        entity_body = {key: copy.deepcopy(installed[key]) for key in ENTITY_BODY_KEYS}
        _convert_new_entity_fallback_nodes(state, before_ids)
        state.nodes[node_id] = {
            "Text": task["CurrentText"],
            "Marker": marker,
            "Body": {"Entity": entity_body},
        }
        state.meta[node_id]["status"] = "expanded"
        return

    if body_type == "Event":
        installed_event = install_event_frame(
            task,
            body["Event"],
            max_depth,
            max_nodes_per_graph,
            allow_self=False,
            fallback_node_handler=_fallback_condition_node,
        )
        state.nodes[node_id] = {
            "Text": task["CurrentText"],
            "Marker": marker,
            "Body": {"Event": installed_event},
        }
        state.meta[node_id]["status"] = "expanded"
        return

    coordination = body["Coordination"]
    coordination_out = {
        "Operator": coordination["Operator"],
        "Items": [],
    }
    local_cache = {}

    for item in coordination["Items"]:
        text = item["Text"]
        parent_hint = {
            "Role": "ConditionCoordinationItem",
            "Operator": coordination["Operator"],
        }
        if depth + 1 > max_depth:
            if len(state.nodes) >= max_nodes_per_graph:
                raise ValueError(
                    f"max_nodes_per_graph exceeded for {state.candidate_id}: "
                    f"{max_nodes_per_graph}"
                )
            child_id = state.new_node(text, depth + 1, parent_hint)
            _fallback_condition_node(state, child_id)
        else:
            child_id = intern_child_node(
                state,
                node_id,
                text,
                depth + 1,
                parent_hint,
                local_cache,
                role=f"condition_coordination:{coordination['Operator']}",
                max_nodes_per_graph=max_nodes_per_graph,
                allow_unique_existing=False,
            )
        coordination_out["Items"].append({"Ref": child_id})

    state.nodes[node_id] = {
        "Text": task["CurrentText"],
        "Marker": marker,
        "Body": {"Coordination": coordination_out},
    }
    state.meta[node_id]["status"] = "expanded"


def _condition_body_graph_reason(
    body: Any,
    node_text: str,
    nodes: Dict[str, Dict],
    root_text: str,
    path: str,
) -> str:
    if not isinstance(body, dict) or len(body) != 1:
        return f"{path}:need_exactly_one_type"
    body_type = next(iter(body))
    if body_type not in CONDITION_BODY_TYPES:
        return f"{path}:invalid_type({body_type!r})"

    if body_type == "Atom":
        atom = body.get("Atom")
        if not isinstance(atom, str) or not phrase_grounded(atom, node_text):
            return f"{path}.Atom:not_grounded({atom!r})"
        return ""

    if body_type == "Entity":
        return entity_body_graph_reason(
            body.get("Entity"),
            node_text,
            nodes,
            root_text,
            f"{path}.Entity",
        )

    if body_type == "Event":
        return event_frame_graph_reason(
            body.get("Event"),
            node_text,
            nodes,
            f"{path}.Event",
            allow_self=False,
        )

    coordination = body.get("Coordination")
    if not isinstance(coordination, dict) or set(coordination) != {"Operator", "Items"}:
        return f"{path}.Coordination:invalid_schema"
    if coordination.get("Operator") not in OPERATORS:
        return f"{path}.Coordination:invalid_operator"
    items = coordination.get("Items")
    if not isinstance(items, list) or len(items) < 2:
        return f"{path}.Coordination.Items:need_at_least_2"
    seen = set()
    for index, item in enumerate(items):
        item_path = f"{path}.Coordination.Items[{index}]"
        if not isinstance(item, dict) or set(item) != REF_KEYS:
            return f"{item_path}:invalid_ref"
        ref_id = item.get("Ref")
        if not isinstance(ref_id, str) or ref_id not in nodes:
            return f"{item_path}:missing_ref({ref_id!r})"
        if ref_id in seen:
            return f"{item_path}:duplicate_ref"
        seen.add(ref_id)
    return ""


def _condition_body_edges(body: Dict) -> Set[str]:
    if not isinstance(body, dict) or len(body) != 1:
        return set()
    body_type = next(iter(body))
    if body_type not in CONDITION_BODY_TYPES:
        return set()
    if body_type == "Entity":
        return entity_body_edges(body["Entity"])
    if body_type == "Event":
        return event_frame_edges(body["Event"])
    if body_type == "Coordination":
        return {
            item["Ref"]
            for item in body["Coordination"].get("Items", [])
            if isinstance(item, dict) and isinstance(item.get("Ref"), str)
        }
    return set()


def _local_condition_graph_reason(
    root_text: str,
    graph: Any,
    _context: str,
) -> str:
    if not isinstance(graph, dict) or set(graph) != GRAPH_KEYS:
        return "graph:invalid_schema"
    root = graph.get("Root")
    nodes = graph.get("Nodes")
    if not isinstance(root, str) or not root.startswith("c"):
        return f"graph.Root:invalid_condition_prefix({root!r})"
    if not isinstance(nodes, dict):
        return "graph.Nodes:not_object"
    if root not in nodes:
        return "graph.Root:missing_node"

    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}
    for node_id, node in nodes.items():
        path = f"Nodes[{node_id!r}]"
        if not isinstance(node_id, str) or not node_id.startswith("c"):
            return f"{path}:invalid_condition_prefix"
        if not isinstance(node, dict) or set(node) != CONDITION_NODE_KEYS:
            return f"{path}:invalid_node_schema"

        text = node.get("Text")
        if not isinstance(text, str) or not phrase_grounded(text, root_text):
            return f"{path}.Text:not_grounded({text!r})"

        marker = node.get("Marker")
        if marker is not None and (
            not isinstance(marker, str) or not phrase_grounded(marker, text)
        ):
            return f"{path}.Marker:not_grounded({marker!r})"

        reason = _condition_body_graph_reason(
            node.get("Body"),
            text,
            nodes,
            root_text,
            f"{path}.Body",
        )
        if reason:
            return reason
        adjacency[node_id].update(_condition_body_edges(node["Body"]))

    visited: Set[str] = set()
    visiting: Set[str] = set()

    def dfs(current: str) -> str:
        if current in visiting:
            return f"graph:cycle_at({current})"
        if current in visited:
            return ""
        visiting.add(current)
        for child in adjacency[current]:
            reason = dfs(child)
            if reason:
                return reason
        visiting.remove(current)
        visited.add(current)
        return ""

    reason = dfs(root)
    if reason:
        return reason
    unreachable = sorted(set(nodes) - visited)
    if unreachable:
        return f"graph:unreachable_nodes({unreachable})"
    return ""


def _remap_refs(value: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_remap_refs(item, mapping) for item in value]
    if isinstance(value, dict):
        if set(value) == REF_KEYS:
            return {"Ref": mapping[value["Ref"]]}
        return {key: _remap_refs(item, mapping) for key, item in value.items()}
    return copy.deepcopy(value)


def _aggregate_local_graphs(local_graphs: List[Dict]) -> Dict:
    roots = []
    nodes = {}
    next_index = 0

    for graph in local_graphs:
        local_nodes = graph["Nodes"]
        mapping = {}
        for local_id in local_nodes:
            mapping[local_id] = f"c{next_index}"
            next_index += 1

        roots.append({"Ref": mapping[graph["Root"]]})
        for local_id, node in local_nodes.items():
            nodes[mapping[local_id]] = _remap_refs(node, mapping)

    return {"Roots": roots, "Nodes": nodes}


def _condition_graph_reason(condition_texts: List[str], graph: Any) -> str:
    if not isinstance(graph, dict) or set(graph) != CONDITION_GRAPH_KEYS:
        return "graph:invalid_schema"
    roots = graph.get("Roots")
    nodes = graph.get("Nodes")
    if not isinstance(roots, list):
        return "graph.Roots:not_list"
    if not isinstance(nodes, dict):
        return "graph.Nodes:not_object"
    if len(roots) != len(condition_texts):
        return "graph.Roots:count_mismatch"

    root_ids = []
    seen_roots = set()
    for index, root_ref in enumerate(roots):
        path = f"Roots[{index}]"
        if not isinstance(root_ref, dict) or set(root_ref) != CONDITION_ROOT_KEYS:
            return f"{path}:invalid_ref"
        root_id = root_ref.get("Ref")
        if not isinstance(root_id, str) or root_id not in nodes:
            return f"{path}:missing_ref({root_id!r})"
        if root_id in seen_roots:
            return f"{path}:duplicate_root"
        seen_roots.add(root_id)
        root_ids.append(root_id)

    if not condition_texts:
        return "" if not nodes else "graph:unexpected_nodes_without_conditions"

    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}
    for node_id, node in nodes.items():
        path = f"Nodes[{node_id!r}]"
        if not isinstance(node_id, str) or not node_id.startswith("c"):
            return f"{path}:invalid_condition_prefix"
        if not isinstance(node, dict) or set(node) != CONDITION_NODE_KEYS:
            return f"{path}:invalid_node_schema"
        adjacency[node_id].update(_condition_body_edges(node["Body"]))

    globally_owned: Set[str] = set()

    for index, root_id in enumerate(root_ids):
        root_text = condition_texts[index]
        root_node = nodes[root_id]
        if normalize_space(root_node.get("Text", "")) != normalize_space(root_text):
            return f"Roots[{index}]:root_text_mismatch"
        if root_node.get("Marker") is None and re.match(
            r"^(?:based(?:\s+solely)?\s+on|including|such\s+as\s+by)\b",
            root_text,
            re.I,
        ):
            return f"Roots[{index}]:missing_outer_marker"

        visiting: Set[str] = set()
        local_visited: Set[str] = set()

        def dfs(node_id: str) -> str:
            if node_id in visiting:
                return f"graph:cycle_at({node_id})"
            if node_id in local_visited:
                return ""
            if node_id in globally_owned:
                return f"graph:node_shared_across_roots({node_id})"

            node = nodes[node_id]
            path = f"Nodes[{node_id!r}]"
            text = node.get("Text")
            if not isinstance(text, str) or not phrase_grounded(text, root_text):
                return f"{path}.Text:not_grounded({text!r})"

            marker = node.get("Marker")
            if marker is not None and (
                not isinstance(marker, str) or not phrase_grounded(marker, text)
            ):
                return f"{path}.Marker:not_grounded({marker!r})"

            reason = _condition_body_graph_reason(
                node.get("Body"),
                text,
                nodes,
                root_text,
                f"{path}.Body",
            )
            if reason:
                return reason

            reason = _condition_node_coverage_reason(node, nodes, path)
            if reason:
                return reason

            visiting.add(node_id)
            for child in adjacency[node_id]:
                reason = dfs(child)
                if reason:
                    return reason
            visiting.remove(node_id)
            local_visited.add(node_id)
            return ""

        reason = dfs(root_id)
        if reason:
            return reason
        globally_owned.update(local_visited)

    unreachable = sorted(set(nodes) - globally_owned)
    if unreachable:
        return f"graph:unreachable_nodes({unreachable})"
    return ""


def _materialize_condition_body(
    body: Dict,
    expand_ref,
    stack: Set[str],
) -> Dict:
    body_type = next(iter(body))
    if body_type == "Atom":
        return {"Atom": body["Atom"]}

    if body_type == "Event":
        return {
            "Event": materialize_event_frame(
                body["Event"],
                expand_ref,
                stack,
            )
        }

    if body_type == "Coordination":
        coordination = body["Coordination"]
        return {
            "Coordination": {
                "Operator": coordination["Operator"],
                "Items": [expand_ref(item, stack) for item in coordination["Items"]],
            }
        }

    entity = body["Entity"]

    def expand_relation(relation: Dict) -> Dict:
        if "Target" in relation:
            return {
                "Predicate": relation["Predicate"],
                "Target": expand_ref(relation["Target"], stack),
            }
        return {
            "Source": expand_ref(relation["Source"], stack),
            "Predicate": relation["Predicate"],
        }

    def expand_description(item: Dict) -> Dict:
        keys = set(item)
        if keys == {"Modifier"}:
            return {"Modifier": item["Modifier"]}
        if keys == {"Relation"}:
            return {"Relation": expand_relation(item["Relation"])}
        if keys == {"Event"}:
            return {
                "Event": materialize_event_frame(
                    item["Event"],
                    expand_ref,
                    stack,
                )
            }
        return {
            "Operator": item["Operator"],
            "Items": [expand_description(child) for child in item["Items"]],
        }

    coordination = entity.get("Coordination")
    coordination_out = None
    if coordination is not None:
        coordination_out = {
            "Operator": coordination["Operator"],
            "Items": [expand_ref(item, stack) for item in coordination["Items"]],
        }

    return {
        "Entity": {
            "Head": entity["Head"],
            "Coordination": coordination_out,
            "DescriptionGroups": [
                expand_description(item) for item in entity.get("DescriptionGroups", [])
            ],
            "Reference": copy.deepcopy(entity["Reference"]),
        }
    }


def materialize_condition_graph(graph: Dict) -> List[Dict]:
    if not isinstance(graph, dict) or set(graph) != CONDITION_GRAPH_KEYS:
        raise ValueError("Invalid Condition_Elements graph.")

    roots = graph["Roots"]
    nodes = graph["Nodes"]

    def expand_ref(ref: Dict, stack: Set[str]) -> Dict:
        if not isinstance(ref, dict) or set(ref) != REF_KEYS:
            raise ValueError(f"Invalid condition Ref: {ref!r}")
        return expand_node(ref["Ref"], stack)

    def expand_node(node_id: str, stack: Set[str]) -> Dict:
        if node_id not in nodes:
            raise ValueError(f"Unknown condition node ref: {node_id}")
        if node_id in stack:
            raise ValueError(
                f"Cycle detected while materializing condition at {node_id}"
            )

        stack = set(stack)
        stack.add(node_id)
        node = nodes[node_id]
        return {
            "Text": node["Text"],
            "Marker": node["Marker"],
            "Body": _materialize_condition_body(
                node["Body"],
                expand_ref,
                stack,
            ),
        }

    return [expand_ref(root, set()) for root in roots]


def _process_elements(
    llm_client: LLMClient,
    targets: List[Dict],
    organization: str,
    batch_size: int,
    max_retries: int,
    phase: str,
    max_depth: int,
    max_nodes_per_condition: int,
) -> Dict[str, int]:
    candidates = []
    local_graphs: Dict[str, Dict] = {}
    target_condition_ids: Dict[str, List[str]] = {}

    empty_elements = 0
    existing_elements = 0
    deterministic_conditions = 0
    total_conditions = 0

    for target in targets:
        element = target["element"]
        condition_texts = _condition_texts(element)
        total_conditions += len(condition_texts)

        existing = element.get("Condition_Elements")
        if not _condition_graph_reason(condition_texts, existing):
            existing_elements += 1
            continue

        if not condition_texts:
            element["Condition_Elements"] = {"Roots": [], "Nodes": {}}
            empty_elements += 1
            continue

        ids = []
        context = build_context_text(element)
        for condition_index, condition_text in enumerate(condition_texts):
            candidate_id = f"{target['id']}_condition{condition_index}"
            ids.append(candidate_id)

            if simple_token(condition_text):
                local_graphs[candidate_id] = _fallback_condition_graph(
                    condition_text,
                    node_prefix="c",
                )
                deterministic_conditions += 1
                continue

            candidates.append(
                {
                    "id": candidate_id,
                    "element": element,
                    "RootText": condition_text,
                    "Context": context,
                    "NodePrefix": "c",
                    "PromptContext": {
                        "Sentence": element.get("Sentence", ""),
                        "Subject": element.get("Subject", ""),
                        "Predicate": element.get("Predicate", ""),
                        "Object": element.get("Object", ""),
                    },
                }
            )

        target_condition_ids[target["id"]] = ids

    if candidates:
        graphs, expansion_stats = expand_layered_graphs(
            llm_client=llm_client,
            candidates=candidates,
            organization=organization,
            system_prompt=SYSTEM_PROMPT,
            build_prompt=_build_condition_prompt,
            parse_response=_parse_condition_response,
            install_plan=_install_condition_plan,
            phase=phase,
            log_label="SplitConditionGraph",
            batch_size=batch_size,
            max_retries=max_retries,
            max_depth=max_depth,
            max_nodes_per_graph=max_nodes_per_condition,
            final_graph_reason=_local_condition_graph_reason,
            deterministic_expander=_expand_condition_deterministic,
            fallback_node_handler=_fallback_condition_node,
            fallback_graph_factory=_fallback_condition_graph,
        )
        local_graphs.update(graphs)
        if expansion_stats["fallback_nodes"] or expansion_stats["invalid_graphs"]:
            raise ValueError(
                "Condition graph generation produced lossy fallback nodes: "
                f"fallback_nodes={expansion_stats['fallback_nodes']}, "
                f"invalid_graphs={expansion_stats['invalid_graphs']}"
            )
    else:
        expansion_stats = {
            "candidates": 0,
            "node_batches": 0,
            "llm_nodes": 0,
            "deterministic_nodes": 0,
            "retries": 0,
            "fallback_nodes": 0,
            "invalid_graphs": 0,
            "total_nodes": 0,
        }

    for target in targets:
        target_id = target["id"]
        ids = target_condition_ids.get(target_id)
        if ids is None:
            continue
        graphs_in_order = [local_graphs[item_id] for item_id in ids]
        combined = _aggregate_local_graphs(graphs_in_order)
        condition_texts = _condition_texts(target["element"])
        reason = _condition_graph_reason(condition_texts, combined)
        if reason:
            raise ValueError(
                "Condition graph aggregate is invalid; refusing fallback: "
                f"phase={phase}, id={target_id}, reason={reason}"
            )
        target["element"]["Condition_Elements"] = combined

    generated_total_nodes = sum(
        len(local_graphs[item_id]["Nodes"])
        for ids in target_condition_ids.values()
        for item_id in ids
        if item_id in local_graphs
    )

    return {
        "targets": len(targets),
        "conditions": total_conditions,
        "candidate_conditions": len(candidates),
        "empty_elements": empty_elements,
        "existing_elements": existing_elements,
        "deterministic_conditions": deterministic_conditions,
        "node_batches": expansion_stats["node_batches"],
        "llm_nodes": expansion_stats["llm_nodes"],
        "deterministic_nodes": expansion_stats["deterministic_nodes"],
        "retries": expansion_stats["retries"],
        "fallback_nodes": expansion_stats["fallback_nodes"],
        "invalid_graphs": expansion_stats["invalid_graphs"],
        "total_nodes": generated_total_nodes,
    }


def split_condition_groups_batch(
    llm_client: LLMClient,
    rules: List[Dict],
    organization: str,
    batch_size: int = 10,
    max_retries: int = 2,
    max_depth: int = 12,
    max_nodes_per_condition: int = 256,
) -> List[Dict]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")
    if max_depth <= 0:
        raise ValueError("max_depth must be greater than 0.")
    if max_nodes_per_condition <= 0:
        raise ValueError("max_nodes_per_condition must be greater than 0.")

    main_targets = _collect_elements(rules, "main")
    example_targets = _collect_elements(rules, "example")
    usage_before = llm_client.get_usage()

    logger.info(
        "SplitConditionGraph | start | rules=%d | main_targets=%d | example_targets=%d",
        len(rules),
        len(main_targets),
        len(example_targets),
    )

    main_stats = _process_elements(
        llm_client,
        main_targets,
        organization,
        batch_size,
        max_retries,
        "main",
        max_depth,
        max_nodes_per_condition,
    )
    example_stats = _process_elements(
        llm_client,
        example_targets,
        organization,
        batch_size,
        max_retries,
        "example",
        max_depth,
        max_nodes_per_condition,
    )

    usage = llm_client.get_usage_since(usage_before)

    logger.info(
        "SplitConditionGraph | completed | "
        "main_targets=%d | main_conditions=%d | "
        "example_targets=%d | example_conditions=%d | "
        "existing=%d | deterministic=%d | empty=%d | "
        "total_nodes=%d | llm_nodes=%d | deterministic_nodes=%d | "
        "node_batches=%d | retries=%d | fallback_nodes=%d | invalid_graphs=%d | "
        "llm_calls=%d | input_tokens=%d | output_tokens=%d | total_tokens=%d",
        main_stats["targets"],
        main_stats["conditions"],
        example_stats["targets"],
        example_stats["conditions"],
        main_stats["existing_elements"] + example_stats["existing_elements"],
        main_stats["deterministic_conditions"]
        + example_stats["deterministic_conditions"],
        main_stats["empty_elements"] + example_stats["empty_elements"],
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

    return normalize_condition_relation_predicates_batch(
        llm_client,
        rules,
        batch_size=batch_size,
        max_retries=max_retries,
    )
