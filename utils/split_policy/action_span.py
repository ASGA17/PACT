import json
import logging
import re
from typing import Any, Dict, List, Tuple

from utils.split_policy.layered_graph import (
    OPERATORS,
    action_body_reason,
    normalize_space,
    stub_text_reason,
)
from utils.utils import parse_json_response

logger = logging.getLogger(__name__)

ACTION_SPAN_KEYS = {
    "Text",
    "Action",
    "Modifiers",
    "Operators",
    "Coordination",
}

ACTION_SPAN_SYSTEM_PROMPT = """
You decompose source-grounded action/state spans into one-level semantic action plans.

For each CurrentText, analyze only its immediate action layer:
- minimal action or state,
- action modifiers,
- action-level operators,
- action coordination.

Preserve source wording, semantic action boundaries, ellipsis, and explicit logical
scope. Context may be used only for disambiguation. Do not import participants,
arguments, entities, conditions, or policy logic into the action span.

When coordinated actions share leading wording, preserve the shared scope. If needed,
repeat shared source-grounded wording in each child Text so every child remains a
standalone faithful action phrase. Repetition is allowed only from wording already
present in CurrentText; do not paraphrase or invent wording.

Return JSON only.
""".strip()


def needs_action_decomposition(text: str) -> bool:
    """Cheap recall filter; semantic splitting is still decided by the LLM."""
    if not isinstance(text, str):
        return False
    value = normalize_space(text).lower()
    if not value:
        return False
    if "and/or" in value:
        return True
    return bool(re.search(r"\b(and|or|nor)\b", value))


def build_action_span_prompt(
    tasks: List[Dict],
    _organization: str = "",
    *,
    role_name: str = "Action",
) -> str:
    records = []
    for task in tasks:
        context = task.get("PromptContext", {})
        if not isinstance(context, dict):
            context = {}
        records.append(
            {
                "id": task["id"],
                "Role": role_name,
                "RootText": task.get("RootText", task.get("CurrentText", "")),
                "CurrentText": task.get("CurrentText", ""),
                "ParentHint": task.get("ParentHint", {}),
                "Sentence": context.get("Sentence", ""),
                "Subject": context.get("Subject", ""),
                "Predicate": context.get("Predicate", ""),
                "Object": context.get("Object", ""),
                "Condition": context.get("Condition", []),
            }
        )

    input_text = json.dumps(records, ensure_ascii=False, indent=2)
    return f"""
Input:
{input_text}

Return one result per id. Analyze only ONE semantic action layer of CurrentText.

Current action node plan:
{{
  "Text": "CurrentText",
  "Action": "minimal semantic action/state wording" | null,
  "Modifiers": ["source wording"],
  "Operators": ["source wording"],
  "Coordination": null | {{
    "Operator": "AND" | "OR",
    "Items": [{{"Text": "child action span"}}, ...]
  }}
}}

Rules:

1. Action is the minimal lexical action/state. Preserve particles or prepositions
   required by the lexical action or its argument structure, such as "engage in",
   "tamper with", "intrude into", "interfere with", or "lead to".

2. Modifiers contain source wording that changes how the action is performed without
   introducing another action-level scope.

3. Operators scope over the action, such as "attempt to", "try to", "fail to",
   "begin to", "continue to", or source-grounded shared control/purpose wording such
   as "designed to" when it clearly scopes over coordinated child actions.

4. Coordination represents genuine semantic AND/OR between parallel actions. If
   CurrentText coordinates distinct actions, use Action=null and child Text stubs.
   Preserve nested logical scope when needed.

5. Shared wording must not disappear. Example:
   "designed to interfere with, destroy, or limit"
   -> OR children that preserve the shared "designed to" scope, e.g.
      "designed to interfere with", "designed to destroy", "designed to limit".
   The child wording may be a source-grounded subsequence of CurrentText.

6. "insults or abuses" -> OR("insults", "abuses").
   "beautifies and whitewashes" -> AND("beautifies", "whitewashes").
   Do not split lexicalized expressions when splitting changes the action meaning.

7. If CurrentText contains "and/or" lexically but does NOT express action-level
   coordination, keep one Action and Coordination=null.

8. Do not create arguments, entity nodes, Relations, Conditions, or inferred actions.
   Preserve substantive source meaning and do not paraphrase.

9. Child stubs contain only {{"Text": "..."}}.

Output:
[
  {{"id": "...", "Node": action_node_plan}}
]
""".strip()


def action_span_plan_reason(task: Dict, value: Any) -> str:
    current_text = task.get("CurrentText", "")
    if not isinstance(value, dict):
        return "node:not_object"
    if set(value) != ACTION_SPAN_KEYS:
        missing = sorted(ACTION_SPAN_KEYS - set(value))
        extra = sorted(set(value) - ACTION_SPAN_KEYS)
        return f"node:keys(missing={missing},extra={extra})"
    if not isinstance(value.get("Text"), str):
        return "node.Text:not_string"

    reason = action_body_reason(
        {
            "Action": value.get("Action"),
            "Modifiers": value.get("Modifiers"),
            "Operators": value.get("Operators"),
        },
        current_text,
        "node",
        require_action=False,
    )
    if reason:
        return reason

    coordination = value.get("Coordination")
    if coordination is not None:
        if not isinstance(coordination, dict) or set(coordination) != {
            "Operator",
            "Items",
        }:
            return "node.Coordination:invalid_schema"
        operator = coordination.get("Operator")
        if operator not in OPERATORS:
            return f"node.Coordination:invalid_operator({operator!r})"
        items = coordination.get("Items")
        if not isinstance(items, list) or len(items) < 2:
            return "node.Coordination.Items:need_at_least_2"
        seen = set()
        for index, item in enumerate(items):
            reason = stub_text_reason(
                item,
                current_text,
                f"node.Coordination.Items[{index}]",
            )
            if reason:
                return reason
            key = normalize_space(item["Text"]).lower()
            if key in seen:
                return f"node.Coordination.Items[{index}]:duplicate_text"
            seen.add(key)

    action = value.get("Action")
    if action is None and coordination is None:
        return "node:missing_action_or_coordination"
    if action is not None and coordination is not None:
        return "node:action_and_coordination_both_present"
    return ""


def parse_action_span_response(
    response: str,
    tasks: List[Dict],
    parse_json=parse_json_response,
) -> Tuple[Dict[str, Dict], List[Dict], List[str]]:
    parsed = parse_json(response)
    if not isinstance(parsed, list):
        raise ValueError("Shallow action decomposition output must be a list.")

    expected = {task["id"]: task for task in tasks}
    item_map: Dict[str, Dict] = {}
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

    result: Dict[str, Dict] = {}
    failed: List[Dict] = []
    failure_details: List[str] = []
    for task in tasks:
        task_id = task["id"]
        if task_id in duplicates:
            failed.append(task)
            failure_details.append(f"{task_id}:duplicate_id")
            continue
        output_item = item_map.get(task_id)
        if not isinstance(output_item, dict) or "Node" not in output_item:
            failed.append(task)
            failure_details.append(f"{task_id}:missing_Node")
            continue
        plan = output_item.get("Node")
        if isinstance(plan, dict):
            plan = dict(plan)
            plan["Text"] = task.get("CurrentText", "")
        reason = action_span_plan_reason(task, plan)
        if reason:
            failed.append(task)
            failure_details.append(
                f"{task_id}:{reason};CurrentText={task.get('CurrentText', '')!r}"
            )
            continue
        result[task_id] = plan

    return result, failed, failure_details


def analyze_action_spans_batch(
    llm_client: Any,
    requests: List[Dict],
    *,
    batch_size: int = 20,
    max_retries: int = 2,
) -> Dict[str, Dict]:
    """Analyze candidate action spans once; unresolved items remain absent."""
    if not requests:
        return {}
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")

    tasks = []
    for item in requests:
        text = item.get("Text", "")
        tasks.append(
            {
                "id": item["id"],
                "RootText": text,
                "CurrentText": text,
                "ParentHint": item.get("ParentHint", {"Role": "ActionSpan"}),
                "PromptContext": item.get("PromptContext", {}),
            }
        )

    out: Dict[str, Dict] = {}
    for start in range(0, len(tasks), batch_size):
        remaining = tasks[start : start + batch_size]
        for attempt in range(1, max_retries + 1):
            if not remaining:
                break
            try:
                response = llm_client.invoke(
                    build_action_span_prompt(
                        remaining, role_name="EmbeddedRelationAction"
                    ),
                    system_prompt=ACTION_SPAN_SYSTEM_PROMPT,
                )
                if response is None or (
                    isinstance(response, str) and not response.strip()
                ):
                    raise ValueError("Empty action-span response.")
                parsed, failed, details = parse_action_span_response(
                    response, remaining
                )
                out.update(parsed)
                remaining = failed
                if failed and attempt < max_retries:
                    logger.warning(
                        "ActionSpan | retry | attempt=%d/%d | retry=%d | reasons=%s",
                        attempt,
                        max_retries,
                        len(failed),
                        "; ".join(details),
                    )
            except Exception as exc:
                logger.warning(
                    "ActionSpan | retry | attempt=%d/%d | error=%s",
                    attempt,
                    max_retries,
                    exc,
                )
        if remaining:
            logger.warning(
                "ActionSpan | unresolved | ids=%s",
                [task["id"] for task in remaining],
            )
    return out


__all__ = [
    "ACTION_SPAN_KEYS",
    "ACTION_SPAN_SYSTEM_PROMPT",
    "needs_action_decomposition",
    "build_action_span_prompt",
    "action_span_plan_reason",
    "parse_action_span_response",
    "analyze_action_spans_batch",
]
