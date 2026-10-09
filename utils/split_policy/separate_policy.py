import json
import logging
import re
from typing import Dict, List

from llms import LLMClient
from utils.utils import parse_json_response

logger = logging.getLogger(__name__)

MARKER_RE = re.compile(
    r"(?<!\w)(such as|including(?:,?\s+but\s+not\s+limited\s+to)?|"
    r"this(?:\s+also)?\s+includes|for example|for instance|e\.g\.)(?!\w)",
    re.I,
)


DANGLING = (
    "related to",
    "based on",
    "associated with",
    "consisting of",
    "depending on",
    "because of",
    "due to",
    "such as",
    "including",
    "with",
    "without",
    "against",
    "from",
    "into",
    "of",
    "for",
    "by",
)

CONTINUE_RE = re.compile(
    r",\s+(?=that\b|which\b|who\b|whom\b|whose\b|from\b|in\b|on\b|by\b|"
    r"with\b|without\b|under\b|through\b|for\b|when\b|where\b|unless\b|"
    r"except\b|if\b|based\s+on\b|depending\s+on\b|provided\b|subject\s+to\b|"
    r"because\s+of\b|due\s+to\b)",
    re.I,
)

VALID_ROLES = {
    "PHRASE_EXAMPLE",
    "SENTENCE_EXAMPLE",
    "NORMATIVE_CONTENT",
    "UNCERTAIN",
}

EXAMPLE_CLASSIFIER_SYSTEM_PROMPT = """
You are a policy semantic analysis assistant.

Python has identified possible example candidates.
For each candidate, classify it as:
- PHRASE_EXAMPLE
- SENTENCE_EXAMPLE
- NORMATIVE_CONTENT
- UNCERTAIN

Preserve the complete normative meaning of the policy.

PHRASE_EXAMPLE:
- The candidate gives examples, members, types, or instances of a specific
  noun phrase or concept in the main clause.
- The candidate items instantiate that phrase itself.
- It answers: "What are examples of this phrase?"
- anchor must be the shortest exact noun phrase or coordinated noun phrase
  directly illustrated by the candidate.
- illustrative_span must be the exact leading substring of candidate that is
  merely illustrative. Exclude any trailing condition, qualification,
  exception, consequence, or other normative content.

SENTENCE_EXAMPLE:
- The candidate gives a concrete instance, special case, or way of performing
  the prohibited action or proposition expressed by the main clause.
- It answers: "What is a concrete example of this rule or prohibited conduct?"
- It does not need to be a grammatically complete sentence; an action phrase,
  gerund clause, noun phrase naming a prohibited practice, or similar
  proposition may qualify.
- If the candidate illustrates the governing action or the rule as a whole
  rather than a noun phrase, classify it as SENTENCE_EXAMPLE.
- anchor must be null.
- illustrative_span follows the same exact-substring rule.

NORMATIVE_CONTENT:
- The candidate adds required scope, conditions, exceptions, qualifications,
  or substantive restrictions and cannot be removed as merely illustrative.
- anchor must be null.
- illustrative_span must be null.

UNCERTAIN:
- Use only when the candidate cannot be reliably assigned to another role.
- anchor must be null.
- illustrative_span must be null.

Do not classify a candidate as PHRASE_EXAMPLE merely because it follows
"including" or "such as".

"This includes ..." or "This also includes ..." at the start of a following
sentence normally elaborates
the preceding prohibited conduct and should be SENTENCE_EXAMPLE unless it adds
a genuine condition, exception, qualification, or independent rule.

Distinguish by what the candidate illustrates:
- "groups, including nationalities, regions, genders" -> PHRASE_EXAMPLE,
  because the listed items are examples of the group concept.
- "attempt to circumvent security settings, including accessing data,
  logging into servers, or probing networks" -> SENTENCE_EXAMPLE,
  because the listed actions are concrete ways of performing the prohibited
  conduct, not examples of "security settings or network systems".

For all roles except PHRASE_EXAMPLE, anchor must be null.

Do not rewrite, summarize, paraphrase, or invent content.
Return JSON only.
""".strip()


def split_sentences(text):
    result = []

    start = 0
    length = len(text)

    for index, char in enumerate(text):
        if char not in ".!?":
            continue

        if index + 1 < length and not text[index + 1].isspace():
            continue

        # Do not split after abbreviations used as example markers.
        prefix = text[max(0, index - 3) : index + 1].lower()
        if prefix in {"e.g.", "i.e."}:
            continue

        # ``including:1. first;2. second`` is common in normalized policy
        # sources.  The numbered item marker is not a sentence boundary.
        next_index = index + 1
        while next_index < length and text[next_index].isspace():
            next_index += 1

        if (
            char == "."
            and index > 0
            and text[index - 1].isdigit()
            and next_index < length
        ):
            number_start = index - 1
            while number_start > 0 and text[number_start - 1].isdigit():
                number_start -= 1
            enumerator_prefix = text[start:number_start]
            if number_start == 0 or re.search(
                r"(?:^|[;:])\s*(?:and|or)?\s*$", enumerator_prefix, re.I
            ):
                continue

        raw = text[start : index + 1]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())

        if raw.strip():
            result.append((raw.strip(), start + left, start + right))

        start = index + 1

    if start < length:
        raw = text[start:]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())

        if raw.strip():
            result.append((raw.strip(), start + left, start + right))

    return result


def find_candidates(text):
    candidates = []

    for _, sent_start, sent_end in split_sentences(text):
        sentence = text[sent_start:sent_end]

        for match in MARKER_RE.finditer(sentence):
            marker_start = sent_start + match.start()
            marker_end = sent_start + match.end()

            rest = text[marker_end:sent_end]
            skip_match = re.match(r"[\s,:-]*", rest)
            skip = skip_match.end() if skip_match else 0

            value_start = marker_end + skip
            value_end = sent_end

            if marker_start > 0 and text[marker_start - 1] == "(":
                close = text.find(")", value_start, sent_end)
                if close != -1:
                    value_end = close
            else:
                continuation = CONTINUE_RE.search(text[value_start:sent_end])
                if continuation:
                    value_end = value_start + continuation.start()

            value = text[value_start:value_end].strip()
            if not value:
                continue

            value_start = text.find(value, value_start, value_end + 1)
            value_end = value_start + len(value)

            remove_start = marker_start
            remove_end = value_end

            before = text[sent_start:marker_start]
            comma = re.search(r",\s*$", before)
            if comma:
                remove_start = sent_start + comma.start()
            else:
                semicolon = re.search(r";\s*$", before)
                if semicolon:
                    remove_start = sent_start + semicolon.start()

            after = text[remove_end:sent_end]
            comma = re.match(r"\s*,\s*", after)
            if comma:
                remove_end += comma.end()

            if (
                remove_start > 0
                and text[remove_start - 1] == "("
                and remove_end < len(text)
                and text[remove_end] == ")"
            ):
                remove_start -= 1
                remove_end += 1

            # If an example ends one sentence but the policy contains another
            # sentence, retain the sentence-ending punctuation.  Otherwise
            # removing the example joins two independent rules together.
            if text[remove_end:].strip():
                punctuation_end = remove_end
                while (
                    punctuation_end > remove_start
                    and text[punctuation_end - 1].isspace()
                ):
                    punctuation_end -= 1

                if (
                    punctuation_end > remove_start
                    and text[punctuation_end - 1] in ".!?"
                ):
                    remove_end = punctuation_end - 1

            marker = match.group()

            candidates.append(
                {
                    "marker": marker,
                    "value": value,
                    "full_value": text[marker_start:value_end].strip(),
                    "marker_start": marker_start,
                    "value_start": value_start,
                    "value_end": value_end,
                    "remove_start": remove_start,
                    "remove_end": remove_end,
                }
            )

    return candidates


def remove_span(text, start, end):
    text = text[:start] + " " + text[end:]
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s*[—–-]\s*$", "", text)
    return text.strip()


def detachable(text, candidate):
    remaining = remove_span(
        text,
        candidate["remove_start"],
        candidate["remove_end"],
    )

    if not remaining:
        return False

    normalized = remaining.rstrip(".,;:!?").lower()
    return not any(normalized.endswith(item) for item in DANGLING)


def split_sentence_examples(text: str) -> List[str]:
    text = text.strip().rstrip(".")

    # Commas frequently introduce coordinated actions with a shared object, or
    # noun-list items such as "fraud, phishing, malware". Splitting on commas
    # silently detaches that shared structure. Semicolons are the safe boundary
    # for independent example clauses.
    parts = re.split(
        r";\s*(?=(?:and\s+|or\s+)?[A-Za-z][A-Za-z'-]*ing\b)",
        text,
        flags=re.I,
    )

    if len(parts) <= 1:
        return [text]

    results = []

    for part in parts:
        part = re.sub(r"^(?:and|or)\s+", "", part.strip(), flags=re.I)

        if not re.match(r"^[A-Za-z][A-Za-z'-]*ing\b", part, re.I):
            return [text]

        normalized = part.rstrip(".,;:!?").lower()
        if any(normalized.endswith(item) for item in DANGLING):
            return [text]

        results.append(part)

    return results


def _build_batch_prompt(batch):
    items = []

    for task in batch:
        items.append(
            {
                "id": task["id"],
                "policy": task["policy"],
                "candidates": [
                    {
                        "id": i,
                        "marker": candidate["marker"],
                        "candidate": candidate["value"],
                    }
                    for i, candidate in enumerate(task["candidates"])
                ],
            }
        )

    return f"""
Input:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return exactly:
{{
  "items": [
    {{
      "id": 0,
      "candidates": [
        {{
          "id": 0,
          "role": "PHRASE_EXAMPLE",
          "anchor": "exact phrase from policy",
          "illustrative_span": "exact leading substring from candidate"
        }}
      ]
    }}
  ]
}}

Allowed roles:
- PHRASE_EXAMPLE
- SENTENCE_EXAMPLE
- NORMATIVE_CONTENT
- UNCERTAIN

Rules:
1. Return one item for every policy id.
2. Return one result for every candidate id.
3. PHRASE_EXAMPLE gives examples, members, types, or instances of a specific
   noun phrase or concept; the candidate items instantiate that phrase itself.
4. SENTENCE_EXAMPLE gives a concrete instance, special case, or way of
   performing the main prohibited action or proposition.
5. SENTENCE_EXAMPLE does not need to be a complete grammatical sentence.
6. If the candidate illustrates the governing action or rule as a whole rather
   than a noun phrase, classify it as SENTENCE_EXAMPLE.
7. NORMATIVE_CONTENT is required scope, condition, exception, qualification,
   or other content necessary to preserve normative meaning.
8. For PHRASE_EXAMPLE, anchor must be the shortest exact phrase from the policy
   whose meaning is directly instantiated by the candidate.
9. For all other roles, anchor must be null.
10. For PHRASE_EXAMPLE and SENTENCE_EXAMPLE, illustrative_span must be an exact,
    non-empty leading substring of candidate. Include only the illustrative
    material. Exclude a trailing normative condition, qualification, exception,
    consequence, or restriction even when Python included it in candidate.
11. For NORMATIVE_CONTENT and UNCERTAIN, illustrative_span must be null.
12. Do not infer PHRASE_EXAMPLE merely from the marker "including" or "such as".
13. Do not invent or paraphrase content.
14. Output JSON only.
""".strip()


def _parse_batch_response(response, batch):
    parsed = parse_json_response(response)

    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        raise ValueError("Invalid SeparatePolicy response.")

    policy_map = {
        item.get("id"): item for item in parsed["items"] if isinstance(item, dict)
    }

    result = {}

    for task in batch:
        policy_result = policy_map.get(task["id"])

        if not isinstance(policy_result, dict):
            raise ValueError(f"Missing policy id {task['id']}.")

        candidates = policy_result.get("candidates")

        if not isinstance(candidates, list):
            raise ValueError(f"Missing candidates for policy id {task['id']}.")

        candidate_map = {
            item.get("id"): item for item in candidates if isinstance(item, dict)
        }

        expected_ids = set(range(len(task["candidates"])))

        if set(candidate_map) != expected_ids:
            raise ValueError(f"Candidate ids mismatch for policy id {task['id']}.")

        for item in candidate_map.values():
            if item.get("role") not in VALID_ROLES:
                raise ValueError("Invalid candidate role.")

        for candidate_id, item in candidate_map.items():
            role = item.get("role")
            illustrative_span = item.get("illustrative_span")

            if role in {"PHRASE_EXAMPLE", "SENTENCE_EXAMPLE"}:
                if not isinstance(illustrative_span, str):
                    raise ValueError("Missing illustrative_span.")

                illustrative_span = illustrative_span.strip()
                candidate_value = task["candidates"][candidate_id]["value"]

                if not illustrative_span or not candidate_value.startswith(
                    illustrative_span
                ):
                    raise ValueError("illustrative_span must be a candidate prefix.")

                item["illustrative_span"] = illustrative_span
            elif illustrative_span is not None:
                raise ValueError("Unexpected illustrative_span.")

        result[task["id"]] = candidate_map

    return result


def _default_rule(policy, policy_id):
    return {
        "ID": str(policy_id),
        "Original_Policy": policy,
        "Main_Clause": policy,
        "Example_Sentences": [],
        "Examples": {},
    }


def _with_illustrative_span(policy, candidate, item):
    """Restrict a broad Python candidate to the span approved by the model."""
    illustrative_span = item.get("illustrative_span")

    # Kept for callers using stored/legacy analyses. New model responses are
    # validated by _parse_batch_response and always provide this field.
    if not isinstance(illustrative_span, str) or not illustrative_span.strip():
        return candidate

    illustrative_span = illustrative_span.strip()

    if not candidate["value"].startswith(illustrative_span):
        return candidate

    value_end = candidate["value_start"] + len(illustrative_span)
    adjusted = dict(candidate)
    adjusted["value"] = illustrative_span
    adjusted["value_end"] = value_end
    adjusted["full_value"] = policy[candidate["marker_start"] : value_end].strip()
    if value_end != candidate["value_end"]:
        omitted_tail = policy[value_end : candidate["value_end"]].strip()

        if omitted_tail and all(char in ".!?" for char in omitted_tail):
            adjusted["remove_end"] = candidate["remove_end"]
        else:
            adjusted["remove_end"] = value_end
    return adjusted


def _apply_analysis(policy, policy_id, candidates, analysis):
    examples = {}
    example_sentences = []
    remove_spans = []

    accepted = []

    for i, candidate in enumerate(candidates):
        item = analysis.get(i, {})
        role = item.get("role", "UNCERTAIN")

        if role not in {"PHRASE_EXAMPLE", "SENTENCE_EXAMPLE"}:
            continue

        candidate = _with_illustrative_span(policy, candidate, item)

        if not detachable(policy, candidate):
            continue

        if role == "PHRASE_EXAMPLE":
            anchor = item.get("anchor")

            if not isinstance(anchor, str) or not anchor or anchor not in policy:
                continue

            if policy.find(anchor) >= candidate["remove_start"]:
                continue

        accepted.append((i, role, item, candidate))

    # A marker inside another accepted example (for example, "including X,
    # such as Y") must not be extracted and deleted a second time. Besides
    # duplicate examples, applying original offsets for nested deletions can
    # remove text beyond the outer span after the string length changes.
    selected_ids = set()
    selected_spans = []

    for i, _, _, candidate in sorted(
        accepted,
        key=lambda entry: (
            -(entry[3]["remove_end"] - entry[3]["remove_start"]),
            entry[3]["remove_start"],
        ),
    ):
        span = (candidate["remove_start"], candidate["remove_end"])

        if any(
            max(span[0], other[0]) < min(span[1], other[1]) for other in selected_spans
        ):
            continue

        selected_ids.add(i)
        selected_spans.append(span)

    for i, role, item, candidate in accepted:
        if i not in selected_ids:
            continue

        if role == "PHRASE_EXAMPLE":
            anchor = item["anchor"]
            examples.setdefault(anchor, []).append(candidate["full_value"])
        else:
            example_sentences.extend(split_sentence_examples(candidate["value"]))

        remove_spans.append((candidate["remove_start"], candidate["remove_end"]))

    main_clause = policy

    for start, end in sorted(remove_spans, reverse=True):
        main_clause = remove_span(main_clause, start, end)

    if not main_clause or any(
        main_clause.rstrip(".,;:!?").lower().endswith(item) for item in DANGLING
    ):
        return _default_rule(policy, policy_id)

    main_clause = main_clause.rstrip()

    if main_clause and main_clause[-1] not in ".!?":
        main_clause += "."

    return {
        "ID": str(policy_id),
        "Original_Policy": policy,
        "Main_Clause": main_clause,
        "Example_Sentences": example_sentences,
        "Examples": examples,
    }


def separate_policy_batch(
    llm_client: LLMClient,
    policies: List[str],
    batch_size: int = 10,
    max_retries: int = 5,
) -> List[Dict]:

    rules = [
        _default_rule(policy.strip(), i + 1)
        for i, policy in enumerate(policies)
        if isinstance(policy, str) and policy.strip()
    ]

    tasks = []

    for i, rule in enumerate(rules):
        candidates = find_candidates(rule["Original_Policy"])

        if candidates:
            tasks.append(
                {
                    "id": i,
                    "policy": rule["Original_Policy"],
                    "candidates": candidates,
                }
            )

    total_batches = (len(tasks) + batch_size - 1) // batch_size if tasks else 0

    usage_before = llm_client.get_usage()

    logger.info(
        "SeparatePolicy | start | policies=%d | candidate_policies=%d | batches=%d",
        len(rules),
        len(tasks),
        total_batches,
    )

    retries = 0
    failed_batches = 0

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        batch_num = start // batch_size + 1

        batch_usage_before = llm_client.get_usage()
        remaining = batch
        analysis_map = {}

        for attempt in range(1, max_retries + 1):
            if not remaining:
                break

            try:
                response = llm_client.invoke(
                    _build_batch_prompt(remaining),
                    system_prompt=EXAMPLE_CLASSIFIER_SYSTEM_PROMPT,
                )

                if not response:
                    raise ValueError("Empty LLM response.")

                analysis = _parse_batch_response(response, remaining)

                failed = []

                for task in remaining:
                    task_analysis = analysis.get(task["id"])

                    if task_analysis is None:
                        failed.append(task)
                        continue

                    analysis_map[task["id"]] = task_analysis

                if failed and attempt < max_retries:
                    retries += 1

                    logger.warning(
                        "SeparatePolicy | batch=%d/%d | "
                        "attempt=%d/%d failed | retry_items=%d",
                        batch_num,
                        total_batches,
                        attempt,
                        max_retries,
                        len(failed),
                    )

                remaining = failed

            except Exception as exc:
                if attempt < max_retries:
                    retries += 1

                logger.warning(
                    "SeparatePolicy | batch=%d/%d | attempt=%d/%d failed | error=%s",
                    batch_num,
                    total_batches,
                    attempt,
                    max_retries,
                    exc,
                )

        batch_usage = llm_client.get_usage_since(batch_usage_before)

        if remaining:
            failed_batches += 1

            logger.error(
                "SeparatePolicy | batch=%d/%d failed | items=%d | llm_calls=%d",
                batch_num,
                total_batches,
                len(remaining),
                batch_usage["api_calls"],
            )

            continue

        for task in batch:
            rules[task["id"]] = _apply_analysis(
                task["policy"],
                task["id"] + 1,
                task["candidates"],
                analysis_map[task["id"]],
            )

        logger.info(
            "SeparatePolicy | batch=%d/%d | items=%d | llm_calls=%d",
            batch_num,
            total_batches,
            len(batch),
            batch_usage["api_calls"],
        )

    usage = llm_client.get_usage_since(usage_before)

    logger.info(
        "SeparatePolicy | completed | policies=%d | candidate_policies=%d | "
        "batches=%d | llm_calls=%d | retries=%d | failed_batches=%d | "
        "input_tokens=%d | output_tokens=%d | total_tokens=%d",
        len(rules),
        len(tasks),
        total_batches,
        usage["api_calls"],
        retries,
        failed_batches,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
    )

    return rules
