import json
import logging
import re
from typing import Dict, List, Union

from llms import LLMClient
from utils.utils import parse_json_response

logger = logging.getLogger(__name__)


SUBJECT_PROFILES = {
    "Personal User": ("Natural person acting in an individual or personal capacity."),
    "Organization": (
        "Company, institution, agency, or other actor acting in an organizational capacity."
    ),
    "AI Service Provider": (
        "Provider or operator responsible for providing or governing an AI service."
    ),
    "Algorithmic Recommendation Service Provider": (
        "Provider or operator responsible for recommendation, ranking, search-order, "
        "or algorithmic information-distribution services."
    ),
    "Deep Synthesis Service Provider": (
        "Provider or operator responsible for deep synthesis or synthetic-media services."
    ),
    "Employer": ("Actor exercising employment or workplace decision-making authority."),
    "Healthcare Provider": (
        "Actor providing healthcare services or making healthcare-related decisions."
    ),
    "Financial Service Provider": (
        "Actor providing financial, credit, lending, or other regulated financial services."
    ),
    "Education Provider": (
        "Actor providing education or exercising authority over educational decisions."
    ),
    "Insurance Provider": (
        "Actor providing insurance or making insurance-related decisions."
    ),
    "Housing Provider": ("Actor providing or controlling access to housing."),
    "Social Welfare Provider": (
        "Actor administering social welfare, benefits, or public assistance."
    ),
    "Law Enforcement Organization": (
        "Police or other organization exercising official law-enforcement authority."
    ),
    "Government Organization": (
        "Government agency, regulator, or other public authority."
    ),
}

FALLBACK_SUBJECT = "Organization or Personal User"

SPECIFIC_SUBJECT_PROFILES = SUBJECT_PROFILES.copy()
SUBJECT_VOCAB = set(SPECIFIC_SUBJECT_PROFILES)

SYSTEM_PROMPT = """
You complete policy fragments only by supplying missing Subject or governing Predicate information.

Python will construct:
<Subject> must not <PredicateClause>.

Apply this priority strictly:

1. Existing Predicate
- Identify all governing actions already expressed in the original policy, including coordinated actions.
- Preserve every existing governing action and its wording.
- Do not replace, paraphrase, or reorder an existing action.
- Convert an inflected governing verb to its base form only when grammatically required after "must not".
- Remove infinitival "to" from a governing verb only when grammatically required after "must not".
- Do not treat verbs inside objects, relative clauses, modifiers, purposes, methods, conditions, or consequences as governing actions.

2. Subject
- Preserve an explicit regulated actor.
- Infer a specific Subject Profile from explicit actor wording or from role-specific context that clearly identifies the regulated actor.
- Use role-specific context only when it strongly and uniquely corresponds to a Subject Profile.
- Return Organization alone only when the regulated actor is explicitly identified as an organizational actor.
- Return Personal User alone only when the regulated actor is explicitly identified as an individual person or user.
- Treat "you" or "yourself" referring to the regulated actor as explicit evidence for Personal User.
- If the original policy does not distinguish between Organization and Personal User and does not clearly identify another Subject Profile, return:
  ["Organization", "Personal User"].
- Do not infer the Subject merely from the prohibited action, object, content type, general policy topic, source organization, or an inferred Predicate.

3. Missing Predicate
- Infer a governing Predicate only where no usable governing action exists.
- If both Subject and Predicate are missing and the Object is communicative content, such as content, information, material, remarks, or messages, use only:
  produce, copy, publish, disseminate.
- Use every canonical action that naturally applies.
- Do not substitute alternative inferred verbs such as include, contain, provide, generate, or distribute.
- Do not apply these canonical actions to conduct or behavior.
- If an existing noun phrase or nominalized conduct has no governing verb, preserve that phrase verbatim and add only the minimum governing construction needed, such as "engage in".
- Never replace an existing verb with a fallback verb.

4. Preservation
- Preserve all existing substantive policy wording and word order.
- Do not paraphrase, synonymize, simplify, polish, or stylistically rewrite existing text.
- Do not move individual words, adverbs, predicates, objects, modifiers, conditions, exceptions, or method phrases.
- Grammatical normalization of a governing verb must occur in its original position.

PredicateClause may begin with an existing modifier, condition, exception, or method phrase.
The governing Predicate itself must use the base form required after "must not".
PredicateClause must not contain the Subject or an added clause-level prohibition, obligation, permission, or modality operator.
Return JSON only.
""".strip()

INDEPENDENT_NORMATIVE_RE = re.compile(
    r"^.+?\b(?:must\s+not|shall\s+not|may\s+not|should\s+not|"
    r"must\s+never|should\s+never)\b",
    re.I,
)


def _split_independent_normative_sentences(text: str) -> List[str]:
    """Split only later sentences that carry their own normative operator."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())

    if len(parts) <= 1:
        return [text.strip()] if text.strip() else []

    groups = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if groups and not INDEPENDENT_NORMATIVE_RE.match(part):
            groups[-1] = f"{groups[-1]} {part}"
        else:
            groups.append(part)

    return groups


def _merge_colon_continuation_lines(items: List[str]) -> List[str]:
    """Attach colon-introduced continuation lines to their governing policy."""
    merged = []
    index = 0

    while index < len(items):
        current = items[index].strip()

        if not current:
            index += 1
            continue

        if current.endswith(":") and index + 1 < len(items):
            continuations = []
            index += 1

            while index < len(items):
                continuation = items[index].strip()
                if not continuation:
                    index += 1
                    continue

                continuations.append(continuation.rstrip(";"))
                index += 1

                if not continuation.endswith(";"):
                    break

            current = f"{current} {'; '.join(continuations)}"
            merged.append(current)
            continue

        merged.append(current)
        index += 1

    return merged


def _normalize_policies(rules: Union[str, List[str]]) -> List[str]:
    if isinstance(rules, str):
        rules = rules.splitlines()

    if not isinstance(rules, list):
        raise TypeError("rules must be a multiline string or List[str].")

    rules = _merge_colon_continuation_lines(rules)

    return [
        sentence
        for item in rules
        if isinstance(item, str) and item.strip()
        for sentence in _split_independent_normative_sentences(item)
    ]


def _normalize_explicit_complete_policy(text: str):
    text = text.strip()

    match = re.match(
        r"^(.+?)\s+(must\s+not|shall\s+not|may\s+not|should\s+not)\s+(.+)$",
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    subject = _canonicalize_subject_profile(match.group(1).strip())
    predicate_clause = match.group(3).strip()
    predicate_clause = re.sub(
        r",\s*nor\s+should\s+(?:it|they)\s+",
        ", nor ",
        predicate_clause,
        flags=re.I,
    )

    if not subject or not predicate_clause:
        return None

    predicate_clause = predicate_clause.rstrip(". ")

    if not predicate_clause:
        return None

    return f"{subject} must not {predicate_clause}."


def _find_source_text(sentence: str, value):
    if not value:
        return None

    value = str(value).strip()

    if not value:
        return None

    match = re.search(re.escape(value), sentence, re.IGNORECASE)

    if not match:
        return None

    return sentence[match.start() : match.end()]


def _canonicalize_subject_profile(value: str) -> str:
    """Restore the configured spelling of known subject profiles."""
    if not isinstance(value, str):
        return value

    parts = re.split(r"\s+or\s+", value.strip(), flags=re.IGNORECASE)
    canonical = {profile.casefold(): profile for profile in SUBJECT_PROFILES}

    if parts and all(part.casefold() in canonical for part in parts):
        return " or ".join(canonical[part.casefold()] for part in parts)

    return value.strip()


def _find_leading_subject(sentence: str, value: str):
    """Match an explicit actor only at the policy start and on a word boundary."""
    if not isinstance(value, str) or not value.strip():
        return None

    match = re.match(
        rf"^\s*({re.escape(value.strip())})(?=\s|[,;:()\[\]{{}}]|$)",
        sentence,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _split_subjects(value) -> List[str]:
    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    subjects = []

    for item in value:
        if not isinstance(item, str):
            continue

        subjects.extend(
            part.strip()
            for part in re.split(r"\s+or\s+", item, flags=re.IGNORECASE)
            if part.strip()
        )

    return subjects


def _generic_subject() -> str:
    return FALLBACK_SUBJECT


def _resolve_subject(sentence: str, value) -> str:
    subjects = _split_subjects(value)

    for subject in subjects:
        explicit_subject = _find_leading_subject(sentence, subject)

        if explicit_subject:
            return _canonicalize_subject_profile(subject)

    valid_subjects = []

    for subject in subjects:
        canonical_subject = _canonicalize_subject_profile(subject)

        if (
            canonical_subject in SUBJECT_VOCAB
            and canonical_subject not in valid_subjects
        ):
            valid_subjects.append(canonical_subject)

    if valid_subjects:
        return " or ".join(valid_subjects[:2])

    return _generic_subject()


def _normalize_predicate_clause(value) -> str:
    if not isinstance(value, str):
        return ""

    clause = value.strip()

    if not clause:
        return ""

    clause = re.sub(
        r"^\s*(?:do\s+not|don['’]t|do\s+n['’]t|must\s+not|shall\s+not|may\s+not|never)\b\s*",
        "",
        clause,
        flags=re.IGNORECASE,
    )

    clause = re.sub(
        r"^\s*(?:the\s+)?prohibition\s+(?:of|on)\s+",
        "",
        clause,
        flags=re.IGNORECASE,
    )

    return clause.strip().rstrip(". ")


def _build_complete_policy(subject: str, predicate_clause: str) -> str:
    return f"{subject} must not {predicate_clause}."


def _build_prompt(items: List[Dict], organization: str) -> str:
    return f"""
Subject Profiles:
{json.dumps(SPECIFIC_SUBJECT_PROFILES, ensure_ascii=False, indent=2)}

Input:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return exactly:

{{
  "results": [
    {{
      "id": 0,
      "Subject": ["<Subject Profile>"],
      "PredicateClause": "<completed PredicateClause>"
    }}
  ]
}}

Output requirements:
1. Return exactly one result for every input item.
2. Preserve every input id.
3. Subject must be a list containing only Subject Profiles allowed by the system prompt.
4. PredicateClause must not contain the Subject or an added clause-level prohibition, obligation, permission, or modality operator.
5. Preserve all existing substantive wording and word order; only normalize a governing verb where grammatically required after "must not".
6. Follow the reasoning and preservation rules in the system prompt.
7. Output JSON only.
""".strip()


def _parse_results(response: str, expected_ids):
    parsed = parse_json_response(response)
    results = parsed.get("results") if isinstance(parsed, dict) else None

    if not isinstance(results, list):
        raise ValueError(
            "Invalid CompletePolicySentence output: 'results' must be a list."
        )

    item_map = {}
    duplicates = set()

    for item in results:
        if not isinstance(item, dict):
            continue

        item_id = item.get("id")

        if item_id not in expected_ids:
            continue

        if item_id in item_map:
            duplicates.add(item_id)
            continue

        predicate_clause = _normalize_predicate_clause(item.get("PredicateClause"))

        if not predicate_clause:
            continue

        item_map[item_id] = {
            "Subject": item.get("Subject", []),
            "PredicateClause": predicate_clause,
        }

    for item_id in duplicates:
        item_map.pop(item_id, None)

    failed_ids = sorted(expected_ids - set(item_map))

    return item_map, failed_ids


def complete_policy_batch(
    llm_client: LLMClient,
    rules: Union[str, List[str]],
    organization: str,
    batch_size: int = 10,
    max_retries: int = 5,
) -> List[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")

    policies = _normalize_policies(rules)

    if not policies:
        logger.info(
            "CompletePolicySentence | completed | policies=0 | llm_calls=0 | total_tokens=0"
        )
        return []

    usage_before = llm_client.get_usage()

    output_rules = policies.copy()
    rewrite_ids = []

    for index, policy in enumerate(policies):
        normalized = _normalize_explicit_complete_policy(policy)

        if normalized is not None:
            output_rules[index] = normalized
        else:
            rewrite_ids.append(index)

    total_batches = (
        (len(rewrite_ids) + batch_size - 1) // batch_size if rewrite_ids else 0
    )

    retry_count = 0
    failed_batch_count = 0

    logger.info(
        "CompletePolicySentence | start | policies=%d | rewrite=%d | batches=%d | batch_size=%d",
        len(policies),
        len(rewrite_ids),
        total_batches,
        batch_size,
    )

    for start in range(0, len(rewrite_ids), batch_size):
        batch_ids = rewrite_ids[start : start + batch_size]
        batch_num = start // batch_size + 1
        batch_usage_before = llm_client.get_usage()

        rewrites = {}
        remaining_ids = batch_ids.copy()

        for attempt in range(1, max_retries + 1):
            if not remaining_ids:
                break

            items = [
                {
                    "id": index,
                    "text": policies[index],
                }
                for index in remaining_ids
            ]

            try:
                response = llm_client.invoke(
                    _build_prompt(
                        items=items,
                        organization=organization,
                    ),
                    system_prompt=SYSTEM_PROMPT,
                )

                if not response:
                    raise ValueError("Empty LLM response.")

                resolved, failed_ids = _parse_results(
                    response=response,
                    expected_ids=set(remaining_ids),
                )

                for item_id, item in resolved.items():
                    subject = _resolve_subject(
                        sentence=policies[item_id],
                        value=item.get("Subject", []),
                    )

                    completed_policy = _build_complete_policy(
                        subject=subject,
                        predicate_clause=item["PredicateClause"],
                    )

                    rewrites[item_id] = completed_policy

                remaining_ids = failed_ids

                if failed_ids and attempt < max_retries:
                    retry_count += 1
                    logger.warning(
                        "CompletePolicySentence | batch=%d/%d | attempt=%d/%d failed | retry_items=%d",
                        batch_num,
                        total_batches,
                        attempt,
                        max_retries,
                        len(failed_ids),
                    )

            except Exception as exc:
                if attempt < max_retries:
                    retry_count += 1

                logger.warning(
                    "CompletePolicySentence | batch=%d/%d | attempt=%d/%d failed | error=%s",
                    batch_num,
                    total_batches,
                    attempt,
                    max_retries,
                    exc,
                )

        for index, text in rewrites.items():
            output_rules[index] = text

        batch_usage = llm_client.get_usage_since(batch_usage_before)

        if remaining_ids:
            failed_batch_count += 1
            logger.error(
                "CompletePolicySentence | batch=%d/%d failed | items=%d | llm_calls=%d | fallback=keep_original",
                batch_num,
                total_batches,
                len(remaining_ids),
                batch_usage["api_calls"],
            )
            continue

        logger.info(
            "CompletePolicySentence | batch=%d/%d | items=%d | llm_calls=%d",
            batch_num,
            total_batches,
            len(batch_ids),
            batch_usage["api_calls"],
        )

    usage = llm_client.get_usage_since(usage_before)

    logger.info(
        "CompletePolicySentence | completed | input=%d | output=%d | rewritten=%d | batches=%d | llm_calls=%d | retries=%d | failed_batches=%d | input_tokens=%d | output_tokens=%d | total_tokens=%d",
        len(policies),
        len(output_rules),
        len(rewrite_ids),
        total_batches,
        usage["api_calls"],
        retry_count,
        failed_batch_count,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
    )

    return output_rules
