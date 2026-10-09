import json
import logging
import re
from typing import Dict, List

from utils.utils import parse_json_response

logger = logging.getLogger(__name__)

CONTENT_COVERAGE_IGNORED = {
    "a",
    "an",
    "and",
    "any",
    "as",
    "but",
    "by",
    "for",
    "from",
    "in",
    "including",
    "limited",
    "not",
    "of",
    "on",
    "or",
    "such",
    "the",
    "their",
    "to",
    "with",
}

SYSTEM_PROMPT = """
You are a policy semantic parser.

Extract Predicate, Object, and Condition from each sentence.
Subject is handled separately by Python and must not be inferred.
Return JSON only.
""".strip()

EXAMPLE_SYSTEM_PROMPT = """
You are a policy example semantic parser.

Structure only the supplied Examples mapping according to the rules in the prompt.
Return JSON only.
""".strip()


def _extract_explicit_subject(sentence: str) -> str:
    match = re.match(
        r"^\s*(.+?)\s+(?:must\s+not|shall\s+not|may\s+not|should\s+not)\s+",
        sentence,
        re.I,
    )

    if not match:
        return ""

    return match.group(1).strip()


def _find_source_text(sentence: str, value):
    if not value:
        return None

    value = str(value).strip()

    if not value:
        return None

    pattern = re.escape(value)

    # Prevent a value such as "Use" from matching inside "User".
    if re.match(r"[A-Za-z0-9]", value[0]):
        pattern = r"(?<![A-Za-z0-9_’'])" + pattern

    if re.match(r"[A-Za-z0-9]", value[-1]):
        pattern = pattern + r"(?![A-Za-z0-9_’'])"

    match = re.search(pattern, sentence, re.I)
    return sentence[match.start() : match.end()] if match else None


def _normalize_match_text(value: str) -> str:
    """Normalize harmless spacing/case differences for whole-span checks."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _restore_subsequence_case(value: str, source: str):
    if not isinstance(value, str) or not value.strip():
        return None

    candidate_matches = list(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", value))
    source_matches = list(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", source))

    if not candidate_matches:
        return None

    source_index = 0
    replacements = []

    for candidate_match in candidate_matches:
        token = candidate_match.group().lower()

        while (
            source_index < len(source_matches)
            and source_matches[source_index].group().lower() != token
        ):
            source_index += 1

        if source_index >= len(source_matches):
            return None

        replacements.append(
            (
                candidate_match.start(),
                candidate_match.end(),
                source_matches[source_index].group(),
            )
        )
        source_index += 1

    result = value

    for start, end, replacement in reversed(replacements):
        result = result[:start] + replacement + result[end:]

    return result


def _find_grounded_text(sentence: str, value):
    exact = _find_source_text(sentence, value)

    if exact is not None:
        return exact

    if not isinstance(value, str) or not value.strip():
        return None

    return _restore_subsequence_case(value.strip(), sentence)


def _strip_terminal_punctuation(text: str) -> str:
    if not isinstance(text, str):
        return text

    return text.rstrip().rstrip(".!?")


def _element_placeholder(sentence: str) -> Dict:
    return {
        "Sentence": sentence,
        "Subject": "",
        "Predicate": "",
        "Object": "",
        "Condition": [],
    }


def _sync_schema(rule: Dict):
    # Remove obsolete logical fields if they are present in upstream/intermediate data.
    for key in list(rule):
        if "Logical" in key:
            rule.pop(key, None)

    examples_mapping = rule.get("Examples")
    if not isinstance(examples_mapping, dict):
        examples_mapping = {}
    rule["Examples"] = examples_mapping

    main = rule.get("Main_Clause")

    if isinstance(main, str) and main.strip():
        main = main.strip()
        rule["Main_Clause"] = main
        rule.setdefault(
            "Main_Clause_Elements",
            _element_placeholder(main),
        )
    else:
        rule.setdefault("Main_Clause_Elements", {})

    examples = rule.get("Example_Sentences", [])

    if not isinstance(examples, list):
        examples = []

    examples = [
        sentence.strip()
        for sentence in examples
        if isinstance(sentence, str) and sentence.strip()
    ]

    rule["Example_Sentences"] = examples

    element_slots = rule.get("Example_Sentences_Elements")

    if (
        not isinstance(element_slots, list)
        or len(element_slots) != len(examples)
        or any(
            not isinstance(item, dict) or item.get("Sentence") != sentence
            for item, sentence in zip(element_slots, examples)
        )
    ):
        rule["Example_Sentences_Elements"] = [
            _element_placeholder(sentence) for sentence in examples
        ]
    else:
        rule["Example_Sentences_Elements"] = element_slots

    if not isinstance(rule.get("Examples_Elements"), list):
        rule["Examples_Elements"] = []


def _reorder_output_fields(rule: Dict) -> Dict:
    element_keys = {
        "Main_Clause_Elements",
        "Example_Sentences_Elements",
        "Examples_Elements",
    }
    reordered = {
        key: value
        for key, value in rule.items()
        if key not in element_keys and "Logical" not in key
    }

    reordered["Main_Clause_Elements"] = rule.get("Main_Clause_Elements", {})
    reordered["Example_Sentences_Elements"] = rule.get("Example_Sentences_Elements", [])
    reordered["Examples_Elements"] = rule.get("Examples_Elements", [])

    return reordered


def _build_examples_prompt(items: List[Dict]) -> str:
    return f"""
Input:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return exactly:
{{
  "items": [
    {{
      "id": 0,
      "ExampleElements": [
        {{
          "Example": "example text",
          "AnchorType": "Object",
          "AnchorText": "anchor text"
        }}
      ]
    }}
  ]
}}

Rules:
1. Locate each Examples key in Subject, Predicate, Object, or Condition.
   AnchorType must be one of those four fields. AnchorText must be the smallest
   exact phrase that names the illustrated concept. Do not include surrounding
   determiners, prepositions, or reference words unless they are necessary to
   identify the concept.
2. Split a value into atomic examples only when it clearly enumerates distinct alternatives.
3. Preserve modifier scope across coordination. When adjacent modifiers form one
   modifier phrase that applies to the whole coordination, propagate the complete
   phrase together; do not propagate only part of it. Do not propagate prepositions,
   determiners, quantifiers, or other structural words merely because they precede
   the first item.
4. Remove only example-introducing framing that is not part of the example itself.
   Preserve substantive wording and capitalization.
5. Reconstruct an atomic example only from words in that same input value; never
   copy wording from another input item.
6. Preserve every substantive example or alternative. Do not silently omit a
   parenthetical, final coordinated item, or nested example.
7. Preserve every input id and return no extra fields.

Example:
- value: "qualified limited A, B, and C"
  -> qualified limited A / qualified limited B / qualified limited C

Return JSON only.
""".strip()


STRUCTURAL_SCOPE_WORDS = {
    "a",
    "an",
    "the",
    "any",
    "all",
    "each",
    "every",
    "either",
    "neither",
    "some",
    "through",
    "by",
    "with",
    "without",
    "via",
    "from",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "under",
    "over",
    "into",
    "onto",
    "against",
    "among",
    "between",
}


def _word_matches(text: str):
    return list(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text))


def _longest_common_prefix(token_lists: List[List[str]]) -> List[str]:
    if not token_lists:
        return []

    prefix = token_lists[0][:]

    for tokens in token_lists[1:]:
        size = min(len(prefix), len(tokens))
        index = 0

        while index < size and prefix[index].lower() == tokens[index].lower():
            index += 1

        prefix = prefix[:index]

        if not prefix:
            break

    return prefix


def _ground_against_any_source(value: str, source_values: List[str]):
    for source_value in source_values:
        grounded = _restore_subsequence_case(value, source_value)
        if grounded is not None:
            return grounded

    return None


def _repair_shared_modifier_phrases(
    results: List[Dict], source_values: List[str]
) -> List[Dict]:
    """Repair partial propagation of a multi-word shared modifier phrase.

    This is intentionally conservative: it only repairs a group with at least
    three examples, when later examples already share a common prefix and the
    first example contains additional non-structural words immediately before
    that shared prefix. The repaired text must still be grounded in the same
    Examples value.
    """
    groups = {}

    for index, item in enumerate(results):
        key = (item.get("AnchorType"), item.get("AnchorText"))
        groups.setdefault(key, []).append(index)

    for indices in groups.values():
        if len(indices) < 3:
            continue

        examples = [results[index].get("Example", "") for index in indices]
        token_matches = [_word_matches(example) for example in examples]
        token_lists = [
            [match.group() for match in matches] for matches in token_matches
        ]

        if any(not tokens for tokens in token_lists):
            continue

        shared_prefix = _longest_common_prefix(token_lists[1:])

        if not shared_prefix:
            continue

        first_tokens = token_lists[0]
        prefix_len = len(shared_prefix)
        shared_start = None

        for start in range(1, len(first_tokens) - prefix_len + 1):
            segment = first_tokens[start : start + prefix_len]

            if [token.lower() for token in segment] == [
                token.lower() for token in shared_prefix
            ]:
                shared_start = start
                break

        if shared_start is None:
            continue

        extra_tokens = first_tokens[:shared_start]

        if not extra_tokens or any(
            token.lower() in STRUCTURAL_SCOPE_WORDS for token in extra_tokens
        ):
            continue

        first_matches = token_matches[0]
        extra_text = examples[0][
            first_matches[0].start() : first_matches[shared_start].start()
        ].strip()

        if not extra_text:
            continue

        full_prefix_candidate = f"{extra_text} {' '.join(shared_prefix)}"

        if _ground_against_any_source(full_prefix_candidate, source_values) is None:
            continue

        for index in indices[1:]:
            example = results[index].get("Example", "")
            matches = _word_matches(example)
            tokens = [match.group() for match in matches]

            if len(tokens) < prefix_len:
                continue

            if [token.lower() for token in tokens[:prefix_len]] != [
                token.lower() for token in shared_prefix
            ]:
                continue

            candidate = f"{extra_text} {example}"
            grounded = _ground_against_any_source(candidate, source_values)

            if grounded is not None:
                results[index]["Example"] = _strip_terminal_punctuation(grounded)

    return results


def _repair_shared_anchor_head(
    results: List[Dict], source_values: List[str]
) -> List[Dict]:
    """Propagate an omitted shared head noun such as ``privacy rights``."""
    groups = {}

    for index, item in enumerate(results):
        key = (item.get("AnchorType"), item.get("AnchorText"))
        groups.setdefault(key, []).append(index)

    for (_, anchor_text), indices in groups.items():
        if len(indices) < 2 or not isinstance(anchor_text, str):
            continue

        anchor_tokens = _content_tokens(anchor_text)
        if not anchor_tokens:
            continue

        head = anchor_tokens[0]
        endings = [
            _content_tokens(results[index].get("Example", "")) for index in indices
        ]

        if not any(tokens and tokens[-1] == head for tokens in endings):
            continue

        for index, tokens in zip(indices, endings):
            if not tokens or tokens[-1] == head:
                continue

            candidate = f'{results[index]["Example"]} {head}'
            grounded = _ground_against_any_source(candidate, source_values)

            if grounded is not None:
                results[index]["Example"] = _strip_terminal_punctuation(grounded)

    return results


def _content_tokens(text: str):
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text.lower())
        if token not in CONTENT_COVERAGE_IGNORED
    ]


def _covers_example_content(results: List[Dict], source_values: List[str]) -> bool:
    source_tokens = set(_content_tokens(" ".join(source_values)))
    result_tokens = set(
        _content_tokens(" ".join(item.get("Example", "") for item in results))
    )
    return source_tokens.issubset(result_tokens)


def _normalize_example_elements(value, examples_mapping: Dict, main_elements: Dict):
    if not isinstance(value, list) or not isinstance(examples_mapping, dict):
        return None

    source_values = [
        item
        for values in examples_mapping.values()
        if isinstance(values, list)
        for item in values
        if isinstance(item, str) and item.strip()
    ]

    if not source_values:
        return []

    results = []
    allowed = {"Subject", "Predicate", "Object", "Condition"}

    for item in value:
        if not isinstance(item, dict):
            return None

        example = item.get("Example")
        anchor_type = item.get("AnchorType")
        anchor_text = item.get("AnchorText")

        if (
            not isinstance(example, str)
            or not example.strip()
            or anchor_type not in allowed
            or not isinstance(anchor_text, str)
            or not anchor_text.strip()
        ):
            return None

        grounded_example = None

        for source_value in source_values:
            grounded_example = _restore_subsequence_case(
                example.strip(),
                source_value,
            )
            if grounded_example is not None:
                break

        if grounded_example is None:
            return None

        results.append(
            {
                "Example": _strip_terminal_punctuation(grounded_example),
                "AnchorType": anchor_type,
                "AnchorText": _strip_terminal_punctuation(anchor_text.strip()),
            }
        )

    results = _repair_shared_modifier_phrases(results, source_values)
    results = _repair_shared_anchor_head(results, source_values)

    if not _covers_example_content(results, source_values):
        return None

    for item in results:
        anchor_value = main_elements.get(item["AnchorType"], "")

        if isinstance(anchor_value, list):
            anchor_value = " ".join(
                value for value in anchor_value if isinstance(value, str)
            )

        if (
            not isinstance(anchor_value, str)
            or _find_source_text(
                anchor_value,
                item["AnchorText"],
            )
            is None
        ):
            return None

    return results


def extract_examples_for_rules(
    llm_client,
    rules: List[Dict],
    batch_size: int = 4,
    max_retries: int = 5,
) -> List[Dict]:
    tasks = []

    for rule_index, rule in enumerate(rules):
        examples = rule.get("Examples", {})

        if not isinstance(examples, dict) or not examples:
            rule["Examples_Elements"] = []
            continue

        elements = rule.get("Main_Clause_Elements")
        if not isinstance(elements, dict):
            rule["Examples_Elements"] = []
            continue

        tasks.append(
            {
                "id": rule_index,
                "Examples": examples,
                "Main_Clause_Elements": elements,
            }
        )

    if not tasks:
        return rules

    total_batches = (len(tasks) + batch_size - 1) // batch_size

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        batch_num = start // batch_size + 1
        remaining = batch
        usage_before = llm_client.get_usage()

        for attempt in range(max_retries):
            if not remaining:
                break

            try:
                response = llm_client.invoke(
                    _build_examples_prompt(remaining),
                    system_prompt=EXAMPLE_SYSTEM_PROMPT,
                )

                parsed = parse_json_response(response)
                items = parsed.get("items") if isinstance(parsed, dict) else None

                if not isinstance(items, list):
                    raise ValueError("Invalid Examples output")

                item_map = {
                    item.get("id"): item
                    for item in items
                    if isinstance(item, dict) and isinstance(item.get("id"), int)
                }

                failed = []

                for source in remaining:
                    item = item_map.get(source["id"])

                    if not isinstance(item, dict):
                        failed.append(source)
                        continue

                    normalized = _normalize_example_elements(
                        item.get("ExampleElements"),
                        source["Examples"],
                        source["Main_Clause_Elements"],
                    )

                    if normalized is None:
                        failed.append(source)
                        continue

                    rules[source["id"]]["Examples_Elements"] = normalized

                if failed and attempt + 1 < max_retries:
                    logger.warning(
                        "ExtractElements | examples batch=%d/%d | "
                        "attempt=%d/%d failed | retry_items=%d",
                        batch_num,
                        total_batches,
                        attempt + 1,
                        max_retries,
                        len(failed),
                    )

                remaining = failed

            except Exception as e:
                logger.warning(
                    "ExtractElements | examples batch=%d/%d | "
                    "attempt=%d/%d failed | error=%s",
                    batch_num,
                    total_batches,
                    attempt + 1,
                    max_retries,
                    e,
                )

        usage = llm_client.get_usage_since(usage_before)

        if not remaining:
            logger.info(
                "ExtractElements | examples batch=%d/%d | items=%d | llm_calls=%d",
                batch_num,
                total_batches,
                len(batch),
                usage["api_calls"],
            )
        else:
            logger.error(
                "ExtractElements | examples batch=%d/%d failed | "
                "items=%d | llm_calls=%d",
                batch_num,
                total_batches,
                len(remaining),
                usage["api_calls"],
            )

            for source in remaining:
                rules[source["id"]]["Examples_Elements"] = []

    return rules


def _build_prompt(sentences: List[str], organization: str) -> str:
    items = [
        {
            "id": i,
            "sentence": sentence,
        }
        for i, sentence in enumerate(sentences)
    ]

    return f"""
Policies:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return exactly:
{{
  "items": [
    {{
      "id": 0,
      "Predicate": "A",
      "Object": "B",
      "Condition": ["C"]
    }}
  ]
}}

Rules:
1. Predicate
- Extract the governed action phrase, including required particles and prepositions.
  Keep an infinitival complement inside Predicate only when it directly completes
  the governing action before its Object. If the governing action already has an
  explicit Object before the infinitival phrase, do not absorb that Object or the
  following infinitival phrase into Predicate; classify the infinitival phrase
  according to its attachment.

  A to B C
  -> Predicate: A to B
  -> Object: C

  A B to C D
  -> Predicate: A
  -> Object: B
  -> Condition: to C D

- Keep coordinated main predicates together. Exclude clause-level policy operators
  and do not replace the governed action with a purpose, method, restriction, or consequence.
- Predicate must not be empty when an explicit governed action exists.
- Cover every governing action, especially actions separated by semicolons. Never
  select only the final action from a coordinated policy.
- For an action-like example fragment, Predicate must still identify a grammatical
  action. Use the source action wording; normalization to a base verb is handled
  after extraction.
- If an example fragment only names activities, categories, techniques, or other
  noun phrases (for example, "fraud, scams, or other deceptive practices"), return
  null for Predicate. Its governing action will be inherited from the parent clause.
- Never copy the whole governed clause into Predicate merely to preserve coverage.
  Put each immediate target in Object and each modifier in Condition.

2. Object and Condition
- Object is the immediate target of Predicate. Keep modifiers that structurally
  define or restrict the Object, including relative, reduced-relative, and participial modifiers.
- Condition contains modifiers of Predicate or the whole clause, such as restrictions,
  purposes, exceptions, methods, applicability, scope, circumstances, or consequences.
- Classify by attachment, not by keywords: if a phrase modifies the Object noun phrase,
  keep it in Object; if it modifies Predicate or the whole clause, place it in Condition.
- Preserve shared or coordinated object text when coordinated predicates take different objects.
  Return null when there is no explicit Object. Split Conditions only when they express
  different semantic dimensions.

3. Preservation
- Copy Object and Condition text with exact source wording and capitalization.
- Predicate must be a grammatical action verb or verb phrase. Use its base verb
  form for an action-like example fragment. Convert a nominalized action head
  when necessary (for example, "generation of" -> "generate" and
  "creation or promotion of" -> "create or promote"). Never use a bare action
  noun such as "generation", "creation", "promotion", or "propagation" as Predicate.
- Preserve every input id.

Return JSON only.
""".strip()


EXAMPLE_PREDICATE_CANONICAL_FORMS = {
    "propagation of": "propagate",
    "generation of": "generate",
    "creation of": "create",
    "promotion of": "promote",
    "creation or promotion": "create or promote",
    "creation or promotion of": "create or promote",
    "catfishing": "catfish",
    "phishing": "phish",
    "instructions for how to conceal": "conceal",
    "spoofing and social engineering": "spoof or socially engineer",
    "advocacy or direct communication with": "advocate or communicate directly with",
    "jailbreaking or prompt injection": "jailbreak or inject",
    "dissemination of": "disseminate",
}


def _canonicalize_example_predicate(predicate: str) -> str:
    """Turn known action nominalizations into usable predicate verbs.

    The mapping is deliberately conservative. It avoids a general suffix-based
    lemmatizer, which can silently corrupt valid policy terms.
    """
    if not isinstance(predicate, str):
        return predicate

    stripped = predicate.strip()
    return EXAMPLE_PREDICATE_CANONICAL_FORMS.get(stripped.casefold(), stripped)


def _has_infinitive_object_boundary_issue(
    sentence: str, predicate: str, obj: str
) -> bool:
    """Flag a likely Predicate/Object split where Object begins with an infinitive.

    The check is intentionally narrow. It only fires when the extracted Object
    begins with "to ..." and the source contains Predicate immediately followed
    by that Object, indicating that the model may have cut a governed action
    phrase too early.
    """
    if not predicate or not obj:
        return False

    if not re.match(r"^to\s+[A-Za-z][A-Za-z0-9'’_-]*\b", obj.strip(), re.I):
        return False

    combined = f"{predicate.strip()} {obj.strip()}"
    return _find_source_text(sentence, combined) is not None


def _has_material_coverage_gap(sentence: str, normalized: Dict) -> bool:
    """Reject parses that silently discard most of a coordinated policy."""
    modal = re.search(r"\b(?:must|shall|may|should)\s+not\b", sentence, re.I)
    governed_text = sentence[modal.end() :] if modal else sentence
    source_tokens = _content_tokens(governed_text)

    if len(source_tokens) < 8:
        return False

    extracted_text = " ".join(
        [
            normalized.get("Predicate", ""),
            normalized.get("Object", ""),
            *normalized.get("Condition", []),
        ]
    )
    extracted_tokens = set(_content_tokens(extracted_text))
    covered = sum(token in extracted_tokens for token in source_tokens)
    return covered / len(source_tokens) < 0.7


def _has_predicate_object_collapse(sentence: str, normalized: Dict) -> bool:
    """Reject a long governed span collapsed into Predicate with no Object.

    This is deliberately limited to long, coordinated spans. Short genuinely
    intransitive predicates remain valid, while failures such as ``send ... or
    allow ...`` cannot pass merely because putting the whole sentence in
    Predicate gives perfect token coverage.
    """
    predicate = normalized.get("Predicate", "")
    obj = normalized.get("Object", "")
    if not predicate or obj:
        return False

    # Passive predicates can legitimately govern complements without exposing
    # a direct Object in this schema (for example, "be interpreted as ...").
    if re.match(r"^be\b", predicate.strip(), re.I):
        return False

    modal = re.search(r"\b(?:must|shall|may|should)\s+not\b", sentence, re.I)
    governed_text = sentence[modal.end() :] if modal else sentence
    governed_text = _strip_terminal_punctuation(governed_text.strip())

    if _normalize_match_text(predicate) != _normalize_match_text(governed_text):
        return False

    if len(_content_tokens(governed_text)) < 6:
        return False

    return bool(re.search(r"[,;]|\b(?:and|or)\b", governed_text, re.I))


def _has_whole_sentence_object_collapse(sentence: str, normalized: Dict) -> bool:
    """Reject a long action example copied wholesale into Object.

    Short nominal examples can legitimately become an Object through parent
    inheritance. A long coordinated action span cannot: it must expose its own
    actions, objects, and conditions.
    """
    obj = normalized.get("Object", "")
    if not obj:
        return False
    source = _strip_terminal_punctuation(sentence.strip())
    if _normalize_match_text(obj) != _normalize_match_text(source):
        return False
    return len(_content_tokens(source)) >= 8 and bool(
        re.search(r"[,;]|\b(?:and|or)\b", source, re.I)
    )


def _is_nominal_example_object_fragment(sentence: str, normalized: Dict) -> bool:
    """Recognize known short noun-list examples that should inherit Predicate.

    Keep this conservative: gerund clauses can be real actions, so only stable
    noun-list forms observed in policy examples are normalized automatically.
    """
    if normalized.get("Object"):
        return False

    predicate = normalized.get("Predicate", "")
    if not predicate:
        candidate = sentence
    elif _normalize_match_text(predicate) == _normalize_match_text(sentence):
        candidate = predicate
    else:
        return False

    candidate = _normalize_match_text(_strip_terminal_punctuation(candidate))
    return bool(
        re.fullmatch(
            r"(?:fraud,?\s+scams?,?\s+or\s+other\s+deceptive\s+practices|"
            r"social\s+scoring\s+or\s+predictive\s+profiling|"
            r"[“\"]?model\s+scraping[”\"]?)",
            candidate,
            re.I,
        )
    )


def _normalize_item(sentence: str, item: Dict) -> Dict:
    subject = _extract_explicit_subject(sentence)

    predicate = _find_grounded_text(sentence, item.get("Predicate")) or ""
    predicate = _strip_terminal_punctuation(predicate)

    obj = _find_grounded_text(sentence, item.get("Object")) or ""
    obj = _strip_terminal_punctuation(obj)

    conditions = item.get("Condition", [])

    if not isinstance(conditions, list):
        conditions = []

    grounded_conditions = []

    for condition in conditions:
        grounded = _find_source_text(sentence, condition)

        if grounded:
            grounded_conditions.append(_strip_terminal_punctuation(grounded))

    return {
        "Sentence": sentence,
        "Subject": subject,
        "Predicate": predicate,
        "Object": obj,
        "Condition": grounded_conditions,
    }


def _fallback_item(sentence: str, preserve_governed_text: bool = False) -> Dict:
    predicate = ""

    if preserve_governed_text:
        modal = re.search(r"\b(?:must|shall|may|should)\s+not\b", sentence, re.I)

        if modal:
            predicate = _strip_terminal_punctuation(sentence[modal.end() :].strip())

    return {
        "Sentence": sentence,
        "Subject": _extract_explicit_subject(sentence),
        "Predicate": predicate,
        "Object": "",
        "Condition": [],
    }


def _inherit_example_fragment_context(clause: Dict, parent: Dict) -> Dict:
    """Ground non-sentential examples in their governing main-clause context."""
    if clause.get("Predicate") or not parent.get("Predicate"):
        return clause

    sentence = clause.get("Sentence", "").strip()
    clause["Predicate"] = parent["Predicate"]

    if re.match(r"^(?:by|through|via|with|using)\b", sentence, re.I):
        clause["Object"] = parent.get("Object", "")
        clause["Condition"] = [sentence]
    else:
        clause["Object"] = _strip_terminal_punctuation(sentence)
        clause["Condition"] = []

    return clause


def extract_sentences_batch(
    llm_client,
    sentences: List[str],
    organization: str,
    batch_size: int = 4,
    max_retries: int = 5,
    phase: str = "main",
) -> List[Dict]:

    results = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size if sentences else 0

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        batch_num = start // batch_size + 1
        usage_before = llm_client.get_usage()

        batch_results = [None] * len(batch)
        remaining = list(range(len(batch)))

        for attempt in range(max_retries):
            if not remaining:
                break

            retry_sentences = [batch[index] for index in remaining]

            try:
                response = llm_client.invoke(
                    _build_prompt(retry_sentences, organization),
                    system_prompt=SYSTEM_PROMPT,
                )

                parsed = parse_json_response(response)
                items = parsed.get("items") if isinstance(parsed, dict) else None

                if not isinstance(items, list):
                    raise ValueError("Invalid LLM output")

                item_map = {
                    item.get("id"): item
                    for item in items
                    if isinstance(item, dict) and isinstance(item.get("id"), int)
                }

                failed = []

                for local_index, batch_index in enumerate(remaining):
                    item = item_map.get(local_index)

                    if not isinstance(item, dict):
                        failed.append(batch_index)
                        continue

                    normalized = _normalize_item(
                        batch[batch_index],
                        item,
                    )

                    nominal_example_object = False
                    if phase == "example":
                        normalized["Predicate"] = _canonicalize_example_predicate(
                            normalized["Predicate"]
                        )
                        nominal_example_object = _is_nominal_example_object_fragment(
                            batch[batch_index], normalized
                        )
                        if nominal_example_object:
                            normalized["Predicate"] = ""

                    invalid = not normalized["Predicate"] and not nominal_example_object
                    invalid = invalid or (
                        not nominal_example_object
                        and _has_infinitive_object_boundary_issue(
                            batch[batch_index],
                            normalized["Predicate"],
                            normalized["Object"],
                        )
                    )
                    invalid = invalid or (
                        not nominal_example_object
                        and _has_material_coverage_gap(
                            batch[batch_index],
                            normalized,
                        )
                    )
                    invalid = invalid or _has_predicate_object_collapse(
                        batch[batch_index], normalized
                    )
                    invalid = invalid or _has_whole_sentence_object_collapse(
                        batch[batch_index], normalized
                    )

                    if invalid:
                        failed.append(batch_index)
                        continue

                    batch_results[batch_index] = normalized

                if failed and attempt + 1 < max_retries:
                    logger.warning(
                        "ExtractElements | %s batch=%d/%d | "
                        "attempt=%d/%d failed | retry_items=%d",
                        phase,
                        batch_num,
                        total_batches,
                        attempt + 1,
                        max_retries,
                        len(failed),
                    )

                remaining = failed

            except Exception as e:
                logger.warning(
                    "ExtractElements | %s batch=%d/%d | "
                    "attempt=%d/%d failed | error=%s",
                    phase,
                    batch_num,
                    total_batches,
                    attempt + 1,
                    max_retries,
                    e,
                )

        usage = llm_client.get_usage_since(usage_before)

        if remaining:
            logger.error(
                "ExtractElements | %s batch=%d/%d failed | " "items=%d | llm_calls=%d",
                phase,
                batch_num,
                total_batches,
                len(remaining),
                usage["api_calls"],
            )

            failed_sentences = [batch[index] for index in remaining]
            raise ValueError(
                "Element extraction failed after retries; refusing to write "
                f"lossy Predicate/Object fallbacks: {failed_sentences!r}"
            )
        else:
            logger.info(
                "ExtractElements | %s batch=%d/%d | items=%d | llm_calls=%d",
                phase,
                batch_num,
                total_batches,
                len(batch),
                usage["api_calls"],
            )

        results.extend(batch_results)

    return results


def extract_elements_for_rules(
    llm_client,
    rules: List[Dict],
    organization: str,
    batch_size: int = 4,
    max_retries: int = 5,
) -> List[Dict]:

    for rule in rules:
        _sync_schema(rule)

    usage_before = llm_client.get_usage() if hasattr(llm_client, "get_usage") else None

    main_tasks = []

    for rule_index, rule in enumerate(rules):
        sentence = rule.get("Main_Clause")

        if isinstance(sentence, str) and sentence.strip():
            main_tasks.append((rule_index, sentence.strip()))

    logger.info(
        "ExtractElements | start | rules=%d | main_clauses=%d | example_sentences=%d",
        len(rules),
        len(main_tasks),
        sum(len(rule.get("Example_Sentences", [])) for rule in rules),
    )

    main_clauses = extract_sentences_batch(
        llm_client=llm_client,
        sentences=[sentence for _, sentence in main_tasks],
        organization=organization,
        batch_size=batch_size,
        max_retries=max_retries,
        phase="main",
    )

    if len(main_clauses) != len(main_tasks):
        raise ValueError(
            f"Main task/result mismatch: "
            f"tasks={len(main_tasks)}, clauses={len(main_clauses)}"
        )

    for (
        rule_index,
        _,
    ), clause in zip(
        main_tasks,
        main_clauses,
    ):
        rule = rules[rule_index]
        rule["Main_Clause_Elements"] = clause

    rules = extract_examples_for_rules(
        llm_client=llm_client,
        rules=rules,
        batch_size=batch_size,
        max_retries=max_retries,
    )

    example_tasks = []

    for rule_index, rule in enumerate(rules):
        parent_elements = rule.get("Main_Clause_Elements", {})
        parent_subject = parent_elements.get("Subject", "")

        for example_index, sentence in enumerate(rule.get("Example_Sentences", [])):
            if isinstance(sentence, str) and sentence.strip():
                example_tasks.append(
                    (
                        rule_index,
                        example_index,
                        sentence.strip(),
                        parent_subject,
                        parent_elements,
                    )
                )

    example_clauses = extract_sentences_batch(
        llm_client=llm_client,
        sentences=[sentence for _, _, sentence, _, _ in example_tasks],
        organization=organization,
        batch_size=batch_size,
        max_retries=max_retries,
        phase="example",
    )

    if len(example_clauses) != len(example_tasks):
        raise ValueError(
            f"Example task/result mismatch: "
            f"tasks={len(example_tasks)}, clauses={len(example_clauses)}"
        )

    for (
        rule_index,
        example_index,
        sentence,
        parent_subject,
        parent_elements,
    ), clause in zip(
        example_tasks,
        example_clauses,
    ):
        if parent_subject:
            clause["Subject"] = parent_subject

        clause = _inherit_example_fragment_context(clause, parent_elements)

        rule = rules[rule_index]
        rule["Example_Sentences_Elements"][example_index] = clause

    if usage_before is not None:
        usage = llm_client.get_usage_since(usage_before)

        logger.info(
            "ExtractElements | completed | rules=%d | main_clauses=%d | "
            "example_sentences=%d | llm_calls=%d | "
            "input_tokens=%d | output_tokens=%d | total_tokens=%d",
            len(rules),
            len(main_tasks),
            len(example_tasks),
            usage["api_calls"],
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
        )
    else:
        logger.info(
            "ExtractElements | completed | rules=%d | main_clauses=%d | "
            "example_sentences=%d",
            len(rules),
            len(main_tasks),
            len(example_tasks),
        )

    return [_reorder_output_fields(rule) for rule in rules]
