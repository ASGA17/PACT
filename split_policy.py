import argparse
import logging
from pathlib import Path
from typing import List, Dict
from llms import LLMClient
from utils.utils import parse_json_response
from utils.split_policy.complete_policy import complete_policy_batch
from utils.split_policy.separate_policy import separate_policy_batch
from utils.split_policy.extract_elements import extract_elements_for_rules
from utils.split_policy.subject_graph import split_subject_groups_batch
from utils.split_policy.object_graph import split_object_groups_batch
from utils.split_policy.predicate_graph import split_predicate_groups_batch
from utils.split_policy.condition_graph import split_condition_groups_batch

from utils.logging_utils import setup_logging
from policy_loader import load_regulation_text

from utils.global_tool import get_llm_client, load_save

# Shared batch and retry settings.
BATCH_SIZE = 10  # Maximum rules per batch.
MAX_RETRIES = 5  # Maximum retry attempts.

logger = logging.getLogger(__name__)


# Normalize each policy into an explicit prohibition.
def complete_policy(
    llm_client: LLMClient,
    rules: List[str],
    organization: str,
) -> List[str]:
    rules = complete_policy_batch(
        llm_client=llm_client,
        rules=rules,
        organization=organization,
        batch_size=BATCH_SIZE,
        max_retries=MAX_RETRIES,
    )

    load_save(
        rules,
        f"./output/{organization}/policy_completed.txt",
    )

    return rules


# Separate normative clauses from examples.
def separate_main_examples(
    llm_client: LLMClient, rules: List[str], organization: str
) -> List[Dict]:

    rules = separate_policy_batch(
        llm_client=llm_client,
        policies=rules,
        batch_size=BATCH_SIZE,
        max_retries=MAX_RETRIES,
    )

    load_save(rules, f"./output/{organization}/policy_main_examples.txt")

    return rules


# Extract subjects, predicates, objects, and conditions.
def extract_elements(
    llm_client: LLMClient,
    rules: List[Dict],
    organization: str,
) -> List[Dict]:

    rules = extract_elements_for_rules(
        llm_client=llm_client,
        rules=rules,
        organization=organization,
        batch_size=BATCH_SIZE,
        max_retries=MAX_RETRIES,
    )

    load_save(
        rules,
        f"./output/{organization}/policy_elements.txt",
    )

    return rules


# Parse subject spans.
def split_subject_groups(llm_client: LLMClient, rules: List[Dict], organization: str):
    rules = split_subject_groups_batch(
        llm_client=llm_client,
        rules=rules,
        organization=organization,
        batch_size=10,
        max_retries=MAX_RETRIES,
    )

    load_save(rules, f"./output/{organization}/policy_subject_groups.txt")

    return rules


# Parse object spans.
def split_object_groups(llm_client: LLMClient, rules: List[Dict], organization: str):
    rules = split_object_groups_batch(
        llm_client=llm_client,
        rules=rules,
        organization=organization,
        batch_size=4,
        max_retries=MAX_RETRIES,
    )

    load_save(rules, f"./output/{organization}/policy_object_groups.txt")

    return rules


# Parse predicate spans.
def split_predicate_groups(llm_client: LLMClient, rules: List[Dict], organization: str):
    rules = split_predicate_groups_batch(
        llm_client=llm_client,
        rules=rules,
        organization=organization,
        batch_size=10,
        max_retries=MAX_RETRIES,
    )

    load_save(rules, f"./output/{organization}/policy_predicate_groups.txt")

    return rules


# Parse condition spans.
def split_condition_groups(llm_client: LLMClient, rules: List[Dict], organization: str):
    rules = split_condition_groups_batch(
        llm_client=llm_client,
        rules=rules,
        organization=organization,
        batch_size=4,
        max_retries=MAX_RETRIES,
    )

    load_save(rules, f"./output/{organization}/policy_condition_groups.txt")

    return rules


STAGES = (
    ("complete", None, complete_policy),
    ("separate", "policy_completed.txt", separate_main_examples),
    ("extract", "policy_main_examples.txt", extract_elements),
    ("subject", "policy_elements.txt", split_subject_groups),
    ("object", "policy_subject_groups.txt", split_object_groups),
    ("predicate", "policy_object_groups.txt", split_predicate_groups),
    ("condition", "policy_predicate_groups.txt", split_condition_groups),
)


def main():
    parser = argparse.ArgumentParser(
        description="Run policy splitting and save each stage under output/<organization>/."
    )
    parser.add_argument(
        "--stage", required=True, choices=[name for name, _, _ in STAGES] + ["all"]
    )
    parser.add_argument("--organization", default="Baidu")
    parser.add_argument("--policy-path", default="policy/baidu.txt", type=Path)
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args()

    if (
        not args.organization
        or Path(args.organization).name != args.organization
        or args.organization in {".", ".."}
    ):
        parser.error("--organization must be a single directory name")

    selected = (
        STAGES
        if args.stage == "all"
        else [stage for stage in STAGES if stage[0] == args.stage]
    )
    output_dir = Path("output") / args.organization
    first_input = (
        args.policy_path if selected[0][1] is None else output_dir / selected[0][1]
    )
    if not first_input.is_file():
        parser.error(f"input file not found: {first_input}")

    setup_logging(task_name="policy_extract")
    llm_client = get_llm_client(args.model)
    previous_result = None
    for name, source_name, operation in selected:
        if previous_result is not None:
            rules = previous_result
        elif source_name is None:
            rules = load_regulation_text(str(args.policy_path))
        else:
            source_path = output_dir / source_name
            rules = parse_json_response(load_regulation_text(str(source_path)))
        previous_result = operation(
            llm_client=llm_client, rules=rules, organization=args.organization
        )
        logger.info("Finished stage %s for %s", name, args.organization)


if __name__ == "__main__":
    main()
