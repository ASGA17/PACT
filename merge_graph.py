import logging
from typing import Dict, List, Optional

from llms import LLMClient
from policy_loader import load_regulation_text
from utils.global_tool import get_llm_client, load_save
from utils.logging_utils import setup_logging
from utils.merge_graph.policy_graph import build_policy_graph_batch
from utils.merge_graph.enrich_policy_graph import enrich_policy_graph_batch
from utils.merge_graph.global_graph import (
    build_global_graph_batch,
    build_global_graph_from_files,
)

from utils.utils import parse_json_response

# Shared batch and retry settings.
BATCH_SIZE = 10
MAX_RETRIES = 5

GLOBAL_BATCH_SIZE = 20
GLOBAL_MAX_RETRIES = 3

logger = logging.getLogger(__name__)


# Bind role graphs from clauses and examples into frame graphs.
def merge_graph(
    llm_client: LLMClient,
    rules: List[Dict],
    organization: str,
) -> List[Dict]:
    rules = build_policy_graph_batch(
        llm_client=llm_client,
        rules=rules,
        organization=organization,
        batch_size=BATCH_SIZE,
        max_retries=MAX_RETRIES,
    )

    load_save(rules, f"./output/{organization}/graph_policy.txt")

    return rules


# Resolve cross-node references and add instantiation and distinctness constraints.
def enrich_graph(
    llm_client: LLMClient, rules: List[Dict], organization: str
) -> List[Dict]:
    rules = enrich_policy_graph_batch(
        llm_client=llm_client,
        rules=rules,
        organization=organization,
        batch_size=BATCH_SIZE,
        max_retries=MAX_RETRIES,
    )

    load_save(
        rules,
        f"./output/{organization}/enriched_graph_policy.txt",
    )

    return rules


# Merge graphs held in memory.
def merge_global_graph(
    llm_client: Optional[LLMClient],
    sources: List[Dict],
    output_path: str = "./output/global/global_graph.txt",
) -> Dict:
    """Fuse multiple already-enriched rule lists into one Global Graph.

    sources example:
        [
            {
                "source_id": "baidu_terms",
                "organization": "Baidu",
                "rules": baidu_enriched_rules,
            },
            {
                "source_id": "openai_usage",
                "organization": "OpenAI",
                "rules": openai_enriched_rules,
            },
        ]

    `source_id` identifies a file/dataset and must be unique.
    `organization` identifies the policy issuer; multiple source_ids may share it.
    """
    graph = build_global_graph_batch(
        llm_client=llm_client,
        sources=sources,
        batch_size=GLOBAL_BATCH_SIZE,
        max_retries=GLOBAL_MAX_RETRIES,
        strict_validation=True,
    )
    load_save(graph, output_path)
    return graph


# Merge graphs loaded from files.
def merge_global_graph_from_files(
    llm_client: Optional[LLMClient],
    inputs: List[Dict],
    output_path: str = "./output/global/global_graph.txt",
) -> Dict:
    """Fuse multiple enriched_graph_policy files.

    inputs example:
        [
            {
                "source_id": "baidu_terms",
                "organization": "Baidu",
                "path": "./output/Baidu/enriched_graph_policy.txt",
            },
            {
                "source_id": "openai_usage",
                "organization": "OpenAI",
                "path": "./output/OpenAI/enriched_graph_policy.txt",
            },
        ]
    """
    graph = build_global_graph_from_files(
        llm_client=llm_client,
        inputs=inputs,
        batch_size=GLOBAL_BATCH_SIZE,
        max_retries=GLOBAL_MAX_RETRIES,
        strict_validation=True,
    )
    load_save(graph, output_path)
    return graph


if __name__ == "__main__":
    setup_logging(task_name="graph_merge")

    organization = "Baidu"
    llm_client = get_llm_client("deepseek-v4-flash")

    # Run policy and frame graph assembly.
    # Input records must contain parsed graphs for all four roles.
    # regulation_text = load_regulation_text(f"./output/{organization}/policy_condition_groups.txt")
    # rules = parse_json_response(regulation_text)
    # result = merge_graph(llm_client=llm_client,rules=rules,organization=organization)

    # regulation_text = load_regulation_text(f"./output/{organization}/graph_policy.txt")
    # rules = parse_json_response(regulation_text)
    # result = enrich_graph(llm_client=llm_client,rules=rules,organization=organization)

    inputs = [
        {
            "source_id": "baidu_terms",
            "organization": "Baidu",
            "path": "./output/Baidu/enriched_graph_policy.txt",
        }
    ]
    graph = merge_global_graph_from_files(
        llm_client=llm_client,
        inputs=inputs,
        output_path="./output/global/global_graph.txt",
    )
