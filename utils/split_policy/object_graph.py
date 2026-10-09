from typing import Dict, List

from llms import LLMClient
from utils.utils import parse_json_response
from utils.split_policy.relation_predicate_normalizer import (
    normalize_entity_relation_predicates_batch,
)
from utils.split_policy.layered_graph import (
    materialize_entity_graph,
    split_entity_groups_batch,
)


def materialize_object_graph(graph: Dict) -> Dict:
    return materialize_entity_graph(
        graph,
        node_prefix="o",
        entity_name="Object",
    )


def split_object_groups_batch(
    llm_client: LLMClient,
    rules: List[Dict],
    organization: str,
    batch_size: int = 10,
    max_retries: int = 2,
    max_depth: int = 12,
    max_nodes_per_object: int = 256,
) -> List[Dict]:
    rules = split_entity_groups_batch(
        llm_client,
        rules,
        organization,
        entity_name="Object",
        element_field="Object",
        output_field="Object_Elements",
        node_prefix="o",
        context_fields=("Subject", "Predicate", "Condition"),
        parse_json=parse_json_response,
        batch_size=batch_size,
        max_retries=max_retries,
        max_depth=max_depth,
        max_nodes_per_graph=max_nodes_per_object,
    )
    return normalize_entity_relation_predicates_batch(
        llm_client,
        rules,
        graph_field="Object_Elements",
        batch_size=batch_size,
        max_retries=max_retries,
    )
