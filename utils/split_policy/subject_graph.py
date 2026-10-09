from typing import Dict, List

from llms import LLMClient
from utils.utils import parse_json_response
from utils.split_policy.layered_graph import (
    materialize_entity_graph,
    split_entity_groups_batch,
)


def materialize_subject_graph(graph: Dict) -> Dict:
    return materialize_entity_graph(
        graph,
        node_prefix="s",
        entity_name="Subject",
    )


def split_subject_groups_batch(
    llm_client: LLMClient,
    rules: List[Dict],
    organization: str,
    batch_size: int = 10,
    max_retries: int = 2,
    max_depth: int = 12,
    max_nodes_per_subject: int = 256,
) -> List[Dict]:
    return split_entity_groups_batch(
        llm_client,
        rules,
        organization,
        entity_name="Subject",
        element_field="Subject",
        output_field="Subject_Elements",
        node_prefix="s",
        context_fields=("Predicate", "Object", "Condition"),
        parse_json=parse_json_response,
        batch_size=batch_size,
        max_retries=max_retries,
        max_depth=max_depth,
        max_nodes_per_graph=max_nodes_per_subject,
    )
