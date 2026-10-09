import logging
from typing import Any, Dict, List, Optional

from llms import LLMClient
from policy_loader import load_regulation_text
from utils.utils import parse_json_response

from utils.merge_graph.global_graph_builder import (
    build_occurrence_graph,
    strip_internal_fields,
)
from utils.merge_graph.global_graph_fusion import apply_global_fusion
from utils.merge_graph.global_graph_validator import validate_global_graph

logger = logging.getLogger(__name__)


def _nonempty_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def build_global_graph_batch(
    llm_client: Optional[LLMClient],
    sources: List[Dict],
    batch_size: int = 20,
    max_retries: int = 2,
    strict_validation: bool = True,
) -> Dict:
    """Build one cross-policy, cross-source global graph.

    Parameters
    ----------
    llm_client:
        Used only for genuinely ambiguous canonical fusion and service reference
        refinement. Pass None to run deterministic fusion only.

    sources:
        Each item must contain:
            {
                "source_id": "baidu",
                "organization": "Baidu",
                "rules": [... enriched policy rules ...]
            }

        source_id identifies the input dataset/file. organization identifies the
        policy issuer. Multiple source_ids may share the same organization.

    The input Policy Graphs are never mutated.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than 0.")

    graph = build_occurrence_graph(sources)
    apply_global_fusion(
        graph,
        llm_client,
        batch_size=batch_size,
        max_retries=max_retries,
    )

    errors = validate_global_graph(graph)
    if errors:
        logger.error("GlobalGraph | validation failed | errors=%s", "; ".join(errors))
        if strict_validation:
            raise ValueError("Global graph validation failed: " + "; ".join(errors))

    # Validation and fusion stats are internal gates/diagnostics, not exported fields.
    fusion_stats = graph.get("_FusionStats", {})
    if not isinstance(fusion_stats, dict):
        fusion_stats = {}
    result = strip_internal_fields(graph)

    logger.info(
        "GlobalGraph | completed | sources=%d | policies=%d | occurrences=%d | "
        "canonical_nodes=%d | edges=%d | eligible_occurrences=%s | "
        "deterministic_merges=%s | llm_fusion_tasks=%s | llm_merges=%s | "
        "fusion_unresolved=%s | subtype_edges=%s | "
        "service_ref_deterministic=%s | service_ref_llm=%s | "
        "service_ref_unresolved=%s | api_calls=%s | input_tokens=%s | "
        "output_tokens=%s | total_tokens=%s | valid=%s",
        len(result.get("Sources", {})),
        len(result.get("Policies", {})),
        len(result.get("Occurrences", {})),
        len(result.get("CanonicalNodes", {})),
        len(result.get("Edges", [])),
        fusion_stats.get("EligibleOccurrences", "?"),
        fusion_stats.get("DeterministicMerges", "?"),
        fusion_stats.get("LLMFusionTasks", "?"),
        fusion_stats.get("LLMMerges", "?"),
        fusion_stats.get("FusionUnresolved", "?"),
        fusion_stats.get("SubtypeEdges", "?"),
        fusion_stats.get("ServiceReferencesDeterministic", "?"),
        fusion_stats.get("ServiceReferencesLLM", "?"),
        fusion_stats.get("ServiceReferencesUnresolved", "?"),
        fusion_stats.get("APICalls", "?"),
        fusion_stats.get("InputTokens", "?"),
        fusion_stats.get("OutputTokens", "?"),
        fusion_stats.get("TotalTokens", "?"),
        not errors,
    )
    return result


def build_global_graph_from_files(
    llm_client: Optional[LLMClient],
    inputs: List[Dict],
    batch_size: int = 20,
    max_retries: int = 2,
    strict_validation: bool = True,
) -> Dict:
    """Load multiple enriched graph files and fuse them into one global graph.

    Example input:
        [
            {
                "source_id": "baidu",
                "organization": "Baidu",
                "path": "./output/Baidu/enriched_graph_policy.txt",
            },
            {
                "source_id": "openai",
                "organization": "OpenAI",
                "path": "./output/OpenAI/enriched_graph_policy.txt",
            },
        ]
    """
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a non-empty list.")

    sources: List[Dict] = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise TypeError(f"inputs[{index}] must be a dict.")
        source_id = _nonempty_text(item.get("source_id"))
        organization = _nonempty_text(item.get("organization"))
        path = _nonempty_text(item.get("path"))
        if source_id is None:
            raise ValueError(f"inputs[{index}].source_id is required.")
        if organization is None:
            raise ValueError(f"inputs[{index}].organization is required.")
        if path is None:
            raise ValueError(f"inputs[{index}].path is required.")

        raw = load_regulation_text(path)
        rules = parse_json_response(raw)
        if not isinstance(rules, list):
            raise ValueError(f"Input file did not parse to a rule list: {path}")

        sources.append(
            {
                "source_id": source_id,
                "organization": organization,
                "rules": rules,
                "path": path,
            }
        )

    return build_global_graph_batch(
        llm_client=llm_client,
        sources=sources,
        batch_size=batch_size,
        max_retries=max_retries,
        strict_validation=strict_validation,
    )


__all__ = [
    "build_global_graph_batch",
    "build_global_graph_from_files",
    "validate_global_graph",
]
