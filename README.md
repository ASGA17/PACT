# PACT: Preserving Policy Semantics in LLM Safety Tests

Code and experimental data for PACT. This repository contains policy-graph construction and behavior-selection code, contract prompt builders, frozen experimental inputs, item-level evaluations, and reported results.

## Quick start

Use Python 3.11 for the pinned model dependencies. The offline check needs no API key, GPU, or third-party package:

```bash
python scripts/validate.py
```

This checks question/response/label alignment and recomputes graph metrics, contract ablation aggregates, and five-model ASR. It writes `results/validation.json`.

Export the four ablation configurations for the frozen panel:

```bash
python scripts/export_ablation_prompts.py
```

See [Reproduction](docs/REPRODUCING.md) for model-backed steps and their requirements.

## Contents

| Path | Contents |
|---|---|
| `split_policy.py`, `merge_graph.py`, `utils/` | Layered parsing, grounding, policy graphs, constrained exploration, and contract compilation |
| `utils/type_exploration/` | Graph indexing, candidate behavior-path exploration, branch selection, and validity-evaluation helpers |
| `prompts/` | Task-specific output requirements, 17 question slots, and contract ablation prompts |
| `policy/` | Thirteen policy source texts; per-policy source labels are in `data/graphs/global_graph.json` |
| `data/graphs/` | Constructed global policy graph |
| `data/graph_evaluation/` | 100-policy sample, Gold facts, extracted representations, and per-policy scores |
| `data/contract_ablation/` | 50 scenarios, five configurations, blinded inputs, and three judges' labels |
| `data/final/` | 28,662 questions, five models' responses, and LG/HB labels |
| `data/reference_sets/` | Ten coverage/novelty reference sets |
| `results/` | Reported aggregates and offline verification results |
| `experiment_config.json` | Manuscript scope and terminology |

The final set contains 1,686 scenarios with 17 questions each. The five target models are Gemma-7B, Llama-2-7B-chat, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, and Vicuna-7B-v1.5. There are 143,310 responses and 286,620 safety labels across two judges.

The contract ablation uses 50 of these scenarios, comprising 25 with explicit conditions or exceptions and 25 without. Each configuration contains 850 questions. Judge results remain separate; no majority-vote aggregation is used for the paper's metrics.

## Evaluation

Policy graph evaluation and contract ablation score records and final safety labels support offline aggregation. Behavior exploration results and the traversal implementation are also included. `results/safety_testing/coverage_reported.json` contains paper-reported coverage values; `scripts/coverage.py` evaluates the released questions and writes a separate results file. `results/safety_testing/budget_allocation_reported.json` compares coverage and novelty between the breadth-first (slot-major) and depth-first (scene-major) budget-allocation strategies.

See [Data dictionary](docs/DATA_DICTIONARY.md) and [Third-party materials](docs/THIRD_PARTY.md). No API keys, model weights, local caches, or author configuration files are included.

## Supplementary material

The paper's Data Availability statement names this repository as supplementary material. Where each promised item lives:

- Representation and algorithm details: [Data dictionary](docs/DATA_DICTIONARY.md) plus the pipeline code (`split_policy.py`, `merge_graph.py`, `utils/`, `prompts/`).
- Evaluation protocols: [Reproduction](docs/REPRODUCING.md) and `scripts/`.
- Policy-source distribution: per-policy `SourceID` values in `data/graphs/global_graph.json` (282 policies across 13 sources).
- Sample composition of the ablation panel: `ConstraintStratum` and `PrimaryTaskType` in `data/contract_ablation/selected_scenes.jsonl`.
- Scenario-cluster bootstrap intervals: `results/contract_ablation/bootstrap_intervals.json` (procedure: `scripts/cluster_bootstrap.py`).
- Budget-allocation comparison: `results/safety_testing/budget_allocation_reported.json` (breadth-first vs depth-first).
- Additional experimental results: `results/behavior_exploration/summary.json`.

Large datasets use gzip compression. The evaluation scripts read them directly; manual extraction is optional. For example, `gzip -dc data/final/questions.jsonl.gz | head -n 1` displays the first question.
