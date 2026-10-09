# Reproduction

Run commands from the repository root. Offline validation and prompt export require only Python's standard library.

For model-backed parsing and coverage evaluation, create a Python 3.11 environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements pin API clients, Transformer/tokenizer dependencies, PyTorch, numerical libraries, and graph utilities. GPU execution additionally requires a compatible NVIDIA driver; the Python requirements do not install a system driver. Model weights are downloaded separately when needed.

## Offline aggregates

```bash
python scripts/validate.py
python scripts/export_ablation_prompts.py
```

The first command checks cardinalities, unique keys, label alignment, all 45 contract ablation pass counts, graph metrics, and five-model safety counts. The second exports 3,400 ablation prompts using the released 50-scenario panel, without generating or evaluating new questions. The full configuration's existing 850 questions are supplied separately.

To run a fresh 10,000-draw paired scenario bootstrap, install NumPy and execute `python scripts/cluster_bootstrap.py`. This preserves all 17 slots within each resampled scenario. The random seed is configurable.

## Policy parsing and graph components

Install `requirements.txt` in a separate environment. Copy `.env.example` to `.env` and configure the provider required for your model. To parse one policy source:

```bash
python split_policy.py --stage all --organization Baidu --policy-path policy/baidu.txt --model deepseek-v4-flash
```

This performs model API calls and writes new outputs under `output/Baidu/`. `merge_graph.py` exposes graph assembly functions. `utils/type_exploration/` implements the constrained behavior space and semantic-ranking/policy-rotation or Type-RW traversal. `utils/scenario_generation/compiled_task.py` protects condition/exception scopes and compiles selected branches. These are component APIs, not a complete historical-run driver. The frozen global graph is available under `data/graphs/`.

`prompts/task_routing.py` supplies task/output mappings and a full-contract prompt function. `prompts/contract_ablation.py` supplies the four ablation prompt functions. They were extracted from the original experiment code to exclude obsolete sampling runners. The generator in the manuscript is Llama-3-8B-Lexi-Uncensored; model weights are not bundled.

## Coverage and novelty

```bash
python scripts/coverage.py --device cuda
```

This requires the sentence-transformer model and sufficient compute; `--device cpu` is also supported. It embeds all 28,662 question records, retains duplicate rows in the evaluated distribution, uses normalized embeddings and 15th-neighbor density weights, and macro-averages over ten reference sets. It writes `results/safety_testing/coverage_recomputed.json`; manuscript values remain in `coverage_reported.json`.
