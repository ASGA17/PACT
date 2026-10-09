# Data dictionary

JSONL files contain one JSON object per line. `.jsonl.gz` uses standard gzip compression. Texts, responses, and judgment values are preserved; local runtime paths and obsolete processing metadata are omitted.

Large JSONL datasets are gzip-compressed; smaller scene indexes remain plain JSONL:

| File | Records | Purpose |
|---|---:|---|
| `data/final/questions.jsonl.gz` | 28,662 | Final test inputs and behavior/task fields |
| `data/final/scenes.jsonl` | 1,686 | Scenario/slot index |
| `data/contract_ablation/selected_scenes.jsonl` | 50 | Ablation scenarios and contracts |
| `data/contract_ablation/full_reference_questions.jsonl.gz` | 850 | PACT configuration |
| `data/contract_ablation/questions.jsonl.gz` | 3,400 | Four ablations, 850 each |

## Final evaluation

`SceneIndex` runs consecutively from 1 to 1,686. `QuestionSlot` is zero-based (0--16) within each scenario. Final question IDs run from `question:000001` to `question:028662`; each ID identifies the same input in model responses and safety labels. The `(SceneIndex, QuestionSlot)` pair is an equivalent unique key.

All final question records have the same fields. Optional metadata absent from a source record is represented by JSON `null`; this does not indicate an empty question or a missing response. No semantic metadata is inferred to fill these fields.

- `Question`: actual test input; `Description`: scenario context.
- `TargetBehavior`, `MustKeep`, `PrimaryTaskType`: behavioral and output-task requirements.
- `BehaviorID`, `BranchID`: retained provenance identifiers.
- `ExecutionStyle`, `DirectnessLevel`: expression settings.
- `output`: target-model response; `target_generation_settings`: retained decoding settings.
- `attack_success`: response judged unsafe by the indicated safety judge.
- `judge_raw_output`: judge's original textual output.

`scenes.jsonl` is an index of the final question set, not a new generation record. `asr_counts.json` contains `pact_*` success counts and `polaris_*` reported counts; denominators are 28,662 and 28,660 respectively.

## Contract ablation

`questions.jsonl.gz` contains four ablations; `full_reference_questions.jsonl.gz` contains PACT. Both question files use the same field set. `SampleSceneIndex` runs from 0 to 49; ablation IDs encode the configuration, sample, and slot. PACT reference questions reuse the corresponding final question IDs. Join using `(AblationArm, SampleSceneIndex, QuestionSlot)`. `SourceSceneIndex` maps to the final scenario identifiers. `blind_key.jsonl.gz` maps blinded `id` values to experimental cells. `blind_input.jsonl.gz` preserves the inputs supplied for judging. Judge files use `pcr`, `bpr`, and `task_match` Boolean labels.

Configuration identifiers are stable code keys. In prose, `full_contract` is PACT and `scene_only` is Scenario Only; the other identifiers map directly to No Structure, No Constraints, and No Routing.

## Graph evaluation

`gold.json` contains the policy annotations used for the reference comparison. Gold denotes the human-verified reference described in the manuscript; this packaging step does not constitute a new annotation verification.

Per-policy CSVs retain integer counts and replace the older score-column name `VSR` with the manuscript term `PSR`. `Ours`/`Full` method labels become `PACT`. No factual scores are changed.

## Precision

Aggregate rates should be computed from counts without intermediate rounding. The manuscript's printed comparisons can reflect subtraction or averaging of displayed two-decimal rates; last-decimal differences from direct count-based calculations are therefore possible. Keep the counts intact.

`data/graphs/selected_branch_inventory.jsonl.gz` contains the 692 branch records referenced by available final-question `BranchID` fields. An optional `BranchID` is `null` when no identifier is supplied. Scenario/slot alignment remains complete for every question and response.

The scripts automatically accept either a `.jsonl` file or its `.jsonl.gz` counterpart. To inspect compressed records locally, use `gzip -dc data/final/questions.jsonl.gz | head -n 1`.
