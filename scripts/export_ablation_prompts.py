#!/usr/bin/env python3
"""Export 3,400 prompts for the four ablations on the frozen 50-scenario panel."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prompts.contract_ablation import slot_cells, task_facts, build_prompt, SYSTEM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/ablation_prompts.jsonl"
    )
    args = parser.parse_args()
    with (ROOT / "data/contract_ablation/selected_scenes.jsonl").open(
        encoding="utf-8"
    ) as output_file:
        scenes = [json.loads(l) for l in output_file]
    assert len(scenes) == 50
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt_count = 0
    with args.output.open("w") as output_file:
        for scene in scenes:
            for arm in ["no_structure", "no_constraints", "no_routing", "scene_only"]:
                for cell in slot_cells(scene["PrimaryTaskType"]):
                    row = {
                        **scene,
                        **cell,
                        "AblationArm": arm,
                        "ContractFactSentence": task_facts(scene["ContractTask"]),
                    }
                    record = {
                        k: row[k]
                        for k in [
                            "SampleSceneIndex",
                            "SourceSceneIndex",
                            "AblationArm",
                            "QuestionSlot",
                        ]
                    }
                    record.update(system=SYSTEM, user=build_prompt(row))
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    prompt_count += 1
    assert prompt_count == 3400
    print(f"Exported {prompt_count} prompts to {args.output}")


if __name__ == "__main__":
    main()
