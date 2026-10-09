#!/usr/bin/env python3
"""Scenario-paired bootstrap using each judge's independent labels (requires numpy)."""

import argparse
import json
from pathlib import Path

from data_io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/cluster_bootstrap.json"
    )
    args = parser.parse_args()
    import numpy as np

    blind_key = {
        record["id"]: record
        for record in read_jsonl(ROOT / "data/contract_ablation/blind_key.jsonl")
    }
    scene_ids = sorted({record["SampleSceneIndex"] for record in blind_key.values()})
    assert len(scene_ids) == 50
    rng = np.random.default_rng(args.seed)
    # Resample whole scenarios to keep their 17 question variants together.
    indices = rng.integers(0, 50, size=(args.draws, 50))
    results = []
    for path in sorted((ROOT / "data/contract_ablation/judges").glob("*.jsonl*")):
        labels = {record["id"]: record for record in read_jsonl(path)}
        for arm, metric in [("no_constraints", "bpr"), ("no_routing", "task_match")]:
            values = {}
            for identifier, record in blind_key.items():
                if record["AblationArm"] in ["full_contract", arm]:
                    values[
                        record["AblationArm"],
                        record["SampleSceneIndex"],
                        record["QuestionSlot"],
                    ] = float(labels[identifier][metric])
            differences = np.array(
                [
                    100
                    * np.mean(
                        [
                            values["full_contract", scene_index, slot_index]
                            - values[arm, scene_index, slot_index]
                            for slot_index in range(17)
                        ]
                    )
                    for scene_index in scene_ids
                ]
            )
            samples = differences[indices].mean(axis=1)
            lower, upper = np.percentile(samples, [2.5, 97.5])
            results.append(
                {
                    "judge": path.name.removesuffix(".gz").removesuffix(".jsonl"),
                    "ablation": arm,
                    "metric": metric,
                    "difference_pp": float(differences.mean()),
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "draws": args.draws,
                "seed": args.seed,
                "note": "New bootstrap run; historical random draws are not reconstructed.",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
