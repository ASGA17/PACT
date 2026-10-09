#!/usr/bin/env python3
"""Check frozen artifact alignment and recompute available paper aggregates offline."""

import collections
import csv
import json
from pathlib import Path

from data_io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return read_jsonl(ROOT / name)


def table(name):
    with (ROOT / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def position(record):
    """Return the scene/slot key shared by questions, responses, and labels."""
    return int(record["SceneIndex"]), int(record["QuestionSlot"])


def main():
    config = json.loads((ROOT / "experiment_config.json").read_text())
    report = {}
    questions = read("data/final/questions.jsonl")
    positions = {position(record) for record in questions}
    assert len(questions) == len(positions) == 28662
    question_ids = {record["QuestionID"] for record in questions}
    assert len(question_ids) == 28662
    question_fields = set(questions[0])
    assert all(set(record) == question_fields for record in questions)
    question_lookup = {position(record): record for record in questions}
    for record in questions:
        expected_id = (record["SceneIndex"] - 1) * 17 + record["QuestionSlot"] + 1
        assert record["QuestionID"] == f"question:{expected_id:06d}"
    scenes = collections.Counter(s for s, _ in positions)
    assert set(scenes) == set(range(1, 1687))
    scene_index = read("data/final/scenes.jsonl")
    assert len(scene_index) == 1686
    assert {r["SceneIndex"] for r in scene_index} == set(scenes)
    assert all(
        r["QuestionSlots"] == list(range(17)) and r["QuestionCount"] == 17
        for r in scene_index
    )
    assert len(scenes) == 1686 and set(scenes.values()) == {17}
    assert all(
        {slot for scene, slot in positions if scene == s} == set(range(17))
        for s in scenes
    )
    counts = json.loads((ROOT / "data/final/asr_counts.json").read_text())["per_model"]
    counts = {record["model"]: record for record in counts}
    asr = {}
    for model in config["models"]:
        responses = read(f"data/final/responses/{model}.jsonl.gz")
        assert (
            len(responses) == 28662
            and {position(record) for record in responses} == positions
        )
        assert all(isinstance(record["output"], str) for record in responses)
        assert all(set(record) == set(responses[0]) for record in responses)
        assert all(
            record["QuestionID"] == question_lookup[position(record)]["QuestionID"]
            for record in responses
        )
        asr[model] = {}
        for judge in ["LG", "HB"]:
            labels = read(f"data/final/safety/{judge}/{model}.jsonl.gz")
            assert (
                len(labels) == 28662
                and {position(record) for record in labels} == positions
            )
            assert all(
                record["attack_success"] in [True, False, 0, 1] for record in labels
            )
            assert all(set(record) == set(labels[0]) for record in labels)
            assert all(
                record["QuestionID"] == question_lookup[position(record)]["QuestionID"]
                for record in labels
            )
            n = sum(bool(record["attack_success"]) for record in labels)
            assert n == counts[model][f"pact_{judge.lower()}_success"], (
                model,
                judge,
                n,
            )
            asr[model][judge] = round(100 * n / 28662, 2)
    report["final"] = {
        "questions": len(questions),
        "scenarios": len(scenes),
        "responses": 5 * len(questions),
        "safety_labels": 10 * len(questions),
        "asr_percent": asr,
        "macro_asr_percent": {
            judge: round(sum(asr[model][judge] for model in asr) / 5, 2)
            for judge in ["LG", "HB"]
        },
    }
    panel = read("data/contract_ablation/selected_scenes.jsonl")
    assert len(panel) == 50
    assert collections.Counter(record["ConstraintStratum"] for record in panel) == {
        "constrained": 25,
        "unconstrained": 25,
    }
    assert all(int(record["SourceSceneIndex"]) in scenes for record in panel)
    questions = read("data/contract_ablation/questions.jsonl") + read(
        "data/contract_ablation/full_reference_questions.jsonl"
    )
    assert len(questions) == 4250
    assert all(set(record) == set(questions[0]) for record in questions)
    assert len({record["QuestionID"] for record in questions}) == 4250
    for record in questions:
        assert record["SceneIndex"] == record["SourceSceneIndex"]
        if record["AblationArm"] == "full_contract":
            source = question_lookup[position(record)]
            assert record["Question"] == source["Question"]
            assert record["QuestionID"] == source["QuestionID"]
    ablation_lookup = {
        (r["AblationArm"], r["SampleSceneIndex"], r["QuestionSlot"]): r
        for r in questions
    }
    key = {
        record["id"]: record
        for record in read("data/contract_ablation/blind_key.jsonl")
    }
    assert len(key) == 4250
    blind_inputs = {
        r["id"]: r for r in read("data/contract_ablation/blind_input.jsonl")
    }
    assert set(blind_inputs) == set(key)
    for identifier, record in key.items():
        question = ablation_lookup[
            record["AblationArm"], record["SampleSceneIndex"], record["QuestionSlot"]
        ]
        assert record["QuestionID"] == question["QuestionID"]
        assert record["SourceSceneIndex"] == question["SourceSceneIndex"]
        assert blind_inputs[identifier]["question"] == question["Question"]
    cells = {
        (
            record["AblationArm"],
            int(record["SampleSceneIndex"]),
            int(record["QuestionSlot"]),
        )
        for record in questions
    }
    assert len(cells) == 4250
    assert cells == {
        (
            record["AblationArm"],
            int(record["SampleSceneIndex"]),
            int(record["QuestionSlot"]),
        )
        for record in key.values()
    }
    expected = {
        (record["judge"], record["arm"]): record
        for record in table("results/contract_ablation/judge_metrics.csv")
    }
    out = []
    for judge in config["contract_ablation_judges"]:
        labels = read(f"data/contract_ablation/judges/{judge}.jsonl")
        assert len(labels) == 4250 and {record["id"] for record in labels} == set(key)
        groups = collections.defaultdict(list)
        for record in labels:
            groups[key[record["id"]]["AblationArm"]].append(record)
        for arm, records in groups.items():
            assert len(records) == 850
            record = {"judge": judge, "arm": arm, "n": len(records)}
            for field in ["pcr", "bpr", "task_match"]:
                n = sum(record[field] is True for record in records)
                assert n == int(expected[judge, arm][field + "_pass"])
                record[field] = round(100 * n / len(records), 2)
            out.append(record)
    report["contract_ablation"] = out
    report["graph_evaluation"] = {}
    for file in ["graph_scores.csv", "graph_ablation.csv"]:
        groups = collections.defaultdict(list)
        for record in table("data/graph_evaluation/" + file):
            groups[record["Method"]].append(record)
        vals = {}
        for method, records in groups.items():
            assert len(records) == 100
            complex_rs = [
                record for record in records if int(record["ComplexityConfirmed"]) == 1
            ]
            assert len(complex_rs) == 83
            result = {
                "PSR": round(sum(float(record["PSR"]) for record in records), 2),
                "StructuralPSR": round(
                    100 * sum(float(record["PSR"]) for record in complex_rs) / 83, 2
                ),
                "USFR": round(
                    100
                    * sum(float(record["UnsupportedClaims"]) for record in records)
                    / sum(float(record["ProjectedClaims"]) for record in records),
                    2,
                ),
            }
            if "SAO_TP" in records[0]:
                tp = sum(float(record["SAO_TP"]) for record in records)
                fp = sum(float(record["SAO_FP"]) for record in records)
                fn = sum(float(record["SAO_FN"]) for record in records)
                result["CommonSAOmicroF1"] = round(200 * tp / (2 * tp + fp + fn), 2)
                assert sum(int(record["GoldSAO"]) for record in records) == 623
            else:
                assert sum(int(record["GoldFacts"]) for record in records) == 657
                result["StrictPFR"] = round(
                    100 * sum(int(record["FactsFull"]) for record in records) / 657, 2
                )
            vals[method] = result
        report["graph_evaluation"][file] = vals
    assert report["graph_evaluation"]["graph_scores.csv"]["PACT"] == {
        "PSR": 93.0,
        "StructuralPSR": 91.57,
        "USFR": 0.89,
        "CommonSAOmicroF1": 92.67,
    }
    assert report["graph_evaluation"]["graph_scores.csv"]["KGGen"] == {
        "PSR": 44.0,
        "StructuralPSR": 33.73,
        "USFR": 5.79,
        "CommonSAOmicroF1": 85.1,
    }
    assert report["graph_evaluation"]["graph_ablation.csv"] == {
        "PACT": {"PSR": 93.0, "StructuralPSR": 91.57, "USFR": 0.89, "StrictPFR": 98.78},
        "OneShot": {
            "PSR": 92.0,
            "StructuralPSR": 90.36,
            "USFR": 19.74,
            "StrictPFR": 98.33,
        },
        "NoGrounding": {
            "PSR": 86.0,
            "StructuralPSR": 83.13,
            "USFR": 6.37,
            "StrictPFR": 96.65,
        },
    }
    assert report["final"]["macro_asr_percent"] == {"LG": 18.92, "HB": 33.18}
    assert collections.Counter(r["PrimaryTaskType"] for r in panel) == {
        "advice": 7,
        "artifact": 19,
        "code": 3,
        "document": 3,
        "procedural": 18,
    }
    for path in ROOT.rglob("*"):
        if path.is_file():
            assert path.stat().st_size < 100_000_000, ("Large GitHub file", str(path))
    (ROOT / "results/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: final questions, five-model responses/safety labels, 50-scene panel, 45 contract ablation metric counts, policy graph evaluation aggregates, and file sizes."
    )
    print("Safety testing macro ASR:", report["final"]["macro_asr_percent"])


if __name__ == "__main__":
    main()
