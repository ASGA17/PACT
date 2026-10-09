#!/usr/bin/env python3
"""Recompute density-weighted coverage/novelty on all 28,662 question rows.

Requires sentence-transformers and PyTorch. Writes a new result, never overwrites
paper-reported values. Exact duplicate rows remain in the metric distribution.
"""

import argparse
import csv
import json
from pathlib import Path

from data_io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    "AdvBench": "goal",
    "DAN": "question",
    "JBB-Behaviors": "Goal",
    "LLM-Fuzz": "text",
    "MaliciousInstruct": "question",
    "MasterKey": "Questions",
    "airbench": "question",
    "harmbench": "Behavior",
    "sorrybench": "question",
    "sosbench": "goal",
}


def density_weights(x, k=15, chunk=512):
    import torch

    if len(x) <= k:
        raise ValueError("The distribution must contain more than k rows")
    parts = []
    for start in range(0, len(x), chunk):
        d = (1 - x[start : start + chunk] @ x.T).clamp(0, 2)
        parts.append(torch.topk(d, k + 1, largest=False).values[:, k])
    w = torch.cat(parts)
    if w.sum() <= 0:
        w = torch.ones_like(w)
    return w / w.sum()


def nearest(x, y, chunk=512):
    import torch

    return torch.cat(
        [
            (1 - (x[s : s + chunk] @ y.T).max(dim=1).values).clamp(0, 2)
            for s in range(0, len(x), chunk)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/safety_testing/coverage_recomputed.json",
    )
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    texts = [
        record["Question"] for record in read_jsonl(ROOT / "data/final/questions.jsonl")
    ]
    assert len(texts) == 28662
    encoder = SentenceTransformer(args.encoder, device=args.device)

    def embed(t):
        return encoder.encode(
            t, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        )

    generated = embed(texts)
    generated_weights = density_weights(generated)
    results = []
    for name, column in REFERENCES.items():
        with (ROOT / "data/reference_sets" / f"{name}.csv").open(
            encoding="utf-8-sig"
        ) as f:
            reference_texts = [
                r[column].strip()
                for r in csv.DictReader(f)
                if r.get(column, "").strip()
            ]
        x = embed(reference_texts)
        reference_weights = density_weights(x)
        reference_distances = nearest(x, generated)
        generated_distances = nearest(generated, x)
        for tau in [0.4, 0.5, 0.6]:
            results.append(
                {
                    "reference": name,
                    "threshold": tau,
                    "coverage": 100
                    * float(reference_weights[reference_distances <= tau].sum()),
                    "novelty": 100
                    * float(generated_weights[generated_distances > tau].sum()),
                }
            )
    macro = [
        {
            "threshold": t,
            **{
                m: sum(r[m] for r in results if r["threshold"] == t) / 10
                for m in ["coverage", "novelty"]
            },
        }
        for t in [0.4, 0.5, 0.6]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "questions": len(texts),
                "encoder": args.encoder,
                "k": 15,
                "macro": macro,
                "per_reference": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
