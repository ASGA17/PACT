"""Read plain or gzip-compressed JSONL datasets using the same interface."""

import gzip
import json
from pathlib import Path


def read_jsonl(path):
    """Read records, accepting a compressed counterpart of a plain JSONL path."""
    path = Path(path)
    if not path.exists() and path.suffix != ".gz":
        compressed = path.with_name(path.name + ".gz")
        if compressed.exists():
            path = compressed
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]
