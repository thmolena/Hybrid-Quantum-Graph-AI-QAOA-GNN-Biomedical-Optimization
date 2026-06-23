from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def ensure_parent_dir(file_path: str | Path) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_records(records: Iterable[Mapping[str, object]], output_path: str | Path) -> Path:
    rows = list(records)
    if not rows:
        raise ValueError("Expected at least one record to write")

    path = ensure_parent_dir(output_path)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
