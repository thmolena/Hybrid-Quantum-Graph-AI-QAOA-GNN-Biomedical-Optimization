"""Run the transcriptomic QAOA depolarizing-noise benchmark and export summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qaoa.transcriptomic import run_transcriptomic_noise_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("outputs/tables/qaoa_noise_summary.csv"),
        help="Path for the aggregated noisy-QAOA summary table.",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=Path("outputs/tables/qaoa_noise_summary_meta.json"),
        help="Path for experiment metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, metadata = run_transcriptomic_noise_experiment()
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_table, index=False)
    with args.output_meta.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    print(summary.to_string(index=False))
    print(f"\nWrote noise summary to {args.output_table}")
    print(f"Wrote metadata to {args.output_meta}")


if __name__ == "__main__":
    main()