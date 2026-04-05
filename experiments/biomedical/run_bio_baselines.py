from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.biomedical.baselines import run_biomedical_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical biomedical baselines on the processed CTG artifact.")
    parser.add_argument(
        "--data-path",
        default="outputs/ctg_processed.csv",
        help="Path to the processed CTG CSV artifact.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/tables/biomedical_baselines.csv",
        help="Path where the baseline table will be written.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for deterministic baselines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_biomedical_baselines(
        data_path=Path(args.data_path),
        output_path=Path(args.output_path),
        seed=args.seed,
    )
    print(f"Wrote biomedical baseline table to {output_path}")


if __name__ == "__main__":
    main()