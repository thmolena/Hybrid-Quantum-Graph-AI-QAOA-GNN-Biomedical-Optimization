from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qaoa.baselines import run_qaoa_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline QAOA evaluations on the repository edge-list artifact.")
    parser.add_argument(
        "--graph-path",
        default="outputs/maxcut_graph.csv",
        help="Path to the weighted graph CSV artifact.",
    )
    parser.add_argument(
        "--angles-path",
        default="outputs/qaoa_classical_angles.csv",
        help="Path to the saved classical QAOA angle CSV artifact.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/tables/qaoa_baselines.csv",
        help="Path where the baseline table will be written.",
    )
    parser.add_argument(
        "--num-random-samples",
        type=int,
        default=256,
        help="Number of random angle samples used for the heuristic search baseline.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the heuristic search baseline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_qaoa_baselines(
        graph_path=Path(args.graph_path),
        angles_path=Path(args.angles_path),
        output_path=Path(args.output_path),
        num_random_samples=args.num_random_samples,
        seed=args.seed,
    )
    print(f"Wrote QAOA baseline table to {output_path}")


if __name__ == "__main__":
    main()
