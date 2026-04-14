from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qaoa.transcriptomic import TranscriptomicBenchmarkConfig, run_transcriptomic_generalization_benchmark


DEFAULT_SUMMARY_PATH = REPO_ROOT / "outputs" / "tables" / "qaoa_adaptation_sweep.csv"
DEFAULT_DETAILED_PATH = REPO_ROOT / "outputs" / "tables" / "qaoa_adaptation_sweep_detailed.csv"
DEFAULT_META_PATH = REPO_ROOT / "outputs" / "tables" / "qaoa_adaptation_sweep_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an adaptation-family-size study for the transcriptomic QAOA benchmark.")
    parser.add_argument("--adaptation-sizes", nargs="+", type=int, default=[8, 12, 16, 24], help="Adaptation-family sizes to evaluate.")
    parser.add_argument("--benchmark-size", type=int, default=6, help="Number of held-out benchmark graphs.")
    parser.add_argument("--subsample-size", type=int, default=60, help="Number of patients per graph subsample.")
    parser.add_argument("--num-starts", type=int, default=6, help="Number of classical optimization restarts per graph.")
    parser.add_argument("--maxiter", type=int, default=240, help="Maximum Nelder-Mead iterations per restart.")
    parser.add_argument("--training-seed", type=int, default=7, help="Training seed for the GNN.")
    parser.add_argument("--epochs", type=int, default=350, help="Maximum GNN training epochs.")
    parser.add_argument("--patience", type=int, default=45, help="Early-stopping patience for GNN training.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for the GNN.")
    parser.add_argument("--learning-rate", type=float, default=5e-3, help="Learning rate for GNN training.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for GNN training.")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH, help="Output CSV for aggregated adaptation-size results.")
    parser.add_argument("--detailed-out", type=Path, default=DEFAULT_DETAILED_PATH, help="Output CSV for per-graph results.")
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_PATH, help="Output JSON for run metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_frames: List[pd.DataFrame] = []
    detailed_frames: List[pd.DataFrame] = []
    metadata_rows = []

    for adaptation_size in args.adaptation_sizes:
        config = TranscriptomicBenchmarkConfig(
            adaptation_size=adaptation_size,
            benchmark_size=args.benchmark_size,
            subsample_size=args.subsample_size,
            num_starts=args.num_starts,
            maxiter=args.maxiter,
            training_seed=args.training_seed,
        )
        training_kwargs = {
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
        }

        detailed_frame, summary_frame, metadata = run_transcriptomic_generalization_benchmark(
            config,
            training_kwargs=training_kwargs,
        )

        detailed_frame = detailed_frame.copy()
        detailed_frame.insert(0, "adaptation_size", adaptation_size)
        detailed_frames.append(detailed_frame)

        summary_frame = summary_frame.copy()
        summary_frame.insert(0, "adaptation_size", adaptation_size)
        summary_frame["training_best_loss"] = metadata["training"]["best_loss"]
        summary_frame["training_best_epoch"] = metadata["training"]["best_epoch"]
        summary_frame["training_epochs_run"] = metadata["training"]["epochs_run"]
        summary_frames.append(summary_frame)

        metadata_rows.append({"config": asdict(config), "training": metadata["training"]})
        print(f"adaptation_size={adaptation_size}")
        print(summary_frame[["method", "mean_ratio", "std_ratio", "median_runtime_ms", "retention_vs_classical"]].to_string(index=False))

    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_detailed = pd.concat(detailed_frames, ignore_index=True)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(args.summary_out, index=False)
    combined_detailed.to_csv(args.detailed_out, index=False)
    with args.meta_out.open("w", encoding="utf-8") as handle:
        json.dump({"runs": metadata_rows}, handle, indent=2)


if __name__ == "__main__":
    main()