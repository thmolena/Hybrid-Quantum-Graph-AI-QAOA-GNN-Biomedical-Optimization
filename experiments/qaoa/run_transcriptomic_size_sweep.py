from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qaoa.transcriptomic import (
    TranscriptomicBenchmarkConfig,
    run_transcriptomic_generalization_benchmark,
    target_edge_count_for_gene_count,
)


DEFAULT_SUMMARY_PATH = REPO_ROOT / "outputs" / "tables" / "qaoa_size_sweep.csv"
DEFAULT_DETAILED_PATH = REPO_ROOT / "outputs" / "tables" / "qaoa_size_sweep_detailed.csv"
DEFAULT_META_PATH = REPO_ROOT / "outputs" / "tables" / "qaoa_size_sweep_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real transcriptomic QAOA size sweep.")
    parser.add_argument("--gene-counts", nargs="+", type=int, default=[4, 6, 8, 10, 12, 14, 16], help="Top-gene panel sizes to evaluate.")
    parser.add_argument("--target-density", type=float, default=0.4, help="Target graph density used to derive edge budgets.")
    parser.add_argument("--adaptation-size", type=int, default=12, help="Number of adaptation graphs per size.")
    parser.add_argument("--benchmark-size", type=int, default=4, help="Number of held-out benchmark graphs per size.")
    parser.add_argument("--subsample-size", type=int, default=50, help="Number of patients per graph subsample.")
    parser.add_argument("--num-starts", type=int, default=5, help="Number of classical optimization restarts per graph.")
    parser.add_argument("--maxiter", type=int, default=160, help="Maximum Nelder-Mead iterations per restart.")
    parser.add_argument("--epochs", type=int, default=250, help="Maximum GNN training epochs.")
    parser.add_argument("--patience", type=int, default=40, help="Early-stopping patience for GNN training.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for the GNN.")
    parser.add_argument("--learning-rate", type=float, default=5e-3, help="Learning rate for GNN training.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for GNN training.")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH, help="Output CSV for sweep summaries.")
    parser.add_argument("--detailed-out", type=Path, default=DEFAULT_DETAILED_PATH, help="Output CSV for per-graph results.")
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_PATH, help="Output JSON for sweep metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_frames: List[pd.DataFrame] = []
    detailed_frames: List[pd.DataFrame] = []
    metadata_rows = []

    for top_gene_count in args.gene_counts:
        target_edge_count = target_edge_count_for_gene_count(top_gene_count, args.target_density)
        config = TranscriptomicBenchmarkConfig(
            top_gene_count=top_gene_count,
            target_edge_count=target_edge_count,
            adaptation_size=args.adaptation_size,
            benchmark_size=args.benchmark_size,
            subsample_size=args.subsample_size,
            num_starts=args.num_starts,
            maxiter=args.maxiter,
        )
        training_kwargs = {
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
        }

        started_at = time.perf_counter()
        detailed_frame, summary_frame, metadata = run_transcriptomic_generalization_benchmark(config, training_kwargs=training_kwargs)
        sweep_runtime_s = time.perf_counter() - started_at

        detailed_frame = detailed_frame.copy()
        detailed_frame.insert(0, "top_gene_count", top_gene_count)
        detailed_frame.insert(1, "target_edge_count", target_edge_count)
        detailed_frame["sweep_runtime_s"] = sweep_runtime_s
        detailed_frames.append(detailed_frame)

        summary_frame = summary_frame.copy()
        summary_frame.insert(0, "top_gene_count", top_gene_count)
        summary_frame.insert(1, "target_edge_count", target_edge_count)
        summary_frame["adaptation_size"] = config.adaptation_size
        summary_frame["benchmark_size"] = config.benchmark_size
        summary_frame["training_best_loss"] = metadata["training"]["best_loss"]
        summary_frame["training_best_epoch"] = metadata["training"]["best_epoch"]
        summary_frame["training_epochs_run"] = metadata["training"]["epochs_run"]
        summary_frame["sweep_runtime_s"] = sweep_runtime_s
        summary_frames.append(summary_frame)

        metadata_rows.append(
            {
                "config": asdict(config),
                "training": metadata["training"],
                "sweep_runtime_s": sweep_runtime_s,
            }
        )

        print(
            f"top_gene_count={top_gene_count} target_edge_count={target_edge_count} "
            f"runtime={sweep_runtime_s:.2f}s"
        )
        print(summary_frame[["method", "mean_ratio", "median_runtime_ms", "retention_vs_classical"]].to_string(index=False))

    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_detailed = pd.concat(detailed_frames, ignore_index=True)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(args.summary_out, index=False)
    combined_detailed.to_csv(args.detailed_out, index=False)
    with args.meta_out.open("w", encoding="utf-8") as handle:
        json.dump({"runs": metadata_rows}, handle, indent=2)


if __name__ == "__main__":
    main()