"""Run a cross-family transfer study across transcriptomic weighted graph families."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.qaoa.transcriptomic import (
    headline_training_kwargs,
    headline_transcriptomic_benchmark_config,
    TranscriptomicBenchmarkConfig,
    run_cross_family_transfer_experiment,
)


OUTPUT_DIR = Path("outputs") / "tables"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headline_config = headline_transcriptomic_benchmark_config()
    source_config = TranscriptomicBenchmarkConfig(
        top_gene_count=headline_config.top_gene_count,
        target_edge_count=headline_config.target_edge_count,
        benchmark_size=4,
        benchmark_seed=headline_config.benchmark_seed,
        adaptation_size=12,
        adaptation_seed=headline_config.adaptation_seed,
        subsample_size=headline_config.subsample_size,
        depth=headline_config.depth,
        num_starts=5,
        maxiter=160,
        training_seed=headline_config.training_seed,
    )
    training_kwargs = headline_training_kwargs()
    training_kwargs.update({"epochs": 500, "patience": 60})
    detailed, summary, metadata = run_cross_family_transfer_experiment(
        source_config=source_config,
        target_gene_counts=(10, 14),
        training_kwargs=training_kwargs,
        include_target_oracle=False,
    )

    detailed.to_csv(OUTPUT_DIR / "qaoa_cross_family_transfer_detailed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "qaoa_cross_family_transfer.csv", index=False)
    with (OUTPUT_DIR / "qaoa_cross_family_transfer_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_config": asdict(metadata["source_config"]),
                "training_kwargs": metadata["training_kwargs"],
                "source_training": metadata["source_training"],
                "targets": metadata["targets"],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()