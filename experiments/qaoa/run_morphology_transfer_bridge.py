"""Run cross-domain transfer from transcriptomic graphs to a morphology-derived graph family."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.qaoa.morphology import (
    headline_morphology_benchmark_config,
    run_morphology_transfer_bridge_experiment,
)
from src.qaoa.transcriptomic import TranscriptomicBenchmarkConfig, headline_transcriptomic_benchmark_config


OUTPUT_DIR = Path("outputs") / "tables"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_headline = headline_transcriptomic_benchmark_config()
    source_config = TranscriptomicBenchmarkConfig(
        top_gene_count=source_headline.top_gene_count,
        target_edge_count=source_headline.target_edge_count,
        benchmark_size=4,
        benchmark_seed=source_headline.benchmark_seed,
        adaptation_size=12,
        adaptation_seed=source_headline.adaptation_seed,
        subsample_size=source_headline.subsample_size,
        depth=source_headline.depth,
        num_starts=5,
        maxiter=160,
        training_seed=source_headline.training_seed,
    )
    target_config = headline_morphology_benchmark_config()

    detailed, summary, bridge, metadata = run_morphology_transfer_bridge_experiment(
        source_config=source_config,
        target_config=target_config,
    )

    detailed.to_csv(OUTPUT_DIR / "qaoa_morphology_transfer_detailed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "qaoa_morphology_transfer.csv", index=False)
    bridge.to_csv(OUTPUT_DIR / "qaoa_morphology_concentration_bridge.csv", index=False)
    with (OUTPUT_DIR / "qaoa_morphology_transfer_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_config": asdict(metadata["source_config"]),
                "target_config": asdict(metadata["target_config"]),
                "training_kwargs": metadata["training_kwargs"],
                "source_training": metadata["source_training"],
                "target_training": metadata["target_training"],
                "source_stats": metadata["source_stats"],
                "target_stats": metadata["target_stats"],
                "target_feature_table": metadata["target_feature_table"],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()