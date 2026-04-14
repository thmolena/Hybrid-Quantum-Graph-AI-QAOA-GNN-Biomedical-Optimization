"""Run a cross-family transfer study across transcriptomic weighted graph families."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.qaoa.transcriptomic import (
    headline_training_kwargs,
    headline_transcriptomic_benchmark_config,
    run_cross_family_transfer_experiment,
)


OUTPUT_DIR = Path("outputs") / "tables"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_config = headline_transcriptomic_benchmark_config()
    detailed, summary, metadata = run_cross_family_transfer_experiment(
        source_config=source_config,
        target_gene_counts=(10, 12, 14, 16),
        training_kwargs=headline_training_kwargs(),
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