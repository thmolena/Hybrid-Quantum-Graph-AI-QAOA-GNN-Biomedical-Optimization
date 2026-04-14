"""Run the extracted-script transcriptomic headline benchmark and ablation suite."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.qaoa.transcriptomic import (
    headline_training_kwargs,
    headline_transcriptomic_benchmark_config,
    run_transcriptomic_headline_benchmark,
)


OUTPUT_DIR = Path("outputs") / "tables"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = headline_transcriptomic_benchmark_config()
    detailed, summary, metadata = run_transcriptomic_headline_benchmark(
        config=config,
        training_kwargs=headline_training_kwargs(),
    )

    detailed.to_csv(OUTPUT_DIR / "qaoa_headline_benchmark_detailed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "qaoa_headline_benchmark_summary.csv", index=False)
    with (OUTPUT_DIR / "qaoa_headline_benchmark_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(metadata["config"]),
                "training": metadata["training"],
                "training_kwargs": metadata["training_kwargs"],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()