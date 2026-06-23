from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qaoa.transcriptomic import select_gene_panel


DEFAULT_PANEL_PATH = REPO_ROOT / "data" / "prostate_top32_variance_panel.csv.gz"
DEFAULT_META_PATH = REPO_ROOT / "data" / "prostate_top32_variance_panel_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the cached prostate transcriptomic reduced panel from OpenML.")
    parser.add_argument("--dataset-name", default="prostate")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--top-gene-count", type=int, default=32)
    parser.add_argument("--target-edge-count", type=int, default=18)
    parser.add_argument("--panel-out", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = fetch_openml(name=args.dataset_name, version=args.version, as_frame=True)
    expression_frame_full = dataset.data.apply(pd.to_numeric, errors="coerce")
    expression_frame_full = expression_frame_full.loc[:, expression_frame_full.notna().all(axis=0)]
    labels = pd.Series(dataset.target.astype(str), name=dataset.target_names[0] if dataset.target_names else "class")

    gene_table = select_gene_panel(expression_frame_full, args.top_gene_count)
    selected_genes = gene_table["gene"].tolist()
    expression_frame = expression_frame_full.loc[:, selected_genes].copy()

    args.panel_out.parent.mkdir(parents=True, exist_ok=True)
    cache_frame = expression_frame.copy()
    cache_frame.insert(0, "__sample_id__", expression_frame.index.astype(str))
    cache_frame["__target__"] = labels.to_numpy()
    cache_frame.to_csv(args.panel_out, index=False, compression="gzip")

    metadata = {
        "dataset_name": dataset.details.get("name", args.dataset_name),
        "dataset_version": args.version,
        "full_feature_count": int(expression_frame_full.shape[1]),
        "top_gene_count": int(args.top_gene_count),
        "target_edge_count": int(args.target_edge_count),
        "label_name": labels.name,
        "gene_table": gene_table.to_dict(orient="records"),
    }
    with args.meta_out.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        json.dumps(
            {
                "panel_path": str(args.panel_out),
                "meta_path": str(args.meta_out),
                "samples": int(expression_frame.shape[0]),
                "cached_gene_count": int(expression_frame.shape[1]),
                "full_feature_count": int(expression_frame_full.shape[1]),
                "top_genes_preview": selected_genes[:10],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()