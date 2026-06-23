"""Morphology-derived weighted graph families and cross-domain QAOA transfer helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.qaoa.transcriptomic import (
    TranscriptomicBenchmarkConfig,
    attach_classical_targets,
    build_gene_correlation_graph,
    build_prior_style_regressor,
    build_transcriptomic_benchmark,
    classical_optimize_instance,
    evaluate_angle_initializer,
    graph_descriptor,
    headline_training_kwargs,
    headline_transcriptomic_benchmark_config,
    normalize_angles,
    predict_instance_with_gnn,
    qaoa_value_for_angles,
    train_adapted_qaoa_gnn,
    build_cut_diagonal,
    brute_force_maxcut,
)


DEFAULT_DATA_PATH = Path("data/breast_cancer.csv")


@dataclass(frozen=True)
class MorphologyBenchmarkConfig:
    top_feature_count: int = 12
    target_edge_count: int = 18
    benchmark_size: int = 4
    benchmark_seed: int = 84
    adaptation_size: int = 12
    adaptation_seed: int = 400
    subsample_size: int = 100
    depth: int = 2
    num_starts: int = 5
    maxiter: int = 160
    seed_offset: int = 5000
    training_seed: int = 31


def headline_morphology_benchmark_config() -> MorphologyBenchmarkConfig:
    return MorphologyBenchmarkConfig()


def _load_morphology_frame(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Morphology feature table not found: {data_path}")
    frame = pd.read_csv(data_path)
    frame.index = [f"morphology_{index:04d}" for index in range(len(frame))]
    return frame


def select_morphology_panel(feature_frame: pd.DataFrame, top_feature_count: int) -> pd.DataFrame:
    correlation_matrix = feature_frame.corr().abs().fillna(0.0)
    values = correlation_matrix.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)
    correlation_matrix.iloc[:, :] = values
    ranking = correlation_matrix.mean(axis=1).sort_values(ascending=False).head(top_feature_count)
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(ranking) + 1),
            "gene": ranking.index.to_list(),
            "mean_abs_correlation": ranking.to_numpy(),
        }
    )


def random_subsample_indices(index: Sequence[str], sample_size: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(index), size=sample_size, replace=False).tolist()
    rng.shuffle(chosen)
    return chosen


def create_morphology_instance(
    graph_id: int,
    graph: nx.Graph,
    sample_indices: Sequence[str],
    split_name: str,
) -> Dict[str, object]:
    n_qubits = graph.number_of_nodes()
    edges = list(graph.edges())
    best_cut, best_mask = brute_force_maxcut(n_qubits, edges)
    adjacency = nx.to_numpy_array(graph, dtype=np.float64) + np.eye(n_qubits)
    features = adjacency.sum(axis=1, keepdims=True).astype(np.float32)
    return {
        "graph_id": graph_id,
        "split": split_name,
        "graph": graph,
        "n": n_qubits,
        "edges": edges,
        "edge_count": len(edges),
        "density": nx.density(graph),
        "adjacency": adjacency,
        "features": features,
        "cut_diagonal": build_cut_diagonal(n_qubits, edges),
        "best_cut": best_cut,
        "best_mask": best_mask,
        "sample_indices": list(sample_indices),
        "sample_count": len(sample_indices),
        "gene_labels": [graph.nodes[node]["gene"] for node in graph.nodes()],
    }


def build_morphology_split(
    feature_frame: pd.DataFrame,
    feature_table: pd.DataFrame,
    target_edge_count: int,
    split_name: str,
    split_size: int,
    subsample_size: int,
    base_seed: int,
) -> List[Dict[str, object]]:
    instances: List[Dict[str, object]] = []
    for offset in range(split_size):
        graph_seed = base_seed + offset
        sample_indices = random_subsample_indices(feature_frame.index.to_list(), subsample_size, seed=graph_seed)
        subset_frame = feature_frame.loc[sample_indices]
        graph, corr_matrix, edge_table = build_gene_correlation_graph(subset_frame, feature_table, target_edge_count)
        instance = create_morphology_instance(graph_seed, graph, sample_indices, split_name)
        instance["correlation_matrix"] = corr_matrix
        instance["edge_table"] = edge_table
        instances.append(instance)
    return instances


def build_morphology_benchmark(
    config: MorphologyBenchmarkConfig | None = None,
    data_path: Path = DEFAULT_DATA_PATH,
) -> Dict[str, object]:
    config = config or MorphologyBenchmarkConfig()
    feature_frame = _load_morphology_frame(data_path)
    feature_table = select_morphology_panel(feature_frame, config.top_feature_count)

    representative_graph, representative_corr, representative_edge_table = build_gene_correlation_graph(
        feature_frame,
        feature_table,
        config.target_edge_count,
    )
    representative = create_morphology_instance(
        graph_id=0,
        graph=representative_graph,
        sample_indices=feature_frame.index.tolist(),
        split_name="representative",
    )
    representative["correlation_matrix"] = representative_corr
    representative["edge_table"] = representative_edge_table

    adaptation_instances = build_morphology_split(
        feature_frame,
        feature_table,
        config.target_edge_count,
        "adaptation",
        config.adaptation_size,
        config.subsample_size,
        config.adaptation_seed,
    )
    benchmark_instances = build_morphology_split(
        feature_frame,
        feature_table,
        config.target_edge_count,
        "benchmark",
        config.benchmark_size,
        config.subsample_size,
        config.benchmark_seed,
    )
    return {
        "config": config,
        "meta": {
            "dataset": str(data_path),
            "family_name": "breast_cancer_morphology",
            "selection_rule": "top mean absolute correlation",
        },
        "feature_frame": feature_frame,
        "feature_table": feature_table,
        "representative": representative,
        "adaptation_instances": adaptation_instances,
        "benchmark_instances": benchmark_instances,
    }


def angle_concentration_stats(instances: Sequence[Dict[str, object]], family_name: str) -> Dict[str, object]:
    angles = np.vstack([np.asarray(instance["target_angles"], dtype=np.float64) for instance in instances])
    centroid = angles.mean(axis=0)
    distances = np.linalg.norm(angles - centroid, axis=1)
    return {
        "family": family_name,
        "num_instances": int(len(instances)),
        "centroid": centroid,
        "mean_radius": float(distances.mean()),
        "rms_radius": float(np.sqrt(np.mean(distances**2))),
        "max_radius": float(distances.max()),
    }


def _evaluate_initializer_with_angles(
    instance: Dict[str, object],
    method: str,
    raw_angles: np.ndarray,
    proposal_ms: float,
    depth: int,
    source_centroid: np.ndarray | None = None,
) -> Dict[str, object]:
    normalized = np.concatenate(normalize_angles(raw_angles, depth)).astype(np.float64)
    row = evaluate_angle_initializer(instance, method, normalized, proposal_ms, depth)
    target_angles = np.asarray(instance["target_angles"], dtype=np.float64)
    row["angle_l2_to_target"] = float(np.linalg.norm(normalized - target_angles))
    row["raw_angles"] = normalized
    if source_centroid is not None:
        row["distance_to_source_centroid"] = float(np.linalg.norm(normalized - source_centroid))
    else:
        row["distance_to_source_centroid"] = float("nan")
    return row


def summarize_domain_transfer(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            num_nodes=("num_nodes", "median"),
            num_edges=("num_edges", "median"),
            mean_ratio=("approximation_ratio", "mean"),
            std_ratio=("approximation_ratio", "std"),
            mean_retention=("retention_vs_classical", "mean"),
            mean_angle_l2=("angle_l2_to_target", "mean"),
            mean_distance_to_source_centroid=("distance_to_source_centroid", "mean"),
            median_runtime_ms=("total_ms", "median"),
        )
        .reset_index(drop=True)
    )
    summary["std_ratio"] = summary["std_ratio"].fillna(0.0)
    method_order = [
        "Classical depth-2 search",
        "Transcriptomic heuristic",
        "Transcriptomic descriptor regressor",
        "Transcriptomic GNN transfer",
        "Morphology heuristic",
        "Morphology descriptor regressor",
        "Morphology GNN (oracle)",
    ]
    summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
    return summary.sort_values("method").reset_index(drop=True)


def concentration_bridge_summary(
    source_stats: Dict[str, object],
    target_stats: Dict[str, object],
    detailed_rows: pd.DataFrame,
) -> pd.DataFrame:
    centroid_distance = float(np.linalg.norm(source_stats["centroid"] - target_stats["centroid"]))
    bridge_rows = [
        {
            "metric": "source_num_instances",
            "value": float(source_stats["num_instances"]),
        },
        {
            "metric": "target_num_instances",
            "value": float(target_stats["num_instances"]),
        },
        {
            "metric": "source_mean_radius",
            "value": float(source_stats["mean_radius"]),
        },
        {
            "metric": "source_rms_radius",
            "value": float(source_stats["rms_radius"]),
        },
        {
            "metric": "source_max_radius",
            "value": float(source_stats["max_radius"]),
        },
        {
            "metric": "target_mean_radius",
            "value": float(target_stats["mean_radius"]),
        },
        {
            "metric": "target_rms_radius",
            "value": float(target_stats["rms_radius"]),
        },
        {
            "metric": "target_max_radius",
            "value": float(target_stats["max_radius"]),
        },
        {
            "metric": "centroid_distance",
            "value": centroid_distance,
        },
        {
            "metric": "rms_separation_margin",
            "value": centroid_distance - float(source_stats["rms_radius"]) - float(target_stats["rms_radius"]),
        },
        {
            "metric": "max_separation_margin",
            "value": centroid_distance - float(source_stats["max_radius"]) - float(target_stats["max_radius"]),
        },
    ]

    transfer_rows = detailed_rows[detailed_rows["method"] == "Transcriptomic GNN transfer"]
    if not transfer_rows.empty:
        transfer_mean_distance = float(transfer_rows["distance_to_source_centroid"].mean())
        target_max_radius = float(target_stats["max_radius"])
        bridge_rows.extend(
            [
                {
                    "metric": "transfer_mean_angle_l2",
                    "value": float(transfer_rows["angle_l2_to_target"].mean()),
                },
                {
                    "metric": "transfer_mean_distance_to_source_centroid",
                    "value": transfer_mean_distance,
                },
                {
                    "metric": "transfer_mean_retention",
                    "value": float(transfer_rows["retention_vs_classical"].mean()),
                },
                {
                    "metric": "transfer_excursion_beyond_source_max_radius",
                    "value": transfer_mean_distance - float(source_stats["max_radius"]),
                },
                {
                    "metric": "transfer_angle_error_lower_bound",
                    "value": transfer_mean_distance - centroid_distance - target_max_radius,
                },
            ]
        )
    return pd.DataFrame(bridge_rows)


def run_morphology_transfer_bridge_experiment(
    source_config: TranscriptomicBenchmarkConfig | None = None,
    target_config: MorphologyBenchmarkConfig | None = None,
    training_kwargs: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    source_config = source_config or TranscriptomicBenchmarkConfig(
        top_gene_count=headline_transcriptomic_benchmark_config().top_gene_count,
        target_edge_count=headline_transcriptomic_benchmark_config().target_edge_count,
        benchmark_size=4,
        benchmark_seed=headline_transcriptomic_benchmark_config().benchmark_seed,
        adaptation_size=12,
        adaptation_seed=headline_transcriptomic_benchmark_config().adaptation_seed,
        subsample_size=headline_transcriptomic_benchmark_config().subsample_size,
        depth=headline_transcriptomic_benchmark_config().depth,
        num_starts=5,
        maxiter=160,
        training_seed=headline_transcriptomic_benchmark_config().training_seed,
    )
    target_config = target_config or headline_morphology_benchmark_config()

    fit_kwargs = headline_training_kwargs()
    fit_kwargs.update({"epochs": 500, "patience": 60})
    if training_kwargs:
        fit_kwargs.update(training_kwargs)

    source_bundle = build_transcriptomic_benchmark(source_config)
    source_adaptation = attach_classical_targets(source_bundle["adaptation_instances"], source_config)
    source_training = train_adapted_qaoa_gnn(source_adaptation, depth=source_config.depth, **fit_kwargs)
    source_prior = build_prior_style_regressor(source_adaptation, seed=int(fit_kwargs.get("seed", 19)))
    source_heuristic = np.mean([instance["target_angles"] for instance in source_adaptation], axis=0)
    source_stats = angle_concentration_stats(source_adaptation, "transcriptomic")

    target_bundle = build_morphology_benchmark(target_config)
    target_adaptation = attach_classical_targets(target_bundle["adaptation_instances"], target_config)
    target_benchmark = attach_classical_targets(target_bundle["benchmark_instances"], target_config)
    target_training = train_adapted_qaoa_gnn(target_adaptation, depth=target_config.depth, **fit_kwargs)
    target_prior = build_prior_style_regressor(target_adaptation, seed=int(fit_kwargs.get("seed", 19)))
    target_heuristic = np.mean([instance["target_angles"] for instance in target_adaptation], axis=0)
    target_stats = angle_concentration_stats(target_adaptation, "morphology")

    rows: List[Dict[str, object]] = []
    source_centroid = np.asarray(source_stats["centroid"], dtype=np.float64)
    for instance in target_benchmark:
        best_cut = float(instance["best_cut"])
        classical_reference = instance["classical_reference"]
        rows.append(
            {
                "method": "Classical depth-2 search",
                "graph_id": int(instance["graph_id"]),
                "num_nodes": int(instance["n"]),
                "num_edges": int(instance["edge_count"]),
                "expected_cut": float(classical_reference["value"]),
                "best_cut": best_cut,
                "approximation_ratio": float(classical_reference["value"] / best_cut),
                "classical_ratio": float(classical_reference["value"] / best_cut),
                "retention_vs_classical": 1.0,
                "proposal_ms": float(classical_reference.get("runtime_ms", np.nan)),
                "evaluation_ms": 0.0,
                "total_ms": float(classical_reference.get("runtime_ms", np.nan)),
                "angle_l2_to_target": 0.0,
                "distance_to_source_centroid": float(np.linalg.norm(np.asarray(instance["target_angles"], dtype=np.float64) - source_centroid)),
            }
        )

        start = time.perf_counter()
        source_heuristic_raw = np.asarray(source_heuristic, dtype=np.float64).copy()
        proposal_ms = 1000.0 * (time.perf_counter() - start)
        rows.append(
            _evaluate_initializer_with_angles(
                instance,
                "Transcriptomic heuristic",
                source_heuristic_raw,
                proposal_ms,
                target_config.depth,
                source_centroid=source_centroid,
            )
        )

        descriptor = graph_descriptor(instance).reshape(1, -1)
        start = time.perf_counter()
        source_prior_raw = source_prior.predict(descriptor).reshape(-1)
        proposal_ms = 1000.0 * (time.perf_counter() - start)
        rows.append(
            _evaluate_initializer_with_angles(
                instance,
                "Transcriptomic descriptor regressor",
                source_prior_raw,
                proposal_ms,
                target_config.depth,
                source_centroid=source_centroid,
            )
        )

        start = time.perf_counter()
        source_prediction = predict_instance_with_gnn(instance, source_training["model"], target_config.depth)
        proposal_ms = 1000.0 * (time.perf_counter() - start)
        rows.append(
            _evaluate_initializer_with_angles(
                instance,
                "Transcriptomic GNN transfer",
                source_prediction["raw_output"],
                proposal_ms,
                target_config.depth,
                source_centroid=source_centroid,
            )
        )

        start = time.perf_counter()
        target_heuristic_raw = np.asarray(target_heuristic, dtype=np.float64).copy()
        proposal_ms = 1000.0 * (time.perf_counter() - start)
        rows.append(
            _evaluate_initializer_with_angles(
                instance,
                "Morphology heuristic",
                target_heuristic_raw,
                proposal_ms,
                target_config.depth,
                source_centroid=source_centroid,
            )
        )

        start = time.perf_counter()
        target_prior_raw = target_prior.predict(descriptor).reshape(-1)
        proposal_ms = 1000.0 * (time.perf_counter() - start)
        rows.append(
            _evaluate_initializer_with_angles(
                instance,
                "Morphology descriptor regressor",
                target_prior_raw,
                proposal_ms,
                target_config.depth,
                source_centroid=source_centroid,
            )
        )

        start = time.perf_counter()
        target_prediction = predict_instance_with_gnn(instance, target_training["model"], target_config.depth)
        proposal_ms = 1000.0 * (time.perf_counter() - start)
        rows.append(
            _evaluate_initializer_with_angles(
                instance,
                "Morphology GNN (oracle)",
                target_prediction["raw_output"],
                proposal_ms,
                target_config.depth,
                source_centroid=source_centroid,
            )
        )

    detailed = pd.DataFrame(rows)
    summary = summarize_domain_transfer(detailed)
    bridge = concentration_bridge_summary(source_stats, target_stats, detailed)
    metadata = {
        "source_config": source_config,
        "target_config": target_config,
        "training_kwargs": fit_kwargs,
        "source_training": {
            "best_loss": float(source_training["best_loss"]),
            "best_epoch": int(source_training["best_epoch"]),
            "epochs_run": int(source_training["epochs_run"]),
        },
        "target_training": {
            "best_loss": float(target_training["best_loss"]),
            "best_epoch": int(target_training["best_epoch"]),
            "epochs_run": int(target_training["epochs_run"]),
        },
        "source_stats": {
            "family": str(source_stats["family"]),
            "num_instances": int(source_stats["num_instances"]),
            "centroid": np.asarray(source_stats["centroid"], dtype=np.float64).tolist(),
            "mean_radius": float(source_stats["mean_radius"]),
            "rms_radius": float(source_stats["rms_radius"]),
            "max_radius": float(source_stats["max_radius"]),
        },
        "target_stats": {
            "family": str(target_stats["family"]),
            "num_instances": int(target_stats["num_instances"]),
            "centroid": np.asarray(target_stats["centroid"], dtype=np.float64).tolist(),
            "mean_radius": float(target_stats["mean_radius"]),
            "rms_radius": float(target_stats["rms_radius"]),
            "max_radius": float(target_stats["max_radius"]),
        },
        "target_feature_table": json.loads(target_bundle["feature_table"].to_json(orient="records")),
    }
    return detailed, summary, bridge, metadata