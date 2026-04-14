"""Transcriptomic QAOA benchmark construction and noisy evaluation helpers."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch import optim

from src.gnn import SimpleGCN


DEFAULT_CACHE_PANEL = Path("data/prostate_top10_variance_panel.csv.gz")
DEFAULT_CACHE_META = Path("data/prostate_top10_variance_panel_meta.json")


@dataclass(frozen=True)
class TranscriptomicBenchmarkConfig:
    top_gene_count: int = 10
    target_edge_count: int = 18
    benchmark_size: int = 6
    benchmark_seed: int = 42
    adaptation_size: int = 24
    adaptation_seed: int = 200
    subsample_size: int = 60
    depth: int = 2
    num_starts: int = 8
    maxiter: int = 320
    seed_offset: int = 1000
    training_seed: int = 7


def _load_cached_panel(
    panel_path: Path = DEFAULT_CACHE_PANEL,
    meta_path: Path = DEFAULT_CACHE_META,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, object]]:
    if not panel_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Cached transcriptomic panel not found. Expected data/prostate_top10_variance_panel.csv.gz "
            "and data/prostate_top10_variance_panel_meta.json."
        )

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    panel = pd.read_csv(panel_path, compression="gzip")
    sample_ids = panel.pop("__sample_id__").astype(str)
    labels = pd.Series(panel.pop("__target__").astype(str).to_numpy(), index=sample_ids, name=meta.get("label_name", "class"))
    expression_frame = panel.copy()
    expression_frame.index = sample_ids
    gene_table = pd.DataFrame(meta["gene_table"])
    return expression_frame, labels, gene_table, meta


def build_cut_diagonal(n_qubits: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Return the cut value for every computational basis state."""
    cut_diagonal = np.zeros(2**n_qubits, dtype=np.float64)
    for state_index in range(2**n_qubits):
        bits = [(state_index >> bit) & 1 for bit in range(n_qubits)]
        cut_diagonal[state_index] = sum(bits[u] != bits[v] for u, v in edges)
    return cut_diagonal


def apply_rx_all(state: np.ndarray, n_qubits: int, beta: float) -> np.ndarray:
    rx = np.array(
        [
            [np.cos(beta), -1j * np.sin(beta)],
            [-1j * np.sin(beta), np.cos(beta)],
        ],
        dtype=np.complex128,
    )
    psi = state.reshape((2,) * n_qubits)
    for axis in range(n_qubits):
        psi = np.moveaxis(psi, axis, 0)
        psi = np.tensordot(rx, psi, axes=([1], [0]))
        psi = np.moveaxis(psi, 0, axis)
    return psi.reshape(-1)


def qaoa_state_fast(cut_diagonal: np.ndarray, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    num_states = cut_diagonal.shape[0]
    n_qubits = int(np.log2(num_states))
    state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * cut_diagonal)
        state = apply_rx_all(state, n_qubits, beta)
    return state


def expected_cut_fast(cut_diagonal: np.ndarray, state: np.ndarray) -> float:
    return float(np.dot(cut_diagonal, np.abs(state) ** 2))


def brute_force_maxcut(n_qubits: int, edges: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    best_cut = -1
    best_mask = 0
    for mask in range(1, 2**n_qubits):
        cut_value = sum(1 for u, v in edges if ((mask >> u) & 1) != ((mask >> v) & 1))
        if cut_value > best_cut:
            best_cut = cut_value
            best_mask = mask
    return best_cut, best_mask


def normalize_angles(raw_angles: Sequence[float], depth: int) -> Tuple[np.ndarray, np.ndarray]:
    raw_angles = np.asarray(raw_angles, dtype=np.float64).reshape(-1)
    if raw_angles.size < 2 * depth:
        raise ValueError(f"Expected at least {2 * depth} raw angles, received {raw_angles.size}.")
    gammas = np.mod(raw_angles[:depth], math.pi)
    betas = np.mod(raw_angles[depth : 2 * depth], math.pi / 2)
    return gammas, betas


def qaoa_value_for_angles(cut_diagonal: np.ndarray, gammas: Sequence[float], betas: Sequence[float]) -> Tuple[float, np.ndarray]:
    state = qaoa_state_fast(cut_diagonal, gammas, betas)
    return expected_cut_fast(cut_diagonal, state), state


def classical_optimize_instance(
    instance: Dict[str, object],
    depth: int,
    num_starts: int = 8,
    maxiter: int = 320,
    seed: int = 0,
) -> Dict[str, object]:
    cut_diagonal = instance["cut_diagonal"]
    rng = np.random.default_rng(seed)
    best = None
    started_at = time.perf_counter()

    def objective(raw_angles: np.ndarray) -> float:
        gammas, betas = normalize_angles(raw_angles, depth)
        value, _ = qaoa_value_for_angles(cut_diagonal, gammas, betas)
        return -value

    for _ in range(num_starts):
        x0 = np.concatenate(
            [
                rng.uniform(0.0, math.pi, size=depth),
                rng.uniform(0.0, math.pi / 2, size=depth),
            ]
        )
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6},
        )
        gammas, betas = normalize_angles(result.x, depth)
        value, state = qaoa_value_for_angles(cut_diagonal, gammas, betas)
        candidate = {
            "gammas": gammas,
            "betas": betas,
            "value": value,
            "state": state,
            "nit": result.nit,
            "nfev": result.nfev,
            "success": bool(result.success),
            "raw_angles": np.concatenate([gammas, betas]),
        }
        if best is None or candidate["value"] > best["value"]:
            best = candidate
    best["runtime_ms"] = 1000.0 * (time.perf_counter() - started_at)
    return best


def select_gene_panel(expression_frame: pd.DataFrame, top_gene_count: int) -> pd.DataFrame:
    gene_table = (
        expression_frame.var(axis=0)
        .sort_values(ascending=False)
        .head(top_gene_count)
        .rename("variance")
        .reset_index()
        .rename(columns={"index": "gene"})
    )
    gene_table["rank"] = np.arange(1, len(gene_table) + 1)
    return gene_table[["rank", "gene", "variance"]]


def build_gene_correlation_graph(
    expression_frame: pd.DataFrame,
    gene_table: pd.DataFrame,
    target_edge_count: int,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    genes = gene_table["gene"].tolist()
    correlation_matrix = expression_frame[genes].corr().abs().fillna(0.0).copy()
    correlation_values = correlation_matrix.to_numpy(copy=True)
    np.fill_diagonal(correlation_values, 0.0)
    correlation_matrix.iloc[:, :] = correlation_values

    complete_graph = nx.Graph()
    for gene_index, gene_name in enumerate(genes):
        complete_graph.add_node(gene_index, gene=gene_name)
    for i, gene_i in enumerate(genes):
        for j in range(i + 1, len(genes)):
            gene_j = genes[j]
            complete_graph.add_edge(i, j, weight=float(correlation_matrix.loc[gene_i, gene_j]))

    spanning_tree = nx.maximum_spanning_tree(complete_graph, weight="weight")
    remaining_edges = sorted(
        (
            (u, v, data["weight"])
            for u, v, data in complete_graph.edges(data=True)
            if not spanning_tree.has_edge(u, v)
        ),
        key=lambda item: item[2],
        reverse=True,
    )

    graph = nx.Graph()
    for node_index, gene_name in enumerate(genes):
        graph.add_node(node_index, gene=gene_name)
    for u, v, data in spanning_tree.edges(data=True):
        graph.add_edge(u, v, weight=data["weight"])
    for u, v, weight in remaining_edges:
        if graph.number_of_edges() >= target_edge_count:
            break
        graph.add_edge(u, v, weight=weight)

    edge_table = pd.DataFrame(
        [
            {
                "gene_u": graph.nodes[u]["gene"],
                "gene_v": graph.nodes[v]["gene"],
                "abs_correlation": data["weight"],
            }
            for u, v, data in graph.edges(data=True)
        ]
    ).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return graph, correlation_matrix, edge_table


def stratified_subsample_indices(labels: pd.Series, sample_size: int, seed: int) -> List[str]:
    label_counts = labels.value_counts().sort_index()
    desired = label_counts / label_counts.sum() * sample_size
    counts = np.floor(desired).astype(int)
    remainder = sample_size - int(counts.sum())
    if remainder > 0:
        fractional = (desired - counts).sort_values(ascending=False)
        for label in fractional.index[:remainder]:
            counts.loc[label] += 1

    rng = np.random.default_rng(seed)
    chosen: List[str] = []
    for label, count in counts.items():
        label_indices = labels[labels == label].index.to_numpy()
        chosen.extend(rng.choice(label_indices, size=int(count), replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def create_graph_instance(
    graph_id: int,
    graph: nx.Graph,
    sample_indices: Sequence[str],
    labels: pd.Series,
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
        "class_balance": pd.Series(labels.loc[list(sample_indices)]).value_counts().sort_index().to_dict(),
        "gene_labels": [graph.nodes[node]["gene"] for node in graph.nodes()],
    }


def build_graph_split(
    expression_frame: pd.DataFrame,
    labels: pd.Series,
    gene_table: pd.DataFrame,
    target_edge_count: int,
    split_name: str,
    split_size: int,
    subsample_size: int,
    base_seed: int,
) -> List[Dict[str, object]]:
    instances = []
    for offset in range(split_size):
        graph_seed = base_seed + offset
        sample_indices = stratified_subsample_indices(labels, subsample_size, seed=graph_seed)
        subset_expression = expression_frame.loc[sample_indices]
        graph, corr_matrix, edge_table = build_gene_correlation_graph(subset_expression, gene_table, target_edge_count)
        instance = create_graph_instance(graph_seed, graph, sample_indices, labels, split_name)
        instance["correlation_matrix"] = corr_matrix
        instance["edge_table"] = edge_table
        instances.append(instance)
    return instances


def build_transcriptomic_benchmark(
    config: TranscriptomicBenchmarkConfig | None = None,
    panel_path: Path = DEFAULT_CACHE_PANEL,
    meta_path: Path = DEFAULT_CACHE_META,
) -> Dict[str, object]:
    config = config or TranscriptomicBenchmarkConfig()
    expression_frame, labels, gene_table, meta = _load_cached_panel(panel_path, meta_path)
    if len(gene_table) != config.top_gene_count:
        gene_table = select_gene_panel(expression_frame, config.top_gene_count)

    representative_graph, representative_corr, representative_edge_table = build_gene_correlation_graph(
        expression_frame,
        gene_table,
        config.target_edge_count,
    )
    representative = create_graph_instance(
        graph_id=0,
        graph=representative_graph,
        sample_indices=expression_frame.index.tolist(),
        labels=labels,
        split_name="representative",
    )
    representative["correlation_matrix"] = representative_corr
    representative["edge_table"] = representative_edge_table

    adaptation_instances = build_graph_split(
        expression_frame,
        labels,
        gene_table,
        config.target_edge_count,
        "adaptation",
        config.adaptation_size,
        config.subsample_size,
        config.adaptation_seed,
    )
    benchmark_instances = build_graph_split(
        expression_frame,
        labels,
        gene_table,
        config.target_edge_count,
        "benchmark",
        config.benchmark_size,
        config.subsample_size,
        config.benchmark_seed,
    )
    return {
        "config": config,
        "meta": meta,
        "expression_frame": expression_frame,
        "labels": labels,
        "gene_table": gene_table,
        "representative": representative,
        "adaptation_instances": adaptation_instances,
        "benchmark_instances": benchmark_instances,
    }


def attach_classical_targets(instances: Sequence[Dict[str, object]], config: TranscriptomicBenchmarkConfig) -> List[Dict[str, object]]:
    enriched_instances = []
    for index, instance in enumerate(instances):
        reference = classical_optimize_instance(
            instance,
            depth=config.depth,
            num_starts=config.num_starts,
            maxiter=config.maxiter,
            seed=config.seed_offset + int(instance["graph_id"]) + index,
        )
        enriched = dict(instance)
        enriched["classical_reference"] = reference
        enriched["target_angles"] = np.concatenate([reference["gammas"], reference["betas"]]).astype(np.float32)
        enriched_instances.append(enriched)
    return enriched_instances


def train_adapted_qaoa_gnn(
    train_instances: Sequence[Dict[str, object]],
    depth: int,
    hidden_dim: int = 64,
    epochs: int = 500,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    seed: int = 7,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    trained_model = SimpleGCN(in_feats=1, hidden=hidden_dim, out_feats=2, p=depth)
    optimizer = optim.Adam(trained_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(trained_model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        trained_model.train()
        running_loss = 0.0
        for instance in train_instances:
            adjacency_tensor = torch.tensor(instance["adjacency"], dtype=torch.float32)
            feature_tensor = torch.tensor(instance["features"], dtype=torch.float32)
            target_tensor = torch.tensor(instance["target_angles"], dtype=torch.float32)

            prediction = trained_model(feature_tensor, adjacency_tensor).view(-1)
            loss = ((prediction - target_tensor) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        mean_loss = running_loss / max(1, len(train_instances))
        loss_history.append(mean_loss)
        if mean_loss + 1e-8 < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = copy.deepcopy(trained_model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    trained_model.load_state_dict(best_state)
    trained_model.eval()
    return {
        "model": trained_model,
        "history": loss_history,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(loss_history),
    }


def predict_instance_with_gnn(instance: Dict[str, object], model: SimpleGCN, depth: int) -> Dict[str, object]:
    adjacency_tensor = torch.tensor(instance["adjacency"], dtype=torch.float32)
    feature_tensor = torch.tensor(instance["features"], dtype=torch.float32)
    with torch.no_grad():
        raw_output = model(feature_tensor, adjacency_tensor).view(-1).cpu().numpy()
    gammas, betas = normalize_angles(raw_output, depth)
    value, state = qaoa_value_for_angles(instance["cut_diagonal"], gammas, betas)
    return {
        "raw_output": raw_output,
        "gammas": gammas,
        "betas": betas,
        "value": value,
        "state": state,
    }


def target_edge_count_for_gene_count(top_gene_count: int, target_density: float = 0.4) -> int:
    if top_gene_count < 2:
        raise ValueError("top_gene_count must be at least 2")
    max_edges = top_gene_count * (top_gene_count - 1) // 2
    suggested_edges = int(round(target_density * max_edges))
    return min(max(top_gene_count - 1, suggested_edges), max_edges)


def evaluate_transcriptomic_benchmark(
    benchmark_instances: Sequence[Dict[str, object]],
    model: SimpleGCN,
    adaptation_instances: Sequence[Dict[str, object]],
    depth: int,
) -> pd.DataFrame:
    heuristic_angles = np.mean([instance["target_angles"] for instance in adaptation_instances], axis=0)
    rows: List[Dict[str, object]] = []

    for instance in benchmark_instances:
        graph_id = int(instance["graph_id"])
        best_cut = float(instance["best_cut"])

        inference_started_at = time.perf_counter()
        learned = predict_instance_with_gnn(instance, model, depth)
        learned_runtime_ms = 1000.0 * (time.perf_counter() - inference_started_at)

        heuristic_started_at = time.perf_counter()
        heuristic_gammas, heuristic_betas = normalize_angles(heuristic_angles, depth)
        heuristic_value, _ = qaoa_value_for_angles(instance["cut_diagonal"], heuristic_gammas, heuristic_betas)
        heuristic_runtime_ms = 1000.0 * (time.perf_counter() - heuristic_started_at)

        classical_reference = instance["classical_reference"]
        rows.extend(
            [
                {
                    "method": "Classical depth-2 search",
                    "graph_id": graph_id,
                    "num_nodes": int(instance["n"]),
                    "num_edges": int(instance["edge_count"]),
                    "expected_cut": float(classical_reference["value"]),
                    "best_cut": best_cut,
                    "approximation_ratio": float(classical_reference["value"] / best_cut),
                    "runtime_ms": float(classical_reference.get("runtime_ms", np.nan)),
                },
                {
                    "method": "Heuristic mean-angle initializer",
                    "graph_id": graph_id,
                    "num_nodes": int(instance["n"]),
                    "num_edges": int(instance["edge_count"]),
                    "expected_cut": float(heuristic_value),
                    "best_cut": best_cut,
                    "approximation_ratio": float(heuristic_value / best_cut),
                    "runtime_ms": heuristic_runtime_ms,
                },
                {
                    "method": "Graph-conditioned GNN (ours)",
                    "graph_id": graph_id,
                    "num_nodes": int(instance["n"]),
                    "num_edges": int(instance["edge_count"]),
                    "expected_cut": float(learned["value"]),
                    "best_cut": best_cut,
                    "approximation_ratio": float(learned["value"] / best_cut),
                    "runtime_ms": learned_runtime_ms,
                },
            ]
        )

    return pd.DataFrame(rows)


def summarize_transcriptomic_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            num_nodes=("num_nodes", "median"),
            num_edges=("num_edges", "median"),
            mean_ratio=("approximation_ratio", "mean"),
            std_ratio=("approximation_ratio", "std"),
            median_runtime_ms=("runtime_ms", "median"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    summary["std_ratio"] = summary["std_ratio"].fillna(0.0)

    classical_mean_ratio = float(
        summary.loc[summary["method"] == "Classical depth-2 search", "mean_ratio"].iloc[0]
    )
    summary["retention_vs_classical"] = summary["mean_ratio"] / classical_mean_ratio
    return summary


def run_transcriptomic_generalization_benchmark(
    config: TranscriptomicBenchmarkConfig | None = None,
    training_kwargs: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    config = config or TranscriptomicBenchmarkConfig()
    bundle = build_transcriptomic_benchmark(config)
    adaptation_instances = attach_classical_targets(bundle["adaptation_instances"], config)
    benchmark_instances = attach_classical_targets(bundle["benchmark_instances"], config)

    fit_kwargs = {
        "depth": config.depth,
        "seed": config.training_seed,
    }
    if training_kwargs:
        fit_kwargs.update(training_kwargs)
    training_result = train_adapted_qaoa_gnn(adaptation_instances, **fit_kwargs)

    benchmark_frame = evaluate_transcriptomic_benchmark(
        benchmark_instances,
        training_result["model"],
        adaptation_instances,
        config.depth,
    )
    summary = summarize_transcriptomic_benchmark(benchmark_frame)
    metadata = {
        "config": config,
        "training": {
            "best_loss": training_result["best_loss"],
            "best_epoch": training_result["best_epoch"],
            "epochs_run": training_result["epochs_run"],
        },
    }
    return benchmark_frame, summary, metadata


def rx_unitary(beta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(beta), -1j * np.sin(beta)],
            [-1j * np.sin(beta), np.cos(beta)],
        ],
        dtype=np.complex128,
    )


def apply_single_qubit_unitary_density(
    density: np.ndarray,
    unitary: np.ndarray,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    identity = np.eye(2, dtype=np.complex128)
    operator = np.array([[1.0 + 0.0j]])
    for index in range(n_qubits):
        operator = np.kron(operator, unitary if index == qubit else identity)
    return operator @ density @ operator.conj().T


def apply_local_depolarizing(density: np.ndarray, error_rate: float, n_qubits: int) -> np.ndarray:
    if error_rate <= 0.0:
        return density
    paulis = [
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    ]
    mixed = density
    for qubit in range(n_qubits):
        updated = (1.0 - error_rate) * mixed
        for pauli in paulis:
            updated += (error_rate / 3.0) * apply_single_qubit_unitary_density(mixed, pauli, qubit, n_qubits)
        mixed = updated
    return mixed


def qaoa_density_with_local_depolarizing(
    cut_diagonal: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    error_rate: float,
) -> np.ndarray:
    state = np.ones(cut_diagonal.shape[0], dtype=np.complex128) / np.sqrt(cut_diagonal.shape[0])
    density = np.outer(state, state.conj())
    n_qubits = int(np.log2(cut_diagonal.shape[0]))
    phase = np.eye(cut_diagonal.shape[0], dtype=np.complex128)
    for gamma, beta in zip(gammas, betas):
        phase = np.diag(np.exp(-1j * gamma * cut_diagonal))
        density = phase @ density @ phase.conj().T
        density = apply_local_depolarizing(density, error_rate, n_qubits)
        for qubit in range(n_qubits):
            density = apply_single_qubit_unitary_density(density, rx_unitary(beta), qubit, n_qubits)
        density = apply_local_depolarizing(density, error_rate, n_qubits)
    return density


def noisy_expected_cut(cut_diagonal: np.ndarray, density: np.ndarray) -> float:
    diagonal_probabilities = np.real(np.diag(density))
    return float(np.dot(cut_diagonal, diagonal_probabilities))


def evaluate_method_under_noise(
    benchmark_instances: Sequence[Dict[str, object]],
    method_name: str,
    angle_lookup: Dict[int, np.ndarray],
    error_rates: Iterable[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for error_rate in error_rates:
        ratios = []
        for instance in benchmark_instances:
            raw_angles = angle_lookup[int(instance["graph_id"])]
            gammas, betas = normalize_angles(raw_angles, raw_angles.size // 2)
            density = qaoa_density_with_local_depolarizing(instance["cut_diagonal"], gammas, betas, error_rate)
            value = noisy_expected_cut(instance["cut_diagonal"], density)
            ratio = value / float(instance["best_cut"])
            ratios.append(ratio)
            rows.append(
                {
                    "method": method_name,
                    "graph_id": int(instance["graph_id"]),
                    "noise_rate": error_rate,
                    "noisy_value": value,
                    "noisy_ratio": ratio,
                }
            )
        rows.append(
            {
                "method": method_name,
                "graph_id": "mean",
                "noise_rate": error_rate,
                "noisy_value": float(np.mean(ratios)),
                "noisy_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios, ddof=0)),
            }
        )
    return rows


def run_transcriptomic_noise_experiment(
    config: TranscriptomicBenchmarkConfig | None = None,
    noise_rates: Sequence[float] = (0.0, 0.01, 0.02, 0.05),
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    config = config or TranscriptomicBenchmarkConfig()
    bundle = build_transcriptomic_benchmark(config)
    adaptation_instances = attach_classical_targets(bundle["adaptation_instances"], config)
    benchmark_instances = attach_classical_targets(bundle["benchmark_instances"], config)
    training_result = train_adapted_qaoa_gnn(adaptation_instances, depth=config.depth, seed=config.training_seed)
    model = training_result["model"]

    learned_lookup: Dict[int, np.ndarray] = {}
    classical_lookup: Dict[int, np.ndarray] = {}
    heuristic_angles = np.mean([instance["target_angles"] for instance in adaptation_instances], axis=0)
    heuristic_lookup: Dict[int, np.ndarray] = {}
    for instance in benchmark_instances:
        graph_id = int(instance["graph_id"])
        prediction = predict_instance_with_gnn(instance, model, config.depth)
        learned_lookup[graph_id] = np.concatenate([prediction["gammas"], prediction["betas"]])
        classical_lookup[graph_id] = instance["target_angles"]
        heuristic_lookup[graph_id] = heuristic_angles

    rows: List[Dict[str, object]] = []
    rows.extend(evaluate_method_under_noise(benchmark_instances, "Graph-conditioned GNN (ours)", learned_lookup, noise_rates))
    rows.extend(evaluate_method_under_noise(benchmark_instances, "Classical depth-2 search angles", classical_lookup, noise_rates))
    rows.extend(evaluate_method_under_noise(benchmark_instances, "Heuristic mean-angle initializer", heuristic_lookup, noise_rates))

    frame = pd.DataFrame(rows)
    summary = frame[frame["graph_id"] == "mean"].copy()
    summary = summary[["method", "noise_rate", "noisy_ratio", "std_ratio"]].rename(columns={"noisy_ratio": "mean_ratio"})
    summary = summary.sort_values(["noise_rate", "method"]).reset_index(drop=True)
    metadata = {
        "config": config,
        "training": {
            "best_loss": training_result["best_loss"],
            "best_epoch": training_result["best_epoch"],
            "epochs_run": training_result["epochs_run"],
        },
        "noise_rates": list(noise_rates),
    }
    return summary, metadata
