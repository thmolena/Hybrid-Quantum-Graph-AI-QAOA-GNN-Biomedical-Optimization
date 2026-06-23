from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd

from src.common.io import write_records


@dataclass(frozen=True)
class WeightedGraphInstance:
    node_names: list[str]
    weighted_edges: list[tuple[int, int, float]]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)


def load_weighted_graph(graph_path: str | Path) -> WeightedGraphInstance:
    frame = pd.read_csv(graph_path)
    expected_columns = {"source", "target", "weight"}
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing graph columns: {sorted(missing)}")

    node_names = sorted(set(frame["source"]).union(frame["target"]))
    node_to_index = {name: index for index, name in enumerate(node_names)}
    weighted_edges = [
        (node_to_index[row.source], node_to_index[row.target], float(row.weight))
        for row in frame.itertuples(index=False)
    ]
    return WeightedGraphInstance(node_names=node_names, weighted_edges=weighted_edges)


def load_reference_angles(angles_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(angles_path)
    if frame.empty:
        raise ValueError("Angle file is empty")

    row = frame.iloc[0]
    gamma_columns = sorted([column for column in frame.columns if column.startswith("gamma_")])
    beta_columns = sorted([column for column in frame.columns if column.startswith("beta_")])
    if not gamma_columns or not beta_columns:
        raise ValueError("Angle file must contain gamma_* and beta_* columns")

    gammas = row[gamma_columns].to_numpy(dtype=float)
    betas = row[beta_columns].to_numpy(dtype=float)
    return gammas, betas


def _weighted_cut_values(num_nodes: int, weighted_edges: Iterable[tuple[int, int, float]]) -> np.ndarray:
    num_states = 2**num_nodes
    values = np.zeros(num_states, dtype=float)
    for state_index in range(num_states):
        bits = [(state_index >> bit_index) & 1 for bit_index in range(num_nodes)]
        cut_value = 0.0
        for source, target, weight in weighted_edges:
            if bits[source] != bits[target]:
                cut_value += weight
        values[state_index] = cut_value
    return values


def exact_weighted_maxcut(num_nodes: int, weighted_edges: Iterable[tuple[int, int, float]]) -> float:
    return float(_weighted_cut_values(num_nodes, weighted_edges).max())


def qaoa_state_weighted(
    num_nodes: int,
    weighted_edges: Iterable[tuple[int, int, float]],
    gammas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    num_states = 2**num_nodes
    state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    cut_values = _weighted_cut_values(num_nodes, weighted_edges)
    if len(gammas) != len(betas):
        raise ValueError("Expected matching gamma and beta lengths")

    for gamma, beta in zip(gammas, betas):
        phase = np.exp(-1j * gamma * cut_values)
        state = state * phase
        state = _apply_mixer(state, num_nodes, beta)
    return state


def expected_weighted_cut(
    num_nodes: int,
    weighted_edges: Iterable[tuple[int, int, float]],
    state: np.ndarray,
) -> float:
    cut_values = _weighted_cut_values(num_nodes, weighted_edges)
    probabilities = np.abs(state) ** 2
    return float(np.dot(probabilities, cut_values))


def _apply_mixer(state: np.ndarray, num_nodes: int, beta: float) -> np.ndarray:
    mixed_state = state.copy()
    cosine = np.cos(beta)
    sine = -1j * np.sin(beta)
    for qubit_index in range(num_nodes):
        stride = 2**qubit_index
        block = stride * 2
        for base in range(0, mixed_state.shape[0], block):
            for offset in range(stride):
                index_zero = base + offset
                index_one = index_zero + stride
                amplitude_zero = mixed_state[index_zero]
                amplitude_one = mixed_state[index_one]
                mixed_state[index_zero] = cosine * amplitude_zero + sine * amplitude_one
                mixed_state[index_one] = sine * amplitude_zero + cosine * amplitude_one
    return mixed_state


def evaluate_angle_baseline(
    name: str,
    instance: WeightedGraphInstance,
    gammas: np.ndarray,
    betas: np.ndarray,
    exact_cut: float,
    runtime_ms: float | None = None,
) -> dict[str, object]:
    state = qaoa_state_weighted(instance.num_nodes, instance.weighted_edges, gammas, betas)
    expected_cut = expected_weighted_cut(instance.num_nodes, instance.weighted_edges, state)
    approximation_ratio = expected_cut / exact_cut if exact_cut else 0.0
    record = {
        "baseline": name,
        "num_nodes": instance.num_nodes,
        "expected_cut": round(expected_cut, 6),
        "exact_maxcut": round(exact_cut, 6),
        "approximation_ratio": round(approximation_ratio, 6),
        "gammas": ";".join(f"{value:.6f}" for value in gammas),
        "betas": ";".join(f"{value:.6f}" for value in betas),
    }
    if runtime_ms is not None:
        record["runtime_ms"] = round(runtime_ms, 6)
    return record


def random_search_baseline(
    instance: WeightedGraphInstance,
    depth: int,
    exact_cut: float,
    num_samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    best_record = None
    best_value = -np.inf
    started_at = time.perf_counter()
    for _ in range(num_samples):
        gammas = rng.uniform(0.0, np.pi, size=depth)
        betas = rng.uniform(0.0, np.pi / 2.0, size=depth)
        record = evaluate_angle_baseline(
            name=f"random_search_best_of_{num_samples}",
            instance=instance,
            gammas=gammas,
            betas=betas,
            exact_cut=exact_cut,
        )
        value = float(record["expected_cut"])
        if value > best_value:
            best_value = value
            best_record = record
    if best_record is None:
        raise RuntimeError("Random search baseline did not produce a result")
    best_record["runtime_ms"] = round((time.perf_counter() - started_at) * 1e3, 6)
    return best_record


def run_qaoa_baselines(
    graph_path: str | Path,
    angles_path: str | Path,
    output_path: str | Path,
    num_random_samples: int = 256,
    seed: int = 7,
) -> Path:
    instance = load_weighted_graph(graph_path)
    gammas, betas = load_reference_angles(angles_path)
    exact_cut = exact_weighted_maxcut(instance.num_nodes, instance.weighted_edges)

    zero_started_at = time.perf_counter()
    zero_record = evaluate_angle_baseline(
        name="zero_angles",
        instance=instance,
        gammas=np.zeros_like(gammas),
        betas=np.zeros_like(betas),
        exact_cut=exact_cut,
        runtime_ms=(time.perf_counter() - zero_started_at) * 1e3,
    )

    reference_started_at = time.perf_counter()
    reference_record = evaluate_angle_baseline(
        name="reference_classical_angles",
        instance=instance,
        gammas=gammas,
        betas=betas,
        exact_cut=exact_cut,
        runtime_ms=(time.perf_counter() - reference_started_at) * 1e3,
    )

    records = [
        zero_record,
        reference_record,
        random_search_baseline(
            instance=instance,
            depth=len(gammas),
            exact_cut=exact_cut,
            num_samples=num_random_samples,
            seed=seed,
        ),
    ]
    return write_records(records, output_path)
