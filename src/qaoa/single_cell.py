"""
GNN-Informed QAOA for Batch-Aware Single-Cell Graph Partitioning
=================================================================
Core implementation for the paper:
  "Structure-Aware QAOA for Single-Cell Graph Partitioning
   via GNN-Informed Hamiltonians"

Public API
----------
build_cell_graph(X, batch_labels, k, seed)        -> CellGraph
BatchAwareHamiltonian(cell_graph, lam_bal, lam_batch)
EdgeRefinementGNN(in_feats, hidden)               -> nn.Module
ParameterInitGNN(in_feats, hidden, depth)         -> nn.Module
qaoa_value_batched(hamiltonian, gammas, betas)    -> float
classical_optimize_partition(hamiltonian, ...)    -> dict
gnn_optimize_partition(hamiltonian, gnn, ...)     -> dict
evaluate_partition(z, true_labels, batch_labels)  -> dict
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph


# ---------------------------------------------------------------------------
# 1. Cell graph construction
# ---------------------------------------------------------------------------

@dataclass
class CellGraph:
    """Compact representation of a cell-cell similarity graph."""
    n: int                          # number of cells
    edges: List[Tuple[int, int]]    # (i, j) with i < j
    weights: np.ndarray             # w[k] for edge k
    adjacency: np.ndarray           # n x n symmetric weighted adj matrix
    features: np.ndarray            # n x d expression features
    batch_labels: np.ndarray        # integer batch id per cell  (0-indexed)
    true_labels: Optional[np.ndarray] = None  # ground-truth cluster id
    cell_ids: Optional[List[str]] = None


def build_cell_graph(
    X: np.ndarray,
    batch_labels: np.ndarray,
    k: int = 5,
    similarity: str = "cosine",
    true_labels: Optional[np.ndarray] = None,
    seed: int = 0,
    n_cells: Optional[int] = None,
) -> CellGraph:
    """
    Build a kNN cell-cell similarity graph from expression matrix X.

    Parameters
    ----------
    X            : (N, D) expression matrix (z-scored recommended)
    batch_labels : (N,) integer batch ids
    k            : neighbourhood size
    similarity   : 'cosine' | 'euclidean'
    true_labels  : optional ground-truth labels for evaluation
    seed         : RNG seed for reproducible subsampling
    n_cells      : if set, subsample X to this many cells first
    """
    rng = np.random.default_rng(seed)
    if n_cells is not None and X.shape[0] > n_cells:
        idx = rng.choice(X.shape[0], size=n_cells, replace=False)
        X = X[idx]
        batch_labels = batch_labels[idx]
        if true_labels is not None:
            true_labels = true_labels[idx]

    n = X.shape[0]

    # kNN adjacency (connectivity)
    knn_conn = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                                include_self=False, metric="euclidean")
    # Make symmetric
    knn_sym = (knn_conn + knn_conn.T).toarray()
    knn_sym = (knn_sym > 0).astype(np.float64)

    # Edge weights: cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    sim_full = Xn @ Xn.T          # n x n cosine similarity matrix
    # Clip to [0, 1] and zero non-neighbours
    sim_full = np.clip(sim_full, 0.0, 1.0)
    adj = knn_sym * sim_full      # weighted adjacency

    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edges.append((i, j))
                weights.append(adj[i, j])

    return CellGraph(
        n=n,
        edges=edges,
        weights=np.array(weights, dtype=np.float64),
        adjacency=adj,
        features=X.astype(np.float32),
        batch_labels=np.asarray(batch_labels, dtype=np.int32),
        true_labels=np.asarray(true_labels, dtype=np.int32) if true_labels is not None else None,
    )


# ---------------------------------------------------------------------------
# 2. Batch-Aware Ising Hamiltonian
# ---------------------------------------------------------------------------

@dataclass
class BatchAwareHamiltonian:
    """
    H = H_cut + lambda_bal * H_bal + lambda_batch * H_batch

    H_cut   = sum_{(i,j) in E} w_ij * (1 - Z_i Z_j) / 2
              Note: sign convention gives +value for bisection.
    H_bal   = (sum_i Z_i)^2          penalises imbalanced partitions
    H_batch = sum_{(i,j) in E} delta(b_i == b_j) * Z_i Z_j
              penalises grouping same-batch cells together

    The diagonal of H in the computational basis is stored in `diagonal`
    for use directly in QAOA simulation.
    """
    cell_graph: CellGraph
    lam_bal: float = 0.1
    lam_batch: float = 0.3
    diagonal: np.ndarray = field(init=False)
    n_states: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_states = 2 ** self.cell_graph.n
        self.diagonal = self._build_diagonal()

    def _build_diagonal(self) -> np.ndarray:
        n = self.cell_graph.n
        edges = self.cell_graph.edges
        weights = self.cell_graph.weights
        batch = self.cell_graph.batch_labels
        diag = np.zeros(self.n_states, dtype=np.float64)

        for state in range(self.n_states):
            z = np.array([1 - 2 * ((state >> i) & 1) for i in range(n)], dtype=np.float64)

            # H_cut = sum w_ij * (1 - z_i z_j) / 2
            h_cut = sum(
                weights[k] * (1.0 - z[i] * z[j]) / 2.0
                for k, (i, j) in enumerate(edges)
            )

            # H_bal = (sum z_i)^2
            h_bal = float(z.sum()) ** 2

            # H_batch = sum_{(i,j)} delta(b_i==b_j) * z_i z_j
            h_batch = sum(
                (1.0 if batch[i] == batch[j] else 0.0) * z[i] * z[j]
                for (i, j) in edges
            )

            diag[state] = h_cut + self.lam_bal * h_bal + self.lam_batch * h_batch

        return diag

    def best_partition_brute(self) -> Tuple[np.ndarray, float]:
        """Brute-force minimum of H over all 2^n bitstrings."""
        best_idx = int(np.argmin(self.diagonal))
        n = self.cell_graph.n
        z_best = np.array([(best_idx >> i) & 1 for i in range(n)], dtype=np.int32)
        return z_best, float(self.diagonal[best_idx])


# ---------------------------------------------------------------------------
# 3. QAOA simulation on BatchAwareHamiltonian
# ---------------------------------------------------------------------------

def _apply_rx_all(state: np.ndarray, n: int, beta: float) -> np.ndarray:
    rx = np.array([[np.cos(beta), -1j * np.sin(beta)],
                   [-1j * np.sin(beta), np.cos(beta)]], dtype=np.complex128)
    psi = state.reshape((2,) * n)
    for axis in range(n):
        psi = np.moveaxis(psi, axis, 0)
        psi = np.tensordot(rx, psi, axes=([1], [0]))
        psi = np.moveaxis(psi, 0, axis)
    return psi.reshape(-1)


def qaoa_state(diagonal: np.ndarray, gammas: Sequence[float],
               betas: Sequence[float]) -> np.ndarray:
    """Execute depth-p QAOA circuit and return final state vector."""
    n_states = diagonal.shape[0]
    n = int(np.log2(n_states))
    state = np.ones(n_states, dtype=np.complex128) / np.sqrt(n_states)
    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * diagonal)
        state = _apply_rx_all(state, n, beta)
    return state


def qaoa_energy(diagonal: np.ndarray, gammas: Sequence[float],
                betas: Sequence[float]) -> float:
    """Expected energy <psi|H|psi> for a given angle set."""
    state = qaoa_state(diagonal, gammas, betas)
    return float(np.dot(diagonal, np.abs(state) ** 2))


def _normalize(raw: np.ndarray, depth: int) -> Tuple[np.ndarray, np.ndarray]:
    gammas = np.mod(raw[:depth], math.pi)
    betas  = np.mod(raw[depth:2 * depth], math.pi / 2)
    return gammas, betas


def classical_optimize_partition(
    hamiltonian: BatchAwareHamiltonian,
    depth: int = 2,
    num_starts: int = 8,
    maxiter: int = 320,
    seed: int = 0,
) -> Dict:
    """Classical Nelder-Mead search over QAOA angles."""
    diag = hamiltonian.diagonal
    rng  = np.random.default_rng(seed)
    best_energy, best_angles = np.inf, None
    t0 = time.perf_counter()

    def obj(raw):
        g, b = _normalize(raw, depth)
        return qaoa_energy(diag, g, b)

    for _ in range(num_starts):
        x0 = np.concatenate([rng.uniform(0.0, math.pi, depth),
                              rng.uniform(0.0, math.pi / 2, depth)])
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-5})
        if res.fun < best_energy:
            best_energy = res.fun
            best_angles = res.x

    gammas, betas = _normalize(best_angles, depth)
    state = qaoa_state(diag, gammas, betas)
    probs = np.abs(state) ** 2
    best_state = int(np.argmax(probs))
    n = hamiltonian.cell_graph.n
    z = np.array([(best_state >> i) & 1 for i in range(n)], dtype=np.int32)

    return {
        "method": "Classical QAOA",
        "energy": best_energy,
        "gammas": gammas.tolist(),
        "betas": betas.tolist(),
        "partition": z,
        "runtime_ms": 1000.0 * (time.perf_counter() - t0),
    }


# ---------------------------------------------------------------------------
# 4. GNN modules
# ---------------------------------------------------------------------------

class EdgeRefinementGNN(nn.Module):
    """
    GNN Role A: refine edge weights w_ij given node expression features.

    Architecture: 2-layer GCN -> per-node embedding -> MLP edge scorer
    Input:  node features (n, d_in)
    Output: refined edge weight per edge (n_edges,)  in (0, 1)
    """

    def __init__(self, in_feats: int, hidden: int = 64) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_feats, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.edge_score = nn.Linear(2 * hidden, 1)

    def _gcn_pass(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Symmetric normalised adjacency (D^{-1/2} A D^{-1/2})
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        a_norm = adj / (deg * deg.T).sqrt()
        h = F.relu(self.lin1(torch.matmul(a_norm, x)))
        h = F.relu(self.lin2(torch.matmul(a_norm, h)))
        return h

    def forward(
        self,
        x: torch.Tensor,            # (n, d_in)
        adj: torch.Tensor,          # (n, n)
        edge_index: torch.Tensor,   # (2, E) source/target for each edge
    ) -> torch.Tensor:              # (E,) refined weights in (0,1)
        h = self._gcn_pass(x, adj)                          # (n, hidden)
        src = h[edge_index[0]]                               # (E, hidden)
        dst = h[edge_index[1]]                               # (E, hidden)
        score = self.edge_score(torch.cat([src, dst], dim=1))  # (E, 1)
        return torch.sigmoid(score).squeeze(1)              # (E,)


class ParameterInitGNN(nn.Module):
    """
    GNN Role B: predict QAOA initial angles (gamma_1..p, beta_1..p) from graph.

    Architecture: 2-layer GCN -> global mean pool -> MLP -> 2p angles
    Input:  node features (n, d_in)
    Output: angle vector (2p,)  [gammas] ++ [betas]
    """

    def __init__(self, in_feats: int, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.lin1 = nn.Linear(in_feats, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, 2 * depth)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        a_norm = adj / (deg * deg.T).sqrt()
        h = F.relu(self.lin1(torch.matmul(a_norm, x)))
        h = F.relu(self.lin2(torch.matmul(a_norm, h)))
        g = h.mean(dim=0)              # global mean pool (hidden,)
        angles = self.readout(g)       # (2*depth,)
        return angles


# ---------------------------------------------------------------------------
# 5. GNN-informed QAOA inference
# ---------------------------------------------------------------------------

def _make_edge_index(edges: List[Tuple[int, int]]) -> torch.Tensor:
    src = [i for i, j in edges] + [j for i, j in edges]
    dst = [j for i, j in edges] + [i for i, j in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def gnn_optimize_partition(
    hamiltonian: BatchAwareHamiltonian,
    param_gnn: ParameterInitGNN,
    edge_gnn: Optional[EdgeRefinementGNN] = None,
    depth: int = 2,
    num_starts: int = 3,
    maxiter: int = 160,
    seed: int = 0,
) -> Dict:
    """
    GNN-informed QAOA:
      1. (optionally) refine edge weights via EdgeRefinementGNN
      2. predict initial angles via ParameterInitGNN
      3. short Nelder-Mead polish starting from GNN init
    """
    cg = hamiltonian.cell_graph
    x   = torch.tensor(cg.features, dtype=torch.float32)
    adj = torch.tensor(cg.adjacency, dtype=torch.float32)
    edge_index = _make_edge_index(cg.edges)

    t0 = time.perf_counter()

    # --- Optional: refine edge weights ---
    active_diag = hamiltonian.diagonal
    if edge_gnn is not None:
        edge_gnn.eval()
        with torch.no_grad():
            refined_w = edge_gnn(x, adj, edge_index).numpy()
        # Build a new Hamiltonian with refined weights
        refined_cg = CellGraph(
            n=cg.n,
            edges=cg.edges,
            weights=refined_w,
            adjacency=cg.adjacency,
            features=cg.features,
            batch_labels=cg.batch_labels,
            true_labels=cg.true_labels,
        )
        refined_ham = BatchAwareHamiltonian(
            refined_cg, hamiltonian.lam_bal, hamiltonian.lam_batch)
        active_diag = refined_ham.diagonal

    # --- GNN initial angles ---
    param_gnn.eval()
    with torch.no_grad():
        raw_init = param_gnn(x, adj).numpy()   # (2*depth,)

    proposal_ms = 1000.0 * (time.perf_counter() - t0)

    # --- Short Nelder-Mead polish from GNN init ---
    rng = np.random.default_rng(seed)
    best_energy, best_angles = np.inf, None

    def obj(raw):
        g, b = _normalize(raw, depth)
        return qaoa_energy(active_diag, g, b)

    # First start: GNN prediction
    candidates = [raw_init]
    for _ in range(num_starts - 1):
        candidates.append(raw_init + rng.normal(0, 0.1, raw_init.shape))

    for x0 in candidates:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-5})
        if res.fun < best_energy:
            best_energy = res.fun
            best_angles = res.x

    gammas, betas = _normalize(best_angles, depth)
    state = qaoa_state(active_diag, gammas, betas)
    probs = np.abs(state) ** 2
    best_state = int(np.argmax(probs))
    n = cg.n
    z = np.array([(best_state >> i) & 1 for i in range(n)], dtype=np.int32)

    total_ms = 1000.0 * (time.perf_counter() - t0)

    return {
        "method": "GNN-QAOA",
        "energy": best_energy,
        "gammas": gammas.tolist(),
        "betas": betas.tolist(),
        "partition": z,
        "proposal_ms": proposal_ms,
        "runtime_ms": total_ms,
        "gnn_init_angles": raw_init.tolist(),
    }


# ---------------------------------------------------------------------------
# 6. Training helpers
# ---------------------------------------------------------------------------

def train_parameter_gnn(
    train_instances: List[Dict],
    depth: int = 2,
    hidden: int = 64,
    epochs: int = 500,
    lr: float = 2.5e-3,
    weight_decay: float = 5e-5,
    patience: int = 60,
    seed: int = 7,
) -> Dict:
    """
    Supervised training of ParameterInitGNN.
    Each instance must contain 'cell_graph', 'target_angles' (2*depth,).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_feats = train_instances[0]["cell_graph"].features.shape[1]
    model = ParameterInitGNN(in_feats=in_feats, hidden=hidden, depth=depth)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inst in train_instances:
            cg = inst["cell_graph"]
            x   = torch.tensor(cg.features, dtype=torch.float32)
            adj = torch.tensor(cg.adjacency, dtype=torch.float32)
            target = torch.tensor(inst["target_angles"], dtype=torch.float32)

            pred = model(x, adj)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / max(1, len(train_instances))
        losses.append(mean_loss)

        if mean_loss + 1e-8 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "best_loss": best_loss, "loss_history": losses}


# ---------------------------------------------------------------------------
# 7. Evaluation metrics
# ---------------------------------------------------------------------------

def batch_mixing_entropy(partition: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Higher value = more batch mixing within each cluster.
    Defined as the average entropy of batch composition per cluster.
    """
    scores = []
    for cluster_id in np.unique(partition):
        mask = partition == cluster_id
        batches = batch_labels[mask]
        if batches.size == 0:
            continue
        _, counts = np.unique(batches, return_counts=True)
        p = counts / counts.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        scores.append(entropy)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_partition(
    partition: np.ndarray,
    true_labels: Optional[np.ndarray],
    batch_labels: np.ndarray,
) -> Dict:
    result: Dict = {}
    result["batch_mixing_entropy"] = batch_mixing_entropy(partition, batch_labels)
    result["n_clusters"] = int(np.unique(partition).size)
    if true_labels is not None:
        result["ari"] = float(adjusted_rand_score(true_labels, partition))
        result["nmi"] = float(normalized_mutual_info_score(true_labels, partition))
    return result


def heuristic_partition(cell_graph: CellGraph) -> np.ndarray:
    """
    Spectral bisection baseline: sign of Fiedler eigenvector.
    Works on the weighted adjacency as a standard classical baseline.
    """
    n = cell_graph.n
    W = cell_graph.adjacency
    D = np.diag(W.sum(axis=1))
    L = D - W
    _, vecs = np.linalg.eigh(L)
    fiedler = vecs[:, 1]               # second smallest eigenvector
    return (fiedler >= 0).astype(np.int32)
