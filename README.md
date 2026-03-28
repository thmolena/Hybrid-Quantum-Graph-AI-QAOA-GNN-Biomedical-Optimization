# Hybrid Quantum–Graph AI: QAOA + GNN for Biomedical Optimization

> A research-grade hybrid quantum-classical system that unifies variational quantum optimization and graph neural network learning under a single graph-theoretic framework, demonstrated on real genomic and clinical datasets.

---

## Abstract

This project addresses a fundamental challenge at the intersection of quantum computing and biomedical AI: how can the structure inherent in biological data — gene co-expression networks, clinical similarity graphs — be exploited simultaneously by quantum optimization algorithms and graph-based deep learning?

The core hypothesis is that **graphs are the common representation language** connecting two otherwise distinct paradigms. On the quantum side, MaxCut on a graph is a canonical NP-hard combinatorial problem that the Quantum Approximate Optimization Algorithm (QAOA) targets directly. On the classical side, Graph Convolutional Networks (GCNs) aggregate information through the same adjacency structure to learn expressive node and graph representations. This repository demonstrates that a single trained GNN can replace the expensive variational parameter search in QAOA — and that the same graph-learning machinery transfers directly to real-world clinical risk detection.

The system is validated on two real datasets: a transcriptomic cohort of 102 prostate cancer patient samples (12,600 gene expression features) and a retrospective fetal monitoring cohort of 2,126 cardiotocography exams (21 physiologic features). Across both domains, the pipeline achieves strong quantitative results while remaining fully reproducible and interpretable.

---

## Highlights

| Domain | Result |
|---|---|
| QAOA parameter prediction | GNN inference matches classical Nelder-Mead search quality at a fraction of the latency |
| Biomedical graph classification | **94.1% accuracy**, **0.963 ROC AUC**, **85.7% pathologic recall** on held-out CTG test set |
| Balanced accuracy | **0.903** — robust to the class imbalance inherent in rare pathologic events |
| Graph scale (clinical) | 2,126-node exam graph, 14,776 edges via 10-NN physiologic similarity |
| Quantum simulation | Exact statevector QAOA on biologically motivated co-expression graphs derived from 10-gene panels |

---

## Motivation and Problem Statement

### The QAOA parameter bottleneck

QAOA is a leading near-term variational quantum algorithm for combinatorial optimization. Its performance depends critically on the choice of variational angles `(γ, β)`, which are classically optimized through repeated circuit evaluations — a process that scales poorly with problem size and circuit depth. Learning to predict these parameters directly from graph structure, bypassing the inner optimization loop, is an active research direction with significant practical implications for quantum advantage on near-term hardware.

### The biomedical graph learning opportunity

Clinical datasets are rarely i.i.d. Patients, exams, and biological entities exist in relational contexts defined by physiologic similarity, genomic co-regulation, and temporal proximity. Standard tabular classifiers discard this relational structure entirely. Graph convolutional networks recover it, enabling message passing that aggregates information from physiologically similar neighbors — a natural inductive bias for medical risk stratification.

### The unifying insight

Both problems reduce to learning from graphs. The same representation — nodes with features, edges encoding similarity or interaction — supports variational quantum circuits on the optimization side and deep message-passing networks on the prediction side. This project makes that connection concrete, executable, and empirically grounded on real data.

---

## Technical Contributions

1. **Transcriptomic graph construction for QAOA.** A real prostate cancer gene expression matrix is transformed into biologically motivated co-expression graphs via absolute Pearson correlation, providing tractable MaxCut instances with genomic semantics.

2. **GNN-accelerated QAOA parameter prediction.** A `SimpleGCN` model is trained to regress classical QAOA angles from graph structure descriptors, replacing multi-start Nelder-Mead search with a single forward pass. Benchmarks across patient-resampled co-expression graphs quantify the latency-quality trade-off.

3. **Leakage-aware clinical graph learning.** The cardiotocography pipeline enforces strict train/validation/test separation before any feature standardization, uses class-weighted cross-entropy to handle the imbalanced pathologic class, and evaluates the held-out cohort with clinical screening metrics (recall, balanced accuracy, ROC AUC) rather than raw accuracy.

4. **Portable, dependency-light implementation.** The codebase runs in a standard PyTorch environment. PyTorch Geometric is used when available but is not required; the GCN falls back to explicit adjacency-matrix message passing otherwise.

5. **End-to-end deployment path.** The system includes a trained model checkpoint, a Flask inference API, and a static frontend demo, providing a complete path from notebook research to interactive deployment.

---

## System Architecture

```
Real Data Sources
    ├── OpenML prostate dataset (102 patients × 12,600 gene features)
    └── UCI Cardiotocography dataset (2,126 exams × 21 physiologic features)
                │
                ▼
    Graph Construction Layer
    ├── Co-expression graph  (absolute Pearson correlation → adjacency)
    └── Exam-similarity graph (10-nearest-neighbor → symmetric adjacency)
                │
          ┌─────┴─────┐
          ▼           ▼
  QAOA Branch     Biomedical Branch
  ──────────────  ──────────────────
  Exact statevec  ClinicalGCN / BioGCN
  simulator       (2-layer, dropout,
  SimpleGCN →     class-weighted loss)
  predict (γ, β)  → pathologic risk score
  MaxCut eval     ROC / confusion matrix
          │           │
          └─────┬─────┘
                ▼
    Flask API  (src/server.py)
    Static frontend  (website/)
    Model checkpoint  (model.pt)
```

---

## Notebook Guide

### `notebooks/qaoa_demo.ipynb` — Genomics-Derived QAOA Benchmark

**Data.** OpenML `prostate` gene-expression dataset: 102 patient samples, 12,600 expression features. A 10-gene panel is selected by variance ranking. A representative co-expression graph is built from the full cohort; six additional benchmark graphs are derived from stratified patient subsamples.

**Workflow.**

1. Introductory walkthrough of QAOA, MaxCut, and the exact statevector simulation logic.
2. Construction of biologically motivated co-expression graphs from real transcriptomic data.
3. Multi-start Nelder-Mead classical optimization of QAOA angles on the representative graph.
4. GNN forward pass to predict `(γ, β)` directly from graph structure.
5. Visualization: co-expression graph with MaxCut partition, 1D QAOA response slice, full 2D angle landscape.
6. Benchmark comparison of classical search versus GNN inference across the patient-resampled graph collection.

**Key findings.** Real transcriptomic data yields tractable graph optimization instances. Exact QAOA remains feasible when the biological source is reduced to a small co-expression graph. The trained GNN matches classical search solution quality at substantially lower inference cost.

---

### `notebooks/bio_demo.ipynb` — Biomedical Graph Classification on CTG Cohort

**Data.** UCI Cardiotocography dataset: 2,126 fetal monitoring exams, 21 physiologic summary features, three-class obstetric risk labels (normal / suspect / pathologic).

**Workflow.**

1. Reframe the three-class label into a binary screening target: pathologic versus non-pathologic.
2. Stratified train / validation / test split with leakage-free feature standardization (scaler fit on training partition only).
3. Construction of a symmetric 10-nearest-neighbor exam-similarity graph (14,776 edges) with self-loops and degree normalization.
4. Two-layer `ClinicalGCN` with dropout and class-weighted cross-entropy, trained with early stopping on validation balanced accuracy.
5. Held-out evaluation with clinical screening metrics.

**Held-out test performance.**

| Metric | Value |
|---|---|
| Accuracy | 94.1% |
| Balanced Accuracy | 0.903 |
| Pathologic Recall | 85.7% |
| ROC AUC | 0.963 |

The pipeline is explicitly scoped as a retrospective research demonstration, not a clinical deployment system.

---

### `notebooks/quantum_ai_bio_combined.ipynb` — Integrative Narrative

This notebook stitches the two branches into a single coherent demonstration. It is the primary entry point for readers who want to see the full hybrid system end to end.

**Workflow.**

1. Reproducibility and environment audit.
2. Compact QAOA walkthrough: small MaxCut graph, exact solution, classical angle optimization, GNN prediction comparison.
3. Full CTG graph-classification pipeline from raw data through training to evaluation dashboard.
4. Synthesis section explaining how graph structure supports both variational optimization and discriminative learning under a unified framework.

**Key outputs.** `outputs/maxcut_graph.csv`, `outputs/qaoa_classical_angles.csv`, `outputs/ctg_raw.csv`, `outputs/ctg_processed.csv`, QAOA landscape figures, BioGCN six-panel evaluation dashboard (confusion matrix, ROC curve, training dynamics, PCA projections, cohort feature shifts).

---

## Repository Structure

```text
Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/
├── README.md
├── LICENSE
├── requirements.txt
├── model.pt                        # Trained SimpleGCN checkpoint
├── index.html                      # GitHub Pages entry point
├── data/
│   └── breast_cancer.csv           # Earlier supporting artifact
├── notebooks/
│   ├── qaoa_demo.ipynb             # Genomics-derived QAOA benchmark
│   ├── bio_demo.ipynb              # Clinical graph classification
│   └── quantum_ai_bio_combined.ipynb  # Full integrative narrative
├── outputs/
│   ├── ctg_raw.csv
│   ├── ctg_processed.csv
│   ├── maxcut_graph.csv
│   └── qaoa_classical_angles.csv
├── src/
│   ├── __init__.py
│   ├── data.py                     # Graph samplers and dataset utilities
│   ├── gnn.py                      # SimpleGCN / ClinicalGCN definitions
│   ├── qaoa_sim.py                 # Exact statevector QAOA simulator
│   ├── server.py                   # Flask inference API
│   └── train.py                    # GNN training pipeline
└── website/
    ├── index.html
    ├── style.css
    ├── demo.js
    ├── README_SITE.md
    └── notebooks_html/
        ├── qaoa_demo.html
        ├── bio_demo.html
        └── quantum_ai_bio_combined.html
```

Current development centers on the CTG and prostate-data workflows. The `breast_cancer.csv` dataset and related output files remain in the repository as earlier supporting artifacts.

---

## Implementation Notes

### Graph Neural Network (`src/gnn.py`)

`SimpleGCN` is the QAOA parameter predictor.

- Input node feature: augmented degree
- Hidden width: 32 units
- Output: `2 × p` values interpreted as `(γ₁, …, γₚ, β₁, …, βₚ)`
- Readout: global mean pooling
- Backend: `torch_geometric.nn.GCNConv` when PyTorch Geometric is available; falls back to explicit adjacency-matrix message passing otherwise

`ClinicalGCN` / `BioGCN` are the biomedical classifiers (defined inline in the notebooks).

- Two GCN layers with ReLU activation and dropout
- Class-weighted cross-entropy loss to handle pathologic class imbalance
- Early stopping on validation balanced accuracy

### QAOA Simulator (`src/qaoa_sim.py`)

A lightweight exact statevector simulator with no external quantum library dependency.

- Initializes the uniform superposition state analytically
- Applies the MaxCut phase-separation operator via exact diagonal exponentiation
- Applies the mixer operator via dense matrix exponentiation
- Computes expected cut value from the final statevector probability distribution

Complexity scales exponentially with the number of qubits, making this design appropriate for the small-graph regime used in the notebooks (up to ~10 nodes / qubits). It is transparent and portable by design.

### Training Pipeline (`src/train.py`)

Generates connected Erdős-Rényi random graphs, computes reference QAOA angles by classical optimization, and trains `SimpleGCN` to regress those parameters from graph structure.

```bash
python -m src.train --dataset-size 20 --n 6 --p 1 --epochs 10 --model-path model.pt
```

### Inference API (`src/server.py`)

A minimal Flask endpoint for interactive inference.

- Route: `POST /predict`
- Input: JSON edge list and optional node count
- Output: predicted `gammas`, `betas`, and resulting expected cut value

```json
{
    "edges": [[0, 1], [1, 2], [0, 2]],
    "n": 3
}
```

The static site in `website/` calls the Flask API and visualizes predictions for user-supplied graphs. By default, `website/demo.js` targets `http://localhost:5000/predict`. The website README documents how to override the API base URL for a different deployment target.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train a fresh GNN checkpoint
python -m src.train --dataset-size 20 --n 6 --p 1 --epochs 10 --model-path model.pt

# 3. Run the notebooks
#    Recommended order:
#      notebooks/quantum_ai_bio_combined.ipynb   ← full integrative narrative
#      notebooks/qaoa_demo.ipynb                 ← QAOA benchmark only
#      notebooks/bio_demo.ipynb                  ← biomedical workflow only

# 4. Start the inference API
python -m src.server

# 5. Serve the static demo site
python -m http.server 8000
# → open http://localhost:8000/
```

> GitHub Pages: the published site is served from the repository root `index.html`, which loads assets from `website/`.

---

## Generated Artifacts

| File | Description |
|---|---|
| `model.pt` | Trained `SimpleGCN` checkpoint |
| `outputs/ctg_raw.csv` | CTG cohort with original three-class and binary screening labels |
| `outputs/ctg_processed.csv` | Standardized CTG features with train / val / test split annotations |
| `outputs/maxcut_graph.csv` | Edge list for the QAOA demo graph |
| `outputs/qaoa_classical_angles.csv` | Classically optimized depth-1 QAOA angles |

---

## Roadmap

### Near-term
- Extend QAOA benchmarks to deeper circuits (`p > 1`) and denser co-expression graphs from larger transcriptomic cohorts.
- Add systematic experiment tracking for QAOA quality-vs-speed trade-offs across varied graph topologies and node counts.
- Enrich GNN node features beyond degree to include local graph statistics (clustering coefficient, betweenness centrality) for improved parameter prediction accuracy.

### Mid-term
- Explore alternative graph construction strategies for the clinical branch: temporal graphs, multi-modal heterogeneous graphs combining physiologic and demographic features.
- Systematic ablation of GCN depth, aggregation scheme, and edge-weighting strategy on the CTG classification task.
- Incorporate uncertainty quantification for clinical risk scores.

### Longer-term
- Connect the exact statevector simulation layer to hardware-aware backends (noise models, transpilation) to study how classically predicted parameters transfer to real quantum hardware.
- Investigate warm-starting strategies where GNN-predicted angles initialize variational circuits rather than serving as the final answer, targeting hybrid quantum-classical fine-tuning.
- Apply the graph-mediated QAOA workflow to protein interaction networks and drug-target interaction graphs.

---

## Contributing

Contributions are welcome in the form of algorithmic improvements, notebook refinements, additional real-data graph benchmarks, evaluation upgrades, and reproducibility fixes. Pull requests are most useful when they include a concise technical rationale and clear validation evidence.

---

## License

This project is released under the terms of the LICENSE file in this repository.
