# Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization

## Overview

This repository is a research-oriented hybrid quantum-classical AI project organized around one recurring idea: express a problem as a graph, then use graph-aware learning and optimization to study it from multiple angles.

The current codebase supports three closely related workflows:

- a genomics-derived QAOA benchmark in which real gene-expression data is converted into small co-expression graphs for exact MaxCut and QAOA analysis
- a biomedical graph-learning pipeline in which fetal monitoring exams are linked through physiologic similarity and classified with a graph convolutional network
- a combined demonstration notebook that places the optimization and biomedical branches in one coherent narrative

The implementation uses a lightweight exact statevector simulator for small QAOA instances, a portable GCN implementation with an optional PyTorch Geometric backend, a training script for graph-to-QAOA-parameter regression, and a Flask API plus static website for interactive inference.

---

## Current Project Scope

The repository is no longer just a generic hybrid-optimization scaffold. The notebooks now focus on two concrete, real-data stories:

1. Quantum optimization from real transcriptomic data. The QAOA notebook starts from the OpenML `prostate` gene-expression dataset, selects a compact gene panel, derives biologically motivated co-expression graphs, and compares direct classical QAOA parameter search against GNN-predicted parameters.
2. Biomedical graph classification on a real CTG cohort. The biomedical notebook uses the UCI Cardiotocography dataset to build a risk-sensitive fetal-state classifier on a k-nearest-neighbor exam graph, with attention to class imbalance, leakage-aware preprocessing, and clinically meaningful evaluation.

The combined notebook then shows how these two branches fit together under a single graph-centered hybrid-AI framing.

---

## Notebook Guide

### `notebooks/qaoa_demo.ipynb`

This notebook is the repository's main quantum-optimization case study. It reframes the QAOA demo around a real genomics source dataset instead of a purely synthetic graph.

**What goes in**

- OpenML `prostate` gene-expression dataset
- 102 patient samples and 12,600 expression features
- a derived 10-gene panel chosen by variance ranking
- co-expression graphs built from absolute Pearson correlations

**What the notebook does**

1. Introduces QAOA, MaxCut, and the exact statevector simulation logic in tutorial form.
2. Loads the repository's `SimpleGCN` checkpoint from `model.pt` when available.
3. Defines helper functions for exact MaxCut, QAOA state evaluation, angle normalization, classical optimization, and learned inference.
4. Fetches the real genomics dataset and constructs one representative full-cohort co-expression graph plus six additional benchmark graphs from stratified patient subsamples.
5. Runs multi-start Nelder-Mead search on the representative graph to obtain a depth-1 QAOA baseline.
6. Runs the GNN on the same graph to predict `gamma` and `beta` directly from graph structure.
7. Visualizes the representative graph, a one-dimensional landscape slice, and the full two-parameter QAOA landscape.
8. Benchmarks classical search versus GNN inference across the patient-resampled graph collection.

**What it displays**

- dataset summary tables for the prostate cohort
- the selected gene panel and benchmark overview
- classical and GNN-predicted QAOA angle summaries
- a representative gene co-expression graph with the exact MaxCut partition
- a 1D expected-cut slice and a 2D QAOA heatmap
- benchmark-wide latency and quality summaries
- the highest-probability output bit strings for the representative graph

**What it highlights**

- real transcriptomic data can be turned into tractable graph optimization instances
- exact QAOA remains feasible when the biological source is reduced to a small graph
- a learned graph model can propose useful QAOA parameters much faster than repeated direct search

### `notebooks/bio_demo.ipynb`

This notebook is the repository's main biomedical learning workflow. It builds an end-to-end graph convolutional pipeline for retrospective fetal-state risk detection on a real public cohort.

**What goes in**

- UCI Cardiotocography dataset via `ucimlrepo.fetch_ucirepo(id=193)`
- 2,126 fetal monitoring exams
- 21 physiologic summary features
- original three-class labels: normal, suspect, and pathologic

**What the notebook does**

1. Reframes the original three-class obstetric label into a binary screening target: pathologic versus non-pathologic.
2. Audits cohort composition and writes an auditable raw cohort table.
3. Creates a stratified train, validation, and test split before fitting the scaler.
4. Standardizes features using training-set statistics only.
5. Constructs a symmetric 10-nearest-neighbor exam-similarity graph with self-loops and degree normalization.
6. Defines a two-layer `ClinicalGCN` with dropout and class-weighted cross-entropy.
7. Trains with validation monitoring and early stopping.
8. Produces held-out evaluation outputs focused on pathologic detection.

**What it writes and displays**

- `outputs/ctg_raw.csv` with case IDs, original labels, and binary framing
- `outputs/ctg_processed.csv` with standardized features and split assignments
- cohort, split, and graph statistics printed in the notebook
- a PCA-based cohort audit and prediction panel
- a feature-shift bar chart showing the strongest physiologic differences
- a held-out confusion matrix, ROC curve, and training-dynamics figure

**What it highlights**

- the CTG cohort contains 2,126 exams split into 1,445 train, 255 validation, and 426 test cases
- the exam graph contains 14,776 undirected edges with low density, so each exam exchanges information only with a small physiologic neighborhood
- the saved notebook run reports 94.1% accuracy, 0.903 balanced accuracy, 85.7% pathologic recall, and 0.963 ROC AUC on the held-out test set
- the notebook is explicit about scope: this is a research demonstration for retrospective risk screening, not a clinical deployment system

### `notebooks/quantum_ai_bio_combined.ipynb`

This notebook is the integrative narrative for the repository. It combines a compact QAOA walkthrough with the real CTG graph-classification branch so readers can see the hybrid theme end to end.

**What goes in**

- the repository source modules and plotting stack
- a small connected Erdős-Rényi MaxCut graph sampled from `src.data.sample_erdos_renyi`
- the `SimpleGCN` checkpoint for QAOA parameter prediction when available
- the same real CTG cohort used in `bio_demo.ipynb`

**What the notebook does**

1. Runs a reproducibility and environment audit.
2. Builds a small MaxCut graph, computes the exact MaxCut solution, and classically optimizes depth-1 QAOA angles.
3. Loads or initializes `SimpleGCN` and compares GNN-predicted QAOA angles against the classical baseline.
4. Visualizes the graph, the one-dimensional QAOA response slice, and the full two-parameter landscape.
5. Loads the UCI CTG cohort, standardizes the features, and writes `outputs/ctg_raw.csv` and `outputs/ctg_processed.csv`.
6. Builds a 10-nearest-neighbor exam graph and trains a two-layer `BioGCN` with class-weighted loss and early stopping.
7. Produces a six-panel evaluation dashboard with confusion matrix, ROC, training curves, PCA projections, and cohort feature shifts.
8. Closes with a synthesis that explains how graph structure supports both optimization and prediction.

**What it writes and displays**

- `outputs/maxcut_graph.csv`
- `outputs/qaoa_classical_angles.csv`
- `outputs/ctg_raw.csv`
- `outputs/ctg_processed.csv`
- QAOA baseline, predicted-angle, and landscape figures
- BioGCN evaluation dashboard on the held-out CTG cohort

**What it highlights**

- the QAOA branch is intentionally small and exact, making the optimization geometry easy to inspect
- the biomedical branch is larger and more applied, showing how graph learning behaves on a real imbalanced cohort
- the notebook makes the repository's main conceptual claim explicit: graphs are the common language connecting hybrid quantum optimization and biomedical learning

---

## Repository Structure

```text
Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/
├── README.md
├── LICENSE
├── requirements.txt
<<<<<<< Updated upstream
├── model.pt
├── index.html
=======
<<<<<<< HEAD
├── model.pt
├── index.html
=======
├── index.html                  ← GitHub Pages root
>>>>>>> e34a9c0f4b8da436cfc29f3055c13028d5cf44d6
>>>>>>> Stashed changes
├── data/
│   └── breast_cancer.csv
├── notebooks/
│   ├── bio_demo.ipynb
│   ├── qaoa_demo.ipynb
│   └── quantum_ai_bio_combined.ipynb
├── outputs/
│   ├── breast_cancer_expanded_processed.csv
│   ├── breast_cancer_processed.csv
│   ├── breast_cancer_raw.csv
│   ├── ctg_processed.csv
│   ├── ctg_raw.csv
│   ├── maxcut_graph.csv
│   └── qaoa_classical_angles.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── gnn.py
│   ├── qaoa_sim.py
│   ├── server.py
│   └── train.py
└── website/
        ├── README_SITE.md
        ├── demo.js
        ├── index.html
        ├── style.css
        └── notebooks_html/
                ├── bio_demo.html
                ├── qaoa_demo.html
                └── quantum_ai_bio_combined.html
```

Current notebook development centers on the CTG and prostate-data workflows. The `breast_cancer.csv` dataset and related output files remain in the repository as earlier supporting artifacts, but they are no longer the main narrative of the README or the current notebooks.

---

## Implementation Notes

### Graph model

`src/gnn.py` defines `SimpleGCN`, the model used for graph-to-QAOA-parameter prediction.

- input node feature: augmented degree
- hidden width: 32
- output: `2 * p` values interpreted as `gamma` and `beta`
- readout: global mean pooling
- backend behavior:
    - uses `torch_geometric.nn.GCNConv` when PyTorch Geometric is available
    - otherwise falls back to adjacency-matrix message passing with standard PyTorch layers

### QAOA simulator

`src/qaoa_sim.py` implements a lightweight exact statevector simulator.

- initializes the uniform superposition state directly
- applies the MaxCut phase operator exactly
- applies the mixer through dense matrix exponentiation
- computes the expected cut from the final statevector

This design is transparent and portable, but it scales exponentially with the number of nodes. It is appropriate for the small exact cases used in the notebooks, not for large quantum instances.

### Training pipeline

`src/train.py` generates connected Erdős-Rényi graphs, computes reference QAOA angles by classical optimization, and trains `SimpleGCN` to regress those parameters from graph structure.

The default training command is:

```bash
python -m src.train --dataset-size 20 --n 6 --p 1 --epochs 10 --model-path model.pt
```

### API server

`src/server.py` exposes a Flask prediction endpoint.

- route: `POST /predict`
- input: graph edge list and optional node count
- output: predicted `gammas`, `betas`, and the resulting expected cut

Example request body:

```json
{
    "edges": [[0, 1], [1, 2], [0, 2]],
    "n": 3
}
```

### Website demo

The static site in `website/` calls the Flask API and visualizes predictions for user-supplied graphs. By default, `website/demo.js` targets `http://localhost:5000/predict`. The website README also documents how to override the API base URL for a different deployment target.

---

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Train the QAOA-parameter predictor if you want a fresh checkpoint.

```bash
python -m src.train --dataset-size 20 --n 6 --p 1 --epochs 10 --model-path model.pt
```

3. Run the notebooks in the order that matches your goal.

- `notebooks/qaoa_demo.ipynb` for the genomics-derived QAOA benchmark
- `notebooks/bio_demo.ipynb` for the standalone biomedical CTG workflow
- `notebooks/quantum_ai_bio_combined.ipynb` for the full integrative narrative

4. Start the inference API.

```bash
python -m src.server
```

5. Optionally serve the repository locally for the website and GitHub Pages entry point.

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`.

If you want to serve only the website directory instead, use:

```bash
python -m http.server 8000 --directory website
```

Then open `http://localhost:8000/index.html`.

> GitHub Pages: the published site is served from the repository root `index.html`, which loads assets from `website/`.

---

## Generated Artifacts

The repository currently contains several generated CSV artifacts produced by the training scripts and notebooks.

- `outputs/ctg_raw.csv`: raw CTG cohort with original three-class labels and binary screening labels
- `outputs/ctg_processed.csv`: standardized CTG features with train, validation, and test split annotations
- `outputs/maxcut_graph.csv`: edge list for the small QAOA demo graph written by the combined notebook
- `outputs/qaoa_classical_angles.csv`: classically optimized QAOA angles written by the combined notebook
- `model.pt`: trained `SimpleGCN` checkpoint used by the QAOA notebooks and the Flask API when available

---

## Key Features

### Real-data quantum-to-graph workflow

The QAOA notebook does not stop at synthetic graph generation. It starts from a real transcriptomic matrix, derives compact co-expression graphs, and studies how graph structure affects exact MaxCut and QAOA behavior.

### Leakage-aware biomedical graph learning

The CTG pipeline explicitly separates train, validation, and test partitions before standardization, uses class weighting for the rare pathologic class, and evaluates the held-out cohort with risk-sensitive metrics rather than accuracy alone.

### Portable GNN implementation

The repository runs in a standard PyTorch environment and does not require PyTorch Geometric, while still taking advantage of it when installed.

### Interactive inference path

The codebase includes not only notebooks and training scripts, but also a small deployment path through `src/server.py` and the static website demo.

---

## Roadmap

### Near-term

Expand benchmark coverage to larger and more diverse graph families, add stronger experiment tracking for QAOA quality-speed trade-offs, and enrich the graph features used for learned parameter prediction.

### Mid-term

Strengthen the biomedical branch with richer graph construction strategies, additional physiologic or metadata features, and broader evaluation protocols for clinically motivated screening analysis.

### Longer-term

Connect the simulation-centric workflow to hardware-aware backends and more advanced hybrid training loops, while preserving the repository's current emphasis on interpretability and reproducibility.

---

## Contributing

Contributions are welcome in the form of algorithmic improvements, notebook refinements, additional real-data graph benchmarks, evaluation upgrades, and reproducibility fixes. Pull requests are most useful when they include a concise technical rationale and clear validation evidence.

---

## License

This project is released under the terms of the LICENSE file in this repository.
