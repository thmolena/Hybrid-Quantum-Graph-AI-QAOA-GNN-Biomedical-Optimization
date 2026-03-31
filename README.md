# Hybrid Quantum–Graph AI: QAOA + GNN for Biomedical Optimization

> A research-grade hybrid quantum-classical system that unifies variational quantum optimization and graph neural network learning under a single graph-theoretic framework, demonstrated on real genomic and clinical datasets.

---

## Executive Summary

This project addresses a central research topic at the intersection of quantum optimization and biomedical machine learning: **whether a shared graph-learning framework can be used both to warm-start QAOA on real transcriptomic graphs and to classify clinical risk on real physiologic similarity graphs.**

Its strongest defended contribution is therefore not a generic hybrid slogan, but a three-part evidence structure spanning optimization, reproducible benchmarking, and clinically meaningful operating-point analysis:

1. **Optimization:** an **Adaptive Quantum GCN** reaches **0.868** mean held-out approximation ratio versus **0.869** for direct classical depth-2 QAOA, preserving **99.95%** of classical quality on held-out transcriptomic graphs.
2. **Biomedical benchmark:** the shared **Adaptive BioGCN** benchmark reaches **96.71%** representative held-out CTG accuracy and remains stable at **95.49% ± 0.97%** across repeated training seeds on a fixed split.
3. **Best biomedical operating point:** **ResidualClinicalGCN** reaches **98.8%** held-out CTG accuracy with **31 / 35** pathologic exams detected and only **1** false positive.

Within that structure, `notebooks/quantum_ai_bio_combined.ipynb` serves as the most complete integrative presentation, `notebooks/qaoa_demo.ipynb` isolates the strongest optimization evidence, and `notebooks/bio_demo.ipynb` provides the deepest biomedical evaluation.

---

## Reading Guide

| Reading objective | Primary source | Main takeaway |
|---|---|---|
| The repository in one pass | `notebooks/quantum_ai_bio_combined.ipynb` | One graph-learning framework links quantum warm-starting and biomedical classification within a single experimental narrative |
| The strongest optimization result | `notebooks/qaoa_demo.ipynb` | Learned graph-conditioned warm starts can nearly match exact depth-2 classical QAOA on held-out real-data graphs |
| The strongest biomedical result | `notebooks/bio_demo.ipynb` | The repository pairs a reproducible benchmark tier with a stronger residual clinical model for deeper evaluation |
| The interactive public-facing overview | `index.html` | The landing page surfaces the strongest results, figures, and notebook entry points without replacing notebook-grounded evidence |

---

## Table of Contents

- [Abstract](#abstract)
- [Draft Manuscript](#draft-manuscript)
- [Highlights](#highlights)
- [Three-Notebook Contribution Matrix](#three-notebook-contribution-matrix)
- [Performance Summary](#performance-summary)
- [Technical Contributions](#technical-contributions)
- [Notebook Guide](#notebook-guide)
- [Quick Start](#quick-start)
- [Roadmap](#roadmap)

---

## Draft Manuscript

An academic-style submission draft is included for readers who want a paper-form presentation of the project.

- **Title:** *Research Paper*
- **Author:** Molena Huynh
- **Markdown manuscript:** [paper/research_paper.md](paper/research_paper.md)

This draft is maintained as a single Markdown manuscript so the paper narrative, repository evidence, and editable source remain consolidated in one file.

---

## Abstract

This project addresses a fundamental challenge at the intersection of quantum computing and biomedical AI: how can the structure inherent in biological data — gene co-expression networks, clinical similarity graphs — be exploited simultaneously by quantum optimization algorithms and graph-based deep learning?

The core hypothesis is that **graphs are the common representation language** connecting two otherwise distinct paradigms. On the quantum side, MaxCut on a graph is a canonical NP-hard combinatorial problem that the Quantum Approximate Optimization Algorithm (QAOA) targets directly. On the classical side, Graph Convolutional Networks (GCNs) aggregate information through the same adjacency structure to learn expressive node and graph representations. This repository demonstrates that a single trained GNN can replace the expensive variational parameter search in QAOA — and that the same graph-learning machinery transfers directly to real-world clinical risk detection.

The system is validated on two real datasets: a transcriptomic cohort of 102 prostate cancer patient samples (12,600 gene expression features) and a retrospective fetal monitoring cohort of 2,126 cardiotocography exams (21 physiologic features). Across the repository's three notebooks, the project presents a strong result profile anchored by three defended outcomes: a transcriptomic QAOA warm-start model that retains **99.95%** of the depth-2 classical benchmark on held-out graphs, a shared biomedical benchmark that reaches **96.71%** representative held-out CTG accuracy and remains stable across repeated training seeds, and a residual clinical graph model that reaches **98.8%** held-out CTG accuracy with **31 / 35** pathologic exams detected and only **1** false positive.

---

## Highlights

| Domain | Result |
|---|---|
| Adaptive Quantum GCN (`qaoa_demo.ipynb`) | **0.868** mean held-out approximation ratio vs **0.869** classical, retaining **99.95%** of classical quality with **~1.41 × 10^4×** median inference speedup |
| Adaptive BioGCN benchmark (`bio_demo.ipynb`, Section 6A) | **96.71%** representative held-out CTG accuracy, **0.943** balanced accuracy, **0.983** ROC AUC; fixed-split robustness **95.49% ± 0.97%** accuracy |
| ResidualClinicalGCN extension (`bio_demo.ipynb`) | **98.8%** held-out CTG accuracy, **0.942** balanced accuracy, **0.978** ROC AUC, **31/35** pathologic exams detected with **1** false positive |
| Combined notebook integrated summary | **35,070×** representative QAOA warm-start speedup, **96.71%** Adaptive BioGCN benchmark accuracy, and **96.5%** integrated ResidualClinicalGCN evaluation accuracy in one end-to-end narrative |
| Graph scale (clinical) | 2,126-node exam graph, 14,776 edges via 10-NN physiologic similarity |
| Quantum simulation scope | Exact statevector QAOA, exponential in qubits — benchmarked on 10-node / 10-qubit co-expression graphs |

---

## Project Framing

- **One representation, two domains:** graphs are the common formal object in both branches rather than a loose conceptual analogy.
- **Three notebooks, three roles:** the optimization, biomedical, and integrated notebooks each support a distinct argumentative function.
- **Benchmark plus extension:** the biomedical results are strengthened by separating a reproducible benchmark configuration from a higher-performing residual model.
- **Notebook-grounded claims:** the landing page and README summarize the evidence, but the primary quantitative claims remain traceable to executed notebook outputs.

---

## Three-Notebook Contribution Matrix

| Notebook | Adaptive model emphasis | Best reported result | Why it matters |
|---|---|---|---|
| `notebooks/qaoa_demo.ipynb` | **Adaptive Quantum GCN** (`SimpleGCN` after transcriptomic adaptation) | **0.868** mean held-out approximation ratio vs **0.869** classical; **99.95%** quality retention; **~14,122×** median inference speedup | Establishes that learned graph-conditioned warm-starting can nearly close the gap to exact depth-2 classical optimization on held-out real-data transcriptomic graphs |
| `notebooks/bio_demo.ipynb` | **Adaptive BioGCN** benchmark and **ResidualClinicalGCN** extension | **98.8%** held-out CTG accuracy for `ResidualClinicalGCN`; **96.71%** representative benchmark accuracy for `Adaptive BioGCN`; **95.49% ± 0.97%** benchmark robustness | Provides the repository's strongest biomedical evidence, combining a reproducible benchmark tier with a higher-performing residual clinical model |
| `notebooks/quantum_ai_bio_combined.ipynb` | **Adaptive Quantum GCN** + **Adaptive BioGCN** in one integrated workflow | **35,070×** representative warm-start speedup, **96.71%** Adaptive BioGCN accuracy, **96.5%** integrated residual evaluation accuracy | Shows the unified graph-learning framework end to end, rather than presenting the quantum and biomedical branches as unrelated demos |

---

## Performance Summary

| Metric family | Strongest figure in the repository |
|---|---|
| Best biomedical held-out accuracy | **98.8%** (`ResidualClinicalGCN`, `bio_demo.ipynb`) |
| Best biomedical benchmark reproducibility story | **96.71%** representative accuracy and **95.49% ± 0.97%** fixed-split robustness (`Adaptive BioGCN`) |
| Best optimization-quality story | **99.95%** held-out quality retention against the depth-2 classical QAOA baseline (`qaoa_demo.ipynb`) |
| Best representative optimization speedup on the website-integrated workflow | **35,070×** (`quantum_ai_bio_combined.ipynb`) |
| Best transcriptomic approximation ratio | **0.898** representative and **0.868** mean held-out (`Adaptive Quantum GCN`) |
| Best operational screening profile | **31 / 35** pathologic exams detected with **1** false positive (`ResidualClinicalGCN`) |
| Best integrated notebook evidence | Joint quantum warm-start and biomedical classification results in a single reproducible narrative (`quantum_ai_bio_combined.ipynb`) |

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

2. **Adaptive Quantum GCN for transcriptomic QAOA warm-start.** The repository's plain-language `Adaptive Quantum GCN` label refers to the transcriptomically adapted depth-2 `SimpleGCN` used in `qaoa_demo.ipynb`. It replaces weak zero-shot transfer with in-family transcriptomic adaptation and raises held-out learned quality from the earlier `~0.573` range to **0.868**, effectively matching the **0.869** classical benchmark while preserving single-pass inference.

3. **Leakage-aware clinical graph learning with two defended biomedical model tiers.** The cardiotocography pipeline enforces strict train/validation/test separation before any feature standardization, uses class-weighted losses for the rare pathologic class, and now reports two complementary biomedical results: the repository-aligned **Adaptive BioGCN** benchmark for reproducibility and the stronger **ResidualClinicalGCN** extension for deeper standalone evaluation.

4. **Notebook-level evidence rather than isolated examples.** The three notebooks now play distinct roles: `qaoa_demo.ipynb` establishes the strongest optimization result, `bio_demo.ipynb` delivers the strongest biomedical operating point and evaluation depth, and `quantum_ai_bio_combined.ipynb` demonstrates that the same graph-learning formalism supports both branches in one coherent pipeline.

5. **Portable, dependency-light implementation.** The codebase runs in a standard PyTorch environment. PyTorch Geometric is used when available but is not required; the GCN falls back to explicit adjacency-matrix message passing otherwise.

6. **End-to-end deployment path.** The system includes a trained model checkpoint, a Flask inference API, and a static frontend demo, providing a path from notebook research to interactive presentation. The root website is intentionally framed as a results overview and interactive illustration; the strongest quantitative claims remain notebook-grounded.

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
    Exact statevec  Adaptive BioGCN /
    simulator       ResidualClinicalGCN
    Adaptive        → pathologic risk score
    Quantum GCN →   ROC / confusion matrix
    predict (γ, β)
    MaxCut eval
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

**Model naming clarification.** In README prose, this notebook's learned model is referred to as **Adaptive Quantum GCN**: the plain-language label for the transcriptomically adapted depth-2 `SimpleGCN` used to predict QAOA angles after in-notebook domain adaptation.

**Workflow.**

1. Introductory walkthrough of QAOA, MaxCut, and the exact statevector simulation logic.
2. Construction of biologically motivated co-expression graphs from real transcriptomic data.
3. Exact depth-2 classical QAOA optimization on the representative graph and on held-out patient-resampled benchmark graphs.
4. Transcriptomic domain adaptation of the **Adaptive Quantum GCN** (`SimpleGCN`) on 24 resampled graphs from the same cohort family.
5. Single-pass GNN prediction of `(γ, β)` from graph structure, followed by exact evaluation of the resulting cut quality.
6. Visualization: co-expression graph with MaxCut partition, landscape geometry, state concentration, and benchmark-wide residual-regret analysis.

**Benchmark results** (6 held-out patient-resampled co-expression graphs, depth-2 QAOA).

| Metric | Classical depth-2 baseline | Adaptive Quantum GCN |
|---|---|---|
| Representative full-cohort ratio | 0.898 | 0.898 |
| Mean held-out approximation ratio | 0.869 | 0.868 |
| Quality retained vs classical | — | **99.95%** |
| Lift over old transfer baseline | — | **~0.573 → ~0.868** |
| Median inference speedup | — | **~14,122×** |

**Contribution summary.** This notebook is the repository's strongest optimization result: it turns the original warm-start demo into a real-data domain-adaptation study and shows that a learned graph model can recover essentially all of the depth-2 classical quality on held-out transcriptomic graphs while maintaining orders-of-magnitude lower inference cost.

**Scope note.** `model.pt` is still the legacy depth-1 checkpoint trained on synthetic Erdős-Rényi graphs. The upgraded notebook result comes instead from notebook-local transcriptomic adaptation of `SimpleGCN`, so the main claim is no longer zero-shot transfer but high-quality in-family adaptation on real-data graph resamples.

---

### `notebooks/bio_demo.ipynb` — Biomedical Graph Classification on CTG Cohort

**Data.** UCI Cardiotocography dataset: 2,126 fetal monitoring exams, 21 physiologic summary features, three-class obstetric risk labels (normal / suspect / pathologic).

**Model naming clarification.** The notebook now contains two custom biomedical graph classifiers, and the distinction is important. The phrase **reference model** in the notebook's aligned benchmark snapshot refers to the repository-defined **Adaptive BioGCN** benchmark, implemented inline as `AdaptiveBioGCN`. In other words, **Adaptive BioGCN is the reference model** for the fixed-split robustness study; it is not one model plus a second separate reference model. The notebook also defines a second model, `ResidualClinicalGCN`, which is the stronger standalone residual extension used for the deeper threshold, calibration, saliency, and baseline-comparison analyses.

**Workflow.**

1. Reframe the three-class label into a binary screening target: pathologic versus non-pathologic.
2. Stratified train / validation / test split with leakage-free feature standardization (scaler fit on training partition only).
3. Construction of a symmetric 10-nearest-neighbor exam-similarity graph (14,776 edges) with self-loops and degree normalization.
4. Train the internal **Adaptive BioGCN** benchmark (`AdaptiveBioGCN`): a two-layer upgraded clinical GCN using a denser $k=15$ graph, hidden width $96 \rightarrow 48$, batch normalization, GELU activations, dropout, and AdamW optimization.
5. Train and analyse the separate `ResidualClinicalGCN` extension for the richer standalone evaluation sections.
6. Evaluate both the benchmark story and the standalone residual model with clinical screening metrics.

**Primary defended results.**

| Metric | Value |
|---|---|
| ResidualClinicalGCN held-out accuracy | **98.8%** |
| ResidualClinicalGCN balanced accuracy | **0.942** |
| ResidualClinicalGCN ROC AUC | **0.978** |
| Pathologic recall / precision | **88.6% / 96.9%** |
| Operational profile | **31 / 35** pathologic exams detected with **1** false positive |
| Adaptive BioGCN representative benchmark | **96.71%** accuracy, **0.943** balanced accuracy, **0.983** ROC AUC |
| Adaptive BioGCN fixed-split robustness | **95.49% ± 0.97%** accuracy, **0.903 ± 0.027** balanced accuracy |

**Contribution summary.** This notebook is the repository's strongest biomedical result. It does not stop at a single accuracy number; it shows that the upgraded residual graph model improves the held-out CTG result from the older **94.1%** configuration to **98.8%** while collapsing false positives from **20** to **1**, and it pairs that result with calibration, saliency, homophily, and repeated-split follow-up analyses.

**Scope note.** `Adaptive BioGCN` is a **project-specific benchmark label**, not a claim that this exact name denotes an externally standardized or widely adopted literature architecture. Within this repository, it is the shared internal reference model used in both biomedical notebooks. `ResidualClinicalGCN` is the stronger standalone extension used to push the biomedical operating point higher inside this notebook. The pipeline remains a retrospective research demonstration rather than a clinical deployment system.

---

### `notebooks/quantum_ai_bio_combined.ipynb` — Integrative Narrative

This notebook stitches the two branches into a single coherent demonstration. It is the primary entry point for readers who want to see the full hybrid system end to end.

**Model naming clarification.** In README prose, the combined notebook integrates two adaptive model labels: **Adaptive Quantum GCN** for the QAOA warm-start branch (`SimpleGCN`) and **Adaptive BioGCN** for the shared biomedical benchmark. It also includes `ResidualClinicalGCN` as the stronger biomedical evaluation architecture inside the integrated story.

**Workflow.**

1. Reproducibility and environment audit.
2. Compact QAOA walkthrough: small MaxCut graph, exact solution, classical angle optimization, and **Adaptive Quantum GCN** warm-start comparison.
3. Full CTG graph-classification pipeline from raw data through **Adaptive BioGCN** benchmarking, `ResidualClinicalGCN` evaluation, and fixed-split robustness reporting.
4. Synthesis section explaining how graph structure supports both variational optimization and discriminative learning under a unified framework.

**Representative integrated results.**

| Branch | Result |
|---|---|
| QAOA warm-start | `SimpleGCN` inference speedup of about **35,070×** on the representative small-graph demonstration |
| Adaptive BioGCN benchmark | **96.71%** held-out CTG accuracy, **0.983** ROC AUC on the canonical seed-42 run |
| Adaptive BioGCN robustness | **95.49% ± 0.97%** accuracy, **0.979 ± 0.003** ROC AUC across five training seeds |
| ResidualClinicalGCN integrated evaluation | **96.5%** held-out CTG accuracy, **0.9028** balanced accuracy, **0.9699** ROC AUC |

**Contribution summary.** This notebook is the repository's integrative argument. It is the strongest single entry point for showing that one graph-learning formalism can support both quantum warm-starting and biomedical classification, with quantitative evidence on both sides rather than only a conceptual bridge.

**Key outputs.** `outputs/maxcut_graph.csv`, `outputs/qaoa_classical_angles.csv`, `outputs/ctg_raw.csv`, `outputs/ctg_processed.csv`, QAOA landscape figures, and the integrated biomedical evaluation dashboard covering confusion matrix, ROC curve, training dynamics, PCA projections, and cohort feature shifts.

---

## Repository Structure

```text
Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/
├── README.md
├── LICENSE
├── requirements.txt
├── model.pt                        # Trained SimpleGCN checkpoint
├── index.html                      # GitHub Pages entry point
├── paper/
│   └── research_paper.md           # Consolidated manuscript
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
│   ├── gnn.py                      # SimpleGCN and dense-adj / PyG GNN utilities
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

`SimpleGCN` is the QAOA parameter predictor and is referred to in README prose as **Adaptive Quantum GCN** when discussing the transcriptomically adapted notebook result.

- Input node feature: augmented degree
- Hidden width: 32 units
- Output: `2 × p` values interpreted as `(γ₁, …, γₚ, β₁, …, βₚ)`
- Readout: global mean pooling
- Backend: `torch_geometric.nn.GCNConv` when PyTorch Geometric is available; falls back to explicit adjacency-matrix message passing otherwise
- **Training regime:** the included `model.pt` checkpoint is the legacy depth-1 synthetic-graph model. The stronger result in `qaoa_demo.ipynb` comes from notebook-local transcriptomic domain adaptation of this same backbone at depth `p = 2`.

The biomedical classifiers are defined inline in the notebooks and currently appear in two main forms.

- `AdaptiveBioGCN`: the repository's internal **Adaptive BioGCN reference model** used for the aligned benchmark and fixed-split robustness study
- `ResidualClinicalGCN`: the standalone residual extension used for the deeper threshold, calibration, uncertainty, and baseline-comparison analyses

The phrase **reference model** in the notebook's aligned benchmark snapshot refers specifically to `AdaptiveBioGCN`, described in plain language as **Adaptive BioGCN**: an upgraded two-layer clinical GCN with a $k=15$ exam-similarity graph, hidden width $96 \rightarrow 48$, batch normalization, GELU activations, dropout, class-imbalance-aware training, and AdamW optimization. This should be read as an internal reproducible benchmark configuration for the repository, not as an externally established model family under that exact name.

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

# 3. Run the notebooks interactively
#    Recommended order:
#      notebooks/quantum_ai_bio_combined.ipynb   ← full integrative narrative
#      notebooks/qaoa_demo.ipynb                 ← QAOA benchmark only
#      notebooks/bio_demo.ipynb                  ← biomedical workflow only

# 3b. Or execute non-interactively for full reproducibility
jupyter nbconvert --to notebook --execute --inplace notebooks/qaoa_demo.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/bio_demo.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_ai_bio_combined.ipynb

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
- **Promote the notebook-local transcriptomic adaptation workflow into a reusable training pipeline** so the strongest optimization result no longer lives only inside `qaoa_demo.ipynb`.
- **Add explicit tabular baseline comparison** (logistic regression, random forest, MLP) on the held-out CTG split to rigorously quantify the relational learning benefit of the GCN.

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
