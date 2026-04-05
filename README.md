# Hybrid Quantum-Graph AI: Research Prototype for QAOA Warm-Starting and Biomedical Graph Learning

> A research prototype that explores whether one graph-learning workflow can support two related but distinct tasks: QAOA warm-start prediction on transcriptomic graphs and screening-oriented graph classification on biomedical data.

[![Project Website](https://img.shields.io/badge/Project_Website-Live_Overview-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)

---

## What This Repository Is

This repository should be read as a **research prototype and communication artifact**, not as either:

- a production-grade machine learning system
- a finalized research contribution with fully closed evaluation questions

Its value is in making one experimental thesis concrete:

> graph-conditioned learning may be useful both for amortizing QAOA parameter search on families of graph instances and for performing relational prediction on biomedical similarity graphs.

The repository contains real data, executable notebooks, a manuscript draft, and a website. That makes it stronger than a toy demo. At the same time, the strongest claims remain notebook-led, the evaluation story is still incomplete, and the engineering layer is intentionally lightweight.

---

## Current Status

The current artifact is best described as:

- **research prototype** in method and experimentation
- **demo project** in presentation and accessibility
- **not yet** a production engineering system
- **not yet** a submission-ready empirical paper

Three limits matter upfront.

1. The strongest optimization results are reported from notebook experiments rather than a reusable scripted pipeline.
2. The biomedical branch has promising results, but its baseline and external-validation story is not yet strong enough for a serious research claim.
3. The website demonstrates the ideas, but it is not evidence of deployment readiness or clinical utility.

---

## Working Thesis

The unifying idea in this repository is not that quantum optimization and biomedical screening are the same problem. It is that both can be cast as **graph-conditioned prediction problems**:

- In the QAOA branch, a GNN predicts useful QAOA angles from graph structure.
- In the biomedical branch, a graph model predicts pathologic risk from an exam-similarity graph.

That shared framing is the real intellectual center of the project. The repository is therefore strongest when treated as one cross-domain research program rather than three unrelated demos.

---

## Notebook Strategy

The repository currently contains three notebooks, but they should not be read as three separate contributions.

### Canonical notebook

`notebooks/quantum_ai_bio_combined.ipynb`

This should be treated as the **canonical narrative notebook**:

- it states the project thesis
- it shows how the graph-learning viewpoint connects the two branches
- it is the best entry point for readers evaluating the overall idea

### Supporting notebooks

`notebooks/qaoa_demo.ipynb`

- branch-specific technical notebook for the QAOA warm-start workflow
- best place to inspect optimization assumptions, simulation setup, and transcriptomic graph construction

`notebooks/bio_demo.ipynb`

- branch-specific technical notebook for the biomedical graph workflow
- best place to inspect screening metrics, threshold analysis, and cohort-specific evaluation

The recommended future direction is **not** to keep presenting these as three demos. It is to maintain:

- one canonical notebook for the full story
- two supporting notebooks for branch-level detail and ablations

---

## What the Repository Currently Demonstrates

### QAOA branch

The QAOA branch explores whether graph neural networks can predict depth-2 QAOA parameters on transcriptomic co-expression graphs well enough to reduce classical search cost.

Representative reported result:

- Adaptive Quantum GCN reaches a mean held-out approximation ratio of **0.868** versus **0.869** for direct classical depth-2 optimization on the evaluated held-out family.

Interpretation:

- promising warm-start evidence within exact simulation on small graph instances
- not evidence of hardware-level quantum advantage
- not yet a complete scaling argument

### Biomedical branch

The biomedical branch explores whether graph-based relational modeling improves retrospective screening-oriented prediction on the cardiotocography cohort.

Representative reported results:

- Adaptive BioGCN reaches **96.71%** representative held-out accuracy and **95.49% +/- 0.97%** fixed-split robustness.
- ResidualClinicalGCN reaches **98.8%** held-out accuracy with **31 / 35** pathologic exams detected and **1** false positive at the reported operating point.

Interpretation:

- strong retrospective graph-learning results on the chosen cohort
- not a deployment claim
- not yet sufficient for broader clinical generalization claims

---

## What the Repository Does Not Yet Establish

This repository does **not** yet establish the following.

### 1. A clearly isolated novel algorithmic contribution

The current work combines known ingredients in a coherent way, but the novelty boundary is still soft. A critical reader can still describe the project as an integration of:

- GNN-based structure learning
- QAOA warm-start prediction
- biomedical graph classification

without being forced to acknowledge a distinct new method.

### 2. A research-grade evaluation package

The main missing pieces are:

- controlled baseline comparisons under a shared protocol
- ablations that test whether the graph-conditioning mechanism itself is necessary
- clearer reporting of selection criteria for benchmark versus strongest-case models

### 3. A convincing scaling story

The QAOA branch is small-graph and exact-simulation based. That is acceptable for a prototype, but it means the work must be framed around the regime it actually targets rather than broad quantum-performance implications.

### 4. A deployment-grade engineering story

The codebase is intentionally thin:

- one lightweight training script
- one lightweight Flask server
- notebook-heavy experiment logic

That is appropriate for a prototype and insufficient for a production artifact.

---

## Repository Structure

```text
Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/
|- README.md
|- LICENSE
|- requirements.txt
|- model.pt
|- index.html
|- data/
|- notebooks/
|- outputs/
|- paper/
|  |- research_paper.md
|- scripts/
|- src/
|- website/
|- docs/
|  |- research_gap_assessment.md
|  |- repo_restructure_plan.md
```

---

## Core Code Status

### `src/train.py`

Prototype training entry point.

- useful for demonstrating the basic learning loop
- not yet a reproducible experiment framework
- does not yet encode the full transcriptomic adaptation workflow used for the strongest notebook claims

### `src/server.py`

Thin demo API.

- supports website interaction
- not designed as a hardened inference service

### `src/gnn.py`

Minimal GNN backbone.

- clear and inspectable
- appropriate for a prototype
- not yet the center of a fully characterized method section with architecture ablations

### `src/qaoa_sim.py`

Exact statevector simulator.

- transparent and useful for small-instance experimentation
- intrinsically limited in scale

---

## Recommended Reading Order

1. Start with `notebooks/quantum_ai_bio_combined.ipynb` for the full project thesis.
2. Read `notebooks/qaoa_demo.ipynb` if you want the optimization branch in detail.
3. Read `notebooks/bio_demo.ipynb` if you want the biomedical branch in detail.
4. Read `paper/research_paper.md` as the draft manuscript, with the understanding that it is a positioning document for an in-progress research prototype.
5. Read `docs/research_gap_assessment.md` for the exact evidence still missing.
6. Read `docs/repo_restructure_plan.md` for the path from demo repo to stronger research and engineering artifact.

---

## Quick Start

```bash
conda activate qaoa
pip install -r requirements.txt
python -m src.server
python -m http.server 8000
```

Then open `http://localhost:8000/` for the landing page.

To execute the notebooks directly:

```bash
conda activate qaoa
jupyter nbconvert --to notebook --execute --inplace notebooks/qaoa_demo.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/bio_demo.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_ai_bio_combined.ipynb
```

To refresh the static notebook exports:

```bash
conda activate qaoa
python scripts/export_notebook_html.py notebooks/qaoa_demo.ipynb --output qaoa_demo.html --output-dir website/notebooks_html
python scripts/export_notebook_html.py notebooks/bio_demo.ipynb --output bio_demo.html --output-dir website/notebooks_html
python scripts/export_notebook_html.py notebooks/quantum_ai_bio_combined.ipynb --output quantum_ai_bio_combined.html --output-dir website/notebooks_html
```

---

## Research Gaps and Engineering Plan

Two project-planning documents are included to make the next steps explicit.

- `docs/research_gap_assessment.md`: exact gaps between the current prototype and a stronger submission candidate
- `docs/repo_restructure_plan.md`: concrete plan to restructure the repository around one canonical notebook and a more credible research/engineering layout

---

## Scope and Limitations

- The current artifact is optimized for **clarity and exploration**, not for operational robustness.
- The strongest QAOA evidence is in **small-graph exact simulation**.
- The biomedical results are **retrospective** and should not be interpreted as clinical deployment claims.
- The integrated narrative is strongest as a **research direction** rather than a finished conclusion.

---

## Draft Manuscript

The repository includes an academic-style manuscript draft at `paper/research_paper.md`. It should be read as an in-progress paper built around the current prototype rather than as a final submission-ready manuscript.

---

## License

This project is released under the terms of the LICENSE file in this repository.