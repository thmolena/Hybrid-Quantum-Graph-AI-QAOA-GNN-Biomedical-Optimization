# Hybrid Quantum-Graph Learning for Structured Biomedical Optimization

### GNN-Informed Initialization for QAOA and Graph-Based Clinical Modeling

[![Project Website](https://img.shields.io/badge/Project_Website-Live_Overview-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)

## Abstract

This repository presents a hybrid quantum-classical framework for structured optimization and prediction in biomedical domains. We investigate whether graph neural networks (GNNs) can learn structure-aware representations that improve parameter initialization for the Quantum Approximate Optimization Algorithm (QAOA).

We evaluate this hypothesis on graph-structured biomedical settings, including transcriptomic co-expression networks and clinical similarity graphs. Results suggest that GNN-informed initialization yields competitive approximation performance relative to classical baselines in small-graph regimes, while the same graph-learning paradigm supports strong predictive performance in biomedical classification tasks.

This work is positioned as a research prototype: it demonstrates a unified graph-conditioned learning framework across quantum optimization and biomedical prediction, while clearly delineating current limitations and future scaling challenges. The accompanying manuscript in `paper/research_paper.md` is the NeurIPS-oriented paper version of the project narrative.

## 1. Motivation

Many real-world biomedical problems are naturally graph-structured and combinatorial, including:

* gene co-expression and pathway interaction networks
* patient similarity and cohort stratification
* molecular interaction and therapeutic selection graphs

These problems exhibit:

* high-dimensional relational structure
* non-convex combinatorial objectives
* sensitivity to initialization and inductive bias

QAOA offers a principled quantum approach to combinatorial optimization, but its performance is highly dependent on parameter initialization. This repository explores whether learned graph representations can provide meaningful inductive bias for QAOA parameterization.

## 2. Core Hypothesis

Structure-aware embeddings learned by GNNs can improve QAOA parameter initialization, leading to stronger convergence behavior and competitive solution quality on structured graph optimization problems.

## 3. Method Overview

The framework consists of three tightly coupled components:

### 3.1 Graph Representation Learning

* GNN encodes graph topology and node features
* produces embeddings capturing structural and relational information

### 3.2 Quantum Optimization (QAOA)

* optimization problem encoded as a cost Hamiltonian
* variational parameters determine solution quality

### 3.3 Hybrid Integration

* GNN outputs are used to initialize QAOA parameters as a warm start
* the shared graph representation also supports biomedical prediction tasks

## 4. System Architecture

```text
Graph Data
   -> GNN Encoder
      -> (A) QAOA Parameter Initialization -> QAOA Circuit -> Optimization Output
      -> (B) Graph Classifier -> Biomedical Prediction Output
```

This is a unified graph-conditioned learning framework, not a collection of separate pipelines.

## 5. Experimental Design

We evaluate three regimes:

| Method | Description |
| --- | --- |
| Classical Baselines | Standard optimization and machine-learning approaches |
| QAOA Baseline | QAOA with random or heuristic initialization |
| Hybrid Method | GNN-informed QAOA initialization plus graph-based prediction |

The strongest narrative is organized around one canonical notebook, supported by branch-specific technical appendices and extracted baseline scripts.

## 6. Evaluation Metrics

### Optimization (QAOA branch)

* approximation ratio
* objective value
* convergence behavior

### Biomedical Prediction

* accuracy and balanced accuracy
* ROC AUC
* sensitivity at clinically relevant operating points

## 7. Results (Prototype-Scale)

| Method | Result | Interpretation |
| --- | --- | --- |
| Classical QAOA | ~0.869 approximation ratio | Strong baseline in small graphs |
| GNN-Informed QAOA | ~0.868 approximation ratio | Competitive warm-start behavior |
| Biomedical Graph Models | up to 98.8% accuracy | Strong retrospective signal |

### Key Takeaways

* GNN initialization is competitive, not yet superior
* performance depends strongly on graph structure and evaluation protocol
* biomedical results are promising but not yet externally validated

## 8. Contributions

### Technical

* concrete prototype linking GNN embeddings to QAOA initialization
* unified treatment of optimization and prediction under graph learning

### Conceptual

* reframes QAOA initialization as a representation learning problem
* positions biomedical modeling and quantum optimization under a shared graph-conditioned paradigm

### Practical

* end-to-end prototype spanning graph learning, quantum simulation, and biomedical evaluation

## 9. Limitations

* QAOA experiments are limited to small-scale exact simulation
* no claim of quantum advantage is made
* results are not hardware-validated
* biomedical experiments are retrospective and cohort-specific
* the pipeline remains partially notebook-driven rather than fully automated

## 10. Future Work

### Research

* scaling to larger graphs and harder optimization regimes
* formal ablations on when GNN initialization is beneficial
* theoretical analysis of learned initialization landscapes

### Quantum

* integration with hardware backends or realistic noise models
* benchmarking against stronger classical optimizers

### Biomedical

* external validation on independent cohorts
* expansion to molecular and multi-modal graph data

### Engineering

* fully reproducible experiment pipelines
* modular experiment configuration system
* improved packaging and deployment

## 11. Implementation

### Stack

* Python
* PyTorch
* NumPy, SciPy, pandas, and scikit-learn
* custom QAOA simulator

Optional:

* PyTorch Geometric

## 12. Repository Structure

```text
notebooks/      # Core experiments and demonstrations
experiments/    # Scripted baselines
src/            # Models, simulators, utilities
data/           # Biomedical datasets
outputs/        # Results and artifacts
paper/          # Manuscript draft
docs/           # Research analysis and planning
website/        # Demo site and visualizations
```

## 13. Reproducibility

Install:

```bash
pip install -r requirements.txt
```

Run core notebook:

```bash
jupyter notebook notebooks/quantum_ai_bio_combined.ipynb
```

Run experiments:

```bash
python experiments/qaoa/run_qaoa_baselines.py
python experiments/biomedical/run_bio_baselines.py
```

For the manuscript-oriented version of the project narrative, see `paper/research_paper.md`.

## License

This project is released under the terms of the LICENSE file in this repository.