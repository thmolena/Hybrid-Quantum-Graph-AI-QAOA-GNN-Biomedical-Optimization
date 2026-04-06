# Graph-Conditioned Parameterization as a Task-Agnostic Interface

### Bridging Combinatorial Optimization and Clinical Prediction

[![Project Website](https://img.shields.io/badge/Project_Website-Open-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)

The GitHub Pages root landing page is the single public overview for this repository. The `website/` directory exists to hold the static assets and exported notebook HTML that power that landing page.

## Abstract

This repository studies whether graph-conditioned learning can serve as a reusable computational interface across two different problems: QAOA parameter initialization on transcriptomic co-expression graphs and graph-based biomedical screening on cardiotocography data.

On six held-out transcriptomic graphs, the adapted GNN initializer achieves a mean depth-2 QAOA approximation ratio of 0.8682 ± 0.0312, versus 0.8686 ± 0.0308 for direct classical search, while reducing median end-to-end latency from 675.9 ms to 0.256 ms. In the biomedical branch, the strongest graph model reaches 98.8% held-out accuracy and 0.942 balanced accuracy, while the expanded tabular baseline suite now includes calibrated logistic regression, calibrated random forest, XGBoost, and LightGBM variants.

The project is best read as a research study with explicit scope limits: it does not claim quantum advantage, clinical readiness, or head-to-head superiority over the latest learned-QAOA literature. The accompanying manuscript in `paper/research_paper.md` is the paper-form version of this positioning.

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

Structure-aware embeddings learned by GNNs can compress graph structure into actionable downstream signals, yielding competitive QAOA warm starts in one branch and clinically meaningful graph-based discrimination in another.

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

## 5. Evidence Snapshot

| Branch | Main result | Interpretation |
| --- | --- | --- |
| QAOA | 0.8682 ± 0.0312 vs. 0.8686 ± 0.0308 classical | Quality near parity, value is in the compute-quality tradeoff |
| QAOA runtime | 0.256 ms median vs. 675.9 ms classical | About 2640x lower median latency |
| Biomedical graph model | 98.8% accuracy, 0.942 balanced accuracy | Strong graph-based operating point |
| Biomedical tabular benchmark | 99.06% accuracy, 0.956 balanced accuracy | Calibrated LightGBM is the strongest current tabular baseline |

## 6. Experimental Design

We evaluate three regimes:

| Method | Description |
| --- | --- |
| Classical Baselines | Standard optimization and machine-learning approaches |
| QAOA Baseline | QAOA with random or heuristic initialization |
| Hybrid Method | GNN-informed QAOA initialization plus graph-based prediction |

The strongest narrative is organized around one canonical notebook, supported by branch-specific technical appendices and extracted baseline scripts.

## 7. Evaluation Metrics

### Optimization (QAOA branch)

* approximation ratio
* objective value
* convergence behavior

### Biomedical Prediction

* accuracy and balanced accuracy
* ROC AUC
* sensitivity at clinically relevant operating points

## 8. Results

| Method | Result | Interpretation |
| --- | --- | --- |
| Classical QAOA | 0.8686 ± 0.0308 | Exact held-out reference |
| GNN-Informed QAOA | 0.8682 ± 0.0312 | Retains 99.95% of classical quality |
| Best tabular biomedical baseline | 99.06% accuracy, 0.956 balanced accuracy | Calibrated LightGBM currently leads the non-graph table |
| Best graph biomedical model | 98.8% accuracy, 0.942 balanced accuracy | Strong graph operating point with 31/35 pathologic cases detected |

### Key Takeaways

* GNN initialization is competitive, not superior, and the key benefit is inference efficiency
* the QAOA claim is strengthened by explicit ablation, runtime, and convergence evidence
* the biomedical branch is now benchmarked against stronger tabular baselines rather than only lightweight comparators
* biomedical results remain retrospective and cohort-specific

## 9. Comparison to Prior Learned-QAOA Work

Recent learned-initialization papers such as Graph Learning for Parameter Prediction of Quantum Approximate Optimization Algorithm, QSeer: A Quantum-Inspired Graph Neural Network for Parameter Initialization in Quantum Approximate Optimization Algorithm Circuits, and Conditional Diffusion-based Parameter Generation for Quantum Approximate Optimization Algorithm push more directly on learned QAOA parameter generation itself. This repository makes a narrower claim. Its distinguishing contribution is the combination of biologically derived graph instances, held-out quality and runtime analysis, explicit ablations, and a shared graph-conditioned framing that also supports biomedical evaluation. A direct experimental comparison to those learned-QAOA methods remains future work.

## 10. Contributions

### Technical

* concrete prototype linking GNN embeddings to QAOA initialization
* unified treatment of optimization and prediction under graph learning

### Conceptual

* reframes QAOA initialization as a representation learning problem
* positions biomedical modeling and quantum optimization under a shared graph-conditioned paradigm

### Practical

* end-to-end prototype spanning graph learning, quantum simulation, and biomedical evaluation

## 11. Limitations

* QAOA experiments are limited to small-scale exact simulation
* no claim of quantum advantage is made
* results are not hardware-validated
* biomedical experiments are retrospective and cohort-specific
* the strongest tabular biomedical baselines are currently as strong as or stronger than the graph models on this dataset
* direct head-to-head comparison with recent learned-QAOA papers is still missing
* the pipeline remains partially notebook-driven rather than fully automated

## 12. Future Work

### Research

* scaling to larger graphs and harder optimization regimes
* formal ablations on when GNN initialization is beneficial
* theoretical analysis of learned initialization landscapes

### Quantum

* integration with hardware backends or realistic noise models
* benchmarking against recent learned-QAOA initialization methods and stronger classical optimizers

### Biomedical

* external validation on independent cohorts
* expansion to molecular and multi-modal graph data
* calibration-focused comparison between graph and tabular models

### Engineering

* fully reproducible experiment pipelines
* modular experiment configuration system
* improved packaging and deployment

## 13. Implementation

### Stack

* Python
* PyTorch
* NumPy, SciPy, pandas, and scikit-learn
* custom QAOA simulator

Optional:

* PyTorch Geometric

## 14. Repository Structure

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

## 15. Reproducibility

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