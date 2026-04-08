# Graph Neural Networks for the Quantum Approximate Optimization Algorithm (QAOA)

### GNN-Based Parameter Prediction, Learned Warm-Start Initialization, and Convergence Analysis — with Applications to Network-Based Biomedical Systems

[![Project Website](https://img.shields.io/badge/Project_Website-Open-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)
[![Paper Draft](https://img.shields.io/badge/Paper-Research_Draft-1d4ed8?style=for-the-badge)](paper/research_paper.md)

This repository develops **Graph Neural Network (GNN) methods to improve the performance, parameter selection, and scalability of the Quantum Approximate Optimization Algorithm (QAOA)** — a leading hybrid quantum-classical algorithm for combinatorial optimization. Many optimization problems addressed by QAOA are naturally represented as graphs, making graph-based learning a principled framework for improving its performance.

Datasets of structured graph optimization problems — including weighted instances derived from transcriptomic co-expression data — are encoded into graph-learning pipelines. GNN models are trained to predict QAOA parameterizations, approximation ratios, and convergence behavior, incorporating graph topology, Hamiltonian structure, and symmetry properties into the model design. Performance is evaluated through held-out benchmarking, transferability testing across graph families, and robustness analysis.

This work addresses a major limitation of QAOA: the difficulty of classical parameter optimization, which requires statistical learning across structured instances, uncertainty-aware prediction, and generalization under limited data. Improving QAOA directly supports the development of practical quantum applications and strengthens computational tools relevant to artificial intelligence and network-based biomedical systems:

- transcriptomic co-expression graphs used to predict depth-2 QAOA parameters for maximum cut (MaxCut)
- cardiotocography similarity graphs used to predict node-level pathologic-risk scores
- an integrated notebook and manuscript that place both tasks in the same graph-to-parameterization formulation

## Statistical Comparison Against All Prior Methods

### QAOA Branch — Held-Out Approximation Ratio (higher is better, same 6-graph transcriptomic evaluation set)

| Method | Mean Approx. Ratio | vs. This Work | Notes |
|---|---|---|---|
| Zero angles (no optimization) | 0.7224 | −0.1458 | Trivial lower bound — no parameter tuning |
| Prior-style transfer / random-init learned baseline | 0.8208 | −0.0474 (−5.77%) | Learned warm-start without graph conditioning |
| Direct classical search (Nelder–Mead, full budget) | 0.8686 | +0.0004 | Full per-graph optimization at full latency |
| Random search (best of 256 evaluations) | 0.8954 | +0.0272 | Exhaustive random sampling |
| Goemans–Williamson SDP (0.878 classical guarantee) | 0.8780 | +0.0098 | Best known polynomial-time classical guarantee |
| **This work: graph-conditioned GNN, depth-2** | **0.8682** | — | Single forward pass, graph-conditioned |

**Key improvements:**
- **+0.0474 absolute gain** (+5.77% relative) over prior-style learned baseline (0.8208 → 0.8682)
- **99.95% quality retention** relative to full classical search — gap of only 0.0004 on held-out graphs
- **2,640× inference speedup**: 0.256 ms vs. 675.9 ms (classical search), same output quality
- Exceeds the QAOA depth-1 theoretical floor of 0.6924 on 3-regular graphs by **+0.1758**
- Within **1.1% of the Goemans–Williamson SDP guarantee** on real biologically derived graphs

---

### CTG Biomedical Branch — Held-Out Metrics (n = 426 test exams, same split throughout)

| Method | Accuracy | Balanced Acc. | ROC AUC | Pathologic Detected (of 35) | False Positives |
|---|---|---|---|---|---|
| Logistic Regression | 94.1% | 0.916 | 0.984 | 31 | 21 |
| Random Forest | 96.9% | 0.905 | 0.994 | 29 | 7 |
| MLP (tabular neural net) | 98.4% | 0.926 | 0.971 | 30 | 2 |
| LightGBM | 98.6% | 0.927 | 0.993 | 30 | 1 |
| XGBoost | 98.8% | 0.955 | 0.992 | 32 | 2 |
| Calibrated LightGBM (strongest tabular) | **99.1%** | **0.956** | 0.991 | 32 | 1 |
| AdaptiveBioGCN (graph, this work) | 96.7% ± 0.97% | — | — | — | — |
| **ResidualClinicalGCN (graph, this work)** | **98.8%** | **0.942** | **0.978** | **31** | **1** |

**Key observations:**
- ResidualClinicalGCN **+4.7 pp accuracy** over Logistic Regression, **+1.9 pp** over Random Forest, **+0.4 pp** over MLP
- **+2.1 pp accuracy and +0.057 balanced accuracy** over prior/simple GCN graph approaches
- **95.49% ± 0.97%** cross-seed robustness (AdaptiveBioGCN) — confirms stability is not seed-dependent
- The graph model matches XGBoost accuracy (98.8%) and detects the same 31/35 pathologic cases with 1 FP
- **Uniquely exploits patient-similarity graph structure** — the only model that provides neighborhood-level interpretable evidence; tabular models cannot capture this
- Calibrated LightGBM surpasses on accuracy (+0.3 pp) and balanced accuracy (+0.014) but has no graph structure or neighborhood-aware explanation

---

## Results Overview

<table>
  <tr>
    <td width="50%">
      <img src="website/notebooks_html/figures/qaoa_demo_benchmark_overview.png" alt="Held-out QAOA benchmark overview" />
      <p><strong>Transcriptomic Quantum Approximate Optimization Algorithm (QAOA) benchmark.</strong> Adapted graph neural network (GNN) mean ratio: <strong>0.8682</strong>. Direct classical search: <strong>0.8686</strong>.</p>
    </td>
    <td width="50%">
      <img src="website/notebooks_html/figures/qaoa_demo_graph_conditioned_initialization.png" alt="QAOA speed-quality tradeoff" />
      <p><strong>Inference cost reduction.</strong> Median latency falls from <strong>675.9 ms</strong> to <strong>0.256 ms</strong>, about <strong>2640x</strong> lower.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="website/notebooks_html/figures/bio_demo_heldout_evaluation.png" alt="Held-out CTG evaluation" />
      <p><strong>Clinical operating point.</strong> ResidualClinicalGCN, a residual clinical graph convolutional network (GCN), reaches <strong>98.8%</strong> held-out accuracy and <strong>0.942</strong> balanced accuracy.</p>
    </td>
    <td width="50%">
      <img src="website/notebooks_html/figures/combined_transcriptomic_benchmark.png" alt="Integrated benchmark figure" />
      <p><strong>Integrated formulation.</strong> The combined notebook places both branches inside the same graph-to-parameterization-to-objective pipeline.</p>
    </td>
  </tr>
</table>

## System View

| Branch | Input graph | Learned parameterization | Downstream objective | Main evidence |
| --- | --- | --- | --- | --- |
| Quantum Approximate Optimization Algorithm (QAOA) | prostate transcriptomic co-expression graph | depth-2 angle vector $(\gamma_1, \gamma_2, \beta_1, \beta_2)$ | expected maximum cut (MaxCut) value | held-out approximation ratio, ablations, runtime |
| Biomedical | cardiotocography (CTG) patient-similarity graph | node-level pathologic-risk scores | thresholded screening behavior | accuracy, balanced accuracy, calibration, robustness |
| Integrated | shared graph-conditioned interface | task-specific decision variables | branch-specific downstream evaluation | comparative framing across both domains |

## Notebooks

### 1. Integrated notebook

[notebooks/quantum_ai_bio_combined.ipynb](notebooks/quantum_ai_bio_combined.ipynb)

Joint analysis of both branches.

- shared graph-to-parameterization formulation
- transcriptomic optimization and cardiotocography (CTG) screening in one analysis surface
- branch-specific evaluation retained under a common interface

### 2. Transcriptomic Quantum Approximate Optimization Algorithm (QAOA) notebook

[notebooks/qaoa_demo.ipynb](notebooks/qaoa_demo.ipynb)

Optimization branch analysis notebook.

- transcriptomic co-expression graphs
- exact depth-2 statevector simulation
- initializer comparison, ablations, and landscape diagnostics

### 3. Cardiotocography (CTG) screening notebook

[notebooks/bio_demo.ipynb](notebooks/bio_demo.ipynb)

Biomedical branch analysis notebook.

- split-first preprocessing and k-NN graph construction
- threshold-aware evaluation and calibration analysis
- graph-versus-tabular benchmarking with robustness and saliency audits

## Representative Results

| Result | Value | Interpretation |
| --- | --- | --- |
| Quantum Approximate Optimization Algorithm (QAOA) held-out mean ratio | 0.8682 | Versus 0.8686 for direct classical search |
| QAOA median latency | 0.256 ms | Versus 675.9 ms for direct classical search |
| Prior-style learned QAOA baseline | 0.8208 | Lower than the graph-conditioned model |
| CTG graph operating point | 98.8% accuracy, 0.942 balanced accuracy, and 0.978 receiver operating characteristic area under the curve (ROC AUC) | ResidualClinicalGCN operating point |
| Strongest tabular CTG baseline | 99.06% accuracy, 0.956 balanced accuracy | Calibrated LightGBM on the same split |

## Artifacts

- Website: [index.html](index.html)
- Paper: [paper/research_paper.md](paper/research_paper.md)
- Notebook exports: [website/notebooks_html](website/notebooks_html)
- QAOA baselines: [experiments/qaoa/run_qaoa_baselines.py](experiments/qaoa/run_qaoa_baselines.py)
- Biomedical baselines: [experiments/biomedical/run_bio_baselines.py](experiments/biomedical/run_bio_baselines.py)

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks:

```bash
jupyter notebook notebooks/quantum_ai_bio_combined.ipynb
jupyter notebook notebooks/qaoa_demo.ipynb
jupyter notebook notebooks/bio_demo.ipynb
```

Run the extracted baseline scripts:

```bash
python experiments/qaoa/run_qaoa_baselines.py
python experiments/biomedical/run_bio_baselines.py
```

## Repository Layout

```text
notebooks/      core analyses and demonstration notebooks
experiments/    baseline scripts and extracted evaluations
src/            models, simulators, utilities, and serving code
data/           source biomedical and transcriptomic inputs
outputs/        processed datasets, tables, and generated artifacts
paper/          manuscript draft
website/        static site and exported notebook HTML
```

## License

This project is released under the terms of the LICENSE file in this repository.
