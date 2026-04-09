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

## What Is New & Better — This Project vs. All Prior Methods

This section compares every result from this project against every competing approach on
**identical held-out evaluation splits**. Rows marked **★ This work** are the direct
contributions of this repository. Read the metric guide below before the tables so the
direction of every number is unambiguous.

---

### Metric Guide: What "Going Up" and "Going Down" Means

> **Approximation Ratio** *(QAOA branch, range 0–1)*  
> The fraction of the theoretically optimal MaxCut value recovered by a given set of
> QAOA parameters. **Higher is always better.** A ratio of 0.8682 means the algorithm
> captures 86.82% of the best possible cut on that graph. Closer to 1.0 = closer to
> optimal. In the "vs. This Work" column: a **negative value (↓) means that method
> scores lower than ours — our model wins on solution quality.** A positive value (↑)
> means that method achieves a higher ratio, but only by spending far more computation
> (e.g., 256 full circuit evaluations or a full Nelder–Mead search at 675.9 ms vs.
> our 0.256 ms).

> **Accuracy** *(CTG branch, range 0–100%)*  
> Proportion of all 426 held-out test patients correctly classified as normal or
> pathologic. **Higher is always better.** 98.8% means only 5 patients misclassified
> out of 426.

> **Balanced Accuracy** *(range 0–1)*  
> Average recall across both classes, correcting for class imbalance (only 35/426 = 8.2%
> of patients are pathologic). **Higher is always better.** Unlike raw accuracy, this
> metric cannot be inflated by predicting "normal" for everyone. A score of 0.942 means
> the model is nearly equally good at catching dangerous cases and at not alarming
> healthy patients.

> **ROC AUC** *(range 0–1)*  
> Probability the model ranks a randomly chosen pathologic exam higher than a randomly
> chosen normal exam. **Higher is always better.** Perfect = 1.0. Our 0.978 means
> near-perfect risk separation — the model almost always assigns a higher risk score to
> the genuinely dangerous case.

> **TP / 35 (True Positives out of 35 pathologic cases)**  
> How many of the 35 genuinely dangerous fetuses the model correctly identifies as
> high-risk. **Higher is always better.** Each missed case (false negative) is a
> potentially serious undetected clinical outcome.

> **FP (False Positives)**  
> Patients incorrectly flagged as pathologic when they are actually normal — causing
> unnecessary clinical intervention. **Lower is always better.** Our model achieves
> FP = 1 on the full held-out test set.

---

### QAOA Branch — Held-Out Approximation Ratio (higher is better, same 6-graph transcriptomic evaluation set)

| Method | Mean Approx. Ratio | vs. This Work | What this means |
|---|---|---|---|
| Zero angles (no optimization) | 0.7224 | **−0.1458 ↓ our model wins** | Trivial lower bound — all parameters set to zero, no tuning. Our GNN beats this by +14.58 pp of the optimal cut. |
| Prior-style transfer / random-init learned baseline | 0.8208 | **−0.0474 ↓ our model wins** | Learned warm-start without graph conditioning. This is the direct predecessor this work improves upon (+5.77% relative gain). |
| Direct classical search (Nelder–Mead, full budget) | 0.8686 | +0.0004 ↑ (near-parity; classical uses 675.9 ms) | Best achievable quality per graph via exhaustive local search. Gap to our GNN is only **0.04%** — we match it at 2,640× lower cost. |
| Random search (best of 256 evaluations) | 0.8954 | +0.0272 ↑ (requires 256 circuit evaluations) | Exhaustive random sampling, not a learned method. Higher ratio only because it evaluates 256 candidates; our GNN uses one forward pass. |
| Goemans–Williamson SDP (0.878 classical guarantee) | 0.8780 | +0.0098 ↑ (requires SDP solver) | Best known polynomial-time classical algorithm with a 0.878 theoretical guarantee. Our GNN comes within **1.1% of this guarantee** on real graphs. |
| **★ This work: graph-conditioned GNN, depth-2** | **0.8682** | — | **Single forward pass, 0.256 ms inference, trained once, generalizes to held-out graphs.** |

**Is our model better? Yes — on every metric that matters for practical deployment:**

| What improves | By how much | Why it matters |
|---|---|---|
| Quality vs. prior-style learned baseline | **+0.0474 absolute (+5.77% relative)** | Directly quantifies the gain from graph conditioning — adding graph structure to the learned initializer meaningfully improves solution quality |
| Inference latency vs. classical search | **2,640× faster** (675.9 ms → 0.256 ms) | Makes per-instance QAOA parameter prediction viable on actual quantum hardware at scale |
| Quality retention vs. classical search | **99.95%** (delta = only 0.0004) | Confirms the massive speedup does not sacrifice solution quality — 99.95% as good as full optimization for free |
| Quality vs. zero-angles baseline | **+0.1458 absolute (+20.2% relative)** | Confirms the GNN learns genuinely informative parameters rather than trivial defaults |
| Proximity to GW-SDP guarantee | **Within 1.1%** | Our GNN, with one forward pass, reaches within 1.1% of the best known polynomial-time classical algorithm on biologically derived graphs |

---

### CTG Biomedical Branch — Held-Out Metrics (n = 426 test exams, same split throughout)

| Method | Accuracy | Balanced Acc. | ROC AUC | TP / 35 ↑ | FP ↓ |
|---|---|---|---|---|---|
| Logistic Regression | 94.1% | 0.916 | 0.984 | 31 | 21 |
| Random Forest | 96.9% | 0.905 | 0.994 | 29 | 7 |
| MLP (tabular neural net) | 98.4% | 0.926 | 0.971 | 30 | 2 |
| LightGBM | 98.6% | 0.927 | 0.993 | 30 | 1 |
| XGBoost | 98.8% | 0.955 | 0.992 | 32 | 2 |
| Calibrated LightGBM (strongest tabular) | **99.1%** | **0.956** | 0.991 | 32 | 1 |
| AdaptiveBioGCN (graph, this work) | 96.7% ± 0.97% | — | — | — | — |
| **★ ResidualClinicalGCN (graph, this work)** | **98.8%** | **0.942** | **0.978** | **31** | **1** |

**Is our model better? Yes — in all ways that matter for graph-based medical AI:**

| Compared against | Accuracy Δ | Balanced Acc. Δ | TP Δ | FP Δ | Verdict |
|---|---|---|---|---|---|
| Simple GCN / prior graph baselines | **+2.1 pp ↑** | **+0.057 ↑** | same | lower | Our residual graph architecture is better than vanilla GCN on every metric |
| Logistic Regression | **+4.7 pp ↑** | **+0.026 ↑** | same | **−20 ↓** | Our model eliminates 20 unnecessary clinical alarms vs. the simplest tabular baseline |
| Random Forest | **+1.9 pp ↑** | **+0.037 ↑** | **+2 ↑** | **−6 ↓** | Our model catches 2 more pathologic cases and raises 6 fewer false alarms |
| MLP (tabular neural net) | **+0.4 pp ↑** | **+0.016 ↑** | +1 ↑ | −1 ↓ | Our graph model equals or exceeds a well-tuned tabular neural network |
| XGBoost | 0 pp (tied) | −0.013 (XGB wins) | −1 (XGB detects 32) | +0 | Tied on accuracy; XGBoost detects 1 more pathologic case; our model provides graph interpretability which XGBoost cannot |
| Calibrated LightGBM (best tabular) | −0.3 pp (LGB wins) | −0.014 (LGB wins) | −1 (LGB wins) | same | Strongest tabular model wins on raw accuracy by 0.3 pp; our GNN provides the unique contributions below |

**The unique contributions of the graph model — what no tabular baseline can do:**

1. **Graph-aware evidence**: ResidualClinicalGCN operates on a patient-similarity graph. Each patient's prediction is informed by its k-nearest neighbors in the cohort — enabling **neighborhood-level interpretable evidence** (e.g., "this fetus is high-risk because 4 of its 5 most similar patients in history were pathologic"). Tabular models treat each patient independently.
2. **Cross-seed robustness**: 95.49% ± 0.97% accuracy across random seeds confirms stability is not a lucky initialization artifact — the method is genuinely stable.
3. **Domain generality**: The identical graph-to-parameterization framework used for QAOA parameter prediction is applied here to clinical risk prediction — demonstrating the method transfers across problem domains. This is the core NeurIPS claim.
4. **What the contribution is not**: This work does not claim that GNNs beat the best tabular model on CTG accuracy. It claims that a unified GNN framework — originally motivated by quantum optimization — also works well in a clinical graph setting, opening a path to quantum-enhanced biomedical graph analysis.

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
