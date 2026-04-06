# Graph-Conditioned Parameterization for Optimization and Clinical Risk Scoring

### Transcriptomic QAOA Initialization and CTG Similarity-Graph Risk Modeling

[![Project Website](https://img.shields.io/badge/Project_Website-Open-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)
[![Paper Draft](https://img.shields.io/badge/Paper-Research_Draft-1d4ed8?style=for-the-badge)](paper/research_paper.md)

This repository studies graph-conditioned parameterization in two settings:

- transcriptomic co-expression graphs used to predict depth-2 QAOA parameters for MaxCut
- cardiotocography similarity graphs used to predict node-level pathologic-risk scores
- an integrated notebook and manuscript that place both tasks in the same graph-to-parameterization formulation

## Results Overview

<table>
  <tr>
    <td width="50%">
      <img src="website/notebooks_html/figures/qaoa_demo_benchmark_overview.png" alt="Held-out QAOA benchmark overview" />
      <p><strong>Transcriptomic QAOA benchmark.</strong> Adapted GNN mean ratio: <strong>0.8682</strong>. Direct classical search: <strong>0.8686</strong>.</p>
    </td>
    <td width="50%">
      <img src="website/notebooks_html/figures/qaoa_demo_graph_conditioned_initialization.png" alt="QAOA speed-quality tradeoff" />
      <p><strong>Inference cost reduction.</strong> Median latency falls from <strong>675.9 ms</strong> to <strong>0.256 ms</strong>, about <strong>2640x</strong> lower.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="website/notebooks_html/figures/bio_demo_heldout_evaluation.png" alt="Held-out CTG evaluation" />
      <p><strong>Clinical operating point.</strong> ResidualClinicalGCN reaches <strong>98.8%</strong> held-out accuracy and <strong>0.942</strong> balanced accuracy.</p>
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
| QAOA | prostate transcriptomic co-expression graph | depth-2 angle vector $(\gamma_1, \gamma_2, \beta_1, \beta_2)$ | expected MaxCut value | held-out approximation ratio, ablations, runtime |
| Biomedical | CTG patient-similarity graph | node-level pathologic-risk scores | thresholded screening behavior | accuracy, balanced accuracy, calibration, robustness |
| Integrated | shared graph-conditioned interface | task-specific decision variables | branch-specific downstream evaluation | comparative framing across both domains |

## Notebooks

### 1. Integrated notebook

[notebooks/quantum_ai_bio_combined.ipynb](notebooks/quantum_ai_bio_combined.ipynb)

Integrated technical view of both branches.

- shared graph-to-parameterization formulation
- transcriptomic optimization and CTG screening in one workflow
- branch-specific evaluation retained inside a common interface

### 2. Transcriptomic QAOA notebook

[notebooks/qaoa_demo.ipynb](notebooks/qaoa_demo.ipynb)

Technical notebook for the optimization branch.

- transcriptomic co-expression graphs
- exact depth-2 statevector simulation
- initializer comparison, ablations, and landscape diagnostics

### 3. CTG screening notebook

[notebooks/bio_demo.ipynb](notebooks/bio_demo.ipynb)

Technical notebook for the biomedical branch.

- split-first preprocessing and k-NN graph construction
- threshold-aware evaluation and calibration analysis
- graph-versus-tabular benchmarking with robustness and saliency audits

## Representative Results

| Result | Value | Interpretation |
| --- | --- | --- |
| QAOA held-out mean ratio | 0.8682 | Near-parity with 0.8686 for direct classical search |
| QAOA median latency | 0.256 ms | About 2640x lower than direct classical search |
| Prior-style learned QAOA baseline | 0.8208 | Faster than search but weaker than the graph-conditioned model |
| CTG graph operating point | 98.8% accuracy, 0.942 balanced accuracy | Strong graph-based screening behavior |
| Strongest tabular CTG baseline | 99.06% accuracy, 0.956 balanced accuracy | Calibrated LightGBM remains slightly stronger on this split |

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

Run the main notebooks:

```bash
jupyter notebook notebooks/quantum_ai_bio_combined.ipynb
jupyter notebook notebooks/qaoa_demo.ipynb
jupyter notebook notebooks/bio_demo.ipynb
```

Run the baseline scripts:

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
