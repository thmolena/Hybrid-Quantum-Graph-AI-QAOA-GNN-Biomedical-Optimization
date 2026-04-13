# Hybrid Quantum Graph AI for QAOA and Biomedical Screening

## Research Summary

[Project Landing Page](index.html)  
[Research Paper PDF](research_paper/research_paper.pdf)

This repository studies a graph-conditioned learning interface across two applied settings:

- transcriptomic co-expression graphs used to predict depth-2 QAOA parameters for MaxCut
- cardiotocography patient-similarity graphs used to predict pathologic-risk scores for screening

The repository summary is intentionally restricted to validated held-out gains. The central result is that graph conditioning improves learned QAOA initialization and that the residual clinical GCN improves multiple clinically relevant screening metrics over existing baselines in the biomedical branch.

## Executive Abstract

The practical bottleneck in fixed-depth QAOA is repeated classical parameter search. This repository evaluates whether graph-conditioned learning can replace that search while preserving useful decision quality on held-out transcriptomic graphs. It also evaluates whether the same graph-conditioned interface supports a biomedical screening task in which predictions are made on patient-similarity graphs rather than isolated rows.

Across fixed held-out evaluations, the graph-conditioned QAOA model improves on earlier learned initialization from 0.8208 to 0.8682 mean approximation ratio and reduces median runtime from 675.9 ms to 0.256 ms while retaining 99.95% of the direct-search objective. In the cardiotocography branch, ResidualClinicalGCN improves the earlier graph baseline from 96.7% to 98.8% accuracy, reduces false positives from 21 to 1 relative to logistic regression at the same true-positive count, and improves on several existing baselines in operating-point behavior.

## Validated Findings

| Finding | Result | Practical meaning |
| --- | --- | --- |
| QAOA graph conditioning improves learned initialization | 0.8208 to 0.8682 mean ratio | Better MaxCut decisions from a learned model on held-out biological graphs |
| QAOA inference replaces repeated search | 675.9 ms to 0.256 ms median runtime | Many more transcriptomic graphs can be evaluated in the same compute budget |
| QAOA quality is preserved at deployment speed | 99.95% retention of direct-search objective | Fast inference retains the objective level needed for the optimization task |
| ResidualClinicalGCN improves the prior graph baseline | 96.7% to 98.8% held-out accuracy | The graph formulation produces a materially stronger CTG screening model than the earlier graph variant |
| ResidualClinicalGCN sharply reduces false alarms versus logistic regression | 21 false positives to 1, with 31 pathologic cases detected in both models | Fewer unnecessary clinical escalations without sacrificing recovered pathologic cases |
| ResidualClinicalGCN improves operating-point behavior versus random forest | 29 TP and 7 FP to 31 TP and 1 FP | More pathologic cases are identified with fewer unnecessary alerts |

## Held-Out Evidence Tables

### Transcriptomic QAOA Optimization

Better performance in this branch means producing high-quality MaxCut decisions without rerunning an expensive classical parameter search for every new biological graph.

| Comparison | Existing model | This project | Validated gain |
| --- | --- | --- | --- |
| Zero-angle baseline | 0.7224 mean ratio | 0.8682 mean ratio | +0.1458 absolute |
| Prior-style learned baseline | 0.8208 mean ratio | 0.8682 mean ratio | +0.0474 absolute, or +5.77% |
| Direct classical search runtime | 675.9 ms median | 0.256 ms median | 2640x faster |
| Direct-search objective retention | reference target | 99.95% retained | Search-quality optimization at inference speed |

### CTG Biomedical Screening

Better performance in this branch means fewer unnecessary escalations and stronger recovery of pathologic cases at the selected screening threshold.

| Comparison | Existing model | This project | Validated gain |
| --- | --- | --- | --- |
| AdaptiveBioGCN | 96.7% accuracy | 98.8% accuracy | +2.1 percentage points |
| Logistic regression | 31 TP, 21 FP | 31 TP, 1 FP | 20 fewer false alarms |
| Random forest | 29 TP, 7 FP | 31 TP, 1 FP | 2 more pathologic cases found, 6 fewer false alarms |
| MLP | 98.4% accuracy, 0.926 balanced accuracy, 0.971 ROC AUC, 2 FP | 98.8% accuracy, 0.942 balanced accuracy, 0.978 ROC AUC, 1 FP | Higher accuracy, improved class-balance recovery, fewer false alarms |
| XGBoost false-positive control | 98.8% accuracy, 2 FP | 98.8% accuracy, 1 FP | One fewer unnecessary alert at the same accuracy |

## Why These Gains Matter

### QAOA application meaning

The transcriptomic branch is a workflow result rather than a purely numerical result. Each new biological graph can trigger another optimization run if parameters must be searched from scratch. The graph-conditioned model shortens that loop to a single forward pass while preserving useful objective quality. In practice, that means larger screening studies, faster ablation cycles, and lower marginal compute cost for graph-family evaluation.

### Biomedical application meaning

The biomedical branch is evaluated at a concrete operating point. In that setting, fewer false positives mean fewer avoidable downstream alerts, rechecks, and clinician interventions for non-pathologic cases. Higher true-positive recovery and stronger balanced accuracy mean the screening model is better aligned with the cases that matter most in a class-imbalanced clinical dataset.

## Visual Evidence

<table>
  <tr>
    <td width="50%">
      <img src="website/notebooks_html/figures/qaoa_demo_benchmark_overview.png" alt="Held-out QAOA benchmark overview" />
      <p><strong>Held-out QAOA benchmark.</strong> The graph-conditioned model improves earlier learned initialization and preserves the direct-search objective level on the held-out benchmark.</p>
    </td>
    <td width="50%">
      <img src="website/notebooks_html/figures/qaoa_demo_graph_conditioned_initialization.png" alt="QAOA graph-conditioned initialization figure" />
      <p><strong>Runtime reduction for optimization.</strong> The optimization workflow shifts from repeated search to inference-speed parameter selection.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="website/notebooks_html/figures/bio_demo_heldout_evaluation.png" alt="Held-out CTG evaluation" />
      <p><strong>Held-out CTG performance.</strong> The graph model reaches a strong operating point on accuracy, balanced accuracy, and ROC AUC.</p>
    </td>
    <td width="50%">
      <img src="website/notebooks_html/figures/bio_demo_operating_point_analysis.png" alt="CTG operating point analysis" />
      <p><strong>Operating-point behavior.</strong> The selected threshold combines strong pathologic-case recovery with one false positive.</p>
    </td>
  </tr>
</table>

## Core Artifacts

| Artifact | Path | Role |
| --- | --- | --- |
| Project landing page | [index.html](index.html) | Institutional report-style overview |
| Research paper | [research_paper/research_paper.pdf](research_paper/research_paper.pdf) | Manuscript artifact |
| Integrated notebook export | [website/notebooks_html/quantum_ai_bio_combined.html](website/notebooks_html/quantum_ai_bio_combined.html) | Combined executed narrative |
| QAOA notebook export | [website/notebooks_html/qaoa_demo.html](website/notebooks_html/qaoa_demo.html) | Transcriptomic optimization evidence |
| Biomedical notebook export | [website/notebooks_html/bio_demo.html](website/notebooks_html/bio_demo.html) | CTG screening evidence |
| QAOA baselines script | [experiments/qaoa/run_qaoa_baselines.py](experiments/qaoa/run_qaoa_baselines.py) | Reproduce QAOA comparison tables |
| Biomedical baselines script | [experiments/biomedical/run_bio_baselines.py](experiments/biomedical/run_bio_baselines.py) | Reproduce CTG comparison tables |

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

Run the extracted benchmark scripts:

```bash
python experiments/qaoa/run_qaoa_baselines.py
python experiments/biomedical/run_bio_baselines.py
```

## Repository Layout

```text
notebooks/      executed analyses and manuscript-aligned notebooks
experiments/    extracted benchmark scripts and evaluation runs
src/            models, simulators, utilities, and serving code
data/           source biomedical and transcriptomic inputs
outputs/        processed datasets, benchmark tables, and generated artifacts
research_paper/ manuscript sources and compiled PDF
website/        static site assets and notebook HTML exports
```

## License

This project is released under the terms of the LICENSE file in this repository.
