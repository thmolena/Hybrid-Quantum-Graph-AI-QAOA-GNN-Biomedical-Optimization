# Learning QAOA Parameters with Graph Neural Networks

## Summary

We propose a method to learn fixed-depth QAOA parameters using graph neural networks, improving optimization efficiency and generalization across held-out problem instances in a structured weighted-graph family.

Primary artifacts:

- [Project landing page](index.html)
- [Research paper PDF](research_paper/research_paper.pdf)
- [QAOA notebook export](website/notebooks_html/qaoa_demo.html)

## Problem

QAOA requires parameter tuning that is both computationally expensive and instance-specific. In the fixed-depth setting studied here, each new graph would normally trigger another classical outer-loop optimization.

For structured biological graph families, that creates two practical problems:

- parameter search does not transfer reliably across graph instances
- repeated optimization increases runtime and compute cost

## Approach

We model QAOA parameter selection as a learning problem.

- Input: weighted graph structure and node attributes
- Model: graph neural network encoder
- Output: predicted QAOA parameters
- Evaluation: exact downstream objective on held-out graphs

The central idea is to replace repeated online search with graph-conditioned inference while preserving the objective quality that direct search would otherwise deliver.

## Results

On the transcriptomic MaxCut benchmark in this repository, the learned graph-conditioned initializer demonstrates:

- improved approximation ratio relative to the affine learned baseline: `0.8208 -> 0.8682`
- near-parity with direct multi-start classical search: `0.8682` versus `0.8686`
- sharply reduced online optimization time: `936.1255 ms -> 0.3652 ms`
- competitive noisy performance under local depolarizing noise: `0.7850` mean ratio at `eta = 0.05`

These results support the bounded claim that learning QAOA parameters via graph neural networks improves optimization performance and generalization across held-out graph instances in this fixed-depth regime.

## Method Overview

Graph -> GNN

GNN -> parameters

Parameters -> QAOA

The manuscript additionally provides:

- an exact QUBO/Ising encoding for the auxiliary biomedical objective
- a second-order regret bound connecting angle error to objective loss
- an expressivity separation from an affine graph-feature regressor baseline
- a finite-family non-plateau statement for the studied regime
- explicit resource-scaling discussion

## Paper

- [Formatted manuscript PDF](research_paper/research_paper.pdf)
- [LaTeX source](research_paper/main.tex)

## Code

- [Repository source](https://github.com/thmolena/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization)
- [QAOA baseline script](experiments/qaoa/run_qaoa_baselines.py)
- [Biomedical baseline script](experiments/biomedical/run_bio_baselines.py)
- [Core training and simulation modules](src/train.py)
- [QAOA simulator](src/qaoa_sim.py)

## Secondary Biomedical Note

This repository also contains a cardiotocography similarity-graph study. That branch is useful for demonstrating the broader graph-to-decision formulation, but it is not the main physics claim of the project. The strongest tabular baseline remains slightly better on the reported split, so the biomedical material should be read as secondary supporting evidence rather than the headline contribution.

## Requirements

```bash
pip install -r requirements.txt
```

## Reproducibility

```bash
python experiments/qaoa/run_qaoa_baselines.py
python experiments/biomedical/run_bio_baselines.py
```

## License

This project is released under the terms of the LICENSE file in this repository.
