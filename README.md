# Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization

## Overview

This repository presents a research-oriented hybrid quantum-classical AI pipeline that integrates:

- Quantum Approximate Optimization Algorithm (QAOA)
- Graph Neural Networks (GNNs)
- Biomedical and network data analysis

The primary objective is to examine how **hybrid quantum-classical methods** can improve optimization quality and inference efficiency in graph-structured problems.

The workflow formulates domain-specific tasks as graph optimization problems and combines machine learning with variational quantum optimization to evaluate solution quality, computational behavior, and practical transferability.

This work highlights how **quantum-enhanced AI methods** may contribute to high-impact application areas such as:

- Healthcare resource allocation
- Biomedical experimental design
- Network resilience and infrastructure optimization

---

## Motivation

Many scientific and engineering systems naturally form **complex networks**, including:

- Biological interaction networks
- Healthcare infrastructure
- Supply chains
- Communication networks

These settings frequently require solving **large combinatorial optimization problems** that become computationally demanding under purely classical strategies, particularly as graph size and constraint complexity increase.

Hybrid quantum-classical approaches provide a principled framework:

- **Graph Neural Networks (GNNs)** learn structural representations from graph data.
- **QAOA** explores combinatorial objective landscapes through parameterized quantum circuits.
- **Classical ML pipelines** support data processing, model training, and empirical evaluation.

By combining these components, the project provides an experimental testbed for **high-dimensional graph optimization** with explicit hybrid feedback loops.

---

## System Architecture

The pipeline contains three major components.

### 1. Graph Representation

Raw data is transformed into graph structures.

Nodes represent entities such as:

- patients
- experiments
- medical facilities
- biological components

Edges represent relationships including:

- interactions
- dependencies
- resource flows
- network connections

Graph construction enables structured learning and optimization.

---

### 2. Graph Neural Networks (GNN)

Graph Neural Networks learn representations that capture the structure and relationships within the graph.

Typical models used include:

- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Message Passing Neural Networks (MPNN)

These models generate embeddings that summarize graph structure and guide downstream optimization.

---

### 3. Quantum Optimization with QAOA

Optimization problems are encoded into **QUBO (Quadratic Unconstrained Binary Optimization)** or **Ising models**.

The **Quantum Approximate Optimization Algorithm (QAOA)** is then used to search for high-quality solutions.

The process includes:

1. Mapping optimization problems to a quantum Hamiltonian
2. Constructing parameterized quantum circuits
3. Iteratively optimizing circuit parameters
4. Measuring candidate solutions

The classical and quantum components interact in a hybrid loop.

---

## Example Applications

### Healthcare Resource Allocation

Optimize the distribution of limited resources across healthcare systems, such as:

- hospital capacity
- equipment allocation
- emergency response logistics

---

### Biomedical Experimental Design

Determine optimal experiment combinations under constraints like:

- limited budgets
- limited lab time
- experimental dependencies

---

### Network Resilience

Identify critical nodes in biological or infrastructure networks and optimize reinforcement strategies to improve system robustness.

---

## Repository Structure

The repository is organized to separate core modeling code, datasets, reproducible notebooks, generated outputs, and demonstration assets.

```text
Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/
├── README.md
├── LICENSE
├── requirements.txt
├── index.html                  ← GitHub Pages root
├── data/
│   └── breast_cancer.csv
├── notebooks/
│   ├── bio_demo.ipynb
│   ├── qaoa_demo.ipynb
│   └── quantum_ai_bio_combined.ipynb
├── outputs/
│   ├── breast_cancer_processed.csv
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
    ├── index.html              ← local dev copy (served from website/)
    ├── style.css
    └── notebooks_html/
        ├── bio_demo.html
        ├── qaoa_demo.html
        └── quantum_ai_bio_combined.html
```

---

## Project Summary

This project serves as a reproducible research artifact for studying hybrid quantum-classical optimization on graph-structured tasks. The current implementation combines graph representation learning, QAOA parameter optimization, and lightweight deployment interfaces to evaluate how learned priors can accelerate variational optimization.

In addition to model code, the repository includes notebooks, exported HTML reports, and an interactive demo interface to support transparent communication of methods and outcomes. The overall design is intended for interdisciplinary audiences in AI, optimization, and computational biomedicine.

---

## Key Features

### End-to-End Hybrid Workflow

The codebase links graph generation, feature construction, GNN-based parameter prediction, and QAOA state simulation in a unified pipeline. This structure enables controlled experiments on whether learned graph representations improve optimization efficiency and consistency.

### Classical-Quantum Bridge via QAOA Parameter Learning

The training procedure samples graph instances, computes reference QAOA angles via classical optimization, and trains a GNN to infer those angles directly from graph structure. The resulting model approximates expensive search with fast, data-driven inference.

### Lightweight and Portable Implementation

The implementation is designed for standard Python environments with minimal dependencies, while optionally leveraging PyTorch Geometric when available. A fallback adjacency-based GNN path preserves reproducibility when specialized graph ML tooling is unavailable.

### Biomedical and Applied Optimization Demonstrations

Notebook artifacts include biomedical classification and hybrid optimization demonstrations, illustrating how a shared methodological framework can be adapted to real-world datasets and decision-support contexts.

### API and Web Demo Support

A Flask inference endpoint and static web demo provide an interactive interface for evaluating graph inputs and inspecting predicted QAOA parameters with associated expected-cut estimates. This supports rapid validation and communication of experimental behavior.

---

## Quick Start

The following steps reproduce a baseline workflow, from model training to API-based inference.

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Train the GNN predictor on generated graph data.

```bash
python -m src.train --dataset-size 20 --n 6 --p 1 --epochs 10 --model-path model.pt
```

3. Start the prediction API.

```bash
python -m src.server
```

4. Optionally serve the website demo locally in another terminal.

```bash
python -m http.server 8000
```

Then open `http://localhost:8000` and use the demo to interact with the `/predict` endpoint.

> **GitHub Pages**: the live demo is published at  
> `https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/`  
> The root `index.html` is the GitHub Pages entry point; it loads assets from `website/`.

---

## Roadmap

### Near-Term

Expand benchmark coverage to larger and more diverse graph families, with clearer experiment tracking for convergence behavior, optimization quality, and runtime comparisons between baseline and learned-parameter approaches.

### Mid-Term

Strengthen biomedical modeling with richer graph construction strategies, additional node and edge attributes, and stronger evaluation protocols tailored to applied healthcare and biological network tasks.

### Longer-Term

Integrate hardware-aware quantum backends and advanced hybrid training loops to move from simulation-centric studies toward deployment-oriented quantum-AI experimentation at larger scale.

---

## Contributing

Contributions are welcome in the form of algorithmic improvements, additional datasets, ablation studies, notebook enhancements, and reproducibility upgrades. Pull requests with concise technical rationale, experimental assumptions, and validation evidence are especially valuable for maintaining a rigorous and extensible research codebase.

---

## License

This project is released under the terms of the LICENSE file in this repository.
