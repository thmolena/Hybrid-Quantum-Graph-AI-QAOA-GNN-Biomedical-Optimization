# Hybrid Quantum-Graph AI for Biomedical Optimization

### GNN-Initialized QAOA for Structure-Aware Optimization

> A research prototype exploring whether graph neural networks can provide useful structure-aware initialization for QAOA while also motivating a broader graph-learning view of biomedical optimization.

[![Project Website](https://img.shields.io/badge/Project_Website-Live_Overview-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)

---

## 🧠 Overview

This project explores a hybrid quantum-classical framework for graph-based biomedical optimization by combining:

* Graph Neural Networks (GNNs) for structure-aware representation learning
* Quantum Approximate Optimization Algorithm (QAOA) for combinatorial optimization

The repository currently operates as a research prototype rather than a finished research system or production engineering artifact. The strongest evidence is in notebook-driven experiments on transcriptomic graph optimization and biomedical screening-oriented graph learning.

---

## 🎯 Core Hypothesis

> GNN-derived graph representations can improve QAOA parameter initialization, leading to stronger convergence behavior and competitive solution quality on structured graph optimization problems relevant to biomedical settings.

In this repository, that hypothesis is explored most concretely through transcriptomic co-expression graphs for the QAOA branch and graph-based cardiotocography screening for the biomedical branch.

---

## 🔬 Problem Setting

We focus on graph-structured learning and optimization problems motivated by biomedical data, including:

* transcriptomic co-expression graphs used as structured optimization instances
* patient or exam similarity graphs for screening-oriented prediction
* broader biomedical graph settings such as pathway, molecular interaction, or therapeutic selection problems

These settings are typically:

* combinatorial or relational rather than purely tabular
* high-dimensional
* structurally complex

---

## ⚙️ Method Overview

Our working pipeline has three parts:

1. Graph representation

	* a GNN learns structure-aware embeddings from graph topology and node features
	* those embeddings are used to predict informative optimization parameters or relational predictions

2. Quantum optimization

	* QAOA encodes the target optimization objective as a cost Hamiltonian
	* variational parameters determine the quality of the approximate solution

3. Hybrid integration

	* the learned graph representation guides QAOA initialization in the optimization branch
	* the same graph-conditioned viewpoint supports biomedical prediction in the screening branch

---

## 🧩 Architecture

Pipeline:

Graph Data → GNN → Parameter Initialization / Risk Representation → QAOA Circuit or Graph Classifier → Optimization / Prediction → Output

For this repository, the architecture is best understood as one graph-conditioned learning thesis expressed across two branches rather than as three disconnected demos.

---

## 🧪 Experimental Setup

The current repository compares several modes of analysis:

| Method | Description |
| --- | --- |
| Classical Baseline | Direct classical optimization or standard ML baselines on the processed data |
| QAOA Baseline | QAOA with reference or random-style initialization |
| Hybrid Prototype | GNN-informed QAOA warm starts and graph-based biomedical prediction |

Current extracted baseline scripts are available for both branches under the experiments directory, but the strongest end-to-end story is still notebook-led.

---

## 📊 Evaluation Metrics

The repository currently reports or motivates metrics such as:

* approximation ratio
* convergence behavior
* objective value
* held-out accuracy
* balanced accuracy
* ROC AUC
* operating-point sensitivity for pathologic detection

---

## 📈 Results

Representative outputs in the current prototype include:

| Method | Representative Result | Interpretation |
| --- | --- | --- |
| Classical QAOA Optimization | Approximation ratio around 0.869 on the reported held-out family | Strong classical reference in the small-graph regime |
| GNN-Informed QAOA | Approximation ratio around 0.868 on the same family | Competitive warm-start evidence, but not a quantum-advantage claim |
| Biomedical Graph Models | Held-out accuracy up to 98.8% at the reported operating point | Strong retrospective signal, but not a clinical deployment claim |

Key observations:

* hybrid initialization appears competitive with direct classical angle search in the evaluated small-graph setting
* gains depend on graph family, protocol choice, and evaluation scope
* the biomedical branch shows strong retrospective classification performance, but broader validation is still missing

---

## ⚠️ Limitations

* the QAOA branch is still limited to small-graph exact simulation
* the strongest claims are still concentrated in notebook workflows rather than a fully scripted experiment framework
* no hardware-level quantum advantage is established
* the biomedical results are retrospective and cohort-specific
* the engineering layer remains lightweight and prototype-oriented

---

## 🚀 Future Work

* scale the QAOA branch to broader graph families and stronger baselines
* add clearer ablations showing when graph-conditioned initialization is actually necessary
* evaluate larger biomedical graph settings and external cohorts
* migrate more notebook logic into reproducible experiment scripts
* strengthen the serving and packaging layer so the research and engineering stories stop competing

---

## 🖥️ Implementation Notes

The current implementation is built around:

* Python
* PyTorch
* NumPy, pandas, SciPy, and scikit-learn
* an exact small-scale QAOA simulator in the repository codebase
* notebook-first experimentation plus lightweight Flask-based demo serving

Optional PyTorch Geometric support is included where available, but the repository is not currently built around a heavy external quantum SDK stack.

---

## 📂 Repository Structure

* `notebooks/` → canonical notebook plus two branch-specific technical appendices
* `experiments/` → extracted baseline scripts for QAOA and biomedical evaluation
* `src/` → shared code, simulators, baselines, and serving components
* `data/` → source biomedical and metadata assets
* `outputs/` → processed artifacts, graphs, and generated result tables
* `paper/` → manuscript draft
* `website/` → static demo site and exported notebook HTML
* `docs/` → research-gap and repository restructure plans

---

## 🧭 Key Contribution

The main contribution of the repository in its current form is not a finalized new algorithm. It is a concrete hybrid research prototype showing how classical graph learning can inform quantum optimization while also supporting a broader graph-based biomedical modeling narrative.

That is the right level of claim for the artifact today.

---

## 📌 Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the canonical notebook first:

```bash
jupyter notebook notebooks/quantum_ai_bio_combined.ipynb
```

Run the extracted baseline scripts:

```bash
./.venv/bin/python experiments/qaoa/run_qaoa_baselines.py
./.venv/bin/python experiments/biomedical/run_bio_baselines.py
```

Refresh the exported notebook HTML used by the site:

```bash
python scripts/export_notebook_html.py notebooks/qaoa_demo.ipynb --output qaoa_demo.html --output-dir website/notebooks_html
python scripts/export_notebook_html.py notebooks/bio_demo.ipynb --output bio_demo.html --output-dir website/notebooks_html
python scripts/export_notebook_html.py notebooks/quantum_ai_bio_combined.ipynb --output quantum_ai_bio_combined.html --output-dir website/notebooks_html
```

---

## 📚 References

This project is grounded in three surrounding literatures:

* QAOA and hybrid quantum-classical optimization
* Graph neural networks for structured representation learning
* graph-based and computational methods in biomedical modeling

For the current manuscript framing and the remaining evidence gaps, see:

* `paper/research_paper.md`
* `docs/research_gap_assessment.md`
* `docs/repo_restructure_plan.md`

---

## License

This project is released under the terms of the LICENSE file in this repository.