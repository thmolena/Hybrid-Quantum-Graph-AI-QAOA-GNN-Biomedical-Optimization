# Repo Restructure Plan

This document proposes a concrete restructuring path that makes the repository more credible as both a research artifact and a small engineering system.

## 1. Target outcome

The goal is not to turn this repository into a large product. The goal is to make it read as:

- one coherent research program
- with one canonical notebook
- supported by reproducible scripts
- and a thin but credible software structure

## 2. Main design decision

Treat the repository as a **single project with branch-specific modules**, not as three independent demos.

That means:

- one canonical experiment story
- two supporting branches
- one consistent terminology set
- one consistent artifact pipeline

## 3. Notebook architecture

### Canonical notebook

Keep:

- `notebooks/quantum_ai_bio_combined.ipynb`

Role:

- thesis notebook
- highest-level figures
- final summary tables
- cross-domain interpretation

This notebook should answer:

1. What problem is being addressed?
2. Why is graph-conditioned learning the shared lens?
3. What exactly is the proposed workflow?
4. What evidence supports it?

### Supporting notebook 1

Keep:

- `notebooks/qaoa_demo.ipynb`

Role:

- detailed QAOA branch notebook
- optimization setup
- transcriptomic graph construction
- warm-start baselines and ablations

### Supporting notebook 2

Keep:

- `notebooks/bio_demo.ipynb`

Role:

- detailed biomedical branch notebook
- cohort audit
- graph construction details
- operating-point and robustness analysis

### Rule

The supporting notebooks should feed the canonical notebook, not compete with it.

## 4. Proposed folder structure

```text
Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/
|- README.md
|- docs/
|  |- research_gap_assessment.md
|  |- repo_restructure_plan.md
|  |- contribution_statement.md
|- notebooks/
|  |- 00_canonical_pipeline.ipynb
|  |- 10_qaoa_branch.ipynb
|  |- 20_biomedical_branch.ipynb
|  |- figures/
|- experiments/
|  |- qaoa/
|  |  |- configs/
|  |  |- run_qaoa_baselines.py
|  |  |- run_qaoa_ablation.py
|  |- biomedical/
|  |  |- configs/
|  |  |- run_bio_baselines.py
|  |  |- run_bio_ablation.py
|- src/
|  |- common/
|  |- qaoa/
|  |- biomedical/
|  |- serving/
|- outputs/
|  |- tables/
|  |- figures/
|  |- models/
|- website/
```

You do not need to implement all of this at once. The important structural move is separating:

- research scripts
- reusable code
- presentation assets

## 5. Proposed code split

### `src/common/`

Shared utilities:

- seed handling
- metrics helpers
- plotting helpers
- configuration loading

### `src/qaoa/`

QAOA-specific code:

- graph construction for optimization branch
- simulator wrappers
- warm-start models
- evaluation metrics and baseline runners

### `src/biomedical/`

Biomedical-specific code:

- cohort loading
- preprocessing
- graph construction
- model definitions
- evaluation and calibration utilities

### `src/serving/`

Demo-only code:

- website inference API
- model loading for interactive demo

This separation matters because the current server should not implicitly define the research codebase.

## 6. Canonical experiment flow

The strongest version of this repo should let a reviewer do the following.

1. Run branch-specific scripts to generate branch outputs.
2. Load those outputs into the canonical notebook.
3. Inspect the final integrated tables and figures without rerunning exploratory analysis cells.

That creates a clean distinction between:

- experiment generation
- result synthesis
- presentation

## 7. Minimal engineering upgrades

These changes would materially improve credibility without turning the repo into a product effort.

1. Add a config file per experiment family.
2. Add one results directory per experiment run.
3. Add lightweight logging.
4. Add a small test suite for core graph and metric utilities.
5. Pin key dependency versions more tightly.

## 8. Minimal research upgrades

1. Add `experiments/qaoa/run_qaoa_baselines.py`.
2. Add `experiments/biomedical/run_bio_baselines.py`.
3. Export baseline tables into `outputs/tables/`.
4. Make the canonical notebook consume those tables directly.

## 9. Recommended migration sequence

### Phase 1: Narrative cleanup

1. Reposition README and manuscript.
2. Declare one canonical notebook.
3. Treat the other two notebooks as supporting technical notebooks.

### Phase 2: Script extraction

1. Move the strongest notebook logic into branch scripts.
2. Standardize outputs and tables.
3. Freeze baseline protocols.

### Phase 3: Code organization

1. Split `src/` into common, qaoa, biomedical, and serving.
2. Reduce notebook-local logic.
3. Add tests for shared utilities.

### Phase 4: Website cleanup

1. Make the website explicitly a demo layer.
2. Point the website at the canonical thesis, not at three separate demos.
3. Keep strong caveats around prototype status.

## 10. Bottom line

The most important structural change is conceptual, not cosmetic:

> move from three demos to one research story with two supporting branches.

Once that is done, the code and notebook structure can reinforce the same message instead of fragmenting it.