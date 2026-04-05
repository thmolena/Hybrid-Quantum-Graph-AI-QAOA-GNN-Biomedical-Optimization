# Research Gap Assessment

This document identifies the exact gaps between the current repository and a stronger research submission candidate.

## 1. Core diagnosis

The current repository has a coherent idea, real data, and promising results. The main weakness is not lack of ambition. The weakness is that the project still reads as an integrated prototype rather than a tightly validated contribution.

The current review-risk sentence is:

> interesting integration of known components, but the novel contribution and evaluation discipline remain under-specified.

## 2. Exact evidence gaps

### 2.1 Novelty gap

Current issue:

- the repository demonstrates a plausible cross-domain framing
- it does not yet force a reader to say what is algorithmically or empirically new

Why this matters:

- a strong paper needs one claim that is hard to collapse into "GNN + QAOA + biology"

What is needed:

1. One narrow primary contribution statement.
2. One sentence defining what prior work does not already show.
3. One experiment designed specifically to validate that delta.

Examples of acceptable contribution framing:

- graph-conditioned QAOA warm-starting on transcriptomic graph families under held-out family evaluation
- a unified graph-learning pipeline with task-specific evaluation that shows useful transfer of modeling principles across optimization and biomedical screening

Examples of weak contribution framing:

- hybrid quantum AI for biology
- combining GNNs, QAOA, and biomedical data

### 2.2 Baseline gap

Current issue:

- results are reported, but the baseline story is still incomplete and not uniformly controlled

Minimum required QAOA baselines:

1. Direct classical depth-2 QAOA optimization on the same held-out graphs.
2. A naive or random initialization baseline.
3. A non-graph learned regressor or MLP baseline for angle prediction.
4. If feasible, a parameter-transfer baseline from prior graph families.

Minimum required biomedical baselines:

1. Logistic regression.
2. Random forest or gradient-boosted tree ensemble.
3. MLP on the same tabular features.
4. A simpler graph baseline with fewer architectural degrees of freedom.

The important part is not just adding rows to a table. The important part is using:

- the same splits
- the same preprocessing discipline
- the same model-selection rule
- the same threshold-selection policy when thresholded metrics are reported

### 2.3 Ablation gap

Current issue:

- the repository shows that the full workflow can work
- it does not yet show which parts of the workflow are responsible

Needed QAOA ablations:

1. Remove graph structure or replace with weaker features.
2. Compare degree-only features against richer node features.
3. Compare transcriptomic adaptation against no adaptation.
4. Compare depth-1 and depth-2 prediction settings where applicable.

Needed biomedical ablations:

1. Remove graph edges and evaluate a purely tabular version.
2. Vary graph construction choices such as k-nearest-neighbor policy.
3. Compare benchmark and residual variants under matched training settings.
4. Test threshold sensitivity explicitly.

### 2.4 Statistical reporting gap

Current issue:

- some robustness reporting exists, but the uncertainty story is not fully unified

Needed improvements:

1. Report mean and variance for each primary model under a fixed protocol.
2. Separate representative run reporting from robustness reporting.
3. State how hyperparameters and thresholds were selected.
4. Distinguish exploratory analysis from confirmatory evaluation.

### 2.5 Scaling gap

Current issue:

- the optimization branch is strong in exact small-instance simulation
- it lacks a clearly bounded regime-of-use statement

Needed improvements:

1. State the target regime explicitly: small graph families, exact simulation, warm-start study.
2. Explain why that regime is scientifically interesting even without hardware scale.
3. If no larger-scale experiment is feasible, add a clear non-claim statement rather than implying broad scalability.

### 2.6 Biomedical depth gap

Current issue:

- the biomedical branch is well executed as a retrospective graph-learning exercise
- it still risks being read as a generic ML-on-health-data add-on

Needed improvements:

1. Make the clinical question narrower and more explicit.
2. Justify why the graph construction is domain-relevant rather than cosmetic.
3. Show operating-point reasoning, not just headline accuracy.
4. Be explicit that the current cohort is retrospective and single-source.

## 3. Notebook problem

Right now the notebooks can still be read as:

- QAOA demo
- biomedical demo
- combined demo

That is an exploratory progression, not a mature research presentation.

The correct target state is:

- one canonical notebook that tells the whole thesis
- two branch notebooks that support the canonical notebook with technical depth and ablations

## 4. Submission bar

To move from prototype to stronger submission candidate, the project should satisfy all of the following.

1. One primary claim.
2. One canonical evaluation table per branch.
3. One ablation table per branch.
4. One explicit statement of scope and non-claims.
5. One reproducible script path for the strongest reported results.

## 5. Recommended next actions

### Immediate

1. Reframe the repository around one canonical narrative.
2. Tighten the contribution statement to one sentence.
3. Standardize baseline reporting across both branches.

### Next research cycle

1. Add ablations.
2. Encode notebook logic into scripts.
3. Separate benchmark models from strongest-case models under clear selection rules.

### Before submission

1. Freeze splits and configs.
2. Regenerate final tables from scripted runs.
3. Rewrite the manuscript around the narrowed claim.