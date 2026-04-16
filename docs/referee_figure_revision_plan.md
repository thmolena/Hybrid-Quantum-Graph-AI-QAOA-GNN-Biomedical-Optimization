# Referee Figure Revision Plan

This note records the figure-side revisions implied by the PRX Quantum referee comment without changing experiments, datasets, or evaluation protocols.

## Core interpretation

The current figures are scientifically correct, but several of them communicate benchmark values more strongly than they communicate the geometric mechanism of the paper.

The manuscript's central scientific claim is:

> Parameter concentration governs within-family success and cross-family failure in amortized fixed-depth QAOA.

The referee concern is therefore a presentation mismatch:

- the paper is fundamentally about geometry, concentration, and transfer failure
- some of the current figures still read as flat performance summaries

## Figure-specific diagnosis

### Within-family benchmark figures

- The Pareto benchmark is valid, but the quality values are tightly clustered because the family is concentrated.
- The ablation figure is also valid, but the near-equality of the strong baselines makes it look visually flat.

Interpretation:

- these are concentration figures, not slope figures
- the visual message should be runtime separation plus narrow objective band

### Morphology transfer figure

- This is already the strongest figure in the paper.
- It contains both the empirical failure and the geometric reason for that failure.

Interpretation:

- this should be treated as the main mechanism figure
- the caption and surrounding text should emphasize that leaving the target concentration region causes the performance collapse

### Transcriptomic family-shift figure

- The current figure shows a clear ordering, but not an obvious monotone slope.

Interpretation:

- the figure should be introduced as an ordering plot that exposes robust degradation under family shift

### Size sweep and adaptation sweep

- These figures are useful because they show bounded operating regimes rather than universal monotone scaling.

Interpretation:

- the narrative should explicitly say that the scientifically relevant pattern is plateau-then-breakdown, not universal improvement or decline

## Required presentation fixes

### Immediate text-side fixes

1. Rewrite captions so they interpret rather than merely describe.
2. Add one sentence in the main text before or after each key figure explaining the idea the reader should see immediately.
3. Frame flat figures as evidence of concentration when that is the real scientific conclusion.

### Figure-side fixes using existing results

1. Re-express flat baseline figures in terms of objective gap to classical rather than raw approximation ratio.
2. Promote the morphology transfer figure to the lead mechanism figure.
3. Add one monotonic plot from existing outputs:
   - x-axis: angle distance, target-angle error, or excursion beyond concentration radius
   - y-axis: retained objective or approximation ratio
4. Add one data-driven geometry visualization from existing angle outputs:
   - transcriptomic cluster
   - morphology cluster
   - transferred predictions

## Constraints

Do not change:

- experiments
- datasets
- evaluation protocols

Change only:

- figure axes
- figure composition
- captions
- local interpretation in the manuscript

## Acceptance standard for revised figures

Each main-text figure should communicate one idea quickly:

1. within-family concentration enables search-free reuse
2. cross-family shift moves predictions outside the target concentration region
3. larger angle displacement produces larger objective loss
