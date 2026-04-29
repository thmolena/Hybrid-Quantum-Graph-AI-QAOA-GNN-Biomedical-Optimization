"""
gen_fig2_evals.py
-----------------
Generates fig2_evals.pdf: mean circuit evaluations comparison across six
warm-start methods on n=14, p=2 MaxCut instances (48 graphs).
Data from Table I (tab:efficiency) of the manuscript.

Output: submission/figures/fig2_evals.pdf
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# PRX Quantum / APS two-column style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'lines.linewidth':    1.0,
})

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data  (mean ± 1σ circuit evaluations)
# ---------------------------------------------------------------------------
methods = ['Random\n(R=4)', 'Heuristic', 'k-NN', 'TQA', 'GNN-\npoint', 'UQ-QAOA']
means   = np.array([343, 172, 85, 109, 86, 45], dtype=float)
errors  = np.array([ 38,  19, 11,  14, 10,  7], dtype=float)
colors  = ['#888888', '#4477AA', '#228833', '#CCBB44', '#AA3377', '#EE6677']

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.375, 2.6))
x = np.arange(len(methods))

ax.bar(
    x, means, yerr=errors, capsize=3,
    color=colors, edgecolor='#333333', linewidth=0.5, width=0.65,
    error_kw=dict(linewidth=0.8, capthick=0.8, ecolor='#333333'),
    zorder=3,
)

ax.set_ylabel('Mean circuit evaluations')
ax.set_xticks(x)
ax.set_xticklabels(methods, linespacing=0.9)
ax.tick_params(axis='x', which='both', bottom=False)
ax.set_ylim(0, 420)
ax.grid(axis='y', linewidth=0.4, linestyle='--', color='#cccccc', zorder=0)

# Annotate reduction on UQ-QAOA
ax.text(5, 56, '86.9%\nreduction', ha='center', va='bottom', fontsize=7,
        color='#c00000', fontweight='bold', linespacing=1.1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.4)
out_path = os.path.join(FIGURES_DIR, 'fig2_evals.pdf')
fig.savefig(out_path)
plt.close(fig)
print(f'Saved: {out_path}')
