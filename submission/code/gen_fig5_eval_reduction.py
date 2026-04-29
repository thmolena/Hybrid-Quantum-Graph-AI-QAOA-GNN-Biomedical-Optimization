"""
gen_fig5_eval_reduction.py
--------------------------
Generates fig5_eval_reduction.pdf: percentage reduction in circuit
evaluations relative to the Random(R=4) baseline for each warm-start method.
Data from Table II (tab:eff_quality) of the manuscript.

Output: submission/figures/fig5_eval_reduction.pdf
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
# Data  (evaluation reduction % relative to Random, mean ± 1σ)
# ---------------------------------------------------------------------------
methods  = ['Random\n(R=4)', 'Heuristic', 'k-NN', 'TQA', 'GNN-\npoint', 'UQ-QAOA']
red_pct  = np.array([ 0.0, 49.9, 75.2, 68.2, 74.9, 86.9])
errors   = np.array([ 0.0,  5.6,  3.1,  4.0,  3.0,  2.1])
colors   = ['#888888', '#4477AA', '#228833', '#CCBB44', '#AA3377', '#EE6677']

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.375, 2.6))
x = np.arange(len(methods))

bars = ax.bar(
    x, red_pct, yerr=errors, capsize=3,
    color=colors, edgecolor='#333333', linewidth=0.5, width=0.65,
    error_kw=dict(linewidth=0.8, capthick=0.8, ecolor='#333333'),
    zorder=3,
)

ax.set_ylabel('Circuit evaluation reduction (%)')
ax.set_xticks(x)
ax.set_xticklabels(methods, linespacing=0.9)
ax.tick_params(axis='x', which='both', bottom=False)
ax.set_ylim(0, 105)
ax.grid(axis='y', linewidth=0.4, linestyle='--', color='#cccccc', zorder=0)

# Label the UQ-QAOA bar
ax.text(5, 86.9 + 2.1 + 1.5, '86.9%', ha='center', va='bottom',
        fontsize=7.5, color='#c00000', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.4)
out_path = os.path.join(FIGURES_DIR, 'fig5_eval_reduction.pdf')
fig.savefig(out_path)
plt.close(fig)
print(f'Saved: {out_path}')
