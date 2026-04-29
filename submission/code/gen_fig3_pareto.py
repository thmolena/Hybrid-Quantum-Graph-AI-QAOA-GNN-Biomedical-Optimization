"""
gen_fig3_pareto.py
------------------
Generates fig3_pareto.pdf: approximation ratio per 100 circuit evaluations
(efficiency-adjusted quality) across six warm-start methods.
Data from Table II (tab:eff_quality) of the manuscript.

Output: submission/figures/fig3_pareto.pdf
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
# Data  (ratio / 100 evaluations, mean ± 1σ)
# ---------------------------------------------------------------------------
methods = ['Random\n(R=4)', 'Heuristic', 'k-NN', 'TQA', 'GNN-\npoint', 'UQ-QAOA']
r100    = np.array([0.242, 0.503, 0.981, 0.794, 0.991, 1.862])
errors  = np.array([0.015, 0.031, 0.065, 0.048, 0.061, 0.098])
colors  = ['#888888', '#4477AA', '#228833', '#CCBB44', '#AA3377', '#EE6677']

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.375, 2.6))
x = np.arange(len(methods))

ax.bar(
    x, r100, yerr=errors, capsize=3,
    color=colors, edgecolor='#333333', linewidth=0.5, width=0.65,
    error_kw=dict(linewidth=0.8, capthick=0.8, ecolor='#333333'),
    zorder=3,
)

ax.axhline(1.0, color='#333333', linewidth=0.6, linestyle=':', zorder=2)

ax.set_ylabel(r'Approx. ratio per 100 circuit evals')
ax.set_xticks(x)
ax.set_xticklabels(methods, linespacing=0.9)
ax.tick_params(axis='x', which='both', bottom=False)
ax.set_ylim(0, 2.2)
ax.grid(axis='y', linewidth=0.4, linestyle='--', color='#cccccc', zorder=0)

# Label UQ-QAOA value
ax.text(5, 1.862 + 0.098 + 0.06, f'1.86', ha='center', va='bottom',
        fontsize=7.5, color='#c00000', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.4)
out_path = os.path.join(FIGURES_DIR, 'fig3_pareto.pdf')
fig.savefig(out_path)
plt.close(fig)
print(f'Saved: {out_path}')
