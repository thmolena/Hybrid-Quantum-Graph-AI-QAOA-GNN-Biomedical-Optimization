"""
gen_fig1_runtime.py
-------------------
Generates fig1_runtime.pdf: median wall-clock time (log scale) comparison
across six warm-start methods on n=14, p=2 MaxCut instances (48 graphs).
Data from Table I (tab:efficiency) of the manuscript.

Output: submission/figures/fig1_runtime.pdf
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
# Data  (median ± 1σ, ms)
# ---------------------------------------------------------------------------
methods  = ['Random\n(R=4)', 'Heuristic', 'k-NN', 'TQA', 'GNN-\npoint', 'UQ-QAOA']
medians  = np.array([528, 267, 132, 138, 133, 69], dtype=float)
errors   = np.array([ 62,  31,  17,  18,  16, 10], dtype=float)
colors   = ['#888888', '#4477AA', '#228833', '#CCBB44', '#AA3377', '#EE6677']

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.375, 2.6))
x = np.arange(len(methods))

bars = ax.bar(
    x, medians, yerr=errors, capsize=3,
    color=colors, edgecolor='#333333', linewidth=0.5, width=0.65,
    error_kw=dict(linewidth=0.8, capthick=0.8, ecolor='#333333'),
    zorder=3,
)

ax.set_yscale('log')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f'{int(v)}' if v >= 1 else f'{v:.1f}'
))
ax.set_ylim(20, 2000)
ax.set_yticks([30, 100, 300, 1000])
ax.yaxis.set_major_formatter(mticker.FixedFormatter(['30', '100', '300', '1000']))

ax.set_ylabel('Median wall-clock time (ms)')
ax.set_xticks(x)
ax.set_xticklabels(methods, linespacing=0.9)
ax.tick_params(axis='x', which='both', bottom=False)
ax.grid(axis='y', linewidth=0.4, linestyle='--', color='#cccccc', zorder=0)

# Speedup annotation on UQ-QAOA bar
ax.text(5, 90, '7.7×', ha='center', va='bottom', fontsize=7.5,
        color='#c00000', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.4)
out_path = os.path.join(FIGURES_DIR, 'fig1_runtime.pdf')
fig.savefig(out_path)
plt.close(fig)
print(f'Saved: {out_path}')
