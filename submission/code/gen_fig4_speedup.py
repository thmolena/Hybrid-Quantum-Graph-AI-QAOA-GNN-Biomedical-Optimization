"""
gen_fig4_speedup.py
-------------------
Generates fig4_speedup.pdf: wall-clock speedup over Random baseline as a
function of graph size n ∈ {8, 10, 12, 14, 16} for five warm-start methods.
All models trained only at n=14 (zero-shot generalization).
Data from Table IV (tab:crosssize) of the manuscript.

Output: submission/figures/fig4_speedup.pdf
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
    'legend.fontsize':    7.5,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'lines.linewidth':    1.2,
})

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data  (speedup × ± 1σ): trained at n=14, evaluated at n=8–16
# ---------------------------------------------------------------------------
sizes = np.array([8, 10, 12, 14, 16])

data = {
    'Heuristic': dict(
        mu=np.array([1.8, 1.9, 1.9, 2.0, 1.9]),
        sd=np.array([0.2, 0.2, 0.3, 0.3, 0.3]),
        color='#4477AA', marker='s', ls='--',
    ),
    'k-NN': dict(
        mu=np.array([3.6, 3.7, 3.9, 4.0, 3.6]),
        sd=np.array([0.4, 0.4, 0.5, 0.5, 0.5]),
        color='#228833', marker='^', ls='-.',
    ),
    'TQA': dict(
        mu=np.array([3.5, 3.6, 3.7, 3.8, 3.5]),
        sd=np.array([0.4, 0.4, 0.5, 0.5, 0.5]),
        color='#CCBB44', marker='D', ls=':',
    ),
    'GNN-point': dict(
        mu=np.array([3.8, 3.9, 4.0, 4.0, 3.7]),
        sd=np.array([0.4, 0.5, 0.5, 0.5, 0.5]),
        color='#AA3377', marker='o', ls='-.',
    ),
    'UQ-QAOA': dict(
        mu=np.array([8.5, 8.3, 8.4, 7.7, 6.3]),
        sd=np.array([0.9, 0.8, 0.9, 0.8, 0.7]),
        color='#EE6677', marker='*', ls='-', lw=1.5,
    ),
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.375, 2.6))

for label, d in data.items():
    lw = d.get('lw', 1.2)
    ax.plot(sizes, d['mu'], label=label, color=d['color'],
            marker=d['marker'], markersize=4.5 if label == 'UQ-QAOA' else 3.5,
            linestyle=d['ls'], linewidth=lw, zorder=3)
    ax.fill_between(sizes, d['mu'] - d['sd'], d['mu'] + d['sd'],
                    color=d['color'], alpha=0.15, zorder=2)

ax.axhline(1.0, color='#888888', linewidth=0.6, linestyle=':', zorder=1,
           label='Random (1×)')
ax.set_xlabel('Graph size $n$')
ax.set_ylabel('Speedup over Random baseline (×)')
ax.set_xticks(sizes)
ax.set_ylim(0.5, 11)
ax.legend(loc='upper right', framealpha=0.85, edgecolor='#cccccc',
          labelspacing=0.3, handlelength=1.8)
ax.grid(linewidth=0.4, linestyle='--', color='#cccccc', zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.4)
out_path = os.path.join(FIGURES_DIR, 'fig4_speedup.pdf')
fig.savefig(out_path)
plt.close(fig)
print(f'Saved: {out_path}')
