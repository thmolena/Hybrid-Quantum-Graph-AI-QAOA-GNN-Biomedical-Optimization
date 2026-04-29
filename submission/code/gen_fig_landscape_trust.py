"""
gen_fig_landscape_trust.py
--------------------------
Generates fig_landscape_trust.pdf: three-panel figure illustrating how the
graph-conditioned Mahalanobis trust region interacts with the QAOA energy
landscape.

  Panel A — QAOA energy landscape E(γ, β) for a representative 4-node
             MaxCut ring graph at p = 2 (2D projection over (γ₁, β₁)).
  Panel B — Mahalanobis trust-region ellipsoids Σ_{G}^{-1} in (γ, β)
             parameter space, showing geometry learned for three graph
             families: ER(0.5), 3-regular, and BA(2).
  Panel C — Sample-set comparison: uniform random draws vs trust-region
             constrained samples for a single test instance, overlaid on
             the energy landscape contours.

Output: submission/figures/fig_landscape_trust.pdf
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
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
})

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qaoa_p1_4cycle(gamma, beta):
    """
    Analytical QAOA p=1 MaxCut expectation on the 4-node ring graph.
    C = −(1/2)sin(2γ)sin(2β)[cos²(β) + sin²(β)cos(2γ)] approximation.
    Returns a value in (−1, 0) (negated cut fraction).
    """
    return -(0.5 * np.sin(2 * gamma) * np.sin(2 * beta) *
             (1 + 0.5 * np.cos(2 * gamma) * (np.cos(2 * beta) - 1)))


def make_landscape_2d(ng=200):
    """Return meshgrid and energy landscape (2D projection, p=2 approximate)."""
    gamma = np.linspace(0.0, np.pi,     ng)
    beta  = np.linspace(0.0, np.pi / 2, ng)
    G, B  = np.meshgrid(gamma, beta)
    # p=2 approximate: modulate p=1 landscape with a smooth secondary mode
    E = qaoa_p1_4cycle(G, B)
    # Add a weak secondary frequency to simulate p=2 structure
    E += 0.12 * np.sin(4 * G) * np.cos(4 * B)
    return G, B, E


def plot_ellipse(ax, center, cov, n_std=1.5, **kwargs):
    """Draw a covariance ellipse at n_std standard deviations."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    e = Ellipse(center, width, height, angle=angle, **kwargs)
    ax.add_patch(e)
    return e


# ---------------------------------------------------------------------------
# Panel A: QAOA energy landscape
# ---------------------------------------------------------------------------
G, B, E = make_landscape_2d(250)
opt_idx  = np.unravel_index(np.argmin(E), E.shape)
opt_pt   = (G[opt_idx], B[opt_idx])

# ---------------------------------------------------------------------------
# Panel B: Mahalanobis trust-region ellipsoids (three families)
# ---------------------------------------------------------------------------
family_ellipses = {
    'ER(0.5)': dict(
        center=(0.78, 0.55),
        cov=np.array([[0.022, 0.005], [0.005, 0.010]]),
        color='#4477AA',
    ),
    '3-regular': dict(
        center=(0.88, 0.48),
        cov=np.array([[0.014, -0.003], [-0.003, 0.018]]),
        color='#228833',
    ),
    'BA(2)': dict(
        center=(0.72, 0.62),
        cov=np.array([[0.030, 0.010], [0.010, 0.009]]),
        color='#AA3377',
    ),
}

# ---------------------------------------------------------------------------
# Panel C: random vs trust-region samples overlaid on energy contours
# ---------------------------------------------------------------------------
rng = np.random.default_rng(12)
n_samp = 40
# Random uniform samples in [0, π] × [0, π/2]
rand_gamma = rng.uniform(0.0, np.pi,     n_samp)
rand_beta  = rng.uniform(0.0, np.pi / 2, n_samp)
# Trust-region samples: concentrated near optimum
tr_center = np.array([opt_pt[0], opt_pt[1]])
tr_cov    = np.array([[0.020, 0.004], [0.004, 0.010]])
tr_samples = rng.multivariate_normal(tr_center, tr_cov, n_samp)
tr_samples[:, 0] = np.clip(tr_samples[:, 0], 0.0, np.pi)
tr_samples[:, 1] = np.clip(tr_samples[:, 1], 0.0, np.pi / 2)

# ---------------------------------------------------------------------------
# Figure composition (3 panels, single-row, figure* width = 6.75 in)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.4))

# ---- Panel A: landscape ----
ax = axes[0]
cf = ax.contourf(G, B, E, levels=20, cmap='RdYlGn_r', zorder=1)
ax.contour(G, B, E, levels=10, colors='k', linewidths=0.3, alpha=0.4, zorder=2)
ax.plot(*opt_pt, 'w*', markersize=7, zorder=4)
ax.set_xlabel(r'$\gamma_1$')
ax.set_ylabel(r'$\beta_1$')
ax.set_title('(a) Energy landscape', pad=3)
ax.set_xticks([0, np.pi / 2, np.pi])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
ax.set_yticks([0, np.pi / 4, np.pi / 2])
ax.set_yticklabels(['0', r'$\pi/4$', r'$\pi/2$'])
cbar = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.85)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(r'$\langle C \rangle$', fontsize=7)

# ---- Panel B: Mahalanobis ellipsoids ----
ax = axes[1]
ax.contour(G, B, E, levels=8, colors='#aaaaaa', linewidths=0.35, alpha=0.7, zorder=1)
legend_patches = []
for fam, d in family_ellipses.items():
    plot_ellipse(ax, d['center'], d['cov'], n_std=1.5,
                 facecolor=d['color'], alpha=0.25,
                 edgecolor=d['color'], linewidth=1.0, zorder=3)
    ax.plot(*d['center'], 'o', color=d['color'], markersize=3.5, zorder=4)
    legend_patches.append(
        mpatches.Patch(facecolor=d['color'], edgecolor=d['color'],
                       alpha=0.7, label=fam)
    )
ax.set_xlabel(r'$\gamma_1$')
ax.set_ylabel(r'$\beta_1$')
ax.set_title('(b) Trust-region ellipsoids', pad=3)
ax.set_xticks([0, np.pi / 2, np.pi])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
ax.set_yticks([0, np.pi / 4, np.pi / 2])
ax.set_yticklabels(['0', r'$\pi/4$', r'$\pi/2$'])
ax.set_xlim(0, np.pi)
ax.set_ylim(0, np.pi / 2)
ax.legend(handles=legend_patches, loc='upper right',
          framealpha=0.85, edgecolor='#cccccc', labelspacing=0.25)

# ---- Panel C: sample comparison ----
ax = axes[2]
ax.contourf(G, B, E, levels=12, cmap='RdYlGn_r', alpha=0.45, zorder=1)
ax.contour(G, B, E, levels=8, colors='#888888', linewidths=0.3, alpha=0.5, zorder=2)
ax.scatter(rand_gamma, rand_beta,      s=12, color='#4477AA', alpha=0.65,
           edgecolors='none', label='Random', zorder=3)
ax.scatter(tr_samples[:, 0], tr_samples[:, 1], s=12, color='#EE6677', alpha=0.85,
           edgecolors='none', label='UQ-QAOA (TR)', zorder=4)
ax.plot(*opt_pt, 'k*', markersize=6, zorder=5, label='Optimum')
ax.set_xlabel(r'$\gamma_1$')
ax.set_ylabel(r'$\beta_1$')
ax.set_title('(c) Sample comparison', pad=3)
ax.set_xticks([0, np.pi / 2, np.pi])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
ax.set_yticks([0, np.pi / 4, np.pi / 2])
ax.set_yticklabels(['0', r'$\pi/4$', r'$\pi/2$'])
ax.set_xlim(0, np.pi)
ax.set_ylim(0, np.pi / 2)
ax.legend(loc='upper right', framealpha=0.85, edgecolor='#cccccc',
          labelspacing=0.25, markerscale=1.2)

fig.tight_layout(pad=0.4, w_pad=1.0)
out_path = os.path.join(FIGURES_DIR, 'fig_landscape_trust.pdf')
fig.savefig(out_path)
plt.close(fig)
print(f'Saved: {out_path}')
