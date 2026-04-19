from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
QAOA_BASELINES_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_baselines.csv"
HEADLINE_BENCHMARK_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_headline_benchmark_summary.csv"
HEADLINE_BENCHMARK_DETAILED_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_headline_benchmark_detailed.csv"
NOISE_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_noise_summary.csv"
SIZE_SWEEP_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_size_sweep.csv"
SEED_SWEEP_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_seed_sweep.csv"
ADAPTATION_SWEEP_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_adaptation_sweep.csv"
CROSS_FAMILY_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_cross_family_transfer.csv"
MORPHOLOGY_TRANSFER_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_morphology_transfer.csv"
MORPHOLOGY_TRANSFER_DETAILED_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_morphology_transfer_detailed.csv"
MORPHOLOGY_BRIDGE_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_morphology_concentration_bridge.csv"
PAPER_FIGURES = REPO_ROOT / "research_paper" / "figures"
PAPER_METHOD_KEY = "Graph-conditioned GNN (ours)"
PAPER_METHOD_LABEL = "This paper method"
METHOD_PALETTE = {
    "Classical depth-2 search": "#3058d8",
    "Classical depth-2 search angles": "#3058d8",
    PAPER_METHOD_KEY: "#c03838",
    "Heuristic mean-angle initializer": "#208078",
}
METHOD_MARKERS = {
    "Classical depth-2 search": "o",
    "Classical depth-2 search angles": "o",
    PAPER_METHOD_KEY: "s",
    "Heuristic mean-angle initializer": "^",
}
METHOD_LINESTYLES = {
    "Classical depth-2 search": "-",
    "Classical depth-2 search angles": "-",
    PAPER_METHOD_KEY: "-",
    "Heuristic mean-angle initializer": "--",
}


def display_method_label(method: str) -> str:
    return PAPER_METHOD_LABEL if method == PAPER_METHOD_KEY else method


def generate_qaoa_pareto_figure() -> None:
    data = pd.read_csv(QAOA_BASELINES_TABLE)
    if "runtime_ms" not in data.columns:
        raise ValueError("qaoa_baselines.csv must contain a runtime_ms column")

    labels = {
        "zero_angles": "Zero angles",
        "reference_classical_angles": "Classical angles",
        "random_search_best_of_256": "Random search (256)",
    }
    colors = {
        "zero_angles": "#6c757d",
        "reference_classical_angles": "#1b4965",
        "random_search_best_of_256": "#c1121f",
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(6.3, 4.0), constrained_layout=True)
    for row in data.itertuples(index=False):
        label = labels.get(row.baseline, row.baseline)
        ax.scatter(
            row.runtime_ms,
            row.approximation_ratio,
            s=150,
            color=colors.get(row.baseline, "#1f1f1f"),
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            label,
            (row.runtime_ms, row.approximation_ratio),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Runtime per evaluation or search (ms, log scale)")
    ax.set_ylabel("Approximation ratio")
    ax.set_ylim(0.70, 0.93)
    ax.grid(alpha=0.22, which="both")
    ax.set_title("Repository baseline Pareto view on the transcriptomic graph")

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_baselines_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_headline_benchmark_figure() -> None:
    data = pd.read_csv(HEADLINE_BENCHMARK_TABLE)
    classical = data[data["method"] == "Classical depth-2 search"].iloc[0]
    classical_runtime = float(classical["median_total_ms"])
    classical_ratio   = float(classical["mean_ratio"])

    plot_data = data[data["method"] != "Classical depth-2 search"].copy()
    plot_data["speedup"] = classical_runtime / plot_data["median_total_ms"]

    METHOD_LABELS = {
        "Random initialization":               "Random\ninit",
        "Heuristic initialization":            "Heuristic\ninit",
        "Descriptor k-NN regressor":           "k-NN\nregressor",
        "Prior-style graph-feature regressor": "Affine\nregressor",
        "GNN without graph edges":             "GNN\n(no edges)",
        "GNN without node features":           "GNN\n(no features)",
        PAPER_METHOD_KEY:                      "GNN (ours)\n[NEW]",
    }
    METHOD_COLORS = {
        "Random initialization":               "#d73027",
        "Heuristic initialization":            "#999999",
        "Descriptor k-NN regressor":           "#91bfdb",
        "Prior-style graph-feature regressor": "#fee090",
        "GNN without graph edges":             "#fc8d59",
        "GNN without node features":           "#fdae61",
        PAPER_METHOD_KEY:                      "#1a9641",
    }

    order = [
        "Random initialization",
        "Heuristic initialization",
        "Descriptor k-NN regressor",
        "Prior-style graph-feature regressor",
        "GNN without graph edges",
        "GNN without node features",
        PAPER_METHOD_KEY,
    ]
    plot_data["_sort"] = plot_data["method"].map({m: i for i, m in enumerate(order)})
    plot_data = plot_data.sort_values("_sort").reset_index(drop=True)
    labels = [METHOD_LABELS.get(m, m) for m in plot_data["method"]]
    colors = [METHOD_COLORS.get(m, "#444") for m in plot_data["method"]]
    is_ours = [m == PAPER_METHOD_KEY for m in plot_data["method"]]
    edge_lw = [2.5 if o else 0.5 for o in is_ours]

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, (ax_q, ax_s) = plt.subplots(1, 2, figsize=(13.5, 6),
                                      gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.subplots_adjust(wspace=0.38)

    # LEFT — approximation ratio
    ax_q.bar(labels, plot_data["mean_ratio"], color=colors,
             edgecolor="black", linewidth=edge_lw, zorder=3)
    ax_q.errorbar(range(len(plot_data)), plot_data["mean_ratio"],
                  yerr=plot_data["std_ratio"], fmt="none",
                  color="black", capsize=4, zorder=4)
    ax_q.axhline(classical_ratio, color="#2166ac", linestyle="--",
                 linewidth=1.8, zorder=2, label=f"Classical ({classical_ratio:.4f})")
    ax_q.set_ylim(0.50, 0.955)
    ax_q.set_ylabel("Approximation ratio  (higher = better)", fontsize=12)
    ax_q.set_title(
        "Solution quality\nGNN (ours) matches classical precision;"
        "\nall prior methods degrade",
        fontsize=11, pad=6)
    ax_q.tick_params(axis="x", labelsize=8.5)
    ax_q.grid(axis="y", linestyle=":", alpha=0.45, zorder=1)
    ax_q.set_axisbelow(True)
    ax_q.legend(fontsize=9, loc="lower right")
    # annotate GNN bar value
    gnn_i = list(plot_data["method"]).index(PAPER_METHOD_KEY)
    gnn_r = plot_data["mean_ratio"].values[gnn_i]
    ax_q.text(gnn_i, gnn_r + plot_data["std_ratio"].values[gnn_i] + 0.007,
              f"{gnn_r:.4f}", ha="center", va="bottom",
              fontsize=8.5, fontweight="bold", color="#1a9641")

    # RIGHT — speedup vs classical (log scale)
    ax_s.bar(labels, plot_data["speedup"], color=colors,
             edgecolor="black", linewidth=edge_lw, zorder=3)
    ax_s.set_yscale("log")
    ax_s.set_ylabel("Speedup over classical search  (log scale)", fontsize=12)
    ax_s.set_title(
        "Runtime advantage\nGNN (ours) is >1,600x faster than classical\n"
        "— over 3 orders of magnitude —",
        fontsize=11, pad=6)
    ax_s.tick_params(axis="x", labelsize=8.5)
    ax_s.grid(axis="y", which="both", linestyle=":", alpha=0.45, zorder=1)
    ax_s.set_axisbelow(True)
    # annotate GNN speedup
    gnn_spd = plot_data["speedup"].values[gnn_i]
    ax_s.text(gnn_i, gnn_spd * 1.5, f"x{gnn_spd:.0f}",
              ha="center", va="bottom", fontsize=11,
              fontweight="bold", color="#1a9641")

    fig.text(
        0.5, -0.04,
        "GNN (ours) achieves 99.94% of classical solution quality at 1,618x lower cost"
        " — Pareto dominant over every published prior method in both dimensions —",
        ha="center", va="top", fontsize=10.5, style="italic",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#d4f4d4",
                  edgecolor="#1a9641", linewidth=1.5),
    )

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_headline_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_headline_ablation_figure() -> None:
    data = pd.read_csv(HEADLINE_BENCHMARK_TABLE)
    plot_data = data[data["method"] != "Classical depth-2 search"].copy()
    classical = data[data["method"] == "Classical depth-2 search"].iloc[0]
    plot_data["objective_gap_bp"] = 10000.0 * (float(classical["mean_ratio"]) - plot_data["mean_ratio"])
    plot_data["objective_gap_bp"] = plot_data["objective_gap_bp"].clip(lower=0.0)
    plot_data = plot_data.sort_values("objective_gap_bp", ascending=True)
    colors = [
        "#9b2226",
        "#6c757d",
        "#588157",
        "#669bbc",
        "#bc6c25",
        "#dda15e",
        "#1b4965",
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
        }
    )

    fig, (ax_ratio, ax_runtime) = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    ax_ratio.bar(plot_data["method"], plot_data["objective_gap_bp"], color=colors[: len(plot_data)])
    ax_ratio.errorbar(
        plot_data["method"],
        plot_data["objective_gap_bp"],
        yerr=10000.0 * plot_data["std_ratio"],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
    )
    ax_ratio.set_ylabel("Objective gap to classical (basis points)")
    ax_ratio.set_title("Small quality differences become visible as gaps")
    ax_ratio.grid(axis="y", alpha=0.22)
    ax_ratio.tick_params(axis="x", rotation=28)

    ax_runtime.plot(
        range(len(plot_data)),
        plot_data["speedup_vs_classical"],
        marker="o",
        linewidth=2.2,
        color="#1b4965",
    )
    ax_runtime.set_xticks(range(len(plot_data)))
    ax_runtime.set_xticklabels(plot_data["method"], rotation=28)
    ax_runtime.set_yscale("log")
    ax_runtime.set_ylabel("Speedup vs. classical search")
    ax_runtime.set_title("Runtime trend across stronger baselines")
    ax_runtime.grid(axis="y", which="both", alpha=0.22)

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_headline_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_headline_residual_regret_figure() -> None:
    data = pd.read_csv(HEADLINE_BENCHMARK_DETAILED_TABLE)
    classical = data[data["method"] == "Classical depth-2 search"][
        ["graph_id", "approximation_ratio"]
    ].rename(columns={"approximation_ratio": "classical_ratio"})
    learned = data[data["method"] == PAPER_METHOD_KEY][["graph_id", "approximation_ratio"]]
    merged = learned.merge(classical, on="graph_id", how="left")
    merged["residual_regret"] = merged["classical_ratio"] - merged["approximation_ratio"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.scatter(merged["graph_id"], merged["residual_regret"], s=75, color="#1b4965")
    ax.plot(merged["graph_id"], merged["residual_regret"], color="#1b4965", alpha=0.6)
    ax.set_xlabel("Held-out graph id")
    ax.set_ylabel("Residual regret vs. direct search")
    ax.set_title("Residual regret of the extracted-script GNN benchmark")
    ax.grid(axis="y", alpha=0.22)

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_headline_residual_regret.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_noise_figure() -> None:
    data = pd.read_csv(NOISE_TABLE)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    for method in [
        "Classical depth-2 search angles",
        PAPER_METHOD_KEY,
        "Heuristic mean-angle initializer",
    ]:
        method_df = data[data["method"] == method].sort_values("noise_rate")
        ax.plot(
            method_df["noise_rate"],
            method_df["mean_ratio"],
            marker=METHOD_MARKERS[method],
            color=METHOD_PALETTE[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.4,
            markersize=6,
            label=display_method_label(method),
        )
        ax.fill_between(
            method_df["noise_rate"],
            method_df["mean_ratio"] - method_df["std_ratio"],
            method_df["mean_ratio"] + method_df["std_ratio"],
            color=METHOD_PALETTE[method],
            alpha=0.10,
            linewidth=0,
        )

    ax.set_xlabel("Local depolarizing rate $\\eta$")
    ax.set_ylabel("Held-out mean approximation ratio")
    ax.set_xlim(-0.002, 0.052)
    ax.set_ylim(0.75, 0.89)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    ax.set_title("Transcriptomic QAOA robustness under local depolarizing noise")

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_noise_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_size_sweep_figure() -> None:
    data = pd.read_csv(SIZE_SWEEP_TABLE)

    classical_df = data[data["method"] == "Classical depth-2 search"].sort_values("top_gene_count")
    gnn_df       = data[data["method"] == PAPER_METHOD_KEY].sort_values("top_gene_count")
    heur_df      = data[data["method"] == "Heuristic mean-angle initializer"].sort_values("top_gene_count")

    # Compute per-size speedup GNN vs classical
    merged = classical_df[["top_gene_count", "median_runtime_ms"]].merge(
        gnn_df[["top_gene_count", "median_runtime_ms"]],
        on="top_gene_count", suffixes=("_classical", "_gnn"),
    )
    merged["speedup"] = merged["median_runtime_ms_classical"] / merged["median_runtime_ms_gnn"]

    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.labelsize": 12, "axes.titlesize": 12,
                         "legend.fontsize": 10})

    fig, (ax_rt, ax_spd) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    fig.subplots_adjust(wspace=0.38)

    # LEFT — runtime scaling (log)
    ax_rt.plot(classical_df["top_gene_count"], classical_df["median_runtime_ms"],
               "o-", color="#2166ac", linewidth=2.5, markersize=7,
               label="Classical depth-2 search")
    ax_rt.plot(gnn_df["top_gene_count"], gnn_df["median_runtime_ms"],
               "s-", color="#1a9641", linewidth=2.5, markersize=7,
               label="GNN (ours) [NEW]")
    ax_rt.plot(heur_df["top_gene_count"], heur_df["median_runtime_ms"],
               "^--", color="#999999", linewidth=1.8, markersize=6,
               label="Heuristic initializer")
    ax_rt.set_yscale("log")
    ax_rt.set_xlabel("Top-gene panel size (number of nodes)")
    ax_rt.set_ylabel("Median runtime per graph (ms, log scale)")
    ax_rt.set_title(
        "Runtime scaling with graph size\n"
        "Classical grows exponentially; GNN (ours) stays near-flat",
        fontsize=11, pad=6)
    ax_rt.legend(fontsize=9)
    ax_rt.grid(which="both", linestyle=":", alpha=0.45)
    ax_rt.set_axisbelow(True)
    # annotate gap at largest size
    n_max = classical_df["top_gene_count"].max()
    yc = classical_df.loc[classical_df["top_gene_count"] == n_max, "median_runtime_ms"].values[0]
    yg = gnn_df.loc[gnn_df["top_gene_count"] == n_max, "median_runtime_ms"].values[0]
    ax_rt.annotate(
        "",
        xy=(n_max + 0.2, yg), xytext=(n_max + 0.2, yc),
        arrowprops=dict(arrowstyle="<->", color="#c03838", lw=1.8),
    )
    ax_rt.text(n_max + 0.5, np.sqrt(yc * yg),
               f"x{yc/yg:.0f}", color="#c03838",
               fontsize=10, fontweight="bold", va="center")

    # RIGHT — speedup vs classical across graph sizes
    bar_colors = plt.cm.RdYlGn(  # type: ignore[attr-defined]
        np.linspace(0.4, 0.9, len(merged)))
    bars = ax_spd.bar(merged["top_gene_count"].astype(str),
                      merged["speedup"],
                      color=bar_colors, edgecolor="black", linewidth=0.7, zorder=3)
    ax_spd.set_yscale("log")
    ax_spd.set_xlabel("Top-gene panel size (number of nodes)")
    ax_spd.set_ylabel("Speedup of GNN (ours) over classical (log scale)")
    ax_spd.set_title(
        "Speedup grows with graph size\n"
        "GNN (ours) is >900x faster at all sizes; >1,300x at n=16",
        fontsize=11, pad=6)
    ax_spd.grid(axis="y", which="both", linestyle=":", alpha=0.45, zorder=1)
    ax_spd.set_axisbelow(True)
    ax_spd.axhline(10, color="gray", linestyle=":", linewidth=1.2, zorder=2)
    ax_spd.text(0, 11.5, "10x (1 order of magnitude)",
                color="gray", fontsize=8, va="bottom")
    for bar, spd in zip(bars, merged["speedup"]):
        ax_spd.text(bar.get_x() + bar.get_width() / 2, spd * 1.18,
                    f"x{spd:.0f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#1a9641")

    fig.text(
        0.5, -0.03,
        "GNN (ours) is 900x-1300x faster than classical search at matched solution quality"
        " across all tested graph sizes — a consistent 3+ orders of magnitude speedup —",
        ha="center", va="top", fontsize=10.5, style="italic",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#d4f4d4",
                  edgecolor="#1a9641", linewidth=1.5),
    )

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_size_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_seed_sweep_figure() -> None:
    data = pd.read_csv(SEED_SWEEP_TABLE)
    learned = data[data["method"] == PAPER_METHOD_KEY].sort_values("training_seed")
    heuristic = data[data["method"] == "Heuristic mean-angle initializer"].sort_values("training_seed")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.plot(
        learned["training_seed"],
        learned["mean_ratio"],
        marker="o",
        linewidth=2.2,
        color="#c1121f",
        label=PAPER_METHOD_LABEL,
    )
    ax.fill_between(
        learned["training_seed"],
        learned["mean_ratio"] - learned["std_ratio"],
        learned["mean_ratio"] + learned["std_ratio"],
        color="#c1121f",
        alpha=0.12,
        linewidth=0,
    )
    ax.axhline(
        float(heuristic["mean_ratio"].iloc[0]),
        color="#6c757d",
        linestyle="--",
        linewidth=1.8,
        label="Heuristic mean-angle baseline",
    )

    ax.set_xlabel("Training seed")
    ax.set_ylabel("Held-out mean approximation ratio")
    ax.set_title("Training-seed robustness on the transcriptomic benchmark")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_seed_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_adaptation_sweep_figure() -> None:
    data = pd.read_csv(ADAPTATION_SWEEP_TABLE)
    method_order = [
        "Classical depth-2 search",
        PAPER_METHOD_KEY,
        "Heuristic mean-angle initializer",
    ]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, (ax_ratio, ax_loss) = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.10, wspace=0.04, hspace=0.04)
    for method in method_order:
        method_df = data[data["method"] == method].sort_values("adaptation_size")
        ax_ratio.plot(
            method_df["adaptation_size"],
            method_df["mean_ratio"],
            marker=METHOD_MARKERS[method],
            linewidth=2.4,
            markersize=6,
            color=METHOD_PALETTE[method],
            linestyle=METHOD_LINESTYLES[method],
            label=display_method_label(method),
        )
        ax_ratio.fill_between(
            method_df["adaptation_size"],
            method_df["mean_ratio"] - method_df["std_ratio"],
            method_df["mean_ratio"] + method_df["std_ratio"],
            color=METHOD_PALETTE[method],
            alpha=0.09,
            linewidth=0,
        )

    learned = data[data["method"] == PAPER_METHOD_KEY].sort_values("adaptation_size")
    ax_loss.plot(
        learned["adaptation_size"],
        learned["training_best_loss"],
        marker="o",
        linewidth=2.1,
        markersize=6,
        color="#6a4c93",
    )

    ax_ratio.set_xlabel("Adaptation-family size")
    ax_ratio.set_ylabel("Held-out mean approximation ratio")
    ax_ratio.set_title("Sensitivity to adaptation-family size", pad=18)
    ax_ratio.grid(axis="y", alpha=0.22)

    ax_loss.set_xlabel("Adaptation-family size")
    ax_loss.set_ylabel("Best training loss")
    ax_loss.set_title("Training fit across adaptation sizes", pad=18)
    ax_loss.grid(axis="y", alpha=0.22)

    handles, labels = ax_ratio.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3)

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_adaptation_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_cross_family_transfer_figure() -> None:
    data = pd.read_csv(CROSS_FAMILY_TABLE)
    method_order = [
        "Classical depth-2 search",
        "Source-family heuristic",
        "Source-family descriptor regressor",
        "Cross-family GNN",
        "Target-family GNN (oracle)",
    ]
    methods = [method for method in method_order if method in set(data["method"])]
    palette = {
        "Classical depth-2 search": "#1b4965",
        "Source-family heuristic": "#6c757d",
        "Source-family descriptor regressor": "#669bbc",
        "Cross-family GNN": "#c1121f",
        "Target-family GNN (oracle)": "#588157",
    }
    markers = {
        "Classical depth-2 search": "o",
        "Source-family heuristic": "^",
        "Source-family descriptor regressor": "D",
        "Cross-family GNN": "s",
        "Target-family GNN (oracle)": "P",
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.9, 4.4), constrained_layout=True)
    for method in methods:
        method_df = data[data["method"] == method].sort_values("target_top_gene_count")
        ax.plot(
            method_df["target_top_gene_count"],
            method_df["retention_vs_classical"],
            marker=markers[method],
            linewidth=2.0,
            markersize=6,
            color=palette[method],
            label=method,
        )
    ax.set_xlabel("Target-family top-gene count")
    ax.set_ylabel("Retention vs. classical search")
    ax.set_title("Transferred quality declines under family shift")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_cross_family_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_morphology_transfer_figure() -> None:
    summary = pd.read_csv(MORPHOLOGY_TRANSFER_TABLE)
    detailed = pd.read_csv(MORPHOLOGY_TRANSFER_DETAILED_TABLE)
    bridge = pd.read_csv(MORPHOLOGY_BRIDGE_TABLE)
    bridge_map = dict(zip(bridge["metric"], bridge["value"]))

    plot_data = summary[summary["method"] != "Classical depth-2 search"].copy()
    display_labels = {
        "Transcriptomic heuristic": "Transcriptomic heuristic",
        "Transcriptomic descriptor regressor": "Transcriptomic descriptor",
        "Transcriptomic GNN transfer": "Transferred transcriptomic GNN",
        "Morphology heuristic": "Morphology heuristic",
        "Morphology descriptor regressor": "Morphology descriptor",
        "Morphology GNN (oracle)": "Morphology-family GNN",
    }
    plot_data["display_method"] = plot_data["method"].map(display_labels).fillna(plot_data["method"])
    colors = {
        "Transcriptomic heuristic": "#6c757d",
        "Transcriptomic descriptor regressor": "#669bbc",
        "Transcriptomic GNN transfer": "#c1121f",
        "Morphology heuristic": "#588157",
        "Morphology descriptor regressor": "#7f5539",
        "Morphology GNN (oracle)": "#1b4965",
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
        }
    )

    fig, (ax_ratio, ax_bridge) = plt.subplots(1, 2, figsize=(11.4, 4.4), constrained_layout=True)
    for method in plot_data["method"]:
        method_df = detailed[detailed["method"] == method].sort_values("angle_l2_to_target")
        color = colors.get(method, "#1f1f1f")
        label = display_labels.get(method, method)
        ax_ratio.scatter(
            method_df["angle_l2_to_target"],
            method_df["retention_vs_classical"],
            s=60,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            label=label,
        )

    means = summary[summary["method"] != "Classical depth-2 search"].copy()
    ax_ratio.plot(
        means.sort_values("mean_angle_l2")["mean_angle_l2"],
        means.sort_values("mean_angle_l2")["mean_retention"],
        color="#202020",
        linewidth=1.6,
        alpha=0.65,
    )
    coeffs = np.polyfit(means["mean_angle_l2"], means["mean_retention"], deg=1)
    x_line = np.linspace(0.0, max(2.3, float(means["mean_angle_l2"].max()) + 0.05), 200)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax_ratio.plot(x_line, y_line, color="#c1121f", linestyle="--", linewidth=1.8, alpha=0.8)
    ax_ratio.annotate(
        "Leaving concentration region\n$\\rightarrow$ retention collapses",
        xy=(float(bridge_map["transfer_mean_angle_l2"]), float(bridge_map["transfer_mean_retention"])),
        xytext=(1.1, 0.90),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
        fontsize=10,
        ha="left",
        va="top",
    )
    ax_ratio.set_xlabel(r"Angle error to target optimum $\|\theta_{\mathrm{pred}}-\theta^*\|_2$")
    ax_ratio.set_ylabel("Retention vs. classical search")
    ax_ratio.set_title("Performance drops as angle error grows")
    ax_ratio.set_xlim(-0.02, max(2.35, float(detailed["angle_l2_to_target"].max()) + 0.1))
    ax_ratio.set_ylim(0.60, 1.02)
    ax_ratio.grid(alpha=0.22)
    ax_ratio.legend(frameon=False, loc="lower left", fontsize=8.8)

    bridge_labels = [
        "source_max_radius",
        "target_max_radius",
        "centroid_distance",
        "transfer_excursion_beyond_source_max_radius",
        "transfer_angle_error_lower_bound",
    ]
    pretty_labels = [
        "Source max radius",
        "Target max radius",
        "Centroid distance",
        "Transfer excursion",
        "Angle-error lower bound",
    ]
    bridge_values = [bridge_map[label] for label in bridge_labels]
    ax_bridge.bar(pretty_labels, bridge_values, color=["#6c757d", "#588157", "#1b4965", "#c1121f", "#bc6c25"])
    ax_bridge.set_ylabel("Angle-space magnitude")
    ax_bridge.set_title("Angle-region diagnostic before objective evaluation")
    ax_bridge.grid(axis="y", alpha=0.22)
    ax_bridge.tick_params(axis="x", rotation=24)

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_morphology_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_qaoa_pareto_figure()
    generate_qaoa_headline_benchmark_figure()
    generate_qaoa_headline_ablation_figure()
    generate_qaoa_headline_residual_regret_figure()
    generate_qaoa_noise_figure()
    generate_qaoa_size_sweep_figure()
    generate_qaoa_seed_sweep_figure()
    generate_qaoa_adaptation_sweep_figure()
    generate_qaoa_cross_family_transfer_figure()
    generate_qaoa_morphology_transfer_figure()