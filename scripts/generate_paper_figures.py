from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
QAOA_BASELINES_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_baselines.csv"
NOISE_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_noise_summary.csv"
SIZE_SWEEP_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_size_sweep.csv"
SEED_SWEEP_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_seed_sweep.csv"
ADAPTATION_SWEEP_TABLE = REPO_ROOT / "outputs" / "tables" / "qaoa_adaptation_sweep.csv"
PAPER_FIGURES = REPO_ROOT / "research_paper" / "figures"


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

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    palette = {
        "Classical depth-2 search angles": "#1b4965",
        "Graph-conditioned GNN (ours)": "#c1121f",
        "Heuristic mean-angle initializer": "#6c757d",
    }
    markers = {
        "Classical depth-2 search angles": "o",
        "Graph-conditioned GNN (ours)": "s",
        "Heuristic mean-angle initializer": "^",
    }

    for method in [
        "Classical depth-2 search angles",
        "Graph-conditioned GNN (ours)",
        "Heuristic mean-angle initializer",
    ]:
        method_df = data[data["method"] == method].sort_values("noise_rate")
        ax.plot(
            method_df["noise_rate"],
            method_df["mean_ratio"],
            marker=markers[method],
            color=palette[method],
            linewidth=2.2,
            markersize=6,
            label=method,
        )
        ax.fill_between(
            method_df["noise_rate"],
            method_df["mean_ratio"] - method_df["std_ratio"],
            method_df["mean_ratio"] + method_df["std_ratio"],
            color=palette[method],
            alpha=0.14,
            linewidth=0,
        )

    ax.set_xlabel("Local depolarizing rate $\\eta$")
    ax.set_ylabel("Held-out mean approximation ratio")
    ax.set_xlim(-0.002, 0.052)
    ax.set_ylim(0.75, 0.89)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Transcriptomic QAOA robustness under local depolarizing noise")

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_noise_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_size_sweep_figure() -> None:
    data = pd.read_csv(SIZE_SWEEP_TABLE)
    method_order = [
        "Classical depth-2 search",
        "Graph-conditioned GNN (ours)",
        "Heuristic mean-angle initializer",
    ]
    palette = {
        "Classical depth-2 search": "#1b4965",
        "Graph-conditioned GNN (ours)": "#c1121f",
        "Heuristic mean-angle initializer": "#6c757d",
    }
    markers = {
        "Classical depth-2 search": "o",
        "Graph-conditioned GNN (ours)": "s",
        "Heuristic mean-angle initializer": "^",
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

    fig, (ax_ratio, ax_runtime) = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    for method in method_order:
        method_df = data[data["method"] == method].sort_values("top_gene_count")
        ax_ratio.plot(
            method_df["top_gene_count"],
            method_df["mean_ratio"],
            marker=markers[method],
            linewidth=2.1,
            markersize=6,
            color=palette[method],
            label=method,
        )
        ax_ratio.fill_between(
            method_df["top_gene_count"],
            method_df["mean_ratio"] - method_df["std_ratio"],
            method_df["mean_ratio"] + method_df["std_ratio"],
            color=palette[method],
            alpha=0.12,
            linewidth=0,
        )

        ax_runtime.plot(
            method_df["top_gene_count"],
            method_df["median_runtime_ms"],
            marker=markers[method],
            linewidth=2.1,
            markersize=6,
            color=palette[method],
            label=method,
        )

    ax_ratio.set_xlabel("Top-gene panel size")
    ax_ratio.set_ylabel("Held-out mean approximation ratio")
    ax_ratio.set_title("Real transcriptomic size sweep")
    ax_ratio.grid(axis="y", alpha=0.22)
    ax_ratio.set_ylim(0.72, 0.96)

    ax_runtime.set_xlabel("Top-gene panel size")
    ax_runtime.set_ylabel("Median runtime per held-out graph (ms)")
    ax_runtime.set_title("Online cost across graph size")
    ax_runtime.set_yscale("log")
    ax_runtime.grid(axis="y", which="both", alpha=0.22)

    handles, labels = ax_ratio.get_legend_handles_labels()
    ax_runtime.legend(handles, labels, frameon=False, loc="upper left")

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_size_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_seed_sweep_figure() -> None:
    data = pd.read_csv(SEED_SWEEP_TABLE)
    learned = data[data["method"] == "Graph-conditioned GNN (ours)"].sort_values("training_seed")
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

    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    ax.plot(
        learned["training_seed"],
        learned["mean_ratio"],
        marker="o",
        linewidth=2.2,
        color="#c1121f",
        label="Graph-conditioned GNN (ours)",
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
    ax.legend(frameon=False, loc="lower right")

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_seed_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_adaptation_sweep_figure() -> None:
    data = pd.read_csv(ADAPTATION_SWEEP_TABLE)
    method_order = [
        "Classical depth-2 search",
        "Graph-conditioned GNN (ours)",
        "Heuristic mean-angle initializer",
    ]
    palette = {
        "Classical depth-2 search": "#1b4965",
        "Graph-conditioned GNN (ours)": "#c1121f",
        "Heuristic mean-angle initializer": "#6c757d",
    }
    markers = {
        "Classical depth-2 search": "o",
        "Graph-conditioned GNN (ours)": "s",
        "Heuristic mean-angle initializer": "^",
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

    fig, (ax_ratio, ax_loss) = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    for method in method_order:
        method_df = data[data["method"] == method].sort_values("adaptation_size")
        ax_ratio.plot(
            method_df["adaptation_size"],
            method_df["mean_ratio"],
            marker=markers[method],
            linewidth=2.1,
            markersize=6,
            color=palette[method],
            label=method,
        )
        ax_ratio.fill_between(
            method_df["adaptation_size"],
            method_df["mean_ratio"] - method_df["std_ratio"],
            method_df["mean_ratio"] + method_df["std_ratio"],
            color=palette[method],
            alpha=0.12,
            linewidth=0,
        )

    learned = data[data["method"] == "Graph-conditioned GNN (ours)"].sort_values("adaptation_size")
    ax_loss.plot(
        learned["adaptation_size"],
        learned["training_best_loss"],
        marker="o",
        linewidth=2.1,
        markersize=6,
        color="#7f5539",
    )

    ax_ratio.set_xlabel("Adaptation-family size")
    ax_ratio.set_ylabel("Held-out mean approximation ratio")
    ax_ratio.set_title("Sensitivity to adaptation-family size")
    ax_ratio.grid(axis="y", alpha=0.22)

    ax_loss.set_xlabel("Adaptation-family size")
    ax_loss.set_ylabel("Best training loss")
    ax_loss.set_title("Training fit across adaptation sizes")
    ax_loss.grid(axis="y", alpha=0.22)

    handles, labels = ax_ratio.get_legend_handles_labels()
    ax_loss.legend(handles, labels, frameon=False, loc="upper right")

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_adaptation_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_qaoa_pareto_figure()
    generate_qaoa_noise_figure()
    generate_qaoa_size_sweep_figure()
    generate_qaoa_seed_sweep_figure()
    generate_qaoa_adaptation_sweep_figure()