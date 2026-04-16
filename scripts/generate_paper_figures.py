from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    plot_data = data[data["method"] != "Classical depth-2 search"].copy()
    colors = {
        "Random initialization": "#9b2226",
        "Heuristic initialization": "#6c757d",
        "Descriptor k-NN regressor": "#588157",
        "Prior-style graph-feature regressor": "#669bbc",
        "GNN without graph edges": "#bc6c25",
        "GNN without node features": "#dda15e",
        PAPER_METHOD_KEY: "#1b4965",
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

    label_map = {
        "Random initialization": "Random",
        "Heuristic initialization": "Heuristic",
        "Descriptor k-NN regressor": "k-NN regressor",
        "Prior-style graph-feature regressor": "Affine regressor",
        "GNN without graph edges": "GNN without edges",
        "GNN without node features": "GNN without features",
        PAPER_METHOD_KEY: PAPER_METHOD_LABEL,
    }

    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    for row in plot_data.itertuples(index=False):
        ax.scatter(
            row.median_total_ms,
            row.mean_ratio,
            s=150,
            color=colors.get(row.method, "#1f1f1f"),
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            label=label_map.get(row.method, row.method),
        )

    classical = data[data["method"] == "Classical depth-2 search"].iloc[0]
    ax.axhline(classical["mean_ratio"], color="#1b4965", linestyle="--", linewidth=1.6, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Median end-to-end runtime per held-out graph (ms, log scale)")
    ax.set_ylabel("Held-out mean approximation ratio")
    ax.set_ylim(0.60, float(classical["mean_ratio"]) + 0.03)
    ax.grid(alpha=0.22, which="both")
    ax.set_title("Extracted-script headline benchmark on transcriptomic graphs")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "qaoa_headline_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qaoa_headline_ablation_figure() -> None:
    data = pd.read_csv(HEADLINE_BENCHMARK_TABLE)
    plot_data = data[data["method"] != "Classical depth-2 search"].copy()
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
    ax_ratio.bar(plot_data["method"], plot_data["mean_ratio"], color=colors)
    ax_ratio.errorbar(
        plot_data["method"],
        plot_data["mean_ratio"],
        yerr=plot_data["std_ratio"],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
    )
    ax_ratio.set_ylabel("Held-out mean approximation ratio")
    ax_ratio.set_title("Initializer ablations and stronger baselines")
    ax_ratio.set_ylim(0.60, 0.91)
    ax_ratio.grid(axis="y", alpha=0.22)
    ax_ratio.tick_params(axis="x", rotation=28)

    ax_runtime.bar(plot_data["method"], plot_data["speedup_vs_classical"], color=colors)
    ax_runtime.set_yscale("log")
    ax_runtime.set_ylabel("Speedup vs. classical search")
    ax_runtime.set_title("Online speedup relative to direct search")
    ax_runtime.grid(axis="y", which="both", alpha=0.22)
    ax_runtime.tick_params(axis="x", rotation=28)

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

    fig, (ax_ratio, ax_runtime) = plt.subplots(1, 2, figsize=(10.9, 4.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.10, wspace=0.04, hspace=0.04)
    for method in method_order:
        method_df = data[data["method"] == method].sort_values("top_gene_count")
        ax_ratio.plot(
            method_df["top_gene_count"],
            method_df["mean_ratio"],
            marker=METHOD_MARKERS[method],
            linewidth=2.4,
            markersize=6,
            color=METHOD_PALETTE[method],
            linestyle=METHOD_LINESTYLES[method],
            label=display_method_label(method),
        )
        ax_ratio.fill_between(
            method_df["top_gene_count"],
            method_df["mean_ratio"] - method_df["std_ratio"],
            method_df["mean_ratio"] + method_df["std_ratio"],
            color=METHOD_PALETTE[method],
            alpha=0.09,
            linewidth=0,
        )

        ax_runtime.plot(
            method_df["top_gene_count"],
            method_df["median_runtime_ms"],
            marker=METHOD_MARKERS[method],
            linewidth=2.4,
            markersize=6,
            color=METHOD_PALETTE[method],
            linestyle=METHOD_LINESTYLES[method],
            label=display_method_label(method),
        )

    ax_ratio.set_xlabel("Top-gene panel size")
    ax_ratio.set_ylabel("Held-out mean approximation ratio")
    ax_ratio.set_title("Real transcriptomic size sweep", pad=18)
    ax_ratio.grid(axis="y", alpha=0.22)
    ax_ratio.set_ylim(0.72, 0.96)

    ax_runtime.set_xlabel("Top-gene panel size")
    ax_runtime.set_ylabel("Median runtime per held-out graph (ms)")
    ax_runtime.set_title("Online cost across graph size", pad=18)
    ax_runtime.set_yscale("log")
    ax_runtime.grid(axis="y", which="both", alpha=0.22)

    handles, labels = ax_ratio.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3)

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
            method_df["mean_ratio"],
            marker=markers[method],
            linewidth=2.0,
            markersize=6,
            color=palette[method],
            label=method,
        )
    ax.set_xlabel("Target-family top-gene count")
    ax.set_ylabel("Held-out mean approximation ratio")
    ax.set_title("Cross-family transfer from a 16-gene source family")
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
    ax_ratio.bar(
        plot_data["display_method"],
        plot_data["mean_ratio"],
        color=[colors.get(method, "#1f1f1f") for method in plot_data["method"]],
    )
    ax_ratio.errorbar(
        plot_data["display_method"],
        plot_data["mean_ratio"],
        yerr=plot_data["std_ratio"],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
    )
    ax_ratio.axhline(
        float(summary.loc[summary["method"] == "Classical depth-2 search", "mean_ratio"].iloc[0]),
        color="#1b4965",
        linestyle="--",
        linewidth=1.6,
        alpha=0.6,
    )
    ax_ratio.set_ylabel("Held-out mean approximation ratio")
    ax_ratio.set_title("Transfer to morphology graphs")
    ax_ratio.set_ylim(0.52, 0.89)
    ax_ratio.grid(axis="y", alpha=0.22)
    ax_ratio.tick_params(axis="x", rotation=24)

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