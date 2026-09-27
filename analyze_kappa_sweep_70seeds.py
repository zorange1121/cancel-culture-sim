"""Analyze the combined 70-seed topology x kappa baseline."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze_paired_topology_kappa import (
    contrast_interaction_tests,
    difference_in_differences,
    kappa_effects,
    repeated_measures_anova,
    seed_vs_topology_variation,
    topology_differences,
)
from diagnose_seed_topology_mechanisms import bootstrap_topology_winners
from network_generation import SUPPORTED_TOPOLOGIES


INPUT_DIR = Path("outputs") / "step1_kappa_sweep_70seeds"
INPUT_PATH = INPUT_DIR / "combined_raw_run_results_70seeds.csv"
OLD_INPUT_PATH = (
    Path("outputs")
    / "step1_kappa_sweep_20seeds_corrected_parsing"
    / "raw_run_results.csv"
)
OUTPUT_DIR = INPUT_DIR / "analysis"
COLORS = {"ER": "#1f77b4", "WS": "#ff7f0e", "BA": "#2ca02c", "SBM": "#d62728"}


def condition_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (topology, kappa), group in data.groupby(["topology", "kappa"], sort=False):
        values = group.final_activated_proportion.to_numpy(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        margin = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
        rows.append(
            {
                "topology": topology,
                "kappa": kappa,
                "n_runs": len(values),
                "mean_final_cascade_size": mean,
                "median_final_cascade_size": float(np.median(values)),
                "std_final_cascade_size": sd,
                "q25_final_cascade_size": float(np.quantile(values, 0.25)),
                "q75_final_cascade_size": float(np.quantile(values, 0.75)),
                "min_final_cascade_size": float(values.min()),
                "max_final_cascade_size": float(values.max()),
                "ci95_lower": max(0.0, mean - margin),
                "ci95_upper": min(1.0, mean + margin),
                "ci95_half_width": margin,
                "ci_method": "two-sided t interval",
            }
        )
    return pd.DataFrame(rows)


def plot_mean_curves(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for topology in SUPPORTED_TOPOLOGIES:
        group = summary[summary.topology == topology].sort_values("kappa")
        axis.plot(
            group.kappa,
            group.mean_final_cascade_size,
            "o-",
            lw=2.3,
            color=COLORS[topology],
            label=topology,
        )
        axis.fill_between(
            group.kappa,
            group.ci95_lower,
            group.ci95_upper,
            color=COLORS[topology],
            alpha=0.16,
        )
    axis.set_xlabel("Kappa")
    axis.set_ylabel("Mean final activated proportion")
    axis.set_xticks(sorted(summary.kappa.unique()))
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(title="Topology")
    axis.set_title("Topology × Kappa Baseline (70 paired seeds, 95% t CI)", weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_distributions(data: pd.DataFrame, path: Path) -> None:
    kappas = sorted(data.kappa.unique())
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        subset = data[data.topology == topology]
        values = [
            subset[subset.kappa == kappa].final_activated_proportion.to_numpy()
            for kappa in kappas
        ]
        boxes = axis.boxplot(values, positions=range(len(kappas)), widths=0.65, patch_artist=True)
        for box in boxes["boxes"]:
            box.set_facecolor(COLORS[topology])
            box.set_alpha(0.5)
        axis.set_title(topology, weight="bold")
        axis.set_xticks(range(len(kappas)), [f"{value:.1f}" for value in kappas])
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", alpha=0.25)
    figure.supxlabel("Kappa")
    figure.supylabel("Final activated proportion")
    figure.suptitle("Final Cascade Distributions (70 paired seeds)", fontsize=16, weight="bold")
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_winners(probabilities: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for topology in SUPPORTED_TOPOLOGIES:
        group = probabilities[probabilities.topology == topology].sort_values("kappa")
        axis.plot(
            group.kappa,
            group.bootstrap_probability_of_highest_mean,
            "o-",
            lw=2.2,
            color=COLORS[topology],
            label=topology,
        )
    axis.set_xlabel("Kappa")
    axis.set_ylabel("Bootstrap probability of highest mean")
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(title="Topology")
    axis.set_title("Leading Topology Uncertainty (70 paired seeds)", weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_20_and_70(old: pd.DataFrame, new_summary: pd.DataFrame) -> pd.DataFrame:
    old_summary = condition_summary(old).rename(
        columns={
            "mean_final_cascade_size": "mean_20",
            "ci95_half_width": "ci95_half_width_20",
        }
    )
    comparison = new_summary.merge(
        old_summary[["topology", "kappa", "mean_20", "ci95_half_width_20"]],
        on=["topology", "kappa"],
    )
    comparison = comparison.rename(
        columns={
            "mean_final_cascade_size": "mean_70",
            "ci95_half_width": "ci95_half_width_70",
        }
    )
    comparison["mean_change_70_minus_20"] = comparison.mean_70 - comparison.mean_20
    comparison["ci_half_width_ratio_70_to_20"] = (
        comparison.ci95_half_width_70 / comparison.ci95_half_width_20
    )
    return comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    old = pd.read_csv(OLD_INPUT_PATH)
    if len(data) != 4 * 11 * 70:
        raise AssertionError("Expected 3,080 combined runs")
    if set(data.groupby(["topology", "kappa"]).size()) != {70}:
        raise AssertionError("Every topology/kappa condition must contain 70 seeds")

    summary = condition_summary(data)
    topology_raw, topology_summary = topology_differences(data)
    kappa_raw, kappa_summary = kappa_effects(data)
    variation = seed_vs_topology_variation(data)
    anova = repeated_measures_anova(data)
    contrast_tests = contrast_interaction_tests(topology_raw)
    did = difference_in_differences(topology_raw)
    winners, single_seed_winners = bootstrap_topology_winners(data, replicates=50_000)
    comparison = compare_20_and_70(old, summary)

    summary.to_csv(OUTPUT_DIR / "condition_summary_70seeds.csv", index=False)
    topology_raw.to_csv(OUTPUT_DIR / "paired_topology_differences_raw_70seeds.csv", index=False)
    topology_summary.to_csv(OUTPUT_DIR / "paired_topology_differences_summary_70seeds.csv", index=False)
    kappa_raw.to_csv(OUTPUT_DIR / "paired_kappa_effects_raw_70seeds.csv", index=False)
    kappa_summary.to_csv(OUTPUT_DIR / "paired_kappa_effects_summary_70seeds.csv", index=False)
    variation.to_csv(OUTPUT_DIR / "seed_vs_topology_variation_70seeds.csv", index=False)
    anova.to_csv(OUTPUT_DIR / "repeated_measures_anova_70seeds.csv", index=False)
    contrast_tests.to_csv(OUTPUT_DIR / "contrast_interaction_tests_70seeds.csv", index=False)
    did.to_csv(OUTPUT_DIR / "difference_in_differences_70seeds.csv", index=False)
    winners.to_csv(OUTPUT_DIR / "bootstrap_winner_probabilities_70seeds.csv", index=False)
    single_seed_winners.to_csv(OUTPUT_DIR / "single_seed_winners_70seeds.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "comparison_20_vs_70_seeds.csv", index=False)

    plot_mean_curves(summary, OUTPUT_DIR / "figure_1_mean_final_size_70seeds.png")
    plot_distributions(data, OUTPUT_DIR / "figure_2_distributions_70seeds.png")
    plot_winners(winners, OUTPUT_DIR / "figure_3_winner_probabilities_70seeds.png")

    print("Repeated-measures ANOVA")
    print(anova.to_string(index=False))
    print("\nMean final cascade size")
    print(
        summary.pivot(
            index="kappa", columns="topology", values="mean_final_cascade_size"
        ).round(4).to_string()
    )
    print("\nBootstrap winner probabilities")
    print(
        winners.pivot(
            index="kappa",
            columns="topology",
            values="bootstrap_probability_of_highest_mean",
        ).round(3).to_string()
    )
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
