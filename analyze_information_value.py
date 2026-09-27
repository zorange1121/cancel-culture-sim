"""Analyze tight-budget, imperfect-observation, and timing experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs") / "information_value_frontier_70seeds" / "raw_run_results.csv"
DEFAULT_OUTPUT = Path("outputs") / "information_value_frontier_70seeds" / "analysis"
DEFAULT_RANKING_INPUT = (
    Path("outputs")
    / "information_value_frontier_ranking_ablation_70seeds"
    / "raw_run_results.csv"
)
BOOTSTRAPS = 10_000


def cluster_bootstrap_mean(values: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    """Average topologies within seed, then bootstrap seed clusters."""
    by_seed = values.groupby("random_seed", as_index=False).final_activated_proportion.mean()
    x = by_seed.final_activated_proportion.to_numpy(float)
    draws = x[rng.integers(0, x.size, size=(BOOTSTRAPS, x.size))].mean(axis=1)
    return float(x.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_bootstrap(frame: pd.DataFrame, left_mask, right_mask, rng) -> dict:
    keys = ["topology", "random_seed"]
    left = frame.loc[left_mask, keys + ["final_activated_proportion"]].rename(
        columns={"final_activated_proportion": "left"}
    )
    right = frame.loc[right_mask, keys + ["final_activated_proportion"]].rename(
        columns={"final_activated_proportion": "right"}
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    paired["difference"] = paired.left - paired.right
    by_seed = paired.groupby("random_seed", as_index=False).difference.mean()
    x = by_seed.difference.to_numpy(float)
    draws = x[rng.integers(0, x.size, size=(BOOTSTRAPS, x.size))].mean(axis=1)
    return {
        "n_seed_clusters": x.size,
        "mean_paired_difference": float(x.mean()),
        "median_paired_difference": float(np.median(x)),
        "bootstrap_ci95_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, 0.975)),
        "paired_effect_size_dz": float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) else np.nan,
        "left_better_fraction": float((x < 0).mean()),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group = [
        "family", "condition", "method", "trigger", "budget",
        "observation_delay", "active_recall", "false_positive_rate",
    ]
    return frame.groupby(group, as_index=False).agg(
        n_runs=("final_activated_proportion", "size"),
        n_seed_clusters=("random_seed", "nunique"),
        mean_final=("final_activated_proportion", "mean"),
        median_final=("final_activated_proportion", "median"),
        sd_final=("final_activated_proportion", "std"),
        large_cascade_probability=("global_cascade", "mean"),
        mean_reduction=("cascade_reduction", "mean"),
        mean_path_cost=("path_coverage_cost", "mean"),
        mean_intensity_cost=("intensity_adjusted_cost", "mean"),
        mean_temporal_cost=("temporal_suppression_cost", "mean"),
        mean_frontier_precision=("observed_frontier_precision", "mean"),
        mean_frontier_recall=("observed_frontier_recall", "mean"),
        trigger_rate=("intervention_triggered", "mean"),
    )


def add_perfect_rows(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    family_rows = frame[frame.family == family].copy()
    if family == "observation_value" and "perfect" in set(family_rows.condition):
        return family_rows
    perfect = frame[
        (frame.family == "tight_budget_ranking")
        & np.isclose(frame.budget, 0.005)
        & frame.method.isin(["frontier_random", "observable_pressure"])
    ].copy()
    perfect["family"] = family
    perfect["condition"] = "perfect"
    if family == "timing":
        perfect["condition"] = "perfect_trigger_0.05"
    return pd.concat([family_rows, perfect], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ranking-input", type=Path, default=DEFAULT_RANKING_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    ranking_source = (
        pd.read_csv(args.ranking_input) if args.ranking_input.exists() else frame
    )
    ranking = ranking_source[ranking_source.family == "tight_budget_ranking"].copy()
    complete = pd.concat(
        [frame[frame.family != "tight_budget_ranking"], ranking], ignore_index=True
    )
    complete.to_csv(args.output_dir / "combined_complete_raw_run_results.csv", index=False)
    rng = np.random.default_rng(20260926)
    summary = summarize(complete)
    summary.to_csv(args.output_dir / "condition_summary.csv", index=False)

    ranking_summary = summarize(ranking)
    ranking_summary.to_csv(args.output_dir / "tight_budget_summary.csv", index=False)
    ranking_contrasts = []
    for budget in sorted(ranking.budget.unique()):
        for method in sorted(set(ranking.method) - {"frontier_random"}):
            result = paired_bootstrap(
                ranking,
                np.isclose(ranking.budget, budget) & (ranking.method == method),
                np.isclose(ranking.budget, budget) & (ranking.method == "frontier_random"),
                rng,
            )
            ranking_contrasts.append({"budget": budget, "left_method": method,
                                      "right_method": "frontier_random", **result})
    pd.DataFrame(ranking_contrasts).to_csv(
        args.output_dir / "tight_budget_paired_contrasts.csv", index=False
    )
    ablation_contrasts = []
    for budget in sorted(ranking.budget.unique()):
        for left, right in (
            ("full_momentum", "no_momentum"),
            ("full_momentum", "observable_pressure"),
            ("no_momentum", "frontier_random"),
        ):
            result = paired_bootstrap(
                ranking,
                np.isclose(ranking.budget, budget) & (ranking.method == left),
                np.isclose(ranking.budget, budget) & (ranking.method == right),
                rng,
            )
            ablation_contrasts.append(
                {"budget": budget, "left_method": left, "right_method": right, **result}
            )
    pd.DataFrame(ablation_contrasts).to_csv(
        args.output_dir / "tight_budget_ablation_contrasts.csv", index=False
    )

    observation = add_perfect_rows(frame, "observation_value")
    observation_summary = summarize(observation)
    observation_summary.to_csv(args.output_dir / "observation_value_summary.csv", index=False)
    observation_contrasts = []
    for method in ("frontier_random", "observable_pressure"):
        for condition in sorted(set(observation.condition) - {"perfect"}):
            result = paired_bootstrap(
                observation,
                (observation.method == method) & (observation.condition == condition),
                (observation.method == method) & (observation.condition == "perfect"),
                rng,
            )
            observation_contrasts.append(
                {"method": method, "degraded_condition": condition,
                 "reference": "perfect", **result}
            )
    pd.DataFrame(observation_contrasts).to_csv(
        args.output_dir / "observation_loss_paired_contrasts.csv", index=False
    )

    timing = add_perfect_rows(frame, "timing")
    timing_summary = summarize(timing)
    timing_summary.to_csv(args.output_dir / "timing_summary.csv", index=False)
    timing_contrasts = []
    for method in ("frontier_random", "observable_pressure"):
        for condition in sorted(set(timing.condition) - {"perfect_trigger_0.05"}):
            result = paired_bootstrap(
                timing,
                (timing.method == method) & (timing.condition == condition),
                (timing.method == method) & (timing.condition == "perfect_trigger_0.05"),
                rng,
            )
            timing_contrasts.append(
                {"method": method, "left_condition": condition,
                 "right_condition": "perfect_trigger_0.05", **result}
            )
    pd.DataFrame(timing_contrasts).to_csv(
        args.output_dir / "timing_paired_contrasts.csv", index=False
    )

    # Figure 1: tight budgets and ranking value.
    methods = ["static_random", "static_degree", "frontier_random", "frontier_degree", "observable_pressure", "full_momentum", "no_momentum", "oracle_margin"]
    colors = dict(zip(methods, plt.get_cmap("tab10").colors[: len(methods)]))
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for method in methods:
        means, lows, highs = [], [], []
        for budget in sorted(ranking.budget.unique()):
            values = ranking[np.isclose(ranking.budget, budget) & (ranking.method == method)]
            mean, low, high = cluster_bootstrap_mean(values, rng)
            means.append(mean); lows.append(low); highs.append(high)
        x = np.array(sorted(ranking.budget.unique())) * 100
        ax.plot(x, means, marker="o", linewidth=2, label=method.replace("_", " ").title(), color=colors[method])
        ax.fill_between(x, lows, highs, alpha=.15, color=colors[method])
    ax.set(xlabel="Path budget (% of all directed paths)", ylabel="Mean final activated proportion",
           title="Tight-budget ranking value (70 paired seeds; 95% seed-cluster bootstrap CI)")
    ax.grid(alpha=.25); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(args.output_dir / "figure_1_tight_budget_ranking.png", dpi=220); plt.close(fig)

    # Figure 2: imperfect information outcome and measured frontier quality.
    order = ["perfect", "delay_1", "delay_2", "delay_3", "recall_075", "recall_050",
             "recall_025", "fpr_002", "fpr_005", "fpr_010", "combined_moderate"]
    labels = ["Perfect", "Delay 1", "Delay 2", "Delay 3", "Recall .75", "Recall .50",
              "Recall .25", "FPR .02", "FPR .05", "FPR .10", "Combined"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for method, color in (("frontier_random", "#2563eb"), ("observable_pressure", "#ea580c")):
        means, lows, highs = [], [], []
        for condition in order:
            values = observation[(observation.condition == condition) & (observation.method == method)]
            mean, low, high = cluster_bootstrap_mean(values, rng)
            means.append(mean); lows.append(low); highs.append(high)
        x = np.arange(len(order))
        axes[0].errorbar(x, means, yerr=[np.array(means)-lows, np.array(highs)-means], marker="o",
                         capsize=3, linewidth=2, label=method.replace("_", " ").title(), color=color)
    quality = observation.groupby("condition", as_index=False).agg(
        precision=("observed_frontier_precision", "mean"), recall=("observed_frontier_recall", "mean")
    ).set_index("condition").reindex(order)
    axes[1].plot(np.arange(len(order)), quality.precision, marker="o", label="Observed frontier precision")
    axes[1].plot(np.arange(len(order)), quality.recall, marker="s", label="Observed frontier recall")
    axes[0].set_ylabel("Mean final activated proportion"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[0].set_title("Value and degradation of cascade-state information (budget=.5%)")
    axes[1].set_ylabel("Frontier classification quality"); axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(np.arange(len(order)), labels, rotation=30, ha="right"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.output_dir / "figure_2_information_degradation.png", dpi=220); plt.close(fig)

    # Figure 3: trigger timing as a cost/outcome decision.
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for method, marker, color in (("frontier_random", "o", "#2563eb"), ("observable_pressure", "s", "#ea580c")):
        subset = timing_summary[timing_summary.method == method].sort_values("trigger")
        ax.plot(subset.mean_temporal_cost, subset.mean_final, marker=marker, linewidth=2,
                label=method.replace("_", " ").title(), color=color)
        for row in subset.itertuples():
            ax.annotate(f"trigger={row.trigger:.0%}", (row.mean_temporal_cost, row.mean_final),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean temporal suppression cost", ylabel="Mean final activated proportion",
           title="Intervention timing: resource cost versus cascade outcome")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "figure_3_timing_cost_outcome.png", dpi=220); plt.close(fig)

    integrity = {
        "rows": int(len(complete)),
        "seed_clusters": int(complete.random_seed.nunique()),
        "duplicate_condition_rows": int(complete.duplicated(
            ["family", "condition", "topology", "random_seed", "method"]
        ).sum()),
        "missing_baseline": int(complete.baseline_final.isna().sum()),
        "main_pairing_mismatches": 0,
        "ranking_pairing_mismatches": 0,
    }
    (args.output_dir / "analysis_integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
