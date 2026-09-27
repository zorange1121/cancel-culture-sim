"""Compare intervention methods across ER, WS, BA, and SBM using existing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM


OLD_RESULTS = Path("outputs/information_value_frontier_final_70seeds/analysis/combined_complete_raw_run_results.csv")
NESTED_RESULTS = Path("outputs/nested_momentum_70seeds/raw_run_results.csv")
OUTPUT_DIR = Path("outputs/topology_method_interaction_70seeds")
BOOTSTRAPS = 10_000
RNG_SEED = 20260927

METHOD_ORDER = [
    "Static Random",
    "Static Degree",
    "Frontier Random",
    "Frontier Degree",
    "Observable Pressure",
    "Pressure + Momentum (λ=.35)",
    "Oracle Margin",
]
DEPLOYABLE = [method for method in METHOD_ORDER if method != "Oracle Margin"]
TOPOLOGY_ORDER = ["ER", "WS", "BA", "SBM"]


def load_combined(old_path: Path, nested_path: Path) -> pd.DataFrame:
    old = pd.read_csv(old_path)
    old = old[
        (old.family == "tight_budget_ranking")
        & old.method.isin(
            ["static_random", "static_degree", "frontier_random", "frontier_degree", "oracle_margin"]
        )
    ].copy()
    old["method_label"] = old.method.map(
        {
            "static_random": "Static Random",
            "static_degree": "Static Degree",
            "frontier_random": "Frontier Random",
            "frontier_degree": "Frontier Degree",
            "oracle_margin": "Oracle Margin",
        }
    )

    nested = pd.read_csv(nested_path)
    nested = nested[nested.momentum_lambda.isin([0.0, 0.35])].copy()
    nested["method_label"] = np.where(
        np.isclose(nested.momentum_lambda, 0.0),
        "Observable Pressure",
        "Pressure + Momentum (λ=.35)",
    )
    columns = [
        "topology", "random_seed", "budget", "method_label",
        "final_activated_proportion", "cascade_reduction", "cascade_duration", "t50",
        "selected_path_count", "path_coverage_cost", "baseline_final",
    ]
    combined = pd.concat([old[columns], nested[columns]], ignore_index=True)
    combined["topology"] = pd.Categorical(combined.topology, TOPOLOGY_ORDER, ordered=True)
    combined["method_label"] = pd.Categorical(combined.method_label, METHOD_ORDER, ordered=True)
    return combined.sort_values(["budget", "topology", "method_label", "random_seed"])


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, values.size, size=(BOOTSTRAPS, values.size))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for (budget, topology, method), group in frame.groupby(
        ["budget", "topology", "method_label"], observed=True, sort=True
    ):
        values = group.final_activated_proportion.to_numpy(float)
        mean, low, high = bootstrap_ci(values, rng)
        rows.append(
            {
                "budget": budget,
                "topology": str(topology),
                "method": str(method),
                "n_seeds": values.size,
                "mean_final": mean,
                "median_final": float(np.median(values)),
                "std_final": float(np.std(values, ddof=1)),
                "ci95_lower": low,
                "ci95_upper": high,
                "mean_reduction_vs_no_intervention": float(group.cascade_reduction.mean()),
                "mean_path_coverage": float(group.path_coverage_cost.mean()),
            }
        )
    return pd.DataFrame(rows)


def repeated_measures(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = frame.copy()
    work["topology"] = work.topology.astype(str)
    work["method_label"] = work.method_label.astype(str)
    for budget, group in work.groupby("budget", sort=True):
        result = AnovaRM(
            group,
            depvar="final_activated_proportion",
            subject="random_seed",
            within=["topology", "method_label"],
        ).fit().anova_table.reset_index().rename(columns={"index": "effect"})
        for _, item in result.iterrows():
            f_value = float(item["F Value"])
            num_df = float(item["Num DF"])
            den_df = float(item["Den DF"])
            rows.append(
                {
                    "budget": budget,
                    "effect": item.effect,
                    "f_value": f_value,
                    "num_df": num_df,
                    "den_df": den_df,
                    "p_value": float(item["Pr > F"]),
                    "partial_eta_squared": (f_value * num_df) / (f_value * num_df + den_df),
                }
            )
    return pd.DataFrame(rows)


def targeted_contrasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 1)
    definitions = [
        ("Frontier information", "Frontier Random", "Static Degree"),
        ("Pressure scoring", "Observable Pressure", "Frontier Random"),
        ("Momentum addition", "Pressure + Momentum (λ=.35)", "Observable Pressure"),
        ("Oracle gap", "Oracle Margin", "Pressure + Momentum (λ=.35)"),
    ]
    rows = []
    difference_rows = []
    for budget, budget_group in frame.groupby("budget", sort=True):
        for label, left, right in definitions:
            for topology, group in budget_group.groupby("topology", observed=True, sort=True):
                wide = group.pivot(index="random_seed", columns="method_label", values="final_activated_proportion")
                values = (wide[left] - wide[right]).to_numpy(float)
                mean, low, high = bootstrap_ci(values, rng)
                rows.append(
                    {
                        "budget": budget,
                        "contrast": label,
                        "topology": str(topology),
                        "left_method": left,
                        "right_method": right,
                        "mean_paired_difference": mean,
                        "ci95_lower": low,
                        "ci95_upper": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                    }
                )
                for seed, value in zip(wide.index, values):
                    difference_rows.append(
                        {
                            "budget": budget,
                            "contrast": label,
                            "topology": str(topology),
                            "random_seed": seed,
                            "difference": value,
                        }
                    )

    interaction_rows = []
    differences = pd.DataFrame(difference_rows)
    for (budget, contrast), group in differences.groupby(["budget", "contrast"], sort=True):
        table = AnovaRM(group, depvar="difference", subject="random_seed", within=["topology"]).fit().anova_table.iloc[0]
        f_value = float(table["F Value"])
        num_df = float(table["Num DF"])
        den_df = float(table["Den DF"])
        interaction_rows.append(
            {
                "budget": budget,
                "contrast": contrast,
                "f_value": f_value,
                "num_df": num_df,
                "den_df": den_df,
                "p_value": float(table["Pr > F"]),
                "partial_eta_squared": (f_value * num_df) / (f_value * num_df + den_df),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(interaction_rows)


def exploratory_winners(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 2)
    rows = []
    deployable = frame[frame.method_label.isin(DEPLOYABLE)].copy()
    for (budget, topology), group in deployable.groupby(["budget", "topology"], observed=True, sort=True):
        means = group.groupby("method_label", observed=True).final_activated_proportion.mean().sort_values()
        best, runner_up = means.index[:2]
        wide = group.pivot(index="random_seed", columns="method_label", values="final_activated_proportion")
        values = (wide[best] - wide[runner_up]).to_numpy(float)
        mean, low, high = bootstrap_ci(values, rng)
        rows.append(
            {
                "budget": budget,
                "topology": str(topology),
                "best_by_mean": str(best),
                "best_mean_final": float(means.iloc[0]),
                "runner_up": str(runner_up),
                "runner_up_mean_final": float(means.iloc[1]),
                "best_minus_runner_up": mean,
                "ci95_lower": low,
                "ci95_upper": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "note": "Exploratory: winner selected from the same data",
            }
        )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, contrasts: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    budgets = sorted(summary.budget.unique())
    colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    x = np.arange(len(TOPOLOGY_ORDER))
    for ax, budget in zip(axes.flat, budgets):
        subset = summary[np.isclose(summary.budget, budget)]
        for index, method in enumerate(METHOD_ORDER):
            group = subset[subset.method == method].set_index("topology").loc[TOPOLOGY_ORDER]
            ax.plot(x, group.mean_final, marker="o", label=method, color=colors[index])
        ax.set_xticks(x, TOPOLOGY_ORDER)
        ax.set_title(f"Budget {100*budget:g}%")
        ax.set_ylabel("Mean final activated proportion")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Intervention methods across network topologies", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "figure_1_topology_method_means.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    contrast_order = ["Frontier information", "Pressure scoring", "Momentum addition", "Oracle gap"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, budget in zip(axes.flat, budgets):
        subset = contrasts[np.isclose(contrasts.budget, budget)]
        width = 0.19
        x = np.arange(len(TOPOLOGY_ORDER))
        for index, contrast in enumerate(contrast_order):
            group = subset[subset.contrast == contrast].set_index("topology").loc[TOPOLOGY_ORDER]
            position = x + (index - 1.5) * width
            y = group.mean_paired_difference.to_numpy(float)
            lower = y - group.ci95_lower.to_numpy(float)
            upper = group.ci95_upper.to_numpy(float) - y
            ax.bar(position, y, width, label=contrast, color=colors[index], alpha=0.82)
            ax.errorbar(position, y, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=2, linewidth=0.7)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x, TOPOLOGY_ORDER)
        ax.set_title(f"Budget {100*budget:g}%")
        ax.set_ylabel("Paired difference (left − right)\nnegative favors the added information")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Where does each intervention improvement come from?", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "figure_2_targeted_topology_contrasts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-results", type=Path, default=OLD_RESULTS)
    parser.add_argument("--nested-results", type=Path, default=NESTED_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_combined(args.old_results, args.nested_results)
    summary = summarize(frame)
    anova = repeated_measures(frame)
    contrasts, contrast_interactions = targeted_contrasts(frame)
    winners = exploratory_winners(frame)

    frame.to_csv(args.output_dir / "combined_existing_run_results.csv", index=False)
    summary.to_csv(args.output_dir / "topology_method_summary.csv", index=False)
    anova.to_csv(args.output_dir / "topology_method_repeated_measures.csv", index=False)
    contrasts.to_csv(args.output_dir / "targeted_topology_contrasts.csv", index=False)
    contrast_interactions.to_csv(args.output_dir / "targeted_contrast_interactions.csv", index=False)
    winners.to_csv(args.output_dir / "exploratory_method_winners.csv", index=False)
    make_figures(summary, contrasts, args.output_dir)

    integrity = {
        "rows": int(len(frame)),
        "expected_rows": 4 * 70 * 4 * len(METHOD_ORDER),
        "unique_runs": int(frame[["budget", "topology", "random_seed", "method_label"]].drop_duplicates().shape[0]),
        "seeds": int(frame.random_seed.nunique()),
        "topologies": TOPOLOGY_ORDER,
        "methods": METHOD_ORDER,
        "budgets": sorted(map(float, frame.budget.unique())),
        "missing_outcomes": int(frame.final_activated_proportion.isna().sum()),
    }
    (args.output_dir / "analysis_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
