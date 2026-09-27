"""Analyze the paired nested Pressure-plus-Momentum experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM


DEFAULT_INPUT = Path("outputs") / "nested_momentum_70seeds" / "raw_run_results.csv"
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260927


def markdown_table(frame: pd.DataFrame, decimals: int = 6) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.{decimals}f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, values.size, size=(BOOTSTRAPS, values.size))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def pooled_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (budget, lam), group in frame.groupby(["budget", "momentum_lambda"], sort=True):
        seed_means = group.groupby("random_seed").final_activated_proportion.mean()
        mean, low, high = bootstrap_mean_ci(seed_means.to_numpy(), rng)
        rows.append(
            {
                "budget": budget,
                "momentum_lambda": lam,
                "n_seed_clusters": seed_means.size,
                "mean_final": mean,
                "median_final": float(group.final_activated_proportion.median()),
                "std_across_seed_means": float(seed_means.std(ddof=1)),
                "ci95_lower": low,
                "ci95_upper": high,
                "mean_cascade_reduction": float(group.cascade_reduction.mean()),
                "mean_duration": float(group.cascade_duration.mean()),
                "mean_t50": float(group.t50.mean()),
                "large_cascade_rate": float(group.global_cascade.mean()),
                "trigger_rate": float(group.intervention_triggered.mean()),
                "mean_selected_paths": float(group.selected_path_count.mean()),
            }
        )
    return pd.DataFrame(rows)


def topology_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows = []
    for (budget, topology, lam), group in frame.groupby(
        ["budget", "topology", "momentum_lambda"], sort=True
    ):
        values = group.sort_values("random_seed").final_activated_proportion.to_numpy()
        mean, low, high = bootstrap_mean_ci(values, rng)
        rows.append(
            {
                "budget": budget,
                "topology": topology,
                "momentum_lambda": lam,
                "n_seeds": values.size,
                "mean_final": mean,
                "median_final": float(np.median(values)),
                "std_final": float(np.std(values, ddof=1)),
                "ci95_lower": low,
                "ci95_upper": high,
                "mean_cascade_reduction": float(group.cascade_reduction.mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_contrasts(frame: pd.DataFrame, by_topology: bool) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + (3 if by_topology else 2))
    rows = []
    group_keys = ["budget", "topology"] if by_topology else ["budget"]
    for keys, group in frame.groupby(group_keys, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        index = ["random_seed", "topology"] if not by_topology else ["random_seed"]
        wide = group.pivot(index=index, columns="momentum_lambda", values="final_activated_proportion")
        for lam in sorted(value for value in frame.momentum_lambda.unique() if value > 0):
            differences = (wide[lam] - wide[0.0])
            if not by_topology:
                differences = differences.groupby(level="random_seed").mean()
            values = differences.to_numpy(float)
            mean, low, high = bootstrap_mean_ci(values, rng)
            row = {
                "budget": keys[0],
                "momentum_lambda": lam,
                "contrast": f"lambda_{lam:g}_minus_lambda_0",
                "n_seed_clusters": values.size,
                "mean_paired_difference": mean,
                "median_paired_difference": float(np.median(values)),
                "std_paired_difference": float(np.std(values, ddof=1)),
                "ci95_lower": low,
                "ci95_upper": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
            if by_topology:
                row["topology"] = keys[1]
            rows.append(row)
    columns = ["budget"] + (["topology"] if by_topology else []) + [
        "momentum_lambda", "contrast", "n_seed_clusters", "mean_paired_difference",
        "median_paired_difference", "std_paired_difference", "ci95_lower",
        "ci95_upper", "ci_excludes_zero",
    ]
    return pd.DataFrame(rows)[columns]


def repeated_measures(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget, group in frame.groupby("budget", sort=True):
        model = AnovaRM(
            group,
            depvar="final_activated_proportion",
            subject="random_seed",
            within=["topology", "momentum_lambda"],
        ).fit()
        table = model.anova_table.reset_index().rename(columns={"index": "effect"})
        for _, result in table.iterrows():
            f_value = float(result["F Value"])
            num_df = float(result["Num DF"])
            den_df = float(result["Den DF"])
            partial_eta_sq = (f_value * num_df) / (f_value * num_df + den_df)
            rows.append(
                {
                    "budget": budget,
                    "effect": result["effect"],
                    "f_value": f_value,
                    "num_df": num_df,
                    "den_df": den_df,
                    "p_value": float(result["Pr > F"]),
                    "partial_eta_squared": partial_eta_sq,
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    pooled: pd.DataFrame,
    topology: pd.DataFrame,
    topology_contrasts: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {0.001: "#2563eb", 0.0025: "#16a34a", 0.005: "#ea580c", 0.01: "#7c3aed"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for budget, group in pooled.groupby("budget", sort=True):
        group = group.sort_values("momentum_lambda")
        x = group.momentum_lambda.to_numpy(float)
        y = group.mean_final.to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=2, label=f"Budget {100*budget:g}%", color=colors[budget])
        ax.fill_between(x, group.ci95_lower, group.ci95_upper, alpha=0.16, color=colors[budget])
    ax.set_xlabel("Momentum weight λ")
    ax.set_ylabel("Final activated proportion")
    ax.set_title("Nested Pressure + Momentum experiment (70 paired seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_lambda_sweep_pooled.png", dpi=200)
    plt.close(fig)

    topologies = ["ER", "WS", "BA", "SBM"]
    budgets = sorted(topology.budget.unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    palette = {"ER": "#2563eb", "WS": "#16a34a", "BA": "#ea580c", "SBM": "#7c3aed"}
    for ax, budget in zip(axes.flat, budgets):
        subset = topology[np.isclose(topology.budget, budget)]
        for network in topologies:
            group = subset[subset.topology == network].sort_values("momentum_lambda")
            ax.plot(group.momentum_lambda, group.mean_final, marker="o", label=network, color=palette[network])
        ax.set_title(f"Budget {100*budget:g}%")
        ax.set_xlabel("λ")
        ax.set_ylabel("Final activated proportion")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Topology-specific nested Momentum response", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_topology_lambda_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, budget in zip(axes.flat, budgets):
        subset = topology_contrasts[np.isclose(topology_contrasts.budget, budget)]
        width = 0.22
        x = np.arange(len(topologies))
        for offset, lam in enumerate((0.15, 0.35, 0.5)):
            group = subset[np.isclose(subset.momentum_lambda, lam)].set_index("topology").loc[topologies]
            position = x + (offset - 1) * width
            y = group.mean_paired_difference.to_numpy(float)
            lower = y - group.ci95_lower.to_numpy(float)
            upper = group.ci95_upper.to_numpy(float) - y
            ax.bar(position, y, width=width, label=f"λ={lam:g}", alpha=0.82)
            ax.errorbar(position, y, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=2, linewidth=0.8)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x, topologies)
        ax.set_title(f"Budget {100*budget:g}%")
        ax.set_ylabel("Paired final difference vs λ=0\n(lower is better)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Does Momentum help differently by topology?", y=0.985)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "figure_3_topology_paired_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    pooled: pd.DataFrame,
    contrasts: pd.DataFrame,
    topology_contrasts: pd.DataFrame,
    anova: pd.DataFrame,
    output_dir: Path,
) -> None:
    integrity = {
        "rows": int(len(frame)),
        "unique_runs": int(frame[["topology", "random_seed", "budget", "momentum_lambda"]].drop_duplicates().shape[0]),
        "seeds": int(frame.random_seed.nunique()),
        "topologies": sorted(frame.topology.unique().tolist()),
        "budgets": sorted(map(float, frame.budget.unique())),
        "lambdas": sorted(map(float, frame.momentum_lambda.unique())),
        "missing_final": int(frame.final_activated_proportion.isna().sum()),
    }
    (output_dir / "analysis_integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )

    lines = [
        "# Nested Pressure + Momentum experiment",
        "",
        "## Design",
        "",
        "The score is `S_lambda = (1-lambda) * ObservablePressure + lambda * newly_active_sender`.",
        "Lambda 0 is exactly Observable Pressure. The experiment uses ER, WS, BA, and SBM; 70 paired seeds; kappa=.5; trigger=5%; selected-path weight=.75; and budgets .1%, .25%, .5%, and 1%.",
        "",
        f"Integrity: {integrity['rows']:,} rows, {integrity['unique_runs']:,} unique runs, {integrity['missing_final']} missing outcomes.",
        "",
        "## Pooled results",
        "",
        markdown_table(pooled[["budget", "momentum_lambda", "mean_final", "ci95_lower", "ci95_upper"]]),
        "",
        "## Paired differences relative to lambda=0",
        "",
        "Negative differences favor adding Momentum.",
        "",
        markdown_table(contrasts),
        "",
        "## Topology-specific paired differences",
        "",
        markdown_table(topology_contrasts),
        "",
        "## Repeated-measures topology x lambda tests",
        "",
        markdown_table(anova),
        "",
        "## Files",
        "",
        "- `raw_run_results.csv` and `raw_timestep_results.csv`: raw results.",
        "- `pooled_summary.csv`: seed-cluster summaries across topologies.",
        "- `pooled_paired_contrasts.csv`: paired lambda-vs-zero effects.",
        "- `topology_summary.csv` and `topology_paired_contrasts.csv`: topology-specific results.",
        "- `repeated_measures_anova.csv`: topology, lambda, and interaction tests by budget.",
        "- `figure_1_lambda_sweep_pooled.png` to `figure_3_topology_paired_effects.png`: figures.",
    ]
    (output_dir / "nested_momentum_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    pooled = pooled_summary(frame)
    topology = topology_summary(frame)
    contrasts = paired_contrasts(frame, by_topology=False)
    topology_contrasts = paired_contrasts(frame, by_topology=True)
    anova = repeated_measures(frame)

    pooled.to_csv(output_dir / "pooled_summary.csv", index=False)
    topology.to_csv(output_dir / "topology_summary.csv", index=False)
    contrasts.to_csv(output_dir / "pooled_paired_contrasts.csv", index=False)
    topology_contrasts.to_csv(output_dir / "topology_paired_contrasts.csv", index=False)
    anova.to_csv(output_dir / "repeated_measures_anova.csv", index=False)
    make_figures(pooled, topology, topology_contrasts, output_dir)
    write_report(frame, pooled, contrasts, topology_contrasts, anova, output_dir)
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
