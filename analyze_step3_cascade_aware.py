"""Paired analysis for STEP 3 cascade-aware interventions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_DIR = Path("outputs") / "step3_cascade_aware_70seeds"
DEFAULT_STEP2 = Path("outputs") / "step2_intervention_70seeds"


def paired_stats(values: pd.Series) -> dict[str, float | int]:
    x = values.dropna().to_numpy(float)
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {
        "n": n,
        "mean_difference": mean,
        "median_difference": float(np.median(x)),
        "sd_difference": sd,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
        "positive_fraction": float((x > 0).mean()),
        "zero_fraction": float((x == 0).mean()),
        "two_sided_t_p": float(stats.ttest_1samp(x, 0).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0,
    }


def contrasts(comparison: pd.DataFrame) -> pd.DataFrame:
    requested = [
        ("observable_pressure", "degree"),
        ("observable_momentum", "degree"),
        ("oracle_margin", "degree"),
        ("observable_momentum", "observable_pressure"),
        ("oracle_margin", "observable_momentum"),
    ]
    rows = []
    keys = ["topology", "kappa", "intervention_trigger", "intervention_budget"]
    for key, group in comparison[comparison.intervention_strategy != "none"].groupby(keys):
        wide = group.pivot(index="random_seed", columns="intervention_strategy", values="cascade_reduction")
        for left, right in requested:
            if left in wide and right in wide:
                rows.append(
                    {
                        **dict(zip(keys, key)),
                        "contrast": f"{left} - {right}",
                        **paired_stats(wide[left] - wide[right]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--step2-dir", type=Path, default=DEFAULT_STEP2)
    args = parser.parse_args()

    dynamic_path = args.input_dir / "raw_cascade_aware_run_results.csv"
    dynamic = pd.read_csv(dynamic_path)
    step2 = pd.read_csv(args.step2_dir / "raw_run_results.csv")
    path_counts = step2[
        (step2.intervention_strategy == "random")
        & (step2.kappa == 0.5)
        & (step2.intervention_trigger == 0.05)
        & (step2.intervention_budget == 0.05)
    ][["topology", "random_seed", "available_path_count"]].rename(
        columns={"available_path_count": "total_path_count"}
    )
    dynamic = dynamic.drop(columns=["total_path_count", "actual_budget_fraction"], errors="ignore")
    dynamic = dynamic.merge(path_counts, on=["topology", "random_seed"], validate="many_to_one")
    dynamic["actual_budget_fraction"] = dynamic.selected_path_count / dynamic.total_path_count
    dynamic["requested_budget_usage"] = dynamic.actual_budget_fraction / dynamic.intervention_budget
    dynamic["actual_intervention_efficiency"] = np.where(
        dynamic.actual_budget_fraction > 0,
        dynamic.cascade_reduction / dynamic.actual_budget_fraction,
        np.nan,
    )
    dynamic.to_csv(dynamic_path, index=False)

    utilization = (
        dynamic[dynamic.intervention_triggered]
        .groupby(["intervention_trigger", "intervention_budget", "intervention_strategy"])
        .agg(
            n=("random_seed", "size"),
            mean_actual_budget=("actual_budget_fraction", "mean"),
            min_actual_budget=("actual_budget_fraction", "min"),
            max_actual_budget=("actual_budget_fraction", "max"),
            mean_requested_budget_usage=("requested_budget_usage", "mean"),
            min_requested_budget_usage=("requested_budget_usage", "min"),
        )
        .reset_index()
    )
    utilization.to_csv(args.input_dir / "budget_utilization.csv", index=False)

    comparison = pd.read_csv(args.input_dir / "combined_static_dynamic_run_results.csv")
    contrast_frame = contrasts(comparison)
    contrast_frame.to_csv(args.input_dir / "paired_strategy_contrasts.csv", index=False)

    focus = contrast_frame[
        (contrast_frame.kappa == 0.5)
        & (contrast_frame.intervention_trigger == 0.05)
        & (contrast_frame.intervention_budget.isin([0.02, 0.05]))
    ]
    print(focus.round(5).to_string(index=False))
    print("\nBudget utilization")
    print(utilization.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
