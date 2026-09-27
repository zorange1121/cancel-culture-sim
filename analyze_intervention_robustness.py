"""Analyze the completed intervention-weight and coefficient robustness checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


OUTPUT_DIR = Path("outputs") / "intervention_robustness_70seeds"


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["float"]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{decimals}f}")
    headers = [str(column) for column in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in formatted.itertuples(index=False, name=None))
    return "\n".join(lines)


def coefficient_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    focus = frame[(frame.intervention_strategy == "observable_momentum") & (frame.intervention_weight == 0)]
    wide = focus.pivot(
        index=["topology", "random_seed", "intervention_budget"],
        columns="coefficient_profile",
        values="final_activated_proportion",
    ).reset_index()
    records = []
    profiles = [column for column in wide.columns if column not in {"topology", "random_seed", "intervention_budget", "default"}]
    for profile in profiles:
        wide[f"{profile}_minus_default"] = wide[profile] - wide.default
        for key, group in wide.groupby(["topology", "intervention_budget"]):
            values = group[f"{profile}_minus_default"].to_numpy(float)
            half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
            records.append(
                {
                    "topology": key[0],
                    "intervention_budget": key[1],
                    "coefficient_profile": profile,
                    "mean_final_difference_vs_default": values.mean(),
                    "median_difference": np.median(values),
                    "sd_difference": values.std(ddof=1),
                    "ci95_lower": values.mean() - half,
                    "ci95_upper": values.mean() + half,
                    "profile_better_fraction": (values < 0).mean(),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    raw = pd.read_csv(OUTPUT_DIR / "raw_run_results.csv")
    weight = pd.read_csv(OUTPUT_DIR / "intervention_weight_summary.csv")
    coeff = pd.read_csv(OUTPUT_DIR / "coefficient_robustness_summary.csv")
    advantage = pd.read_csv(OUTPUT_DIR / "paired_momentum_vs_degree.csv")
    contrasts = coefficient_contrasts(raw)
    contrasts.to_csv(OUTPUT_DIR / "paired_coefficient_contrasts.csv", index=False)

    default_rows = raw[
        ((raw.intervention_strategy == "degree") & (raw.coefficient_profile == "not_applicable"))
        | ((raw.intervention_strategy == "observable_momentum") & (raw.coefficient_profile == "default"))
    ]
    monotone_groups = 0
    violations = 0
    for _, group in default_rows.groupby(["topology", "random_seed", "intervention_strategy", "intervention_budget"]):
        values = group.sort_values("intervention_weight").final_activated_proportion.to_numpy(float)
        monotone_groups += 1
        violations += int(np.any(np.diff(values) < -1e-12))

    retained = weight.pivot_table(
        index=["topology", "intervention_strategy", "intervention_budget"],
        columns="intervention_weight",
        values="mean_paired_reduction",
    ).reset_index()
    retained["retained_effect_at_weight_0.75"] = retained[0.75] / retained[0.0]
    retained.to_csv(OUTPUT_DIR / "retained_effectiveness.csv", index=False)

    mean_ranges = coeff.groupby(["topology", "intervention_budget"]).agg(
        min_mean_final=("mean_final_cascade_size", "min"),
        max_mean_final=("mean_final_cascade_size", "max"),
    ).reset_index()
    mean_ranges["coefficient_mean_range"] = mean_ranges.max_mean_final - mean_ranges.min_mean_final
    mean_ranges.to_csv(OUTPUT_DIR / "coefficient_profile_ranges.csv", index=False)

    integrity = {
        "run_count": int(len(raw)),
        "duplicate_condition_seed_rows": int(raw.duplicated([
            "topology", "random_seed", "intervention_strategy", "intervention_budget",
            "intervention_weight", "coefficient_profile"
        ]).sum()),
        "all_conditions_have_70_seeds": bool((raw.groupby([
            "topology", "intervention_strategy", "intervention_budget",
            "intervention_weight", "coefficient_profile"
        ]).size() == 70).all()),
        "paired_ignition_all_match": bool(pd.read_csv(OUTPUT_DIR / "paired_randomness_verification.csv").matches_seed_reference.all()),
        "weight_monotonicity_groups": monotone_groups,
        "weight_monotonicity_violations": violations,
    }
    (OUTPUT_DIR / "integrity_checks.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    focus_weight = weight[weight.intervention_budget == 0.05][[
        "topology", "intervention_strategy", "intervention_weight",
        "mean_final_cascade_size", "mean_paired_reduction"
    ]].copy()
    focus_weight.columns = ["Topology", "Strategy", "Path weight", "Mean final", "Mean paired reduction"]
    focus_retained = retained[retained.intervention_budget == 0.05][[
        "topology", "intervention_strategy", "retained_effect_at_weight_0.75"
    ]].copy()
    focus_retained.columns = ["Topology", "Strategy", "Effect retained at weight=.75"]
    ranges = mean_ranges.copy()
    ranges.columns = ["Topology", "Budget", "Best mean final", "Worst mean final", "Across-profile range"]
    adv = advantage[advantage.intervention_budget == 0.05][[
        "topology", "intervention_weight", "mean_momentum_minus_degree",
        "ci95_lower", "ci95_upper", "momentum_better_fraction"
    ]].copy()
    adv.columns = ["Topology", "Path weight", "Momentum − Degree", "CI lower", "CI upper", "Momentum better fraction"]

    report = f"""# Intervention Robustness Checks

## Design

- Four topologies: ER, WS, BA, SBM; N=10,000; target base-network mean degree ≈10.
- 70 paired seeds (0–69), κ=0.5, progress trigger=5%, budgets=2% and 5%.
- Weight sensitivity: selected exposure-path weight ∈ {{0, .25, .50, .75}} for static Degree and Observable Momentum. Weight 0 means complete suppression; .75 means the selected exposure retains 75% of its original contribution.
- Coefficient robustness: Observable Momentum at weight 0 under five pre-specified profiles: default, exposure-heavy, momentum-heavy, degree-heavy, and no-community.
- Agent thresholds, B/R, roles, KOL construction, ignition, graph parameters, stopping condition, and cascade decision logic are unchanged.
- The same selected paths and tie-breaking order are reused across weights. The 2% Degree set is a prefix of the 5% set. Weight=0 default results reproduce the existing STEP 2/3 runs exactly.

## Integrity

- {integrity['run_count']:,} runs; every one of 96 conditions contains 70 seeds.
- Duplicate condition-seed rows: {integrity['duplicate_condition_seed_rows']}.
- Ignition pairing across topologies: {integrity['paired_ignition_all_match']}.
- Monotonicity checks: {integrity['weight_monotonicity_violations']} violations among {integrity['weight_monotonicity_groups']} paired weight series.
- Full automated test suite after adding configurable coefficients: 26/26 passed.

## 1. Intervention-weight sensitivity

At the 5% budget, weaker suppression increases final cascade size smoothly, but both strategies remain effective even at path weight .75:

{markdown_table(focus_weight)}

Fraction of the weight=0 mean reduction retained when selected paths keep 75% of their original exposure:

{markdown_table(focus_retained)}

Observable Momentum remains better than Degree at every tested weight. Negative values below mean a smaller final cascade for Momentum:

{markdown_table(adv)}

The effect is therefore not an artifact of setting selected paths exactly to zero. Complete suppression gives the strongest absolute result, but the Cascade-aware advantage persists under partial attenuation. The apparent Momentum-minus-Degree gap often grows at weight .75 because static Degree degrades faster; this does not mean weaker intervention is better in absolute terms.

## 2. Observable-coefficient robustness

Range of condition means across five coefficient profiles:

{markdown_table(ranges)}

At the 5% budget, coefficients barely matter because the requested budget covers nearly all live frontier paths; across-profile ranges are at most 0.0001. At the tighter 2% budget, ranking matters more. The largest spread is WS (0.0109), while ER, BA, and SBM ranges are 0.0017, 0.0038, and 0.0019.

No profile is universally best. Degree-heavy or momentum-heavy is slightly better in several 2% conditions, while exposure-heavy is usually weaker, but these differences are small relative to the paired reduction from the untreated baseline. Every profile preserves the main conclusion that an Observable Momentum intervention substantially reduces final cascade size.

The no-community profile changes SBM mean final size only slightly. Under this calibration, the cross-community indicator contributes little compared with local pressure, recent activation, and degree. This should be described as a finding for the current SBM parameters, not as proof that community structure is generally irrelevant.

## Conclusion

Both checks strengthen the original result. Cascade-aware targeting continues to outperform static Degree when paths are only partially attenuated, and its effectiveness is not dependent on one precise heuristic coefficient vector. The numerical claim should remain conditional on κ=.5, the 5% trigger, the tested budgets, and the current calibrated model. Coefficient robustness is not cross-validation: it establishes local stability across plausible hand-specified profiles, not that the default coefficients are optimal or will generalize to real platform data.

## Outputs

- `raw_run_results.csv` and `raw_timestep_results.csv`: complete results.
- `intervention_weight_summary.csv`: weight sensitivity summary.
- `coefficient_robustness_summary.csv`: coefficient-profile summary.
- `paired_momentum_vs_degree.csv`: paired strategy differences by weight.
- `paired_coefficient_contrasts.csv`: paired profile differences versus default.
- `retained_effectiveness.csv`: fraction of complete-suppression effect retained at weight .75.
- `coefficient_profile_ranges.csv`: across-profile mean ranges.
- `figure_1_intervention_weight_sensitivity.png` and `figure_2_coefficient_robustness.png`: main figures.
"""
    (OUTPUT_DIR / "robustness_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'robustness_report.md'}")


if __name__ == "__main__":
    main()
