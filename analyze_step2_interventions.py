"""Paired descriptive analysis for the focused STEP 2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT = Path("outputs") / "step2_intervention_70seeds"


def paired_summary(values: pd.Series) -> dict[str, float | int]:
    values = values.dropna().to_numpy(float)
    n = values.size
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    mean = float(values.mean()) if n else np.nan
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(values)) if n else np.nan,
        "sd": sd,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
        "positive_fraction": float((values > 0).mean()) if n else np.nan,
        "zero_fraction": float((values == 0).mean()) if n else np.nan,
        "negative_fraction": float((values < 0).mean()) if n else np.nan,
        "cohens_dz": mean / sd if sd > 0 else np.nan,
    }


def reduction_summaries(data: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "topology",
        "kappa",
        "intervention_strategy",
        "intervention_budget",
        "intervention_trigger",
    ]
    rows = []
    for key, group in data[data.intervention_strategy != "none"].groupby(keys):
        rows.append({**dict(zip(keys, key)), **paired_summary(group.cascade_reduction)})
    return pd.DataFrame(rows)


def strategy_contrasts(data: pd.DataFrame) -> pd.DataFrame:
    focus = data[
        (data.kappa == 0.5)
        & (data.intervention_trigger == 0.05)
        & (data.intervention_budget == 0.05)
    ]
    rows = []
    for topology, group in focus.groupby("topology"):
        wide = group.pivot(index="random_seed", columns="intervention_strategy", values="cascade_reduction")
        for strategy in [name for name in ("degree", "betweenness", "bridge") if name in wide]:
            difference = wide[strategy] - wide["random"]
            rows.append(
                {
                    "topology": topology,
                    "contrast": f"{strategy} - random",
                    **paired_summary(difference),
                    "two_sided_t_p": float(stats.ttest_1samp(difference, 0).pvalue),
                    "wilcoxon_p": (
                        float(stats.wilcoxon(difference).pvalue)
                        if np.any(difference != 0)
                        else 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def kappa_reduction_risk(data: pd.DataFrame) -> pd.DataFrame:
    subset = data[data.intervention_strategy == "kappa_reduction"]
    rows = []
    for key, group in subset.groupby(["topology", "kappa", "intervention_trigger"]):
        rows.append(
            {
                "topology": key[0],
                "kappa": key[1],
                "intervention_trigger": key[2],
                "trigger_rate": group.intervention_triggered.mean(),
                "mean_cascade_reduction": group.cascade_reduction.mean(),
                "runs_helped": int((group.cascade_reduction > 0).sum()),
                "runs_unchanged": int((group.cascade_reduction == 0).sum()),
                "runs_harmed": int((group.cascade_reduction < 0).sum()),
                "worst_increase": float((-group.cascade_reduction).max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    data = pd.read_csv(args.input_dir / "raw_run_results.csv")

    reductions = reduction_summaries(data)
    contrasts = strategy_contrasts(data)
    kappa_risk = kappa_reduction_risk(data)
    reductions.to_csv(args.input_dir / "paired_reduction_summary.csv", index=False)
    contrasts.to_csv(args.input_dir / "paired_strategy_contrasts.csv", index=False)
    kappa_risk.to_csv(args.input_dir / "kappa_reduction_risk.csv", index=False)

    config_path = args.input_dir / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    import matplotlib
    import networkx
    import scipy
    config["software_versions"] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "matplotlib": matplotlib.__version__,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Paired strategy contrasts at kappa=.5, trigger=5%, budget=5%")
    print(contrasts.round(5).to_string(index=False))
    print("\nKappa reduction risk")
    print(kappa_risk.round(5).to_string(index=False))


if __name__ == "__main__":
    main()

