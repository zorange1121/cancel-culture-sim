"""Re-run the Step 1 kappa=0 baseline after correcting survey parsing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_simulation import (
    B,
    K,
    KOL_REACH,
    N,
    NUM_IGNITE,
    NUM_KOLS,
    P,
    R,
    RANDOM_SEED,
    SURVEY_PATH,
    create_simulation,
    load_survey_and_fit,
    parse_pct,
)
from network_generation import SUPPORTED_TOPOLOGIES
from run_topology_baseline_experiment import cascade_metrics, write_csv


KAPPA = 0.0
MAX_STEPS = 60
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
DEFAULT_OUTPUT_DIR = (
    Path("outputs") / "step1_topology_baseline_corrected_parsing_kappa0_seed42"
)
BEFORE_THRESHOLD_PATH = (
    Path("outputs")
    / "activation_mechanism_diagnostic_kappa0_seed42"
    / "threshold_distribution.csv"
)


def describe(values: np.ndarray) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    for quantile, quantile_value in zip(QUANTILES, np.quantile(values, QUANTILES)):
        result[f"q{int(quantile * 100):02d}"] = float(quantile_value)
    return result


def survey_fit_summary() -> tuple[dict, tuple[float, float], tuple[float, float]]:
    dataframe = pd.read_excel(SURVEY_PATH)
    q1_col, q4_col, q5_col = dataframe.columns[1], dataframe.columns[7], dataframe.columns[8]
    poster_mask = dataframe[q1_col].str.startswith("是", na=False)
    q4 = dataframe.loc[poster_mask, q4_col].apply(parse_pct).dropna().to_numpy(float)
    q5 = dataframe.loc[poster_mask, q5_col].apply(parse_pct).dropna().to_numpy(float)
    _, beta_private, beta_public = load_survey_and_fit(SURVEY_PATH)
    summary = {
        "survey_path": str(SURVEY_PATH),
        "percentage_scale": "proportions in [0, 1]",
        "Q4": describe(q4),
        "Q5": describe(q5),
        "fitted_beta_parameters": {
            "private_Q4": {"alpha": beta_private[0], "beta": beta_private[1]},
            "public_Q5": {"alpha": beta_public[0], "beta": beta_public[1]},
        },
    }
    return summary, beta_private, beta_public


def threshold_rows(simulation) -> list[dict]:
    kol_mask = np.zeros(N, dtype=bool)
    kol_mask[simulation.kol_idx] = True
    poster_mask = simulation.roles == "Poster"
    raw_costs = np.where(
        simulation.roles == "Lurker",
        simulation.c_private,
        simulation.c_public,
    )
    thresholds = (raw_costs + R) / B
    groups = {
        "all_agents": np.ones(N, dtype=bool),
        "non_kol_agents": ~kol_mask,
        "lurkers": simulation.roles == "Lurker",
        "posters": poster_mask,
        "non_kol_posters": poster_mask & ~kol_mask,
        "kols": kol_mask,
    }
    rows = []
    for group, mask in groups.items():
        rows.append({"group": group, **describe(thresholds[mask])})
    return rows


def threshold_before_after(after_rows: list[dict]) -> list[dict]:
    if not BEFORE_THRESHOLD_PATH.exists():
        return []
    with BEFORE_THRESHOLD_PATH.open(newline="", encoding="utf-8") as stream:
        before = {row["group"]: row for row in csv.DictReader(stream)}
    after = {row["group"]: row for row in after_rows}
    rows = []
    for group in ("all_agents", "lurkers", "posters", "kols"):
        rows.append(
            {
                "group": group,
                "before_mean": float(before[group]["threshold_mean"]),
                "after_mean": after[group]["mean"],
                "before_median": float(before[group]["threshold_median"]),
                "after_median": after[group]["median"],
                "before_min": float(before[group]["threshold_min"]),
                "after_min": after[group]["min"],
                "before_max": float(before[group]["threshold_max"]),
                "after_max": after[group]["max"],
            }
        )
    return rows


def run_baseline(seed: int) -> tuple[list[dict], list[dict], dict[str, list[float]], list[dict], dict]:
    raw_rows: list[dict] = []
    metric_rows: list[dict] = []
    histories: dict[str, list[float]] = {}
    thresholds: list[dict] = []
    topology_parameters: dict[str, dict] = {}

    for topology in SUPPORTED_TOPOLOGIES:
        simulation = create_simulation(
            topology=topology,
            n=N,
            target_avg_degree=K,
            network_seed=seed,
            agent_seed=seed,
        )
        if not thresholds:
            thresholds = threshold_rows(simulation)
        history, _ = simulation.run(
            kappa=KAPPA,
            benefit=B,
            outside_payoff=R,
            max_steps=MAX_STEPS,
            num_ignite=NUM_IGNITE,
        )
        counts = np.rint(history * N).astype(int)
        histories[topology] = history.tolist()
        topology_parameters[topology] = simulation.base_graph.graph["topology_parameters"]
        for timestep, (count, proportion) in enumerate(zip(counts, history)):
            raw_rows.append(
                {
                    "topology": topology,
                    "kappa": KAPPA,
                    "random_seed": seed,
                    "timestep": timestep,
                    "activated_count": int(count),
                    "activated_proportion": float(proportion),
                }
            )
        metrics = cascade_metrics(counts, N)
        metric_rows.append(
            {
                "topology": topology,
                "kappa": KAPPA,
                "random_seed": seed,
                "cascade_grew_beyond_seeds": bool(counts[-1] > NUM_IGNITE),
                **metrics,
            }
        )

    config = {
        "experiment": "Step 1 topology baseline after corrected percentage parsing",
        "only_model_change": "correct survey percentage parsing before Beta fitting",
        "N": N,
        "target_average_degree": K,
        "WS_rewiring_probability": P,
        "B": B,
        "R": R,
        "kappa": KAPPA,
        "num_kols": NUM_KOLS,
        "kol_reach": KOL_REACH,
        "num_initial_seeds": NUM_IGNITE,
        "network_seed": seed,
        "agent_seed": seed,
        "max_steps": MAX_STEPS,
        "stopping_condition": "no state changes or max_steps reached",
        "topology_parameters": topology_parameters,
    }
    return raw_rows, metric_rows, histories, thresholds, config


def plot_histories(path: Path, histories: dict[str, list[float]], seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "ER": {"color": "#1f77b4", "marker": "o"},
        "WS": {"color": "#ff7f0e", "marker": "s"},
        "BA": {"color": "#2ca02c", "marker": "^"},
        "SBM": {"color": "#d62728", "marker": "D"},
    }
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for topology in SUPPORTED_TOPOLOGIES:
        values = histories[topology]
        axis.plot(
            range(len(values)),
            values,
            linewidth=2.3,
            markersize=6,
            label=topology,
            **styles[topology],
        )
    axis.set_title(
        f"Cascade Dynamics by Network Topology (kappa=0, seed={seed})",
        fontsize=15,
        weight="bold",
    )
    axis.set_xlabel("Simulation timestep", fontsize=12)
    axis.set_ylabel("Activated proportion", fontsize=12)
    observed_max = max(max(values) for values in histories.values())
    axis.set_ylim(0, min(1.03, max(0.05, observed_max * 1.15)))
    axis.set_xlim(left=0)
    axis.grid(alpha=0.25)
    axis.legend(title="Topology", frameon=True)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    survey_summary, _, _ = survey_fit_summary()
    raw_rows, metric_rows, histories, thresholds, config = run_baseline(args.seed)
    comparisons = threshold_before_after(thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "topology_cascade_timesteps.csv"
    metrics_path = args.output_dir / "topology_cascade_metrics.csv"
    threshold_path = args.output_dir / "effective_threshold_distribution.csv"
    comparison_path = args.output_dir / "threshold_before_after.csv"
    survey_path = args.output_dir / "survey_fit_summary.json"
    config_path = args.output_dir / "experiment_config.json"
    figure_path = args.output_dir / "topology_cascade_curves_kappa0_seed42.png"
    write_csv(raw_path, raw_rows)
    write_csv(metrics_path, metric_rows)
    write_csv(threshold_path, thresholds)
    if comparisons:
        write_csv(comparison_path, comparisons)
    survey_path.write_text(json.dumps(survey_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_histories(figure_path, histories, args.seed)
    print(json.dumps(survey_summary, ensure_ascii=False, indent=2))
    print("Cascade metrics:")
    for row in metric_rows:
        print(row)
    print(f"Figure: {figure_path.resolve()}")
    print(f"Raw timesteps: {raw_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Thresholds: {threshold_path.resolve()}")
    print(f"Before/after: {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
