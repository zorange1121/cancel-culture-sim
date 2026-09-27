"""Sensitivity checks for intervention weight and Observable Momentum coefficients."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from baseline_simulation import K, N, NUM_IGNITE, create_simulation
from cascade_aware import select_cascade_aware_paths
from intervention import (
    InterventionSpec,
    build_exposure_paths,
    run_intervention,
    select_exposure_paths,
)
from network_generation import SUPPORTED_TOPOLOGIES


OUTPUT_DIR = Path("outputs") / "intervention_robustness_70seeds"
STEP2_RUNS = Path("outputs") / "step2_intervention_70seeds" / "raw_run_results.csv"
SEEDS = tuple(range(70))
KAPPA = 0.5
TRIGGER = 0.05
BUDGETS = (0.02, 0.05)
WEIGHTS = (0.0, 0.25, 0.5, 0.75)
COEFFICIENT_PROFILES = {
    "default": (0.35, 0.35, 0.15, 0.10, 0.05),
    "exposure_heavy": (0.55, 0.20, 0.10, 0.10, 0.05),
    "momentum_heavy": (0.20, 0.55, 0.10, 0.10, 0.05),
    "degree_heavy": (0.25, 0.25, 0.25, 0.20, 0.05),
    "no_community": (0.375, 0.375, 0.15, 0.10, 0.0),
}


def metrics(counts: np.ndarray, n: int) -> dict[str, int | float]:
    final_count = int(counts[-1])
    return {
        "final_activated_count": final_count,
        "final_activated_proportion": final_count / n,
        "cascade_duration": int(counts.size - 1),
        "t50": int(np.flatnonzero(counts >= 0.5 * final_count)[0]),
    }


def append_result(
    rows: list[dict],
    timesteps: list[dict],
    *,
    topology: str,
    seed: int,
    strategy: str,
    budget: float,
    weight: float,
    profile: str,
    coefficients: tuple[float, ...] | None,
    result,
    total_paths: int,
) -> None:
    outcome = metrics(result.activated_counts, N)
    common = {
        "topology": topology,
        "kappa": KAPPA,
        "random_seed": seed,
        "intervention_strategy": strategy,
        "intervention_budget": budget,
        "intervention_weight": weight,
        "intervention_trigger": TRIGGER,
        "coefficient_profile": profile,
        "observable_coefficients": json.dumps(coefficients) if coefficients else "",
    }
    rows.append(
        {
            **common,
            "intervention_triggered": result.intervention_triggered,
            "intervention_actual_ratio": result.actual_activated_ratio,
            "intervention_actual_time": result.actual_timestep,
            "selected_path_count": result.selected_path_count,
            "available_path_count": result.available_path_count,
            "total_path_count": total_paths,
            "actual_budget_fraction": result.selected_path_count / total_paths,
            **outcome,
        }
    )
    for timestep, count in enumerate(result.activated_counts):
        timesteps.append(
            {
                **common,
                "timestep": timestep,
                "activated_count": int(count),
                "activated_proportion": count / N,
            }
        )


def run_seed(seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rows, timesteps, pairing = [], [], []
    reference_ignition = None
    for topology in SUPPORTED_TOPOLOGIES:
        simulation = create_simulation(
            topology,
            n=N,
            target_avg_degree=K,
            network_seed=seed,
            agent_seed=seed,
        )
        ignition = simulation.rng.choice(simulation.kol_idx, size=NUM_IGNITE, replace=False)
        ignition_tuple = tuple(sorted(map(int, ignition)))
        if reference_ignition is None:
            reference_ignition = ignition_tuple
        pairing.append(
            {
                "topology": topology,
                "random_seed": seed,
                "initial_kol_seed_nodes": json.dumps(ignition_tuple),
                "matches_seed_reference": ignition_tuple == reference_ignition,
            }
        )
        paths = build_exposure_paths(simulation)

        degree_template = InterventionSpec(
            strategy="degree",
            trigger_progress=TRIGGER,
            budget=max(BUDGETS),
            intervention_weight=0.0,
        )
        degree_selected_max, degree_available = select_exposure_paths(
            simulation,
            paths,
            degree_template,
            np.random.default_rng(seed * 10_000 + 211),
        )

        for budget in BUDGETS:
            degree_count = min(int(round(budget * paths.count)), degree_selected_max.size)
            degree_selected = (degree_selected_max[:degree_count].copy(), degree_available)
            for weight in WEIGHTS:
                degree_spec = InterventionSpec(
                    strategy="degree",
                    trigger_progress=TRIGGER,
                    budget=budget,
                    intervention_weight=weight,
                )
                degree_result = run_intervention(
                    simulation,
                    KAPPA,
                    degree_spec,
                    intervention_seed=seed * 10_000 + 211,
                    ignition_nodes=ignition,
                    prepared_paths=paths,
                    preselected_paths=degree_selected,
                )
                append_result(
                    rows,
                    timesteps,
                    topology=topology,
                    seed=seed,
                    strategy="degree",
                    budget=budget,
                    weight=weight,
                    profile="not_applicable",
                    coefficients=None,
                    result=degree_result,
                    total_paths=paths.count,
                )

                coefficients = COEFFICIENT_PROFILES["default"]
                momentum_spec = InterventionSpec(
                    strategy="observable_momentum",
                    trigger_progress=TRIGGER,
                    budget=budget,
                    intervention_weight=weight,
                    observable_coefficients=coefficients,
                )
                momentum_result = run_intervention(
                    simulation,
                    KAPPA,
                    momentum_spec,
                    intervention_seed=seed * 10_000 + 701,
                    ignition_nodes=ignition,
                    prepared_paths=paths,
                    dynamic_selector=select_cascade_aware_paths,
                )
                append_result(
                    rows,
                    timesteps,
                    topology=topology,
                    seed=seed,
                    strategy="observable_momentum",
                    budget=budget,
                    weight=weight,
                    profile="default",
                    coefficients=coefficients,
                    result=momentum_result,
                    total_paths=paths.count,
                )

            for profile, coefficients in COEFFICIENT_PROFILES.items():
                if profile == "default":
                    continue
                spec = InterventionSpec(
                    strategy="observable_momentum",
                    trigger_progress=TRIGGER,
                    budget=budget,
                    intervention_weight=0.0,
                    observable_coefficients=coefficients,
                )
                result = run_intervention(
                    simulation,
                    KAPPA,
                    spec,
                    intervention_seed=seed * 10_000 + 701,
                    ignition_nodes=ignition,
                    prepared_paths=paths,
                    dynamic_selector=select_cascade_aware_paths,
                )
                append_result(
                    rows,
                    timesteps,
                    topology=topology,
                    seed=seed,
                    strategy="observable_momentum",
                    budget=budget,
                    weight=0.0,
                    profile=profile,
                    coefficients=coefficients,
                    result=result,
                    total_paths=paths.count,
                )
    return rows, timesteps, pairing


def attach_baselines(frame: pd.DataFrame, step2_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(step2_path)
    baseline = baseline[(baseline.intervention_strategy == "none") & (baseline.kappa == KAPPA)]
    baseline = baseline[["topology", "random_seed", "final_activated_proportion"]].drop_duplicates(
        ["topology", "random_seed"]
    )
    baseline = baseline.rename(columns={"final_activated_proportion": "baseline_final_proportion"})
    result = frame.merge(baseline, on=["topology", "random_seed"], validate="many_to_one")
    result["cascade_reduction"] = result.baseline_final_proportion - result.final_activated_proportion
    return result


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    records = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        final = group.final_activated_proportion.to_numpy(float)
        reduction = group.cascade_reduction.to_numpy(float)
        half = stats.t.ppf(0.975, len(group) - 1) * reduction.std(ddof=1) / np.sqrt(len(group))
        records.append(
            {
                **dict(zip(keys, key)),
                "n_runs": len(group),
                "mean_final_cascade_size": final.mean(),
                "sd_final_cascade_size": final.std(ddof=1),
                "mean_paired_reduction": reduction.mean(),
                "median_paired_reduction": np.median(reduction),
                "sd_paired_reduction": reduction.std(ddof=1),
                "reduction_ci95_lower": reduction.mean() - half,
                "reduction_ci95_upper": reduction.mean() + half,
                "mean_duration": group.cascade_duration.mean(),
                "mean_actual_budget_fraction": group.actual_budget_fraction.mean(),
            }
        )
    return pd.DataFrame(records)


def paired_advantage(frame: pd.DataFrame) -> pd.DataFrame:
    focus = frame[frame.coefficient_profile.isin(["default", "not_applicable"])]
    wide = focus.pivot(
        index=["topology", "random_seed", "intervention_budget", "intervention_weight"],
        columns="intervention_strategy",
        values="final_activated_proportion",
    ).reset_index()
    wide["momentum_minus_degree"] = wide.observable_momentum - wide.degree
    records = []
    keys = ["topology", "intervention_budget", "intervention_weight"]
    for key, group in wide.groupby(keys):
        values = group.momentum_minus_degree.to_numpy(float)
        half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        records.append(
            {
                **dict(zip(keys, key)),
                "n_pairs": len(values),
                "mean_momentum_minus_degree": values.mean(),
                "median_momentum_minus_degree": np.median(values),
                "sd_difference": values.std(ddof=1),
                "ci95_lower": values.mean() - half,
                "ci95_upper": values.mean() + half,
                "momentum_better_fraction": (values < 0).mean(),
            }
        )
    return pd.DataFrame(records)


def plot_results(weight_summary: pd.DataFrame, coefficient_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"degree": "#F58518", "observable_momentum": "#4C78A8"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = weight_summary[weight_summary.topology == topology]
        for (strategy, budget), group in part.groupby(["intervention_strategy", "intervention_budget"]):
            group = group.sort_values("intervention_weight")
            axis.plot(
                group.intervention_weight,
                group.mean_final_cascade_size,
                "-o" if budget == 0.05 else "--o",
                color=colors[strategy],
                label=f"{strategy.replace('_', ' ')}; budget={budget:.0%}",
            )
        axis.set_title(topology)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Partial suppression sensitivity (kappa=.5, trigger=5%)", y=1.02)
    fig.supxlabel("Selected path weight (0 = blocked; 1 = unchanged)")
    fig.supylabel("Mean final activated proportion")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "figure_1_intervention_weight_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = coefficient_summary[coefficient_summary.topology == topology].copy()
        for budget, group in part.groupby("intervention_budget"):
            group = group.sort_values("coefficient_profile")
            axis.plot(
                group.coefficient_profile,
                group.mean_final_cascade_size,
                "-o",
                label=f"budget={budget:.0%}",
            )
        axis.set_title(topology)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Observable Momentum coefficient robustness (weight=0)", y=1.02)
    fig.supylabel("Mean final activated proportion")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "figure_2_coefficient_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--step2-runs", type=Path, default=STEP2_RUNS)
    parser.add_argument("--seeds", type=int, default=70)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_seeds = SEEDS[: args.seeds]
    started = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in selected_seeds}
        for future in as_completed(futures):
            seed = futures[future]
            results[seed] = future.result()
            print(f"Completed seed {seed} ({len(results)}/{len(selected_seeds)})", flush=True)

    rows, timestep_rows, pairing_rows = [], [], []
    for seed in sorted(results):
        run_part, timestep_part, pairing_part = results[seed]
        rows.extend(run_part)
        timestep_rows.extend(timestep_part)
        pairing_rows.extend(pairing_part)
    frame = attach_baselines(pd.DataFrame(rows), args.step2_runs)
    weight_frame = frame[
        ((frame.intervention_strategy == "degree") & (frame.coefficient_profile == "not_applicable"))
        | ((frame.intervention_strategy == "observable_momentum") & (frame.coefficient_profile == "default"))
    ]
    coefficient_frame = frame[
        (frame.intervention_strategy == "observable_momentum")
        & (frame.intervention_weight == 0.0)
    ]
    weight_keys = ["topology", "intervention_strategy", "intervention_budget", "intervention_weight"]
    coefficient_keys = ["topology", "intervention_budget", "coefficient_profile"]
    weight_summary = summarize(weight_frame, weight_keys)
    coefficient_summary = summarize(coefficient_frame, coefficient_keys)
    advantage = paired_advantage(frame)

    frame.to_csv(args.output_dir / "raw_run_results.csv", index=False)
    pd.DataFrame(timestep_rows).to_csv(args.output_dir / "raw_timestep_results.csv", index=False)
    pd.DataFrame(pairing_rows).to_csv(args.output_dir / "paired_randomness_verification.csv", index=False)
    weight_summary.to_csv(args.output_dir / "intervention_weight_summary.csv", index=False)
    coefficient_summary.to_csv(args.output_dir / "coefficient_robustness_summary.csv", index=False)
    advantage.to_csv(args.output_dir / "paired_momentum_vs_degree.csv", index=False)
    plot_results(weight_summary, coefficient_summary, args.output_dir)

    config = {
        "n": N,
        "target_average_degree": K,
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "random_seeds": list(selected_seeds),
        "kappa": KAPPA,
        "trigger": TRIGGER,
        "budgets": BUDGETS,
        "intervention_weights": WEIGHTS,
        "coefficient_profiles": COEFFICIENT_PROFILES,
        "run_count": len(frame),
        "timestep_row_count": len(timestep_rows),
        "elapsed_seconds": time.time() - started,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
        },
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
