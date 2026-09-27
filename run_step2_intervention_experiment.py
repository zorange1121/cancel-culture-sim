"""Focused STEP 2 intervention experiment with 70 paired random seeds.

The design screens strategy, timing, budget, and kappa without running the full
Cartesian product.  It preserves the calibrated baseline and uses identical
agent/KOL/ignition realizations within each random seed across topologies.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import scipy

from baseline_simulation import K, N, NUM_IGNITE, create_simulation
from intervention import (
    InterventionSpec,
    build_exposure_paths,
    run_intervention,
    select_exposure_paths,
)
from network_generation import SUPPORTED_TOPOLOGIES


OUTPUT_DIR = Path("outputs") / "step2_intervention_70seeds"
RANDOM_SEEDS = tuple(range(70))
KAPPAS = (0.4, 0.5, 0.6)
TRIGGERS = (0.05, 0.20, 0.50)
BUDGETS = (0.01, 0.02, 0.05, 0.10)
INTERVENTION_WEIGHT = 0.0
KAPPA_AFTER = 0.0
BETWEENNESS_SAMPLES = 16
GLOBAL_CASCADE_THRESHOLD = 0.5
STRATEGY_SEED_OFFSETS = {
    "random": 101,
    "degree": 211,
    "betweenness": 307,
    "bridge": 401,
    "kappa_reduction": 503,
}


def _metrics(counts: np.ndarray, n: int) -> dict[str, int | float]:
    final_count = int(counts[-1])
    t50_target = 0.5 * final_count
    t90_target = 0.9 * final_count
    return {
        "final_activated_count": final_count,
        "final_activated_proportion": final_count / n,
        "cascade_duration": int(counts.size - 1),
        "t50": int(np.flatnonzero(counts >= t50_target)[0]),
        "t90": int(np.flatnonzero(counts >= t90_target)[0]),
    }


def _conditions(topology: str) -> list[tuple[float, str, float | None, float]]:
    conditions: list[tuple[float, str, float | None, float]] = []
    for kappa in KAPPAS:
        conditions.append((kappa, "none", None, 0.0))

    network_strategies = ["random", "degree", "betweenness"]
    if topology == "SBM":
        network_strategies.append("bridge")

    # Kappa sensitivity at the early trigger and the common 5% budget.
    for kappa in KAPPAS:
        for strategy in network_strategies:
            conditions.append((kappa, strategy, 0.05, 0.05))

    # Timing comparison in the center of the transition region.
    for strategy in network_strategies:
        for trigger in (0.20, 0.50):
            conditions.append((0.5, strategy, trigger, 0.05))

    # Budget response at kappa=.5 and the early trigger.  The 5% condition is
    # already present above.
    for strategy in network_strategies:
        for budget in (0.01, 0.02, 0.10):
            conditions.append((0.5, strategy, 0.05, budget))

    # Global algorithmic intervention has no edge budget.
    for kappa in KAPPAS:
        conditions.append((kappa, "kappa_reduction", 0.05, 0.0))
    for trigger in (0.20, 0.50):
        conditions.append((0.5, "kappa_reduction", trigger, 0.0))
    return conditions


def _preselect_nested_budgets(simulation, paths, strategy: str, seed: int):
    selection_spec = InterventionSpec(
        strategy=strategy,
        trigger_progress=0.05,
        budget=max(BUDGETS),
        intervention_weight=INTERVENTION_WEIGHT,
        betweenness_samples=BETWEENNESS_SAMPLES,
    )
    selection_seed = seed * 10_000 + STRATEGY_SEED_OFFSETS[strategy]
    selected_max, available = select_exposure_paths(
        simulation,
        paths,
        selection_spec,
        np.random.default_rng(selection_seed),
    )
    nested = {}
    for budget in BUDGETS:
        count = min(int(round(budget * paths.count)), selected_max.size)
        nested[budget] = (selected_max[:count].copy(), available)
    return nested


def _run_seed(seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    run_rows: list[dict] = []
    timestep_rows: list[dict] = []
    pairing_rows: list[dict] = []
    reference_ignition: tuple[int, ...] | None = None

    for topology in SUPPORTED_TOPOLOGIES:
        simulation = create_simulation(
            topology,
            n=N,
            target_avg_degree=K,
            network_seed=seed,
            agent_seed=seed,
        )
        ignition = simulation.rng.choice(
            simulation.kol_idx, size=NUM_IGNITE, replace=False
        )
        ignition_tuple = tuple(sorted(int(node) for node in ignition))
        if reference_ignition is None:
            reference_ignition = ignition_tuple
        pairing_rows.append(
            {
                "topology": topology,
                "random_seed": seed,
                "initial_kol_seed_nodes": json.dumps(ignition_tuple),
                "matches_seed_reference": ignition_tuple == reference_ignition,
            }
        )

        paths = build_exposure_paths(simulation)
        selections = {}
        for strategy in ("random", "degree", "betweenness"):
            selections[strategy] = _preselect_nested_budgets(
                simulation, paths, strategy, seed
            )
        if topology == "SBM":
            selections["bridge"] = _preselect_nested_budgets(
                simulation, paths, "bridge", seed
            )

        for condition_id, (kappa, strategy, trigger, budget) in enumerate(
            _conditions(topology)
        ):
            if strategy == "none":
                spec = InterventionSpec()
                preselected = None
            elif strategy == "kappa_reduction":
                spec = InterventionSpec(
                    strategy=strategy,
                    trigger_progress=trigger,
                    kappa_after=KAPPA_AFTER,
                )
                preselected = None
            else:
                spec = InterventionSpec(
                    strategy=strategy,
                    trigger_progress=trigger,
                    budget=budget,
                    intervention_weight=INTERVENTION_WEIGHT,
                    betweenness_samples=BETWEENNESS_SAMPLES,
                )
                preselected = selections[strategy][budget]

            intervention_seed = seed * 10_000 + STRATEGY_SEED_OFFSETS.get(
                strategy, 0
            )
            result = run_intervention(
                simulation,
                kappa,
                spec,
                intervention_seed=intervention_seed,
                ignition_nodes=ignition,
                prepared_paths=paths,
                preselected_paths=preselected,
            )
            metrics = _metrics(result.activated_counts, simulation.n)
            row = {
                "topology": topology,
                "kappa": kappa,
                "random_seed": seed,
                "seed_strategy": "original_kol",
                "seed_fraction": NUM_IGNITE / simulation.n,
                "intervention_strategy": strategy,
                "intervention_budget": budget,
                "intervention_weight": (
                    INTERVENTION_WEIGHT if strategy not in ("none", "kappa_reduction") else np.nan
                ),
                "intervention_trigger": trigger,
                "intervention_triggered": result.intervention_triggered,
                "intervention_actual_ratio": result.actual_activated_ratio,
                "intervention_actual_time": result.actual_timestep,
                "selected_path_count": result.selected_path_count,
                "available_path_count": result.available_path_count,
                "kappa_after": result.kappa_after,
                "global_cascade": metrics["final_activated_proportion"] >= GLOBAL_CASCADE_THRESHOLD,
                **metrics,
            }
            run_rows.append(row)
            for timestep, count in enumerate(result.activated_counts):
                timestep_rows.append(
                    {
                        "topology": topology,
                        "kappa": kappa,
                        "random_seed": seed,
                        "intervention_strategy": strategy,
                        "intervention_budget": budget,
                        "intervention_trigger": trigger,
                        "timestep": timestep,
                        "activated_count": int(count),
                        "activated_proportion": count / simulation.n,
                    }
                )
    return run_rows, timestep_rows, pairing_rows


def _add_paired_reductions(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame.intervention_strategy == "none"][
        ["topology", "kappa", "random_seed", "final_activated_proportion"]
    ].rename(columns={"final_activated_proportion": "baseline_final_proportion"})
    result = frame.merge(
        baseline, on=["topology", "kappa", "random_seed"], how="left", validate="many_to_one"
    )
    result["cascade_reduction"] = (
        result.baseline_final_proportion - result.final_activated_proportion
    )
    result["intervention_efficiency"] = np.where(
        result.intervention_budget > 0,
        result.cascade_reduction / result.intervention_budget,
        np.nan,
    )
    return result


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "topology",
        "kappa",
        "intervention_strategy",
        "intervention_budget",
        "intervention_trigger",
    ]
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        final = group.final_activated_proportion.to_numpy(float)
        reduction = group.cascade_reduction.to_numpy(float)
        n = len(group)
        half_width = float(stats.t.ppf(0.975, n - 1) * final.std(ddof=1) / np.sqrt(n))
        global_group = group[group.global_cascade]
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_runs": n,
                "trigger_rate": group.intervention_triggered.mean(),
                "mean_final_cascade_size": final.mean(),
                "median_final_cascade_size": np.median(final),
                "sd_final_cascade_size": final.std(ddof=1),
                "ci95_lower": final.mean() - half_width,
                "ci95_upper": final.mean() + half_width,
                "global_cascade_probability": group.global_cascade.mean(),
                "conditional_global_cascade_size": (
                    global_group.final_activated_proportion.mean()
                    if len(global_group)
                    else np.nan
                ),
                "mean_cascade_reduction": reduction.mean(),
                "median_cascade_reduction": np.median(reduction),
                "sd_cascade_reduction": reduction.std(ddof=1),
                "mean_intervention_efficiency": group.intervention_efficiency.mean(),
                "mean_duration": group.cascade_duration.mean(),
                "mean_t50": group.t50.mean(),
                "mean_t90": group.t90.mean(),
            }
        )
    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "none": "#555555",
        "random": "#4C78A8",
        "degree": "#F58518",
        "betweenness": "#54A24B",
        "bridge": "#B279A2",
        "kappa_reduction": "#E45756",
    }

    focus = summary[
        (summary.kappa == 0.5)
        & (summary.intervention_budget.isin([0.0, 0.05]))
        & (summary.intervention_trigger.fillna(0).isin([0, 0.05]))
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = focus[focus.topology == topology].sort_values("intervention_strategy")
        labels = part.intervention_strategy.tolist()
        means = part.mean_final_cascade_size.to_numpy(float)
        lower = means - part.ci95_lower.to_numpy(float)
        upper = part.ci95_upper.to_numpy(float) - means
        axis.bar(
            labels,
            means,
            yerr=np.vstack([lower, upper]),
            color=[colors[label] for label in labels],
            capsize=3,
        )
        axis.set_title(topology)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("STEP 2 strategy screen: kappa=0.5, trigger=5%, budget=5% (95% CI)")
    fig.supxlabel("Intervention strategy")
    fig.supylabel("Final activated proportion")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_strategy_screen.png", dpi=180)
    plt.close(fig)

    timing = summary[
        (summary.kappa == 0.5)
        & (summary.intervention_budget == 0.05)
        & summary.intervention_trigger.isin(TRIGGERS)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = timing[timing.topology == topology]
        for strategy in part.intervention_strategy.unique():
            group = part[part.intervention_strategy == strategy].sort_values("intervention_trigger")
            axis.plot(
                100 * group.intervention_trigger,
                group.mean_cascade_reduction,
                "-o",
                label=strategy,
                color=colors[strategy],
            )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(topology)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Timing effect at kappa=0.5 and 5% path budget")
    fig.supxlabel("Requested cascade-progress trigger (%)")
    fig.supylabel("Mean paired cascade reduction")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_timing_effect.png", dpi=180)
    plt.close(fig)

    budget = summary[
        (summary.kappa == 0.5)
        & (summary.intervention_trigger == 0.05)
        & summary.intervention_budget.isin(BUDGETS)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = budget[budget.topology == topology]
        for strategy in part.intervention_strategy.unique():
            group = part[part.intervention_strategy == strategy].sort_values("intervention_budget")
            axis.plot(
                100 * group.intervention_budget,
                group.mean_cascade_reduction,
                "-o",
                label=strategy,
                color=colors[strategy],
            )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(topology)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Budget response at kappa=0.5 and 5% progress trigger")
    fig.supxlabel("Exposure-path budget (%)")
    fig.supylabel("Mean paired cascade reduction")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_budget_effect.png", dpi=180)
    plt.close(fig)

    sensitivity = summary[
        (summary.intervention_trigger == 0.05)
        & (summary.intervention_budget.isin([0.0, 0.05]))
        & (summary.intervention_strategy != "none")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = sensitivity[sensitivity.topology == topology]
        for strategy in part.intervention_strategy.unique():
            group = part[part.intervention_strategy == strategy].sort_values("kappa")
            axis.plot(
                group.kappa,
                group.mean_cascade_reduction,
                "-o",
                label=strategy,
                color=colors[strategy],
            )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(topology)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Intervention effect across kappa (5% trigger, 5% path budget)")
    fig.supxlabel("Kappa")
    fig.supylabel("Mean paired cascade reduction")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_4_kappa_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=len(RANDOM_SEEDS))
    args = parser.parse_args()
    seeds = RANDOM_SEEDS[: args.seeds]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed = futures[future]
            results[seed] = future.result()
            print(f"Completed seed {seed} ({len(results)}/{len(seeds)})", flush=True)

    runs, timesteps, pairing = [], [], []
    for seed in sorted(results):
        seed_runs, seed_timesteps, seed_pairing = results[seed]
        runs.extend(seed_runs)
        timesteps.extend(seed_timesteps)
        pairing.extend(seed_pairing)

    run_frame = _add_paired_reductions(pd.DataFrame(runs))
    timestep_frame = pd.DataFrame(timesteps)
    pairing_frame = pd.DataFrame(pairing)
    summary = _summarize(run_frame)
    run_frame.to_csv(args.output_dir / "raw_run_results.csv", index=False)
    timestep_frame.to_csv(args.output_dir / "raw_timestep_results.csv", index=False)
    pairing_frame.to_csv(args.output_dir / "paired_randomness_verification.csv", index=False)
    summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
    _plot(run_frame, summary, args.output_dir)

    config = {
        "experiment": "STEP 2 focused intervention study",
        "n": N,
        "target_average_degree": K,
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "random_seeds": list(seeds),
        "kappas": list(KAPPAS),
        "triggers": list(TRIGGERS),
        "budgets": list(BUDGETS),
        "intervention_weight": INTERVENTION_WEIGHT,
        "kappa_after": KAPPA_AFTER,
        "betweenness_samples": BETWEENNESS_SAMPLES,
        "global_cascade_threshold": GLOBAL_CASCADE_THRESHOLD,
        "global_cascade_threshold_reason": "0.5 lies in the sparse middle of the observed bimodal baseline distribution in the kappa 0.4-0.6 transition region",
        "design": "Full timing comparison at kappa=.5/budget=.05; full budget response at kappa=.5/trigger=.05; kappa sensitivity at trigger=.05/budget=.05; all use paired seeds",
        "elapsed_seconds": time.time() - started,
        "run_count": len(run_frame),
        "timestep_row_count": len(timestep_frame),
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
        },
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
