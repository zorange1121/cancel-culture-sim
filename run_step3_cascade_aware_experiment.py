"""STEP 3 static versus cascade-aware intervention experiment."""

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
from cascade_aware import CASCADE_AWARE_STRATEGIES, select_cascade_aware_paths
from intervention import InterventionSpec, build_exposure_paths, run_intervention
from network_generation import SUPPORTED_TOPOLOGIES


OUTPUT_DIR = Path("outputs") / "step3_cascade_aware_70seeds"
STEP2_DIR = Path("outputs") / "step2_intervention_70seeds"
RANDOM_SEEDS = tuple(range(70))
INTERVENTION_WEIGHT = 0.0
GLOBAL_CASCADE_THRESHOLD = 0.5
STRATEGY_OFFSETS = {
    "observable_pressure": 601,
    "observable_momentum": 701,
    "oracle_margin": 809,
}


def conditions() -> list[tuple[float, float, float]]:
    return [
        (0.5, 0.05, 0.02),
        (0.5, 0.05, 0.05),
        (0.5, 0.20, 0.05),
        (0.4, 0.05, 0.05),
        (0.6, 0.05, 0.05),
    ]


def metrics(counts: np.ndarray, n: int) -> dict[str, int | float]:
    final_count = int(counts[-1])
    return {
        "final_activated_count": final_count,
        "final_activated_proportion": final_count / n,
        "cascade_duration": int(counts.size - 1),
        "t50": int(np.flatnonzero(counts >= 0.5 * final_count)[0]),
        "t90": int(np.flatnonzero(counts >= 0.9 * final_count)[0]),
    }


def run_seed(seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    run_rows, timestep_rows, pairing_rows = [], [], []
    reference_ignition = None
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
        ignition_tuple = tuple(sorted(map(int, ignition)))
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

        for kappa, trigger, budget in conditions():
            for strategy in CASCADE_AWARE_STRATEGIES:
                spec = InterventionSpec(
                    strategy=strategy,
                    trigger_progress=trigger,
                    budget=budget,
                    intervention_weight=INTERVENTION_WEIGHT,
                )
                result = run_intervention(
                    simulation,
                    kappa,
                    spec,
                    intervention_seed=seed * 10_000 + STRATEGY_OFFSETS[strategy],
                    ignition_nodes=ignition,
                    prepared_paths=paths,
                    dynamic_selector=select_cascade_aware_paths,
                )
                outcome = metrics(result.activated_counts, simulation.n)
                run_rows.append(
                    {
                        "topology": topology,
                        "kappa": kappa,
                        "random_seed": seed,
                        "seed_strategy": "original_kol",
                        "seed_fraction": NUM_IGNITE / simulation.n,
                        "intervention_strategy": strategy,
                        "intervention_budget": budget,
                        "intervention_weight": INTERVENTION_WEIGHT,
                        "intervention_trigger": trigger,
                        "intervention_triggered": result.intervention_triggered,
                        "intervention_actual_ratio": result.actual_activated_ratio,
                        "intervention_actual_time": result.actual_timestep,
                        "selected_path_count": result.selected_path_count,
                        "available_path_count": result.available_path_count,
                        "total_path_count": paths.count,
                        "actual_budget_fraction": result.selected_path_count / paths.count,
                        "kappa_after": result.kappa_after,
                        "global_cascade": outcome["final_activated_proportion"] >= GLOBAL_CASCADE_THRESHOLD,
                        **outcome,
                    }
                )
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


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["topology", "kappa", "intervention_strategy", "intervention_budget", "intervention_trigger"]
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        final = group.final_activated_proportion.to_numpy(float)
        reduction = group.cascade_reduction.to_numpy(float)
        n = len(group)
        sd = final.std(ddof=1)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        global_group = group[group.global_cascade]
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_runs": n,
                "trigger_rate": group.intervention_triggered.mean(),
                "mean_final_cascade_size": final.mean(),
                "median_final_cascade_size": np.median(final),
                "sd_final_cascade_size": sd,
                "ci95_lower": final.mean() - half,
                "ci95_upper": final.mean() + half,
                "global_cascade_probability": group.global_cascade.mean(),
                "conditional_global_cascade_size": global_group.final_activated_proportion.mean() if len(global_group) else np.nan,
                "mean_cascade_reduction": reduction.mean(),
                "median_cascade_reduction": np.median(reduction),
                "sd_cascade_reduction": reduction.std(ddof=1),
                "mean_intervention_efficiency": group.intervention_efficiency.mean(),
                "mean_t50": group.t50.mean(),
                "mean_t90": group.t90.mean(),
                "mean_duration": group.cascade_duration.mean(),
            }
        )
    return pd.DataFrame(rows)


def comparator_rows(step2: pd.DataFrame) -> pd.DataFrame:
    requested = pd.DataFrame(conditions(), columns=["kappa", "intervention_trigger", "intervention_budget"])
    static = step2[step2.intervention_strategy.isin(["random", "degree", "betweenness", "bridge"])].merge(
        requested,
        on=["kappa", "intervention_trigger", "intervention_budget"],
        how="inner",
    )
    baseline_keys = step2[step2.intervention_strategy == "none"]
    return pd.concat([baseline_keys, static], ignore_index=True, sort=False)


def add_baseline_reduction(dynamic: pd.DataFrame, step2: pd.DataFrame) -> pd.DataFrame:
    baseline = step2[step2.intervention_strategy == "none"][[
        "topology", "kappa", "random_seed", "final_activated_proportion"
    ]].rename(columns={"final_activated_proportion": "baseline_final_proportion"})
    result = dynamic.merge(baseline, on=["topology", "kappa", "random_seed"], validate="many_to_one")
    result["cascade_reduction"] = result.baseline_final_proportion - result.final_activated_proportion
    result["intervention_efficiency"] = result.cascade_reduction / result.intervention_budget
    result["actual_intervention_efficiency"] = np.where(
        result.actual_budget_fraction > 0,
        result.cascade_reduction / result.actual_budget_fraction,
        np.nan,
    )
    return result


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "none": "#555555",
        "random": "#4C78A8",
        "degree": "#F58518",
        "betweenness": "#54A24B",
        "bridge": "#B279A2",
        "observable_pressure": "#72B7B2",
        "observable_momentum": "#ECA82C",
        "oracle_margin": "#E45756",
    }
    order = ["none", "random", "degree", "betweenness", "bridge", "observable_pressure", "observable_momentum", "oracle_margin"]
    labels = {
        "none": "None",
        "random": "Random",
        "degree": "Degree",
        "betweenness": "Betweenness",
        "bridge": "Bridge",
        "observable_pressure": "Obs. Pressure",
        "observable_momentum": "Obs. Momentum",
        "oracle_margin": "Oracle Margin",
    }

    focus = summary[(summary.kappa == 0.5) & (summary.intervention_trigger.fillna(0).isin([0, 0.05])) & (summary.intervention_budget.isin([0, 0.05]))]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = focus[focus.topology == topology].copy()
        part["rank"] = part.intervention_strategy.map({name: i for i, name in enumerate(order)})
        part = part.sort_values("rank")
        strategy_names = part.intervention_strategy.tolist()
        means = part.mean_final_cascade_size.to_numpy(float)
        axis.bar([labels[x] for x in strategy_names], means, color=[colors[x] for x in strategy_names])
        axis.set_title(topology)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Static vs Cascade-aware: kappa=.5, trigger=5%, budget=5%")
    fig.supxlabel("Strategy")
    fig.supylabel("Mean final activated proportion")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_static_vs_cascade_aware.png", dpi=180)
    plt.close(fig)

    adaptive = summary[(summary.kappa == 0.5) & (summary.intervention_trigger == 0.05) & summary.intervention_budget.isin([0.02, 0.05])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = adaptive[adaptive.topology == topology]
        for strategy in part.intervention_strategy.unique():
            group = part[part.intervention_strategy == strategy].sort_values("intervention_budget")
            axis.plot(100 * group.intervention_budget, group.mean_cascade_reduction, "-o", label=labels[strategy], color=colors[strategy])
        axis.set_title(topology)
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Budget comparison: Static vs Cascade-aware at 5% trigger")
    fig.supxlabel("Exposure-path budget (%)")
    fig.supylabel("Mean paired cascade reduction")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "figure_2_budget_comparison.png", dpi=180)
    plt.close(fig)

    timing = summary[(summary.kappa == 0.5) & (summary.intervention_budget == 0.05) & summary.intervention_trigger.isin([0.05, 0.20])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = timing[timing.topology == topology]
        for strategy in part.intervention_strategy.unique():
            group = part[part.intervention_strategy == strategy].sort_values("intervention_trigger")
            axis.plot(100 * group.intervention_trigger, group.mean_cascade_reduction, "-o", label=labels[strategy], color=colors[strategy])
        axis.set_title(topology)
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Timing comparison: Static vs Cascade-aware at 5% budget")
    fig.supxlabel("Cascade-progress trigger (%)")
    fig.supylabel("Mean paired cascade reduction")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "figure_3_timing_comparison.png", dpi=180)
    plt.close(fig)

    kappa = summary[(summary.intervention_budget == 0.05) & (summary.intervention_trigger == 0.05) & (summary.intervention_strategy != "none")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        part = kappa[kappa.topology == topology]
        for strategy in part.intervention_strategy.unique():
            group = part[part.intervention_strategy == strategy].sort_values("kappa")
            axis.plot(group.kappa, group.mean_cascade_reduction, "-o", label=labels[strategy], color=colors[strategy])
        axis.set_title(topology)
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Kappa sensitivity: Static vs Cascade-aware")
    fig.supxlabel("Kappa")
    fig.supylabel("Mean paired cascade reduction")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "figure_4_kappa_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--step2-dir", type=Path, default=STEP2_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=70)
    args = parser.parse_args()
    seeds = RANDOM_SEEDS[: args.seeds]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed = futures[future]
            results[seed] = future.result()
            print(f"Completed seed {seed} ({len(results)}/{len(seeds)})", flush=True)

    runs, timesteps, pairing = [], [], []
    for seed in sorted(results):
        a, b, c = results[seed]
        runs.extend(a)
        timesteps.extend(b)
        pairing.extend(c)
    dynamic = pd.DataFrame(runs)
    step2 = pd.read_csv(args.step2_dir / "raw_run_results.csv")
    dynamic = add_baseline_reduction(dynamic, step2)
    comparators = comparator_rows(step2)
    comparison = pd.concat([comparators, dynamic], ignore_index=True, sort=False)
    summary = summarize(comparison)

    dynamic.to_csv(args.output_dir / "raw_cascade_aware_run_results.csv", index=False)
    pd.DataFrame(timesteps).to_csv(args.output_dir / "raw_cascade_aware_timestep_results.csv", index=False)
    pd.DataFrame(pairing).to_csv(args.output_dir / "paired_randomness_verification.csv", index=False)
    comparison.to_csv(args.output_dir / "combined_static_dynamic_run_results.csv", index=False)
    summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
    plot_results(summary, args.output_dir)

    config = {
        "experiment": "STEP 3 static versus cascade-aware intervention",
        "n": N,
        "target_average_degree": K,
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "random_seeds": list(seeds),
        "dynamic_strategies": list(CASCADE_AWARE_STRATEGIES),
        "conditions": [
            {"kappa": k, "trigger": t, "budget": b} for k, t, b in conditions()
        ],
        "intervention_weight": INTERVENTION_WEIGHT,
        "global_cascade_threshold": GLOBAL_CASCADE_THRESHOLD,
        "elapsed_seconds": time.time() - started,
        "dynamic_run_count": len(dynamic),
        "dynamic_timestep_row_count": len(timesteps),
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
        },
    }
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
