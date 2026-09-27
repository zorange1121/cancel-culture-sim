"""Fair live-frontier baselines and Observable Momentum feature ablations.

The experiment keeps the calibrated game, graph, ignition, trigger, budget, and
path weight fixed.  Methods differ only in the candidate set/ranking used for
the one-time exposure-path intervention.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd

from baseline_simulation import K, N, NUM_IGNITE, create_simulation
from cascade_aware import select_cascade_aware_paths
from intervention import (
    InterventionSpec,
    build_exposure_paths,
    run_intervention,
    select_exposure_paths,
)
from network_generation import SUPPORTED_TOPOLOGIES


OUTPUT_DIR = Path("outputs") / "frontier_ablation_70seeds"
BASELINE_RUNS = Path("outputs") / "step2_intervention_70seeds" / "raw_run_results.csv"
SEEDS = tuple(range(70))
KAPPA = 0.5
TRIGGER = 0.05
BUDGET = 0.05
INTERVENTION_WEIGHT = 0.75
GLOBAL_CASCADE_THRESHOLD = 0.5

# The four removal profiles preserve the relative weights of retained features.
MOMENTUM_PROFILES = {
    "full_momentum": (0.35, 0.35, 0.15, 0.10, 0.05),
    "no_momentum": (0.5384615385, 0.0, 0.2307692308, 0.1538461538, 0.0769230769),
    "no_pressure": (0.0, 0.5384615385, 0.2307692308, 0.1538461538, 0.0769230769),
    "no_degree": (0.4666666667, 0.4666666667, 0.0, 0.0, 0.0666666666),
    "no_community": (0.3684210526, 0.3684210526, 0.1578947368, 0.1052631579, 0.0),
}


def outcome_metrics(counts: np.ndarray, n: int) -> dict[str, int | float]:
    final_count = int(counts[-1])
    return {
        "final_activated_count": final_count,
        "final_activated_proportion": final_count / n,
        "cascade_duration": int(counts.size - 1),
        "t50": int(np.flatnonzero(counts >= 0.5 * final_count)[0]),
    }


def append_result(
    rows: list[dict],
    *,
    topology: str,
    seed: int,
    method: str,
    strategy: str,
    coefficients: tuple[float, ...] | None,
    result,
    total_paths: int,
) -> None:
    coverage = result.selected_path_count / total_paths
    intensity = coverage * (1.0 - INTERVENTION_WEIGHT)
    outcome = outcome_metrics(result.activated_counts, N)
    rows.append(
        {
            "topology": topology,
            "kappa": KAPPA,
            "random_seed": seed,
            "method": method,
            "intervention_strategy": strategy,
            "observable_coefficients": json.dumps(coefficients) if coefficients else "",
            "intervention_trigger": TRIGGER,
            "intervention_budget": BUDGET,
            "intervention_weight": INTERVENTION_WEIGHT,
            "intervention_triggered": result.intervention_triggered,
            "intervention_actual_ratio": result.actual_activated_ratio,
            "intervention_actual_time": result.actual_timestep,
            "selected_path_count": result.selected_path_count,
            "available_path_count": result.available_path_count,
            "total_path_count": total_paths,
            "path_coverage_cost": coverage,
            "intensity_adjusted_cost": intensity,
            "affected_transition_count": result.affected_transition_count,
            "temporal_suppression_cost": result.temporal_suppression_cost,
            "active_source_suppression_cost": result.active_source_suppression_cost,
            "frontier_suppression_cost": result.frontier_suppression_cost,
            "global_cascade": outcome["final_activated_proportion"] >= GLOBAL_CASCADE_THRESHOLD,
            **outcome,
        }
    )


def run_seed(seed: int) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    pairing: list[dict] = []
    reference_ignition: tuple[int, ...] | None = None
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

        # Static controls use the same graph and trigger but select from all paths.
        for strategy, offset in (("random", 101), ("degree", 211)):
            spec = InterventionSpec(
                strategy=strategy,
                trigger_progress=TRIGGER,
                budget=BUDGET,
                intervention_weight=INTERVENTION_WEIGHT,
            )
            selected = select_exposure_paths(
                simulation, paths, spec, np.random.default_rng(seed * 10_000 + offset)
            )
            result = run_intervention(
                simulation,
                KAPPA,
                spec,
                intervention_seed=seed * 10_000 + offset,
                ignition_nodes=ignition,
                prepared_paths=paths,
                preselected_paths=selected,
            )
            append_result(
                rows,
                topology=topology,
                seed=seed,
                method=f"static_{strategy}",
                strategy=strategy,
                coefficients=None,
                result=result,
                total_paths=paths.count,
            )

        dynamic_specs: list[tuple[str, str, tuple[float, ...] | None, int]] = [
            ("frontier_random", "frontier_random", None, 301),
            ("frontier_degree", "frontier_degree", None, 401),
            ("observable_pressure", "observable_pressure", None, 601),
        ]
        dynamic_specs.extend(
            (profile, "observable_momentum", coefficients, 701)
            for profile, coefficients in MOMENTUM_PROFILES.items()
        )
        dynamic_specs.append(("oracle_margin", "oracle_margin", None, 809))

        for method, strategy, coefficients, offset in dynamic_specs:
            spec = InterventionSpec(
                strategy=strategy,
                trigger_progress=TRIGGER,
                budget=BUDGET,
                intervention_weight=INTERVENTION_WEIGHT,
                observable_coefficients=coefficients,
            )
            result = run_intervention(
                simulation,
                KAPPA,
                spec,
                intervention_seed=seed * 10_000 + offset,
                ignition_nodes=ignition,
                prepared_paths=paths,
                dynamic_selector=select_cascade_aware_paths,
            )
            append_result(
                rows,
                topology=topology,
                seed=seed,
                method=method,
                strategy=strategy,
                coefficients=coefficients,
                result=result,
                total_paths=paths.count,
            )
    return rows, pairing


def attach_baseline(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(path)
    baseline = baseline[
        (baseline.intervention_strategy == "none") & np.isclose(baseline.kappa, KAPPA)
    ][["topology", "random_seed", "final_activated_proportion"]].drop_duplicates(
        ["topology", "random_seed"]
    )
    baseline = baseline.rename(columns={"final_activated_proportion": "baseline_final_proportion"})
    merged = frame.merge(baseline, on=["topology", "random_seed"], validate="many_to_one")
    merged["cascade_reduction"] = (
        merged.baseline_final_proportion - merged.final_activated_proportion
    )
    for cost in (
        "path_coverage_cost",
        "intensity_adjusted_cost",
        "temporal_suppression_cost",
        "active_source_suppression_cost",
        "frontier_suppression_cost",
    ):
        merged[f"reduction_per_{cost}"] = np.divide(
            merged.cascade_reduction,
            merged[cost],
            out=np.full(len(merged), np.nan),
            where=merged[cost].to_numpy(float) > 0,
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--baseline-runs", type=Path, default=BASELINE_RUNS)
    parser.add_argument("--seeds", type=int, default=70)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_seeds = SEEDS[: args.seeds]
    started = time.time()
    results: dict[int, tuple[list[dict], list[dict]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in selected_seeds}
        for future in as_completed(futures):
            seed = futures[future]
            results[seed] = future.result()
            print(f"Completed seed {seed} ({len(results)}/{len(selected_seeds)})", flush=True)

    rows: list[dict] = []
    pairing: list[dict] = []
    for seed in sorted(results):
        run_rows, pair_rows = results[seed]
        rows.extend(run_rows)
        pairing.extend(pair_rows)
    frame = attach_baseline(pd.DataFrame(rows), args.baseline_runs)
    frame.to_csv(args.output_dir / "raw_run_results.csv", index=False)
    pd.DataFrame(pairing).to_csv(
        args.output_dir / "paired_randomness_verification.csv", index=False
    )
    config = {
        "n": N,
        "target_average_degree": K,
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "random_seeds": list(selected_seeds),
        "kappa": KAPPA,
        "trigger": TRIGGER,
        "budget": BUDGET,
        "intervention_weight": INTERVENTION_WEIGHT,
        "momentum_profiles": MOMENTUM_PROFILES,
        "run_count": len(frame),
        "elapsed_seconds": time.time() - started,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
        },
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
