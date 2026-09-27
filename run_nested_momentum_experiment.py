"""Paired nested Pressure-plus-Momentum experiment.

This experiment changes only the frontier-ranking score:

    S_lambda = (1 - lambda) * S_pressure + lambda * newly_active_sender

Lambda zero is exactly Observable Pressure. All graph, agent, threshold, KOL,
ignition, cascade, trigger, budget, and attenuation settings remain unchanged.
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
from intervention import InterventionSpec, build_exposure_paths, run_intervention
from network_generation import SUPPORTED_TOPOLOGIES


OUTPUT_DIR = Path("outputs") / "nested_momentum_70seeds"
BASELINE_RUNS = Path("outputs") / "step2_intervention_70seeds" / "raw_run_results.csv"
SEEDS = tuple(range(70))
LAMBDAS = (0.0, 0.15, 0.35, 0.50)
BUDGETS = (0.001, 0.0025, 0.005, 0.01)
KAPPA = 0.5
TRIGGER = 0.05
INTERVENTION_WEIGHT = 0.75
GLOBAL_CASCADE_THRESHOLD = 0.5


def nested_coefficients(momentum_lambda: float) -> tuple[float, ...]:
    """Return coefficients in observable_momentum feature order."""

    remaining = 1.0 - momentum_lambda
    # Pressure feature order: local, sender degree, receiver degree, community.
    return (
        remaining * 0.55,
        momentum_lambda,
        remaining * 0.25,
        remaining * 0.15,
        remaining * 0.05,
    )


def outcome_metrics(counts: np.ndarray) -> dict[str, int | float]:
    final_count = int(counts[-1])
    return {
        "final_activated_count": final_count,
        "final_activated_proportion": final_count / N,
        "cascade_duration": int(counts.size - 1),
        "t50": int(np.flatnonzero(counts >= 0.5 * final_count)[0]),
    }


def run_seed(seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    run_rows: list[dict] = []
    timestep_rows: list[dict] = []
    pairing_rows: list[dict] = []
    reference_ignition: tuple[int, ...] | None = None

    for topology_index, topology in enumerate(SUPPORTED_TOPOLOGIES):
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

        # A shared tie-breaking stream makes lambda and budget comparisons paired.
        intervention_seed = seed * 1_000_000 + topology_index * 10_000 + 4_242
        for budget in BUDGETS:
            for momentum_lambda in LAMBDAS:
                coefficients = nested_coefficients(momentum_lambda)
                spec = InterventionSpec(
                    strategy="observable_momentum",
                    trigger_progress=TRIGGER,
                    budget=budget,
                    intervention_weight=INTERVENTION_WEIGHT,
                    observable_coefficients=coefficients,
                )
                result = run_intervention(
                    simulation,
                    KAPPA,
                    spec,
                    intervention_seed=intervention_seed,
                    ignition_nodes=ignition,
                    prepared_paths=paths,
                    dynamic_selector=select_cascade_aware_paths,
                )
                outcome = outcome_metrics(result.activated_counts)
                coverage = result.selected_path_count / paths.count
                method = f"pressure_momentum_lambda_{momentum_lambda:g}"
                common = {
                    "topology": topology,
                    "kappa": KAPPA,
                    "random_seed": seed,
                    "method": method,
                    "momentum_lambda": momentum_lambda,
                    "budget": budget,
                    "trigger": TRIGGER,
                    "intervention_weight": INTERVENTION_WEIGHT,
                }
                run_rows.append(
                    {
                        **common,
                        "score_local_weight": coefficients[0],
                        "score_momentum_weight": coefficients[1],
                        "score_sender_degree_weight": coefficients[2],
                        "score_receiver_degree_weight": coefficients[3],
                        "score_cross_community_weight": coefficients[4],
                        "intervention_triggered": result.intervention_triggered,
                        "actual_trigger_time": result.actual_timestep,
                        "actual_trigger_ratio": result.actual_activated_ratio,
                        "selected_path_count": result.selected_path_count,
                        "total_path_count": paths.count,
                        "path_coverage_cost": coverage,
                        "intensity_adjusted_cost": coverage
                        * (1.0 - INTERVENTION_WEIGHT),
                        "true_frontier_count": result.true_frontier_count_at_trigger,
                        "global_cascade": outcome["final_activated_proportion"]
                        >= GLOBAL_CASCADE_THRESHOLD,
                        **outcome,
                    }
                )
                for timestep, count in enumerate(result.activated_counts):
                    timestep_rows.append(
                        {
                            **common,
                            "timestep": timestep,
                            "activated_count": int(count),
                            "activated_proportion": float(count / N),
                        }
                    )

    return run_rows, timestep_rows, pairing_rows


def attach_baseline(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(path)
    baseline = baseline[
        (baseline.intervention_strategy == "none")
        & np.isclose(baseline.kappa, KAPPA)
    ][["topology", "random_seed", "final_activated_proportion"]].drop_duplicates(
        ["topology", "random_seed"]
    )
    baseline = baseline.rename(
        columns={"final_activated_proportion": "baseline_final"}
    )
    merged = frame.merge(
        baseline, on=["topology", "random_seed"], validate="many_to_one"
    )
    merged["cascade_reduction"] = (
        merged.baseline_final - merged.final_activated_proportion
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
    completed: dict[int, tuple[list[dict], list[dict], list[dict]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in selected_seeds}
        for future in as_completed(futures):
            seed = futures[future]
            completed[seed] = future.result()
            print(
                f"Completed seed {seed} ({len(completed)}/{len(selected_seeds)})",
                flush=True,
            )

    run_rows: list[dict] = []
    timestep_rows: list[dict] = []
    pairing_rows: list[dict] = []
    for seed in sorted(completed):
        seed_runs, seed_timesteps, seed_pairing = completed[seed]
        run_rows.extend(seed_runs)
        timestep_rows.extend(seed_timesteps)
        pairing_rows.extend(seed_pairing)

    frame = attach_baseline(pd.DataFrame(run_rows), args.baseline_runs)
    frame.to_csv(args.output_dir / "raw_run_results.csv", index=False)
    pd.DataFrame(timestep_rows).to_csv(
        args.output_dir / "raw_timestep_results.csv", index=False
    )
    pd.DataFrame(pairing_rows).to_csv(
        args.output_dir / "paired_randomness_verification.csv", index=False
    )
    config = {
        "design": "S_lambda = (1-lambda) * ObservablePressure + lambda * newly_active_sender",
        "n": N,
        "target_average_degree": K,
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "random_seeds": list(selected_seeds),
        "kappa": KAPPA,
        "trigger": TRIGGER,
        "intervention_weight": INTERVENTION_WEIGHT,
        "budgets": list(BUDGETS),
        "momentum_lambdas": list(LAMBDAS),
        "pressure_coefficients": [0.55, 0.25, 0.15, 0.05],
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
