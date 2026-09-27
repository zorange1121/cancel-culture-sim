"""State-information value experiment for one-shot frontier intervention.

The calibrated cascade dynamics are unchanged.  Only the snapshot shown to the
path selector, intervention budget, trigger, and ranking rule vary.  Selected
paths persist after the one-time trigger, matching the existing intervention
semantics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd

from baseline_simulation import K, N, NUM_IGNITE, create_simulation
from cascade_aware import select_cascade_aware_paths
from intervention import (
    CascadeObservationSpec,
    InterventionSpec,
    build_exposure_paths,
    run_intervention,
)
from network_generation import SUPPORTED_TOPOLOGIES


OUTPUT_DIR = Path("outputs") / "information_value_frontier_70seeds"
BASELINE_RUNS = Path("outputs") / "step2_intervention_70seeds" / "raw_run_results.csv"
SEEDS = tuple(range(70))
KAPPA = 0.5
WEIGHT = 0.75
GLOBAL_CASCADE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Condition:
    family: str
    name: str
    method: str
    strategy: str
    budget: float
    trigger: float
    delay: int = 0
    recall: float = 1.0
    false_positive_rate: float = 0.0
    coefficients: tuple[float, ...] | None = None


METHODS = {
    "static_random": "random",
    "static_degree": "degree",
    "frontier_random": "frontier_random",
    "frontier_degree": "frontier_degree",
    "observable_pressure": "observable_pressure",
    "full_momentum": "observable_momentum",
    "no_momentum": "observable_momentum",
    "oracle_margin": "oracle_margin",
}

NO_MOMENTUM_COEFFICIENTS = (
    0.5384615385,
    0.0,
    0.2307692308,
    0.1538461538,
    0.0769230769,
)


def build_conditions() -> tuple[Condition, ...]:
    conditions: list[Condition] = []
    # Ranking only becomes meaningful when the global path budget cannot cover
    # most of the live frontier.
    for budget in (0.001, 0.0025, 0.005, 0.01):
        for method in (
            "static_random",
            "static_degree",
            "frontier_random",
            "frontier_degree",
            "observable_pressure",
            "full_momentum",
            "no_momentum",
            "oracle_margin",
        ):
            strategy = METHODS[method]
            conditions.append(
                Condition(
                    family="tight_budget_ranking",
                    name=f"perfect_budget_{budget:g}",
                    method=method,
                    strategy=strategy,
                    budget=budget,
                    trigger=0.05,
                    coefficients=(
                        NO_MOMENTUM_COEFFICIENTS if method == "no_momentum" else None
                    ),
                )
            )

    observation_scenarios = (
        ("perfect", 0, 1.0, 0.0),
        ("delay_1", 1, 1.0, 0.0),
        ("delay_2", 2, 1.0, 0.0),
        ("delay_3", 3, 1.0, 0.0),
        ("recall_075", 0, 0.75, 0.0),
        ("recall_050", 0, 0.50, 0.0),
        ("recall_025", 0, 0.25, 0.0),
        ("fpr_002", 0, 1.0, 0.02),
        ("fpr_005", 0, 1.0, 0.05),
        ("fpr_010", 0, 1.0, 0.10),
        ("combined_moderate", 1, 0.75, 0.02),
    )
    for name, delay, recall, false_positive_rate in observation_scenarios:
        for method in ("frontier_random", "observable_pressure"):
            conditions.append(
                Condition(
                    family="observation_value",
                    name=name,
                    method=method,
                    strategy=METHODS[method],
                    budget=0.005,
                    trigger=0.05,
                    delay=delay,
                    recall=recall,
                    false_positive_rate=false_positive_rate,
                )
            )

    # The perfect-observation 5% trigger is already present in the ranking grid.
    for trigger in (0.02, 0.10, 0.20):
        for method in ("frontier_random", "observable_pressure"):
            conditions.append(
                Condition(
                    family="timing",
                    name=f"perfect_trigger_{trigger:g}",
                    method=method,
                    strategy=METHODS[method],
                    budget=0.005,
                    trigger=trigger,
                )
            )
    return tuple(conditions)


CONDITIONS = build_conditions()


def outcome_metrics(counts: np.ndarray) -> dict[str, int | float]:
    final_count = int(counts[-1])
    return {
        "final_activated_count": final_count,
        "final_activated_proportion": final_count / N,
        "cascade_duration": int(counts.size - 1),
        "t50": int(np.flatnonzero(counts >= 0.5 * final_count)[0]),
    }


def run_seed(seed: int, families: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    pairing: list[dict] = []
    selected_conditions = [c for c in CONDITIONS if c.family in families]
    reference_ignition: tuple[int, ...] | None = None
    for topology_index, topology in enumerate(SUPPORTED_TOPOLOGIES):
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
        for condition_index, condition in enumerate(selected_conditions):
            spec = InterventionSpec(
                strategy=condition.strategy,
                trigger_progress=condition.trigger,
                budget=condition.budget,
                intervention_weight=WEIGHT,
                observable_coefficients=condition.coefficients,
            )
            # Observation randomness is identical across methods for a shared
            # scenario, topology, and seed. Selection/tie randomness is separate.
            observation_seed = seed * 1_000_000 + topology_index * 10_000 + 777
            method_number = tuple(METHODS).index(condition.method)
            intervention_seed = (
                seed * 1_000_000 + topology_index * 10_000 + method_number + 1_000
            )
            is_state_conditioned = condition.strategy not in ("random", "degree")
            result = run_intervention(
                simulation,
                KAPPA,
                spec,
                intervention_seed=intervention_seed,
                observation_seed=observation_seed,
                observation_spec=(
                    CascadeObservationSpec(
                        delay_steps=condition.delay,
                        active_recall=condition.recall,
                        false_positive_rate=condition.false_positive_rate,
                    )
                    if is_state_conditioned
                    else None
                ),
                ignition_nodes=ignition,
                prepared_paths=paths,
                dynamic_selector=(
                    select_cascade_aware_paths if is_state_conditioned else None
                ),
            )
            outcome = outcome_metrics(result.activated_counts)
            coverage = result.selected_path_count / paths.count
            rows.append(
                {
                    "family": condition.family,
                    "condition": condition.name,
                    "topology": topology,
                    "kappa": KAPPA,
                    "random_seed": seed,
                    "method": condition.method,
                    "strategy": condition.strategy,
                    "trigger": condition.trigger,
                    "budget": condition.budget,
                    "intervention_weight": WEIGHT,
                    "observation_delay": condition.delay,
                    "active_recall": condition.recall,
                    "false_positive_rate": condition.false_positive_rate,
                    "intervention_triggered": result.intervention_triggered,
                    "actual_trigger_time": result.actual_timestep,
                    "actual_trigger_ratio": result.actual_activated_ratio,
                    "selected_path_count": result.selected_path_count,
                    "total_path_count": paths.count,
                    "path_coverage_cost": coverage,
                    "intensity_adjusted_cost": coverage * (1.0 - WEIGHT),
                    "temporal_suppression_cost": result.temporal_suppression_cost,
                    "true_frontier_count": result.true_frontier_count_at_trigger,
                    "observed_frontier_count": result.observed_frontier_count_at_trigger,
                    "observed_frontier_precision": result.observed_frontier_precision,
                    "observed_frontier_recall": result.observed_frontier_recall,
                    "global_cascade": outcome["final_activated_proportion"]
                    >= GLOBAL_CASCADE_THRESHOLD,
                    **outcome,
                }
            )
    return rows, pairing


def attach_baseline(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(path)
    baseline = baseline[
        (baseline.intervention_strategy == "none") & np.isclose(baseline.kappa, KAPPA)
    ][["topology", "random_seed", "final_activated_proportion"]].drop_duplicates(
        ["topology", "random_seed"]
    )
    baseline = baseline.rename(columns={"final_activated_proportion": "baseline_final"})
    merged = frame.merge(baseline, on=["topology", "random_seed"], validate="many_to_one")
    merged["cascade_reduction"] = merged.baseline_final - merged.final_activated_proportion
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--baseline-runs", type=Path, default=BASELINE_RUNS)
    parser.add_argument("--seeds", type=int, default=70)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--family",
        choices=("all", "tight_budget_ranking", "observation_value", "timing"),
        default="all",
    )
    args = parser.parse_args()
    families = (
        ("tight_budget_ranking", "observation_value", "timing")
        if args.family == "all"
        else (args.family,)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_seeds = SEEDS[: args.seeds]
    started = time.time()
    completed: dict[int, tuple[list[dict], list[dict]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_seed, seed, families): seed for seed in selected_seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            completed[seed] = future.result()
            print(f"Completed seed {seed} ({len(completed)}/{len(selected_seeds)})", flush=True)

    rows: list[dict] = []
    pairing: list[dict] = []
    for seed in sorted(completed):
        seed_rows, seed_pairing = completed[seed]
        rows.extend(seed_rows)
        pairing.extend(seed_pairing)
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
        "intervention_weight": WEIGHT,
        "families": list(families),
        "conditions": [c.__dict__ for c in CONDITIONS if c.family in families],
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
