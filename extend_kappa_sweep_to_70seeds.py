"""Extend the unchanged topology x kappa baseline from 20 to 70 paired seeds."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

import baseline_simulation as baseline_module
from baseline_simulation import K, N, create_simulation
from network_generation import SUPPORTED_TOPOLOGIES
from run_kappa_sweep_multiseed import (
    KAPPA_VALUES,
    capture_ignition_nodes,
    paired_signature,
    prepare_vectorized_simulation,
    run_vectorized,
)
from run_topology_baseline_experiment import cascade_metrics


NEW_SEEDS = tuple(range(20, 70))
EXISTING_DIR = Path("outputs") / "step1_kappa_sweep_20seeds_corrected_parsing"
OUTPUT_DIR = Path("outputs") / "step1_kappa_sweep_70seeds"

baseline_module.load_survey_and_fit = lru_cache(maxsize=1)(
    baseline_module.load_survey_and_fit
)


def run_setup(topology: str, seed: int) -> tuple[list[dict], list[dict], dict]:
    simulation = create_simulation(
        topology=topology,
        n=N,
        target_avg_degree=K,
        network_seed=seed,
        agent_seed=seed,
    )
    ignition_nodes = capture_ignition_nodes(simulation)
    signature = paired_signature(simulation, ignition_nodes)
    prepared = prepare_vectorized_simulation(simulation)
    run_rows = []
    timestep_rows = []
    for kappa in KAPPA_VALUES:
        counts, _ = run_vectorized(
            prepared, simulation.n, ignition_nodes, kappa
        )
        metrics = cascade_metrics(counts, simulation.n)
        run_rows.append(
            {
                "topology": topology,
                "kappa": kappa,
                "random_seed": seed,
                "final_activated_proportion": metrics["final_activated_proportion"],
                "final_activated_count": metrics["final_activated_count"],
                "cascade_duration": metrics["total_cascade_duration"],
                "t50": metrics["t50"],
            }
        )
        for timestep, count in enumerate(counts):
            timestep_rows.append(
                {
                    "topology": topology,
                    "kappa": kappa,
                    "random_seed": seed,
                    "timestep": timestep,
                    "activated_count": int(count),
                    "activated_proportion": float(count / simulation.n),
                }
            )
    pairing = {
        "topology": topology,
        "random_seed": seed,
        "agent_kol_ignition_sha256": signature,
        "initial_kol_seed_nodes": json.dumps(sorted(map(int, ignition_nodes))),
    }
    return run_rows, timestep_rows, pairing


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(topology, seed) for topology in SUPPORTED_TOPOLOGIES for seed in NEW_SEEDS]
    run_rows = []
    timestep_rows = []
    pairing_rows = []
    completed = 0
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_setup, topology, seed): (topology, seed)
            for topology, seed in tasks
        }
        for future in as_completed(futures):
            topology, seed = futures[future]
            runs, timesteps, pairing = future.result()
            run_rows.extend(runs)
            timestep_rows.extend(timesteps)
            pairing_rows.append(pairing)
            completed += 1
            print(
                f"Completed {completed}/{len(tasks)} setups: {topology} seed={seed}",
                flush=True,
            )

    new_runs = pd.DataFrame(run_rows).sort_values(
        ["topology", "random_seed", "kappa"]
    )
    new_timesteps = pd.DataFrame(timestep_rows).sort_values(
        ["topology", "random_seed", "kappa", "timestep"]
    )
    pairing = pd.DataFrame(pairing_rows).sort_values(["random_seed", "topology"])
    signature_counts = pairing.groupby("random_seed").agent_kol_ignition_sha256.nunique()
    if not signature_counts.eq(1).all():
        raise AssertionError("Non-topology randomness is not paired across topologies")
    pairing["matches_seed_reference"] = True

    old_runs = pd.read_csv(EXISTING_DIR / "raw_run_results.csv")
    old_timesteps = pd.read_csv(EXISTING_DIR / "raw_timestep_results.csv")
    combined_runs = pd.concat([old_runs, new_runs], ignore_index=True).sort_values(
        ["topology", "random_seed", "kappa"]
    )
    combined_timesteps = pd.concat(
        [old_timesteps, new_timesteps], ignore_index=True
    ).sort_values(["topology", "random_seed", "kappa", "timestep"])

    if len(new_runs) != 4 * 11 * 50:
        raise AssertionError("Additional run count is incomplete")
    if len(combined_runs) != 4 * 11 * 70:
        raise AssertionError("Combined run count is incomplete")
    if set(combined_runs.random_seed) != set(range(70)):
        raise AssertionError("Combined seed list is not 0 through 69")
    if set(combined_runs.groupby(["topology", "kappa"]).size()) != {70}:
        raise AssertionError("Every combined condition must contain 70 seeds")

    new_runs.to_csv(OUTPUT_DIR / "additional_raw_run_results_seeds20_69.csv", index=False)
    new_timesteps.to_csv(
        OUTPUT_DIR / "additional_raw_timestep_results_seeds20_69.csv", index=False
    )
    combined_runs.to_csv(OUTPUT_DIR / "combined_raw_run_results_70seeds.csv", index=False)
    combined_timesteps.to_csv(
        OUTPUT_DIR / "combined_raw_timestep_results_70seeds.csv", index=False
    )
    pairing.to_csv(
        OUTPUT_DIR / "additional_paired_randomness_verification.csv", index=False
    )
    config = {
        "experiment": "extension of unchanged Step 1 topology x kappa baseline to 70 paired seeds",
        "existing_seeds_reused": list(range(20)),
        "additional_seeds_run": list(NEW_SEEDS),
        "combined_seeds": list(range(70)),
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "kappa_values": list(KAPPA_VALUES),
        "additional_runs": len(new_runs),
        "combined_runs": len(combined_runs),
        "simulation_parameters_changed": False,
        "paired_protocol": "network_seed=agent_seed=random_seed; identical agent/KOL/ignition realization across topologies for each numeric seed",
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Additional runs: {len(new_runs)}", flush=True)
    print(f"Combined runs: {len(combined_runs)}", flush=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
