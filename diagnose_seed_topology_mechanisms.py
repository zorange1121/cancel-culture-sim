"""Controlled diagnostics for topology ranking changes and seed sensitivity."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import baseline_simulation as baseline_module
from baseline_simulation import B, K, N, NUM_IGNITE, R, create_simulation
from network_generation import SUPPORTED_TOPOLOGIES


FOCUS_KAPPAS = (0.4, 0.5, 0.6, 0.7)
DIAGNOSTIC_SEEDS = tuple(range(20))
FIXED_SEED = 42
MAX_STEPS = 60
INPUT_PATH = (
    Path("outputs")
    / "step1_kappa_sweep_20seeds_corrected_parsing"
    / "raw_run_results.csv"
)
OUTPUT_DIR = Path("outputs") / "step1_seed_topology_mechanism_diagnostics"
COLORS = {"ER": "#1f77b4", "WS": "#ff7f0e", "BA": "#2ca02c", "SBM": "#d62728"}
SCENARIO_LABELS = {
    "network_only": "Vary network seed",
    "agent_bundle": "Vary agent/threshold/KOL seed",
    "ignition_only": "Vary initial KOL draw",
}

# Survey data and fitted Beta parameters are immutable across every setup.
# Cache this pure calibration step inside each worker without changing values.
baseline_module.load_survey_and_fit = lru_cache(maxsize=1)(
    baseline_module.load_survey_and_fit
)


def prepare(simulation):
    adjacency = nx.to_scipy_sparse_array(
        simulation.graph,
        nodelist=range(simulation.n),
        dtype=np.int32,
        format="csr",
    )
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float)
    poster_mask = simulation.roles == "Poster"
    thresholds = np.where(
        poster_mask, simulation.c_public, simulation.c_private
    )
    thresholds = (thresholds + R) / B
    return adjacency, degrees, poster_mask, thresholds


def run_detailed(prepared, ignition_nodes: np.ndarray, kappa: float) -> dict:
    adjacency, degrees, poster_mask, thresholds = prepared
    n = len(poster_mask)
    state = np.zeros(n, dtype=np.int32)
    state[ignition_nodes] = 1
    history = [int(state.sum())]
    role_history = [(int(state[poster_mask].sum()), int(state[~poster_mask].sum()))]

    for _ in range(MAX_STEPS):
        active_neighbor_counts = np.asarray(adjacency @ state).ravel()
        local = np.divide(
            active_neighbor_counts,
            degrees,
            out=np.zeros(n, dtype=float),
            where=degrees > 0,
        )
        global_exposure = float(state[poster_mask].mean())
        effective = local.copy()
        effective[poster_mask] = (
            (1 - kappa) * local[poster_mask] + kappa * global_exposure
        )
        new_state = state.copy()
        new_state[(state == 0) & (effective >= thresholds)] = 1
        history.append(int(new_state.sum()))
        role_history.append(
            (int(new_state[poster_mask].sum()), int(new_state[~poster_mask].sum()))
        )
        if np.array_equal(new_state, state):
            state = new_state
            break
        state = new_state

    def carried_value(values, timestep):
        return values[min(timestep, len(values) - 1)]

    poster_count = int(poster_mask.sum())
    lurker_count = n - poster_count
    final_posters, final_lurkers = role_history[-1]
    return {
        "final_activated_proportion": history[-1] / n,
        "final_poster_proportion": final_posters / poster_count,
        "final_lurker_proportion": final_lurkers / lurker_count,
        "duration": len(history) - 1,
        "activated_proportion_t1": carried_value(history, 1) / n,
        "activated_proportion_t5": carried_value(history, 5) / n,
        "activated_proportion_t10": carried_value(history, 10) / n,
        "poster_proportion_t10": carried_value(role_history, 10)[0] / poster_count,
        "lurker_proportion_t10": carried_value(role_history, 10)[1] / lurker_count,
    }


def static_diagnostics(simulation, prepared, ignition_nodes: np.ndarray) -> dict:
    adjacency, degrees, poster_mask, thresholds = prepared
    initial_state = np.zeros(simulation.n, dtype=np.int32)
    initial_state[ignition_nodes] = 1
    active_neighbor_counts = np.asarray(adjacency @ initial_state).ravel()
    initial_local = np.divide(
        active_neighbor_counts,
        degrees,
        out=np.zeros(simulation.n, dtype=float),
        where=degrees > 0,
    )
    seeded_neighbor_mask = active_neighbor_counts > 0
    return {
        "poster_count": int(poster_mask.sum()),
        "base_edge_count": int(simulation.base_graph.number_of_edges()),
        "realized_edge_count_after_kol": int(simulation.graph.number_of_edges()),
        "degree_variance_after_kol": float(degrees.var()),
        "seeded_kol_degree_mean": float(degrees[ignition_nodes].mean()),
        "seeded_kol_degree_max": float(degrees[ignition_nodes].max()),
        "initial_one_hop_coverage": int(seeded_neighbor_mask.sum()),
        "initial_one_hop_coverage_proportion": float(seeded_neighbor_mask.mean()),
        "initial_local_exposure_mean": float(initial_local.mean()),
        "poster_threshold_mean": float(thresholds[poster_mask].mean()),
        "poster_threshold_q10": float(np.quantile(thresholds[poster_mask], 0.10)),
        "lurker_threshold_mean": float(thresholds[~poster_mask].mean()),
        "lurker_threshold_q10": float(np.quantile(thresholds[~poster_mask], 0.10)),
    }


def run_one_setup(
    topology: str,
    scenario: str,
    varying_seed: int,
) -> list[dict]:
    if scenario == "network_only":
        network_seed, agent_seed = varying_seed, FIXED_SEED
    elif scenario == "agent_bundle":
        network_seed, agent_seed = FIXED_SEED, varying_seed
    else:
        raise ValueError(scenario)

    simulation = create_simulation(
        topology=topology,
        n=N,
        target_avg_degree=K,
        network_seed=network_seed,
        agent_seed=agent_seed,
    )
    ignition_nodes = simulation.rng.choice(
        simulation.kol_idx, size=NUM_IGNITE, replace=False
    )
    prepared = prepare(simulation)
    diagnostics = static_diagnostics(simulation, prepared, ignition_nodes)
    rows = []
    for kappa in FOCUS_KAPPAS:
        rows.append(
            {
                "scenario": scenario,
                "topology": topology,
                "varying_seed": varying_seed,
                "network_seed": network_seed,
                "agent_seed": agent_seed,
                "ignition_seed": agent_seed,
                "kappa": kappa,
                **diagnostics,
                **run_detailed(prepared, ignition_nodes, kappa),
            }
        )
    return rows


def run_ignition_only(topology: str) -> list[dict]:
    simulation = create_simulation(
        topology=topology,
        n=N,
        target_avg_degree=K,
        network_seed=FIXED_SEED,
        agent_seed=FIXED_SEED,
    )
    prepared = prepare(simulation)
    rows = []
    for ignition_seed in DIAGNOSTIC_SEEDS:
        ignition_nodes = np.random.default_rng(ignition_seed).choice(
            simulation.kol_idx, size=NUM_IGNITE, replace=False
        )
        diagnostics = static_diagnostics(simulation, prepared, ignition_nodes)
        for kappa in FOCUS_KAPPAS:
            rows.append(
                {
                    "scenario": "ignition_only",
                    "topology": topology,
                    "varying_seed": ignition_seed,
                    "network_seed": FIXED_SEED,
                    "agent_seed": FIXED_SEED,
                    "ignition_seed": ignition_seed,
                    "kappa": kappa,
                    **diagnostics,
                    **run_detailed(prepared, ignition_nodes, kappa),
                }
            )
    return rows


def run_controlled_experiment() -> pd.DataFrame:
    rows = []
    setup_tasks = [
        (topology, scenario, seed)
        for topology in SUPPORTED_TOPOLOGIES
        for scenario in ("network_only", "agent_bundle")
        for seed in DIAGNOSTIC_SEEDS
    ]
    ignition_tasks = list(SUPPORTED_TOPOLOGIES)
    total_setups = len(setup_tasks) + len(ignition_tasks)
    completed = 0
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_one_setup, topology, scenario, seed): (
                topology,
                scenario,
                seed,
            )
            for topology, scenario, seed in setup_tasks
        }
        futures.update(
            {
                executor.submit(run_ignition_only, topology): (
                    topology,
                    "ignition_only",
                    None,
                )
                for topology in ignition_tasks
            }
        )
        for future in as_completed(futures):
            topology, scenario, seed = futures[future]
            rows.extend(future.result())
            completed += 1
            suffix = "" if seed is None else f" seed={seed}"
            print(
                f"Completed {completed}/{total_setups} setups: "
                f"{topology} {scenario}{suffix}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_variance(controlled: pd.DataFrame) -> pd.DataFrame:
    summary = (
        controlled.groupby(["scenario", "topology", "kappa"])
        .final_activated_proportion.agg(
            n="size",
            mean="mean",
            median="median",
            standard_deviation="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )
    return summary


def variance_source_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(
        index=["topology", "kappa"],
        columns="scenario",
        values="standard_deviation",
    ).reset_index()
    wide["agent_to_network_sd_ratio"] = wide.agent_bundle / wide.network_only
    wide["ignition_to_network_sd_ratio"] = wide.ignition_only / wide.network_only
    return wide


def bootstrap_topology_winners(existing: pd.DataFrame, replicates: int = 20_000):
    rng = np.random.default_rng(20260922)
    probability_rows = []
    seed_winner_rows = []
    for kappa, group in existing.groupby("kappa", sort=True):
        matrix = group.pivot(
            index="random_seed",
            columns="topology",
            values="final_activated_proportion",
        )[list(SUPPORTED_TOPOLOGIES)].to_numpy(float)
        sampled_indices = rng.integers(0, matrix.shape[0], size=(replicates, matrix.shape[0]))
        bootstrap_means = matrix[sampled_indices].mean(axis=1)
        bootstrap_is_winner = np.isclose(
            bootstrap_means,
            bootstrap_means.max(axis=1, keepdims=True),
            atol=1e-12,
        )
        bootstrap_weights = bootstrap_is_winner / bootstrap_is_winner.sum(
            axis=1, keepdims=True
        )
        single_seed_is_winner = np.isclose(
            matrix,
            matrix.max(axis=1, keepdims=True),
            atol=1e-12,
        )
        single_seed_weights = single_seed_is_winner / single_seed_is_winner.sum(
            axis=1, keepdims=True
        )
        for index, topology in enumerate(SUPPORTED_TOPOLOGIES):
            probability_rows.append(
                {
                    "kappa": float(kappa),
                    "topology": topology,
                    "bootstrap_probability_of_highest_mean": float(
                        bootstrap_weights[:, index].mean()
                    ),
                    "single_seed_winner_fraction": float(
                        single_seed_weights[:, index].mean()
                    ),
                    "tie_handling": "ties split equally among tied topologies",
                }
            )
        for seed, winner_mask in zip(
            sorted(group.random_seed.unique()), single_seed_is_winner
        ):
            seed_winner_rows.append(
                {
                    "kappa": float(kappa),
                    "random_seed": int(seed),
                    "winning_topology": "|".join(
                        topology
                        for topology, is_winner in zip(SUPPORTED_TOPOLOGIES, winner_mask)
                        if is_winner
                    ),
                }
            )
    return pd.DataFrame(probability_rows), pd.DataFrame(seed_winner_rows)


def mechanism_correlations(controlled: pd.DataFrame) -> pd.DataFrame:
    predictors = (
        "activated_proportion_t1",
        "activated_proportion_t5",
        "activated_proportion_t10",
        "seeded_kol_degree_mean",
        "initial_one_hop_coverage_proportion",
        "poster_threshold_mean",
        "poster_threshold_q10",
        "lurker_threshold_mean",
        "lurker_threshold_q10",
    )
    rows = []
    for keys, group in controlled.groupby(["scenario", "topology", "kappa"]):
        for predictor in predictors:
            rows.append(
                {
                    "scenario": keys[0],
                    "topology": keys[1],
                    "kappa": keys[2],
                    "predictor": predictor,
                    "pearson_correlation_with_final": group[predictor].corr(
                        group.final_activated_proportion, method="pearson"
                    ),
                    "spearman_correlation_with_final": group[predictor].corr(
                        group.final_activated_proportion, method="spearman"
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_variance_sources(comparison: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    styles = {
        "network_only": ("o-", "#1f77b4"),
        "agent_bundle": ("s-", "#d62728"),
        "ignition_only": ("^-", "#2ca02c"),
    }
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        group = comparison[comparison.topology == topology].sort_values("kappa")
        for scenario, (style, color) in styles.items():
            axis.plot(
                group.kappa,
                group[scenario],
                style,
                color=color,
                lw=2,
                label=SCENARIO_LABELS[scenario],
            )
        axis.set_title(topology, weight="bold")
        axis.text(
            0.02,
            0.96,
            topology,
            transform=axis.transAxes,
            va="top",
            weight="bold",
            fontsize=12,
        )
        axis.set_xlabel("Kappa")
        axis.set_ylabel("SD of final cascade size")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Controlled Seed-Source Variance", fontsize=16, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_role_outcomes(controlled: pd.DataFrame, path: Path) -> None:
    subset = controlled[controlled.scenario == "agent_bundle"]
    means = (
        subset.groupby(["topology", "kappa"])[
            ["final_poster_proportion", "final_lurker_proportion"]
        ]
        .mean()
        .reset_index()
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        group = means[means.topology == topology]
        axis.plot(group.kappa, group.final_poster_proportion, "o-", lw=2, label="Posters")
        axis.plot(group.kappa, group.final_lurker_proportion, "s-", lw=2, label="Lurkers")
        axis.set_title(topology, weight="bold")
        axis.text(
            0.02,
            0.96,
            topology,
            transform=axis.transAxes,
            va="top",
            weight="bold",
            fontsize=12,
        )
        axis.set_xlabel("Kappa")
        axis.set_ylabel("Mean final activated proportion by role")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Role-Specific Response to Kappa", fontsize=16, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_early_tipping(controlled: pd.DataFrame, path: Path) -> None:
    subset = controlled[controlled.scenario == "agent_bundle"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        group = subset[subset.topology == topology]
        for kappa in FOCUS_KAPPAS:
            values = group[group.kappa == kappa]
            axis.scatter(
                values.activated_proportion_t10,
                values.final_activated_proportion,
                label=f"κ={kappa:.1f}",
                alpha=0.75,
                s=35,
            )
        axis.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axis.set_title(topology, weight="bold")
        axis.set_xlabel("Activated proportion at timestep 10")
        axis.set_ylabel("Final activated proportion")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Early Activation Predicts the Final Cascade Regime", fontsize=16, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_winner_probabilities(probabilities: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for topology in SUPPORTED_TOPOLOGIES:
        group = probabilities[probabilities.topology == topology].sort_values("kappa")
        axis.plot(
            group.kappa,
            group.bootstrap_probability_of_highest_mean,
            "o-",
            color=COLORS[topology],
            lw=2.2,
            label=topology,
        )
    axis.set_xlabel("Kappa")
    axis.set_ylabel("Bootstrap probability of highest topology mean")
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(title="Topology")
    axis.set_title("Uncertainty in the Leading Topology (paired-seed bootstrap)", weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-controlled",
        action="store_true",
        help="Reuse the existing controlled raw CSV and only rebuild analyses/figures",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(INPUT_PATH)
    controlled_path = OUTPUT_DIR / "controlled_seed_decomposition_raw.csv"
    if args.reuse_controlled:
        controlled = pd.read_csv(controlled_path)
        print(
            f"Reusing {len(controlled)} controlled rows; no simulations executed",
            flush=True,
        )
    else:
        controlled = run_controlled_experiment()
    summary = summarize_variance(controlled)
    comparison = variance_source_comparison(summary)
    winner_probabilities, seed_winners = bootstrap_topology_winners(existing)
    correlations = mechanism_correlations(controlled)

    controlled.to_csv(controlled_path, index=False)
    summary.to_csv(OUTPUT_DIR / "controlled_seed_decomposition_summary.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "variance_source_comparison.csv", index=False)
    winner_probabilities.to_csv(OUTPUT_DIR / "bootstrap_topology_winner_probabilities.csv", index=False)
    seed_winners.to_csv(OUTPUT_DIR / "single_seed_topology_winners.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "mechanism_correlations.csv", index=False)

    plot_variance_sources(comparison, OUTPUT_DIR / "figure_1_variance_sources.png")
    plot_role_outcomes(controlled, OUTPUT_DIR / "figure_2_role_specific_kappa_response.png")
    plot_early_tipping(controlled, OUTPUT_DIR / "figure_3_early_tipping_prediction.png")
    plot_winner_probabilities(
        winner_probabilities, OUTPUT_DIR / "figure_4_topology_winner_uncertainty.png"
    )

    config = {
        "input": str(INPUT_PATH.resolve()),
        "simulation_parameters_changed": False,
        "N": N,
        "target_average_degree": K,
        "kappas": list(FOCUS_KAPPAS),
        "diagnostic_seeds": list(DIAGNOSTIC_SEEDS),
        "fixed_seed": FIXED_SEED,
        "scenarios": {
            "network_only": "vary network_seed 0-19; hold agent_seed=42",
            "agent_bundle": "vary agent_seed 0-19; hold network_seed=42; includes population, thresholds, KOL identities/targets, and baseline ignition draw",
            "ignition_only": "hold network_seed=agent_seed=42; redraw 5 initial KOLs with ignition seed 0-19",
        },
        "controlled_runs": len(controlled),
        "bootstrap_replicates": 20_000,
    }
    (OUTPUT_DIR / "diagnostic_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Completed {len(controlled)} controlled diagnostic runs", flush=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
