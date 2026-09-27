"""Formal Step 1 topology x kappa x 20-seed sweep after parsing correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
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
    SURVEY_PATH,
    create_simulation,
)
from network_generation import SUPPORTED_TOPOLOGIES
from run_topology_baseline_experiment import cascade_metrics, write_csv


KAPPA_VALUES = tuple(round(value / 10, 1) for value in range(11))
RANDOM_SEEDS = tuple(range(20))
MAX_STEPS = 60
TRAJECTORY_KAPPAS = (0.0, 0.4, 0.7, 1.0)
DEFAULT_OUTPUT_DIR = Path("outputs") / "step1_kappa_sweep_20seeds_corrected_parsing"


def capture_ignition_nodes(simulation) -> np.ndarray:
    """Draw the original KOL seeds once for paired reuse across all kappas."""

    return simulation.rng.choice(
        simulation.kol_idx,
        size=NUM_IGNITE,
        replace=False,
    )


def prepare_vectorized_simulation(simulation) -> tuple:
    """Prepare static arrays without changing the baseline decision semantics."""

    adjacency = nx.to_scipy_sparse_array(
        simulation.graph,
        nodelist=range(simulation.n),
        dtype=np.int32,
        format="csr",
    )
    degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(float)
    poster_mask = simulation.roles == "Poster"
    raw_costs = np.where(
        simulation.roles == "Lurker",
        simulation.c_private,
        simulation.c_public,
    )
    thresholds = (raw_costs + R) / B
    return adjacency, degrees, poster_mask, thresholds


def run_vectorized(
    prepared: tuple,
    n: int,
    ignition_nodes: np.ndarray,
    kappa: float,
    max_steps: int = MAX_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the original synchronous one-way cascade using sparse arithmetic."""

    adjacency, degrees, poster_mask, thresholds = prepared
    state = np.zeros(n, dtype=np.int32)
    state[ignition_nodes] = 1
    counts = [int(state.sum())]

    for _ in range(max_steps):
        active_neighbor_counts = np.asarray(adjacency @ state).ravel()
        local = np.divide(
            active_neighbor_counts,
            degrees,
            out=np.zeros(n, dtype=float),
            where=degrees > 0,
        )
        global_exposure = float(state[poster_mask].mean()) if poster_mask.any() else 0.0
        effective = local.copy()
        effective[poster_mask] = (
            (1 - kappa) * local[poster_mask] + kappa * global_exposure
        )
        new_state = state.copy()
        new_state[(state == 0) & (effective >= thresholds)] = 1
        counts.append(int(new_state.sum()))
        if np.array_equal(new_state, state):
            break
        state = new_state

    return np.asarray(counts, dtype=int), state


def paired_signature(simulation, ignition_nodes: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(simulation.roles.astype("U6").tobytes())
    digest.update(simulation.c_private.tobytes())
    digest.update(simulation.c_public.tobytes())
    digest.update(simulation.kol_idx.tobytes())
    digest.update(np.asarray(ignition_nodes, dtype=np.int64).tobytes())
    return digest.hexdigest()


def run_sweep() -> tuple[list[dict], list[dict], list[dict], dict]:
    run_rows: list[dict] = []
    timestep_rows: list[dict] = []
    pairing_rows: list[dict] = []
    reference_signatures: dict[int, str] = {}
    topology_parameters: dict[str, dict] = {}

    for topology in SUPPORTED_TOPOLOGIES:
        for seed in RANDOM_SEEDS:
            simulation = create_simulation(
                topology=topology,
                n=N,
                target_avg_degree=K,
                network_seed=seed,
                agent_seed=seed,
            )
            ignition_nodes = capture_ignition_nodes(simulation)
            signature = paired_signature(simulation, ignition_nodes)
            if seed not in reference_signatures:
                reference_signatures[seed] = signature
            matches_reference = signature == reference_signatures[seed]
            if not matches_reference:
                raise AssertionError(
                    f"Non-topology randomness is not paired for seed={seed}, topology={topology}"
                )
            pairing_rows.append(
                {
                    "topology": topology,
                    "random_seed": seed,
                    "agent_kol_ignition_sha256": signature,
                    "matches_seed_reference": matches_reference,
                    "initial_kol_seed_nodes": json.dumps(
                        sorted(map(int, ignition_nodes))
                    ),
                }
            )
            topology_parameters[topology] = simulation.base_graph.graph[
                "topology_parameters"
            ]
            prepared = prepare_vectorized_simulation(simulation)

            for kappa in KAPPA_VALUES:
                counts, _ = run_vectorized(
                    prepared,
                    simulation.n,
                    ignition_nodes,
                    kappa,
                )
                proportions = counts / N
                metrics = cascade_metrics(counts, N)
                run_rows.append(
                    {
                        "topology": topology,
                        "kappa": kappa,
                        "random_seed": seed,
                        "final_activated_proportion": metrics[
                            "final_activated_proportion"
                        ],
                        "final_activated_count": metrics["final_activated_count"],
                        "cascade_duration": metrics["total_cascade_duration"],
                        "t50": metrics["t50"],
                    }
                )
                for timestep, (count, proportion) in enumerate(
                    zip(counts, proportions)
                ):
                    timestep_rows.append(
                        {
                            "topology": topology,
                            "kappa": kappa,
                            "random_seed": seed,
                            "timestep": timestep,
                            "activated_count": int(count),
                            "activated_proportion": float(proportion),
                        }
                    )
            print(
                f"Completed topology={topology}, seed={seed} "
                f"({len(run_rows)}/{len(SUPPORTED_TOPOLOGIES) * len(RANDOM_SEEDS) * len(KAPPA_VALUES)} runs)"
            )

    config = {
        "experiment": "Step 1 topology x kappa x 20 random seeds",
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "kappa_values": list(KAPPA_VALUES),
        "random_seeds": list(RANDOM_SEEDS),
        "total_runs": len(SUPPORTED_TOPOLOGIES)
        * len(KAPPA_VALUES)
        * len(RANDOM_SEEDS),
        "N": N,
        "target_average_degree": K,
        "WS_rewiring_probability": P,
        "B": B,
        "R": R,
        "num_kols": NUM_KOLS,
        "kol_reach": KOL_REACH,
        "num_initial_kol_seeds": NUM_IGNITE,
        "max_steps": MAX_STEPS,
        "stopping_condition": "no state changes or max_steps reached",
        "survey_path": str(SURVEY_PATH),
        "topology_parameters": topology_parameters,
        "paired_randomness": {
            "within_seed_across_topologies": "identical roles, threshold samples, KOL identities, KOL target draws, and initial KOL seeds",
            "within_topology_seed_across_kappa": "same graph, agents, thresholds, KOL edges, and initial KOL seeds; only kappa changes",
            "topology_specific_randomness": "base graph edges generated by the selected topology with the same numeric seed",
        },
        "vectorized_runner": "sparse implementation of the unchanged synchronous activation formula; parity-tested against BaselineSimulation.run",
    }
    return run_rows, timestep_rows, pairing_rows, config


def summarize_runs(run_rows: list[dict]) -> list[dict]:
    dataframe = pd.DataFrame(run_rows)
    rows = []
    for (topology, kappa), group in dataframe.groupby(
        ["topology", "kappa"], sort=False
    ):
        final = group["final_activated_proportion"]
        mean = float(final.mean())
        standard_error = float(final.std(ddof=1) / np.sqrt(len(final)))
        rows.append(
            {
                "topology": topology,
                "kappa": kappa,
                "n_runs": len(final),
                "mean_final_cascade_size": mean,
                "median_final_cascade_size": float(final.median()),
                "std_final_cascade_size": float(final.std(ddof=1)),
                "q25_final_cascade_size": float(final.quantile(0.25)),
                "q75_final_cascade_size": float(final.quantile(0.75)),
                "min_final_cascade_size": float(final.min()),
                "max_final_cascade_size": float(final.max()),
                "mean_cascade_duration": float(group["cascade_duration"].mean()),
                "mean_t50": float(group["t50"].mean()),
                "ci95_lower": max(0.0, mean - 1.96 * standard_error),
                "ci95_upper": min(1.0, mean + 1.96 * standard_error),
                "uncertainty_method": "normal-approximation 95% CI: mean +/- 1.96*SEM",
            }
        )
    return rows


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


COLORS = {
    "ER": "#1f77b4",
    "WS": "#ff7f0e",
    "BA": "#2ca02c",
    "SBM": "#d62728",
}


def plot_final_size(summary_rows: list[dict], path: Path) -> None:
    plt = _plot_setup()
    dataframe = pd.DataFrame(summary_rows)
    figure, axis = plt.subplots(figsize=(10, 6.5))
    for topology in SUPPORTED_TOPOLOGIES:
        group = dataframe[dataframe.topology == topology].sort_values("kappa")
        x = group.kappa.to_numpy(float)
        mean = group.mean_final_cascade_size.to_numpy(float)
        lower = group.ci95_lower.to_numpy(float)
        upper = group.ci95_upper.to_numpy(float)
        axis.plot(x, mean, "-o", lw=2.3, label=topology, color=COLORS[topology])
        axis.fill_between(x, lower, upper, color=COLORS[topology], alpha=0.16)
    axis.set_title("Kappa vs Final Cascade Size (20 seeds)", fontsize=15, weight="bold")
    axis.set_xlabel("Global algorithmic exposure strength (kappa)", fontsize=12)
    axis.set_ylabel("Final activated proportion", fontsize=12)
    axis.set_xticks(KAPPA_VALUES)
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(title="Topology")
    axis.text(
        0.01,
        0.98,
        "Band: normal-approximation 95% CI (mean +/- 1.96 x SEM)",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_distributions(run_rows: list[dict], path: Path) -> None:
    plt = _plot_setup()
    dataframe = pd.DataFrame(run_rows)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for axis, topology in zip(axes.ravel(), SUPPORTED_TOPOLOGIES):
        topology_data = dataframe[dataframe.topology == topology]
        values = [
            topology_data[topology_data.kappa == kappa][
                "final_activated_proportion"
            ].to_numpy(float)
            for kappa in KAPPA_VALUES
        ]
        boxes = axis.boxplot(
            values,
            positions=range(len(KAPPA_VALUES)),
            widths=0.65,
            patch_artist=True,
            showfliers=True,
        )
        for box in boxes["boxes"]:
            box.set_facecolor(COLORS[topology])
            box.set_alpha(0.5)
        axis.set_title(topology, fontsize=13, weight="bold")
        axis.set_xticks(range(len(KAPPA_VALUES)))
        axis.set_xticklabels([f"{value:.1f}" for value in KAPPA_VALUES])
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", alpha=0.25)
    figure.supxlabel("Kappa", fontsize=13)
    figure.supylabel("Final activated proportion", fontsize=13)
    figure.suptitle(
        "Final Cascade Size Distribution by Kappa (20 seeds per condition)",
        fontsize=16,
        weight="bold",
    )
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate_forward_filled_trajectories(
    timestep_rows: list[dict],
    run_rows: list[dict],
    horizon: int = MAX_STEPS,
) -> list[dict]:
    """Aggregate cumulative trajectories after carrying each final value forward."""

    timestep_frame = pd.DataFrame(timestep_rows)
    run_frame = pd.DataFrame(run_rows)
    aggregate_rows: list[dict] = []
    key_columns = ["topology", "kappa", "random_seed"]

    if run_frame.duplicated(key_columns).any():
        raise ValueError("Run-level input contains duplicate topology/kappa/seed rows")

    for (topology, kappa), condition_runs in run_frame.groupby(
        ["topology", "kappa"], sort=False
    ):
        histories = []
        for run in condition_runs.itertuples(index=False):
            observed = timestep_frame[
                (timestep_frame.topology == topology)
                & (timestep_frame.kappa == kappa)
                & (timestep_frame.random_seed == run.random_seed)
            ].sort_values("timestep")
            if observed.empty:
                raise ValueError(
                    f"Missing timestep data for {topology}, kappa={kappa}, seed={run.random_seed}"
                )
            if int(observed.timestep.iloc[-1]) != int(run.cascade_duration):
                raise ValueError("Termination timestep does not match cascade_duration")
            if not np.isclose(
                float(observed.activated_proportion.iloc[-1]),
                float(run.final_activated_proportion),
            ):
                raise ValueError("Final timestep does not match run-level final proportion")
            if int(observed.timestep.iloc[-1]) > horizon:
                raise ValueError("Plotting horizon is earlier than a run termination")

            history = np.full(
                horizon + 1,
                float(run.final_activated_proportion),
                dtype=float,
            )
            observed_timesteps = observed.timestep.to_numpy(dtype=int)
            history[observed_timesteps] = observed.activated_proportion.to_numpy(
                dtype=float
            )
            histories.append(history)

        matrix = np.asarray(histories)
        means = matrix.mean(axis=0)
        standard_errors = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
        for timestep in range(horizon + 1):
            aggregate_rows.append(
                {
                    "topology": topology,
                    "kappa": float(kappa),
                    "timestep": timestep,
                    "n_runs": int(matrix.shape[0]),
                    "mean_activated_proportion": float(means[timestep]),
                    "standard_error": float(standard_errors[timestep]),
                    "ci95_lower": max(
                        0.0, float(means[timestep] - 1.96 * standard_errors[timestep])
                    ),
                    "ci95_upper": min(
                        1.0, float(means[timestep] + 1.96 * standard_errors[timestep])
                    ),
                    "aggregation": "final value carried forward through common horizon",
                }
            )
    return aggregate_rows


def plot_trajectories(aggregate_rows: list[dict], path: Path) -> None:
    plt = _plot_setup()
    dataframe = pd.DataFrame(aggregate_rows)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False, sharey=True)
    for axis, kappa in zip(axes.ravel(), TRAJECTORY_KAPPAS):
        for topology in SUPPORTED_TOPOLOGIES:
            subset = dataframe[
                (dataframe.topology == topology) & (dataframe.kappa == kappa)
            ].sort_values("timestep")
            timesteps = subset.timestep.to_numpy(dtype=int)
            mean = subset.mean_activated_proportion.to_numpy(dtype=float)
            lower = subset.ci95_lower.to_numpy(dtype=float)
            upper = subset.ci95_upper.to_numpy(dtype=float)
            axis.plot(
                timesteps,
                mean,
                lw=2.1,
                label=topology,
                color=COLORS[topology],
            )
            axis.fill_between(
                timesteps,
                lower,
                upper,
                color=COLORS[topology],
                alpha=0.14,
            )
        axis.set_title(f"kappa = {kappa:.1f}", fontsize=13, weight="bold")
        axis.set_xlabel("Simulation timestep")
        axis.set_ylabel("Activated proportion")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(title="Topology")
    figure.suptitle(
        "Mean Cascade Trajectories with 95% CI (20 seeds)",
        fontsize=16,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_rows, timestep_rows, pairing_rows, config = run_sweep()
    summary_rows = summarize_runs(run_rows)
    runs_path = args.output_dir / "raw_run_results.csv"
    timesteps_path = args.output_dir / "raw_timestep_results.csv"
    summary_path = args.output_dir / "condition_summary.csv"
    pairing_path = args.output_dir / "paired_randomness_verification.csv"
    config_path = args.output_dir / "experiment_config.json"
    figure1_path = args.output_dir / "figure1_kappa_vs_final_cascade.png"
    figure2_path = args.output_dir / "figure2_final_size_distributions.png"
    figure3_path = args.output_dir / "figure3_mean_cascade_trajectories.png"

    write_csv(runs_path, run_rows)
    write_csv(timesteps_path, timestep_rows)
    write_csv(summary_path, summary_rows)
    write_csv(pairing_path, pairing_rows)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_final_size(summary_rows, figure1_path)
    plot_distributions(run_rows, figure2_path)
    aggregate_rows = aggregate_forward_filled_trajectories(
        timestep_rows, run_rows, horizon=MAX_STEPS
    )
    plot_trajectories(aggregate_rows, figure3_path)
    print(f"Completed {len(run_rows)} runs and {len(timestep_rows)} timestep rows")
    print(f"Run CSV: {runs_path.resolve()}")
    print(f"Timestep CSV: {timesteps_path.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Figure 1: {figure1_path.resolve()}")
    print(f"Figure 2: {figure2_path.resolve()}")
    print(f"Figure 3: {figure3_path.resolve()}")


if __name__ == "__main__":
    main()
