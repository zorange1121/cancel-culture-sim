"""Run the formal Step 1 cascade baseline across ER, WS, BA, and SBM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

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
)
from network_generation import SUPPORTED_TOPOLOGIES


DEFAULT_KAPPA = 0.6
DEFAULT_MAX_STEPS = 60
DEFAULT_OUTPUT_DIR = Path("outputs") / "step1_topology_baseline"


def cascade_metrics(activated_counts: np.ndarray, n: int) -> dict[str, int | float]:
    """Calculate final cascade size, t50, and the stopping timestep."""

    final_count = int(activated_counts[-1])
    t50_candidates = np.flatnonzero(activated_counts >= 0.5 * final_count)
    t50 = int(t50_candidates[0]) if len(t50_candidates) else -1
    return {
        "final_activated_count": final_count,
        "final_activated_proportion": final_count / n,
        "t50": t50,
        "total_cascade_duration": len(activated_counts) - 1,
    }


def run_experiment(
    n: int = N,
    target_avg_degree: float = K,
    seed: int = RANDOM_SEED,
    kappa: float = DEFAULT_KAPPA,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[list[dict], list[dict], dict[str, list[float]], dict]:
    """Run one identically configured cascade for each topology."""

    timestep_rows: list[dict] = []
    metric_rows: list[dict] = []
    histories: dict[str, list[float]] = {}
    topology_parameters: dict[str, dict] = {}

    for topology in SUPPORTED_TOPOLOGIES:
        simulation = create_simulation(
            topology=topology,
            n=n,
            target_avg_degree=target_avg_degree,
            network_seed=seed,
            agent_seed=seed,
        )
        history, final_state = simulation.run(
            kappa=kappa,
            benefit=B,
            outside_payoff=R,
            max_steps=max_steps,
            num_ignite=NUM_IGNITE,
        )
        activated_counts = np.rint(history * n).astype(int)
        histories[topology] = history.tolist()
        topology_parameters[topology] = simulation.base_graph.graph["topology_parameters"]

        for timestep, (count, proportion) in enumerate(zip(activated_counts, history)):
            timestep_rows.append(
                {
                    "topology": topology,
                    "random_seed": seed,
                    "timestep": timestep,
                    "activated_count": int(count),
                    "activated_proportion": float(proportion),
                }
            )

        metrics = cascade_metrics(activated_counts, n)
        metric_rows.append(
            {
                "topology": topology,
                "random_seed": seed,
                **metrics,
            }
        )

    config = {
        "experiment": "Step 1 formal topology baseline",
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "N": n,
        "target_average_degree": target_avg_degree,
        "WS_rewiring_probability": P,
        "B": B,
        "R": R,
        "kappa": kappa,
        "num_kols": NUM_KOLS,
        "kol_reach": KOL_REACH,
        "num_initial_seeds": NUM_IGNITE,
        "network_seed": seed,
        "agent_seed": seed,
        "initial_seed_protocol": "first seeded run from identical per-topology agent RNG state; choose NUM_IGNITE KOLs without replacement",
        "stopping_condition": "no state changes or max_steps reached",
        "max_steps": max_steps,
        "survey_path": str(SURVEY_PATH),
        "topology_parameters": topology_parameters,
    }
    return timestep_rows, metric_rows, histories, config


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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

    axis.set_title(f"Cascade Dynamics by Network Topology (seed={seed})", fontsize=15, weight="bold")
    axis.set_xlabel("Simulation timestep", fontsize=12)
    axis.set_ylabel("Activated proportion", fontsize=12)
    axis.set_ylim(0, 1.03)
    axis.set_xlim(left=0)
    axis.grid(alpha=0.25)
    axis.legend(title="Topology", frameon=True)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_metrics(rows: list[dict]) -> None:
    header = f"{'Topology':<10} {'Final count':>12} {'Final proportion':>18} {'t50':>6} {'Duration':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['topology']:<10} {row['final_activated_count']:>12} "
            f"{row['final_activated_proportion']:>18.4f} {row['t50']:>6} "
            f"{row['total_cascade_duration']:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    timestep_rows, metric_rows, histories, config = run_experiment(seed=args.seed)
    curves_path = args.output_dir / f"topology_cascade_curves_seed{args.seed}.png"
    timesteps_path = args.output_dir / f"topology_cascade_timesteps_seed{args.seed}.csv"
    metrics_path = args.output_dir / f"topology_cascade_metrics_seed{args.seed}.csv"
    config_path = args.output_dir / f"topology_baseline_config_seed{args.seed}.json"

    write_csv(timesteps_path, timestep_rows)
    write_csv(metrics_path, metric_rows)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_histories(curves_path, histories, args.seed)
    print_metrics(metric_rows)
    print(f"Figure: {curves_path.resolve()}")
    print(f"Timestep CSV: {timesteps_path.resolve()}")
    print(f"Metrics CSV: {metrics_path.resolve()}")
    print(f"Config JSON: {config_path.resolve()}")


if __name__ == "__main__":
    main()
