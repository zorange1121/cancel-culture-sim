"""Diagnose whether global kappa exposure masks Step 1 topology effects."""

from __future__ import annotations

import argparse
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
from run_topology_baseline_experiment import cascade_metrics, write_csv


KAPPA_VALUES = (0.0, 0.4, 0.7)
MAX_STEPS = 60
DEFAULT_OUTPUT_DIR = Path("outputs") / "step1_kappa_diagnostic_seed42"


def run_diagnostic(
    seed: int = RANDOM_SEED,
    n: int = N,
    target_avg_degree: float = K,
) -> tuple[list[dict], list[dict], dict[float, dict[str, list[float]]], dict]:
    """Run the 12 conditions while holding every non-kappa setting fixed."""

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    histories: dict[float, dict[str, list[float]]] = {}
    topology_parameters: dict[str, dict] = {}

    for kappa in KAPPA_VALUES:
        histories[kappa] = {}
        for topology in SUPPORTED_TOPOLOGIES:
            simulation = create_simulation(
                topology=topology,
                n=n,
                target_avg_degree=target_avg_degree,
                network_seed=seed,
                agent_seed=seed,
            )
            history, _ = simulation.run(
                kappa=kappa,
                benefit=B,
                outside_payoff=R,
                max_steps=MAX_STEPS,
                num_ignite=NUM_IGNITE,
            )
            activated_counts = np.rint(history * n).astype(int)
            histories[kappa][topology] = history.tolist()
            topology_parameters[topology] = simulation.base_graph.graph["topology_parameters"]

            for timestep, (count, proportion) in enumerate(zip(activated_counts, history)):
                raw_rows.append(
                    {
                        "topology": topology,
                        "kappa": kappa,
                        "random_seed": seed,
                        "timestep": timestep,
                        "activated_count": int(count),
                        "activated_proportion": float(proportion),
                    }
                )

            summary_rows.append(
                {
                    "topology": topology,
                    "kappa": kappa,
                    "random_seed": seed,
                    **cascade_metrics(activated_counts, n),
                }
            )

    config = {
        "experiment": "Step 1 topology by kappa diagnostic",
        "topologies": list(SUPPORTED_TOPOLOGIES),
        "kappa_values": list(KAPPA_VALUES),
        "N": n,
        "target_average_degree": target_avg_degree,
        "WS_rewiring_probability": P,
        "B": B,
        "R": R,
        "num_kols": NUM_KOLS,
        "kol_reach": KOL_REACH,
        "num_initial_seeds": NUM_IGNITE,
        "network_seed": seed,
        "agent_seed": seed,
        "random_seed_protocol": "reinitialize every condition with network_seed=agent_seed=42; use the first identical KOL ignition draw",
        "stopping_condition": "no state changes or max_steps reached",
        "max_steps": MAX_STEPS,
        "survey_path": str(SURVEY_PATH),
        "topology_parameters": topology_parameters,
        "only_varied_parameter": "kappa",
    }
    return raw_rows, summary_rows, histories, config


def plot_kappa_curves(
    output_dir: Path,
    histories: dict[float, dict[str, list[float]]],
    seed: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "ER": {"color": "#1f77b4", "marker": "o"},
        "WS": {"color": "#ff7f0e", "marker": "s"},
        "BA": {"color": "#2ca02c", "marker": "^"},
        "SBM": {"color": "#d62728", "marker": "D"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kappa in KAPPA_VALUES:
        figure, axis = plt.subplots(figsize=(10, 6.5))
        for topology in SUPPORTED_TOPOLOGIES:
            values = histories[kappa][topology]
            axis.plot(
                range(len(values)),
                values,
                linewidth=2.3,
                markersize=6,
                label=topology,
                **styles[topology],
            )
        axis.set_title(
            f"Cascade Dynamics by Network Topology (kappa={kappa:.1f}, seed={seed})",
            fontsize=15,
            weight="bold",
        )
        axis.set_xlabel("Simulation timestep", fontsize=12)
        axis.set_ylabel("Activated proportion", fontsize=12)
        axis.set_ylim(0, 1.03)
        axis.set_xlim(left=0)
        axis.grid(alpha=0.25)
        axis.legend(title="Topology", frameon=True)
        figure.tight_layout()
        path = output_dir / f"topology_cascade_curves_kappa_{kappa:.1f}_seed{seed}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def print_summary(rows: list[dict]) -> None:
    header = f"{'Topology':<10} {'kappa':>7} {'Final':>10} {'t50':>6} {'Duration':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['topology']:<10} {row['kappa']:>7.1f} "
            f"{row['final_activated_proportion']:>10.4f} {row['t50']:>6} "
            f"{row['total_cascade_duration']:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    raw_rows, summary_rows, histories, config = run_diagnostic(seed=args.seed)
    raw_path = args.output_dir / f"kappa_topology_raw_timesteps_seed{args.seed}.csv"
    summary_path = args.output_dir / f"kappa_topology_summary_seed{args.seed}.csv"
    config_path = args.output_dir / f"kappa_topology_config_seed{args.seed}.json"
    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary_rows)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_paths = plot_kappa_curves(args.output_dir, histories, args.seed)

    print_summary(summary_rows)
    for path in figure_paths:
        print(f"Figure: {path.resolve()}")
    print(f"Raw CSV: {raw_path.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Config JSON: {config_path.resolve()}")


if __name__ == "__main__":
    main()
