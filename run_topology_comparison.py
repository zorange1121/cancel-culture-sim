"""Generate all Step 1 topologies, report statistics, and smoke-test cascades."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline_simulation import B, K, N, R, RANDOM_SEED, create_simulation
from network_generation import SUPPORTED_TOPOLOGIES, network_statistics


def run_comparison(
    n: int = N,
    target_avg_degree: float = K,
    seed: int = RANDOM_SEED,
    kappa: float = 0.6,
) -> list[dict[str, int | float | str | bool]]:
    rows = []
    for topology in SUPPORTED_TOPOLOGIES:
        simulation = create_simulation(
            topology=topology,
            n=n,
            target_avg_degree=target_avg_degree,
            network_seed=seed,
            agent_seed=seed,
        )
        row = network_statistics(simulation.base_graph)
        row["topology_parameters"] = json.dumps(
            simulation.base_graph.graph["topology_parameters"],
            ensure_ascii=False,
            sort_keys=True,
        )
        history, final_state = simulation.run(kappa=kappa, benefit=B, outside_payoff=R)
        row.update(
            random_seed=seed,
            target_average_degree=target_avg_degree,
            simulation_completed=True,
            simulation_steps=len(history) - 1,
            final_activated_proportion=float(final_state.mean()),
        )
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    header = f"{'Topology':<10} {'N':>7} {'Edges':>8} {'Avg Degree':>12} {'Degree Var':>12} {'Clustering':>12} {'Steps':>7} {'Final':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['topology']:<10} {row['n']:>7} {row['edges']:>8} "
            f"{row['average_degree']:>12.4f} {row['degree_variance']:>12.4f} "
            f"{row['average_clustering']:>12.4f} {row['simulation_steps']:>7} "
            f"{row['final_activated_proportion']:>9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--target-avg-degree", type=float, default=K)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--kappa", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=Path("topology_summary.csv"))
    args = parser.parse_args()

    rows = run_comparison(args.n, args.target_avg_degree, args.seed, args.kappa)
    _print_table(rows)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
