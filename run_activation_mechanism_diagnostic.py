"""Inspect why the unchanged baseline cascades rapidly when kappa is zero."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    R,
    RANDOM_SEED,
    create_simulation,
)
from network_generation import SUPPORTED_TOPOLOGIES


KAPPA = 0.0
MAX_STEPS = 60
DEFAULT_OUTPUT_DIR = Path("outputs") / "activation_mechanism_diagnostic_kappa0_seed42"
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def _edge_set(graph) -> set[tuple[int, int]]:
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()}


def _edge_hash(edges: set[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for u, v in sorted(edges):
        digest.update(f"{u},{v}\n".encode("ascii"))
    return digest.hexdigest()


def _distribution_row(group: str, raw_costs: np.ndarray, thresholds: np.ndarray) -> dict:
    row: dict[str, int | float | str] = {
        "group": group,
        "count": int(len(thresholds)),
        "raw_cost_mean": float(raw_costs.mean()),
        "raw_cost_median": float(np.median(raw_costs)),
        "raw_cost_min": float(raw_costs.min()),
        "raw_cost_max": float(raw_costs.max()),
        "threshold_mean": float(thresholds.mean()),
        "threshold_median": float(np.median(thresholds)),
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
    }
    for quantile, raw_value, threshold_value in zip(
        QUANTILES,
        np.quantile(raw_costs, QUANTILES),
        np.quantile(thresholds, QUANTILES),
    ):
        label = f"q{int(quantile * 100):02d}"
        row[f"raw_cost_{label}"] = float(raw_value)
        row[f"threshold_{label}"] = float(threshold_value)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostic(seed: int = RANDOM_SEED) -> dict[str, list[dict] | dict]:
    threshold_rows: list[dict] = []
    timestep_rows: list[dict] = []
    agent_rows: list[dict] = []
    graph_rows: list[dict] = []
    topology_reports: dict[str, dict] = {}
    reference_threshold_bytes: bytes | None = None
    reference_seed_nodes: list[int] | None = None

    for topology in SUPPORTED_TOPOLOGIES:
        simulation = create_simulation(
            topology=topology,
            n=N,
            target_avg_degree=K,
            network_seed=seed,
            agent_seed=seed,
        )
        poster_mask = simulation.roles == "Poster"
        kol_mask = np.zeros(N, dtype=bool)
        kol_mask[simulation.kol_idx] = True
        raw_costs = np.where(
            simulation.roles == "Lurker",
            simulation.c_private,
            simulation.c_public,
        )
        thresholds = (raw_costs + R) / B

        threshold_bytes = thresholds.tobytes()
        if reference_threshold_bytes is None:
            reference_threshold_bytes = threshold_bytes
            groups = {
                "all_agents": np.ones(N, dtype=bool),
                "lurkers": simulation.roles == "Lurker",
                "posters": poster_mask,
                "kols": kol_mask,
            }
            for group_name, mask in groups.items():
                threshold_rows.append(
                    _distribution_row(group_name, raw_costs[mask], thresholds[mask])
                )
        thresholds_match_reference = threshold_bytes == reference_threshold_bytes

        base_edges = _edge_set(simulation.base_graph)
        full_edges = _edge_set(simulation.graph)
        extra_edges = full_edges - base_edges
        extra_neighbors = [set() for _ in range(N)]
        for u, v in extra_edges:
            extra_neighbors[u].add(v)
            extra_neighbors[v].add(u)

        neighbor_cache_matches = all(
            set(map(int, simulation.neighbors[node]))
            == set(map(int, simulation.graph.neighbors(node)))
            for node in range(N)
        )

        state = np.zeros(N, dtype=int)
        seed_nodes = sorted(
            map(
                int,
                simulation.rng.choice(
                    simulation.kol_idx,
                    size=NUM_IGNITE,
                    replace=False,
                ),
            )
        )
        state[seed_nodes] = 1
        if reference_seed_nodes is None:
            reference_seed_nodes = seed_nodes
        seeds_match_reference = seed_nodes == reference_seed_nodes

        seed_set = set(seed_nodes)
        seed_extra_targets = set()
        for seed_node in seed_nodes:
            seed_extra_targets.update(extra_neighbors[seed_node])
        seed_extra_targets.difference_update(seed_set)

        max_phi_local_difference = 0.0
        for timestep in range(1, MAX_STEPS + 1):
            active_before = int(state.sum())
            local = np.zeros(N, dtype=float)
            active_neighbor_counts = np.zeros(N, dtype=int)
            degrees = np.fromiter(
                (len(neighbors) for neighbors in simulation.neighbors),
                dtype=int,
                count=N,
            )
            for node, neighbors in enumerate(simulation.neighbors):
                if len(neighbors):
                    active_neighbor_counts[node] = int(state[neighbors].sum())
                    local[node] = active_neighbor_counts[node] / len(neighbors)

            global_exposure = float(state[poster_mask].mean())
            effective_exposure = np.where(
                poster_mask,
                (1 - KAPPA) * local + KAPPA * global_exposure,
                local,
            )
            max_phi_local_difference = max(
                max_phi_local_difference,
                float(np.max(np.abs(effective_exposure - local))),
            )

            calculated_new_state = state.copy()
            calculated_new_state[(state == 0) & (effective_exposure >= thresholds)] = 1
            model_new_state = simulation.step(state, KAPPA, B, R)
            if not np.array_equal(calculated_new_state, model_new_state):
                raise AssertionError("Instrumented activation decisions differ from model.step")

            newly_mask = (state == 0) & (model_new_state == 1)
            newly_nodes = np.flatnonzero(newly_mask)
            base_active_counts = np.zeros(len(newly_nodes), dtype=int)
            extra_active_counts = np.zeros(len(newly_nodes), dtype=int)
            direct_seed_extra = np.zeros(len(newly_nodes), dtype=bool)
            for index, node in enumerate(newly_nodes):
                base_active_counts[index] = sum(
                    state[neighbor] for neighbor in simulation.base_graph.neighbors(int(node))
                )
                extra_active_counts[index] = sum(
                    state[neighbor] for neighbor in extra_neighbors[int(node)]
                )
                direct_seed_extra[index] = bool(extra_neighbors[int(node)] & seed_set)
                agent_rows.append(
                    {
                        "topology": topology,
                        "kappa": KAPPA,
                        "random_seed": seed,
                        "timestep_activated": timestep,
                        "node_id": int(node),
                        "role": str(simulation.roles[node]),
                        "is_kol": bool(kol_mask[node]),
                        "degree": int(degrees[node]),
                        "active_neighbor_count": int(active_neighbor_counts[node]),
                        "base_graph_active_neighbor_count": int(base_active_counts[index]),
                        "kol_extra_active_neighbor_count": int(extra_active_counts[index]),
                        "has_direct_seed_kol_extra_edge": bool(direct_seed_extra[index]),
                        "local_active_neighbor_proportion": float(local[node]),
                        "effective_exposure_phi": float(effective_exposure[node]),
                        "threshold": float(thresholds[node]),
                        "exposure_minus_threshold": float(
                            effective_exposure[node] - thresholds[node]
                        ),
                    }
                )

            if len(newly_nodes):
                local_new = local[newly_nodes]
                phi_new = effective_exposure[newly_nodes]
                threshold_new = thresholds[newly_nodes]
                degree_new = degrees[newly_nodes]
                active_neighbor_new = active_neighbor_counts[newly_nodes]
                margins = phi_new - threshold_new
                row = {
                    "topology": topology,
                    "kappa": KAPPA,
                    "random_seed": seed,
                    "timestep": timestep,
                    "active_before": active_before,
                    "newly_activated": int(len(newly_nodes)),
                    "active_after": int(model_new_state.sum()),
                    "active_after_proportion": float(model_new_state.mean()),
                    "new_kols": int(kol_mask[newly_nodes].sum()),
                    "mean_local": float(local_new.mean()),
                    "median_local": float(np.median(local_new)),
                    "q10_local": float(np.quantile(local_new, 0.10)),
                    "q90_local": float(np.quantile(local_new, 0.90)),
                    "mean_effective_phi": float(phi_new.mean()),
                    "mean_threshold": float(threshold_new.mean()),
                    "median_threshold": float(np.median(threshold_new)),
                    "q10_threshold": float(np.quantile(threshold_new, 0.10)),
                    "q90_threshold": float(np.quantile(threshold_new, 0.90)),
                    "mean_exposure_margin": float(margins.mean()),
                    "mean_degree": float(degree_new.mean()),
                    "median_degree": float(np.median(degree_new)),
                    "mean_active_neighbor_count": float(active_neighbor_new.mean()),
                    "share_with_base_active_neighbor": float((base_active_counts > 0).mean()),
                    "share_with_kol_extra_active_neighbor": float(
                        (extra_active_counts > 0).mean()
                    ),
                    "share_with_direct_seed_kol_extra_edge": float(direct_seed_extra.mean()),
                }
            else:
                row = {
                    "topology": topology,
                    "kappa": KAPPA,
                    "random_seed": seed,
                    "timestep": timestep,
                    "active_before": active_before,
                    "newly_activated": 0,
                    "active_after": int(model_new_state.sum()),
                    "active_after_proportion": float(model_new_state.mean()),
                    "new_kols": 0,
                    "mean_local": "",
                    "median_local": "",
                    "q10_local": "",
                    "q90_local": "",
                    "mean_effective_phi": "",
                    "mean_threshold": "",
                    "median_threshold": "",
                    "q10_threshold": "",
                    "q90_threshold": "",
                    "mean_exposure_margin": "",
                    "mean_degree": "",
                    "median_degree": "",
                    "mean_active_neighbor_count": "",
                    "share_with_base_active_neighbor": "",
                    "share_with_kol_extra_active_neighbor": "",
                    "share_with_direct_seed_kol_extra_edge": "",
                }
            timestep_rows.append(row)

            if np.array_equal(model_new_state, state):
                break
            state = model_new_state

        graph_rows.append(
            {
                "topology": topology,
                "base_edge_count": len(base_edges),
                "full_edge_count": len(full_edges),
                "unique_kol_extra_edge_count": len(extra_edges),
                "base_edge_sha256": _edge_hash(base_edges),
                "full_edge_sha256": _edge_hash(full_edges),
                "neighbor_cache_matches_graph": neighbor_cache_matches,
                "thresholds_match_reference": thresholds_match_reference,
                "seed_nodes_match_reference": seeds_match_reference,
                "max_abs_effective_phi_minus_local_at_kappa_0": max_phi_local_difference,
            }
        )
        topology_reports[topology] = {
            "kol_count": NUM_KOLS,
            "kol_reach_requested_per_kol": KOL_REACH,
            "initial_seed_count": NUM_IGNITE,
            "initial_seed_nodes": seed_nodes,
            "all_initial_seeds_are_kols": all(kol_mask[node] for node in seed_nodes),
            "unique_seed_kol_extra_targets": len(seed_extra_targets),
            "initial_activated_count": NUM_IGNITE,
            "initial_activated_proportion": NUM_IGNITE / N,
            "final_activated_count": int(state.sum()),
            "final_activated_proportion": float(state.mean()),
        }

    return {
        "threshold_rows": threshold_rows,
        "timestep_rows": timestep_rows,
        "agent_rows": agent_rows,
        "graph_rows": graph_rows,
        "report": {
            "N": N,
            "target_average_degree": K,
            "B": B,
            "R": R,
            "kappa": KAPPA,
            "random_seed": seed,
            "activation_condition": "inactive and effective_exposure_phi >= (agent_cost + R) / B",
            "kappa_zero_implication": "effective_exposure_phi equals local_active_neighbor_proportion for both Posters and Lurkers",
            "topologies": topology_reports,
            "all_base_graph_hashes_distinct": len(
                {row["base_edge_sha256"] for row in graph_rows}
            )
            == len(SUPPORTED_TOPOLOGIES),
            "all_full_graph_hashes_distinct": len(
                {row["full_edge_sha256"] for row in graph_rows}
            )
            == len(SUPPORTED_TOPOLOGIES),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    results = run_diagnostic(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = args.output_dir / "threshold_distribution.csv"
    timestep_path = args.output_dir / "activation_timestep_summary.csv"
    agents_path = args.output_dir / "newly_activated_agents.csv"
    graphs_path = args.output_dir / "graph_verification.csv"
    report_path = args.output_dir / "diagnostic_report.json"
    _write_csv(threshold_path, results["threshold_rows"])
    _write_csv(timestep_path, results["timestep_rows"])
    _write_csv(agents_path, results["agent_rows"])
    _write_csv(graphs_path, results["graph_rows"])
    report_path.write_text(
        json.dumps(results["report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Thresholds: {threshold_path.resolve()}")
    print(f"Timestep summary: {timestep_path.resolve()}")
    print(f"New agents: {agents_path.resolve()}")
    print(f"Graph verification: {graphs_path.resolve()}")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
