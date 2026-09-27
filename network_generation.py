"""Reproducible network generation for the cancel-culture simulations."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


SUPPORTED_TOPOLOGIES = ("ER", "WS", "BA", "SBM")


def _balanced_community_sizes(n: int, num_communities: int) -> list[int]:
    if num_communities < 2 or num_communities > n:
        raise ValueError("num_communities must be between 2 and n")
    quotient, remainder = divmod(n, num_communities)
    return [quotient + (index < remainder) for index in range(num_communities)]


def _validate_common_inputs(n: int, target_avg_degree: float) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer of at least 2")
    if not 0 < target_avg_degree < n:
        raise ValueError("target_avg_degree must be greater than 0 and less than n")


def _ws_degree(target_avg_degree: float) -> int:
    # NetworkX requires an even k. Prefer the closest even integer and resolve
    # exact ties upward so comparisons do not systematically lose density.
    lower = max(2, 2 * int(target_avg_degree // 2))
    upper = lower + 2
    return lower if target_avg_degree - lower < upper - target_avg_degree else upper


def generate_network(
    topology: str,
    n: int,
    target_avg_degree: float,
    seed: int | None,
    **topology_params: Any,
) -> nx.Graph:
    """Generate ER, WS, BA, or SBM using one reproducible interface.

    The WS defaults (``k=target_avg_degree`` and ``p=0.1``) reproduce the
    project's original ``nx.watts_strogatz_graph`` baseline when called with
    ``n=10_000``, ``target_avg_degree=10``, and ``seed=42``.
    """

    _validate_common_inputs(n, target_avg_degree)
    normalized = topology.upper()
    if normalized not in SUPPORTED_TOPOLOGIES:
        choices = ", ".join(SUPPORTED_TOPOLOGIES)
        raise ValueError(f"Unsupported topology {topology!r}; choose one of {choices}")

    params = dict(topology_params)

    if normalized == "ER":
        probability = float(params.pop("p", target_avg_degree / (n - 1)))
        if not 0 <= probability <= 1:
            raise ValueError("ER probability p must be between 0 and 1")
        graph = nx.erdos_renyi_graph(n, p=probability, seed=seed)
        resolved_params: dict[str, Any] = {"p": probability}

    elif normalized == "WS":
        degree = int(params.pop("k", _ws_degree(target_avg_degree)))
        probability = float(params.pop("p", 0.1))
        if degree <= 0 or degree >= n or degree % 2:
            raise ValueError("WS k must be a positive even integer smaller than n")
        if not 0 <= probability <= 1:
            raise ValueError("WS probability p must be between 0 and 1")
        graph = nx.watts_strogatz_graph(n, k=degree, p=probability, seed=seed)
        resolved_params = {"k": degree, "p": probability}

    elif normalized == "BA":
        edges_per_node = int(params.pop("m", round(target_avg_degree / 2)))
        if edges_per_node < 1 or edges_per_node >= n:
            raise ValueError("BA m must be at least 1 and smaller than n")
        graph = nx.barabasi_albert_graph(n, m=edges_per_node, seed=seed)
        resolved_params = {"m": edges_per_node}

    else:
        requested_sizes = params.pop("community_sizes", None)
        if requested_sizes is None:
            num_communities = int(params.pop("num_communities", 4))
            sizes = _balanced_community_sizes(n, num_communities)
        else:
            sizes = [int(size) for size in requested_sizes]
            if not sizes or any(size <= 0 for size in sizes) or sum(sizes) != n:
                raise ValueError("SBM community_sizes must be positive and sum to n")
            num_communities = len(sizes)
            if num_communities < 2:
                raise ValueError("SBM requires at least two communities")

        affinity_ratio = float(params.pop("affinity_ratio", 10.0))
        if affinity_ratio <= 1:
            raise ValueError("SBM affinity_ratio must be greater than 1")

        within_pairs = sum(size * (size - 1) / 2 for size in sizes)
        all_pairs = n * (n - 1) / 2
        cross_pairs = all_pairs - within_pairs
        target_edges = n * target_avg_degree / 2
        probability_out = target_edges / (affinity_ratio * within_pairs + cross_pairs)
        probability_in = affinity_ratio * probability_out
        if probability_in > 1 or probability_out > 1:
            raise ValueError("Requested SBM degree/affinity combination implies p > 1")

        probabilities = [
            [probability_in if row == column else probability_out for column in range(num_communities)]
            for row in range(num_communities)
        ]
        graph = nx.stochastic_block_model(sizes, probabilities, seed=seed)
        community_labels = nx.get_node_attributes(graph, "block")
        nx.set_node_attributes(graph, community_labels, "community")
        resolved_params = {
            "community_sizes": sizes,
            "affinity_ratio": affinity_ratio,
            "p_in": probability_in,
            "p_out": probability_out,
        }

    if params:
        unexpected = ", ".join(sorted(params))
        raise TypeError(f"Unexpected topology parameters for {normalized}: {unexpected}")

    graph.graph.update(
        topology=normalized,
        target_avg_degree=float(target_avg_degree),
        random_seed=seed,
        topology_parameters=resolved_params,
    )
    return graph


def network_statistics(graph: nx.Graph) -> dict[str, int | float | str]:
    """Return the structural statistics required by Step 1."""

    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=float)
    return {
        "topology": str(graph.graph.get("topology", "unknown")),
        "n": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "average_degree": float(degrees.mean()) if degrees.size else 0.0,
        "degree_variance": float(degrees.var()) if degrees.size else 0.0,
        "average_clustering": float(nx.average_clustering(graph)),
    }
