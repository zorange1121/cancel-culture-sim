"""Cascade-state target selection for STEP 3 interventions."""

from __future__ import annotations

import networkx as nx
import numpy as np

from baseline_simulation import B, R, BaselineSimulation
from intervention import ExposurePaths, InterventionSpec, compute_local_exposure


FRONTIER_BASELINE_STRATEGIES = ("frontier_random", "frontier_degree")
OBSERVABLE_STRATEGIES = ("observable_pressure", "observable_momentum")
CASCADE_AWARE_STRATEGIES = (
    *FRONTIER_BASELINE_STRATEGIES,
    *OBSERVABLE_STRATEGIES,
    "oracle_margin",
)

DEFAULT_OBSERVABLE_COEFFICIENTS = {
    "observable_pressure": (0.55, 0.25, 0.15, 0.05),
    "observable_momentum": (0.35, 0.35, 0.15, 0.10, 0.05),
}


def _unit_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    maximum = float(values.max())
    return values.astype(float) / maximum if maximum > 0 else np.zeros_like(values, dtype=float)


def _cross_community(simulation: BaselineSimulation, paths: ExposurePaths) -> np.ndarray:
    if simulation.graph.graph.get("topology") != "SBM":
        return np.zeros(paths.count, dtype=float)
    labels = nx.get_node_attributes(simulation.graph, "community")
    communities = np.fromiter((labels[node] for node in range(simulation.n)), dtype=int)
    return (communities[paths.sources] != communities[paths.targets]).astype(float)


def _rank_with_random_ties(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or candidate_indices.size == 0:
        return np.empty(0, dtype=int)
    shuffled = rng.permutation(candidate_indices.size)
    order = shuffled[np.argsort(-scores[shuffled], kind="stable")]
    return candidate_indices[order[:count]]


def select_cascade_aware_paths(
    simulation: BaselineSimulation,
    paths: ExposurePaths,
    spec: InterventionSpec,
    state: np.ndarray,
    previous_state: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Rank the live cascade frontier using observable or oracle information.

    Frontier Random samples uniformly from active-to-inactive paths.  Frontier
    Degree ranks the same candidates by sender degree plus receiver degree.

    Observable pressure score:
        .55 receiver active-neighbor ratio
      + .25 normalized sender degree
      + .15 normalized receiver degree
      + .05 cross-community indicator

    Observable momentum score:
        .35 receiver active-neighbor ratio
      + .35 sender newly activated at the previous transition
      + .15 normalized sender degree
      + .10 normalized receiver degree
      + .05 cross-community indicator

    Oracle score uses the true threshold margin and the one-path effect on the
    receiver's perceived exposure.  It is a privileged benchmark, not a claim
    of globally optimal combinatorial targeting.
    """

    if spec.strategy not in CASCADE_AWARE_STRATEGIES:
        raise ValueError(f"Unsupported cascade-aware strategy {spec.strategy!r}")

    frontier_mask = (state[paths.sources] == 1) & (state[paths.targets] == 0)
    candidates = np.flatnonzero(frontier_mask)
    requested = int(round(spec.budget * paths.count))
    count = min(requested, candidates.size)
    if count == 0:
        return np.empty(0, dtype=int), int(candidates.size)

    degree_norm = _unit_scale(paths.original_degrees)
    target = paths.targets[candidates]
    source = paths.sources[candidates]

    if spec.strategy == "frontier_random":
        scores = np.zeros(candidates.size, dtype=float)
    elif spec.strategy == "frontier_degree":
        scores = degree_norm[source] + degree_norm[target]
    elif spec.strategy == "observable_pressure":
        local = compute_local_exposure(state, paths, np.ones(paths.count, dtype=float))
        cross = _cross_community(simulation, paths)
        coefficients = spec.observable_coefficients or DEFAULT_OBSERVABLE_COEFFICIENTS[spec.strategy]
        if len(coefficients) != 4:
            raise ValueError("observable_pressure requires four coefficients")
        local_weight, sender_weight, receiver_weight, cross_weight = coefficients
        scores = (
            local_weight * local[target]
            + sender_weight * degree_norm[source]
            + receiver_weight * degree_norm[target]
            + cross_weight * cross[candidates]
        )
    elif spec.strategy == "observable_momentum":
        local = compute_local_exposure(state, paths, np.ones(paths.count, dtype=float))
        cross = _cross_community(simulation, paths)
        coefficients = spec.observable_coefficients or DEFAULT_OBSERVABLE_COEFFICIENTS[spec.strategy]
        if len(coefficients) != 5:
            raise ValueError("observable_momentum requires five coefficients")
        local_weight, momentum_weight, sender_weight, receiver_weight, cross_weight = coefficients
        newly_active = ((state == 1) & (previous_state == 0)).astype(float)
        scores = (
            local_weight * local[target]
            + momentum_weight * newly_active[source]
            + sender_weight * degree_norm[source]
            + receiver_weight * degree_norm[target]
            + cross_weight * cross[candidates]
        )
    else:
        local = compute_local_exposure(state, paths, np.ones(paths.count, dtype=float))
        poster_mask = simulation.roles == "Poster"
        global_exposure = state[poster_mask].mean() if poster_mask.any() else 0.0
        perceived = local.copy()
        perceived[poster_mask] = (
            (1 - kappa) * local[poster_mask] + kappa * global_exposure
        )
        thresholds = np.where(
            poster_mask,
            (simulation.c_public + R) / B,
            (simulation.c_private + R) / B,
        )
        margin = thresholds - perceived
        susceptibility = 1.0 / (np.abs(margin[target]) + 1e-3)
        susceptibility = _unit_scale(susceptibility)
        imminent = (margin[target] <= 0).astype(float)
        path_effect = np.divide(
            1.0,
            paths.original_degrees[target],
            out=np.zeros(target.size, dtype=float),
            where=paths.original_degrees[target] > 0,
        )
        path_effect[poster_mask[target]] *= 1 - kappa
        path_effect = _unit_scale(path_effect)
        scores = (
            0.50 * susceptibility
            + 0.25 * imminent
            + 0.15 * degree_norm[target]
            + 0.10 * path_effect
        )

    return _rank_with_random_ties(candidates, scores, count, rng), int(candidates.size)
