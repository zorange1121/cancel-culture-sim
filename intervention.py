"""Static exposure interventions for STEP 2.

The baseline agent thresholds, utility rule, KOL construction, and irreversible
activation dynamics are preserved.  Network interventions change directed
exposure-path weights; they never remove graph edges or alter the fixed
pre-intervention degree denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import networkx as nx
import numpy as np

from baseline_simulation import B, NUM_IGNITE, R, BaselineSimulation


NETWORK_STRATEGIES = ("random", "degree", "betweenness", "bridge")
CASCADE_AWARE_STRATEGIES = (
    "frontier_random",
    "frontier_degree",
    "observable_pressure",
    "observable_momentum",
    "oracle_margin",
)
SUPPORTED_INTERVENTIONS = (
    "none",
    *NETWORK_STRATEGIES,
    "kappa_reduction",
    *CASCADE_AWARE_STRATEGIES,
)

DynamicPathSelector = Callable[
    [
        BaselineSimulation,
        "ExposurePaths",
        "InterventionSpec",
        np.ndarray,
        np.ndarray,
        float,
        np.random.Generator,
    ],
    tuple[np.ndarray, int],
]


@dataclass(frozen=True)
class InterventionSpec:
    """Configuration for one static STEP 2 intervention."""

    strategy: str = "none"
    trigger_progress: float | None = None
    trigger_timestep: int | None = None
    budget: float = 0.0
    intervention_weight: float = 1.0
    kappa_after: float | None = None
    betweenness_samples: int = 64
    observable_coefficients: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        strategy = self.strategy.lower()
        object.__setattr__(self, "strategy", strategy)
        if strategy not in SUPPORTED_INTERVENTIONS:
            raise ValueError(
                f"Unknown intervention strategy {self.strategy!r}; "
                f"choose one of {SUPPORTED_INTERVENTIONS}"
            )
        if self.trigger_progress is not None and not 0 <= self.trigger_progress <= 1:
            raise ValueError("trigger_progress must be between 0 and 1")
        if self.trigger_timestep is not None and self.trigger_timestep < 0:
            raise ValueError("trigger_timestep must be non-negative")
        if not 0 <= self.budget <= 1:
            raise ValueError("budget must be between 0 and 1")
        if not 0 <= self.intervention_weight <= 1:
            raise ValueError("intervention_weight must be between 0 and 1")
        if self.kappa_after is not None and not 0 <= self.kappa_after <= 1:
            raise ValueError("kappa_after must be between 0 and 1")
        if self.betweenness_samples < 1:
            raise ValueError("betweenness_samples must be positive")
        if self.observable_coefficients is not None:
            coefficients = tuple(float(value) for value in self.observable_coefficients)
            if not coefficients or any(value < 0 for value in coefficients):
                raise ValueError("observable_coefficients must be non-negative")
            if not np.isclose(sum(coefficients), 1.0):
                raise ValueError("observable_coefficients must sum to 1")
            object.__setattr__(self, "observable_coefficients", coefficients)
        if strategy != "none" and self.trigger_progress is None and self.trigger_timestep is None:
            raise ValueError("An active intervention requires a progress or timestep trigger")
        if self.trigger_progress is not None and self.trigger_timestep is not None:
            raise ValueError("Specify only one intervention trigger")
        if strategy == "kappa_reduction" and self.kappa_after is None:
            raise ValueError("kappa_reduction requires kappa_after")


@dataclass(frozen=True)
class CascadeObservationSpec:
    """Information available when a state-conditioned intervention triggers.

    ``delay_steps`` uses an earlier cascade snapshot. ``active_recall`` models
    missed active events, while ``false_positive_rate`` models inactive nodes
    incorrectly reported as active.  The true state still drives the cascade;
    only path selection sees the degraded snapshot.
    """

    delay_steps: int = 0
    active_recall: float = 1.0
    false_positive_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if not 0 <= self.active_recall <= 1:
            raise ValueError("active_recall must be between 0 and 1")
        if not 0 <= self.false_positive_rate <= 1:
            raise ValueError("false_positive_rate must be between 0 and 1")


def observe_cascade_state(
    true_state: np.ndarray,
    spec: CascadeObservationSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a noisy binary snapshot without altering the true cascade."""

    observed = true_state.copy()
    if spec.active_recall < 1:
        active = np.flatnonzero(true_state == 1)
        missed = rng.random(active.size) >= spec.active_recall
        observed[active[missed]] = 0
    if spec.false_positive_rate > 0:
        inactive = np.flatnonzero(true_state == 0)
        false_positive = rng.random(inactive.size) < spec.false_positive_rate
        observed[inactive[false_positive]] = 1
    return observed


@dataclass
class InterventionResult:
    """A complete cascade and its intervention audit fields."""

    activated_counts: np.ndarray
    final_state: np.ndarray
    intervention_triggered: bool
    requested_trigger: float | None
    requested_timestep: int | None
    actual_activated_ratio: float | None
    actual_timestep: int | None
    selected_path_count: int
    available_path_count: int
    kappa_before: float
    kappa_after: float
    affected_transition_count: int = 0
    temporal_suppression_cost: float = 0.0
    active_source_suppression_cost: float = 0.0
    frontier_suppression_cost: float = 0.0
    true_frontier_count_at_trigger: int | None = None
    observed_frontier_count_at_trigger: int | None = None
    observed_frontier_precision: float | None = None
    observed_frontier_recall: float | None = None

    @property
    def activated_proportions(self) -> np.ndarray:
        return self.activated_counts / self.final_state.size


@dataclass(frozen=True)
class ExposurePaths:
    """CSR-aligned directed paths j -> i for local exposure of target i."""

    indptr: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    original_degrees: np.ndarray

    @property
    def count(self) -> int:
        return int(self.sources.size)


def build_exposure_paths(simulation: BaselineSimulation) -> ExposurePaths:
    degrees = np.fromiter((len(nodes) for nodes in simulation.neighbors), dtype=int)
    indptr = np.concatenate(([0], np.cumsum(degrees, dtype=int)))
    if indptr[-1]:
        sources = np.concatenate(simulation.neighbors).astype(int, copy=False)
        targets = np.repeat(np.arange(simulation.n, dtype=int), degrees)
    else:
        sources = np.empty(0, dtype=int)
        targets = np.empty(0, dtype=int)
    return ExposurePaths(indptr, sources, targets, degrees.astype(float))


def compute_local_exposure(
    state: np.ndarray,
    paths: ExposurePaths,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute weighted exposure with the fixed original-degree denominator."""

    if weights.shape != paths.sources.shape:
        raise ValueError("weights must contain one value per directed exposure path")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("exposure weights must be between 0 and 1")
    numerator = np.bincount(
        paths.targets,
        weights=weights * state[paths.sources],
        minlength=state.size,
    )
    return np.divide(
        numerator,
        paths.original_degrees,
        out=np.zeros_like(numerator),
        where=paths.original_degrees > 0,
    )


def _random_tie_order(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = rng.permutation(scores.size)
    return shuffled[np.argsort(-scores[shuffled], kind="stable")]


def select_exposure_paths(
    simulation: BaselineSimulation,
    paths: ExposurePaths,
    spec: InterventionSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Select directed exposure paths using a static STEP 2 strategy.

    The requested budget is a proportion of all directed paths.  Bridge targets
    are restricted to cross-community paths, so their achieved count is capped
    by the number of available bridges.
    """

    requested_count = min(paths.count, int(round(spec.budget * paths.count)))
    if requested_count == 0 or spec.strategy not in NETWORK_STRATEGIES:
        return np.empty(0, dtype=int), paths.count

    if spec.strategy == "random":
        return rng.choice(paths.count, size=requested_count, replace=False), paths.count

    if spec.strategy == "degree":
        scores = paths.original_degrees[paths.targets] + paths.original_degrees[paths.sources]
        return _random_tie_order(scores, rng)[:requested_count], paths.count

    if spec.strategy == "bridge":
        if simulation.graph.graph.get("topology") != "SBM":
            raise ValueError("bridge intervention is only defined for SBM")
        labels = nx.get_node_attributes(simulation.graph, "community")
        if len(labels) != simulation.n:
            raise ValueError("SBM graph is missing community labels")
        community = np.fromiter((labels[node] for node in range(simulation.n)), dtype=int)
        eligible = np.flatnonzero(
            community[paths.targets] != community[paths.sources]
        )
        count = min(requested_count, eligible.size)
        return rng.choice(eligible, size=count, replace=False), int(eligible.size)

    # Approximate edge betweenness once per intervention run, never per timestep.
    sample_count = min(spec.betweenness_samples, simulation.n)
    scores_by_edge = nx.edge_betweenness_centrality(
        simulation.graph,
        k=sample_count if sample_count < simulation.n else None,
        normalized=True,
        seed=int(rng.integers(0, 2**32 - 1)),
    )
    scores = np.fromiter(
        (
            scores_by_edge.get((int(source), int(target)), scores_by_edge.get((int(target), int(source)), 0.0))
            for target, source in zip(paths.targets, paths.sources)
        ),
        dtype=float,
        count=paths.count,
    )
    return _random_tie_order(scores, rng)[:requested_count], paths.count


def apply_decision_rule(
    simulation: BaselineSimulation,
    state: np.ndarray,
    local_exposure: np.ndarray,
    kappa: float,
    benefit: float,
    outside_payoff: float,
) -> np.ndarray:
    """Apply the unchanged baseline decision rule to supplied local exposure."""

    poster_mask = simulation.roles == "Poster"
    global_exposure = state[poster_mask].mean() if poster_mask.any() else 0.0
    thresholds = np.where(
        poster_mask,
        (simulation.c_public + outside_payoff) / benefit,
        (simulation.c_private + outside_payoff) / benefit,
    )
    perceived = local_exposure.copy()
    perceived[poster_mask] = (
        (1 - kappa) * local_exposure[poster_mask] + kappa * global_exposure
    )
    new_state = state.copy()
    new_state[(state == 0) & (perceived >= thresholds)] = 1
    return new_state


def run_intervention(
    simulation: BaselineSimulation,
    kappa: float,
    spec: InterventionSpec,
    *,
    intervention_seed: int,
    ignition_nodes: Iterable[int] | None = None,
    benefit: float = B,
    outside_payoff: float = R,
    max_steps: int = 60,
    num_ignite: int = NUM_IGNITE,
    prepared_paths: ExposurePaths | None = None,
    preselected_paths: tuple[np.ndarray, int] | None = None,
    dynamic_selector: DynamicPathSelector | None = None,
    observation_spec: CascadeObservationSpec | None = None,
    observation_seed: int | None = None,
) -> InterventionResult:
    """Run one reproducible cascade with a one-time progress intervention."""

    if not 0 <= kappa <= 1:
        raise ValueError("kappa must be between 0 and 1")
    if spec.strategy == "kappa_reduction" and spec.kappa_after > kappa:
        raise ValueError("kappa_after must not exceed the initial kappa")

    paths = prepared_paths or build_exposure_paths(simulation)
    weights = np.ones(paths.count, dtype=float)
    intervention_rng = np.random.default_rng(intervention_seed)
    observation_rng = np.random.default_rng(
        intervention_seed if observation_seed is None else observation_seed
    )
    if ignition_nodes is None:
        ignition = simulation.rng.choice(
            simulation.kol_idx, size=num_ignite, replace=False
        )
    else:
        ignition = np.asarray(list(ignition_nodes), dtype=int)
        if ignition.size != num_ignite:
            raise ValueError("ignition_nodes size must equal num_ignite")

    state = np.zeros(simulation.n, dtype=int)
    state[ignition] = 1
    counts = [int(state.sum())]
    triggered = False
    actual_ratio: float | None = None
    actual_timestep: int | None = None
    selected_count = 0
    selected = np.empty(0, dtype=int)
    available_count = paths.count
    current_kappa = float(kappa)
    previous_state = np.zeros_like(state)
    state_history = [state.copy()]
    true_frontier_count_at_trigger: int | None = None
    observed_frontier_count_at_trigger: int | None = None
    observed_frontier_precision: float | None = None
    observed_frontier_recall: float | None = None

    def trigger_if_due(timestep: int) -> None:
        nonlocal triggered, actual_ratio, actual_timestep
        nonlocal selected, selected_count, available_count, current_kappa
        nonlocal true_frontier_count_at_trigger, observed_frontier_count_at_trigger
        nonlocal observed_frontier_precision, observed_frontier_recall
        if triggered or spec.strategy == "none":
            return
        ratio = float(state.mean())
        if spec.trigger_timestep is not None:
            if timestep < spec.trigger_timestep:
                return
        elif ratio < spec.trigger_progress:
            return
        triggered = True
        actual_ratio = ratio
        actual_timestep = timestep
        if spec.strategy == "kappa_reduction":
            current_kappa = float(spec.kappa_after)
            return
        if dynamic_selector is not None:
            selector_state = state
            selector_previous = previous_state
            if observation_spec is not None:
                snapshot_index = max(
                    0, len(state_history) - 1 - observation_spec.delay_steps
                )
                previous_index = max(0, snapshot_index - 1)
                selector_state = observe_cascade_state(
                    state_history[snapshot_index], observation_spec, observation_rng
                )
                selector_previous = observe_cascade_state(
                    state_history[previous_index], observation_spec, observation_rng
                )
            true_frontier = (state[paths.sources] == 1) & (state[paths.targets] == 0)
            observed_frontier = (
                (selector_state[paths.sources] == 1)
                & (selector_state[paths.targets] == 0)
            )
            overlap = int(np.count_nonzero(true_frontier & observed_frontier))
            true_frontier_count_at_trigger = int(np.count_nonzero(true_frontier))
            observed_frontier_count_at_trigger = int(np.count_nonzero(observed_frontier))
            observed_frontier_precision = (
                overlap / observed_frontier_count_at_trigger
                if observed_frontier_count_at_trigger
                else None
            )
            observed_frontier_recall = (
                overlap / true_frontier_count_at_trigger
                if true_frontier_count_at_trigger
                else None
            )
            selected, available_count = dynamic_selector(
                simulation,
                paths,
                spec,
                selector_state,
                selector_previous,
                current_kappa,
                intervention_rng,
            )
        elif preselected_paths is None:
            selected, available_count = select_exposure_paths(
                simulation, paths, spec, intervention_rng
            )
        else:
            selected, available_count = preselected_paths
        weights[selected] = spec.intervention_weight
        selected_count = int(selected.size)

    affected_transition_count = 0
    temporal_suppression_path_steps = 0.0
    active_source_suppression_path_steps = 0.0
    frontier_suppression_path_steps = 0.0

    trigger_if_due(0)
    for timestep in range(1, max_steps + 1):
        if triggered and selected_count:
            suppression = 1.0 - spec.intervention_weight
            affected_transition_count += 1
            temporal_suppression_path_steps += suppression * selected_count
            selected_sources = paths.sources[selected]
            selected_targets = paths.targets[selected]
            active_sources = state[selected_sources] == 1
            active_source_suppression_path_steps += suppression * int(active_sources.sum())
            frontier_suppression_path_steps += suppression * int(
                (active_sources & (state[selected_targets] == 0)).sum()
            )
        local = compute_local_exposure(state, paths, weights)
        new_state = apply_decision_rule(
            simulation,
            state,
            local,
            current_kappa,
            benefit,
            outside_payoff,
        )
        counts.append(int(new_state.sum()))
        unchanged = np.array_equal(new_state, state)
        previous_state = state
        state = new_state
        state_history.append(state.copy())
        trigger_if_due(timestep)
        if unchanged:
            break

    return InterventionResult(
        activated_counts=np.asarray(counts, dtype=int),
        final_state=state,
        intervention_triggered=triggered,
        requested_trigger=spec.trigger_progress,
        requested_timestep=spec.trigger_timestep,
        actual_activated_ratio=actual_ratio,
        actual_timestep=actual_timestep,
        selected_path_count=selected_count,
        available_path_count=available_count,
        kappa_before=float(kappa),
        kappa_after=current_kappa,
        affected_transition_count=affected_transition_count,
        temporal_suppression_cost=temporal_suppression_path_steps / paths.count,
        active_source_suppression_cost=active_source_suppression_path_steps / paths.count,
        frontier_suppression_cost=frontier_suppression_path_steps / paths.count,
        true_frontier_count_at_trigger=true_frontier_count_at_trigger,
        observed_frontier_count_at_trigger=observed_frontier_count_at_trigger,
        observed_frontier_precision=observed_frontier_precision,
        observed_frontier_recall=observed_frontier_recall,
    )
