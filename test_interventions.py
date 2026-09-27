"""STEP 2 intervention framework tests."""

import copy
import unittest

import numpy as np

from baseline_simulation import create_simulation
from cascade_aware import select_cascade_aware_paths
from early_warning import fit_logistic_risk_model, observe_baseline_cascade
from intervention import (
    CascadeObservationSpec,
    InterventionSpec,
    build_exposure_paths,
    compute_local_exposure,
    observe_cascade_state,
    run_intervention,
)
from network_generation import SUPPORTED_TOPOLOGIES
from run_nested_momentum_experiment import nested_coefficients


class NestedMomentumScoreTests(unittest.TestCase):
    def test_lambda_zero_is_exact_observable_pressure(self):
        self.assertEqual(nested_coefficients(0.0), (0.55, 0.0, 0.25, 0.15, 0.05))

    def test_nested_coefficients_preserve_unit_sum(self):
        for momentum_lambda in (0.0, 0.15, 0.35, 0.5):
            coefficients = nested_coefficients(momentum_lambda)
            self.assertAlmostEqual(sum(coefficients), 1.0)
            self.assertAlmostEqual(coefficients[1], momentum_lambda)


class WeightedExposureTests(unittest.TestCase):
    def setUp(self):
        self.simulation = create_simulation(
            "WS", n=300, target_avg_degree=10, network_seed=12, agent_seed=12
        )
        self.paths = build_exposure_paths(self.simulation)
        self.state = np.zeros(self.simulation.n, dtype=int)
        self.state[self.simulation.kol_idx[:5]] = 1

    def test_unit_weights_equal_unweighted_neighbor_mean(self):
        weighted = compute_local_exposure(
            self.state, self.paths, np.ones(self.paths.count)
        )
        expected = np.zeros(self.simulation.n)
        for node, neighbors in enumerate(self.simulation.neighbors):
            if neighbors.size:
                expected[node] = self.state[neighbors].mean()
        np.testing.assert_allclose(weighted, expected)

    def test_reducing_active_neighbor_weight_cannot_increase_exposure(self):
        baseline = compute_local_exposure(
            self.state, self.paths, np.ones(self.paths.count)
        )
        weights = np.ones(self.paths.count)
        active_paths = np.flatnonzero(self.state[self.paths.sources] == 1)
        weights[active_paths[0]] = 0.0
        reduced = compute_local_exposure(self.state, self.paths, weights)
        self.assertTrue(np.all(reduced <= baseline + 1e-15))
        self.assertLess(reduced[self.paths.targets[active_paths[0]]], baseline[self.paths.targets[active_paths[0]]])

    def test_reducing_inactive_neighbor_weight_cannot_increase_exposure(self):
        baseline = compute_local_exposure(
            self.state, self.paths, np.ones(self.paths.count)
        )
        weights = np.ones(self.paths.count)
        inactive_path = np.flatnonzero(self.state[self.paths.sources] == 0)[0]
        weights[inactive_path] = 0.0
        reduced = compute_local_exposure(self.state, self.paths, weights)
        np.testing.assert_allclose(reduced, baseline)


class CascadeObservationTests(unittest.TestCase):
    def test_perfect_observation_preserves_state(self):
        state = np.array([0, 1, 1, 0, 1], dtype=int)
        observed = observe_cascade_state(
            state, CascadeObservationSpec(), np.random.default_rng(12)
        )
        np.testing.assert_array_equal(observed, state)

    def test_zero_active_recall_hides_all_active_events(self):
        state = np.array([0, 1, 1, 0, 1], dtype=int)
        observed = observe_cascade_state(
            state,
            CascadeObservationSpec(active_recall=0.0),
            np.random.default_rng(12),
        )
        np.testing.assert_array_equal(observed, np.zeros_like(state))

    def test_observation_is_reproducible(self):
        state = np.tile(np.array([0, 1], dtype=int), 100)
        spec = CascadeObservationSpec(active_recall=0.7, false_positive_rate=0.1)
        first = observe_cascade_state(state, spec, np.random.default_rng(44))
        second = observe_cascade_state(state, spec, np.random.default_rng(44))
        np.testing.assert_array_equal(first, second)

    def test_invalid_observation_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            CascadeObservationSpec(delay_steps=-1)
        with self.assertRaises(ValueError):
            CascadeObservationSpec(active_recall=1.1)
        with self.assertRaises(ValueError):
            CascadeObservationSpec(false_positive_rate=-0.1)


class InterventionIntegrationTests(unittest.TestCase):
    def test_none_matches_original_baseline_at_every_timestep(self):
        simulation = create_simulation(
            "WS", n=500, target_avg_degree=10, network_seed=42, agent_seed=42
        )
        rng_state = copy.deepcopy(simulation.rng.bit_generator.state)
        ignition = simulation.rng.choice(simulation.kol_idx, size=5, replace=False)
        simulation.rng.bit_generator.state = copy.deepcopy(rng_state)
        baseline_history, baseline_state = simulation.run(kappa=0.6)

        result = run_intervention(
            simulation,
            0.6,
            InterventionSpec(),
            intervention_seed=99,
            ignition_nodes=ignition,
        )
        np.testing.assert_array_equal(
            result.activated_counts,
            np.rint(baseline_history * simulation.n).astype(int),
        )
        np.testing.assert_array_equal(result.final_state, baseline_state)
        self.assertFalse(result.intervention_triggered)

    def test_perfect_observation_matches_existing_frontier_semantics(self):
        simulation = create_simulation(
            "ER", n=500, target_avg_degree=10, network_seed=42, agent_seed=42
        )
        ignition = simulation.kol_idx[:5]
        spec = InterventionSpec(
            strategy="frontier_random",
            trigger_progress=0.02,
            budget=0.01,
            intervention_weight=0.75,
        )
        existing = run_intervention(
            simulation,
            0.6,
            spec,
            intervention_seed=55,
            ignition_nodes=ignition,
            dynamic_selector=select_cascade_aware_paths,
        )
        observed = run_intervention(
            simulation,
            0.6,
            spec,
            intervention_seed=55,
            ignition_nodes=ignition,
            dynamic_selector=select_cascade_aware_paths,
            observation_spec=CascadeObservationSpec(),
            observation_seed=99,
        )
        np.testing.assert_array_equal(existing.activated_counts, observed.activated_counts)
        self.assertEqual(existing.selected_path_count, observed.selected_path_count)
        self.assertEqual(existing.available_path_count, observed.available_path_count)

    def test_zero_recall_removes_observed_frontier_at_initial_trigger(self):
        simulation = create_simulation(
            "ER", n=300, target_avg_degree=10, network_seed=9, agent_seed=9
        )
        result = run_intervention(
            simulation,
            0.6,
            InterventionSpec(
                strategy="frontier_random",
                trigger_progress=0.0,
                budget=0.05,
                intervention_weight=0.75,
            ),
            intervention_seed=9,
            ignition_nodes=simulation.kol_idx[:5],
            dynamic_selector=select_cascade_aware_paths,
            observation_spec=CascadeObservationSpec(active_recall=0.0),
            observation_seed=123,
        )
        self.assertTrue(result.intervention_triggered)
        self.assertEqual(result.available_path_count, 0)
        self.assertEqual(result.selected_path_count, 0)

    def test_progress_trigger_fires_once_and_records_crossing(self):
        simulation = create_simulation(
            "ER", n=500, target_avg_degree=10, network_seed=3, agent_seed=3
        )
        ignition = simulation.kol_idx[:5]
        spec = InterventionSpec(
            strategy="random",
            trigger_progress=0.02,
            budget=0.05,
            intervention_weight=0.0,
        )
        result = run_intervention(
            simulation, 0.6, spec, intervention_seed=3, ignition_nodes=ignition
        )
        if result.intervention_triggered:
            self.assertGreaterEqual(result.actual_activated_ratio, 0.02)
            before = result.activated_proportions[: result.actual_timestep]
            self.assertTrue(np.all(before < 0.02))
            self.assertEqual(
                result.selected_path_count,
                round(spec.budget * build_exposure_paths(simulation).count),
            )

    def test_all_topologies_run_network_and_kappa_interventions(self):
        for topology in SUPPORTED_TOPOLOGIES:
            simulation = create_simulation(
                topology,
                n=250,
                target_avg_degree=10,
                network_seed=7,
                agent_seed=7,
            )
            ignition = simulation.kol_idx[:5]
            strategies = ["random", "degree", "betweenness"]
            if topology == "SBM":
                strategies.append("bridge")
            for strategy in strategies:
                with self.subTest(topology=topology, strategy=strategy):
                    result = run_intervention(
                        simulation,
                        0.7,
                        InterventionSpec(
                            strategy=strategy,
                            trigger_progress=0.0,
                            budget=0.02,
                            intervention_weight=0.5,
                            betweenness_samples=8,
                        ),
                        intervention_seed=7,
                        ignition_nodes=ignition,
                    )
                    self.assertTrue(result.intervention_triggered)
                    self.assertEqual(result.actual_timestep, 0)
                    self.assertGreater(result.selected_path_count, 0)

            kappa_result = run_intervention(
                simulation,
                0.7,
                InterventionSpec(
                    strategy="kappa_reduction",
                    trigger_progress=0.0,
                    kappa_after=0.4,
                ),
                intervention_seed=7,
                ignition_nodes=ignition,
            )
            self.assertEqual(kappa_result.kappa_after, 0.4)
            self.assertEqual(kappa_result.selected_path_count, 0)

    def test_unreached_trigger_is_retained_as_false(self):
        simulation = create_simulation(
            "WS", n=300, target_avg_degree=10, network_seed=21, agent_seed=21
        )
        result = run_intervention(
            simulation,
            0.0,
            InterventionSpec(
                strategy="degree",
                trigger_progress=0.5,
                budget=0.1,
                intervention_weight=0.0,
            ),
            intervention_seed=21,
            ignition_nodes=simulation.kol_idx[:5],
            max_steps=0,
        )
        self.assertFalse(result.intervention_triggered)
        self.assertIsNone(result.actual_timestep)
        self.assertEqual(result.selected_path_count, 0)

    def test_timestep_trigger_fires_at_requested_time(self):
        simulation = create_simulation(
            "ER", n=300, target_avg_degree=10, network_seed=8, agent_seed=8
        )
        result = run_intervention(
            simulation,
            0.5,
            InterventionSpec(
                strategy="degree",
                trigger_timestep=3,
                budget=0.02,
                intervention_weight=0.5,
            ),
            intervention_seed=8,
            ignition_nodes=simulation.kol_idx[:5],
        )
        self.assertTrue(result.intervention_triggered)
        self.assertEqual(result.actual_timestep, 3)
        self.assertEqual(result.requested_timestep, 3)

    def test_only_one_trigger_type_is_allowed(self):
        with self.assertRaisesRegex(ValueError, "only one"):
            InterventionSpec(
                strategy="degree",
                trigger_progress=0.05,
                trigger_timestep=5,
                budget=0.02,
            )

    def test_bridge_requires_sbm_labels(self):
        simulation = create_simulation(
            "ER", n=200, target_avg_degree=10, network_seed=5, agent_seed=5
        )
        with self.assertRaisesRegex(ValueError, "only defined for SBM"):
            run_intervention(
                simulation,
                0.6,
                InterventionSpec(
                    strategy="bridge",
                    trigger_progress=0.0,
                    budget=0.05,
                    intervention_weight=0.0,
                ),
                intervention_seed=5,
                ignition_nodes=simulation.kol_idx[:5],
            )


class CascadeAwareTests(unittest.TestCase):
    def setUp(self):
        self.simulation = create_simulation(
            "SBM", n=300, target_avg_degree=10, network_seed=13, agent_seed=13
        )
        self.paths = build_exposure_paths(self.simulation)
        self.state = np.zeros(self.simulation.n, dtype=int)
        self.state[self.simulation.kol_idx[:5]] = 1
        self.previous = np.zeros_like(self.state)

    def _select(self, strategy, seed=9):
        spec = InterventionSpec(
            strategy=strategy,
            trigger_progress=0.0,
            budget=0.02,
            intervention_weight=0.0,
        )
        return select_cascade_aware_paths(
            self.simulation,
            self.paths,
            spec,
            self.state,
            self.previous,
            0.5,
            np.random.default_rng(seed),
        )[0]

    def test_all_dynamic_strategies_select_only_active_to_inactive_frontier(self):
        for strategy in (
            "frontier_random",
            "frontier_degree",
            "observable_pressure",
            "observable_momentum",
            "oracle_margin",
        ):
            with self.subTest(strategy=strategy):
                selected = self._select(strategy)
                self.assertGreater(selected.size, 0)
                self.assertTrue(np.all(self.state[self.paths.sources[selected]] == 1))
                self.assertTrue(np.all(self.state[self.paths.targets[selected]] == 0))

    def test_observable_selection_does_not_use_latent_thresholds(self):
        for strategy in (
            "frontier_random",
            "frontier_degree",
            "observable_pressure",
            "observable_momentum",
        ):
            with self.subTest(strategy=strategy):
                before = self._select(strategy)
                self.simulation.c_private[:] = 0.999
                self.simulation.c_public[:] = 0.999
                after = self._select(strategy)
                np.testing.assert_array_equal(before, after)

    def test_dynamic_strategy_runs_through_common_framework(self):
        result = run_intervention(
            self.simulation,
            0.5,
            InterventionSpec(
                strategy="observable_pressure",
                trigger_progress=0.0,
                budget=0.02,
                intervention_weight=0.0,
            ),
            intervention_seed=13,
            ignition_nodes=self.simulation.kol_idx[:5],
            prepared_paths=self.paths,
            dynamic_selector=select_cascade_aware_paths,
        )
        self.assertTrue(result.intervention_triggered)
        self.assertGreater(result.selected_path_count, 0)
        self.assertTrue(np.all(np.diff(result.activated_counts) >= 0))

    def test_custom_observable_coefficients_are_supported(self):
        spec = InterventionSpec(
            strategy="observable_momentum",
            trigger_progress=0.0,
            budget=0.02,
            intervention_weight=0.0,
            observable_coefficients=(0.50, 0.20, 0.10, 0.10, 0.10),
        )
        selected, _ = select_cascade_aware_paths(
            self.simulation,
            self.paths,
            spec,
            self.state,
            self.previous,
            0.5,
            np.random.default_rng(9),
        )
        self.assertGreater(selected.size, 0)

    def test_frontier_random_is_reproducible(self):
        first = self._select("frontier_random", seed=22)
        second = self._select("frontier_random", seed=22)
        np.testing.assert_array_equal(first, second)

    def test_frontier_degree_selects_maximum_degree_score(self):
        selected = self._select("frontier_degree", seed=22)
        frontier = np.flatnonzero(
            (self.state[self.paths.sources] == 1)
            & (self.state[self.paths.targets] == 0)
        )
        degree = self.paths.original_degrees
        all_scores = degree[self.paths.sources[frontier]] + degree[self.paths.targets[frontier]]
        selected_scores = degree[self.paths.sources[selected]] + degree[self.paths.targets[selected]]
        self.assertGreaterEqual(selected_scores.min(), np.partition(all_scores, -selected.size)[-selected.size])

    def test_resource_costs_include_strength_and_duration(self):
        result = run_intervention(
            self.simulation,
            0.5,
            InterventionSpec(
                strategy="frontier_random",
                trigger_progress=0.0,
                budget=0.02,
                intervention_weight=0.75,
            ),
            intervention_seed=13,
            ignition_nodes=self.simulation.kol_idx[:5],
            prepared_paths=self.paths,
            dynamic_selector=select_cascade_aware_paths,
        )
        coverage = result.selected_path_count / self.paths.count
        expected = coverage * 0.25 * result.affected_transition_count
        self.assertAlmostEqual(result.temporal_suppression_cost, expected)
        self.assertLessEqual(result.frontier_suppression_cost, result.active_source_suppression_cost)
        self.assertLessEqual(result.active_source_suppression_cost, result.temporal_suppression_cost)

    def test_observable_coefficients_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            InterventionSpec(
                strategy="observable_pressure",
                trigger_progress=0.0,
                budget=0.02,
                observable_coefficients=(0.5, 0.2, 0.1, 0.1),
            )


class EarlyWarningTests(unittest.TestCase):
    def test_observed_baseline_matches_common_runner(self):
        simulation = create_simulation(
            "WS", n=300, target_avg_degree=10, network_seed=17, agent_seed=17
        )
        ignition = simulation.kol_idx[:5]
        paths = build_exposure_paths(simulation)
        features, counts = observe_baseline_cascade(
            simulation, 0.5, ignition, prepared_paths=paths
        )
        result = run_intervention(
            simulation,
            0.5,
            InterventionSpec(),
            intervention_seed=17,
            ignition_nodes=ignition,
            prepared_paths=paths,
        )
        self.assertEqual(counts[-1], result.activated_counts[-1])
        self.assertAlmostEqual(features["activated_t5"], counts[5] / simulation.n)

    def test_logistic_risk_model_fits_obvious_signal(self):
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        model = fit_logistic_risk_model(x, y, ("signal",), l2=0.01)
        probabilities = model.predict_proba(x)
        self.assertTrue(np.all(np.diff(probabilities) > 0))
        self.assertLess(probabilities[0], 0.5)
        self.assertGreater(probabilities[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
