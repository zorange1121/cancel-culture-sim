"""Step 1 tests for topology generation and baseline compatibility."""

import copy
import unittest

import networkx as nx
import numpy as np

from baseline_simulation import B, R, create_simulation, parse_pct
from network_generation import SUPPORTED_TOPOLOGIES, generate_network, network_statistics
from run_kappa_sweep_multiseed import (
    aggregate_forward_filled_trajectories,
    capture_ignition_nodes,
    prepare_vectorized_simulation,
    run_vectorized,
)
from run_topology_baseline_experiment import cascade_metrics


class PercentageParsingTests(unittest.TestCase):
    def test_numeric_proportions_are_not_scaled_again(self):
        self.assertEqual(parse_pct(0.1), 0.1)
        self.assertEqual(parse_pct(0.6), 0.6)
        self.assertEqual(parse_pct(1.0), 1.0)

    def test_percentage_strings_are_scaled_once(self):
        self.assertEqual(parse_pct("10%"), 0.1)
        self.assertEqual(parse_pct("60%"), 0.6)
        self.assertEqual(parse_pct("100%"), 1.0)

    def test_percentage_range_string_uses_midpoint(self):
        self.assertEqual(parse_pct("90~100%"), 0.95)


class NetworkGenerationTests(unittest.TestCase):
    def test_all_topologies_match_node_and_degree_targets(self):
        for topology in SUPPORTED_TOPOLOGIES:
            with self.subTest(topology=topology):
                graph = generate_network(topology, n=1_000, target_avg_degree=10, seed=42)
                summary = network_statistics(graph)
                self.assertEqual(summary["n"], 1_000)
                self.assertLess(abs(summary["average_degree"] - 10), 1.0)

    def test_same_seed_reproduces_exact_edges(self):
        for topology in SUPPORTED_TOPOLOGIES:
            with self.subTest(topology=topology):
                first = generate_network(topology, n=300, target_avg_degree=10, seed=123)
                second = generate_network(topology, n=300, target_avg_degree=10, seed=123)
                self.assertEqual(set(first.edges()), set(second.edges()))

    def test_sbm_stores_community_for_every_node(self):
        graph = generate_network("SBM", n=401, target_avg_degree=10, seed=42)
        labels = nx.get_node_attributes(graph, "community")
        self.assertEqual(len(labels), 401)
        self.assertEqual(set(labels.values()), {0, 1, 2, 3})

    def test_ws_default_is_exact_original_baseline_graph(self):
        expected = nx.watts_strogatz_graph(10_000, k=10, p=0.1, seed=42)
        actual = generate_network("WS", n=10_000, target_avg_degree=10, seed=42)
        self.assertEqual(set(expected.edges()), set(actual.edges()))


class SimulationIntegrationTests(unittest.TestCase):
    def test_trajectory_aggregation_carries_final_values_to_common_horizon(self):
        run_rows = [
            {
                "topology": "ER",
                "kappa": 0.4,
                "random_seed": 0,
                "final_activated_proportion": 0.2,
                "cascade_duration": 1,
            },
            {
                "topology": "ER",
                "kappa": 0.4,
                "random_seed": 1,
                "final_activated_proportion": 0.8,
                "cascade_duration": 2,
            },
        ]
        timestep_rows = [
            {"topology": "ER", "kappa": 0.4, "random_seed": 0, "timestep": 0, "activated_proportion": 0.1},
            {"topology": "ER", "kappa": 0.4, "random_seed": 0, "timestep": 1, "activated_proportion": 0.2},
            {"topology": "ER", "kappa": 0.4, "random_seed": 1, "timestep": 0, "activated_proportion": 0.1},
            {"topology": "ER", "kappa": 0.4, "random_seed": 1, "timestep": 1, "activated_proportion": 0.4},
            {"topology": "ER", "kappa": 0.4, "random_seed": 1, "timestep": 2, "activated_proportion": 0.8},
        ]

        aggregated = aggregate_forward_filled_trajectories(
            timestep_rows, run_rows, horizon=3
        )
        self.assertEqual([row["n_runs"] for row in aggregated], [2, 2, 2, 2])
        np.testing.assert_allclose(
            [row["mean_activated_proportion"] for row in aggregated],
            [0.1, 0.3, 0.5, 0.5],
        )

    def test_vectorized_sweep_runner_matches_original_simulation_each_timestep(self):
        simulation = create_simulation(
            "WS", n=300, target_avg_degree=10, network_seed=42, agent_seed=42
        )
        rng_state_before_ignition = copy.deepcopy(simulation.rng.bit_generator.state)
        ignition_nodes = capture_ignition_nodes(simulation)
        prepared = prepare_vectorized_simulation(simulation)

        for kappa in (0.0, 0.4, 1.0):
            with self.subTest(kappa=kappa):
                vector_counts, vector_state = run_vectorized(
                    prepared, simulation.n, ignition_nodes, kappa
                )
                simulation.rng.bit_generator.state = copy.deepcopy(
                    rng_state_before_ignition
                )
                original_history, original_state = simulation.run(kappa=kappa)
                original_counts = np.rint(
                    np.asarray(original_history) * simulation.n
                ).astype(int)
                np.testing.assert_array_equal(vector_counts, original_counts)
                np.testing.assert_array_equal(vector_state, original_state)

    def test_cascade_metrics_use_half_of_final_size_for_t50(self):
        metrics = cascade_metrics(np.array([5, 100, 600, 900, 900]), n=1_000)
        self.assertEqual(metrics["final_activated_count"], 900)
        self.assertEqual(metrics["final_activated_proportion"], 0.9)
        self.assertEqual(metrics["t50"], 2)
        self.assertEqual(metrics["total_cascade_duration"], 4)

    def test_agent_inputs_do_not_depend_on_topology(self):
        ws = create_simulation("WS", n=300, target_avg_degree=10, network_seed=42, agent_seed=42)
        for topology in ("ER", "BA", "SBM"):
            with self.subTest(topology=topology):
                other = create_simulation(topology, n=300, target_avg_degree=10, network_seed=42, agent_seed=42)
                np.testing.assert_array_equal(ws.roles, other.roles)
                np.testing.assert_allclose(ws.c_private, other.c_private)
                np.testing.assert_allclose(ws.c_public, other.c_public, equal_nan=True)
                np.testing.assert_array_equal(ws.kol_idx, other.kol_idx)

    def test_original_decision_formula_is_unchanged(self):
        simulation = create_simulation("WS", n=300, target_avg_degree=10, network_seed=42, agent_seed=42)
        state = np.zeros(simulation.n, dtype=int)
        state[simulation.kol_idx[:5]] = 1
        actual = simulation.step(state, kappa=0.6, benefit=B, outside_payoff=R)

        local = np.zeros(simulation.n)
        for node, neighbors in enumerate(simulation.neighbors):
            if len(neighbors):
                local[node] = state[neighbors].mean()
        poster_mask = simulation.roles == "Poster"
        global_exposure = state[poster_mask].mean()
        expected = state.copy()
        for node in range(simulation.n):
            if state[node] == 1:
                continue
            if simulation.roles[node] == "Lurker":
                if local[node] >= (simulation.c_private[node] + R) / B:
                    expected[node] = 1
            else:
                perceived = 0.4 * local[node] + 0.6 * global_exposure
                if perceived >= (simulation.c_public[node] + R) / B:
                    expected[node] = 1
        np.testing.assert_array_equal(actual, expected)

    def test_all_topologies_complete_a_full_simulation(self):
        for topology in SUPPORTED_TOPOLOGIES:
            with self.subTest(topology=topology):
                simulation = create_simulation(topology, n=300, target_avg_degree=10, network_seed=42, agent_seed=42)
                history, final_state = simulation.run(kappa=0.6)
                self.assertGreaterEqual(len(history), 2)
                self.assertTrue(np.all(np.diff(history) >= 0))
                self.assertEqual(final_state.shape, (300,))


if __name__ == "__main__":
    unittest.main()
