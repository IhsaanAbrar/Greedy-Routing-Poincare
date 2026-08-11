from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from graph_generation import (  # noqa: E402
    GraphGenerationError,
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
)
from iteration2_config import GraphSpec  # noqa: E402
from network_metrics import calculate_network_metrics  # noqa: E402


class Iteration2GraphProtocolTests(unittest.TestCase):
    def test_er_probability_is_the_exact_frozen_formula(self):
        for n in (100, 300, 1000):
            for m in (4, 8, 16):
                spec = GraphSpec(0, "erdos_renyi", n, m, 0)
                self.assertEqual(
                    spec.er_probability,
                    2 * m * (n - m) / (n * (n - 1)),
                )

    def test_er_retry_provenance_records_every_attempt_in_order(self):
        n = 14
        m = 2
        numerator = 2 * m * (n - m)
        denominator = n * (n - 1)
        attempt_seeds = tuple(range(100, 150))
        generated_graphs = [nx.empty_graph(n), nx.empty_graph(n), nx.path_graph(n)]
        with patch(
            "graph_generation.nx.gnp_random_graph",
            side_effect=generated_graphs,
        ) as generator:
            generated = generate_connected_erdos_renyi(
                n=n,
                p=numerator / denominator,
                graph_seed=91,
                replicate_index=0,
                max_attempts=50,
                attempt_seeds=attempt_seeds,
                p_exact_numerator=numerator,
                p_exact_denominator=denominator,
            )
        self.assertTrue(nx.is_connected(generated.graph))
        self.assertEqual(generator.call_count, 3)
        self.assertEqual(
            generated.metadata["generation_attempt_seeds"],
            list(attempt_seeds[:3]),
        )
        self.assertEqual(generated.metadata["generation_attempt_count"], 3)
        self.assertEqual(generated.metadata["rejected_disconnected_count"], 2)
        self.assertEqual(generated.metadata["generation_attempt_seed"], 102)

    def test_er_exhaustion_stops_after_exactly_fifty_attempts(self):
        n = 14
        m = 2
        attempt_seeds = tuple(range(200, 250))
        with patch(
            "graph_generation.nx.gnp_random_graph",
            side_effect=lambda *args, **kwargs: nx.empty_graph(n),
        ) as generator:
            with self.assertRaisesRegex(GraphGenerationError, "after 50 attempts"):
                generate_connected_erdos_renyi(
                    n=n,
                    p=2 * m * (n - m) / (n * (n - 1)),
                    graph_seed=92,
                    replicate_index=0,
                    max_attempts=50,
                    attempt_seeds=attempt_seeds,
                    p_exact_numerator=2 * m * (n - m),
                    p_exact_denominator=n * (n - 1),
                )
        self.assertEqual(generator.call_count, 50)

    def test_ba_uses_star_initial_graph_and_exact_finite_size_edges(self):
        n = 14
        m = 2
        generated = generate_connected_barabasi_albert(
            n=n,
            m=m,
            graph_seed=9871,
            replicate_index=0,
        )
        initial = generated.graph.subgraph(range(m + 1)).copy()
        self.assertTrue(nx.utils.graphs_equal(initial, nx.star_graph(m)))
        self.assertEqual(generated.graph.number_of_edges(), m * (n - m))
        self.assertEqual(
            generated.metadata["ba_initial_graph"],
            "networkx.star_graph(m)",
        )
        self.assertEqual(
            generated.metadata["realised_average_degree"],
            2 * m * (n - m) / n,
        )

    def test_excluded_graphs_have_exact_integer_contract_and_metrics(self):
        generated = generate_connected_barabasi_albert(
            n=14,
            m=2,
            graph_seed=9872,
            replicate_index=0,
        )
        graph = generated.graph
        self.assertEqual(tuple(sorted(graph)), tuple(range(14)))
        self.assertFalse(graph.is_directed())
        self.assertFalse(graph.is_multigraph())
        self.assertEqual(nx.number_of_selfloops(graph), 0)
        self.assertTrue(nx.is_connected(graph))
        metrics = calculate_network_metrics(graph)
        self.assertEqual(metrics["number_of_vertices"], 14)
        self.assertEqual(metrics["number_of_edges"], graph.number_of_edges())
        self.assertEqual(
            metrics["average_degree"], 2 * graph.number_of_edges() / 14
        )
        self.assertGreaterEqual(metrics["maximum_degree"], 2)
        self.assertGreaterEqual(metrics["population_degree_variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
