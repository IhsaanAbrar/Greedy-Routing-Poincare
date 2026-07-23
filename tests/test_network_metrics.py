import json
from pathlib import Path
import sys
import unittest

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from network_metrics import (  # noqa: E402
    calculate_network_metrics,
    prepare_all_pairs_shortest_paths,
)


class NetworkMetricsTests(unittest.TestCase):
    def test_four_node_path_metrics_match_hand_calculation(self):
        graph = nx.path_graph(4)

        metrics = calculate_network_metrics(graph)

        self.assertEqual(metrics["number_of_vertices"], 4)
        self.assertEqual(metrics["number_of_edges"], 3)
        self.assertAlmostEqual(metrics["average_degree"], 1.5)
        self.assertEqual(metrics["maximum_degree"], 2)
        self.assertAlmostEqual(metrics["population_degree_variance"], 0.25)
        self.assertAlmostEqual(metrics["average_clustering_coefficient"], 0.0)
        self.assertEqual(metrics["diameter"], 3)
        self.assertAlmostEqual(metrics["average_shortest_path_length"], 10 / 6)

    def test_triangle_metrics_match_hand_calculation(self):
        metrics = calculate_network_metrics(nx.complete_graph(3))

        self.assertEqual(metrics["number_of_vertices"], 3)
        self.assertEqual(metrics["number_of_edges"], 3)
        self.assertAlmostEqual(metrics["average_degree"], 2.0)
        self.assertEqual(metrics["maximum_degree"], 2)
        self.assertAlmostEqual(metrics["population_degree_variance"], 0.0)
        self.assertAlmostEqual(metrics["average_clustering_coefficient"], 1.0)
        self.assertEqual(metrics["diameter"], 1)
        self.assertAlmostEqual(metrics["average_shortest_path_length"], 1.0)

    def test_result_has_stable_fields_and_is_json_serializable(self):
        metrics = calculate_network_metrics(nx.path_graph(2))

        self.assertEqual(
            tuple(metrics),
            (
                "number_of_vertices",
                "number_of_edges",
                "average_degree",
                "maximum_degree",
                "population_degree_variance",
                "average_clustering_coefficient",
                "diameter",
                "average_shortest_path_length",
            ),
        )
        json.dumps(metrics)

    def test_calculation_does_not_mutate_graph(self):
        graph = nx.path_graph(4)
        nodes_before = dict(graph.nodes(data=True))
        edges_before = list(graph.edges(data=True))
        graph_metadata_before = dict(graph.graph)

        calculate_network_metrics(graph)

        self.assertEqual(dict(graph.nodes(data=True)), nodes_before)
        self.assertEqual(list(graph.edges(data=True)), edges_before)
        self.assertEqual(graph.graph, graph_metadata_before)

    def test_precomputed_shortest_paths_are_reused_and_topology_checked(self):
        graph = nx.path_graph(5)
        shortest_paths = prepare_all_pairs_shortest_paths(graph)

        direct = calculate_network_metrics(graph)
        reused = calculate_network_metrics(graph, shortest_paths=shortest_paths)

        self.assertEqual(reused, direct)
        different_graph = nx.star_graph(4)
        with self.assertRaisesRegex(ValueError, "topology"):
            calculate_network_metrics(
                different_graph,
                shortest_paths=shortest_paths,
            )

    def test_singleton_shortest_path_summary_is_well_defined(self):
        metrics = calculate_network_metrics(nx.empty_graph(1))

        self.assertEqual(metrics["diameter"], 0)
        self.assertEqual(metrics["average_shortest_path_length"], 0.0)

    def test_shared_shortest_path_results_match_networkx_references(self):
        graphs = (
            nx.path_graph(7),
            nx.cycle_graph(7),
            nx.complete_graph(6),
            nx.barabasi_albert_graph(20, 2, seed=31),
        )

        for graph in graphs:
            with self.subTest(edges=graph.number_of_edges()):
                metrics = calculate_network_metrics(graph)
                self.assertEqual(metrics["diameter"], nx.diameter(graph))
                self.assertAlmostEqual(
                    metrics["average_shortest_path_length"],
                    nx.average_shortest_path_length(graph),
                    places=14,
                )

    def test_disconnected_graph_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "connected"):
            calculate_network_metrics(nx.empty_graph(2))

    def test_empty_graph_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            calculate_network_metrics(nx.Graph())

    def test_directed_graph_is_rejected(self):
        graph = nx.DiGraph([(0, 1), (1, 0)])
        with self.assertRaisesRegex(ValueError, "undirected"):
            calculate_network_metrics(graph)

    def test_multigraph_and_self_loops_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "simple"):
            calculate_network_metrics(nx.MultiGraph([(0, 1)]))

        graph = nx.path_graph(2)
        graph.add_edge(0, 0)
        with self.assertRaisesRegex(ValueError, "self-loops"):
            calculate_network_metrics(graph)


if __name__ == "__main__":
    unittest.main()
