from math import isfinite, tanh
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import embedding as embedding_module  # noqa: E402
from embedding import (  # noqa: E402
    DEVELOPMENT_EMBEDDING_ID,
    FORCE_LAYOUT_IMPLEMENTATION,
    MAX_RANDOM_SEED,
    calculate_embedding_distortion,
    development_embedding_metadata,
    embed_graph_in_poincare_disk,
    project_to_unit_disk,
)
from network_metrics import prepare_all_pairs_shortest_paths  # noqa: E402
from poincare_distance import poincare_distance  # noqa: E402


class ProjectionTests(unittest.TestCase):
    def test_projection_is_radial_and_strictly_inside_disk(self):
        projected = project_to_unit_disk((3.0, 4.0), disk_epsilon=0.1)

        self.assertAlmostEqual(float(np.linalg.norm(projected)), 0.9, places=14)
        np.testing.assert_allclose(projected / projected[0], (1.0, 4.0 / 3.0))
        self.assertLess(float(np.linalg.norm(projected)), 1.0)

    def test_point_at_safe_radius_is_handled_and_remains_inside(self):
        projected = project_to_unit_disk((0.9, 0.0), disk_epsilon=0.1)

        self.assertAlmostEqual(float(np.linalg.norm(projected)), 0.9, places=14)
        self.assertLess(float(np.linalg.norm(projected)), 1.0)

    def test_zero_vector_is_safe(self):
        np.testing.assert_array_equal(
            project_to_unit_disk((0.0, 0.0), disk_epsilon=0.1),
            np.zeros(2),
        )

    def test_projection_rejects_invalid_points_and_epsilon(self):
        for point in ((0.0,), (float("nan"), 0.0), (float("inf"), 0.0)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                project_to_unit_disk(point)

        for epsilon in (0.0, -1.0, 1.0, float("nan"), float("inf"), True):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                project_to_unit_disk((2.0, 0.0), disk_epsilon=epsilon)

    def test_projection_handles_huge_finite_coordinates_without_mutation(self):
        original = np.array((1e308, -1e308))
        snapshot = original.copy()

        projected = project_to_unit_disk(original, disk_epsilon=1e-5)

        self.assertTrue(np.isfinite(projected).all())
        self.assertLess(float(np.linalg.norm(projected)), 1.0)
        np.testing.assert_array_equal(original, snapshot)


class DevelopmentEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(6)

    def test_embedding_has_complete_finite_disk_coverage(self):
        radius = 0.8
        coordinates = embed_graph_in_poincare_disk(
            self.graph,
            seed=1234,
            embedding_radius=radius,
            iterations=80,
        )

        self.assertEqual(set(coordinates), set(self.graph.nodes()))
        for coordinate in coordinates.values():
            self.assertEqual(coordinate.shape, (2,))
            self.assertTrue(np.isfinite(coordinate).all())
            self.assertLess(float(np.linalg.norm(coordinate)), 1.0)
        self.assertAlmostEqual(
            max(float(np.linalg.norm(point)) for point in coordinates.values()),
            radius,
            places=12,
        )
        np.testing.assert_allclose(
            np.mean(np.vstack(list(coordinates.values())), axis=0),
            np.zeros(2),
            atol=1e-14,
        )

    def test_embedding_is_deterministic_and_uses_stable_node_order(self):
        first = embed_graph_in_poincare_disk(self.graph, seed=77, iterations=60)
        reversed_graph = nx.Graph()
        reversed_graph.add_nodes_from(reversed(tuple(self.graph.nodes())))
        reversed_graph.add_edges_from(reversed(tuple(self.graph.edges())))
        second = embed_graph_in_poincare_disk(reversed_graph, seed=77, iterations=60)

        self.assertEqual(tuple(first), tuple(sorted(self.graph.nodes())))
        for node in self.graph:
            np.testing.assert_array_equal(first[node], second[node])

    def test_embedding_method_is_explicit_and_recorded(self):
        embed_graph_in_poincare_disk(self.graph, seed=77, iterations=2)

        metadata = development_embedding_metadata()
        self.assertEqual(metadata["embedding_id"], DEVELOPMENT_EMBEDDING_ID)
        self.assertEqual(metadata["layout_method"], FORCE_LAYOUT_IMPLEMENTATION)
        self.assertIn("not_approved", metadata["scientific_status"])

    def test_one_embedding_implementation_covers_every_configured_size(self):
        configured_sizes = (30, 100, 300, 1_000)
        original_import = __import__

        def import_without_scipy(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise AssertionError("embedding attempted an undeclared SciPy import")
            return original_import(name, *args, **kwargs)

        with patch(
            "embedding._dependency_free_force_layout",
            wraps=embedding_module._dependency_free_force_layout,
        ) as force_layout:
            for node_count in configured_sizes:
                with self.subTest(node_count=node_count):
                    graph = nx.path_graph(node_count)
                    nodes_before = tuple(graph.nodes())
                    edges_before = tuple(graph.edges())
                    with patch("builtins.__import__", side_effect=import_without_scipy):
                        first = embed_graph_in_poincare_disk(
                            graph,
                            seed=19,
                            iterations=1,
                        )
                        second = embed_graph_in_poincare_disk(
                            graph,
                            seed=19,
                            iterations=1,
                        )

                    self.assertEqual(len(first), node_count)
                    self.assertEqual(tuple(graph.nodes()), nodes_before)
                    self.assertEqual(tuple(graph.edges()), edges_before)
                    for node in graph:
                        np.testing.assert_array_equal(first[node], second[node])
                        self.assertTrue(np.isfinite(first[node]).all())
                        self.assertLess(float(np.linalg.norm(first[node])), 1.0)

        self.assertEqual(force_layout.call_count, 2 * len(configured_sizes))
        self.assertTrue(
            all(call.kwargs["iterations"] == 1 for call in force_layout.call_args_list)
        )
        metadata = development_embedding_metadata()
        self.assertEqual(metadata["embedding_id"], DEVELOPMENT_EMBEDDING_ID)
        self.assertEqual(metadata["networkx_version"], nx.__version__)

    def test_different_seeds_may_produce_different_layouts(self):
        first = embed_graph_in_poincare_disk(self.graph, seed=10, iterations=40)
        second = embed_graph_in_poincare_disk(self.graph, seed=11, iterations=40)

        self.assertTrue(
            any(
                not np.allclose(first[node], second[node], rtol=0.0, atol=1e-10)
                for node in self.graph
            )
        )

    def test_embedding_does_not_mutate_graph(self):
        self.graph.graph["label"] = "original"
        self.graph.nodes[0]["kind"] = "endpoint"
        self.graph.edges[0, 1]["weight"] = 7
        graph_attributes = dict(self.graph.graph)
        nodes = list(self.graph.nodes(data=True))
        edges = list(self.graph.edges(data=True))

        embed_graph_in_poincare_disk(self.graph, seed=4)

        self.assertEqual(self.graph.graph, graph_attributes)
        self.assertEqual(list(self.graph.nodes(data=True)), nodes)
        self.assertEqual(list(self.graph.edges(data=True)), edges)

    def test_invalid_embedding_settings_are_rejected(self):
        invalid_cases = (
            {"seed": -1},
            {"seed": MAX_RANDOM_SEED + 1},
            {"seed": 1.5},
            {"seed": True},
            {"seed": 1, "embedding_radius": 0.0},
            {"seed": 1, "embedding_radius": -0.1},
            {"seed": 1, "embedding_radius": float("nan")},
            {"seed": 1, "embedding_radius": 0.999999, "disk_epsilon": 1e-6},
            {"seed": 1, "iterations": 0},
            {"seed": 1, "iterations": -1},
            {"seed": 1, "iterations": 1.5},
            {"seed": 1, "iterations": True},
            {"seed": 1, "disk_epsilon": 0.0},
            {"seed": 1, "disk_epsilon": 1.0},
            {"seed": 1, "tolerance": 0.0},
            {"seed": 1, "tolerance": float("nan")},
        )

        for arguments in invalid_cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                embed_graph_in_poincare_disk(self.graph, **arguments)

    def test_empty_and_directed_graphs_are_rejected(self):
        for graph in (nx.Graph(), nx.path_graph(3, create_using=nx.DiGraph)):
            with self.subTest(graph=type(graph).__name__), self.assertRaises(ValueError):
                embed_graph_in_poincare_disk(graph, seed=1)

    def test_single_node_is_embedded_at_origin(self):
        coordinates = embed_graph_in_poincare_disk(nx.empty_graph(1), seed=9)

        self.assertEqual(set(coordinates), {0})
        np.testing.assert_array_equal(coordinates[0], np.zeros(2))


class EmbeddingDistortionTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(5)
        self.coordinates = embed_graph_in_poincare_disk(
            self.graph,
            seed=321,
            iterations=60,
        )

    def test_distortion_is_finite_non_negative_and_uses_all_pairs(self):
        result = calculate_embedding_distortion(self.graph, self.coordinates)

        self.assertEqual(result.unordered_pair_count, 5 * 4 // 2)
        self.assertGreater(result.fitted_scale_alpha, 0.0)
        self.assertGreaterEqual(result.mean_relative_distortion, 0.0)
        self.assertGreaterEqual(result.rmse_relative_distortion, 0.0)
        for value in (
            result.fitted_scale_alpha,
            result.mean_relative_distortion,
            result.rmse_relative_distortion,
        ):
            self.assertTrue(isfinite(value))

    def test_distortion_is_deterministic(self):
        first = calculate_embedding_distortion(self.graph, self.coordinates)
        second = calculate_embedding_distortion(self.graph, self.coordinates)

        self.assertEqual(first, second)

    def test_distortion_reuses_validated_shortest_paths(self):
        shortest_paths = prepare_all_pairs_shortest_paths(self.graph)

        direct = calculate_embedding_distortion(self.graph, self.coordinates)
        reused = calculate_embedding_distortion(
            self.graph,
            self.coordinates,
            shortest_paths=shortest_paths,
        )

        self.assertEqual(reused, direct)
        with self.assertRaisesRegex(ValueError, "topology"):
            calculate_embedding_distortion(
                nx.star_graph(4),
                self.coordinates,
                shortest_paths=shortest_paths,
            )

    def test_missing_and_extra_coordinates_are_rejected(self):
        missing = dict(self.coordinates)
        missing.pop(4)
        extra = dict(self.coordinates)
        extra[10] = np.zeros(2)

        for coordinates in (missing, extra):
            with self.subTest(nodes=set(coordinates)), self.assertRaises(ValueError):
                calculate_embedding_distortion(self.graph, coordinates)

    def test_invalid_coordinates_are_rejected(self):
        invalid_points = ((1.0, 0.0), (float("nan"), 0.0), (0.0,))

        for invalid_point in invalid_points:
            coordinates = dict(self.coordinates)
            coordinates[0] = invalid_point
            with self.subTest(point=invalid_point), self.assertRaises(ValueError):
                calculate_embedding_distortion(self.graph, coordinates)

    def test_disconnected_graph_is_rejected(self):
        graph = nx.Graph([(0, 1), (2, 3)])
        coordinates = {
            0: (-0.3, 0.0),
            1: (-0.1, 0.0),
            2: (0.1, 0.0),
            3: (0.3, 0.0),
        }

        with self.assertRaises(ValueError):
            calculate_embedding_distortion(graph, coordinates)

    def test_two_node_exact_scale_fit_has_zero_distortion(self):
        graph = nx.path_graph(2)
        coordinates = {0: (0.0, 0.0), 1: (tanh(0.5), 0.0)}

        result = calculate_embedding_distortion(graph, coordinates)

        self.assertEqual(result.unordered_pair_count, 1)
        self.assertAlmostEqual(result.fitted_scale_alpha, 1.0, places=12)
        self.assertAlmostEqual(result.mean_relative_distortion, 0.0, places=12)
        self.assertAlmostEqual(result.rmse_relative_distortion, 0.0, places=12)

    def test_fitted_alpha_minimises_relative_squared_error(self):
        result = calculate_embedding_distortion(self.graph, self.coordinates)
        shortest_paths = dict(nx.all_pairs_shortest_path_length(self.graph))
        ordered_nodes = sorted(self.graph)
        ratios = []

        for left_index, left in enumerate(ordered_nodes[:-1]):
            for right in ordered_nodes[left_index + 1 :]:
                ratios.append(
                    poincare_distance(
                        self.coordinates[left],
                        self.coordinates[right],
                    )
                    / shortest_paths[left][right]
                )

        def objective(alpha):
            return sum((alpha * ratio - 1.0) ** 2 for ratio in ratios)

        alpha = result.fitted_scale_alpha
        self.assertAlmostEqual(
            alpha,
            sum(ratios) / sum(ratio**2 for ratio in ratios),
            places=14,
        )
        self.assertLessEqual(objective(alpha), objective(alpha * 0.99))
        self.assertLessEqual(objective(alpha), objective(alpha * 1.01))

    def test_distortion_does_not_mutate_graph_or_coordinates(self):
        nodes = list(self.graph.nodes(data=True))
        edges = list(self.graph.edges(data=True))
        coordinate_snapshot = {
            node: coordinate.copy() for node, coordinate in self.coordinates.items()
        }

        calculate_embedding_distortion(self.graph, self.coordinates)

        self.assertEqual(list(self.graph.nodes(data=True)), nodes)
        self.assertEqual(list(self.graph.edges(data=True)), edges)
        for node, coordinate in self.coordinates.items():
            np.testing.assert_array_equal(coordinate, coordinate_snapshot[node])

    def test_zero_geometric_distances_cannot_fit_positive_scale(self):
        coordinates = {node: np.zeros(2) for node in self.graph}

        with self.assertRaises(ValueError):
            calculate_embedding_distortion(self.graph, coordinates)

    def test_some_coincident_coordinates_are_accepted(self):
        coordinates = dict(self.coordinates)
        coordinates[4] = coordinates[0].copy()

        result = calculate_embedding_distortion(self.graph, coordinates)

        self.assertEqual(result.unordered_pair_count, 10)
        self.assertTrue(isfinite(result.fitted_scale_alpha))
        self.assertTrue(isfinite(result.mean_relative_distortion))
        self.assertTrue(isfinite(result.rmse_relative_distortion))

    def test_single_node_and_invalid_tolerance_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_embedding_distortion(nx.empty_graph(1), {0: (0.0, 0.0)})
        for tolerance in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                calculate_embedding_distortion(
                    self.graph,
                    self.coordinates,
                    tolerance=tolerance,
                )


if __name__ == "__main__":
    unittest.main()
