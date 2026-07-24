from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from embedding import (  # noqa: E402
    calculate_embedding_distortion,
    embedding_input_from_distance_matrix,
    prepare_embedding_input,
)
from mds_embedding import (  # noqa: E402
    MDS_CONDITION_IDS,
    MDS_MAXIMUM_RADII,
    ClassicalMDSError,
    classical_mds,
    create_mds_sensitivity_conditions,
    scale_equivariant_euclidean_routing_tolerance,
)
from network_metrics import prepare_all_pairs_shortest_paths  # noqa: E402
from poincare_distance import euclidean_distance, poincare_distance  # noqa: E402
from routing import (  # noqa: E402
    LOCAL_MINIMUM,
    euclidean_greedy_route,
    hyperbolic_greedy_route,
    prepare_routing_coordinates,
)


def euclidean_distance_matrix(points):
    points = np.asarray(points, dtype=float)
    return np.linalg.norm(
        points[:, np.newaxis, :] - points[np.newaxis, :, :],
        axis=-1,
    )


def make_input(matrix, nodes=None):
    node_order = tuple(range(len(matrix))) if nodes is None else tuple(nodes)
    return embedding_input_from_distance_matrix(
        matrix,
        node_order,
        configuration_fingerprint="mds-test-config",
        tolerance=1e-12,
    )


class ClassicalMDSTests(unittest.TestCase):
    def setUp(self):
        self.original_points = np.array(
            [
                [-1.2, 0.3],
                [0.1, -0.8],
                [1.4, 0.5],
                [0.6, 1.7],
                [-0.7, 1.1],
            ],
            dtype=float,
        )
        self.input = make_input(euclidean_distance_matrix(self.original_points))

    def test_invalid_distance_inputs_are_rejected(self):
        invalid_matrices = (
            np.ones((2, 3)),
            np.array([[0.0, 1.0], [2.0, 0.0]]),
            np.array([[0.0, np.inf], [np.inf, 0.0]]),
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                make_input(matrix)

    def test_known_two_dimensional_configuration_is_recovered(self):
        result = classical_mds(self.input)
        recovered = np.asarray(
            [result.coordinates[node] for node in self.input.node_order]
        )

        np.testing.assert_allclose(
            euclidean_distance_matrix(recovered),
            self.input.distance_matrix,
            rtol=0.0,
            atol=2e-12,
        )
        self.assertEqual(result.metadata.embedding_version, "classical_mds_2d_v1")
        self.assertEqual(result.metadata.dimension, 2)
        self.assertEqual(result.metadata.node_order, self.input.node_order)
        self.assertEqual(
            result.metadata.configuration_fingerprint,
            self.input.configuration_fingerprint,
        )
        self.assertEqual(
            result.metadata.embedding_input_fingerprint,
            self.input.input_fingerprint,
        )
        self.assertGreaterEqual(
            result.metadata.retained_eigenvalues[0],
            result.metadata.retained_eigenvalues[1],
        )
        self.assertLess(result.metadata.centroid_residual, 1e-12)
        self.assertEqual(
            (
                result.metadata.coincident_coordinate_group_count,
                result.metadata.coincident_vertex_count,
                result.metadata.coincident_vertex_pair_count,
            ),
            (0, 0, 0),
        )

    def test_double_centering_matches_coordinate_gram_matrix(self):
        result = classical_mds(self.input)
        point_count = len(self.input.node_order)
        centering = np.eye(point_count) - np.ones(
            (point_count, point_count)
        ) / point_count
        expected_gram = (
            -0.5
            * centering
            @ np.square(self.input.distance_matrix)
            @ centering
        )

        np.testing.assert_allclose(
            result.coordinate_matrix @ result.coordinate_matrix.T,
            expected_gram,
            rtol=0.0,
            atol=3e-12,
        )

    def test_output_is_deterministic_centered_and_immutable(self):
        matrix_before = self.input.distance_matrix.copy()
        first = classical_mds(self.input)
        second = classical_mds(self.input)

        self.assertEqual(first.coordinates, second.coordinates)
        np.testing.assert_array_equal(
            first.coordinate_matrix,
            second.coordinate_matrix,
        )
        np.testing.assert_allclose(
            np.mean(first.coordinate_matrix, axis=0),
            np.zeros(2),
            rtol=0.0,
            atol=1e-14,
        )
        self.assertFalse(first.coordinate_matrix.flags.writeable)
        np.testing.assert_array_equal(self.input.distance_matrix, matrix_before)

    def test_repeated_retained_eigenspace_is_rotation_invariant(self):
        square = np.array(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ]
        )
        embedding_input = make_input(euclidean_distance_matrix(square))
        point_count = len(square)
        centering = np.eye(point_count) - np.ones(
            (point_count, point_count)
        ) / point_count
        gram = (
            -0.5
            * centering
            @ np.square(embedding_input.distance_matrix)
            @ centering
        )
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        angle = 0.41
        rotated = eigenvectors.copy()
        rotated[:, -2:] = eigenvectors[:, -2:] @ np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

        with patch(
            "mds_embedding.np.linalg.eigh",
            return_value=(eigenvalues, eigenvectors),
        ):
            first = classical_mds(embedding_input)
        with patch(
            "mds_embedding.np.linalg.eigh",
            return_value=(eigenvalues, rotated),
        ):
            second = classical_mds(embedding_input)

        np.testing.assert_allclose(
            first.coordinate_matrix,
            second.coordinate_matrix,
            rtol=0.0,
            atol=1e-13,
        )
        self.assertEqual(
            first.metadata.canonicalized_eigenspace_group_sizes,
            (2,),
        )

    def test_eigenvalue_tolerance_classifies_tiny_and_significant_negatives(self):
        fixture = make_input(
            np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ]
            )
        )
        eigenvectors = np.eye(3)
        cases = (
            (np.array([-1e-14, 0.5, 1.0]), 1, 0),
            (np.array([-0.1, 0.5, 1.0]), 0, 1),
        )
        for eigenvalues, tiny_count, significant_count in cases:
            with (
                self.subTest(eigenvalues=eigenvalues),
                patch(
                    "mds_embedding.np.linalg.eigh",
                    return_value=(eigenvalues, eigenvectors),
                ),
            ):
                result = classical_mds(
                    fixture,
                    eigenvalue_relative_tolerance=1e-12,
                )
                self.assertEqual(
                    result.metadata.tiny_negative_eigenvalue_count,
                    tiny_count,
                )
                self.assertEqual(
                    result.metadata.discarded_significant_negative_count,
                    significant_count,
                )

    def test_insufficient_positive_dimensions_fail_clearly(self):
        collinear = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        with self.assertRaisesRegex(
            ClassicalMDSError,
            "at least two eigenvalues",
        ):
            classical_mds(make_input(euclidean_distance_matrix(collinear)))

    def test_four_exact_nested_radius_transformations(self):
        base = classical_mds(self.input)
        conditions = create_mds_sensitivity_conditions(base)

        self.assertEqual(
            tuple(item.metadata.coordinate_condition_id for item in conditions),
            MDS_CONDITION_IDS,
        )
        self.assertEqual(
            tuple(item.metadata.requested_maximum_radius for item in conditions),
            MDS_MAXIMUM_RADII,
        )
        for condition, radius in zip(conditions, MDS_MAXIMUM_RADII, strict=True):
            points = np.asarray(list(condition.coordinates.values()))
            self.assertAlmostEqual(
                float(np.max(np.linalg.norm(points, axis=1))),
                radius,
                places=12,
            )
            self.assertTrue(np.all(np.linalg.norm(points, axis=1) < 1.0))
            self.assertTrue(condition.metadata.nested_sensitivity_condition)
            self.assertEqual(
                condition.metadata.embedding_input_fingerprint,
                self.input.input_fingerprint,
            )
            self.assertEqual(
                condition.metadata.configuration_fingerprint,
                self.input.configuration_fingerprint,
            )
            self.assertEqual(
                (
                    condition.metadata.coincident_coordinate_group_count,
                    condition.metadata.coincident_vertex_count,
                    condition.metadata.coincident_vertex_pair_count,
                ),
                (0, 0, 0),
            )

    def test_k6_coincidences_are_recorded_and_preserved_without_jitter(self):
        graph = nx.complete_graph(6)
        embedding_input = prepare_embedding_input(
            graph,
            prepare_all_pairs_shortest_paths(graph),
            configuration_fingerprint="k6-coincidence-regression",
        )

        base = classical_mds(embedding_input)
        coordinate_groups = {}
        for node, point in base.coordinates.items():
            coordinate_groups.setdefault(point, []).append(node)
        coincident_groups = tuple(
            tuple(nodes)
            for nodes in coordinate_groups.values()
            if len(nodes) > 1
        )
        expected_counts = (
            len(coincident_groups),
            sum(len(group) for group in coincident_groups),
            sum(
                len(group) * (len(group) - 1) // 2
                for group in coincident_groups
            ),
        )

        self.assertEqual(expected_counts, (1, 2, 1))
        self.assertEqual(
            (
                base.metadata.coincident_coordinate_group_count,
                base.metadata.coincident_vertex_count,
                base.metadata.coincident_vertex_pair_count,
            ),
            expected_counts,
        )
        conditions = create_mds_sensitivity_conditions(base)
        coincident_nodes = coincident_groups[0]
        for condition in conditions:
            self.assertEqual(
                (
                    condition.metadata.coincident_coordinate_group_count,
                    condition.metadata.coincident_vertex_count,
                    condition.metadata.coincident_vertex_pair_count,
                ),
                expected_counts,
            )
            self.assertEqual(
                condition.coordinates[coincident_nodes[0]],
                condition.coordinates[coincident_nodes[1]],
            )
            distortion = calculate_embedding_distortion(
                graph,
                condition.coordinates,
                shortest_paths=prepare_all_pairs_shortest_paths(graph),
            )
            self.assertTrue(np.isfinite(distortion.mean_relative_distortion))

    def test_twin_vertices_can_remain_coincident(self):
        graph = nx.Graph(
            ((0, 5), (1, 5), (2, 5), (3, 5), (4, 5))
        )
        self.assertEqual(set(graph.neighbors(2)), set(graph.neighbors(4)))
        embedding_input = prepare_embedding_input(
            graph,
            prepare_all_pairs_shortest_paths(graph),
            configuration_fingerprint="twin-leaf-regression",
        )

        base = classical_mds(embedding_input)

        self.assertGreaterEqual(
            base.metadata.coincident_coordinate_group_count,
            1,
        )
        self.assertGreaterEqual(base.metadata.coincident_vertex_count, 2)
        self.assertGreaterEqual(base.metadata.coincident_vertex_pair_count, 1)

    def test_zero_distance_between_vertices_routes_deterministically(self):
        graph = nx.Graph(((0, 1), (0, 2), (1, 3), (2, 3)))
        coordinates = {
            0: (0.5, 0.0),
            1: (0.0, 0.0),
            2: (0.0, 0.0),
            3: (0.0, 0.0),
        }
        cases = (
            (euclidean_distance, euclidean_greedy_route, "euclidean"),
            (poincare_distance, hyperbolic_greedy_route, "poincare"),
        )
        for distance_function, route_function, metric_name in cases:
            with self.subTest(metric=metric_name):
                prepared = prepare_routing_coordinates(
                    graph,
                    coordinates,
                    distance_function,
                    metric_name=metric_name,
                )
                first = route_function(graph, prepared, 0, 3)
                second = route_function(graph, prepared, 0, 3)

                self.assertEqual(first, second)
                self.assertEqual(first.walk, (0, 1))
                self.assertEqual(first.failure_type, LOCAL_MINIMUM)

    def test_uniform_scaling_preserves_rankings_and_greedy_routes(self):
        graph = nx.cycle_graph(8)
        embedding_input = prepare_embedding_input(
            graph,
            prepare_all_pairs_shortest_paths(graph),
            configuration_fingerprint="cycle-eight",
        )
        conditions = create_mds_sensitivity_conditions(
            classical_mds(embedding_input)
        )
        normalized_distances = []
        walks = []
        for condition, radius in zip(
            conditions,
            MDS_MAXIMUM_RADII,
            strict=True,
        ):
            points = np.asarray(
                [condition.coordinates[node] for node in embedding_input.node_order]
            )
            pairwise = euclidean_distance_matrix(points)
            normalized_distances.append(pairwise / radius)
            prepared = prepare_routing_coordinates(
                graph,
                condition.coordinates,
                euclidean_distance,
                metric_name="euclidean",
            )
            walks.append(
                euclidean_greedy_route(
                    graph,
                    prepared,
                    0,
                    3,
                    tolerance=1e-12,
                ).walk
            )

        for distances in normalized_distances[1:]:
            np.testing.assert_allclose(
                distances,
                normalized_distances[0],
                rtol=1e-12,
                atol=1e-12,
            )
        self.assertTrue(all(walk == walks[0] for walk in walks))

    def test_scale_equivariant_tolerance_preserves_a_near_tie_decision(self):
        graph = nx.Graph(((0, 1), (0, 2), (1, 3), (2, 3)))
        reference_coordinates = {
            0: (0.0, 0.95),
            1: (0.2000000000015, 0.0),
            2: (0.2, 0.0),
            3: (0.0, 0.0),
        }
        fixed_tolerance_walks = []
        scaled_tolerance_walks = []
        for radius in (0.50, 0.95):
            factor = radius / 0.95
            coordinates = {
                node: (point[0] * factor, point[1] * factor)
                for node, point in reference_coordinates.items()
            }
            prepared = prepare_routing_coordinates(
                graph,
                coordinates,
                euclidean_distance,
                metric_name="euclidean",
            )
            fixed_tolerance_walks.append(
                euclidean_greedy_route(
                    graph,
                    prepared,
                    0,
                    3,
                    tolerance=1e-12,
                ).walk
            )
            scaled_tolerance_walks.append(
                euclidean_greedy_route(
                    graph,
                    prepared,
                    0,
                    3,
                    tolerance=scale_equivariant_euclidean_routing_tolerance(
                        1e-12,
                        radius,
                    ),
                ).walk
            )

        self.assertNotEqual(
            fixed_tolerance_walks[0],
            fixed_tolerance_walks[1],
        )
        self.assertEqual(scaled_tolerance_walks, [(0, 2, 3), (0, 2, 3)])

    def test_graph_and_distance_input_are_not_mutated(self):
        graph = nx.cycle_graph(7)
        graph.nodes[0]["label"] = "unchanged"
        nodes_before = tuple(graph.nodes(data=True))
        edges_before = tuple(graph.edges(data=True))
        embedding_input = prepare_embedding_input(
            graph,
            prepare_all_pairs_shortest_paths(graph),
            configuration_fingerprint="immutable-cycle",
        )
        matrix_before = embedding_input.distance_matrix.copy()

        create_mds_sensitivity_conditions(classical_mds(embedding_input))

        self.assertEqual(tuple(graph.nodes(data=True)), nodes_before)
        self.assertEqual(tuple(graph.edges(data=True)), edges_before)
        np.testing.assert_array_equal(
            embedding_input.distance_matrix,
            matrix_before,
        )

    def test_invalid_parameters_and_eigendecomposition_failure(self):
        with self.assertRaises(ValueError):
            classical_mds(self.input, dimension=3)
        with self.assertRaises(ValueError):
            classical_mds(self.input, eigenvalue_relative_tolerance=0.0)
        with self.assertRaises(ValueError):
            create_mds_sensitivity_conditions(
                classical_mds(self.input),
                radii=(0.5, 0.7, 0.85, 0.9),
            )
        with patch(
            "mds_embedding.np.linalg.eigh",
            side_effect=np.linalg.LinAlgError("fixture"),
        ):
            with self.assertRaisesRegex(
                ClassicalMDSError,
                "eigendecomposition did not converge",
            ):
                classical_mds(self.input)


if __name__ == "__main__":
    unittest.main()
