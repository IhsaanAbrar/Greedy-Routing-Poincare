from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from embedding import (  # noqa: E402
    canonical_eigenvectors_for_indices,
    embedding_input_from_distance_matrix,
    prepare_embedding_input,
)
import hydra_embedding as hydra_module  # noqa: E402
from hydra_embedding import (  # noqa: E402
    HYDRA_CONDITION_ID,
    HYDRA_CURVATURE,
    HYDRA_DIMENSION,
    HYDRA_KAPPA,
    HydraCenteringError,
    HydraEmbeddingError,
    _mobius_add,
    _pairwise_poincare_distances,
    _standard_hydra_coordinates,
    embed_hydra,
    hyperbolic_frechet_mean,
)
from network_metrics import prepare_all_pairs_shortest_paths  # noqa: E402
from poincare_distance import poincare_distance  # noqa: E402


def distance_matrix(points):
    return np.asarray(
        [
            [poincare_distance(left, right) for right in points]
            for left in points
        ],
        dtype=float,
    )


def make_input(matrix, nodes=None):
    node_order = tuple(range(len(matrix))) if nodes is None else tuple(nodes)
    return embedding_input_from_distance_matrix(
        matrix,
        node_order,
        configuration_fingerprint="hydra-test-config",
        tolerance=1e-12,
    )


class HydraInputTests(unittest.TestCase):
    def test_distance_input_validation(self):
        valid = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ]
        )
        invalid = (
            np.ones((2, 3)),
            np.array([[0.0, 1.0], [2.0, 0.0]]),
            np.array([[0.1, 1.0], [1.0, 0.0]]),
            np.array([[0.0, -1.0], [-1.0, 0.0]]),
            np.array([[0.0, np.inf], [np.inf, 0.0]]),
            np.array([[0.0, np.nan], [np.nan, 0.0]]),
        )
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                make_input(matrix)

        with self.assertRaises(ValueError):
            make_input(valid, nodes=(0, 1))
        with self.assertRaises(ValueError):
            make_input(valid, nodes=(0, 0, 1))

    def test_embedding_input_is_immutable_and_fingerprinted(self):
        matrix = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        snapshot = matrix.copy()
        first = make_input(matrix, nodes=("c", "a", "b"))
        second = make_input(matrix, nodes=("c", "a", "b"))

        np.testing.assert_array_equal(matrix, snapshot)
        self.assertFalse(first.distance_matrix.flags.writeable)
        self.assertEqual(first.input_fingerprint, second.input_fingerprint)
        self.assertEqual(first.node_order, ("c", "a", "b"))
        with self.assertRaises(ValueError):
            first.distance_matrix[0, 1] = 2.0

    def test_unstable_or_mixed_node_labels_are_rejected_before_fingerprinting(self):
        class UnstableCustomLabel:
            pass

        matrix = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        invalid_node_orders = (
            (0, "1", 2),
            (False, True, False),
            (
                UnstableCustomLabel(),
                UnstableCustomLabel(),
                UnstableCustomLabel(),
            ),
        )
        for node_order in invalid_node_orders:
            with (
                self.subTest(node_order=node_order),
                self.assertRaisesRegex(ValueError, "node labels"),
            ):
                make_input(matrix, nodes=node_order)

    def test_repeated_eigenspace_basis_is_rotation_invariant(self):
        eigenvalues = np.array([4.0, 2.0, 2.0, 0.5])
        identity = np.eye(4)
        angle = 0.37
        rotated = identity.copy()
        rotated[:, 1:3] = identity[:, 1:3] @ np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

        first, first_groups = canonical_eigenvectors_for_indices(
            eigenvalues,
            identity,
            (1, 2),
            relative_tolerance=1e-12,
        )
        second, second_groups = canonical_eigenvectors_for_indices(
            eigenvalues,
            rotated,
            (1, 2),
            relative_tolerance=1e-12,
        )

        np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-14)
        self.assertEqual(first_groups, (2,))
        self.assertEqual(second_groups, (2,))


class StandardHydraTests(unittest.TestCase):
    def setUp(self):
        self.original_points = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.3],
                [-0.1, -0.15],
                [0.25, -0.2],
            ],
            dtype=float,
        )
        self.input = make_input(distance_matrix(self.original_points))

    def test_exact_hyperbolic_distance_recovery_and_metadata(self):
        result = embed_hydra(self.input)
        recovered = np.asarray(
            [result.coordinates[node] for node in self.input.node_order]
        )
        recovered_distances = _pairwise_poincare_distances(recovered)

        np.testing.assert_allclose(
            recovered_distances,
            self.input.distance_matrix,
            rtol=0.0,
            atol=2e-12,
        )
        metadata = result.metadata
        self.assertEqual(metadata.coordinate_condition_id, HYDRA_CONDITION_ID)
        self.assertEqual(metadata.dimension, HYDRA_DIMENSION)
        self.assertEqual(metadata.kappa, HYDRA_KAPPA)
        self.assertEqual(metadata.curvature, HYDRA_CURVATURE)
        self.assertEqual(metadata.node_order, self.input.node_order)
        self.assertEqual(
            metadata.embedding_input_fingerprint,
            self.input.input_fingerprint,
        )
        self.assertEqual(
            metadata.configuration_fingerprint,
            self.input.configuration_fingerprint,
        )
        self.assertFalse(metadata.radial_rescaling_after_centering)
        self.assertFalse(metadata.boundary_correction_occurred)
        self.assertEqual(metadata.eigenvalue_tolerance, 1e-12)
        self.assertLess(metadata.final_frechet_mean_residual, 1e-10)
        self.assertLess(metadata.maximum_pairwise_distance_error, 2e-12)

    def test_standard_algorithm_b2_radial_projection_regression(self):
        graph = nx.cycle_graph(5)
        embedding_input = prepare_embedding_input(
            graph,
            prepare_all_pairs_shortest_paths(graph),
            configuration_fingerprint="cycle-five",
        )
        coordinates, _, _, group_sizes = _standard_hydra_coordinates(
            embedding_input.distance_matrix,
            eigenvalue_tolerance=1e-12,
        )
        radii = np.linalg.norm(coordinates, axis=1)

        np.testing.assert_allclose(
            radii,
            np.full(5, 0.45558646),
            rtol=0.0,
            atol=1e-8,
        )
        self.assertTrue(group_sizes)

    def test_standard_algorithm_b2_uses_minimum_time_when_below_one(self):
        graph = nx.star_graph(5)
        embedding_input = prepare_embedding_input(
            graph,
            prepare_all_pairs_shortest_paths(graph),
            configuration_fingerprint="star-six-xmin-branch",
        )
        transformed = np.cosh(embedding_input.distance_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(transformed)
        leading_index = int(np.argmax(eigenvalues))
        leading = eigenvectors[:, leading_index].copy()
        if float(np.sum(leading)) < 0.0:
            leading *= -1.0
        time = np.sqrt(float(eigenvalues[leading_index])) * leading
        minimum_time = float(np.min(time))
        self.assertLess(minimum_time, 1.0)
        expected_radii = np.sqrt(
            np.maximum(
                (time - minimum_time) / (time + minimum_time),
                0.0,
            )
        )

        coordinates, _, _, _ = _standard_hydra_coordinates(
            embedding_input.distance_matrix,
            eigenvalue_tolerance=1e-12,
        )

        np.testing.assert_allclose(
            np.linalg.norm(coordinates, axis=1),
            expected_radii,
            rtol=0.0,
            atol=2e-14,
        )
        self.assertEqual(float(np.min(expected_radii)), 0.0)

    def test_output_is_deterministic_and_strictly_inside_disk(self):
        first = embed_hydra(self.input)
        second = embed_hydra(self.input)

        self.assertEqual(first, second)
        for point in first.coordinates.values():
            self.assertTrue(np.isfinite(point).all())
            self.assertLess(float(np.linalg.norm(point)), 1.0)

    def test_stable_node_order_and_graph_immutability(self):
        graph = nx.Graph()
        graph.add_nodes_from((5, 3, 4, 2, 1, 0))
        graph.add_edges_from(nx.cycle_graph(6).edges())
        nodes_before = tuple(graph.nodes(data=True))
        edges_before = tuple(graph.edges(data=True))
        embedding_input = prepare_embedding_input(
            graph,
            prepare_all_pairs_shortest_paths(graph),
            configuration_fingerprint="stable-order",
        )
        matrix_before = embedding_input.distance_matrix.copy()

        result = embed_hydra(embedding_input)

        self.assertEqual(tuple(result.coordinates), tuple(range(6)))
        self.assertEqual(tuple(graph.nodes(data=True)), nodes_before)
        self.assertEqual(tuple(graph.edges(data=True)), edges_before)
        np.testing.assert_array_equal(
            embedding_input.distance_matrix,
            matrix_before,
        )

    def test_frechet_mean_recovers_known_hyperbolic_translation(self):
        symmetric = np.array(
            [[0.4, 0.0], [-0.4, 0.0], [0.0, 0.2], [0.0, -0.2]]
        )
        translation = np.array([0.3, 0.2])
        translated = _mobius_add(translation, symmetric)

        mean, iterations, residual = hyperbolic_frechet_mean(
            translated,
            tolerance=1e-10,
            max_iterations=256,
        )

        np.testing.assert_allclose(mean, translation, rtol=0.0, atol=2e-10)
        self.assertLessEqual(iterations, 8)
        self.assertLessEqual(residual, 1e-10)

    def test_centering_preserves_every_pairwise_distance(self):
        uncentered, _, _, _ = _standard_hydra_coordinates(
            self.input.distance_matrix,
            eigenvalue_tolerance=1e-12,
        )
        before = _pairwise_poincare_distances(uncentered)
        result = embed_hydra(self.input)
        centered = np.asarray(
            [result.coordinates[node] for node in self.input.node_order]
        )
        after = _pairwise_poincare_distances(centered)

        np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)

    def test_boundary_roundoff_correction_is_recorded_without_global_rescaling(self):
        fixture = make_input(
            np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ]
            )
        )
        uncentered = np.array(
            [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]],
            dtype=float,
        )
        rounded_centered = np.array(
            [[1.0, 0.0], [0.2, 0.0], [0.0, 0.2]],
            dtype=float,
        )
        zero_distances = np.zeros((3, 3), dtype=float)
        with (
            patch.object(
                hydra_module,
                "_standard_hydra_coordinates",
                return_value=(uncentered, 1.0, (-0.2, -0.1), ()),
            ),
            patch.object(
                hydra_module,
                "hyperbolic_frechet_mean",
                return_value=(np.zeros(2), 1, 0.0),
            ),
            patch.object(
                hydra_module,
                "_mobius_add",
                side_effect=lambda *_: rounded_centered.copy(),
            ),
            patch.object(
                hydra_module,
                "_frechet_residual",
                return_value=(np.zeros(2), 0.0),
            ),
            patch.object(
                hydra_module,
                "_pairwise_poincare_distances",
                side_effect=(zero_distances, zero_distances),
            ),
        ):
            result = embed_hydra(fixture)

        self.assertTrue(result.metadata.boundary_correction_occurred)
        self.assertEqual(result.metadata.boundary_correction_count, 1)
        self.assertFalse(result.metadata.radial_rescaling_after_centering)
        self.assertEqual(
            result.coordinates[0][0],
            float(np.nextafter(1.0, 0.0)),
        )
        self.assertLess(float(np.linalg.norm(result.coordinates[0])), 1.0)

    def test_clear_invalid_parameter_and_failure_behaviour(self):
        with self.assertRaises(ValueError):
            embed_hydra(self.input, dimension=3)
        with self.assertRaises(ValueError):
            embed_hydra(self.input, kappa=2.0)
        with self.assertRaises(ValueError):
            embed_hydra(self.input, boundary_roundoff_tolerance=0.1)
        with self.assertRaises(HydraCenteringError):
            hyperbolic_frechet_mean(
                np.array([[0.9, 0.0], [-0.2, 0.3], [0.1, -0.5]]),
                tolerance=1e-15,
                max_iterations=1,
            )

        huge = make_input(
            np.array(
                [
                    [0.0, 1_000.0, 1_000.0],
                    [1_000.0, 0.0, 1_000.0],
                    [1_000.0, 1_000.0, 0.0],
                ]
            )
        )
        with self.assertRaisesRegex(HydraEmbeddingError, "overflowed"):
            embed_hydra(huge)

    def test_eigendecomposition_failure_is_wrapped(self):
        with patch(
            "hydra_embedding.np.linalg.eigh",
            side_effect=np.linalg.LinAlgError("fixture"),
        ):
            with self.assertRaisesRegex(
                HydraEmbeddingError,
                "eigendecomposition did not converge",
            ):
                embed_hydra(self.input)


if __name__ == "__main__":
    unittest.main()
