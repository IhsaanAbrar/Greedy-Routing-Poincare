from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from embedding import prepare_embedding_input  # noqa: E402
import hydra_embedding as hydra_module  # noqa: E402
from hydra_embedding import (  # noqa: E402
    HydraEmbeddingError,
    _standard_hydra_coordinates,
    embed_hydra,
)
from iteration2_coordinates import (  # noqa: E402
    create_iteration2_embeddings,
    native_condition,
)
from iteration2_embedding_oracle import (  # noqa: E402
    independent_classical_mds,
)
from mds_embedding import classical_mds  # noqa: E402
from network_metrics import prepare_all_pairs_shortest_paths  # noqa: E402


def _input(graph: nx.Graph, label: str):
    return prepare_embedding_input(
        graph,
        prepare_all_pairs_shortest_paths(graph),
        configuration_fingerprint=f"excluded-iteration2-{label}",
        tolerance=1e-12,
    )


class Iteration2EmbeddingEdgeCaseTests(unittest.TestCase):
    def test_hydra_xmin_below_one_uses_the_frozen_radial_branch(self):
        embedding_input = _input(nx.star_graph(5), "xmin")
        transformed = np.cosh(embedding_input.distance_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(transformed)
        leading_index = int(np.argmax(eigenvalues))
        leading = eigenvectors[:, leading_index].copy()
        if float(np.sum(leading)) < 0.0:
            leading *= -1.0
        time = np.sqrt(float(eigenvalues[leading_index])) * leading
        xmin = float(np.min(time))
        self.assertLess(xmin, 1.0)
        expected = np.sqrt(
            np.maximum((time - xmin) / (time + xmin), 0.0)
        )
        coordinates, _, _, _, _, _ = _standard_hydra_coordinates(
            embedding_input.distance_matrix,
            eigenvalue_tolerance=1e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(coordinates, axis=1),
            expected,
            rtol=0.0,
            atol=2e-14,
        )

    def test_hydra_corrects_only_roundoff_sized_boundary_excess(self):
        embedding_input = _input(nx.complete_graph(3), "boundary")
        uncentered = np.asarray(
            [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]],
            dtype=np.float64,
        )
        within_roundoff = np.asarray(
            [[1.0 + 5e-13, 0.0], [0.2, 0.0], [0.0, 0.2]],
            dtype=np.float64,
        )
        zeros = np.zeros((3, 3), dtype=np.float64)
        common = (
            patch.object(
                hydra_module,
                "_standard_hydra_coordinates",
                return_value=(
                    uncentered,
                    1.0,
                    (-0.2, -0.1),
                    (),
                    1e-12,
                    2,
                ),
            ),
            patch.object(
                hydra_module,
                "hyperbolic_frechet_mean",
                return_value=(np.zeros(2), 1, 0.0),
            ),
            patch.object(
                hydra_module,
                "_frechet_residual",
                return_value=(np.zeros(2), 0.0),
            ),
            patch.object(
                hydra_module,
                "_pairwise_poincare_distances",
                side_effect=(zeros, zeros),
            ),
        )
        with ExitStack() as stack:
            for manager in common:
                stack.enter_context(manager)
            stack.enter_context(
                patch.object(
                    hydra_module,
                    "_mobius_add",
                    return_value=within_roundoff,
                )
            )
            corrected = embed_hydra(embedding_input)
        self.assertEqual(corrected.metadata.boundary_correction_count, 1)
        self.assertEqual(
            corrected.coordinates[0][0],
            float(np.nextafter(1.0, 0.0)),
        )
        self.assertFalse(corrected.metadata.radial_rescaling_after_centering)

        excessive = np.asarray(
            [[1.0 + 2e-12, 0.0], [0.2, 0.0], [0.0, 0.2]],
            dtype=np.float64,
        )
        with (
            patch.object(
                hydra_module,
                "_standard_hydra_coordinates",
                return_value=(
                    uncentered,
                    1.0,
                    (-0.2, -0.1),
                    (),
                    1e-12,
                    2,
                ),
            ),
            patch.object(
                hydra_module,
                "hyperbolic_frechet_mean",
                return_value=(np.zeros(2), 1, 0.0),
            ),
            patch.object(
                hydra_module,
                "_mobius_add",
                return_value=excessive,
            ),
            patch.object(
                hydra_module,
                "_pairwise_poincare_distances",
                return_value=zeros,
            ),
        ):
            with self.assertRaisesRegex(
                HydraEmbeddingError,
                "beyond roundoff",
            ):
                embed_hydra(embedding_input)

    def test_rank_one_axes_centroids_rank_zero_and_collapse(self):
        embedding_input = _input(nx.path_graph(5), "rank-one")
        hydra = embed_hydra(embedding_input)
        mds = classical_mds(embedding_input)
        self.assertEqual(hydra.metadata.effective_spatial_rank, 1)
        self.assertEqual(mds.metadata.effective_rank, 1)
        for result in (hydra, mds):
            values = np.asarray(
                [result.coordinates[node] for node in embedding_input.node_order]
            )
            self.assertTrue(np.any(values[:, 0] != 0.0))
            np.testing.assert_array_equal(values[:, 1], np.zeros(len(values)))
        self.assertLessEqual(mds.metadata.centroid_residual, 1e-12)
        with self.assertRaisesRegex(ValueError, "rank zero"):
            independent_classical_mds(np.zeros((3, 3)), (0, 1, 2))
        with self.assertRaisesRegex(ValueError, "collapse"):
            native_condition(
                "excluded_collapsed",
                "mds",
                {0: (0.0, 0.0), 1: (0.0, 0.0)},
                (0, 1),
            )

    def test_k6_and_twin_fixtures_are_deterministic_and_match_oracles(self):
        twin = nx.Graph(
            [(0, 1), (0, 4), (4, 5), (1, 2), (1, 3)]
        )
        for label, graph in (
            ("k6", nx.complete_graph(6)),
            ("twin", twin),
        ):
            embedding_input = _input(graph, label)
            first = create_iteration2_embeddings(embedding_input)
            second = create_iteration2_embeddings(embedding_input)
            self.assertEqual(
                first.hydra_result.coordinates,
                second.hydra_result.coordinates,
            )
            self.assertEqual(
                first.mds_result.coordinates,
                second.mds_result.coordinates,
            )
            self.assertEqual(first.mds_result.metadata.effective_rank, 2)
            self.assertGreaterEqual(
                first.mds_result.metadata.retained_eigenvalues[0],
                first.mds_result.metadata.retained_eigenvalues[1],
            )
            self.assertTrue(
                all(
                    row["maximum_pairwise_distance_error"] < 1e-8
                    for row in first.independent_validation
                )
            )

    def test_partial_coincidence_is_preserved_and_all_radii_are_exact(self):
        coincident = native_condition(
            "excluded_coincidence",
            "mds",
            {0: (0.25, 0.0), 1: (0.25, 0.0), 2: (-0.25, 0.0)},
            (0, 1, 2),
        )
        self.assertEqual(coincident.coordinates[0], coincident.coordinates[1])
        self.assertEqual(coincident.coincidence_groups, ((0, 1),))

        embeddings = create_iteration2_embeddings(
            _input(nx.cycle_graph(6), "matched-radii")
        )
        self.assertEqual(len(embeddings.matched_conditions), 8)
        for condition in embeddings.matched_conditions:
            maximum = max(
                np.linalg.norm(point)
                for point in condition.coordinates.values()
            )
            self.assertAlmostEqual(
                maximum,
                float(condition.target_maximum_radius),
                places=14,
            )
            self.assertLess(maximum, 1.0)


if __name__ == "__main__":
    unittest.main()
