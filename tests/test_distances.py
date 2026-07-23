from math import atanh, isfinite
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from poincare_distance import (  # noqa: E402
    euclidean_distance,
    poincare_distance,
    validate_disk_point,
)


class DistanceTests(unittest.TestCase):
    def test_identity_is_zero_for_both_metrics(self):
        for point in ((0.0, 0.0), (0.25, -0.5), (0.99, 0.0)):
            with self.subTest(point=point):
                self.assertEqual(euclidean_distance(point, point), 0.0)
                self.assertEqual(poincare_distance(point, point), 0.0)

    def test_both_metrics_are_symmetric_and_non_negative(self):
        u = (-0.3, 0.4)
        v = (0.2, -0.1)

        for distance in (euclidean_distance, poincare_distance):
            with self.subTest(metric=distance.__name__):
                forward = distance(u, v)
                backward = distance(v, u)
                self.assertGreaterEqual(forward, 0.0)
                self.assertAlmostEqual(forward, backward, places=14)

    def test_known_euclidean_distance(self):
        self.assertEqual(euclidean_distance((0.0, 0.0), (3.0, 4.0)), 5.0)

    def test_poincare_distance_from_origin_matches_closed_form(self):
        for radius in (0.1, 0.5, 0.9):
            with self.subTest(radius=radius):
                self.assertAlmostEqual(
                    poincare_distance((0.0, 0.0), (radius, 0.0)),
                    2.0 * atanh(radius),
                    places=12,
                )

    def test_poincare_distance_is_finite_near_boundary(self):
        radius = np.nextafter(1.0, 0.0)
        distance = poincare_distance((0.0, 0.0), (radius, 0.0))

        self.assertTrue(isfinite(distance))
        self.assertGreater(distance, 0.0)

    def test_near_coincident_distinct_points_have_positive_distance(self):
        separation = 1e-200

        distance = poincare_distance((0.0, 0.0), (separation, 0.0))

        self.assertGreater(distance, 0.0)
        self.assertAlmostEqual(distance / separation, 2.0, places=14)

    def test_triangle_inequality_for_representative_disk_points(self):
        u = (-0.7, 0.1)
        v = (0.05, -0.2)
        w = (0.8, 0.15)

        self.assertLessEqual(
            poincare_distance(u, w),
            poincare_distance(u, v) + poincare_distance(v, w) + 1e-14,
        )

    def test_invalid_point_shapes_are_rejected(self):
        invalid_points = (0.1, (0.1,), (0.1, 0.2, 0.3), ((0.1, 0.2),))

        for point in invalid_points:
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    validate_disk_point(point)
                with self.assertRaises(ValueError):
                    euclidean_distance(point, (0.0, 0.0))
                with self.assertRaises(ValueError):
                    poincare_distance(point, (0.0, 0.0))

    def test_nan_and_infinity_are_rejected(self):
        for invalid_value in (float("nan"), float("inf"), float("-inf")):
            point = (invalid_value, 0.0)
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    validate_disk_point(point)
                with self.assertRaises(ValueError):
                    euclidean_distance(point, (0.0, 0.0))
                with self.assertRaises(ValueError):
                    poincare_distance(point, (0.0, 0.0))

    def test_points_on_or_outside_disk_are_rejected(self):
        for point in ((1.0, 0.0), (-1.0, 0.0), (0.8, 0.6), (1.1, 0.0)):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    validate_disk_point(point)
                with self.assertRaises(ValueError):
                    poincare_distance(point, (0.0, 0.0))

    def test_euclidean_distance_does_not_impose_disk_membership(self):
        self.assertEqual(euclidean_distance((1.0, 1.0), (4.0, 5.0)), 5.0)

    def test_invalid_numerical_tolerances_are_rejected(self):
        for tolerance in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                poincare_distance((0.0, 0.0), (0.1, 0.0), tolerance=tolerance)

    def test_disk_validation_returns_an_independent_array(self):
        original = np.array((0.2, 0.3))
        validated = validate_disk_point(original)

        validated[0] = 0.7
        self.assertEqual(original[0], 0.2)


if __name__ == "__main__":
    unittest.main()
