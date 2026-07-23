"""Validated Euclidean and Poincare-disk distance utilities."""

from __future__ import annotations

from math import asinh, hypot, isfinite, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_NUMERICAL_TOLERANCE = 1e-12


def _require_positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    validated = float(value)
    if not isfinite(validated) or validated <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return validated


def _as_finite_point(point: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(point, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite two-dimensional vector") from exc

    if array.shape != (2,):
        raise ValueError(f"{name} must be a finite two-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return np.array(array, dtype=float, copy=True)


def validate_disk_point(
    point: ArrayLike, *, name: str = "point"
) -> NDArray[np.float64]:
    """Return a validated copy of a finite 2D point in the open unit disk.

    Validation never projects or otherwise changes an invalid point. Projection is
    an embedding-layer operation.
    """

    array = _as_finite_point(point, name=name)
    norm = hypot(float(array[0]), float(array[1]))
    if not norm < 1.0:
        raise ValueError(f"{name} must lie strictly inside the open unit disk")
    return array


def euclidean_distance(u: ArrayLike, v: ArrayLike) -> float:
    """Return the Euclidean distance between two finite 2D points."""

    point_u = _as_finite_point(u, name="u")
    point_v = _as_finite_point(v, name="v")
    distance = hypot(
        float(point_u[0]) - float(point_v[0]),
        float(point_u[1]) - float(point_v[1]),
    )
    if not isfinite(distance):
        raise ValueError("Euclidean distance is not finite")
    return distance


def poincare_distance(
    u: ArrayLike,
    v: ArrayLike,
    *,
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> float:
    r"""Return the Poincare-disk distance between two points.

    The implemented metric is

    .. math::

       \operatorname{arcosh}\left(1 +
       \frac{2\lVert u-v\rVert^2}
       {(1-\lVert u\rVert^2)(1-\lVert v\rVert^2)}\right).

    Points on or outside the unit boundary are rejected rather than projected.
    The mathematically equivalent ``2*asinh`` form is evaluated because it
    preserves small positive distances that ``acosh(1 + x)`` can round to
    zero when ``x`` is tiny. ``tolerance`` validates the caller's numerical
    policy; it does not relax disk membership.
    """

    validated_tolerance = _require_positive_finite("tolerance", tolerance)
    point_u = validate_disk_point(u, name="u")
    point_v = validate_disk_point(v, name="v")

    difference_norm = hypot(
        float(point_u[0]) - float(point_v[0]),
        float(point_u[1]) - float(point_v[1]),
    )
    if difference_norm == 0.0:
        return 0.0

    # (1-r)*(1+r) is more accurate than 1-r**2 near the unit boundary.
    u_norm = hypot(float(point_u[0]), float(point_u[1]))
    v_norm = hypot(float(point_v[0]), float(point_v[1]))
    u_boundary_factor = (1.0 - u_norm) * (1.0 + u_norm)
    v_boundary_factor = (1.0 - v_norm) * (1.0 + v_norm)
    denominator_root = sqrt(u_boundary_factor) * sqrt(v_boundary_factor)
    if not isfinite(denominator_root) or denominator_root <= 0.0:
        raise ValueError("Poincare-distance denominator must be positive and finite")

    sinh_half_distance = difference_norm / denominator_root
    if not isfinite(sinh_half_distance) or sinh_half_distance < -validated_tolerance:
        raise ValueError("Poincare-distance ratio is not finite and non-negative")
    distance = 2.0 * asinh(max(0.0, sinh_half_distance))
    if not isfinite(distance) or distance < 0.0:
        raise ValueError("Poincare distance is not finite and non-negative")
    return distance
