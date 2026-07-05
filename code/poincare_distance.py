import numpy as np


def poincare_distance(u, v):
    """Return the hyperbolic distance between two points inside the Poincare disk."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if u.shape != (2,) or v.shape != (2,):
        raise ValueError("u and v must each be 2D points")

    u_norm_sq = float(np.dot(u, u))
    v_norm_sq = float(np.dot(v, v))

    if u_norm_sq >= 1.0 or v_norm_sq >= 1.0:
        raise ValueError("Points must lie strictly inside the unit disk")

    diff = u - v
    diff_norm_sq = float(np.dot(diff, diff))
    denominator = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
    cosh_value = 1.0 + 2.0 * diff_norm_sq / denominator
    cosh_value = max(cosh_value, 1.0)
    return float(np.arccosh(cosh_value))


if __name__ == "__main__":
    tests = [
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.2, 0.0), (0.5, 0.0)),
        ((-0.3, 0.4), (0.1, -0.2)),
    ]

    for u, v in tests:
        print(f"distance({u}, {v}) = {poincare_distance(u, v):.6f}")

    try:
        poincare_distance((1.2, 0.0), (0.0, 0.0))
    except ValueError as exc:
        print(f"invalid point rejected: {exc}")
    else:
        raise AssertionError("Expected ValueError for a point outside the disk")