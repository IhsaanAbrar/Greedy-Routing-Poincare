"""Read-only Iteration 1 pilot precision planning for Iteration 2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import gzip
from hashlib import blake2s
import json
from math import sqrt
from pathlib import Path

import numpy as np

from iteration2_config import ANALYSIS_PLAN_HASH, GRAPH_REPETITIONS


EXPLORATORY_FULL_CI_WIDTH_REFERENCE_PERCENTAGE_POINTS = 2.0
# Compatibility alias. This is a descriptive reference, not a power target.
PRESPECIFIED_FULL_CI_WIDTH_TARGET_PERCENTAGE_POINTS = (
    EXPLORATORY_FULL_CI_WIDTH_REFERENCE_PERCENTAGE_POINTS
)
PROJECTED_GRAPH_COUNTS = (20, 30, 40, 50)
NORMAL_975 = 1.959963984540054


def load_iteration1_graph_metrics(
    path: Path | str,
) -> list[dict[str, str]]:
    """Load the immutable Step 17 graph table without changing it."""

    resolved = Path(path).resolve(strict=True)
    with gzip.open(resolved, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1_800:
        raise ValueError("Iteration 1 graph metric table must have 1,800 rows")
    return rows


def _pilot_surrogate_interactions(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, int, int, str], list[float]]:
    by_graph: dict[str, dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        graph_id = row["graph_id"]
        condition = row["coordinate_condition_id"]
        if condition in by_graph[graph_id]:
            raise ValueError("duplicate Iteration 1 graph/condition pilot identity")
        by_graph[graph_id][condition] = row
    grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for graph_rows in by_graph.values():
        if "hydra" not in graph_rows:
            raise ValueError("Iteration 1 graph is missing Hydra")
        hydra = graph_rows["hydra"]
        for label in ("r050", "r070", "r085", "r095"):
            mds = graph_rows[f"mds_{label}"]
            interaction = float(hydra["poincare_advantage"]) - float(
                mds["poincare_advantage"]
            )
            grouped[
                (
                    hydra["model"],
                    int(hydra["n"]),
                    int(hydra["m"]),
                    label,
                )
            ].append(interaction)
    if len(grouped) != 72 or any(len(values) != 20 for values in grouped.values()):
        raise ValueError("Iteration 1 pilot interaction grid is incomplete")
    return grouped


def projected_precision(
    rows: Sequence[Mapping[str, str]],
    *,
    candidate_graphs: Sequence[int] = PROJECTED_GRAPH_COUNTS,
    target_full_width_percentage_points: float = (
        EXPLORATORY_FULL_CI_WIDTH_REFERENCE_PERCENTAGE_POINTS
    ),
) -> dict[str, object]:
    """Describe an inexact Iteration 1 surrogate without claiming I2 power."""

    interactions = _pilot_surrogate_interactions(rows)
    projections: list[dict[str, object]] = []
    for model in ("erdos_renyi", "barabasi_albert"):
        for label in ("r050", "r070", "r085", "r095"):
            variances = [
                float(
                    np.var(
                        interactions[(model, n, m, label)],
                        ddof=1,
                    )
                )
                for n in (100, 300, 1000)
                for m in (4, 8, 16)
            ]
            maximum_sd = sqrt(max(variances))
            for graph_count in candidate_graphs:
                if graph_count <= 1:
                    raise ValueError("candidate graph counts must exceed one")
                cell_width = (
                    2.0
                    * NORMAL_975
                    * maximum_sd
                    / sqrt(graph_count)
                    * 100.0
                )
                marginal_variance = (
                    sum(variances) / graph_count / (len(variances) ** 2)
                )
                marginal_width = (
                    2.0 * NORMAL_975 * sqrt(marginal_variance) * 100.0
                )
                projections.append(
                    {
                        "model": model,
                        "radius_label": label,
                        "graphs_per_n_m_cell": graph_count,
                        "projected_worst_cell_full_95_ci_width_pp": cell_width,
                        "projected_model_marginal_full_95_ci_width_pp": (
                            marginal_width
                        ),
                        "method": (
                            "iteration1_inexact_surrogate_graph_variance_normal_projection"
                        ),
                        "target_estimand_matched": False,
                        "use_for_confirmatory_power": False,
                    }
                )
    return {
        "iteration1_used_as_pilot_only": True,
        "iteration2_target_estimand": (
            "[(S_P-S_E)_Hydra_scaled_r]-[(S_P-S_E)_MDS_scaled_r]"
        ),
        "iteration1_surrogate_estimand": (
            "[(S_P-S_E)_Hydra_native]-[(S_P-S_E)_MDS_scaled_r]"
        ),
        "estimand_alignment": "inexact_exploratory_surrogate_only",
        "reason_exact_planning_is_unavailable": (
            "Iteration_1_did_not_route_scaled_Hydra_at_the_four_matched_radii"
        ),
        "exploratory_full_95_ci_width_reference_percentage_points": (
            target_full_width_percentage_points
        ),
        "projections": projections,
        "frozen_iteration2_graphs_per_cell": GRAPH_REPETITIONS,
        "recommended_graphs_per_cell": None,
        "design_changed_from_frozen_protocol": False,
        "sample_size_selected_for_significance": False,
        "confirmatory_equivalence_power_claim": False,
        "simultaneous_four_radius_power_claim": False,
    }


def nested_pair_resampling_sensitivity(
    graph_pair_contrasts: Mapping[str, Sequence[float]],
    *,
    pair_counts: Sequence[int] = (250, 500, 1_000),
    replicates: int = 2_000,
    seed: int | None = None,
) -> list[dict[str, object]]:
    """Quantify pair-level Monte Carlo noise while preserving graph units."""

    if len(graph_pair_contrasts) < 2 or replicates < 2:
        raise ValueError("nested sensitivity requires multiple graphs and draws")
    arrays = {
        graph_id: np.asarray(values, dtype=np.float64)
        for graph_id, values in graph_pair_contrasts.items()
    }
    if any(
        values.ndim != 1
        or len(values) < max(pair_counts)
        or not np.isfinite(values).all()
        for values in arrays.values()
    ):
        raise ValueError("pair contrasts are incomplete or non-finite")
    if seed is None:
        payload = (
            "greedy-routing-iteration2-nested-pair-sensitivity-v1\0"
            + ANALYSIS_PLAN_HASH
        ).encode("utf-8")
        seed = int.from_bytes(
            blake2s(payload, digest_size=16, person=b"GRP2pair").digest(),
            "big",
        )
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    graph_ids = sorted(arrays)
    for pair_count in pair_counts:
        estimates = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            graph_means = []
            for graph_id in graph_ids:
                values = arrays[graph_id]
                draw = rng.integers(0, len(values), size=pair_count)
                graph_means.append(float(values[draw].mean()))
            estimates[replicate] = float(np.mean(graph_means))
        output.append(
            {
                "pairs_per_graph": pair_count,
                "nested_resampling_standard_error_pp": (
                    float(np.std(estimates, ddof=1)) * 100.0
                ),
                "replicates": replicates,
                "graph_count": len(graph_ids),
                "graph_is_independent_unit": True,
                "seed_source": "analysis_plan_identity_domain_v1",
            }
        )
    return output


def load_iteration1_pair_contrasts(
    raw_graph_root: Path | str,
    *,
    replicate_indices: Sequence[int] = (0, 1),
    coordinate_condition_id: str = "hydra",
) -> dict[str, tuple[float, ...]]:
    """Read a prespecified pilot subset; selection never uses route outcomes."""

    root = Path(raw_graph_root).resolve(strict=True)
    selected: dict[str, tuple[float, ...]] = {}
    for model_prefix in ("er", "ba"):
        for n in (100, 300, 1000):
            for m in (4, 8, 16):
                for replicate in replicate_indices:
                    graph_id = (
                        f"{model_prefix}_n{n:04d}_m{m:02d}_rep{replicate:03d}"
                    )
                    path = root / graph_id / "routes.jsonl.gz"
                    by_pair: dict[int, dict[str, object]] = defaultdict(dict)
                    with gzip.open(
                        path, "rt", encoding="utf-8", newline=""
                    ) as stream:
                        for line in stream:
                            row = json.loads(line)
                            if (
                                row["coordinate_condition_id"]
                                == coordinate_condition_id
                                and row["method_id"]
                                in ("euclidean_greedy", "poincare_greedy")
                            ):
                                pair_index = int(row["pair_index"])
                                method = str(row["method_id"])
                                methods = by_pair[pair_index]
                                if method in methods:
                                    raise ValueError("duplicate pilot pair/method identity")
                                pair_identity = (int(row["source"]), int(row["destination"]))
                                previous = methods.setdefault("_pair_identity", pair_identity)
                                if previous != pair_identity:
                                    raise ValueError("inconsistent pilot source/destination identity")
                                methods[method] = bool(row["success"])
                    if len(by_pair) != 1_000 or any(
                        set(methods)
                        != {"_pair_identity", "euclidean_greedy", "poincare_greedy"}
                        for methods in by_pair.values()
                    ):
                        raise ValueError("pilot pair contrast input is incomplete")
                    selected[graph_id] = tuple(
                        float(methods["poincare_greedy"])
                        - float(methods["euclidean_greedy"])
                        for _, methods in sorted(by_pair.items())
                    )
    expected = 18 * len(tuple(replicate_indices))
    if len(selected) != expected:
        raise ValueError("pilot pair-sensitivity graph subset is incomplete")
    return selected
