"""Independent structural validation and Iteration 1 protection for Iteration 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import blake2s, sha256
import json
from math import atanh, hypot, isclose, isfinite
from pathlib import Path
import re
from time import perf_counter

from analyze_full_experiment import validate_derived_directory
from iteration2_analysis import (
    graph_level_interactions,
    validate_graph_level_rows,
)
from iteration2_config import (
    ANALYSIS_PLAN_HASH,
    COMBINED_PROTOCOL_HASH,
    DATA_GENERATION_HASH,
    FAILURE_SCHEMA_VERSION,
    ITERATION1_ANALYSIS_DIRECTORY,
    ITERATION1_ANALYSIS_MANIFEST_SHA256,
    ITERATION1_RAW_BYTE_COUNT,
    ITERATION1_RAW_DIRECTORY,
    ITERATION1_RAW_FILE_COUNT,
    ITERATION1_RAW_MANIFEST_SHA256,
    ITERATION1_RAW_TREE_SHA256,
    ITERATION2_RESULT_SCHEMA,
    ITERATION2_RUN_IDENTITY,
    MATCHED_RADII,
    MATCHED_RADIUS_LABELS,
    OUTPUT_SCHEMA_HASH,
    PAIRS_PER_GRAPH,
    ROUTING_METHODS,
    GraphSpec,
    full_schedule,
    is_full_oracle_graph,
    seeds_for_graph,
    sentinel_pair_indices,
)
from iteration2_experiment import (
    PAIR_IDENTITY_SCHEMA,
    ROUTES_PER_PAIR,
    ROUTE_IDENTITY_SCHEMA,
    execute_scheduled_graph,
)
from iteration2_routing import ORDINARY_FAILURE_TYPES, REPAIR_FAILURE_TYPES
from validate_full_experiment import compute_raw_tree_fingerprint


class Iteration2ValidationError(RuntimeError):
    """Raised when a result violates the frozen Iteration 2 schema."""


def _identity_digest(domain: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        {"domain": domain, **dict(payload)},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return blake2s(encoded, digest_size=16, person=b"i2idv1").hexdigest()


def _expected_pair_identity(
    graph_id: str,
    pair_index: int,
    source: int,
    destination: int,
) -> str:
    return _identity_digest(
        "pair_sampling",
        {
            "schema": PAIR_IDENTITY_SCHEMA,
            "data_generation_hash": DATA_GENERATION_HASH,
            "graph_id": graph_id,
            "pair_index": pair_index,
            "source": source,
            "destination": destination,
        },
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_iteration1_immutable(
    repository_root: Path | str,
    *,
    deep: bool = True,
) -> dict[str, object]:
    """Read-only proof that Step 16 and Step 17 retain their frozen bytes."""

    root = Path(repository_root).resolve(strict=True)
    raw = root / "results" / ITERATION1_RAW_DIRECTORY
    analysis = root / "results" / ITERATION1_ANALYSIS_DIRECTORY
    raw_manifest_hash = _sha256(raw / "run_manifest.json")
    analysis_manifest_hash = _sha256(analysis / "analysis_manifest.json")
    if raw_manifest_hash != ITERATION1_RAW_MANIFEST_SHA256:
        raise Iteration2ValidationError("Iteration 1 raw manifest changed")
    if analysis_manifest_hash != ITERATION1_ANALYSIS_MANIFEST_SHA256:
        raise Iteration2ValidationError("Iteration 1 analysis manifest changed")
    manifest = validate_derived_directory(analysis)
    evidence: dict[str, object] = {
        "raw_directory": raw.name,
        "analysis_directory": analysis.name,
        "raw_manifest_sha256": raw_manifest_hash,
        "analysis_manifest_sha256": analysis_manifest_hash,
        "step17_files": manifest["files"],
        "deep_raw_tree_checked": deep,
    }
    if deep:
        tree = compute_raw_tree_fingerprint(raw, include_entries=False)
        if (
            tree.sha256 != ITERATION1_RAW_TREE_SHA256
            or tree.file_count != ITERATION1_RAW_FILE_COUNT
            or tree.byte_count != ITERATION1_RAW_BYTE_COUNT
        ):
            raise Iteration2ValidationError("Iteration 1 raw tree changed")
        evidence["raw_tree_fingerprint"] = tree.summary()
    return evidence


def _validate_coordinates(
    result: Mapping[str, object],
    n: int,
) -> None:
    coordinates = result.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise Iteration2ValidationError("coordinates must be a mapping")
    expected = {
        "hydra_native",
        "mds_native",
        *(
            f"{family}_scaled_{label}"
            for family in ("hydra", "mds")
            for label in MATCHED_RADIUS_LABELS
        ),
    }
    if set(coordinates) != expected:
        raise Iteration2ValidationError("coordinate condition set mismatch")
    expected_nodes = set(range(n))
    for condition, points in coordinates.items():
        if not isinstance(points, Mapping):
            raise Iteration2ValidationError("coordinate node coverage mismatch")
        try:
            normalized_points = {
                int(node): point for node, point in points.items()
            }
        except (TypeError, ValueError) as exc:
            raise Iteration2ValidationError(
                "coordinate node ID is invalid"
            ) from exc
        if set(normalized_points) != expected_nodes:
            raise Iteration2ValidationError("coordinate node coverage mismatch")
        radii = []
        unique = set()
        for point in normalized_points.values():
            if not isinstance(point, Sequence) or len(point) != 2:
                raise Iteration2ValidationError("coordinate is not a two-vector")
            values = (float(point[0]), float(point[1]))
            if not all(isfinite(value) for value in values):
                raise Iteration2ValidationError("coordinate is non-finite")
            radii.append(hypot(*values))
            unique.add(values)
        if len(unique) <= 1:
            raise Iteration2ValidationError("complete coordinate collapse")
        if condition != "mds_native" and max(radii) >= 1.0:
            raise Iteration2ValidationError("routable coordinate left disk")
        for radius, label in zip(
            MATCHED_RADII,
            MATCHED_RADIUS_LABELS,
            strict=True,
        ):
            if condition.endswith(label) and not isclose(
                max(radii),
                radius,
                rel_tol=0.0,
                abs_tol=1e-13,
            ):
                raise Iteration2ValidationError(
                    "matched coordinate radius mismatch"
                )


def validate_iteration2_graph_result(
    result: Mapping[str, object],
) -> dict[str, int]:
    """Validate identities, pair crossing, routes, and graph-level estimands."""

    if not isinstance(result, Mapping):
        raise Iteration2ValidationError("result must be a mapping")
    run_identity = result.get("run_identity")
    scientific_status = result.get("scientific_status")
    production_identity = (
        run_identity == ITERATION2_RUN_IDENTITY
        and scientific_status == "iteration2_scientific_graph"
    )
    excluded_identity = (
        isinstance(run_identity, str)
        and re.fullmatch(r"iteration2_excluded_raw_[0-9a-f]{16}", run_identity)
        is not None
        and scientific_status == "excluded_non_scientific"
    )
    if (
        result.get("result_schema") != ITERATION2_RESULT_SCHEMA
        or result.get("data_generation_hash") != DATA_GENERATION_HASH
        or result.get("analysis_plan_hash") != ANALYSIS_PLAN_HASH
        or result.get("output_schema_hash") != OUTPUT_SCHEMA_HASH
        or result.get("protocol_hash") != COMBINED_PROTOCOL_HASH
        or not (production_identity or excluded_identity)
    ):
        raise Iteration2ValidationError("result identity mismatch")
    if result.get("checkpoint_completion") != {
        "status": "complete",
        "atomic_payload": True,
    }:
        raise Iteration2ValidationError("checkpoint completion field is invalid")
    identity = result.get("graph_identity")
    if not isinstance(identity, Mapping):
        raise Iteration2ValidationError("graph identity is missing")
    required_identity = {
        "graph_id",
        "model",
        "n",
        "m",
        "replicate_index",
        "node_order",
        "canonical_edges",
    }
    if not required_identity.issubset(identity):
        raise Iteration2ValidationError("graph identity is incomplete")
    try:
        n = int(identity["n"])
        graph_id = str(identity["graph_id"])
        m = int(identity["m"])
        replicate_index = int(identity["replicate_index"])
    except (TypeError, ValueError) as exc:
        raise Iteration2ValidationError("graph identity values are invalid") from exc
    if (
        not graph_id
        or identity.get("model") not in {"erdos_renyi", "barabasi_albert"}
        or n < 2
        or not 0 < m < n
        or replicate_index < 0
    ):
        raise Iteration2ValidationError("graph identity values are invalid")
    if identity.get("node_order") != list(range(n)):
        raise Iteration2ValidationError("node order is not canonical")
    _validate_coordinates(result, n)
    tolerances = result.get("tolerances")
    coordinates = result.get("coordinates")
    if not isinstance(tolerances, Mapping) or not isinstance(coordinates, Mapping):
        raise Iteration2ValidationError("routing tolerance table is missing")
    if set(tolerances) != set(coordinates):
        raise Iteration2ValidationError("routing tolerance condition set mismatch")
    float64_epsilon = 2.0**-52
    for condition_id, points in coordinates.items():
        if not isinstance(points, Mapping):
            raise Iteration2ValidationError("coordinate table is invalid")
        radius = max(
            hypot(float(point[0]), float(point[1]))
            for point in points.values()
        )
        expected_euclidean = 64.0 * float64_epsilon * radius
        expected_poincare = (
            None
            if condition_id == "mds_native"
            else 64.0
            * float64_epsilon
            * max(1.0, 4.0 * atanh(radius))
        )
        row = tolerances.get(condition_id)
        observed_euclidean = (
            row.get("euclidean") if isinstance(row, Mapping) else None
        )
        observed_poincare = (
            row.get("poincare") if isinstance(row, Mapping) else None
        )
        if (
            not isinstance(row, Mapping)
            or isinstance(observed_euclidean, bool)
            or not isinstance(observed_euclidean, (int, float))
            or not isclose(
                float(observed_euclidean),
                expected_euclidean,
                rel_tol=4e-15,
                abs_tol=0.0,
            )
            or (
                expected_poincare is None
                and observed_poincare is not None
            )
            or (
                expected_poincare is not None
                and (
                    isinstance(observed_poincare, bool)
                    or not isinstance(observed_poincare, (int, float))
                    or not isclose(
                        float(observed_poincare),
                        expected_poincare,
                        rel_tol=4e-15,
                        abs_tol=0.0,
                    )
                )
            )
        ):
            raise Iteration2ValidationError(
                "routing tolerance differs from the frozen scale formula: "
                f"{condition_id}"
            )
    provenance = result.get("generation_provenance")
    if not isinstance(provenance, Mapping):
        raise Iteration2ValidationError("generation provenance is missing")
    model = str(identity["model"])
    edge_count = len(identity.get("canonical_edges", ()))
    expected_common_provenance = {
        "schema": "iteration2_graph_generation_provenance_v1",
        "model": model,
        "attempt_count": len(provenance.get("attempted_seeds", ())),
        "accepted_seed": (
            provenance.get("attempted_seeds", ())[-1]
            if provenance.get("attempted_seeds")
            else None
        ),
        "realised_edge_count": edge_count,
        "realised_average_degree": 2.0 * edge_count / n,
        "simple": True,
        "connected": True,
        "undirected": True,
        "unweighted": True,
        "node_ids": "integers_0_through_n_minus_1",
        "acceptance_selected_using_routing_outcomes": False,
    }
    if any(
        provenance.get(key) != value
        for key, value in expected_common_provenance.items()
    ):
        raise Iteration2ValidationError("generation provenance is inconsistent")
    attempts = provenance.get("attempts")
    attempted_seeds = provenance.get("attempted_seeds")
    if (
        not isinstance(attempts, Sequence)
        or isinstance(attempts, (str, bytes))
        or not isinstance(attempted_seeds, Sequence)
        or isinstance(attempted_seeds, (str, bytes))
        or len(attempts) != len(attempted_seeds)
        or any(
            not isinstance(row, Mapping)
            or row.get("attempt_index") != index
            or row.get("seed") != attempted_seeds[index]
            or row.get("outcome")
            != (
                "accepted_connected"
                if index == len(attempts) - 1
                else "rejected_disconnected"
            )
            for index, row in enumerate(attempts)
        )
    ):
        raise Iteration2ValidationError("generation attempt provenance is invalid")
    if model == "erdos_renyi":
        numerator = 2 * m * (n - m)
        denominator = n * (n - 1)
        if (
            provenance.get("rejection_count") != len(attempts) - 1
            or provenance.get("p_exact_numerator") != numerator
            or provenance.get("p_exact_denominator") != denominator
            or provenance.get("p") != numerator / denominator
            or provenance.get("maximum_attempts") != 50
            or provenance.get("largest_component_substitution") is not False
        ):
            raise Iteration2ValidationError("ER generation provenance is invalid")
    else:
        if (
            len(attempts) != 1
            or provenance.get("rejection_count") != 0
            or provenance.get("m") != m
            or provenance.get("expected_edge_count") != m * (n - m)
            or provenance.get("exact_finite_size_expected_average_degree")
            != 2.0 * m * (n - m) / n
            or not provenance.get("initial_graph_convention")
        ):
            raise Iteration2ValidationError("BA generation provenance is invalid")

    pairs = result.get("pairs")
    if not isinstance(pairs, Sequence) or not pairs:
        raise Iteration2ValidationError("pair list is empty")
    pair_count = len(pairs)
    normalized_pairs = tuple(tuple(int(value) for value in pair) for pair in pairs)
    if (
        any(
            len(pair) != 2
            or pair[0] == pair[1]
            or not all(0 <= node < n for node in pair)
            for pair in normalized_pairs
        )
        or len(set(normalized_pairs)) != pair_count
    ):
        raise Iteration2ValidationError("pair list is invalid")
    pair_records = result.get("pair_records")
    if (
        not isinstance(pair_records, Sequence)
        or isinstance(pair_records, (str, bytes))
        or len(pair_records) != pair_count
    ):
        raise Iteration2ValidationError("pair sampling records are incomplete")
    expected_pair_identities: list[str] = []
    for pair_index, (source, destination) in enumerate(normalized_pairs):
        row = pair_records[pair_index]
        expected_sampling_identity = _expected_pair_identity(
            graph_id,
            pair_index,
            source,
            destination,
        )
        expected_pair_identities.append(expected_sampling_identity)
        if (
            not isinstance(row, Mapping)
            or row.get("pair_index") != pair_index
            or row.get("source") != source
            or row.get("destination") != destination
            or row.get("sampling_identity_schema") != PAIR_IDENTITY_SCHEMA
            or row.get("sampling_identity") != expected_sampling_identity
        ):
            raise Iteration2ValidationError("pair sampling identity mismatch")
    if len(set(expected_pair_identities)) != pair_count:
        raise Iteration2ValidationError("pair sampling identities collide")
    pair_sampling_identity = result.get("pair_sampling_identity")
    if (
        not isinstance(pair_sampling_identity, Mapping)
        or pair_sampling_identity.get("schema") != PAIR_IDENTITY_SCHEMA
        or pair_sampling_identity.get("graph_id") != graph_id
        or pair_sampling_identity.get("pair_count") != pair_count
        or pair_sampling_identity.get("data_generation_hash")
        != DATA_GENERATION_HASH
        or pair_sampling_identity.get(
            "shared_across_all_coordinate_and_routing_conditions"
        )
        is not True
    ):
        raise Iteration2ValidationError("pair sampling design identity is invalid")
    if result.get("dijkstra_execution_count") != pair_count:
        raise Iteration2ValidationError("Dijkstra was not executed once per pair")
    dijkstra = result.get("dijkstra_records")
    if not isinstance(dijkstra, Sequence) or len(dijkstra) != pair_count:
        raise Iteration2ValidationError("Dijkstra record count mismatch")
    lengths = []
    canonical_edges = {
        frozenset((int(edge[0]), int(edge[1])))
        for edge in identity.get("canonical_edges", ())
    }
    if not canonical_edges:
        raise Iteration2ValidationError("canonical edge list is empty")
    for index, row in enumerate(dijkstra):
        source, destination = normalized_pairs[index]
        if (
            row.get("graph_id") != graph_id
            or row.get("pair_index") != index
            or row.get("pair_identity") != expected_pair_identities[index]
            or row.get("sampling_identity_schema") != PAIR_IDENTITY_SCHEMA
            or row.get("source") != source
            or row.get("destination") != destination
            or row.get("route_length") != row.get("apsp_length")
            or row.get("route_length") != len(row.get("walk", ())) - 1
        ):
            raise Iteration2ValidationError("Dijkstra record mismatch")
        lengths.append(int(row["route_length"]))
        if any(
            frozenset((left, right)) not in canonical_edges
            for left, right in zip(row.get("walk", ()), row.get("walk", ())[1:])
        ):
            raise Iteration2ValidationError("Dijkstra walk contains a non-edge")

    expected_methods = {
        "hydra_native": ROUTING_METHODS,
        "mds_native": ("euclidean_greedy",),
    }
    for family in ("hydra", "mds"):
        for label in MATCHED_RADIUS_LABELS:
            expected_methods[f"{family}_scaled_{label}"] = ROUTING_METHODS
    expected_keys = {
        (pair_index, condition, method)
        for pair_index in range(pair_count)
        for condition, methods in expected_methods.items()
        for method in methods
    }
    routes = result.get("route_records")
    if (
        not isinstance(routes, Sequence)
        or len(routes) != pair_count * ROUTES_PER_PAIR
        or result.get("routes_per_pair") != ROUTES_PER_PAIR
    ):
        raise Iteration2ValidationError("route record count mismatch")
    observed_keys: set[tuple[int, str, str]] = set()
    priority_contexts: dict[int, str] = {}
    euclidean_signatures: dict[tuple[int, str], list[tuple[object, ...]]] = {}
    reused_euclidean_records = 0
    for row in routes:
        if not isinstance(row, Mapping):
            raise Iteration2ValidationError("route record must be a mapping")
        key = (
            int(row.get("pair_index", -1)),
            str(row.get("coordinate_condition_id", "")),
            str(row.get("method_id", "")),
        )
        if key not in expected_keys or key in observed_keys:
            raise Iteration2ValidationError("route key missing or duplicated")
        observed_keys.add(key)
        pair_index, condition_id, method_id = key
        source, destination = normalized_pairs[pair_index]
        walk = tuple(int(node) for node in row.get("walk", ()))
        success = row.get("success")
        route_length = row.get("route_length")
        if (
            not isinstance(success, bool)
            or row.get("graph_id") != graph_id
            or row.get("pair_identity") != expected_pair_identities[pair_index]
            or row.get("source") != source
            or row.get("destination") != destination
            or route_length != len(walk) - 1
            or row.get("dijkstra_length") != lengths[pair_index]
            or not walk
            or walk[0] != source
        ):
            raise Iteration2ValidationError("route structural mismatch")
        expected_route_identity = {
            "schema": ROUTE_IDENTITY_SCHEMA,
            "graph_id": graph_id,
            "pair_index": pair_index,
            "pair_identity": expected_pair_identities[pair_index],
            "source": source,
            "destination": destination,
            "coordinate_condition_id": condition_id,
            "method_id": method_id,
        }
        if (
            result.get("route_identity_schema") != ROUTE_IDENTITY_SCHEMA
            or row.get("route_identity") != expected_route_identity
            or row.get("route_identity_hash")
            != _identity_digest("route", expected_route_identity)
            or row.get("metric_id")
            != (
                "euclidean"
                if method_id == "euclidean_greedy"
                else "poincare"
            )
        ):
            raise Iteration2ValidationError("explicit route identity mismatch")
        priority_context_id = row.get("priority_context_id")
        if not isinstance(priority_context_id, str) or not priority_context_id:
            raise Iteration2ValidationError("routing priority context is missing")
        prior_context = priority_contexts.setdefault(
            pair_index,
            priority_context_id,
        )
        if prior_context != priority_context_id:
            raise Iteration2ValidationError(
                "routing priority changed across metrics or conditions"
            )
        distance_tolerance = row.get("distance_tolerance")
        expected_tolerance = tolerances[condition_id][
            "euclidean" if method_id == "euclidean_greedy" else "poincare"
        ]
        if (
            isinstance(distance_tolerance, bool)
            or not isinstance(distance_tolerance, (int, float))
            or not isfinite(float(distance_tolerance))
            or float(distance_tolerance) <= 0.0
            or distance_tolerance != expected_tolerance
        ):
            raise Iteration2ValidationError("route tolerance is invalid")
        resources = {
            "physical_edge_traversals": row.get("physical_edge_traversals"),
            "forwarding_decisions": row.get("forwarding_decisions"),
            "logical_distance_evaluations": row.get(
                "logical_distance_evaluations"
            ),
            "peak_history_vertices": row.get("peak_history_vertices"),
        }
        if (
            resources["physical_edge_traversals"] != route_length
            or row.get("physical_hops") != route_length
            or resources["forwarding_decisions"] != route_length
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in resources.values()
            )
            or resources["peak_history_vertices"] < 1
            or resources["peak_history_vertices"] > n
        ):
            raise Iteration2ValidationError("route resource accounting is invalid")
        accounting = row.get("resource_accounting")
        if not isinstance(accounting, Mapping) or set(accounting) != set(resources):
            raise Iteration2ValidationError("route resource metadata is incomplete")
        for resource_name, value in resources.items():
            item = accounting.get(resource_name)
            if (
                not isinstance(item, Mapping)
                or item.get("value") != value
                or item.get("applicability") != "applicable"
                or item.get("na_reason") is not None
            ):
                raise Iteration2ValidationError(
                    "route resource applicability is invalid"
                )
        if row.get("initial_failure_type") in {
            "attempted_revisit",
            "ordinary_revisit",
            "numerical_invariant_failure",
        } or row.get("final_failure_type") in {
            "attempted_revisit",
            "ordinary_revisit",
            "numerical_invariant_failure",
        }:
            raise Iteration2ValidationError(
                "invariant failure entered scientific route records"
            )
        is_reused = row.get("execution_reused")
        expected_reused_from = None
        if method_id == "euclidean_greedy" and condition_id.startswith(
            "hydra_scaled_"
        ):
            expected_reused_from = "hydra_native"
        elif method_id == "euclidean_greedy" and condition_id.startswith(
            "mds_scaled_"
        ):
            expected_reused_from = "mds_native"
        if (
            is_reused is not (expected_reused_from is not None)
            or row.get("reused_from_condition_id") != expected_reused_from
            or row.get("reuse_basis")
            != (
                "verified_uniform_euclidean_scale_invariance"
                if expected_reused_from is not None
                else None
            )
        ):
            raise Iteration2ValidationError("Euclidean reuse metadata is invalid")
        if expected_reused_from is not None:
            reused_euclidean_records += 1
        if method_id == "euclidean_greedy":
            family = "hydra" if condition_id.startswith("hydra_") else "mds"
            euclidean_signatures.setdefault((pair_index, family), []).append(
                (success, walk, row.get("final_failure_type"))
            )
        if success:
            if walk[-1] != destination:
                raise Iteration2ValidationError("successful route missed target")
            expected_stretch = route_length / lengths[pair_index]
            if row.get("stretch") != expected_stretch:
                raise Iteration2ValidationError("stretch denominator mismatch")
        elif row.get("stretch") is not None:
            raise Iteration2ValidationError("failed route has stretch")
        if any(
            frozenset((left, right)) not in canonical_edges
            for left, right in zip(walk, walk[1:])
        ):
            raise Iteration2ValidationError("route walk contains a non-edge")
        if row.get("failure_schema") != FAILURE_SCHEMA_VERSION:
            raise Iteration2ValidationError("failure schema is invalid")
        applicability = row.get("failure_category_applicability")
        stages = row.get("failure_stage_applicability")
        if not isinstance(applicability, Mapping) or not isinstance(
            stages, Mapping
        ):
            raise Iteration2ValidationError("failure applicability is missing")
        if set(applicability) != {"initial", "final"} or set(stages) != {
            "initial",
            "repair",
            "final",
        }:
            raise Iteration2ValidationError("failure stages are incomplete")
        is_repair = key[2] == "repaired_poincare_greedy"
        initial_observed = row.get("initial_failure_type") is not None
        final_observed = row.get("final_failure_type") is not None
        repair_attempted = row.get("repair_attempted")
        repair_count = row.get("repair_attempt_count")
        repair_backtrackable = row.get("repair_backtrackable")
        repair_eligible = row.get("repair_eligible")
        repair_succeeded = row.get("repair_succeeded")
        alternative_existed = row.get("repair_alternative_existed")
        alternative_selected = row.get("repair_alternative_selected")
        selected_alternative = row.get("repair_selected_alternative")
        denominator_membership = row.get("repair_denominator_membership")
        expected_denominator_membership = {
            "ordinary_poincare_failed": is_repair and initial_observed,
            "failure_was_backtrackable": (
                is_repair and repair_backtrackable is True
            ),
            "repair_eligible": is_repair and bool(repair_eligible),
            "alternative_existed": (
                is_repair and alternative_existed is True
            ),
            "repair_attempted": is_repair and bool(repair_attempted),
            "route_recovered": is_repair and bool(repair_succeeded),
            "applicability": "applicable" if is_repair else "not_applicable",
            "na_reason": (
                None if is_repair else "routing_method_has_no_repair_stage"
            ),
        }
        if denominator_membership != expected_denominator_membership:
            raise Iteration2ValidationError(
                "repair denominator membership is inconsistent"
            )
        if (
            not isinstance(repair_attempted, bool)
            or not isinstance(repair_succeeded, bool)
            or not isinstance(repair_eligible, bool)
            or not isinstance(alternative_selected, bool)
            or repair_count not in {0, 1}
            or repair_attempted != (repair_count == 1)
            or repair_succeeded != (repair_attempted and success)
            or alternative_selected != (selected_alternative is not None)
        ):
            raise Iteration2ValidationError("repair resource flags are inconsistent")
        if not is_repair:
            if (
                repair_attempted
                or repair_succeeded
                or repair_backtrackable is not None
                or repair_eligible
                or alternative_existed is not None
                or alternative_selected
            ):
                raise Iteration2ValidationError(
                    "ordinary route has repair-only resource state"
                )
        elif not initial_observed:
            if (
                repair_attempted
                or repair_backtrackable is not None
                or repair_eligible
                or alternative_existed is not None
            ):
                raise Iteration2ValidationError(
                    "successful ordinary phase has repair state"
                )
        elif repair_backtrackable is False:
            if repair_attempted or repair_eligible or alternative_existed is not False:
                raise Iteration2ValidationError(
                    "source failure repair denominator is ambiguous"
                )
        elif repair_backtrackable is True:
            if not repair_attempted or not repair_eligible:
                raise Iteration2ValidationError(
                    "backtrackable repair was not attempted exactly once"
                )
        else:
            raise Iteration2ValidationError(
                "failed repaired route lacks backtrackability status"
            )
        for stage in stages.values():
            if (
                not isinstance(stage, Mapping)
                or stage.get("status") is None
                or stage.get("applicability")
                not in {"applicable", "not_applicable"}
                or (
                    stage.get("applicability") == "not_applicable"
                    and not stage.get("na_reason")
                )
            ):
                raise Iteration2ValidationError(
                    "failure stage metadata is invalid"
                )
        for failure in REPAIR_FAILURE_TYPES:
            if (
                applicability["initial"].get(failure) != "not_applicable"
                or applicability["final"].get(failure)
                != (
                    "applicable"
                    if is_repair and final_observed
                    else "not_applicable"
                )
            ):
                raise Iteration2ValidationError(
                    "repair failure applicability is invalid"
                )
        for failure in ORDINARY_FAILURE_TYPES:
            if (
                applicability["initial"].get(failure)
                != (
                    "applicable" if initial_observed else "not_applicable"
                )
                or applicability["final"].get(failure)
                != (
                    "applicable"
                    if final_observed and not is_repair
                    else "not_applicable"
                )
            ):
                raise Iteration2ValidationError(
                    "ordinary failure applicability is invalid"
                )
        final_diagnostic = row.get("final_failure_diagnostic")
        for diagnostic_name in (
            "initial_failure_diagnostic",
            "final_failure_diagnostic",
        ):
            diagnostic = row.get(diagnostic_name)
            if diagnostic is None:
                continue
            if (
                not isinstance(diagnostic, Mapping)
                or diagnostic.get("distance_tolerance") != distance_tolerance
            ):
                raise Iteration2ValidationError(
                    "failure diagnostic tolerance differs from its condition"
                )
            current_distance = diagnostic.get("current_distance")
            best_distance = diagnostic.get("best_neighbor_distance")
            progress_gap = diagnostic.get("progress_gap")
            if current_distance is None or best_distance is None:
                if progress_gap is not None:
                    raise Iteration2ValidationError(
                        "failure diagnostic has an orphaned progress gap"
                    )
            elif (
                progress_gap is None
                or not isclose(
                    float(progress_gap),
                    float(current_distance) - float(best_distance),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise Iteration2ValidationError(
                    "failure diagnostic progress gap is inconsistent"
                )
        if success and final_diagnostic is not None:
            raise Iteration2ValidationError(
                "successful route has a final failure diagnostic"
            )
        if not success:
            if not isinstance(final_diagnostic, Mapping):
                raise Iteration2ValidationError(
                    "failed route lacks a final diagnostic"
                )
            if (
                final_diagnostic.get("failure_type")
                != row.get("final_failure_type")
                or final_diagnostic.get("stopping_vertex") != walk[-1]
                or final_diagnostic.get("failure_hop") != len(walk) - 1
            ):
                raise Iteration2ValidationError(
                    "final failure diagnostic is inconsistent"
                )
    if observed_keys != expected_keys:
        raise Iteration2ValidationError("route crossing is incomplete")
    if reused_euclidean_records != 8 * pair_count:
        raise Iteration2ValidationError("Euclidean route reuse count is invalid")
    if len(euclidean_signatures) != 2 * pair_count or any(
        len(signatures) != 5 or len(set(signatures)) != 1
        for signatures in euclidean_signatures.values()
    ):
        raise Iteration2ValidationError(
            "uniform scaling changed a Euclidean route"
        )
    reuse = result.get("euclidean_scale_reuse")
    executions = result.get("routing_execution_counts")
    if (
        not isinstance(reuse, Mapping)
        or reuse.get("verification_schema")
        != "iteration2_euclidean_scale_reuse_v1"
        or reuse.get("production_executions_per_pair") != 2
        or reuse.get("route_records_per_pair") != 10
        or reuse.get("reused_route_records_per_pair") != 8
        or not isinstance(executions, Mapping)
        or executions.get("dijkstra") != pair_count
        or executions.get("euclidean_state_machine") != 2 * pair_count
        or executions.get("poincare_ordinary_state_machine") != 9 * pair_count
        or executions.get("poincare_repair_wrappers") != 9 * pair_count
        or executions.get("scientific_route_records")
        != ROUTES_PER_PAIR * pair_count
        or executions.get("euclidean_records_reused_after_scale_proof")
        != 8 * pair_count
    ):
        raise Iteration2ValidationError(
            "routing execution/reuse accounting is invalid"
        )

    graph_rows = result.get("graph_level_rows")
    if not isinstance(graph_rows, Sequence):
        raise Iteration2ValidationError("graph-level rows are missing")
    try:
        validate_graph_level_rows(graph_rows, pair_count=pair_count)
        recomputed = graph_level_rows_from_records(
            routes,
            graph_id=graph_id,
            identity=identity,
            pair_count=pair_count,
        )
    except ValueError as exc:
        raise Iteration2ValidationError(str(exc)) from exc
    if list(graph_rows) != recomputed:
        raise Iteration2ValidationError("graph-level rows do not recompute")
    interactions = result.get("graph_level_interactions")
    if list(interactions or ()) != graph_level_interactions(graph_rows):
        raise Iteration2ValidationError("interaction rows do not recompute")

    sentinel = result.get("high_precision_sentinel")
    if (
        not isinstance(sentinel, Mapping)
        or sentinel.get("selection_is_outcome_independent") is not True
        or sentinel.get("disagreements") != 0
        or sentinel.get("float64_altered_decisions") is not False
    ):
        raise Iteration2ValidationError("high-precision sentinel failed")
    gauge = result.get("gauge_and_centering_diagnostics")
    hydra_gauge = (
        gauge.get("hydra_centering")
        if isinstance(gauge, Mapping)
        else None
    )
    if (
        not isinstance(hydra_gauge, Mapping)
        or hydra_gauge.get("selection_is_outcome_independent") is not True
        or hydra_gauge.get("pair_count") != pair_count
        or hydra_gauge.get("poincare_routing_invariance_required") is not True
        or hydra_gauge.get("poincare_routing_changed_pairs") != 0
        or gauge.get("gauge_selected_using_routing_outcomes") is not False
    ):
        raise Iteration2ValidationError(
            "gauge and centering diagnostics are invalid"
        )
    return {
        "vertices": n,
        "pairs": pair_count,
        "dijkstra_records": len(dijkstra),
        "route_records": len(routes),
        "graph_level_rows": len(graph_rows),
        "interaction_rows": len(interactions),
    }


def _scientific_payload_bytes(result: Mapping[str, object]) -> bytes:
    payload = dict(result)
    payload.pop("timing", None)
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_scheduled_iteration2_graph_result(
    result: Mapping[str, object],
    spec: GraphSpec,
) -> dict[str, object]:
    """Regenerate a frozen graph and compare every scientific result field."""

    started = perf_counter()
    counts = validate_iteration2_graph_result(result)
    identity = result.get("graph_identity")
    expected_identity = {
        "graph_id": spec.graph_id,
        "model": spec.model,
        "n": spec.n,
        "m": spec.m,
        "replicate_index": spec.replicate_index,
    }
    if not isinstance(identity, Mapping) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise Iteration2ValidationError(
            "checkpoint identity differs from the frozen schedule"
        )
    seeds = seeds_for_graph(spec)
    expected_seeds = {
        "graph": seeds.graph,
        "embedding_provenance": seeds.embedding_provenance,
        "pairs": seeds.pairs,
        "routing_priority": seeds.routing_priority,
        "validation_sentinel": seeds.validation_sentinel,
        "er_attempt_schedule": list(seeds.er_attempts),
        "identity_scope": "data_generation_hash_only",
    }
    if result.get("seeds") != expected_seeds:
        raise Iteration2ValidationError(
            "checkpoint seed provenance differs from the frozen schedule"
        )
    pairs = result.get("pairs")
    if not isinstance(pairs, Sequence) or len(pairs) != PAIRS_PER_GRAPH:
        raise Iteration2ValidationError(
            "scheduled checkpoint must contain exactly 1,000 frozen pairs"
        )
    sentinel = result.get("high_precision_sentinel")
    expected_indices = (
        tuple(range(PAIRS_PER_GRAPH))
        if is_full_oracle_graph(spec)
        else sentinel_pair_indices(spec.graph_id, PAIRS_PER_GRAPH)
    )
    expected_mode = (
        "prespecified_full_pair_oracle_graph"
        if is_full_oracle_graph(spec)
        else "domain_separated_random_pair_sentinel"
    )
    if (
        not isinstance(sentinel, Mapping)
        or sentinel.get("selection_mode") != expected_mode
        or tuple(sentinel.get("pair_indices", ())) != expected_indices
        or sentinel.get("route_decisions_checked")
        != len(expected_indices) * ROUTES_PER_PAIR
    ):
        raise Iteration2ValidationError(
            "checkpoint oracle coverage differs from the frozen policy"
        )
    regenerated = execute_scheduled_graph(spec, pair_count=PAIRS_PER_GRAPH)
    validate_iteration2_graph_result(regenerated)
    if _scientific_payload_bytes(result) != _scientific_payload_bytes(
        regenerated
    ):
        raise Iteration2ValidationError(
            "checkpoint scientific payload differs from deterministic "
            "regeneration"
        )
    return {
        **counts,
        "scheduled_graph_id": spec.graph_id,
        "full_oracle_graph": is_full_oracle_graph(spec),
        "scientific_regeneration_performed": True,
        "resume_validation_seconds": perf_counter() - started,
    }


def scheduled_specifications() -> dict[str, GraphSpec]:
    schedule = full_schedule()
    result = {spec.graph_id: spec for spec in schedule}
    if len(result) != len(schedule):
        raise Iteration2ValidationError("frozen schedule graph IDs collide")
    return result


def graph_level_rows_from_records(
    routes: Sequence[Mapping[str, object]],
    *,
    graph_id: str,
    identity: Mapping[str, object],
    pair_count: int,
) -> list[dict[str, object]]:
    from iteration2_analysis import graph_level_rows

    return graph_level_rows(
        routes,
        graph_id=graph_id,
        model=str(identity["model"]),
        n=int(identity["n"]),
        m=int(identity["m"]),
        replicate_index=int(identity["replicate_index"]),
        pair_count=pair_count,
    )
