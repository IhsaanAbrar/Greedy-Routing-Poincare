"""In-memory Iteration 2 graph execution with balanced routing conditions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import blake2s
import json
from math import isclose, isfinite
from time import perf_counter

import networkx as nx

from embedding import prepare_embedding_input
from graph_generation import (
    GeneratedGraph,
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
)
from iteration2_analysis import (
    graph_level_interactions,
    graph_level_rows,
)
from iteration2_config import (
    ANALYSIS_PLAN_HASH,
    COMBINED_PROTOCOL_HASH,
    DATA_GENERATION_HASH,
    FAILURE_SCHEMA_VERSION,
    ITERATION2_RESULT_SCHEMA,
    ITERATION2_RUN_IDENTITY,
    MAX_ER_ATTEMPTS,
    MATCHED_RADIUS_LABELS,
    MDS_NATIVE_ARTIFACT,
    OUTPUT_SCHEMA_HASH,
    GraphSeeds,
    GraphSpec,
    is_full_oracle_graph,
    sample_ordered_pairs,
    seeds_for_graph,
    sentinel_pair_indices,
)
from iteration2_coordinates import (
    CoordinateCondition,
    create_iteration2_embeddings,
    euclidean_routing_tolerance,
    native_condition,
    poincare_routing_tolerance,
)
from iteration2_oracle import audit_production_result
from iteration2_routing import (
    NUMERICAL_INVARIANT_FAILURE,
    ORDINARY_FAILURE_TYPES,
    REPAIR_FAILURE_TYPES,
    RoutingPriorityContext,
    euclidean_greedy_route_v2,
    poincare_greedy_route_v2,
    prepare_iteration2_routing,
    repaired_poincare_greedy_route_v2,
)
from iteration2_runtime_guard import scientific_operation_boundary
from network_metrics import (
    calculate_network_metrics,
    prepare_all_pairs_shortest_paths,
)
from poincare_distance import euclidean_distance, poincare_distance
from routing import (
    dijkstra_benchmark,
)


ROUTES_PER_PAIR = 28
PAIR_IDENTITY_SCHEMA = "iteration2_pair_sampling_identity_v1"
ROUTE_IDENTITY_SCHEMA = "iteration2_route_identity_v1"
FORBIDDEN_SCIENTIFIC_FAILURES = frozenset(
    {NUMERICAL_INVARIANT_FAILURE, "attempted_revisit", "ordinary_revisit"}
)


class NumericalRoutingDecisionError(RuntimeError):
    """Raised when production, independent, and high-precision decisions differ."""


@scientific_operation_boundary("dijkstra")
def _execute_dijkstra_benchmark(
    graph: nx.Graph,
    source: int,
    destination: int,
    *,
    expected_shortest_path_length: int,
) -> object:
    return dijkstra_benchmark(
        graph,
        source,
        destination,
        expected_shortest_path_length=expected_shortest_path_length,
    )


def _identity_digest(domain: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        {"domain": domain, **dict(payload)},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return blake2s(encoded, digest_size=16, person=b"i2idv1").hexdigest()


def _pair_identity(
    *,
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


def _priority_context(
    *,
    graph_id: str,
    pair_index: int,
    source: int,
    destination: int,
) -> RoutingPriorityContext:
    return RoutingPriorityContext(
        data_generation_hash=DATA_GENERATION_HASH,
        graph_id=graph_id,
        pair_index=pair_index,
        source=source,
        destination=destination,
    )


def _call_router(
    function: object,
    graph: nx.Graph,
    coordinates: Mapping[int, Sequence[float]],
    source: int,
    destination: int,
    *,
    tolerance: float,
    priority_context: RoutingPriorityContext,
    ordinary_result: object | None = None,
) -> object:
    keyword: dict[str, object] = {
        "tolerance": tolerance,
        "priority_context": priority_context,
    }
    if ordinary_result is not None:
        keyword["ordinary_result"] = ordinary_result
    return function(  # type: ignore[operator]
        graph,
        coordinates,
        source,
        destination,
        **keyword,
    )


def _result_resource(result: object, name: str) -> int:
    value = getattr(result, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"routing result lacks valid {name}")
    return value


def _canonical_graph(graph: nx.Graph) -> nx.Graph:
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or not nx.is_connected(graph)
    ):
        raise ValueError("Iteration 2 graph must be simple, connected, undirected")
    nodes = tuple(sorted(graph.nodes))
    if nodes != tuple(range(len(nodes))):
        raise ValueError("Iteration 2 graph nodes must be integer IDs 0 through n-1")
    result = nx.Graph()
    result.add_nodes_from(nodes)
    result.add_edges_from(
        sorted(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in graph.edges
        )
    )
    return result


def _generation_provenance(
    *,
    graph: nx.Graph,
    model: str,
    n: int,
    m: int,
    graph_seed: int | None,
    metadata: Mapping[str, object],
    er_attempt_seed_schedule: Sequence[int] | None,
) -> dict[str, object]:
    """Normalize model-specific generator evidence into one exact schema."""

    attempted_seeds = tuple(
        int(seed) for seed in metadata.get("generation_attempt_seeds", ())
    )
    attempt_count = int(metadata.get("generation_attempt_count", 0))
    accepted_seed = metadata.get("generation_attempt_seed")
    rejected_count = int(metadata.get("rejected_disconnected_count", -1))
    if (
        attempt_count <= 0
        or len(attempted_seeds) != attempt_count
        or accepted_seed != attempted_seeds[-1]
        or metadata.get("graph_model") != model
        or int(metadata.get("n", -1)) != n
        or int(metadata.get("realised_edge_count", -1))
        != graph.number_of_edges()
        or not isclose(
            float(metadata.get("realised_average_degree", -1.0)),
            2.0 * graph.number_of_edges() / n,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError("graph generation provenance is incomplete or inconsistent")
    if graph_seed is not None and metadata.get("graph_seed") != graph_seed:
        raise RuntimeError("graph seed and generation provenance disagree")
    attempts = [
        {
            "attempt_index": index,
            "seed": seed,
            "outcome": (
                "accepted_connected"
                if index == attempt_count - 1
                else "rejected_disconnected"
            ),
        }
        for index, seed in enumerate(attempted_seeds)
    ]
    common: dict[str, object] = {
        "schema": "iteration2_graph_generation_provenance_v1",
        "model": model,
        "graph_seed": graph_seed,
        "attempt_count": attempt_count,
        "rejection_count": rejected_count,
        "attempted_seeds": list(attempted_seeds),
        "attempts": attempts,
        "accepted_seed": accepted_seed,
        "realised_edge_count": graph.number_of_edges(),
        "realised_average_degree": 2.0 * graph.number_of_edges() / n,
        "simple": True,
        "connected": True,
        "undirected": True,
        "unweighted": True,
        "node_ids": "integers_0_through_n_minus_1",
        "acceptance_selected_using_routing_outcomes": False,
    }
    if model == "erdos_renyi":
        numerator = 2 * m * (n - m)
        denominator = n * (n - 1)
        scheduled = tuple(er_attempt_seed_schedule or attempted_seeds)
        if (
            rejected_count != attempt_count - 1
            or len(scheduled) < attempt_count
            or (
                er_attempt_seed_schedule is not None
                and len(scheduled) != MAX_ER_ATTEMPTS
            )
            or scheduled[:attempt_count] != attempted_seeds
            or metadata.get("p_exact_numerator") != numerator
            or metadata.get("p_exact_denominator") != denominator
            or not isclose(
                float(metadata.get("p", -1.0)),
                numerator / denominator,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise RuntimeError("ER retry provenance differs from the frozen design")
        common.update(
            {
                "p": numerator / denominator,
                "p_exact_numerator": numerator,
                "p_exact_denominator": denominator,
                "maximum_attempts": MAX_ER_ATTEMPTS,
                "scheduled_attempt_seeds": list(scheduled),
                "acceptance_rule": "first_connected_G_n_p_within_50_attempts",
                "largest_component_substitution": False,
            }
        )
    elif model == "barabasi_albert":
        expected_edges = m * (n - m)
        if (
            attempt_count != 1
            or rejected_count != 0
            or metadata.get("m") != m
            or graph.number_of_edges() != expected_edges
            or metadata.get("ba_initial_graph") is None
        ):
            raise RuntimeError("BA provenance differs from the frozen design")
        common.update(
            {
                "m": m,
                "initial_graph_convention": metadata["ba_initial_graph"],
                "expected_edge_count": expected_edges,
                "exact_finite_size_expected_average_degree": (
                    2.0 * m * (n - m) / n
                ),
                "acceptance_rule": "connected_by_prespecified_BA_construction",
            }
        )
    else:
        raise RuntimeError("unknown graph model in provenance")
    return common


@scientific_operation_boundary("graph_generation")
def generate_iteration2_graph(
    spec: GraphSpec,
) -> tuple[GeneratedGraph, GraphSeeds]:
    """Generate one graph from only the new Iteration 2 seed namespace."""

    seeds = seeds_for_graph(spec)
    if spec.model == "erdos_renyi":
        generated = generate_connected_erdos_renyi(
            n=spec.n,
            p=spec.er_probability,
            graph_seed=seeds.graph,
            replicate_index=spec.replicate_index,
            max_attempts=len(seeds.er_attempts),
            setting_index=spec.schedule_index // 20,
            attempt_seeds=seeds.er_attempts,
            p_exact_numerator=2 * spec.m * (spec.n - spec.m),
            p_exact_denominator=spec.n * (spec.n - 1),
        )
    elif spec.model == "barabasi_albert":
        generated = generate_connected_barabasi_albert(
            n=spec.n,
            m=spec.m,
            graph_seed=seeds.graph,
            replicate_index=spec.replicate_index,
            setting_index=spec.schedule_index // 20,
        )
    else:
        raise ValueError("unknown graph model")
    return generated, seeds


def _condition_contexts(
    graph: nx.Graph,
    conditions: Sequence[CoordinateCondition],
) -> dict[str, dict[str, object]]:
    contexts: dict[str, dict[str, object]] = {}
    for condition in conditions:
        euclidean = prepare_iteration2_routing(
            graph,
            condition.coordinates,
            euclidean_distance,
            metric_name=f"{condition.condition_id}:euclidean",
        )
        poincare_allowed = (
            condition.condition_id != MDS_NATIVE_ARTIFACT
            and condition.poincare_routable
        )
        poincare = (
            prepare_iteration2_routing(
                graph,
                condition.coordinates,
                poincare_distance,
                metric_name=f"{condition.condition_id}:poincare",
            )
            if poincare_allowed
            else None
        )
        if poincare is not None and tuple(euclidean.items()) != tuple(
            poincare.items()
        ):
            raise RuntimeError("routing metrics received different snapshots")
        contexts[condition.condition_id] = {
            "condition": condition,
            "euclidean": euclidean,
            "poincare": poincare,
            "euclidean_tolerance": euclidean_routing_tolerance(condition),
            "poincare_tolerance": (
                poincare_routing_tolerance(condition)
                if poincare_allowed
                else None
            ),
        }
    return contexts


def _route_record(
    *,
    graph_id: str,
    pair_index: int,
    pair_identity: str,
    condition_id: str,
    method_id: str,
    metric_id: str,
    result: object,
    dijkstra_length: int,
    condition_tolerance: float,
    diagnostic_scale: float = 1.0,
    execution_reused: bool = False,
    reused_from_condition_id: str | None = None,
) -> dict[str, object]:
    if (
        result.initial_failure_type in FORBIDDEN_SCIENTIFIC_FAILURES
        or result.final_failure_type in FORBIDDEN_SCIENTIFIC_FAILURES
    ):
        raise RuntimeError(
            "routing invariant failures must abort and cannot enter records"
        )
    stretch = (
        result.route_length / dijkstra_length if result.success else None
    )
    if result.success and result.route_length < dijkstra_length:
        raise RuntimeError("greedy route cannot be shorter than Dijkstra")
    tolerance = float(condition_tolerance)
    scale = float(diagnostic_scale)
    if (
        not isfinite(tolerance)
        or tolerance <= 0.0
        or not isfinite(scale)
        or scale <= 0.0
    ):
        raise RuntimeError("route diagnostic scaling is invalid")
    if execution_reused:
        if not isclose(
            tolerance,
            float(result.distance_tolerance) * scale,
            rel_tol=4e-15,
            abs_tol=0.0,
        ):
            raise RuntimeError("reused route tolerance is not scale equivariant")
    elif tolerance != float(result.distance_tolerance) or scale != 1.0:
        raise RuntimeError("executed route tolerance differs from its condition")

    def diagnostic(value: object | None) -> dict[str, object] | None:
        if value is None:
            return None
        row = value.to_dict()
        for field in (
            "current_distance",
            "best_neighbor_distance",
            "progress_gap",
        ):
            if row[field] is not None:
                row[field] = float(row[field]) * scale
        row["distance_tolerance"] = tolerance
        return row
    is_repair_method = method_id == "repaired_poincare_greedy"
    initial_observed = result.initial_failure_type is not None
    final_observed = result.final_failure_type is not None
    repair_applicable = is_repair_method and initial_observed
    stage_applicability = {
        "initial": {
            "status": "observed" if initial_observed else "not_observed",
            "applicability": (
                "applicable" if initial_observed else "not_applicable"
            ),
            "na_reason": (
                None
                if initial_observed
                else "route_succeeded_before_initial_failure"
            ),
        },
        "repair": {
            "status": (
                "succeeded"
                if repair_applicable and result.repair_succeeded
                else "failed"
                if repair_applicable
                else "not_applicable"
            ),
            "applicability": (
                "applicable" if repair_applicable else "not_applicable"
            ),
            "na_reason": (
                None
                if repair_applicable
                else "ordinary_route_succeeded"
                if is_repair_method
                else "routing_method_has_no_repair_stage"
            ),
        },
        "final": {
            "status": "observed" if final_observed else "not_observed",
            "applicability": (
                "applicable" if final_observed else "not_applicable"
            ),
            "na_reason": None if final_observed else "route_succeeded",
        },
    }
    category_applicability = {
        "initial": {
            **{
                failure: (
                    "applicable" if initial_observed else "not_applicable"
                )
                for failure in ORDINARY_FAILURE_TYPES
            },
            **{
                failure: "not_applicable"
                for failure in REPAIR_FAILURE_TYPES
            },
        },
        "final": {
            **{
                failure: (
                    "applicable"
                    if final_observed and not is_repair_method
                    else "not_applicable"
                )
                for failure in ORDINARY_FAILURE_TYPES
            },
            **{
                failure: (
                    "applicable"
                    if final_observed and is_repair_method
                    else "not_applicable"
                )
                for failure in REPAIR_FAILURE_TYPES
            },
        },
    }
    physical_hops = _result_resource(result, "physical_hops")
    forwarding_decisions = _result_resource(result, "forwarding_decisions")
    logical_distance_evaluations = _result_resource(
        result,
        "logical_distance_evaluations",
    )
    peak_history_vertices = _result_resource(result, "peak_history_vertices")
    if physical_hops != result.route_length:
        raise RuntimeError("physical hop accounting disagrees with the walk")
    route_identity = {
        "schema": ROUTE_IDENTITY_SCHEMA,
        "graph_id": graph_id,
        "pair_index": pair_index,
        "pair_identity": pair_identity,
        "source": result.source,
        "destination": result.destination,
        "coordinate_condition_id": condition_id,
        "method_id": method_id,
    }
    return {
        "graph_id": graph_id,
        "pair_index": pair_index,
        "pair_identity": pair_identity,
        "source": result.source,
        "destination": result.destination,
        "coordinate_condition_id": condition_id,
        "method_id": method_id,
        "metric_id": metric_id,
        "distance_tolerance": tolerance,
        "priority_context_id": result.priority_context_id,
        "route_identity": route_identity,
        "route_identity_hash": _identity_digest("route", route_identity),
        "success": result.success,
        "walk": list(result.walk),
        "route_length": result.route_length,
        "physical_edge_traversals": physical_hops,
        "physical_hops": physical_hops,
        "forwarding_decisions": forwarding_decisions,
        "logical_distance_evaluations": logical_distance_evaluations,
        "peak_history_vertices": peak_history_vertices,
        "resource_accounting": {
            "physical_edge_traversals": {
                "value": physical_hops,
                "unit": "edge_traversals",
                "applicability": "applicable",
                "na_reason": None,
            },
            "forwarding_decisions": {
                "value": forwarding_decisions,
                "unit": "routing_decisions",
                "applicability": "applicable",
                "na_reason": None,
            },
            "logical_distance_evaluations": {
                "value": logical_distance_evaluations,
                "unit": "logical_distance_requests",
                "applicability": "applicable",
                "na_reason": None,
                "cache_hit_independent": True,
            },
            "peak_history_vertices": {
                "value": peak_history_vertices,
                "unit": "vertex_ids_held_simultaneously",
                "applicability": "applicable",
                "na_reason": None,
            },
        },
        "execution_reused": execution_reused,
        "reused_from_condition_id": reused_from_condition_id,
        "reuse_basis": (
            "verified_uniform_euclidean_scale_invariance"
            if execution_reused
            else None
        ),
        "dijkstra_length": dijkstra_length,
        "stretch": stretch,
        "initial_failure_type": result.initial_failure_type,
        "final_failure_type": result.final_failure_type,
        "repair_attempted": result.repair_attempted,
        "repair_succeeded": result.repair_succeeded,
        "repair_backtrackable": result.repair_backtrackable,
        "repair_eligible": result.repair_eligible,
        "repair_alternative_existed": result.repair_alternative_existed,
        "repair_alternative_selected": result.repair_alternative_selected,
        "repair_attempt_count": result.repair_attempt_count,
        "failure_schema": FAILURE_SCHEMA_VERSION,
        "failure_stage_applicability": stage_applicability,
        "failure_category_applicability": category_applicability,
        "initial_failure_diagnostic": diagnostic(
            result.initial_failure_diagnostic
        ),
        "final_failure_diagnostic": diagnostic(
            result.final_failure_diagnostic
        ),
        "repair_backtracked_vertex": result.repair_backtracked_vertex,
        "repair_excluded_branch": result.repair_excluded_branch,
        "repair_selected_alternative": result.repair_selected_alternative,
        "repair_denominator_membership": {
            "ordinary_poincare_failed": (
                is_repair_method and initial_observed
            ),
            "failure_was_backtrackable": (
                is_repair_method and result.repair_backtrackable is True
            ),
            "repair_eligible": is_repair_method and result.repair_eligible,
            "alternative_existed": (
                is_repair_method and result.repair_alternative_existed is True
            ),
            "repair_attempted": is_repair_method and result.repair_attempted,
            "route_recovered": is_repair_method and result.repair_succeeded,
            "applicability": (
                "applicable" if is_repair_method else "not_applicable"
            ),
            "na_reason": (
                None if is_repair_method else "routing_method_has_no_repair_stage"
            ),
        },
    }


def _route_signature(result: object) -> tuple[object, ...]:
    return (
        result.success,
        tuple(result.walk),
        result.final_failure_type,
        result.initial_failure_type,
    )


def _verify_euclidean_scale_reuse(embeddings: object) -> dict[str, object]:
    """Prove the two four-radius families are uniform base rescalings."""

    native_by_family = {
        "hydra": embeddings.hydra_native,
        "mds": embeddings.mds_native,
    }
    conditions_by_family = {
        family: tuple(
            condition
            for condition in embeddings.matched_conditions
            if condition.embedding_family == family
        )
        for family in native_by_family
    }
    details: dict[str, object] = {}
    for family, native in native_by_family.items():
        conditions = conditions_by_family[family]
        if len(conditions) != len(MATCHED_RADIUS_LABELS):
            raise RuntimeError("Euclidean reuse radius family is incomplete")
        native_tolerance = euclidean_routing_tolerance(native)
        for condition in conditions:
            scale = condition.scale_factor
            if not isclose(
                euclidean_routing_tolerance(condition),
                native_tolerance * abs(scale),
                rel_tol=4e-15,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Euclidean tolerance is not equivariant under scaling"
                )
            for node, native_point in native.coordinates.items():
                transformed = condition.coordinates[node]
                expected = (
                    float(native_point[0]) * scale,
                    float(native_point[1]) * scale,
                )
                if not all(
                    isclose(
                        float(observed),
                        expected_value,
                        rel_tol=4e-15,
                        abs_tol=1e-15,
                    )
                    for observed, expected_value in zip(
                        transformed,
                        expected,
                        strict=True,
                    )
                ):
                    raise RuntimeError(
                        "matched coordinates are not a uniform base scaling"
                    )
        details[family] = {
            "base_condition_id": native.condition_id,
            "verified_condition_ids": [
                condition.condition_id for condition in conditions
            ],
            "coordinate_relation": "uniform_positive_scalar_multiple",
            "tolerance_relation": "same_positive_scalar_multiple",
            "route_decision_invariance": "proved_before_reuse",
        }
    return {
        "verification_schema": "iteration2_euclidean_scale_reuse_v1",
        "families": details,
        "production_executions_per_pair": 2,
        "route_records_per_pair": 10,
        "reused_route_records_per_pair": 8,
    }


@scientific_operation_boundary("graph_workload_execution")
def execute_iteration2_graph(
    graph: nx.Graph,
    *,
    graph_id: str,
    model: str,
    n: int,
    m: int,
    replicate_index: int,
    pair_seed: int,
    pair_count: int,
    graph_seed: int | None = None,
    embedding_provenance_seed: int | None = None,
    routing_priority_seed: int | None = None,
    validation_sentinel_seed: int | None = None,
    er_attempt_seed_schedule: Sequence[int] | None = None,
    generation_metadata: Mapping[str, object] | None = None,
    audit_all_pairs: bool = False,
    run_identity: str = ITERATION2_RUN_IDENTITY,
) -> dict[str, object]:
    """Execute one graph without writing any result directory."""

    setup_started = perf_counter()
    canonical = _canonical_graph(graph)
    if canonical.number_of_nodes() != n:
        raise ValueError("declared n differs from graph order")
    shortest_paths = prepare_all_pairs_shortest_paths(canonical)
    embedding_input = prepare_embedding_input(
        canonical,
        shortest_paths,
        configuration_fingerprint=DATA_GENERATION_HASH,
        tolerance=1e-12,
    )
    embeddings = create_iteration2_embeddings(embedding_input)
    euclidean_reuse = _verify_euclidean_scale_reuse(embeddings)
    all_conditions = (
        embeddings.hydra_native,
        embeddings.mds_native,
        *embeddings.matched_conditions,
    )
    contexts = _condition_contexts(canonical, all_conditions)
    hydra_uncentered = native_condition(
        "hydra_uncentered_gauge_diagnostic",
        "hydra",
        embeddings.hydra_uncentered_reference,
        embedding_input.node_order,
    )
    hydra_uncentered_euclidean_tolerance = euclidean_routing_tolerance(
        hydra_uncentered
    )
    hydra_gauge_poincare_tolerance = max(
        poincare_routing_tolerance(hydra_uncentered),
        poincare_routing_tolerance(embeddings.hydra_native),
    )
    hydra_uncentered_euclidean_context = prepare_iteration2_routing(
        canonical,
        hydra_uncentered.coordinates,
        euclidean_distance,
        metric_name="hydra_uncentered_gauge_diagnostic:euclidean",
    )
    hydra_uncentered_poincare_context = prepare_iteration2_routing(
        canonical,
        hydra_uncentered.coordinates,
        poincare_distance,
        metric_name="hydra_uncentered_gauge_diagnostic:poincare",
    )
    pairs = sample_ordered_pairs(
        canonical.nodes,
        pair_count,
        graph_id=graph_id,
        pair_seed=pair_seed,
    )
    if len(pairs) != pair_count or len(set(pairs)) != pair_count:
        raise RuntimeError("pair sampler did not return the frozen design")
    pair_records = [
        {
            "pair_index": pair_index,
            "source": source,
            "destination": destination,
            "sampling_identity_schema": PAIR_IDENTITY_SCHEMA,
            "sampling_identity": _pair_identity(
                graph_id=graph_id,
                pair_index=pair_index,
                source=source,
                destination=destination,
            ),
        }
        for pair_index, (source, destination) in enumerate(pairs)
    ]
    if len({row["sampling_identity"] for row in pair_records}) != pair_count:
        raise RuntimeError("pair sampling identities collided")
    sentinel_indices = (
        set(range(pair_count))
        if audit_all_pairs
        else set(sentinel_pair_indices(graph_id, pair_count))
    )
    setup_seconds = perf_counter() - setup_started
    route_started = perf_counter()

    dijkstra_records: list[dict[str, object]] = []
    route_records: list[dict[str, object]] = []
    sentinel_checks = 0
    sentinel_disagreements = 0
    poincare_sensitivity = {
        "hydra": {
            "poincare_greedy_changed_pairs": 0,
            "repaired_poincare_greedy_changed_pairs": 0,
        },
        "mds": {
            "poincare_greedy_changed_pairs": 0,
            "repaired_poincare_greedy_changed_pairs": 0,
        },
    }
    hydra_gauge_diagnostics = {
        "coordinate_reference": (
            "independent_uncentered_hydra_reconstruction"
        ),
        "selection_is_outcome_independent": True,
        "pair_count": pair_count,
        "euclidean_routing_changed_pairs": 0,
        "poincare_routing_changed_pairs": 0,
        "poincare_routing_invariance_required": True,
        "poincare_common_distance_tolerance": (
            hydra_gauge_poincare_tolerance
        ),
    }

    for pair_index, (source, destination) in enumerate(pairs):
        pair_identity = str(pair_records[pair_index]["sampling_identity"])
        priority_context = _priority_context(
            graph_id=graph_id,
            pair_index=pair_index,
            source=source,
            destination=destination,
        )
        benchmark = _execute_dijkstra_benchmark(
            canonical,
            source,
            destination,
            expected_shortest_path_length=(
                shortest_paths.distances[source][destination]
            ),
        )
        dijkstra_records.append(
            {
                "graph_id": graph_id,
                "pair_index": pair_index,
                "pair_identity": pair_identity,
                "source": source,
                "destination": destination,
                "walk": list(benchmark.walk),
                "route_length": benchmark.route_length,
                "apsp_length": shortest_paths.distances[source][destination],
                "sampling_identity_schema": PAIR_IDENTITY_SCHEMA,
            }
        )
        matched_signatures: dict[
            tuple[str, str],
            list[tuple[object, ...]],
        ] = {}
        euclidean_results: dict[str, tuple[str, object]] = {}
        for condition in all_conditions:
            context = contexts[condition.condition_id]
            reused_euclidean = condition.embedding_family in euclidean_results
            if reused_euclidean:
                reused_from, euclidean_result = euclidean_results[
                    condition.embedding_family
                ]
            else:
                reused_from = condition.condition_id
                euclidean_result = _call_router(
                    euclidean_greedy_route_v2,
                    canonical,
                    context["euclidean"],
                    source,
                    destination,
                    tolerance=float(context["euclidean_tolerance"]),
                    priority_context=priority_context,
                )
                euclidean_results[condition.embedding_family] = (
                    condition.condition_id,
                    euclidean_result,
                )
            methods = [
                (
                    "euclidean_greedy",
                    "euclidean",
                    False,
                    euclidean_result,
                    context["euclidean_tolerance"],
                    reused_euclidean,
                    reused_from if reused_euclidean else None,
                )
            ]
            if (
                condition.condition_id != MDS_NATIVE_ARTIFACT
                and condition.poincare_routable
            ):
                ordinary_poincare = _call_router(
                    poincare_greedy_route_v2,
                    canonical,
                    context["poincare"],
                    source,
                    destination,
                    tolerance=float(context["poincare_tolerance"]),
                    priority_context=priority_context,
                )
                repaired_poincare = _call_router(
                    repaired_poincare_greedy_route_v2,
                    canonical,
                    context["poincare"],
                    source,
                    destination,
                    tolerance=float(context["poincare_tolerance"]),
                    priority_context=priority_context,
                    ordinary_result=ordinary_poincare,
                )
                methods.extend(
                    (
                        (
                            "poincare_greedy",
                            "poincare",
                            False,
                            ordinary_poincare,
                            context["poincare_tolerance"],
                            False,
                            None,
                        ),
                        (
                            "repaired_poincare_greedy",
                            "poincare",
                            True,
                            repaired_poincare,
                            context["poincare_tolerance"],
                            False,
                            None,
                        ),
                    )
                )
            for (
                method_id,
                metric,
                repaired,
                result,
                tolerance,
                execution_reused,
                reused_from_condition_id,
            ) in methods:
                route_records.append(
                    _route_record(
                        graph_id=graph_id,
                        pair_index=pair_index,
                        pair_identity=pair_identity,
                        condition_id=condition.condition_id,
                        method_id=method_id,
                        metric_id=metric,
                        result=result,
                        dijkstra_length=benchmark.route_length,
                        condition_tolerance=float(tolerance),
                        diagnostic_scale=(
                            float(condition.scale_factor)
                            if execution_reused
                            else 1.0
                        ),
                        execution_reused=execution_reused,
                        reused_from_condition_id=reused_from_condition_id,
                    )
                )
                if condition.condition_kind == "matched_radius_sensitivity":
                    matched_signatures.setdefault(
                        (condition.embedding_family, method_id),
                        [],
                    ).append(_route_signature(result))
                if pair_index in sentinel_indices:
                    agreement = audit_production_result(
                        result,
                        graph=canonical,
                        coordinates=condition.coordinates,
                        source=source,
                        destination=destination,
                        metric=metric,
                        tolerance=float(tolerance),
                        repaired=repaired,
                        priority_context=priority_context,
                    )
                    sentinel_checks += 1
                    if not (
                        agreement.float64_matches_production
                        and agreement.high_precision_matches_production
                        and agreement.float64_matches_high_precision
                    ):
                        sentinel_disagreements += 1
                        raise NumericalRoutingDecisionError(
                            "float64/production/high-precision routing "
                            f"disagreement: graph={graph_id} pair={pair_index} "
                            f"condition={condition.condition_id} method={method_id}"
                        )

        for family in ("hydra", "mds"):
            euclidean = matched_signatures[(family, "euclidean_greedy")]
            if len(euclidean) != 4 or len(set(euclidean)) != 1:
                raise NumericalRoutingDecisionError(
                    "uniform scaling changed Euclidean routing decisions"
                )
            for method in (
                "poincare_greedy",
                "repaired_poincare_greedy",
            ):
                signatures = matched_signatures[(family, method)]
                if len(signatures) != 4:
                    raise RuntimeError("Poincare radius grid is incomplete")
                if len(set(signatures)) > 1:
                    poincare_sensitivity[family][
                        f"{method}_changed_pairs"
                    ] += 1
        uncentered_euclidean = _call_router(
            euclidean_greedy_route_v2,
            canonical,
            hydra_uncentered_euclidean_context,
            source,
            destination,
            tolerance=hydra_uncentered_euclidean_tolerance,
            priority_context=priority_context,
        )
        centered_euclidean = euclidean_results["hydra"][1]
        if _route_signature(uncentered_euclidean) != _route_signature(
            centered_euclidean
        ):
            hydra_gauge_diagnostics[
                "euclidean_routing_changed_pairs"
            ] += 1
        uncentered_poincare = _call_router(
            poincare_greedy_route_v2,
            canonical,
            hydra_uncentered_poincare_context,
            source,
            destination,
            tolerance=hydra_gauge_poincare_tolerance,
            priority_context=priority_context,
        )
        centered_poincare = _call_router(
            poincare_greedy_route_v2,
            canonical,
            contexts[embeddings.hydra_native.condition_id]["poincare"],
            source,
            destination,
            tolerance=hydra_gauge_poincare_tolerance,
            priority_context=priority_context,
        )
        if _route_signature(uncentered_poincare) != _route_signature(
            centered_poincare
        ):
            hydra_gauge_diagnostics[
                "poincare_routing_changed_pairs"
            ] += 1

    if len(dijkstra_records) != pair_count:
        raise RuntimeError("Dijkstra did not execute exactly once per pair")
    if len(route_records) != pair_count * ROUTES_PER_PAIR:
        raise RuntimeError("route record count differs from crossed design")
    if hydra_gauge_diagnostics["poincare_routing_changed_pairs"] != 0:
        raise NumericalRoutingDecisionError(
            "Hydra centering changed Poincare routing on the independent "
            "uncentered reconstruction"
        )
    graph_rows = graph_level_rows(
        route_records,
        graph_id=graph_id,
        model=model,
        n=n,
        m=m,
        replicate_index=replicate_index,
        pair_count=pair_count,
    )
    interactions = graph_level_interactions(graph_rows)
    route_seconds = perf_counter() - route_started
    pair_distances = [
        int(shortest_paths.distances[source][destination])
        for source, destination in pairs
    ]
    ordered_pair_distances = sorted(pair_distances)
    graph_diagnostics = _graph_and_pair_diagnostics(
        canonical,
        shortest_paths=shortest_paths,
        model=model,
        n=n,
        m=m,
        generation_metadata=generation_metadata or {},
        ordered_pair_distances=ordered_pair_distances,
    )
    generation_provenance = _generation_provenance(
        graph=canonical,
        model=model,
        n=n,
        m=m,
        graph_seed=graph_seed,
        metadata=generation_metadata or {},
        er_attempt_seed_schedule=er_attempt_seed_schedule,
    )
    result = {
        "result_schema": ITERATION2_RESULT_SCHEMA,
        "data_generation_hash": DATA_GENERATION_HASH,
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "output_schema_hash": OUTPUT_SCHEMA_HASH,
        "protocol_hash": COMBINED_PROTOCOL_HASH,
        "run_identity": run_identity,
        "scientific_status": (
            "iteration2_scientific_graph"
            if graph_id.startswith("i2_")
            else "excluded_non_scientific"
        ),
        "checkpoint_completion": {
            "status": "complete",
            "atomic_payload": True,
        },
        "graph_identity": {
            "graph_id": graph_id,
            "model": model,
            "n": n,
            "m": m,
            "replicate_index": replicate_index,
            "node_order": list(range(n)),
            "canonical_edges": [
                [int(left), int(right)] for left, right in canonical.edges
            ],
        },
        "seeds": {
            "graph": graph_seed,
            "embedding_provenance": embedding_provenance_seed,
            "pairs": pair_seed,
            "routing_priority": routing_priority_seed,
            "validation_sentinel": validation_sentinel_seed,
            "er_attempt_schedule": list(er_attempt_seed_schedule or ()),
            "identity_scope": "data_generation_hash_only",
        },
        "generation_metadata": dict(generation_metadata or {}),
        "generation_provenance": generation_provenance,
        "graph_and_pair_diagnostics": graph_diagnostics,
        "embedding_input_fingerprint": embedding_input.input_fingerprint,
        "embedding_metadata": embeddings.metadata,
        "native_embedding_metadata": {
            "hydra": asdict(embeddings.hydra_result.metadata),
            "mds": asdict(embeddings.mds_result.metadata),
        },
        "coordinates": {
            condition.condition_id: {
                int(node): tuple(point)
                for node, point in condition.coordinates.items()
            }
            for condition in all_conditions
        },
        "tolerances": {
            condition.condition_id: {
                "euclidean": euclidean_routing_tolerance(condition),
                "poincare": (
                    poincare_routing_tolerance(condition)
                    if (
                        condition.condition_id != MDS_NATIVE_ARTIFACT
                        and condition.poincare_routable
                    )
                    else None
                ),
            }
            for condition in all_conditions
        },
        "pairs": [list(pair) for pair in pairs],
        "pair_records": pair_records,
        "pair_sampling_identity": {
            "schema": PAIR_IDENTITY_SCHEMA,
            "graph_id": graph_id,
            "pair_seed": pair_seed,
            "pair_count": pair_count,
            "shared_across_all_coordinate_and_routing_conditions": True,
            "data_generation_hash": DATA_GENERATION_HASH,
        },
        "dijkstra_execution_count": len(dijkstra_records),
        "dijkstra_records": dijkstra_records,
        "route_records": route_records,
        "routes_per_pair": ROUTES_PER_PAIR,
        "route_identity_schema": ROUTE_IDENTITY_SCHEMA,
        "euclidean_scale_reuse": euclidean_reuse,
        "routing_execution_counts": {
            "dijkstra": pair_count,
            "euclidean_state_machine": 2 * pair_count,
            "poincare_ordinary_state_machine": 9 * pair_count,
            "poincare_repair_wrappers": 9 * pair_count,
            "scientific_route_records": ROUTES_PER_PAIR * pair_count,
            "euclidean_records_reused_after_scale_proof": 8 * pair_count,
        },
        "graph_level_rows": graph_rows,
        "graph_level_interactions": interactions,
        "high_precision_sentinel": {
            "selection_is_outcome_independent": True,
            "selection_mode": (
                "prespecified_full_pair_oracle_graph"
                if audit_all_pairs
                else "domain_separated_random_pair_sentinel"
            ),
            "pair_indices": sorted(sentinel_indices),
            "route_decisions_checked": sentinel_checks,
            "disagreements": sentinel_disagreements,
            "float64_altered_decisions": sentinel_disagreements > 0,
        },
        "poincare_radius_sensitivity": poincare_sensitivity,
        "gauge_and_centering_diagnostics": {
            "hydra_centering": hydra_gauge_diagnostics,
            "mds_radius_transformation": poincare_sensitivity["mds"],
            "gauge_selected_using_routing_outcomes": False,
        },
        "timing": {
            "setup_seconds": setup_seconds,
            "route_execution_seconds": route_seconds,
            "total_seconds": setup_seconds + route_seconds,
            "timing_is_scientific_outcome": False,
        },
    }
    _require_finite_result(result)
    return result


def _graph_and_pair_diagnostics(
    graph: nx.Graph,
    *,
    shortest_paths: object,
    model: str,
    n: int,
    m: int,
    generation_metadata: Mapping[str, object],
    ordered_pair_distances: Sequence[int],
) -> dict[str, object]:
    metrics = calculate_network_metrics(
        graph,
        shortest_paths=shortest_paths,
    )
    attempts = int(generation_metadata.get("generation_attempt_count", 1))
    attempt_seeds = tuple(
        int(value)
        for value in generation_metadata.get("generation_attempt_seeds", ())
    )
    accepted_seed = generation_metadata.get("generation_attempt_seed")
    rejection_seeds = (
        attempt_seeds[:-1] if model == "erdos_renyi" else ()
    )
    distances = list(ordered_pair_distances)

    def quantile(fraction: float) -> float:
        position = fraction * (len(distances) - 1)
        lower = int(position)
        upper = min(lower + 1, len(distances) - 1)
        weight = position - lower
        return distances[lower] * (1.0 - weight) + distances[upper] * weight

    realised_average = float(metrics["average_degree"])
    target = 2.0 * m * (n - m) / n
    return {
        "graph_model": model,
        "n": n,
        "m": m,
        "er_nominal_p": (
            2.0 * m * (n - m) / (n * (n - 1))
            if model == "erdos_renyi"
            else None
        ),
        "ba_exact_expected_average_degree": (
            2.0 * m * (n - m) / n
            if model == "barabasi_albert"
            else None
        ),
        "er_generation_attempts": (
            attempts if model == "erdos_renyi" else None
        ),
        "accepted_seed": accepted_seed,
        "rejection_seeds": rejection_seeds,
        "connectivity_conditioning": (
            "G(n,p)_conditioned_on_connectivity"
            if model == "erdos_renyi"
            else "connected_by_construction"
        ),
        "connected_er_observed_acceptance_rate": (
            1.0 / attempts if model == "erdos_renyi" else None
        ),
        "realised_edge_count": graph.number_of_edges(),
        "realised_average_degree": realised_average,
        "target_average_degree": target,
        "average_degree_deviation_from_target": realised_average - target,
        "network_metrics": metrics,
        "pair_sampling": {
            "ordered": True,
            "unique": True,
            "source_destination_excluded": True,
            "reverse_directions_distinct": True,
            "same_pairs_all_conditions_and_methods": True,
            "outcome_dependent_resampling": False,
            "pair_count": len(distances),
            "shortest_path_minimum": min(distances),
            "shortest_path_q1": quantile(0.25),
            "shortest_path_median": quantile(0.5),
            "shortest_path_mean": sum(distances) / len(distances),
            "shortest_path_q3": quantile(0.75),
            "shortest_path_maximum": max(distances),
            "shortest_path_counts": {
                str(distance): distances.count(distance)
                for distance in sorted(set(distances))
            },
        },
    }


@scientific_operation_boundary("scheduled_graph_execution")
def execute_scheduled_graph(
    spec: GraphSpec,
    *,
    pair_count: int,
) -> dict[str, object]:
    generated, seeds = generate_iteration2_graph(spec)
    return execute_iteration2_graph(
        generated.graph,
        graph_id=spec.graph_id,
        model=spec.model,
        n=spec.n,
        m=spec.m,
        replicate_index=spec.replicate_index,
        pair_seed=seeds.pairs,
        pair_count=pair_count,
        graph_seed=seeds.graph,
        embedding_provenance_seed=seeds.embedding_provenance,
        routing_priority_seed=seeds.routing_priority,
        validation_sentinel_seed=seeds.validation_sentinel,
        er_attempt_seed_schedule=seeds.er_attempts,
        generation_metadata=generated.metadata,
        audit_all_pairs=is_full_oracle_graph(spec),
    )


def _require_finite_result(value: object, path: str = "result") -> None:
    if isinstance(value, float) and not isfinite(value):
        raise RuntimeError(f"{path} contains a non-finite float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_result(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_result(item, f"{path}[{index}]")
