"""Excluded development pilot for the approved Hydra/MDS embedding design."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from time import perf_counter_ns
from types import MappingProxyType
import sys

import networkx as nx

from embedding import (
    calculate_embedding_distortion,
    prepare_embedding_input,
)
from experiment_config import (
    BARABASI_ALBERT,
    DEVELOPMENT_CONFIG,
    ERDOS_RENYI,
    FEASIBILITY_PILOT_SEEDS,
    FULL_EXPERIMENT_CONFIG,
    ExperimentConfig,
    audit_feasibility_pilot_seed_collisions,
    make_degree_matched_parameters,
)
from graph_generation import (
    GraphMetadata,
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
)
from hydra_embedding import HydraEmbeddingResult, embed_hydra
from mds_embedding import (
    MDSConditionResult,
    ClassicalMDSResult,
    classical_mds,
    scale_equivariant_euclidean_routing_tolerance,
    transform_mds_to_radius,
)
from network_metrics import (
    calculate_network_metrics,
    prepare_all_pairs_shortest_paths,
)
from poincare_distance import euclidean_distance, poincare_distance
from routing import (
    EUCLIDEAN_GREEDY_METHOD,
    HYPERBOLIC_GREEDY_METHOD,
    REPAIRED_HYPERBOLIC_GREEDY_METHOD,
    RoutingResult,
    dijkstra_benchmark,
    euclidean_greedy_route,
    hyperbolic_greedy_route,
    prepare_routing_coordinates,
    repaired_hyperbolic_greedy_route,
)
from run_development_smoke import sample_ordered_pairs


PILOT_PAIR_COUNT = 5
PILOT_EVIDENCE_LABEL = (
    "EXCLUDED DEVELOPMENT FEASIBILITY EVIDENCE - NOT SCIENTIFIC RESULTS"
)


@dataclass(frozen=True)
class CoordinateCondition:
    embedding_family: str
    coordinate_condition_id: str
    mds_radius: float | None
    coordinates: Mapping[int, Sequence[float]]
    metadata: object


@dataclass(frozen=True)
class ConditionDiagnostics:
    embedding_family: str
    coordinate_condition_id: str
    mds_radius: float | None
    maximum_coordinate_norm: float
    frechet_mean_residual: float | None
    mean_relative_distortion: float
    rmse_relative_distortion: float
    transformation_runtime_ns: int
    euclidean_routing_tolerance: float
    poincare_routing_tolerance: float
    embedding_metadata: object


@dataclass(frozen=True)
class EmbeddingRouteRecord:
    graph_id: str
    graph_family: str
    graph_replicate: int
    pair_id: str
    source: int
    destination: int
    embedding_family: str | None
    coordinate_condition_id: str | None
    mds_radius: float | None
    routing_method: str
    success: bool
    failure_type: str | None
    route_length: int
    dijkstra_length: int
    stretch: float | None
    runtime_ns: int
    repair_attempted: bool
    repair_succeeded: bool
    repair_alternative_existed: bool | None


@dataclass(frozen=True)
class GraphFeasibilityRecord:
    graph_id: str
    graph_family: str
    graph_replicate: int
    n: int
    number_of_edges: int
    graph_seed: int
    pair_seed: int
    er_p: float | None
    ba_m: int | None
    graph_generation_metadata: Mapping[str, str | int | float]
    configuration_fingerprint: str
    embedding_input_fingerprint: str
    hydra_runtime_ns: int
    mds_base_runtime_ns: int
    hydra_metadata: object
    mds_base_metadata: object
    source_destination_pairs: tuple[tuple[int, int], ...]
    pair_ids: tuple[str, ...]
    coordinate_condition_ids: tuple[str, ...]
    condition_diagnostics: tuple[ConditionDiagnostics, ...]
    route_records: tuple[EmbeddingRouteRecord, ...]


@dataclass(frozen=True)
class EmbeddingFeasibilityReport:
    evidence_label: str
    excluded_from_final_experiment: bool
    configuration_fingerprint: str
    pilot_seeds: tuple[int, ...]
    graphs: tuple[GraphFeasibilityRecord, ...]
    workload_projection: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _PilotCase:
    graph_id: str
    graph_family: str
    n: int
    er_p: float | None
    ba_m: int | None
    graph_seed: int
    pair_seed: int

    def __post_init__(self) -> None:
        if self.graph_family == ERDOS_RENYI:
            if (
                self.er_p is None
                or not isfinite(self.er_p)
                or not 0.0 < self.er_p < 1.0
                or self.ba_m is not None
            ):
                raise ValueError("ER pilot cases require only a valid er_p")
        elif self.graph_family == BARABASI_ALBERT:
            if (
                self.ba_m is None
                or isinstance(self.ba_m, bool)
                or not isinstance(self.ba_m, int)
                or not 1 <= self.ba_m < self.n
                or self.er_p is not None
            ):
                raise ValueError("BA pilot cases require only a valid ba_m")
        else:
            raise ValueError("pilot graph family is unsupported")


def _timed_call(function, /, *args, **kwargs):
    start = perf_counter_ns()
    result = function(*args, **kwargs)
    return result, perf_counter_ns() - start


def _timed_mds_sensitivity_conditions(
    base_embedding: ClassicalMDSResult,
    *,
    radii: Sequence[float],
    tolerance: float,
) -> tuple[tuple[MDSConditionResult, ...], tuple[int, ...]]:
    """Transform and time every nested MDS radius separately."""

    measured = tuple(
        _timed_call(
            transform_mds_to_radius,
            base_embedding,
            radius,
            tolerance=tolerance,
        )
        for radius in radii
    )
    return (
        tuple(condition for condition, _ in measured),
        tuple(runtime for _, runtime in measured),
    )


def _validated_configuration_fingerprint(
    configuration_fingerprint: str,
) -> str:
    if (
        not isinstance(configuration_fingerprint, str)
        or len(configuration_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in configuration_fingerprint
        )
    ):
        raise ValueError(
            "configuration_fingerprint must be a lowercase SHA-256 digest"
        )
    return configuration_fingerprint


def _validated_generation_metadata(
    metadata: Mapping[str, str | int | float] | None,
    *,
    graph_family: str,
    n: int,
    graph_replicate: int,
    graph_seed: int,
    er_p: float | None,
    ba_m: int | None,
) -> Mapping[str, str | int | float]:
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise ValueError("graph_generation_metadata must be a mapping or None")
    copied = dict(metadata)
    required = {
        "graph_model": graph_family,
        "n": n,
        "replicate_index": graph_replicate,
        "graph_seed": graph_seed,
    }
    for key, expected in required.items():
        if copied.get(key) != expected:
            raise ValueError(
                f"graph_generation_metadata[{key!r}] does not match the graph record"
            )
    if er_p is not None and copied.get("p") != er_p:
        raise ValueError("graph_generation_metadata['p'] does not match er_p")
    if ba_m is not None and copied.get("m") != ba_m:
        raise ValueError("graph_generation_metadata['m'] does not match ba_m")
    if any(
        not isinstance(key, str)
        or not isinstance(value, (str, int, float))
        or isinstance(value, bool)
        for key, value in copied.items()
    ):
        raise ValueError(
            "graph_generation_metadata must contain string keys and scalar values"
        )
    return MappingProxyType(dict(sorted(copied.items())))


def _validate_route(graph: nx.Graph, result: RoutingResult) -> None:
    if not result.walk or result.walk[0] != result.source:
        raise RuntimeError("route walk does not start at its source")
    if result.route_length != len(result.walk) - 1:
        raise RuntimeError("route length does not equal len(walk)-1")
    if any(
        not graph.has_edge(left, right)
        for left, right in zip(result.walk, result.walk[1:])
    ):
        raise RuntimeError("route walk contains a non-edge")
    if result.success and result.walk[-1] != result.destination:
        raise RuntimeError("successful route does not end at its destination")


def _route_record(
    *,
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    graph_replicate: int,
    pair_id: str,
    embedding_family: str | None,
    coordinate_condition_id: str | None,
    mds_radius: float | None,
    result: RoutingResult,
    dijkstra_length: int,
    runtime_ns: int,
) -> EmbeddingRouteRecord:
    _validate_route(graph, result)
    if result.success and result.route_length < dijkstra_length:
        raise RuntimeError("successful route is shorter than Dijkstra")
    stretch = (
        result.route_length / dijkstra_length
        if result.success and embedding_family is not None
        else None
    )
    return EmbeddingRouteRecord(
        graph_id=graph_id,
        graph_family=graph_family,
        graph_replicate=graph_replicate,
        pair_id=pair_id,
        source=int(result.source),
        destination=int(result.destination),
        embedding_family=embedding_family,
        coordinate_condition_id=coordinate_condition_id,
        mds_radius=mds_radius,
        routing_method=result.method,
        success=result.success,
        failure_type=result.failure_type,
        route_length=result.route_length,
        dijkstra_length=dijkstra_length,
        stretch=stretch,
        runtime_ns=runtime_ns,
        repair_attempted=result.repair_attempted,
        repair_succeeded=result.repair_succeeded,
        repair_alternative_existed=result.repair_alternative_existed,
    )


def _approved_conditions(
    hydra: HydraEmbeddingResult,
    mds_conditions: tuple[MDSConditionResult, ...],
) -> tuple[CoordinateCondition, ...]:
    conditions = (
        CoordinateCondition(
            embedding_family=hydra.metadata.embedding_family,
            coordinate_condition_id=hydra.metadata.coordinate_condition_id,
            mds_radius=None,
            coordinates=hydra.coordinates,
            metadata=hydra.metadata,
        ),
        *(
            CoordinateCondition(
                embedding_family=condition.metadata.embedding_family,
                coordinate_condition_id=condition.metadata.coordinate_condition_id,
                mds_radius=condition.metadata.requested_maximum_radius,
                coordinates=condition.coordinates,
                metadata=condition.metadata,
            )
            for condition in mds_conditions
        ),
    )
    if len(conditions) != 5:
        raise RuntimeError("approved design must contain Hydra plus four MDS conditions")
    return conditions


def run_approved_embedding_pipeline(
    graph: nx.Graph,
    *,
    graph_id: str,
    graph_family: str,
    graph_replicate: int,
    graph_seed: int,
    pair_seed: int,
    pair_count: int,
    config: ExperimentConfig = DEVELOPMENT_CONFIG,
    er_p: float | None = None,
    ba_m: int | None = None,
    graph_generation_metadata: GraphMetadata | None = None,
    configuration_fingerprint: str | None = None,
) -> GraphFeasibilityRecord:
    """Run all approved coordinate conditions once on one in-memory graph."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    if not isinstance(graph_id, str) or not graph_id:
        raise ValueError("graph_id must be a non-empty string")
    if graph_family not in (ERDOS_RENYI, BARABASI_ALBERT):
        raise ValueError("graph_family is unsupported")
    if er_p is not None:
        if (
            graph_family != ERDOS_RENYI
            or isinstance(er_p, bool)
            or not isinstance(er_p, (int, float))
            or not isfinite(er_p)
            or not 0.0 < float(er_p) < 1.0
            or ba_m is not None
        ):
            raise ValueError("er_p is valid only for an ER graph record")
        er_p = float(er_p)
    if ba_m is not None:
        if (
            graph_family != BARABASI_ALBERT
            or isinstance(ba_m, bool)
            or not isinstance(ba_m, int)
            or not 1 <= ba_m < graph.number_of_nodes()
            or er_p is not None
        ):
            raise ValueError("ba_m is valid only for a BA graph record")
    effective_configuration_fingerprint = (
        config.configuration_fingerprint
        if configuration_fingerprint is None
        else _validated_configuration_fingerprint(configuration_fingerprint)
    )
    nodes_before = deepcopy(tuple(graph.nodes(data=True)))
    edges_before = deepcopy(tuple(graph.edges(data=True)))

    shortest_paths = prepare_all_pairs_shortest_paths(graph)
    embedding_input = prepare_embedding_input(
        graph,
        shortest_paths,
        configuration_fingerprint=effective_configuration_fingerprint,
        tolerance=config.numerical_tolerance,
    )
    design = config.approved_embedding_design
    hydra, hydra_runtime = _timed_call(
        embed_hydra,
        embedding_input,
        dimension=design.hydra_dimension,
        kappa=design.hydra_kappa,
        centering_tolerance=design.hydra_centering_tolerance,
        centering_max_iterations=design.hydra_centering_max_iterations,
        eigenvalue_tolerance=design.hydra_eigenvalue_tolerance,
        pairwise_isometry_tolerance=design.hydra_isometry_tolerance,
        boundary_roundoff_tolerance=design.hydra_boundary_roundoff_tolerance,
    )
    mds_base, mds_base_runtime = _timed_call(
        classical_mds,
        embedding_input,
        dimension=design.mds_dimension,
        eigenvalue_relative_tolerance=design.mds_eigenvalue_relative_tolerance,
        centroid_tolerance=design.mds_centroid_tolerance,
    )
    (
        mds_conditions,
        mds_transformation_runtimes,
    ) = _timed_mds_sensitivity_conditions(
        mds_base,
        radii=design.mds_maximum_radii,
        tolerance=config.numerical_tolerance,
    )
    transformation_runtime_by_condition = {
        condition.metadata.coordinate_condition_id: runtime
        for condition, runtime in zip(
            mds_conditions,
            mds_transformation_runtimes,
            strict=True,
        )
    }
    conditions = _approved_conditions(hydra, mds_conditions)
    if tuple(condition.coordinate_condition_id for condition in conditions) != (
        design.coordinate_condition_ids
    ):
        raise RuntimeError("implementation condition IDs do not match configuration")

    metrics = calculate_network_metrics(graph, shortest_paths=shortest_paths)
    diagnostics: list[ConditionDiagnostics] = []
    prepared_contexts = {}
    for condition in conditions:
        distortion = calculate_embedding_distortion(
            graph,
            condition.coordinates,
            tolerance=config.numerical_tolerance,
            shortest_paths=shortest_paths,
        )
        euclidean_context = prepare_routing_coordinates(
            graph,
            condition.coordinates,
            euclidean_distance,
            metric_name="euclidean",
        )
        poincare_context = prepare_routing_coordinates(
            graph,
            condition.coordinates,
            poincare_distance,
            metric_name="poincare",
        )
        if tuple(euclidean_context.items()) != tuple(poincare_context.items()):
            raise RuntimeError("routing metrics received different coordinates")
        euclidean_routing_tolerance = (
            config.numerical_tolerance
            if condition.mds_radius is None
            else scale_equivariant_euclidean_routing_tolerance(
                config.numerical_tolerance,
                condition.mds_radius,
            )
        )
        poincare_routing_tolerance = config.numerical_tolerance
        prepared_contexts[condition.coordinate_condition_id] = (
            euclidean_context,
            poincare_context,
            euclidean_routing_tolerance,
            poincare_routing_tolerance,
        )
        maximum_norm = max(
            float(
                (
                    float(point[0]) * float(point[0])
                    + float(point[1]) * float(point[1])
                )
                ** 0.5
            )
            for point in condition.coordinates.values()
        )
        diagnostics.append(
            ConditionDiagnostics(
                embedding_family=condition.embedding_family,
                coordinate_condition_id=condition.coordinate_condition_id,
                mds_radius=condition.mds_radius,
                maximum_coordinate_norm=maximum_norm,
                frechet_mean_residual=(
                    hydra.metadata.final_frechet_mean_residual
                    if condition.mds_radius is None
                    else None
                ),
                mean_relative_distortion=distortion.mean_relative_distortion,
                rmse_relative_distortion=distortion.rmse_relative_distortion,
                transformation_runtime_ns=(
                    0
                    if condition.mds_radius is None
                    else transformation_runtime_by_condition[
                        condition.coordinate_condition_id
                    ]
                ),
                euclidean_routing_tolerance=euclidean_routing_tolerance,
                poincare_routing_tolerance=poincare_routing_tolerance,
                embedding_metadata=condition.metadata,
            )
        )

    pairs = sample_ordered_pairs(graph.nodes, pair_count, pair_seed)
    pair_ids = tuple(
        f"{graph_id}:pair:{index:04d}" for index in range(len(pairs))
    )
    records: list[EmbeddingRouteRecord] = []
    for pair_id, (source, destination) in zip(pair_ids, pairs, strict=True):
        benchmark, benchmark_runtime = _timed_call(
            dijkstra_benchmark,
            graph,
            source,
            destination,
        )
        records.append(
            _route_record(
                graph=graph,
                graph_id=graph_id,
                graph_family=graph_family,
                graph_replicate=graph_replicate,
                pair_id=pair_id,
                embedding_family=None,
                coordinate_condition_id=None,
                mds_radius=None,
                result=benchmark,
                dijkstra_length=benchmark.route_length,
                runtime_ns=benchmark_runtime,
            )
        )
        mds_euclidean_reference: RoutingResult | None = None
        for condition in conditions:
            (
                euclidean_context,
                poincare_context,
                euclidean_tolerance,
                poincare_tolerance,
            ) = prepared_contexts[condition.coordinate_condition_id]
            for method, context, routing_tolerance in (
                (
                    euclidean_greedy_route,
                    euclidean_context,
                    euclidean_tolerance,
                ),
                (
                    hyperbolic_greedy_route,
                    poincare_context,
                    poincare_tolerance,
                ),
                (
                    repaired_hyperbolic_greedy_route,
                    poincare_context,
                    poincare_tolerance,
                ),
            ):
                result, runtime = _timed_call(
                    method,
                    graph,
                    context,
                    source,
                    destination,
                    tolerance=routing_tolerance,
                )
                if (
                    method is euclidean_greedy_route
                    and condition.mds_radius is not None
                ):
                    if mds_euclidean_reference is None:
                        mds_euclidean_reference = result
                    elif result != mds_euclidean_reference:
                        raise RuntimeError(
                            "uniform MDS radius transformations changed an "
                            "Euclidean greedy-routing decision"
                        )
                records.append(
                    _route_record(
                        graph=graph,
                        graph_id=graph_id,
                        graph_family=graph_family,
                        graph_replicate=graph_replicate,
                        pair_id=pair_id,
                        embedding_family=condition.embedding_family,
                        coordinate_condition_id=(
                            condition.coordinate_condition_id
                        ),
                        mds_radius=condition.mds_radius,
                        result=result,
                        dijkstra_length=benchmark.route_length,
                        runtime_ns=runtime,
                    )
                )

    if len(records) != len(pairs) * 16:
        raise RuntimeError("each pair must have one Dijkstra and 15 greedy records")
    if deepcopy(tuple(graph.nodes(data=True))) != nodes_before or deepcopy(
        tuple(graph.edges(data=True))
    ) != edges_before:
        raise RuntimeError("embedding feasibility pipeline mutated the graph")
    validated_generation_metadata = _validated_generation_metadata(
        graph_generation_metadata,
        graph_family=graph_family,
        n=metrics["number_of_vertices"],
        graph_replicate=graph_replicate,
        graph_seed=graph_seed,
        er_p=er_p,
        ba_m=ba_m,
    )
    return GraphFeasibilityRecord(
        graph_id=graph_id,
        graph_family=graph_family,
        graph_replicate=graph_replicate,
        n=metrics["number_of_vertices"],
        number_of_edges=metrics["number_of_edges"],
        graph_seed=graph_seed,
        pair_seed=pair_seed,
        er_p=er_p,
        ba_m=ba_m,
        graph_generation_metadata=validated_generation_metadata,
        configuration_fingerprint=effective_configuration_fingerprint,
        embedding_input_fingerprint=embedding_input.input_fingerprint,
        hydra_runtime_ns=hydra_runtime,
        mds_base_runtime_ns=mds_base_runtime,
        hydra_metadata=hydra.metadata,
        mds_base_metadata=mds_base.metadata,
        source_destination_pairs=pairs,
        pair_ids=pair_ids,
        coordinate_condition_ids=tuple(
            condition.coordinate_condition_id for condition in conditions
        ),
        condition_diagnostics=tuple(diagnostics),
        route_records=tuple(records),
    )


def _pilot_cases() -> tuple[_PilotCase, ...]:
    er_setting = make_degree_matched_parameters(
        n=30,
        ba_m=4,
        label="excluded_pilot_n30_m4",
        provisional=False,
    )
    return (
        _PilotCase(
            graph_id="excluded_pilot_er_n30_r0",
            graph_family=ERDOS_RENYI,
            n=30,
            er_p=er_setting.er_p,
            ba_m=None,
            graph_seed=FEASIBILITY_PILOT_SEEDS[0],
            pair_seed=FEASIBILITY_PILOT_SEEDS[1],
        ),
        _PilotCase(
            graph_id="excluded_pilot_ba_n30_r0",
            graph_family=BARABASI_ALBERT,
            n=30,
            er_p=None,
            ba_m=4,
            graph_seed=FEASIBILITY_PILOT_SEEDS[2],
            pair_seed=FEASIBILITY_PILOT_SEEDS[3],
        ),
        _PilotCase(
            graph_id="excluded_pilot_ba_n100_r0",
            graph_family=BARABASI_ALBERT,
            n=100,
            er_p=None,
            ba_m=4,
            graph_seed=FEASIBILITY_PILOT_SEEDS[4],
            pair_seed=FEASIBILITY_PILOT_SEEDS[5],
        ),
    )


def _pilot_configuration_fingerprint() -> str:
    manifest = {
        "schema": "excluded_embedding_feasibility_pilot_v1",
        "evidence_label": PILOT_EVIDENCE_LABEL,
        "excluded_from_final_experiment": True,
        "pair_count_per_graph": PILOT_PAIR_COUNT,
        "runtime_method_configuration_fingerprint": (
            DEVELOPMENT_CONFIG.configuration_fingerprint
        ),
        "cases": [
            {
                "graph_id": case.graph_id,
                "graph_family": case.graph_family,
                "n": case.n,
                "er_p_hex": (
                    None if case.er_p is None else float(case.er_p).hex()
                ),
                "ba_m": case.ba_m,
                "graph_seed": case.graph_seed,
                "pair_seed": case.pair_seed,
                "maximum_generation_attempts": 1,
            }
            for case in _pilot_cases()
        ],
    }
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(canonical).hexdigest()


PILOT_CONFIGURATION_FINGERPRINT = _pilot_configuration_fingerprint()


def run_embedding_feasibility_pilot() -> EmbeddingFeasibilityReport:
    """Run three excluded graphs in memory and return descriptive diagnostics."""

    if audit_feasibility_pilot_seed_collisions(FULL_EXPERIMENT_CONFIG):
        raise RuntimeError("pilot seeds collide with final experiment seed uses")
    graph_records: list[GraphFeasibilityRecord] = []
    for case in _pilot_cases():
        if case.graph_family == ERDOS_RENYI:
            if case.er_p is None:
                raise RuntimeError("ER pilot case is missing er_p")
            generated = generate_connected_erdos_renyi(
                n=case.n,
                p=case.er_p,
                graph_seed=case.graph_seed,
                replicate_index=0,
                max_attempts=1,
                attempt_seeds=(case.graph_seed,),
            )
        else:
            if case.ba_m is None:
                raise RuntimeError("BA pilot case is missing ba_m")
            generated = generate_connected_barabasi_albert(
                n=case.n,
                m=case.ba_m,
                graph_seed=case.graph_seed,
                replicate_index=0,
            )
        graph_records.append(
            run_approved_embedding_pipeline(
                generated.graph,
                graph_id=case.graph_id,
                graph_family=case.graph_family,
                graph_replicate=0,
                graph_seed=case.graph_seed,
                pair_seed=case.pair_seed,
                pair_count=PILOT_PAIR_COUNT,
                er_p=case.er_p,
                ba_m=case.ba_m,
                graph_generation_metadata=generated.metadata,
                configuration_fingerprint=PILOT_CONFIGURATION_FINGERPRINT,
            )
        )

    workload = FULL_EXPERIMENT_CONFIG.workload_estimate
    projection_keys = (
        "independent_graph_replicates",
        "hydra_embedding_runs",
        "mds_base_embedding_runs",
        "mds_nested_radius_transformations",
        "sampled_ordered_pairs",
        "dijkstra_routing_runs",
        "routing_method_runs",
    )
    return EmbeddingFeasibilityReport(
        evidence_label=PILOT_EVIDENCE_LABEL,
        excluded_from_final_experiment=True,
        configuration_fingerprint=PILOT_CONFIGURATION_FINGERPRINT,
        pilot_seeds=FEASIBILITY_PILOT_SEEDS,
        graphs=tuple(graph_records),
        workload_projection=tuple(
            (key, workload[key]) for key in projection_keys
        ),
    )


def print_feasibility_summary(report: EmbeddingFeasibilityReport) -> None:
    print(
        f"{report.evidence_label} "
        f"(pilot_fingerprint={report.configuration_fingerprint})"
    )
    for graph in report.graphs:
        parameter = (
            f"p={graph.er_p:.17g}"
            if graph.er_p is not None
            else f"m={graph.ba_m}"
        )
        print(
            f"{graph.graph_id}: {parameter}, n={graph.n}, "
            f"edges={graph.number_of_edges}, "
            f"hydra_ms={graph.hydra_runtime_ns / 1e6:.3f}, "
            f"mds_base_ms={graph.mds_base_runtime_ns / 1e6:.3f}"
        )
        for diagnostic in graph.condition_diagnostics:
            condition_records = [
                record
                for record in graph.route_records
                if record.coordinate_condition_id
                == diagnostic.coordinate_condition_id
            ]
            failure_counts = {
                method: sum(
                    record.routing_method == method and not record.success
                    for record in condition_records
                )
                for method in (
                    EUCLIDEAN_GREEDY_METHOD,
                    HYPERBOLIC_GREEDY_METHOD,
                    REPAIRED_HYPERBOLIC_GREEDY_METHOD,
                )
            }
            repair_records = [
                record
                for record in condition_records
                if record.routing_method == REPAIRED_HYPERBOLIC_GREEDY_METHOD
            ]
            repair_opportunities = sum(
                record.repair_attempted for record in repair_records
            )
            repair_successes = sum(
                record.repair_succeeded for record in repair_records
            )
            print(
                f"  {diagnostic.coordinate_condition_id}: "
                f"valid_max_norm={diagnostic.maximum_coordinate_norm:.9f}, "
                f"frechet_residual={diagnostic.frechet_mean_residual}, "
                f"mean_distortion={diagnostic.mean_relative_distortion:.6f}, "
                f"rmse_distortion={diagnostic.rmse_relative_distortion:.6f}, "
                "routing_failures="
                f"euclidean:{failure_counts[EUCLIDEAN_GREEDY_METHOD]},"
                f"poincare:{failure_counts[HYPERBOLIC_GREEDY_METHOD]},"
                "repaired:"
                f"{failure_counts[REPAIRED_HYPERBOLIC_GREEDY_METHOD]}, "
                f"repair_opportunities={repair_opportunities}, "
                f"repair_successes={repair_successes}"
            )
    print(
        "Approximate full-workload operation counts "
        "(not a linear runtime extrapolation):"
    )
    for key, value in report.workload_projection:
        print(f"  {key}={value}")


def main() -> int:
    try:
        report = run_embedding_feasibility_pilot()
        print_feasibility_summary(report)
    except Exception as exc:
        print(f"Excluded embedding-feasibility pilot failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
