"""Small deterministic Stage 11 integration experiment.

This runner validates the development pipeline only. It does not execute the
full configuration, aggregate results, create plots, or make scientific claims.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
from random import Random
import sys
from typing import Any

import networkx as nx
import numpy as np

from embedding import (
    calculate_embedding_distortion,
    development_embedding_metadata,
    embed_graph_in_poincare_disk,
)
from experiment_config import (
    BARABASI_ALBERT,
    DEVELOPMENT_CONFIG,
    ERDOS_RENYI,
    MAX_SEED,
    ExperimentConfig,
)
from graph_generation import generate_graph
from network_metrics import (
    NetworkMetrics,
    calculate_network_metrics,
    prepare_all_pairs_shortest_paths,
)
from poincare_distance import euclidean_distance, poincare_distance
from routing import (
    DIJKSTRA_METHOD,
    PreparedRoutingCoordinates,
    RoutingResult,
    dijkstra_benchmark,
    euclidean_greedy_route,
    hyperbolic_greedy_route,
    prepare_routing_coordinates,
    repaired_hyperbolic_greedy_route,
)


SMOKE_PAIR_LIMIT = 5
SMOKE_MODELS = (ERDOS_RENYI, BARABASI_ALBERT)
SMOKE_RESULT_SCHEMA_VERSION = 2
IMPLEMENTATION_SOURCE_FILES = (
    "embedding.py",
    "experiment_config.py",
    "graph_generation.py",
    "network_metrics.py",
    "poincare_distance.py",
    "routing.py",
    "run_development_smoke.py",
)
Coordinate = Sequence[float]
MetadataScalar = str | int | float | bool


@dataclass(frozen=True)
class MethodSmokeRecord:
    graph_model: str
    source: int
    destination: int
    method: str
    success: bool
    walk: tuple[int, ...]
    route_length: int
    failure_type: str | None
    stretch: float | None
    forwarding_decisions: int
    repair_attempted: bool
    repair_succeeded: bool
    repair_attempt_count: int
    initial_failure_type: str | None
    final_failure_type: str | None
    repair_alternative_existed: bool | None


@dataclass(frozen=True)
class GraphSmokeRecord:
    graph_model: str
    graph_metadata: tuple[tuple[str, MetadataScalar], ...]
    embedding_metadata: tuple[tuple[str, MetadataScalar], ...]
    pair_sampling_metadata: tuple[tuple[str, MetadataScalar], ...]
    network_metrics: tuple[tuple[str, int | float], ...]
    embedding_distortion: tuple[tuple[str, int | float], ...]
    coordinates: tuple[tuple[int, tuple[float, float]], ...]
    source_destination_pairs: tuple[tuple[int, int], ...]
    method_records: tuple[MethodSmokeRecord, ...]


@dataclass(frozen=True)
class DevelopmentSmokeResult:
    result_schema_version: int
    configuration_name: str
    configuration_fingerprint: str
    implementation_source_fingerprint: str
    configuration_json: str
    runtime_metadata: tuple[tuple[str, str], ...]
    graphs: tuple[GraphSmokeRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-native representation of the complete smoke run."""

        payload = json.loads(json.dumps(asdict(self), allow_nan=False))
        payload["configuration"] = json.loads(payload.pop("configuration_json"))
        return payload

    def to_json(self) -> str:
        """Serialize the complete result canonically for reproducibility checks."""

        return json.dumps(
            self.as_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")


def sample_ordered_pairs(
    nodes: Iterable[int], pair_count: int, seed: int
) -> tuple[tuple[int, int], ...]:
    """Sample unique ordered pairs from a stable node order."""

    ordered_nodes = tuple(nodes)
    if any(isinstance(node, bool) or not isinstance(node, int) for node in ordered_nodes):
        raise ValueError("nodes must be integer IDs")
    if len(set(ordered_nodes)) != len(ordered_nodes):
        raise ValueError("nodes must be unique")
    ordered_nodes = tuple(sorted(ordered_nodes))
    if len(ordered_nodes) < 2:
        raise ValueError("at least two nodes are required")
    if isinstance(pair_count, bool) or not isinstance(pair_count, int):
        raise ValueError("pair_count must be an integer")
    if pair_count < 0:
        raise ValueError("pair_count must be non-negative")
    _validate_seed(seed)

    node_count = len(ordered_nodes)
    available_pair_count = node_count * (node_count - 1)
    if pair_count > available_pair_count:
        raise ValueError(
            f"pair_count exceeds the {available_pair_count} available ordered pairs"
        )

    sampled_indices = Random(seed).sample(range(available_pair_count), pair_count)
    pairs: list[tuple[int, int]] = []
    for pair_index in sampled_indices:
        source_index, destination_offset = divmod(pair_index, node_count - 1)
        destination_index = (
            destination_offset
            if destination_offset < source_index
            else destination_offset + 1
        )
        pairs.append(
            (ordered_nodes[source_index], ordered_nodes[destination_index])
        )

    result = tuple(pairs)
    if len(set(result)) != len(result) or any(source == target for source, target in result):
        raise RuntimeError("ordered-pair sampler violated uniqueness invariants")
    return result


def _validate_walk(graph: nx.Graph, result: RoutingResult) -> None:
    if not result.walk or result.walk[0] != result.source:
        raise RuntimeError("recorded walk does not begin at its source")
    if result.route_length != len(result.walk) - 1:
        raise RuntimeError("route length does not equal len(walk) - 1")
    for left, right in zip(result.walk, result.walk[1:]):
        if not graph.has_edge(left, right):
            raise RuntimeError(f"recorded walk contains non-edge ({left}, {right})")
    if result.success and result.walk[-1] != result.destination:
        raise RuntimeError("successful route does not end at its destination")
    if result.repair_attempt_count not in (0, 1):
        raise RuntimeError("routing attempted more than one repair")


def _method_record(
    *,
    graph_model: str,
    graph: nx.Graph,
    result: RoutingResult,
    dijkstra_length: int,
) -> MethodSmokeRecord:
    _validate_walk(graph, result)
    if result.success and result.route_length < dijkstra_length:
        raise RuntimeError("a successful route is shorter than the Dijkstra benchmark")

    if result.method == DIJKSTRA_METHOD or not result.success:
        stretch = None
    else:
        if dijkstra_length <= 0:
            raise RuntimeError("distinct source-destination pairs require positive distance")
        stretch = result.route_length / dijkstra_length

    if not result.success and stretch is not None:
        raise RuntimeError("a failed route cannot have stretch")

    return MethodSmokeRecord(
        graph_model=graph_model,
        source=result.source,
        destination=result.destination,
        method=result.method,
        success=result.success,
        walk=result.walk,
        route_length=result.route_length,
        failure_type=result.failure_type,
        stretch=stretch,
        forwarding_decisions=result.forwarding_decisions,
        repair_attempted=result.repair_attempted,
        repair_succeeded=result.repair_succeeded,
        repair_attempt_count=result.repair_attempt_count,
        initial_failure_type=result.initial_failure_type,
        final_failure_type=result.final_failure_type,
        repair_alternative_existed=result.repair_alternative_existed,
    )


def run_methods_for_pair(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate],
    source: int,
    destination: int,
    *,
    graph_model: str = "fixture",
    tolerance: float,
    prepared_euclidean: PreparedRoutingCoordinates | None = None,
    prepared_poincare: PreparedRoutingCoordinates | None = None,
) -> tuple[MethodSmokeRecord, ...]:
    """Run the benchmark and three routing variants on one shared input."""

    benchmark = dijkstra_benchmark(graph, source, destination)
    euclidean_coordinates = (
        coordinates if prepared_euclidean is None else prepared_euclidean
    )
    poincare_coordinates = (
        coordinates if prepared_poincare is None else prepared_poincare
    )

    raw_results = (
        benchmark,
        euclidean_greedy_route(
            graph,
            euclidean_coordinates,
            source,
            destination,
            tolerance=tolerance,
        ),
        hyperbolic_greedy_route(
            graph,
            poincare_coordinates,
            source,
            destination,
            tolerance=tolerance,
        ),
        repaired_hyperbolic_greedy_route(
            graph,
            poincare_coordinates,
            source,
            destination,
            tolerance=tolerance,
        ),
    )
    records = tuple(
        _method_record(
            graph_model=graph_model,
            graph=graph,
            result=result,
            dijkstra_length=benchmark.route_length,
        )
        for result in raw_results
    )
    if any(
        (record.source, record.destination) != (source, destination)
        for record in records
    ):
        raise RuntimeError("methods did not use an identical source-destination pair")
    return records


def _stable_coordinates(
    coordinates: Mapping[int, Coordinate],
) -> tuple[tuple[int, tuple[float, float]], ...]:
    return tuple(
        (node, (float(coordinates[node][0]), float(coordinates[node][1])))
        for node in sorted(coordinates)
    )


def _stable_items(
    mapping: Mapping[str, MetadataScalar],
) -> tuple[tuple[str, MetadataScalar], ...]:
    return tuple((key, mapping[key]) for key in sorted(mapping))


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _configuration_json(config: ExperimentConfig) -> str:
    """Capture the exact validated configuration using its canonical serializer."""

    serialized = config.to_json()
    if not isinstance(serialized, str):
        raise RuntimeError("ExperimentConfig.to_json() must return a string")
    # Reject non-standard NaN/Infinity tokens and malformed output now, before
    # provenance could be mistaken for a reproducible configuration snapshot.
    json.loads(serialized, parse_constant=_reject_nonstandard_json_constant)
    return serialized


def _embedding_provenance(
    config: ExperimentConfig, embedding_seed: int
) -> tuple[tuple[str, MetadataScalar], ...]:
    metadata: dict[str, MetadataScalar] = dict(development_embedding_metadata())
    if metadata["embedding_id"] != config.embedding_method:
        raise RuntimeError("configured embedding method does not match implementation")
    if metadata["layout_algorithm"] != config.embedding_algorithm:
        raise RuntimeError("configured embedding algorithm does not match implementation")
    metadata.update(
        {
            "embedding_seed": embedding_seed,
            "embedding_radius": config.embedding_radius,
            "embedding_iterations": config.embedding_iterations,
            "unit_disk_boundary_epsilon": config.unit_disk_boundary_epsilon,
            "numerical_tolerance": config.numerical_tolerance,
        }
    )
    return _stable_items(metadata)


def _runtime_metadata() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "networkx_version": nx.__version__,
                "numpy_version": np.__version__,
            }.items()
        )
    )


def implementation_source_fingerprint() -> str:
    """Hash the exact Stage 1-11 implementation used by the smoke run."""

    code_directory = Path(__file__).resolve().parent
    digest = sha256()
    for filename in IMPLEMENTATION_SOURCE_FILES:
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update((code_directory / filename).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_development_smoke(
    config: ExperimentConfig = DEVELOPMENT_CONFIG,
    *,
    pair_limit: int = SMOKE_PAIR_LIMIT,
) -> DevelopmentSmokeResult:
    """Run the fixed, small Stage 11 pipeline entirely in memory."""

    if not isinstance(config, ExperimentConfig) or config != DEVELOPMENT_CONFIG:
        raise ValueError(
            "the smoke runner accepts only the exact canonical development "
            "configuration"
        )
    if isinstance(pair_limit, bool) or not isinstance(pair_limit, int) or pair_limit <= 0:
        raise ValueError("pair_limit must be a positive integer")

    graph_records: list[GraphSmokeRecord] = []
    setting_index = 0
    replicate_index = 0
    for model in SMOKE_MODELS:
        generated = generate_graph(config, setting_index, model, replicate_index)
        graph = generated.graph
        seeds = config.seeds_for_replicate(setting_index, model, replicate_index)
        shortest_paths = prepare_all_pairs_shortest_paths(graph)
        metrics: NetworkMetrics = calculate_network_metrics(
            graph, shortest_paths=shortest_paths
        )
        coordinates = embed_graph_in_poincare_disk(
            graph,
            seed=seeds.embedding_initialization,
            embedding_radius=config.embedding_radius,
            iterations=config.embedding_iterations,
            disk_epsilon=config.unit_disk_boundary_epsilon,
            tolerance=config.numerical_tolerance,
        )
        distortion = calculate_embedding_distortion(
            graph,
            coordinates,
            tolerance=config.numerical_tolerance,
            shortest_paths=shortest_paths,
        )
        prepared_euclidean = prepare_routing_coordinates(
            graph,
            coordinates,
            euclidean_distance,
            metric_name="euclidean",
        )
        prepared_poincare = prepare_routing_coordinates(
            graph,
            coordinates,
            poincare_distance,
            metric_name="poincare",
        )
        if tuple(prepared_euclidean.items()) != tuple(prepared_poincare.items()):
            raise RuntimeError("routing metrics did not receive identical coordinates")
        pair_count = min(pair_limit, config.source_destination_pairs_per_graph)
        pairs = sample_ordered_pairs(
            graph.nodes, pair_count, seeds.source_destination_sampling
        )

        method_records = tuple(
            method_record
            for source, destination in pairs
            for method_record in run_methods_for_pair(
                graph,
                coordinates,
                source,
                destination,
                graph_model=model,
                tolerance=config.numerical_tolerance,
                prepared_euclidean=prepared_euclidean,
                prepared_poincare=prepared_poincare,
            )
        )
        if len(method_records) != len(pairs) * 4:
            raise RuntimeError("smoke run did not record every pair and method")

        graph_records.append(
            GraphSmokeRecord(
                graph_model=model,
                graph_metadata=_stable_items(generated.metadata),
                embedding_metadata=_embedding_provenance(
                    config, seeds.embedding_initialization
                ),
                pair_sampling_metadata=_stable_items(
                    {
                        "algorithm": "python_random_sample_without_replacement_v1",
                        "source_destination_sampling_seed": (
                            seeds.source_destination_sampling
                        ),
                        "ordered_pairs_are_unique": True,
                    }
                ),
                network_metrics=_stable_items(metrics),
                embedding_distortion=_stable_items(distortion.as_dict()),
                coordinates=_stable_coordinates(coordinates),
                source_destination_pairs=pairs,
                method_records=method_records,
            )
        )

    return DevelopmentSmokeResult(
        result_schema_version=SMOKE_RESULT_SCHEMA_VERSION,
        configuration_name=config.name,
        configuration_fingerprint=config.configuration_fingerprint,
        implementation_source_fingerprint=implementation_source_fingerprint(),
        configuration_json=_configuration_json(config),
        runtime_metadata=_runtime_metadata(),
        graphs=tuple(graph_records),
    )


def print_smoke_summary(result: DevelopmentSmokeResult) -> None:
    """Print a concise deterministic summary without statistical claims."""

    print("Development smoke experiment (pipeline validation; not final results)")
    for graph_record in result.graphs:
        metrics = dict(graph_record.network_metrics)
        distortion = dict(graph_record.embedding_distortion)
        print(
            f"{graph_record.graph_model}: "
            f"n={metrics['number_of_vertices']}, "
            f"edges={metrics['number_of_edges']}, "
            f"distortion={distortion['mean_relative_distortion']:.6f}"
        )
        print(f"  fixed_pairs={graph_record.source_destination_pairs}")
        methods = tuple(dict.fromkeys(record.method for record in graph_record.method_records))
        for method in methods:
            selected = [
                record for record in graph_record.method_records if record.method == method
            ]
            successes = sum(record.success for record in selected)
            print(f"  {method}: {successes}/{len(selected)} successful")


def main() -> int:
    try:
        result = run_development_smoke()
        print_smoke_summary(result)
    except Exception as exc:  # CLI boundary: invariant failures must be non-zero.
        print(f"Development smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
