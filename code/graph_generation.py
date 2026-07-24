"""Deterministic generation of connected experimental graphs.

Erdos-Renyi samples are drawn repeatedly until a connected sample is found.
Each retry uses a distinct seed derived deterministically from the Part 1 graph
seed.  Barabasi-Albert graphs use the Part 1 graph seed directly and are
checked for connectedness even though NetworkX's valid BA construction should
always be connected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import platform

import networkx as nx

from experiment_config import (
    BA_FINITE_DEGREE_MATCH_RULE,
    BA_INITIAL_GRAPH,
    BARABASI_ALBERT,
    CONFIGURATION_SCHEMA_VERSION,
    ERDOS_RENYI,
    GRAPH_GENERATION_ATTEMPT_SEED_DOMAIN,
    MAX_SEED,
    SEED_DERIVATION_ALGORITHM,
    SEED_SPACE_SIZE,
    ExperimentConfig,
    derive_domain_seed,
)


MetadataValue = str | int | float
GraphMetadata = dict[str, MetadataValue]


class GraphGenerationError(RuntimeError):
    """Raised when a valid connected graph cannot be generated."""


@dataclass(frozen=True)
class GeneratedGraph:
    """A generated NetworkX graph and JSON-serializable provenance metadata."""

    graph: nx.Graph
    metadata: GraphMetadata


def _require_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _require_non_negative_int(name: str, value: int) -> None:
    _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_n(n: int) -> None:
    _require_int("n", n)
    if n < 2:
        raise ValueError("n must be at least 2")


def _validate_probability(p: float) -> None:
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError("p must be numeric")
    if not isfinite(p) or not 0 < p < 1:
        raise ValueError("p must satisfy 0 < p < 1")


def _validate_attachment(n: int, m: int) -> None:
    _require_int("m", m)
    if not 1 <= m < n:
        raise ValueError("m must satisfy 1 <= m < n")


def _validate_seed(graph_seed: int) -> None:
    _require_int("graph_seed", graph_seed)
    if not 0 <= graph_seed <= MAX_SEED:
        raise ValueError(f"graph_seed must be between 0 and {MAX_SEED}")


def _validate_max_attempts(max_attempts: int) -> None:
    _require_int("max_attempts", max_attempts)
    if not 1 <= max_attempts <= SEED_SPACE_SIZE:
        raise ValueError(
            f"max_attempts must be between 1 and {SEED_SPACE_SIZE}"
        )


def derive_attempt_seed(graph_seed: int, attempt_index: int) -> int:
    """Derive a deterministic standalone retry seed.

    Configured experiments use :meth:`ExperimentConfig.seed_for_graph_attempt`,
    whose identity also includes the setting, model, and replicate. This helper
    preserves the direct graph-generator API by namespacing the base graph seed
    and zero-based attempt index.
    """

    _validate_seed(graph_seed)
    _require_non_negative_int("attempt_index", attempt_index)
    if attempt_index >= SEED_SPACE_SIZE:
        raise ValueError(
            f"attempt_index must be smaller than {SEED_SPACE_SIZE}"
        )
    return derive_domain_seed(
        graph_seed,
        GRAPH_GENERATION_ATTEMPT_SEED_DOMAIN,
        "standalone_graph_generation",
        graph_seed,
        attempt_index,
    )


def _validate_generation_arguments(
    *, n: int, graph_seed: int, replicate_index: int
) -> None:
    _validate_n(n)
    _validate_seed(graph_seed)
    _require_non_negative_int("replicate_index", replicate_index)


def _assert_graph_invariants(graph: nx.Graph, n: int, model: str) -> None:
    expected_nodes = set(range(n))
    if graph.is_directed() or graph.is_multigraph():
        raise GraphGenerationError(f"{model} generation did not produce a simple graph")
    if set(graph.nodes) != expected_nodes:
        raise GraphGenerationError(
            f"{model} generation did not produce node labels 0 through {n - 1}"
        )
    if nx.number_of_selfloops(graph):
        raise GraphGenerationError(f"{model} generation produced a self-loop")
    if not nx.is_connected(graph):
        raise GraphGenerationError(f"{model} generation produced a disconnected graph")


def _metadata(
    *,
    model: str,
    n: int,
    replicate_index: int,
    graph_seed: int,
    generation_attempt_count: int,
    generation_attempt_index: int,
    generation_attempt_seed: int,
    setting_index: int | None,
    seed_derivation_algorithm: str = SEED_DERIVATION_ALGORITHM,
    p: float | None = None,
    m: int | None = None,
) -> GraphMetadata:
    metadata: GraphMetadata = {
        "graph_model": model,
        "n": n,
        "replicate_index": replicate_index,
        "graph_seed": graph_seed,
        "generation_attempt_count": generation_attempt_count,
        "generation_attempt_index": generation_attempt_index,
        "generation_attempt_seed": generation_attempt_seed,
        "seed_derivation_algorithm": seed_derivation_algorithm,
    }
    if setting_index is not None:
        metadata["setting_index"] = setting_index
    if p is not None:
        metadata["p"] = p
    if m is not None:
        metadata["m"] = m
        metadata["ba_initial_graph"] = BA_INITIAL_GRAPH
    return metadata


def generate_connected_erdos_renyi(
    *,
    n: int,
    p: float,
    graph_seed: int,
    replicate_index: int,
    max_attempts: int,
    setting_index: int | None = None,
    attempt_seeds: Sequence[int] | None = None,
) -> GeneratedGraph:
    """Generate a connected simple ``G(n, p)`` graph deterministically.

    Sampling is from the Erdos-Renyi model conditional on accepting only a
    connected graph.  A failed attempt is discarded rather than replaced by
    its largest connected component.
    """

    _validate_generation_arguments(
        n=n, graph_seed=graph_seed, replicate_index=replicate_index
    )
    _validate_probability(p)
    _validate_max_attempts(max_attempts)
    if setting_index is not None:
        _require_non_negative_int("setting_index", setting_index)
    validated_attempt_seeds: tuple[int, ...] | None = None
    if attempt_seeds is not None:
        if isinstance(attempt_seeds, (str, bytes)) or not isinstance(
            attempt_seeds, Sequence
        ):
            raise ValueError("attempt_seeds must be a sequence of integer seeds")
        validated_attempt_seeds = tuple(attempt_seeds)
        if len(validated_attempt_seeds) != max_attempts:
            raise ValueError("attempt_seeds must contain exactly max_attempts values")
        for attempt_seed in validated_attempt_seeds:
            _validate_seed(attempt_seed)
        if len(set(validated_attempt_seeds)) != len(validated_attempt_seeds):
            raise ValueError("attempt_seeds must be unique")

    used_attempt_seeds: set[int] = set()
    for attempt_index in range(max_attempts):
        if validated_attempt_seeds is None:
            attempt_seed = derive_attempt_seed(graph_seed, attempt_index)
        else:
            attempt_seed = validated_attempt_seeds[attempt_index]
        if attempt_seed in used_attempt_seeds:
            raise GraphGenerationError(
                "graph-attempt seed derivation produced a collision"
            )
        used_attempt_seeds.add(attempt_seed)
        graph = nx.gnp_random_graph(n, p, seed=attempt_seed, directed=False)
        if nx.is_connected(graph):
            _assert_graph_invariants(graph, n, ERDOS_RENYI)
            return GeneratedGraph(
                graph=graph,
                metadata=_metadata(
                    model=ERDOS_RENYI,
                    n=n,
                    p=float(p),
                    replicate_index=replicate_index,
                    graph_seed=graph_seed,
                    generation_attempt_count=attempt_index + 1,
                    generation_attempt_index=attempt_index,
                    generation_attempt_seed=attempt_seed,
                    setting_index=setting_index,
                ),
            )

    raise GraphGenerationError(
        "failed to generate a connected Erdos-Renyi graph "
        f"after {max_attempts} attempts (n={n}, p={p}, graph_seed={graph_seed})"
    )


def generate_connected_barabasi_albert(
    *,
    n: int,
    m: int,
    graph_seed: int,
    replicate_index: int,
    setting_index: int | None = None,
) -> GeneratedGraph:
    """Generate and validate a connected simple ``BA(n, m)`` graph."""

    _validate_generation_arguments(
        n=n, graph_seed=graph_seed, replicate_index=replicate_index
    )
    _validate_attachment(n, m)
    if setting_index is not None:
        _require_non_negative_int("setting_index", setting_index)

    initial_graph = nx.star_graph(m)
    graph = nx.barabasi_albert_graph(
        n,
        m,
        seed=graph_seed,
        initial_graph=initial_graph,
    )
    _assert_graph_invariants(graph, n, BARABASI_ALBERT)
    return GeneratedGraph(
        graph=graph,
        metadata=_metadata(
            model=BARABASI_ALBERT,
            n=n,
            m=m,
            replicate_index=replicate_index,
            graph_seed=graph_seed,
            generation_attempt_count=1,
            generation_attempt_index=0,
            generation_attempt_seed=graph_seed,
            setting_index=setting_index,
        ),
    )


def generate_graph(
    config: ExperimentConfig,
    setting_index: int,
    model: str,
    replicate_index: int,
) -> GeneratedGraph:
    """Generate one configured graph replicate using Part 1 seed derivation."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")

    # This validates setting/model/replicate indices and derives the unique
    # experiment-wide graph seed established in Part 1.
    seeds = config.seeds_for_replicate(setting_index, model, replicate_index)
    setting = config.parameter_settings[setting_index]

    if model == ERDOS_RENYI:
        generated = generate_connected_erdos_renyi(
            n=setting.n,
            p=setting.er_p,
            graph_seed=seeds.graph_generation,
            replicate_index=replicate_index,
            max_attempts=config.max_connected_graph_generation_attempts,
            setting_index=setting_index,
            attempt_seeds=tuple(
                config.seed_for_graph_attempt(
                    setting_index,
                    model,
                    replicate_index,
                    attempt_index,
                )
                for attempt_index in range(
                    config.max_connected_graph_generation_attempts
                )
            ),
        )
    elif model == BARABASI_ALBERT:
        generated = generate_connected_barabasi_albert(
            n=setting.n,
            m=setting.ba_m,
            graph_seed=seeds.graph_generation,
            replicate_index=replicate_index,
            setting_index=setting_index,
        )
    else:
        # ``seeds_for_replicate`` already rejects this case. Keep the explicit
        # branch as a defensive invariant should the configuration API change.
        raise ValueError(f"unsupported graph model: {model!r}")

    metadata = dict(generated.metadata)
    metadata.update(
        {
            "configuration_name": config.name,
            "configuration_schema_version": CONFIGURATION_SCHEMA_VERSION,
            "configuration_fingerprint": config.configuration_fingerprint,
            "setting_label": setting.label,
            "embedding_initialization_seed": seeds.embedding_initialization,
            "source_destination_sampling_seed": (
                seeds.source_destination_sampling
            ),
            "seed_derivation_algorithm": SEED_DERIVATION_ALGORITHM,
            "degree_match_rule": BA_FINITE_DEGREE_MATCH_RULE,
            "erdos_renyi_sampling": "G(n,p) conditioned on connectedness",
            "er_expected_average_degree": setting.er_expected_average_degree,
            "ba_exact_average_degree": setting.ba_exact_average_degree,
            "ba_exact_edge_count": setting.ba_exact_edge_count,
            "max_connected_graph_generation_attempts": (
                config.max_connected_graph_generation_attempts
            ),
            "unit_disk_boundary_epsilon": config.unit_disk_boundary_epsilon,
            "numerical_tolerance": config.numerical_tolerance,
            "approved_embedding_families": ",".join(
                config.approved_embedding_design.embedding_families
            ),
            "coordinate_condition_ids": ",".join(
                config.approved_embedding_design.coordinate_condition_ids
            ),
            "mds_maximum_radii": ",".join(
                format(radius, ".2f")
                for radius in config.approved_embedding_design.mds_maximum_radii
            ),
            "development_force_embedding_algorithm": config.embedding_algorithm,
            "development_force_embedding_method": config.embedding_method,
            "development_force_embedding_radius": config.embedding_radius,
            "development_force_embedding_iterations": config.embedding_iterations,
            "routing_tie_break_rule": config.routing_tie_break_rule,
            "networkx_version": nx.__version__,
            "python_version": platform.python_version(),
        }
    )
    return GeneratedGraph(graph=generated.graph, metadata=metadata)


def reproduce_graph_from_metadata(
    config: ExperimentConfig,
    metadata: Mapping[str, MetadataValue],
) -> GeneratedGraph:
    """Regenerate a configured graph after validating its provenance identity."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    required = (
        "configuration_name",
        "configuration_schema_version",
        "configuration_fingerprint",
        "setting_index",
        "setting_label",
        "graph_model",
        "replicate_index",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"metadata is missing required provenance: {missing}")
    if metadata["configuration_name"] != config.name:
        raise ValueError("metadata configuration_name does not match config")
    if metadata["configuration_schema_version"] != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError("metadata configuration schema version is unsupported")
    if metadata["configuration_fingerprint"] != config.configuration_fingerprint:
        raise ValueError("metadata configuration fingerprint does not match config")

    setting_index = metadata["setting_index"]
    replicate_index = metadata["replicate_index"]
    model = metadata["graph_model"]
    _require_non_negative_int("metadata setting_index", setting_index)
    _require_non_negative_int("metadata replicate_index", replicate_index)
    if not isinstance(model, str):
        raise ValueError("metadata graph_model must be a string")

    regenerated = generate_graph(
        config,
        setting_index,
        model,
        replicate_index,
    )
    if any(
        metadata.get(key) != value
        for key, value in regenerated.metadata.items()
    ):
        raise GraphGenerationError(
            "regenerated graph provenance does not match the supplied metadata"
        )
    return regenerated
