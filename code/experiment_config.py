"""Central experiment settings and reproducibility metadata.

This module defines settings only. It deliberately contains no graph generation,
embedding, routing, measurement, analysis, or plotting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2s, sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Iterator, TypeAlias


MIN_SEED = 0
MAX_SEED = 2**32 - 1
SEED_SPACE_SIZE = MAX_SEED + 1

CONFIGURATION_SCHEMA_VERSION = 2
SEED_DERIVATION_ALGORITHM = "blake2s-32-domain-separated-v1"
GRAPH_GENERATION_SEED_DOMAIN = "graph_generation"
EMBEDDING_INITIALIZATION_SEED_DOMAIN = "embedding_initialization"
SOURCE_DESTINATION_SAMPLING_SEED_DOMAIN = "source_destination_sampling"
GRAPH_GENERATION_ATTEMPT_SEED_DOMAIN = "graph_generation_attempt"
SEED_DERIVATION_PERSON = b"GRPseed1"

BA_INITIAL_GRAPH = "networkx.star_graph(m)"
BA_FINITE_DEGREE_MATCH_RULE = "p*(n-1) == 2*m*(n-m)/n"

EMBEDDING_ALGORITHM = "fruchterman_reingold"
EMBEDDING_METHOD = "dense_fruchterman_reingold_rescaled_v1"
ROUTING_METHOD_COUNT = 4

SeedIdentityPart: TypeAlias = str | int
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

ERDOS_RENYI = "erdos_renyi"
BARABASI_ALBERT = "barabasi_albert"
GRAPH_MODELS = (ERDOS_RENYI, BARABASI_ALBERT)

SMALLEST_NODE_ID = "smallest_node_id"
TIE_BREAK_DESCRIPTION = (
    "When candidate distances are equal within numerical_tolerance, choose the "
    "candidate with the smallest integer node ID."
)


def _require_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _require_positive_int(name: str, value: int) -> None:
    _require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_seed(name: str, value: int) -> None:
    _require_int(name, value)
    if not MIN_SEED <= value <= MAX_SEED:
        raise ValueError(f"{name} must be between {MIN_SEED} and {MAX_SEED}")


def derive_domain_seed(
    master_seed: int,
    domain: str,
    *identity: SeedIdentityPart,
) -> int:
    """Derive a stable domain-separated 32-bit seed.

    Python's process-randomized ``hash`` is deliberately avoided. The canonical
    JSON payload and versioned BLAKE2s personalization make the mapping stable
    across process starts for the same explicit identity.
    """

    _validate_seed("master_seed", master_seed)
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string")
    if not identity:
        raise ValueError("seed identity must not be empty")
    for part in identity:
        if isinstance(part, bool) or not isinstance(part, (str, int)):
            raise ValueError("seed identity parts must be strings or integers")

    payload = json.dumps(
        {
            "algorithm": SEED_DERIVATION_ALGORITHM,
            "domain": domain,
            "identity": list(identity),
            "master_seed": master_seed,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = blake2s(
        payload,
        digest_size=4,
        person=SEED_DERIVATION_PERSON,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _require_positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        finite = isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class DegreeMatchedParameters:
    """One ER/BA comparison at a shared graph size.

    NetworkX's default BA construction starts from ``star_graph(m)``. It has
    exactly ``m * (n - m)`` edges at size ``n``, so the finite-size match is
    ``er_p * (n - 1) == 2 * m * (n - m) / n``.
    """

    label: str
    n: int
    er_p: float
    ba_m: int
    provisional: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")

        _require_int("n", self.n)
        if self.n < 2:
            raise ValueError("n must be at least 2")

        if isinstance(self.er_p, bool) or not isinstance(self.er_p, (int, float)):
            raise ValueError("er_p must be numeric")
        try:
            finite_er_p = isfinite(self.er_p)
        except OverflowError:
            finite_er_p = False
        if not finite_er_p or not 0 < self.er_p < 1:
            raise ValueError("er_p must satisfy 0 < er_p < 1")

        _require_int("ba_m", self.ba_m)
        if not 1 <= self.ba_m < self.n:
            raise ValueError("ba_m must satisfy 1 <= ba_m < n")

        if not isinstance(self.provisional, bool):
            raise ValueError("provisional must be a boolean")

    @property
    def er_expected_average_degree(self) -> float:
        return self.er_p * (self.n - 1)

    @property
    def ba_exact_edge_count(self) -> int:
        return self.ba_m * (self.n - self.ba_m)

    @property
    def ba_exact_average_degree(self) -> float:
        return (2.0 * self.ba_exact_edge_count) / self.n

    @property
    def ba_asymptotic_average_degree(self) -> float:
        return 2.0 * self.ba_m

    @property
    def ba_approximate_average_degree(self) -> float:
        """Compatibility alias for the conventional asymptotic value ``2m``."""

        return self.ba_asymptotic_average_degree

    @property
    def expected_degree_gap(self) -> float:
        return abs(
            self.er_expected_average_degree
            - self.ba_exact_average_degree
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "label": self.label,
            "n": self.n,
            "er_p": self.er_p,
            "ba_m": self.ba_m,
            "ba_initial_graph": BA_INITIAL_GRAPH,
            "ba_exact_edge_count": self.ba_exact_edge_count,
            "er_expected_average_degree": self.er_expected_average_degree,
            "ba_exact_average_degree": self.ba_exact_average_degree,
            "ba_asymptotic_average_degree": self.ba_asymptotic_average_degree,
            "degree_match_rule": BA_FINITE_DEGREE_MATCH_RULE,
            "provisional": self.provisional,
        }


def make_degree_matched_parameters(
    *, n: int, ba_m: int, label: str, provisional: bool
) -> DegreeMatchedParameters:
    """Calculate the ER probability matching NetworkX BA's finite edge count."""

    _require_int("n", n)
    if n < 2:
        raise ValueError("n must be at least 2")
    _require_int("ba_m", ba_m)
    if not 1 <= ba_m < n:
        raise ValueError("ba_m must satisfy 1 <= ba_m < n")

    ba_exact_average_degree = (2.0 * ba_m * (n - ba_m)) / n
    er_p = ba_exact_average_degree / (n - 1)
    return DegreeMatchedParameters(
        label=label,
        n=n,
        er_p=er_p,
        ba_m=ba_m,
        provisional=provisional,
    )


@dataclass(frozen=True)
class ReplicateSeeds:
    graph_generation: int
    embedding_initialization: int
    source_destination_sampling: int

    def as_dict(self) -> dict[str, int]:
        return {
            "graph_generation": self.graph_generation,
            "embedding_initialization": self.embedding_initialization,
            "source_destination_sampling": self.source_destination_sampling,
        }


@dataclass(frozen=True)
class GraphReplicate:
    setting_index: int
    setting_label: str
    model: str
    n: int
    parameter: float | int
    replicate_index: int
    seeds: ReplicateSeeds


@dataclass(frozen=True)
class SeedUse:
    seed: int
    domain: str
    configuration_name: str
    setting_label: str
    model: str
    replicate_index: int
    attempt_index: int | None = None


@dataclass(frozen=True)
class SeedCollision:
    seed: int
    uses: tuple[SeedUse, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    parameter_settings: tuple[DegreeMatchedParameters, ...]
    graph_repetitions: int
    source_destination_pairs_per_graph: int
    graph_generation_master_seed: int
    embedding_initialization_master_seed: int
    source_destination_sampling_master_seed: int
    unit_disk_boundary_epsilon: float
    numerical_tolerance: float
    embedding_algorithm: str
    embedding_method: str
    embedding_radius: float
    embedding_iterations: int
    expected_degree_match_tolerance: float
    max_connected_graph_generation_attempts: int
    routing_tie_break_rule: str
    routing_tie_break_description: str
    is_provisional: bool
    provisional_values: tuple[str, ...] = ()
    workload_note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        settings = tuple(self.parameter_settings)
        object.__setattr__(self, "parameter_settings", settings)
        if not settings:
            raise ValueError("parameter_settings must not be empty")
        if not all(isinstance(item, DegreeMatchedParameters) for item in settings):
            raise ValueError(
                "parameter_settings must contain DegreeMatchedParameters values"
            )

        _require_positive_int("graph_repetitions", self.graph_repetitions)
        _require_positive_int(
            "source_destination_pairs_per_graph",
            self.source_destination_pairs_per_graph,
        )
        _require_positive_int(
            "max_connected_graph_generation_attempts",
            self.max_connected_graph_generation_attempts,
        )
        if self.max_connected_graph_generation_attempts > SEED_SPACE_SIZE:
            raise ValueError(
                "max_connected_graph_generation_attempts exceeds the seed space"
            )

        master_seeds = (
            self.graph_generation_master_seed,
            self.embedding_initialization_master_seed,
            self.source_destination_sampling_master_seed,
        )
        for seed_name, seed_value in zip(
            (
                "graph_generation_master_seed",
                "embedding_initialization_master_seed",
                "source_destination_sampling_master_seed",
            ),
            master_seeds,
            strict=True,
        ):
            _validate_seed(seed_name, seed_value)
        if len(set(master_seeds)) != len(master_seeds):
            raise ValueError("master seeds must be distinct")

        _require_positive_finite(
            "unit_disk_boundary_epsilon", self.unit_disk_boundary_epsilon
        )
        if self.unit_disk_boundary_epsilon >= 1:
            raise ValueError(
                "unit_disk_boundary_epsilon must satisfy 0 < epsilon < 1"
            )
        _require_positive_finite("numerical_tolerance", self.numerical_tolerance)
        if not isinstance(self.embedding_algorithm, str) or not self.embedding_algorithm:
            raise ValueError("embedding_algorithm must be a non-empty string")
        if not isinstance(self.embedding_method, str) or not self.embedding_method:
            raise ValueError("embedding_method must be a non-empty string")
        _require_positive_finite(
            "embedding_radius", self.embedding_radius
        )
        if self.embedding_radius >= 1 - self.unit_disk_boundary_epsilon:
            raise ValueError(
                "embedding_radius must satisfy "
                "0 < radius < 1 - unit_disk_boundary_epsilon"
            )
        _require_positive_int(
            "embedding_iterations", self.embedding_iterations
        )
        _require_positive_finite(
            "expected_degree_match_tolerance",
            self.expected_degree_match_tolerance,
        )

        if self.routing_tie_break_rule != SMALLEST_NODE_ID:
            raise ValueError(
                f"routing_tie_break_rule must be {SMALLEST_NODE_ID!r}"
            )
        if (
            not isinstance(self.routing_tie_break_description, str)
            or not self.routing_tie_break_description.strip()
        ):
            raise ValueError("routing_tie_break_description must be non-empty")

        if not isinstance(self.is_provisional, bool):
            raise ValueError("is_provisional must be a boolean")
        provisional_values = tuple(self.provisional_values)
        object.__setattr__(self, "provisional_values", provisional_values)
        if self.is_provisional and not provisional_values:
            raise ValueError(
                "a provisional configuration must identify provisional_values"
            )

        labels: set[str] = set()
        er_parameters: set[tuple[int, float]] = set()
        ba_parameters: set[tuple[int, int]] = set()
        for setting in settings:
            if setting.label in labels:
                raise ValueError(f"duplicate parameter label: {setting.label}")
            labels.add(setting.label)

            er_key = (setting.n, setting.er_p)
            if er_key in er_parameters:
                raise ValueError(
                    f"duplicate ER parameter for n={setting.n}: p={setting.er_p}"
                )
            er_parameters.add(er_key)

            ba_key = (setting.n, setting.ba_m)
            if ba_key in ba_parameters:
                raise ValueError(
                    f"duplicate BA parameter for n={setting.n}: m={setting.ba_m}"
                )
            ba_parameters.add(ba_key)

            ordered_pair_limit = setting.n * (setting.n - 1)
            if self.source_destination_pairs_per_graph > ordered_pair_limit:
                raise ValueError(
                    "source_destination_pairs_per_graph exceeds the number of "
                    f"ordered pairs for n={setting.n}: {ordered_pair_limit}"
                )

            if setting.expected_degree_gap > self.expected_degree_match_tolerance:
                raise ValueError(
                    f"ER/BA expected-degree mismatch for {setting.label!r} is "
                    f"{setting.expected_degree_gap}, exceeding tolerance "
                    f"{self.expected_degree_match_tolerance}"
                )

        if self.graph_replicate_count > SEED_SPACE_SIZE:
            raise ValueError(
                "the number of graph replicates exceeds the available seed space"
            )

    @property
    def graph_sizes(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(setting.n for setting in self.parameter_settings))

    @property
    def er_probabilities(self) -> tuple[tuple[int, float], ...]:
        return tuple((setting.n, setting.er_p) for setting in self.parameter_settings)

    @property
    def ba_attachment_values(self) -> tuple[tuple[int, int], ...]:
        return tuple((setting.n, setting.ba_m) for setting in self.parameter_settings)

    @property
    def graph_replicate_count(self) -> int:
        return len(self.parameter_settings) * len(GRAPH_MODELS) * self.graph_repetitions

    @property
    def sampled_ordered_pair_count(self) -> int:
        return (
            self.graph_replicate_count
            * self.source_destination_pairs_per_graph
        )

    @property
    def development_embedding_radius(self) -> float:
        """Compatibility alias; embedding settings apply to every configuration."""

        return self.embedding_radius

    @property
    def spring_layout_iterations(self) -> int:
        """Compatibility alias for the configured embedding iteration count."""

        return self.embedding_iterations

    @property
    def seed_derivation_metadata(self) -> dict[str, JsonValue]:
        return {
            "algorithm": SEED_DERIVATION_ALGORITHM,
            "graph_generation_domain": GRAPH_GENERATION_SEED_DOMAIN,
            "embedding_initialization_domain": EMBEDDING_INITIALIZATION_SEED_DOMAIN,
            "source_destination_sampling_domain": (
                SOURCE_DESTINATION_SAMPLING_SEED_DOMAIN
            ),
            "graph_generation_attempt_domain": (
                GRAPH_GENERATION_ATTEMPT_SEED_DOMAIN
            ),
            "identity_fields": [
                "configuration_name",
                "configuration_schema_version",
                "setting_index",
                "setting_label",
                "n",
                "er_p_hex",
                "ba_m",
                "model",
                "replicate_index",
                "attempt_index (attempt seeds only)",
            ],
        }

    @property
    def workload_estimate(self) -> dict[str, int]:
        er_graphs = len(self.parameter_settings) * self.graph_repetitions
        ba_graphs = er_graphs
        distortion_pairs = sum(
            len(GRAPH_MODELS)
            * self.graph_repetitions
            * setting.n
            * (setting.n - 1)
            // 2
            for setting in self.parameter_settings
        )
        return {
            "graph_replicates": self.graph_replicate_count,
            "erdos_renyi_graph_replicates": er_graphs,
            "barabasi_albert_graph_replicates": ba_graphs,
            "embedding_runs": self.graph_replicate_count,
            "sampled_ordered_pairs": self.sampled_ordered_pair_count,
            "routing_methods_per_pair": ROUTING_METHOD_COUNT,
            "dijkstra_routing_runs": self.sampled_ordered_pair_count,
            "euclidean_greedy_routing_runs": self.sampled_ordered_pair_count,
            "hyperbolic_greedy_routing_runs": self.sampled_ordered_pair_count,
            "repaired_hyperbolic_greedy_routing_runs": (
                self.sampled_ordered_pair_count
            ),
            "routing_method_runs": (
                self.sampled_ordered_pair_count * ROUTING_METHOD_COUNT
            ),
            "distortion_unordered_pairs": distortion_pairs,
            "maximum_erdos_renyi_generation_attempts": (
                er_graphs * self.max_connected_graph_generation_attempts
            ),
            "maximum_graph_generation_calls": (
                er_graphs * self.max_connected_graph_generation_attempts
                + ba_graphs
            ),
        }

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a versioned JSON-compatible configuration snapshot."""

        return {
            "configuration_schema_version": CONFIGURATION_SCHEMA_VERSION,
            "name": self.name,
            "parameter_settings": [
                setting.as_dict() for setting in self.parameter_settings
            ],
            "graph_repetitions": self.graph_repetitions,
            "source_destination_pairs_per_graph": (
                self.source_destination_pairs_per_graph
            ),
            "master_seeds": {
                "graph_generation": self.graph_generation_master_seed,
                "embedding_initialization": (
                    self.embedding_initialization_master_seed
                ),
                "source_destination_sampling": (
                    self.source_destination_sampling_master_seed
                ),
            },
            "seed_derivation": self.seed_derivation_metadata,
            "unit_disk_boundary_epsilon": self.unit_disk_boundary_epsilon,
            "numerical_tolerance": self.numerical_tolerance,
            "embedding": {
                "algorithm": self.embedding_algorithm,
                "method": self.embedding_method,
                "radius": self.embedding_radius,
                "iterations": self.embedding_iterations,
            },
            "expected_degree_match_tolerance": (
                self.expected_degree_match_tolerance
            ),
            "max_connected_graph_generation_attempts": (
                self.max_connected_graph_generation_attempts
            ),
            "routing": {
                "tie_break_rule": self.routing_tie_break_rule,
                "tie_break_description": self.routing_tie_break_description,
            },
            "is_provisional": self.is_provisional,
            "provisional_values": list(self.provisional_values),
            "workload": self.workload_estimate,
            "workload_note": self.workload_note,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the configuration deterministically."""

        return json.dumps(
            self.as_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    @property
    def configuration_fingerprint(self) -> str:
        return sha256(self.to_json().encode("ascii")).hexdigest()

    def _validate_replicate_identity(
        self, setting_index: int, model: str, replicate_index: int
    ) -> DegreeMatchedParameters:
        _require_int("setting_index", setting_index)
        if not 0 <= setting_index < len(self.parameter_settings):
            raise ValueError("setting_index is outside the configured range")
        if model not in GRAPH_MODELS:
            raise ValueError(f"model must be one of {GRAPH_MODELS}")
        _require_int("replicate_index", replicate_index)
        if not 0 <= replicate_index < self.graph_repetitions:
            raise ValueError("replicate_index is outside the configured range")
        return self.parameter_settings[setting_index]

    def _seed_identity(
        self,
        setting_index: int,
        setting: DegreeMatchedParameters,
        model: str,
        replicate_index: int,
    ) -> tuple[SeedIdentityPart, ...]:
        return (
            self.name,
            CONFIGURATION_SCHEMA_VERSION,
            setting_index,
            setting.label,
            setting.n,
            setting.er_p.hex(),
            setting.ba_m,
            model,
            replicate_index,
        )

    def seeds_for_replicate(
        self, setting_index: int, model: str, replicate_index: int
    ) -> ReplicateSeeds:
        setting = self._validate_replicate_identity(
            setting_index, model, replicate_index
        )
        identity = self._seed_identity(
            setting_index,
            setting,
            model,
            replicate_index,
        )
        return ReplicateSeeds(
            graph_generation=derive_domain_seed(
                self.graph_generation_master_seed,
                GRAPH_GENERATION_SEED_DOMAIN,
                *identity,
            ),
            embedding_initialization=derive_domain_seed(
                self.embedding_initialization_master_seed,
                EMBEDDING_INITIALIZATION_SEED_DOMAIN,
                *identity,
            ),
            source_destination_sampling=derive_domain_seed(
                self.source_destination_sampling_master_seed,
                SOURCE_DESTINATION_SAMPLING_SEED_DOMAIN,
                *identity,
            ),
        )

    def seed_for_graph_attempt(
        self,
        setting_index: int,
        model: str,
        replicate_index: int,
        attempt_index: int,
    ) -> int:
        setting = self._validate_replicate_identity(
            setting_index, model, replicate_index
        )
        if model != ERDOS_RENYI:
            raise ValueError("graph attempt seeds apply only to Erdos-Renyi graphs")
        _require_int("attempt_index", attempt_index)
        if not 0 <= attempt_index < self.max_connected_graph_generation_attempts:
            raise ValueError("attempt_index is outside the configured range")
        return derive_domain_seed(
            self.graph_generation_master_seed,
            GRAPH_GENERATION_ATTEMPT_SEED_DOMAIN,
            *self._seed_identity(
                setting_index,
                setting,
                model,
                replicate_index,
            ),
            attempt_index,
        )

    def iter_graph_replicates(self) -> Iterator[GraphReplicate]:
        for setting_index, setting in enumerate(self.parameter_settings):
            for model in GRAPH_MODELS:
                parameter: float | int
                if model == ERDOS_RENYI:
                    parameter = setting.er_p
                else:
                    parameter = setting.ba_m

                for replicate_index in range(self.graph_repetitions):
                    yield GraphReplicate(
                        setting_index=setting_index,
                        setting_label=setting.label,
                        model=model,
                        n=setting.n,
                        parameter=parameter,
                        replicate_index=replicate_index,
                        seeds=self.seeds_for_replicate(
                            setting_index, model, replicate_index
                        ),
                    )


def iter_seed_uses(config: ExperimentConfig) -> Iterator[SeedUse]:
    """Yield every configured random-seed use, including all possible ER retries."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    for replicate in config.iter_graph_replicates():
        shared = {
            "configuration_name": config.name,
            "setting_label": replicate.setting_label,
            "model": replicate.model,
            "replicate_index": replicate.replicate_index,
        }
        yield SeedUse(
            seed=replicate.seeds.graph_generation,
            domain=GRAPH_GENERATION_SEED_DOMAIN,
            **shared,
        )
        yield SeedUse(
            seed=replicate.seeds.embedding_initialization,
            domain=EMBEDDING_INITIALIZATION_SEED_DOMAIN,
            **shared,
        )
        yield SeedUse(
            seed=replicate.seeds.source_destination_sampling,
            domain=SOURCE_DESTINATION_SAMPLING_SEED_DOMAIN,
            **shared,
        )
        if replicate.model == ERDOS_RENYI:
            for attempt_index in range(
                config.max_connected_graph_generation_attempts
            ):
                yield SeedUse(
                    seed=config.seed_for_graph_attempt(
                        replicate.setting_index,
                        replicate.model,
                        replicate.replicate_index,
                        attempt_index,
                    ),
                    domain=GRAPH_GENERATION_ATTEMPT_SEED_DOMAIN,
                    attempt_index=attempt_index,
                    **shared,
                )


def audit_seed_collisions(config: ExperimentConfig) -> tuple[SeedCollision, ...]:
    """Return deterministic details for every reused 32-bit seed in ``config``."""

    uses_by_seed: dict[int, list[SeedUse]] = {}
    for use in iter_seed_uses(config):
        uses_by_seed.setdefault(use.seed, []).append(use)
    return tuple(
        SeedCollision(seed=seed, uses=tuple(uses_by_seed[seed]))
        for seed in sorted(uses_by_seed)
        if len(uses_by_seed[seed]) > 1
    )


DEVELOPMENT_CONFIG = ExperimentConfig(
    name="development",
    parameter_settings=tuple(
        make_degree_matched_parameters(
            n=30,
            ba_m=ba_m,
            label=f"dev_n30_m{ba_m}",
            provisional=False,
        )
        for ba_m in (2, 4)
    ),
    graph_repetitions=2,
    source_destination_pairs_per_graph=25,
    graph_generation_master_seed=10_001,
    embedding_initialization_master_seed=20_001,
    source_destination_sampling_master_seed=30_001,
    unit_disk_boundary_epsilon=1e-6,
    numerical_tolerance=1e-12,
    embedding_algorithm=EMBEDDING_ALGORITHM,
    embedding_method=EMBEDDING_METHOD,
    embedding_radius=0.85,
    embedding_iterations=100,
    expected_degree_match_tolerance=1e-12,
    max_connected_graph_generation_attempts=25,
    routing_tie_break_rule=SMALLEST_NODE_ID,
    routing_tie_break_description=TIE_BREAK_DESCRIPTION,
    is_provisional=False,
    workload_note=(
        "Debug-only settings: 8 graph replicates, 200 sampled ordered pairs, "
        "and 800 routing-method runs across those graphs."
    ),
)


FULL_EXPERIMENT_CONFIG = ExperimentConfig(
    name="full_experiment",
    parameter_settings=tuple(
        make_degree_matched_parameters(
            n=n,
            ba_m=ba_m,
            label=f"full_n{n}_m{ba_m}",
            provisional=True,
        )
        for n in (100, 300, 1_000)
        for ba_m in (4, 8, 16)
    ),
    graph_repetitions=20,
    source_destination_pairs_per_graph=1_000,
    graph_generation_master_seed=1_000_003,
    embedding_initialization_master_seed=2_000_003,
    source_destination_sampling_master_seed=3_000_003,
    unit_disk_boundary_epsilon=1e-6,
    numerical_tolerance=1e-12,
    embedding_algorithm=EMBEDDING_ALGORITHM,
    embedding_method=EMBEDDING_METHOD,
    embedding_radius=0.85,
    embedding_iterations=200,
    expected_degree_match_tolerance=1e-12,
    max_connected_graph_generation_attempts=50,
    routing_tie_break_rule=SMALLEST_NODE_ID,
    routing_tie_break_description=TIE_BREAK_DESCRIPTION,
    is_provisional=True,
    provisional_values=(
        "graph sizes",
        "BA attachment values",
        "ER probabilities derived from expected-degree matching",
        "graph repetitions",
        "source-destination pairs per graph",
        "master seeds",
        "unit-disk boundary epsilon",
        "embedding method, radius, and iterations",
        "numerical and expected-degree tolerances",
        "connected-graph generation attempt limit",
        "routing tie-breaking rule",
    ),
    workload_note=(
        "Provisional paper workload: 360 graph replicates and 360,000 sampled "
        "ordered pairs per routing method/benchmark, producing 1,440,000 "
        "routing-method runs and 65,916,000 distortion pairs. Runtime and "
        "memory must be benchmarked before these values are frozen."
    ),
)


CONFIGURATIONS = MappingProxyType(
    {
        DEVELOPMENT_CONFIG.name: DEVELOPMENT_CONFIG,
        FULL_EXPERIMENT_CONFIG.name: FULL_EXPERIMENT_CONFIG,
    }
)


def get_config(name: str) -> ExperimentConfig:
    """Return a named configuration, rejecting unknown names explicitly."""

    try:
        return CONFIGURATIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown configuration {name!r}; choose from {tuple(CONFIGURATIONS)}"
        ) from exc
