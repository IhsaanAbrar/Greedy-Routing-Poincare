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

CONFIGURATION_SCHEMA_VERSION = 4
SEED_IDENTITY_VERSION = 3
SEED_DERIVATION_ALGORITHM = "blake2s-32-domain-separated-v1"
GRAPH_GENERATION_SEED_DOMAIN = "graph_generation"
EMBEDDING_INITIALIZATION_SEED_DOMAIN = "embedding_initialization"
SOURCE_DESTINATION_SAMPLING_SEED_DOMAIN = "source_destination_sampling"
GRAPH_GENERATION_ATTEMPT_SEED_DOMAIN = "graph_generation_attempt"
SEED_DERIVATION_PERSON = b"GRPseed1"

BA_INITIAL_GRAPH = "networkx.star_graph(m)"
BA_FINITE_DEGREE_MATCH_RULE = "p = 2*m*(n-m)/(n*(n-1))"
ER_CONNECTIVITY_CONDITIONING = "accepted ER observations are conditional on connectivity"
ORDERED_PAIR_SAMPLER_DOMAIN = "ordered_pair_sampler_v1"
ORDERED_PAIR_SAMPLER_ALGORITHM = (
    "blake2s_uint64_rejection_without_replacement_v1"
)
ANALYSIS_BOOTSTRAP_DOMAIN = "analysis_bootstrap_v1"
ANALYSIS_BOOTSTRAP_REPLICATES = 10_000
ANALYSIS_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
ANALYSIS_BOOTSTRAP_METHOD = "two_sided_percentile_graph_clustered"
FINGERPRINT_ALGORITHM = "sha256"
CANONICAL_JSON_VERSION = "tagged_float64_canonical_json_v1"

EMBEDDING_ALGORITHM = "fruchterman_reingold"
EMBEDDING_METHOD = "dense_fruchterman_reingold_rescaled_v1"

HYDRA_EMBEDDING_FAMILY = "hydra"
MDS_EMBEDDING_FAMILY = "classical_mds"
APPROVED_EMBEDDING_FAMILIES = (
    HYDRA_EMBEDDING_FAMILY,
    MDS_EMBEDDING_FAMILY,
)
HYDRA_CONDITION_ID = "hydra_2d_k1_frechet_centered_v1"
MDS_BASE_EMBEDDING_ID = "classical_mds_2d_v1"
MDS_MAXIMUM_RADII = (0.50, 0.70, 0.85, 0.95)
MDS_CONDITION_IDS = ("mds_r050", "mds_r070", "mds_r085", "mds_r095")
HYDRA_DIMENSION = 2
HYDRA_KAPPA = 1.0
HYDRA_CURVATURE = -1.0
MDS_DIMENSION = 2
HYDRA_CENTERING_TOLERANCE = 1e-10
HYDRA_CENTERING_MAX_ITERATIONS = 256
HYDRA_EIGENVALUE_TOLERANCE = 1e-12
HYDRA_ISOMETRY_TOLERANCE = 1e-9
HYDRA_BOUNDARY_ROUNDOFF_TOLERANCE = 1e-12
HYDRA_ISOMETRY_ABSOLUTE_TOLERANCE = 1e-10
HYDRA_ISOMETRY_RELATIVE_TOLERANCE = 1e-9
HYDRA_FRECHET_ALGORITHM = (
    "deterministic_riemannian_newton_armijo_backtracking_v1"
)
HYDRA_FRECHET_CONVERGENCE_CRITERION = (
    "hyperbolic_tangent_mean_norm_lte_1e-10"
)
MDS_EIGENVALUE_RELATIVE_TOLERANCE = 1e-12
MDS_CENTROID_TOLERANCE = 1e-12
MDS_EUCLIDEAN_TOLERANCE_POLICY = (
    "scale_by_radius_over_maximum_approved_radius_v1"
)
FEASIBILITY_PILOT_SEEDS = (
    4_000_003,
    4_000_019,
    4_000_037,
    4_000_063,
    4_000_099,
    4_000_121,
)

COORDINATE_CONDITION_COUNT = 1 + len(MDS_CONDITION_IDS)
COORDINATE_DEPENDENT_ROUTING_METHOD_COUNT = 3
DIJKSTRA_RUNS_PER_PAIR = 1
ROUTING_METHOD_COUNT = (
    DIJKSTRA_RUNS_PER_PAIR
    + COORDINATE_CONDITION_COUNT * COORDINATE_DEPENDENT_ROUTING_METHOD_COUNT
)
DISTORTION_METRIC_CONDITION_COUNT = 7

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


def _canonical_fingerprint_value(value):
    """Recursively encode values for exact, portable fingerprint JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("fingerprint payloads reject NaN and infinity")
        return {"__float64__": value.hex()}
    if isinstance(value, MappingProxyType):
        value = dict(value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("fingerprint object keys must be strings")
        return {
            key: _canonical_fingerprint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_fingerprint_value(item) for item in value]
    raise ValueError(
        f"unsupported fingerprint payload value: {type(value).__name__}"
    )


def canonical_fingerprint_json(payload: dict[str, object]) -> str:
    """Return canonical UTF-8 JSON text with tagged exact float64 values."""

    if not isinstance(payload, dict):
        raise ValueError("fingerprint payload must be an object")
    encoded = _canonical_fingerprint_value(payload)
    return json.dumps(
        encoded,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_fingerprint_hash(payload: dict[str, object]) -> str:
    """Return the lowercase SHA-256 digest of canonical fingerprint JSON."""

    canonical = canonical_fingerprint_json(payload).encode("utf-8")
    return sha256(canonical).hexdigest()


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
    def er_probability_numerator(self) -> int:
        return 2 * self.ba_m * (self.n - self.ba_m)

    @property
    def er_probability_denominator(self) -> int:
        return self.n * (self.n - 1)

    @property
    def seed_identity_er_p_hex(self) -> str:
        """Preserve the version-3 two-operation float used by seed identities."""

        legacy_average_degree = (
            2.0 * self.ba_m * (self.n - self.ba_m)
        ) / self.n
        return (legacy_average_degree / (self.n - 1)).hex()

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
            "er_p_float64_hex": float(self.er_p).hex(),
            "seed_identity_er_p_hex_v3": self.seed_identity_er_p_hex,
            "er_p_exact_numerator": self.er_probability_numerator,
            "er_p_exact_denominator": self.er_probability_denominator,
            "ba_m": self.ba_m,
            "ba_initial_graph": BA_INITIAL_GRAPH,
            "ba_exact_edge_count": self.ba_exact_edge_count,
            "er_expected_average_degree": self.er_expected_average_degree,
            "ba_exact_average_degree": self.ba_exact_average_degree,
            "ba_asymptotic_average_degree": self.ba_asymptotic_average_degree,
            "degree_match_rule": BA_FINITE_DEGREE_MATCH_RULE,
            "erdos_renyi_conditioning": ER_CONNECTIVITY_CONDITIONING,
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

    er_p = (2 * ba_m * (n - ba_m)) / (n * (n - 1))
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
class ApprovedEmbeddingDesign:
    """Immutable Hydra/MDS coordinate conditions approved for the experiment."""

    embedding_families: tuple[str, str]
    hydra_condition_id: str
    hydra_dimension: int
    hydra_kappa: float
    hydra_curvature: float
    hydra_centering_tolerance: float
    hydra_centering_max_iterations: int
    hydra_eigenvalue_tolerance: float
    hydra_isometry_absolute_tolerance: float
    hydra_isometry_tolerance: float
    hydra_boundary_roundoff_tolerance: float
    mds_base_embedding_id: str
    mds_dimension: int
    mds_maximum_radii: tuple[float, float, float, float]
    mds_condition_ids: tuple[str, str, str, str]
    mds_eigenvalue_relative_tolerance: float
    mds_centroid_tolerance: float
    mds_euclidean_tolerance_policy: str
    feasibility_pilot_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding_families", tuple(self.embedding_families))
        object.__setattr__(
            self, "mds_maximum_radii", tuple(self.mds_maximum_radii)
        )
        object.__setattr__(self, "mds_condition_ids", tuple(self.mds_condition_ids))
        object.__setattr__(
            self, "feasibility_pilot_seeds", tuple(self.feasibility_pilot_seeds)
        )
        if self.embedding_families != APPROVED_EMBEDDING_FAMILIES:
            raise ValueError(
                f"embedding_families must equal {APPROVED_EMBEDDING_FAMILIES}"
            )
        if self.hydra_condition_id != HYDRA_CONDITION_ID:
            raise ValueError(f"hydra_condition_id must be {HYDRA_CONDITION_ID!r}")
        if self.hydra_dimension != HYDRA_DIMENSION:
            raise ValueError("hydra_dimension must equal 2")
        if self.hydra_kappa != HYDRA_KAPPA or self.hydra_curvature != HYDRA_CURVATURE:
            raise ValueError(
                "Hydra must use kappa=1 under the sectional-curvature -1 convention"
            )
        for name, value in (
            ("hydra_centering_tolerance", self.hydra_centering_tolerance),
            ("hydra_eigenvalue_tolerance", self.hydra_eigenvalue_tolerance),
            (
                "hydra_isometry_absolute_tolerance",
                self.hydra_isometry_absolute_tolerance,
            ),
            ("hydra_isometry_tolerance", self.hydra_isometry_tolerance),
            (
                "hydra_boundary_roundoff_tolerance",
                self.hydra_boundary_roundoff_tolerance,
            ),
            (
                "mds_eigenvalue_relative_tolerance",
                self.mds_eigenvalue_relative_tolerance,
            ),
            ("mds_centroid_tolerance", self.mds_centroid_tolerance),
        ):
            _require_positive_finite(name, value)
        if (
            self.hydra_boundary_roundoff_tolerance
            > HYDRA_BOUNDARY_ROUNDOFF_TOLERANCE
        ):
            raise ValueError(
                "hydra_boundary_roundoff_tolerance must remain roundoff-scale"
            )
        _require_positive_int(
            "hydra_centering_max_iterations",
            self.hydra_centering_max_iterations,
        )
        exact_hydra_values = (
            self.hydra_centering_tolerance == HYDRA_CENTERING_TOLERANCE
            and self.hydra_centering_max_iterations
            == HYDRA_CENTERING_MAX_ITERATIONS
            and self.hydra_eigenvalue_tolerance == HYDRA_EIGENVALUE_TOLERANCE
            and self.hydra_isometry_absolute_tolerance
            == HYDRA_ISOMETRY_ABSOLUTE_TOLERANCE
            and self.hydra_isometry_tolerance
            == HYDRA_ISOMETRY_RELATIVE_TOLERANCE
            and self.hydra_boundary_roundoff_tolerance
            == HYDRA_BOUNDARY_ROUNDOFF_TOLERANCE
        )
        if not exact_hydra_values:
            raise ValueError("Hydra numerical settings must equal the frozen values")
        if self.mds_base_embedding_id != MDS_BASE_EMBEDDING_ID:
            raise ValueError(
                f"mds_base_embedding_id must be {MDS_BASE_EMBEDDING_ID!r}"
            )
        if self.mds_dimension != MDS_DIMENSION:
            raise ValueError("mds_dimension must equal 2")
        if self.mds_maximum_radii != MDS_MAXIMUM_RADII:
            raise ValueError(
                f"mds_maximum_radii must equal {MDS_MAXIMUM_RADII}"
            )
        if self.mds_condition_ids != MDS_CONDITION_IDS:
            raise ValueError(
                f"mds_condition_ids must equal {MDS_CONDITION_IDS}"
            )
        if (
            self.mds_euclidean_tolerance_policy
            != MDS_EUCLIDEAN_TOLERANCE_POLICY
        ):
            raise ValueError(
                "mds_euclidean_tolerance_policy must use the approved "
                "scale-equivariant rule"
            )
        if (
            self.mds_eigenvalue_relative_tolerance
            != MDS_EIGENVALUE_RELATIVE_TOLERANCE
            or self.mds_centroid_tolerance != MDS_CENTROID_TOLERANCE
        ):
            raise ValueError("MDS numerical settings must equal the frozen values")
        if len(set(self.mds_condition_ids)) != len(self.mds_condition_ids):
            raise ValueError("mds_condition_ids must be unique")
        if len(self.feasibility_pilot_seeds) < 2:
            raise ValueError("at least two feasibility_pilot_seeds are required")
        for seed in self.feasibility_pilot_seeds:
            _validate_seed("feasibility_pilot_seed", seed)
        if len(set(self.feasibility_pilot_seeds)) != len(
            self.feasibility_pilot_seeds
        ):
            raise ValueError("feasibility_pilot_seeds must be unique")
        if self.feasibility_pilot_seeds != FEASIBILITY_PILOT_SEEDS:
            raise ValueError("feasibility_pilot_seeds must equal the frozen exclusions")

    @property
    def coordinate_condition_ids(self) -> tuple[str, ...]:
        return (self.hydra_condition_id, *self.mds_condition_ids)

    @property
    def independent_embedding_family_count(self) -> int:
        return len(self.embedding_families)

    @property
    def coordinate_condition_count(self) -> int:
        return len(self.coordinate_condition_ids)

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "embedding_families": list(self.embedding_families),
            "independent_embedding_family_count": (
                self.independent_embedding_family_count
            ),
            "coordinate_condition_count": self.coordinate_condition_count,
            "coordinate_condition_ids": list(self.coordinate_condition_ids),
            "hydra": {
                "condition_id": self.hydra_condition_id,
                "dimension": self.hydra_dimension,
                "kappa": self.hydra_kappa,
                "sectional_curvature": self.hydra_curvature,
                "centering": "unweighted_hyperbolic_frechet_mean_to_origin",
                "centering_algorithm": HYDRA_FRECHET_ALGORITHM,
                "centering_convergence_criterion": (
                    HYDRA_FRECHET_CONVERGENCE_CRITERION
                ),
                "centering_algorithm_parameters": {
                    "initial_point": "poincare_origin",
                    "newton_hessian": "exact_hyperbolic_squared_distance",
                    "line_search": "armijo_backtracking",
                    "initial_step": 1.0,
                    "backtracking_factor": 0.5,
                    "armijo_coefficient": 1e-4,
                    "maximum_line_search_steps": 48,
                    "non_convergence": "error",
                },
                "centering_tolerance": self.hydra_centering_tolerance,
                "centering_max_iterations": self.hydra_centering_max_iterations,
                "eigenvalue_threshold": (
                    "1e-12*max(1,max(abs(all_eigenvalues)))"
                ),
                "eigenvalue_tolerance": self.hydra_eigenvalue_tolerance,
                "spatial_eigenvalues": "two_most_negative",
                "effective_spatial_rank": (
                    "rank_1_allowed_active_coordinate_0_"
                    "coordinate_1_zero_padded_rank_0_error"
                ),
                "isometry_absolute_tolerance": (
                    self.hydra_isometry_absolute_tolerance
                ),
                "isometry_relative_tolerance": self.hydra_isometry_tolerance,
                "isometry_rule": (
                    "abs(after-before) <= abs_tol + rel_tol*"
                    "max(abs(before),abs(after))"
                ),
                "boundary_roundoff_tolerance": (
                    self.hydra_boundary_roundoff_tolerance
                ),
                "post_centering_radial_rescaling": False,
                "standard_hydra_only": True,
            },
            "classical_mds": {
                "base_embedding_id": self.mds_base_embedding_id,
                "dimension": self.mds_dimension,
                "maximum_radii": list(self.mds_maximum_radii),
                "condition_ids": list(self.mds_condition_ids),
                "radii_are_nested_sensitivity_transformations": True,
                "eigenvalue_threshold": (
                    "1e-12*max(1,max(abs(all_eigenvalues)))"
                ),
                "eigenvalue_relative_tolerance": self.mds_eigenvalue_relative_tolerance,
                "effective_rank": "rank_2_normal_rank_1_zero_pad_rank_0_error",
                "centroid_tolerance": self.mds_centroid_tolerance,
                "euclidean_routing_tolerance_policy": (
                    self.mds_euclidean_tolerance_policy
                ),
            },
            "feasibility_pilot": {
                "excluded_from_final_experiment": True,
                "reserved_seeds": list(self.feasibility_pilot_seeds),
            },
        }


APPROVED_EMBEDDING_DESIGN = ApprovedEmbeddingDesign(
    embedding_families=APPROVED_EMBEDDING_FAMILIES,
    hydra_condition_id=HYDRA_CONDITION_ID,
    hydra_dimension=HYDRA_DIMENSION,
    hydra_kappa=HYDRA_KAPPA,
    hydra_curvature=HYDRA_CURVATURE,
    hydra_centering_tolerance=HYDRA_CENTERING_TOLERANCE,
    hydra_centering_max_iterations=HYDRA_CENTERING_MAX_ITERATIONS,
    hydra_eigenvalue_tolerance=HYDRA_EIGENVALUE_TOLERANCE,
    hydra_isometry_absolute_tolerance=HYDRA_ISOMETRY_ABSOLUTE_TOLERANCE,
    hydra_isometry_tolerance=HYDRA_ISOMETRY_TOLERANCE,
    hydra_boundary_roundoff_tolerance=HYDRA_BOUNDARY_ROUNDOFF_TOLERANCE,
    mds_base_embedding_id=MDS_BASE_EMBEDDING_ID,
    mds_dimension=MDS_DIMENSION,
    mds_maximum_radii=MDS_MAXIMUM_RADII,
    mds_condition_ids=MDS_CONDITION_IDS,
    mds_eigenvalue_relative_tolerance=MDS_EIGENVALUE_RELATIVE_TOLERANCE,
    mds_centroid_tolerance=MDS_CENTROID_TOLERANCE,
    mds_euclidean_tolerance_policy=MDS_EUCLIDEAN_TOLERANCE_POLICY,
    feasibility_pilot_seeds=FEASIBILITY_PILOT_SEEDS,
)


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
    approved_embedding_design: ApprovedEmbeddingDesign
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
        if not isinstance(
            self.approved_embedding_design, ApprovedEmbeddingDesign
        ):
            raise ValueError(
                "approved_embedding_design must be an ApprovedEmbeddingDesign"
            )
        if set(self.approved_embedding_design.feasibility_pilot_seeds) & set(
            master_seeds
        ):
            raise ValueError(
                "feasibility-pilot seeds must not equal experiment master seeds"
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
        if self.name == "full_experiment":
            expected_grid = tuple(
                (n, m)
                for n in (100, 300, 1_000)
                for m in (4, 8, 16)
            )
            actual_grid = tuple((setting.n, setting.ba_m) for setting in settings)
            if actual_grid != expected_grid or any(
                setting.er_p
                != (2 * setting.ba_m * (setting.n - setting.ba_m))
                / (setting.n * (setting.n - 1))
                for setting in settings
            ):
                raise ValueError("full_experiment graph grid must equal the frozen grid")
            if (
                self.graph_repetitions != 20
                or self.source_destination_pairs_per_graph != 1_000
                or master_seeds != (1_000_003, 2_000_003, 3_000_003)
                or self.unit_disk_boundary_epsilon != 1e-6
                or self.numerical_tolerance != 1e-12
                or self.max_connected_graph_generation_attempts != 50
                or self.is_provisional
                or provisional_values
            ):
                raise ValueError(
                    "full_experiment settings must equal the Step 13 freeze"
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
                "seed_identity_version",
                "setting_index",
                "setting_label",
                "n",
                "seed_identity_er_p_hex_v3",
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
        distortion_pairs_per_condition = sum(
            len(GRAPH_MODELS)
            * self.graph_repetitions
            * setting.n
            * (setting.n - 1)
            // 2
            for setting in self.parameter_settings
        )
        coordinate_dependent_routing_runs = (
            self.sampled_ordered_pair_count
            * self.approved_embedding_design.coordinate_condition_count
            * COORDINATE_DEPENDENT_ROUTING_METHOD_COUNT
        )
        return {
            "graph_replicates": self.graph_replicate_count,
            "independent_graph_replicates": self.graph_replicate_count,
            "erdos_renyi_graph_replicates": er_graphs,
            "barabasi_albert_graph_replicates": ba_graphs,
            "independent_embedding_families": (
                self.approved_embedding_design.independent_embedding_family_count
            ),
            "coordinate_conditions_per_graph": (
                self.approved_embedding_design.coordinate_condition_count
            ),
            "hydra_embedding_runs": self.graph_replicate_count,
            "mds_base_embedding_runs": self.graph_replicate_count,
            "independent_embedding_family_runs": (
                self.graph_replicate_count
                * self.approved_embedding_design.independent_embedding_family_count
            ),
            "mds_nested_radius_transformations": (
                self.graph_replicate_count
                * len(self.approved_embedding_design.mds_maximum_radii)
            ),
            "embedding_runs": (
                self.graph_replicate_count
                * self.approved_embedding_design.independent_embedding_family_count
            ),
            "sampled_ordered_pairs": self.sampled_ordered_pair_count,
            "routing_methods_per_pair": ROUTING_METHOD_COUNT,
            "dijkstra_routing_runs": self.sampled_ordered_pair_count,
            "actual_dijkstra_executions": self.sampled_ordered_pair_count,
            "coordinate_dependent_routing_executions": (
                coordinate_dependent_routing_runs
            ),
            "euclidean_greedy_routing_runs": (
                self.sampled_ordered_pair_count
                * self.approved_embedding_design.coordinate_condition_count
            ),
            "hyperbolic_greedy_routing_runs": (
                self.sampled_ordered_pair_count
                * self.approved_embedding_design.coordinate_condition_count
            ),
            "repaired_hyperbolic_greedy_routing_runs": (
                self.sampled_ordered_pair_count
                * self.approved_embedding_design.coordinate_condition_count
            ),
            "routing_method_runs": (
                self.sampled_ordered_pair_count * ROUTING_METHOD_COUNT
            ),
            "total_routing_and_benchmark_executions": (
                self.sampled_ordered_pair_count * ROUTING_METHOD_COUNT
            ),
            "distortion_unordered_pairs_per_condition": (
                distortion_pairs_per_condition
            ),
            "distortion_metric_conditions": DISTORTION_METRIC_CONDITION_COUNT,
            "distortion_metric_pair_evaluations": (
                distortion_pairs_per_condition
                * DISTORTION_METRIC_CONDITION_COUNT
            ),
            "distortion_unordered_pairs": (
                distortion_pairs_per_condition
                * DISTORTION_METRIC_CONDITION_COUNT
            ),
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
            "seed_identity_version": SEED_IDENTITY_VERSION,
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
                "approved_design": self.approved_embedding_design.as_dict(),
                "development_force_only": {
                    "algorithm": self.embedding_algorithm,
                    "method": self.embedding_method,
                    "radius": self.embedding_radius,
                    "iterations": self.embedding_iterations,
                    "final_experiment_default": False,
                },
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

    def data_generation_freeze_payload(self) -> dict[str, object]:
        """Return the complete frozen scientific data-generation design."""

        return {
            "payload": "greedy_routing_data_generation_freeze_v1",
            "configuration_schema_version": CONFIGURATION_SCHEMA_VERSION,
            "seed_identity_version": SEED_IDENTITY_VERSION,
            "configuration_name": self.name,
            "graph_design": {
                "models": list(GRAPH_MODELS),
                "simple": True,
                "directed": False,
                "weighted": False,
                "node_ids": "integers_exactly_0_through_n_minus_1",
                "parameter_settings": [
                    setting.as_dict() for setting in self.parameter_settings
                ],
                "accepted_replicates_per_model_n_m": self.graph_repetitions,
                "accepted_graph_count": self.graph_replicate_count,
                "networkx_version": "3.6.1",
                "barabasi_albert": {
                    "generator": "networkx.barabasi_albert_graph",
                    "initial_graph": BA_INITIAL_GRAPH,
                    "connectedness_required": True,
                    "connectedness_violation": "error",
                },
                "erdos_renyi": {
                    "generator": "networkx.gnp_random_graph",
                    "probability_rule": BA_FINITE_DEGREE_MATCH_RULE,
                    "conditioning": ER_CONNECTIVITY_CONDITIONING,
                    "attempt_indices": [0, 49],
                    "maximum_attempts": self.max_connected_graph_generation_attempts,
                    "rejected_disconnected_attempts_are_observations": False,
                    "record": [
                        "all_attempt_seeds",
                        "attempt_count",
                        "rejected_disconnected_count",
                        "realised_edge_count",
                        "realised_average_degree",
                    ],
                },
            },
            "seeds": {
                "master": {
                    "graph_generation": self.graph_generation_master_seed,
                    "embedding_provenance": (
                        self.embedding_initialization_master_seed
                    ),
                    "routing_pairs": self.source_destination_sampling_master_seed,
                },
                "embedding_seed_use": "provenance_only_deterministic_embeddings",
                "seed_derivation_algorithm": SEED_DERIVATION_ALGORITHM,
                "seed_identity_version": SEED_IDENTITY_VERSION,
                "excluded_feasibility_seeds": list(FEASIBILITY_PILOT_SEEDS),
            },
            "ordered_pair_sampling": {
                "count_per_graph": self.source_destination_pairs_per_graph,
                "without_replacement": True,
                "ordered": True,
                "domain": ORDERED_PAIR_SAMPLER_DOMAIN,
                "algorithm": ORDERED_PAIR_SAMPLER_ALGORITHM,
                "random_word": "blake2s_derived_unsigned_64_bit",
                "modulo_bias_control": "reject_before_modulo",
                "duplicate_policy": "reject",
                "pair_order": "acceptance_order",
                "identity_fields": [
                    "pair_master_seed",
                    "seed_identity_version",
                    "graph_identity",
                    "sampler_domain_version",
                    "counter",
                ],
                "index_mapping": {
                    "N": "n*(n-1)",
                    "source": "k//(n-1)",
                    "remainder": "k%(n-1)",
                    "destination": "r if r<source else r+1",
                },
                "reuse": "identical_list_for_all_embeddings_radii_and_methods",
            },
            "embeddings": self.approved_embedding_design.as_dict(),
            "embedding_shared_rules": {
                "distance_matrix": "same_unweighted_all_pairs_shortest_paths",
                "dimension": 2,
                "numeric_type": "float64",
                "node_order": "stable_integer_ascending",
                "routing_tuning": False,
                "eigenvalue_threshold": (
                    "1e-12*max(1,max(abs(all_eigenvalues)))"
                ),
                "eigenvalue_non_finite": "error",
                "eigenvector_convention": (
                    "projector_axis_gram_schmidt_then_largest_magnitude_"
                    "positive_sign_v1"
                ),
                "partial_coordinate_coincidences": "allowed_without_jitter",
                "coincidence_metadata": [
                    "groups",
                    "affected_vertices",
                    "coincident_unordered_pairs",
                ],
                "complete_coordinate_collapse": "error",
            },
            "routing": {
                "coordinate_conditions": list(
                    self.approved_embedding_design.coordinate_condition_ids
                ),
                "methods_per_condition": [
                    "euclidean_greedy",
                    "poincare_greedy",
                    "one_repair_poincare_greedy",
                ],
                "distance_representation": "actual_not_squared",
                "examine_every_neighbour": True,
                "tie_rule": self.routing_tie_break_rule,
                "tie_tolerance_rule": "absolute_distance_difference_lte_tolerance",
                "attempted_revisit_checked_before_strict_progress": True,
                "strict_progress": (
                    "d_next_target < d_current_target - tolerance"
                ),
                "tolerances": {
                    "poincare_all_conditions": self.numerical_tolerance,
                    "euclidean_hydra": self.numerical_tolerance,
                    "euclidean_mds": "1e-12*(radius/0.95)",
                },
                "defensive_step_limits": {
                    "ordinary": "n",
                    "repaired": "2*n",
                    "limit_reached": "implementation_error",
                },
                "repair": {
                    "trigger": "first_local_minimum_or_attempted_revisit_only",
                    "backtrack": "exactly_one_physical_edge_counted",
                    "information": "route_history_only_no_dijkstra_or_lookahead",
                    "excluded": "failed_branch_and_already_explored_vertices",
                    "selection": "poincare_distance_then_smallest_node_id",
                    "escape_move": "one_move_without_strict_improvement",
                    "after_escape": "resume_strict_poincare_no_second_repair",
                    "initial_failure_types": [
                        "none",
                        "local_minimum",
                        "attempted_revisit",
                    ],
                    "final_failure_types": [
                        "local_minimum",
                        "attempted_revisit",
                        "repair_unavailable_at_source",
                        "no_alternative_after_backtracking",
                        "post_repair_local_minimum",
                        "post_repair_attempted_revisit",
                    ],
                },
            },
            "dijkstra": {
                "actual_execution_per_graph_pair": 1,
                "reuse_across_coordinate_conditions": True,
                "verify_against_prepared_apsp": True,
                "disagreement": "implementation_error",
                "timing_excludes": ["apsp_preparation", "apsp_lookup"],
                "stretch_length": "unweighted_physical_edge_count",
            },
            "distortion": {
                "pairs": "all_unordered_graph_pairs_not_routing_sample",
                "ratio": "geometric_distance/graph_distance",
                "alpha": "sum(q)/sum(q**2)",
                "mean_absolute_relative": "mean(abs(alpha*q-1))",
                "relative_rmse": "sqrt(mean((alpha*q-1)**2))",
                "conditions": [
                    "hydra_euclidean",
                    "hydra_poincare",
                    "base_mds_euclidean_once",
                    "mds_poincare_r050",
                    "mds_poincare_r070",
                    "mds_poincare_r085",
                    "mds_poincare_r095",
                ],
                "mds_euclidean_scaling_reuse": True,
            },
            "outcome_records": {
                "graph_success_fraction": "successful_pairs/1000",
                "routing_failures_are_valid_outcomes": True,
                "never_silently_skip": [
                    "graph",
                    "pair",
                    "method",
                    "coordinate_condition",
                ],
                "stretch_success_only": True,
                "failure_type_primary_denominator": 1000,
                "repair_recovery_zero_denominator": "N/A",
            },
            "error_and_rerun_policy": {
                "unusual_final_observations_removed": False,
                "numerical_or_implementation_error": "stop_affected_run",
                "transient_io_or_process_rerun": (
                    "same_deterministic_graph_only_under_identical_freeze"
                ),
                "shared_change": (
                    "new_fingerprint_and_rerun_all_potentially_affected_graphs"
                ),
                "mix_different_fingerprints": False,
                "record": ["graph_identity", "failure", "action", "run_manifest"],
            },
            "runtime": {
                "status": "descriptive_only",
                "timer": "time.perf_counter_ns",
                "component_timings": [
                    "graph_generation",
                    "apsp",
                    "hydra",
                    "mds_base",
                    "each_mds_radius_transformation",
                    "actual_dijkstra",
                    "each_euclidean_greedy",
                    "each_poincare_greedy",
                    "each_repaired_poincare_greedy",
                ],
                "total_wall_time_separate": True,
                "checkpoint_serialization": {
                    "component_timings": "excluded",
                    "total_wall_time": "included",
                },
                "hypothesis_tests": False,
                "environment_record": [
                    "timer_definition",
                    "hardware",
                    "operating_system",
                    "python",
                    "dependencies",
                ],
            },
            "workload": self.workload_estimate,
        }

    def analysis_plan_freeze_payload(self) -> dict[str, object]:
        """Return the complete frozen estimation and resampling design."""

        return {
            "payload": "greedy_routing_analysis_plan_freeze_v1",
            "configuration_schema_version": CONFIGURATION_SCHEMA_VERSION,
            "estimands": {
                "unit": "graph",
                "success_fraction": "S[g,a,c]=successful_sampled_pairs/1000",
                "within_graph_contrasts": {
                    "poincare_advantage": "S_P-S_E",
                    "repair_improvement": "S_R-S_P",
                },
                "cell_estimate": "equal_weight_mean_of_20_graph_values",
                "marginal_weighting": "equal_weight_each_of_nine_n_m_strata",
                "route_rows_as_independent_observations": False,
                "repair_recovery": (
                    "count(P_fails_and_R_succeeds)/count(P_fails)"
                ),
                "repair_recovery_zero_denominator": "N/A_with_denominator",
                "failure_rate_primary_denominator": 1000,
                "conditional_failure_composition": (
                    "secondary_with_explicit_denominator"
                ),
                "stretch": {
                    "defined": "successful_routes_only",
                    "euclidean_vs_poincare": "common_success_pairs",
                    "ordinary_vs_repaired_substantive_common_success": False,
                    "repaired_newly_recovered_pairs": (
                        "report_with_denominator"
                    ),
                    "method_specific": "success_conditioned_descriptive",
                },
            },
            "statistical_method": {
                "orientation": "estimation_focused",
                "null_hypothesis_significance_testing": False,
                "holm_correction": False,
                "effect_sizes": "prespecified",
                "confidence_intervals": ANALYSIS_BOOTSTRAP_METHOD,
                "confidence_level": ANALYSIS_BOOTSTRAP_CONFIDENCE_LEVEL,
                "percentile_quantile_rule": (
                    "noninterpolated_nearest_rank_order_statistics_v1"
                ),
                "bootstrap_replicates": ANALYSIS_BOOTSTRAP_REPLICATES,
                "primary_contrast_units": "percentage_points",
                "secondary_exploratory": ["interactions", "correlations"],
            },
            "bootstrap": {
                "cluster": "whole_graph",
                "strata": ["model", "n", "m"],
                "keep_together": [
                    "pairs",
                    "methods",
                    "embeddings",
                    "radii",
                ],
                "reuse_draws_for_paired_comparisons": True,
                "master_seed": self.source_destination_sampling_master_seed,
                "domain": ANALYSIS_BOOTSTRAP_DOMAIN,
                "algorithm": "blake2s_unsigned_64_bit_unbiased_rejection_v1",
                "stratum_size": self.graph_repetitions,
                "identity_fields": [
                    "combined_freeze_hash",
                    "bootstrap_replicate",
                    "model",
                    "n",
                    "m",
                    "draw_position",
                    "rejection_counter",
                ],
                "er_ba_independent": "model_identity_in_stream",
            },
            "model_contrasts": {
                "status": "descriptive_unpaired",
                "within_matching_n_m_coordinate": [
                    "(P-E)_BA-(P-E)_ER",
                    "(R-P)_BA-(R-P)_ER",
                ],
                "resampling": "independent_ER_and_BA_graph_strata",
            },
            "network_property_associations": {
                "status": "exploratory_non_causal",
                "separate_by": ["graph_family", "coordinate_condition"],
                "raw_pooled_spearman_across_n_m": False,
                "method": (
                    "rank_both_residualise_on_categorical_n_m_indicators_"
                    "correlate_residuals"
                ),
                "bootstrap": "graphs_within_n_m_strata",
                "embedding_distortions": (
                    "separate_euclidean_and_poincare_coordinate_diagnostics"
                ),
            },
        }

    @property
    def data_generation_hash(self) -> str:
        return canonical_fingerprint_hash(self.data_generation_freeze_payload())

    @property
    def analysis_plan_hash(self) -> str:
        return canonical_fingerprint_hash(self.analysis_plan_freeze_payload())

    @property
    def combined_freeze_payload(self) -> dict[str, object]:
        return {
            "payload": "greedy_routing_combined_methodology_freeze_v1",
            "configuration_schema_version": CONFIGURATION_SCHEMA_VERSION,
            "seed_identity_version": SEED_IDENTITY_VERSION,
            "canonical_json_version": CANONICAL_JSON_VERSION,
            "hash_algorithm": FINGERPRINT_ALGORITHM,
            "data_generation_hash": self.data_generation_hash,
            "analysis_plan_hash": self.analysis_plan_hash,
        }

    @property
    def combined_freeze_hash(self) -> str:
        return canonical_fingerprint_hash(self.combined_freeze_payload)

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
            SEED_IDENTITY_VERSION,
            setting_index,
            setting.label,
            setting.n,
            setting.seed_identity_er_p_hex,
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


def audit_feasibility_pilot_seed_collisions(
    config: ExperimentConfig,
) -> tuple[int, ...]:
    """Return reserved pilot seeds reused by any configured experiment stream."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    configured_seeds = {use.seed for use in iter_seed_uses(config)}
    return tuple(
        seed
        for seed in config.approved_embedding_design.feasibility_pilot_seeds
        if seed in configured_seeds
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
    approved_embedding_design=APPROVED_EMBEDDING_DESIGN,
    expected_degree_match_tolerance=1e-12,
    max_connected_graph_generation_attempts=25,
    routing_tie_break_rule=SMALLEST_NODE_ID,
    routing_tie_break_description=TIE_BREAK_DESCRIPTION,
    is_provisional=False,
    workload_note=(
        "Debug-only settings: 8 independent graph replicates and 200 sampled "
        "ordered pairs. Approved-design workload counts Hydra and one base MDS "
        "embedding per graph, four nested MDS transformations, Dijkstra once "
        "per pair, and three greedy methods under five coordinate conditions."
    ),
)


FULL_EXPERIMENT_CONFIG = ExperimentConfig(
    name="full_experiment",
    parameter_settings=tuple(
        make_degree_matched_parameters(
            n=n,
            ba_m=ba_m,
            label=f"full_n{n}_m{ba_m}",
            provisional=False,
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
    approved_embedding_design=APPROVED_EMBEDDING_DESIGN,
    expected_degree_match_tolerance=1e-12,
    max_connected_graph_generation_attempts=50,
    routing_tie_break_rule=SMALLEST_NODE_ID,
    routing_tie_break_description=TIE_BREAK_DESCRIPTION,
    is_provisional=False,
    provisional_values=(),
    workload_note=(
        "Frozen paper workload: 360 graph replicates and 360,000 sampled "
        "ordered pairs. MDS radii are nested transformations, not graph "
        "replicates. Dijkstra runs once per pair and each of three greedy "
        "methods runs under five coordinate conditions."
    ),
)


CONFIGURATIONS = MappingProxyType(
    {
        DEVELOPMENT_CONFIG.name: DEVELOPMENT_CONFIG,
        FULL_EXPERIMENT_CONFIG.name: FULL_EXPERIMENT_CONFIG,
    }
)

DATA_GENERATION_HASH = FULL_EXPERIMENT_CONFIG.data_generation_hash
ANALYSIS_PLAN_HASH = FULL_EXPERIMENT_CONFIG.analysis_plan_hash
COMBINED_FREEZE_HASH = FULL_EXPERIMENT_CONFIG.combined_freeze_hash
FREEZE_HASHES = MappingProxyType(
    {
        "data_generation": DATA_GENERATION_HASH,
        "analysis_plan": ANALYSIS_PLAN_HASH,
        "combined": COMBINED_FREEZE_HASH,
    }
)

if audit_feasibility_pilot_seed_collisions(FULL_EXPERIMENT_CONFIG):
    raise RuntimeError(
        "reserved feasibility-pilot seeds collide with final experiment seeds"
    )


def get_config(name: str) -> ExperimentConfig:
    """Return a named configuration, rejecting unknown names explicitly."""

    try:
        return CONFIGURATIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown configuration {name!r}; choose from {tuple(CONFIGURATIONS)}"
        ) from exc
