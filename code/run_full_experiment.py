"""Production Step 14 runner with safe preflight and per-graph resume.

Importing this module never creates output.  The final 360-graph schedule can
only be executed through the explicit ``run --mode full`` command and the
exact combined-freeze confirmation token.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any

import networkx as nx

from benchmark_experiment_capacity import (
    CapacityProfileError,
    load_capacity_profile,
    volume_identifier,
)
from embedding import calculate_embedding_distortion, prepare_embedding_input
from experiment_checkpoint import (
    CHECKPOINT_DIRECTORY,
    FULL_COORDINATE_CONDITION_IDS,
    RESULT_SCHEMA_VERSION,
    CheckpointAudit,
    CheckpointCompatibilityError,
    GraphCheckpointData,
    audit_run_checkpoints,
    preserve_graph_error,
    publish_graph_checkpoint,
    validate_run_manifest_compatibility,
    write_progress,
    write_run_manifest_once,
)
from experiment_config import (
    ANALYSIS_PLAN_HASH,
    BARABASI_ALBERT,
    COMBINED_FREEZE_HASH,
    DATA_GENERATION_HASH,
    DEVELOPMENT_CONFIG,
    ERDOS_RENYI,
    FEASIBILITY_PILOT_SEEDS,
    FULL_EXPERIMENT_CONFIG,
    MDS_BASE_EMBEDDING_ID,
    ExperimentConfig,
)
from experiment_protocol import (
    build_run_manifest,
    graph_identity,
    sample_ordered_pairs,
)
from graph_generation import (
    GeneratedGraph,
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
    generate_graph,
)
from hydra_embedding import embed_hydra
from mds_embedding import (
    classical_mds,
    scale_equivariant_euclidean_routing_tolerance,
    transform_mds_to_radius,
)
from network_metrics import prepare_all_pairs_shortest_paths
from poincare_distance import euclidean_distance, poincare_distance
from routing import (
    RoutingResult,
    dijkstra_benchmark,
    euclidean_greedy_route,
    hyperbolic_greedy_route,
    prepare_routing_coordinates,
    repaired_hyperbolic_greedy_route,
)


EXPECTED_DATA_GENERATION_HASH = (
    "d7c37cd573e96a0f7c5178d83721c596e4451a9277fef9591d8a319df89611d7"
)
EXPECTED_ANALYSIS_PLAN_HASH = (
    "a3650d2ad45c935500334fa145df2880702059db480f9eb50f1558e8229045d8"
)
EXPECTED_COMBINED_FREEZE_HASH = (
    "8e002ef20f96a4f66c80440c9734cd28b6c0851a95a7977d5e2b7cf905f7a78a"
)

OUTPUT_SCHEMA_ID = "greedy_routing_per_graph_checkpoint_v1"
RUN_MANIFEST_SCHEMA = "greedy_routing_full_run_manifest_v2"
DEFAULT_OUTPUT_DIRECTORY = "results"
EXECUTION_MODEL = "single_process_sequential_per_graph"
SCIENTIFIC_TIMER = "time.perf_counter_ns"
DEVELOPMENT_FIXTURE_LABEL = (
    "NON-SCIENTIFIC STEP 14 DEVELOPMENT FIXTURE - EXCLUDED FROM FINAL RESULTS"
)

_METHOD_IDS = {
    "euclidean_greedy": "euclidean_greedy",
    "hyperbolic_greedy": "poincare_greedy",
    "repaired_hyperbolic_greedy": "repaired_poincare_greedy",
}
_DISTORTION_ORDER = (
    "hydra_euclidean",
    "hydra_poincare",
    "base_mds_euclidean",
    "mds_poincare_r050",
    "mds_poincare_r070",
    "mds_poincare_r085",
    "mds_poincare_r095",
)


class FullRunAuthorizationError(RuntimeError):
    """Raised before any output when full-run authorization is incomplete."""


class GraphExecutionStageError(RuntimeError):
    """Attach the exact per-graph stage to a fatal implementation error."""

    def __init__(self, stage: str, cause: BaseException):
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")


@dataclass(frozen=True)
class GraphScheduleEntry:
    """One canonical graph in the deterministic production schedule."""

    schedule_index: int
    setting_index: int
    model: str
    n: int
    m: int
    replicate_index: int
    graph_id: str
    configuration_name: str
    setting_label: str

    @property
    def canonical_pair_graph_identity(self) -> str:
        return graph_identity(
            configuration_name=self.configuration_name,
            setting_index=self.setting_index,
            setting_label=self.setting_label,
            model=self.model,
            replicate_index=self.replicate_index,
        )


@dataclass(frozen=True)
class PreflightReport:
    """Read-only full-run preflight report."""

    authorized: bool
    authorization_reasons: tuple[str, ...]
    output_root: Path
    run_root: Path
    schedule_ids: tuple[str, ...]
    run_manifest: Mapping[str, object]
    checkpoint_audit: CheckpointAudit
    free_disk_bytes: int
    capacity_status: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "authorized": self.authorized,
            "authorization_reasons": list(self.authorization_reasons),
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "free_disk_bytes": self.free_disk_bytes,
            "freeze_hashes": {
                "data_generation": self.run_manifest["data_generation_hash"],
                "analysis_plan": self.run_manifest["analysis_plan_hash"],
                "combined": self.run_manifest["combined_freeze_hash"],
            },
            "git_commit_hash": self.run_manifest["git_commit_hash"],
            "git_working_tree": self.run_manifest["git_working_tree"],
            "source_fingerprint": self.run_manifest["source_fingerprint"],
            "python_version": self.run_manifest["python_version"],
            "operating_system": self.run_manifest["operating_system"],
            "dependency_versions": self.run_manifest["dependency_versions"],
            "workload": self.run_manifest["workload"],
            "scheduled_graph_count": len(self.schedule_ids),
            "schedule_ids": list(self.schedule_ids),
            "complete_graph_count": len(
                self.checkpoint_audit.complete_graph_ids
            ),
            "remaining_graph_count": len(
                self.checkpoint_audit.remaining_graph_ids
            ),
            "resumable": self.checkpoint_audit.resumable,
            "checkpoint_errors": list(self.checkpoint_audit.errors),
            "capacity": (
                dict(self.capacity_status)
                if self.capacity_status is not None
                else None
            ),
            "disk_space_policy": (
                "step15_profile_required_for_full_mode"
            ),
        }


@dataclass(frozen=True)
class _CoordinateCondition:
    condition_id: str
    coordinates: Mapping[int, Sequence[float]]
    metadata: object
    radius: float | None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolved_output_root(output_root: Path | str | None) -> Path:
    root = repository_root()
    declared = (
        root / DEFAULT_OUTPUT_DIRECTORY
        if output_root is None
        else Path(output_root)
    )
    if not declared.is_absolute():
        declared = root / declared
    return declared.resolve(strict=False)


def _resolved_inside(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"path escapes declared output root: {candidate}") from exc
    return candidate


def build_full_schedule(
    config: ExperimentConfig = FULL_EXPERIMENT_CONFIG,
) -> tuple[GraphScheduleEntry, ...]:
    """Build the stable ER-then-BA, n-m-replicate schedule."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    entries: list[GraphScheduleEntry] = []
    for model in (ERDOS_RENYI, BARABASI_ALBERT):
        prefix = "er" if model == ERDOS_RENYI else "ba"
        for setting_index, setting in enumerate(config.parameter_settings):
            for replicate_index in range(config.graph_repetitions):
                graph_id = (
                    f"{prefix}_n{setting.n:04d}_m{setting.ba_m:02d}_"
                    f"rep{replicate_index:03d}"
                )
                entries.append(
                    GraphScheduleEntry(
                        schedule_index=len(entries),
                        setting_index=setting_index,
                        model=model,
                        n=setting.n,
                        m=setting.ba_m,
                        replicate_index=replicate_index,
                        graph_id=graph_id,
                        configuration_name=config.name,
                        setting_label=setting.label,
                    )
                )
    graph_ids = tuple(entry.graph_id for entry in entries)
    if len(graph_ids) != len(set(graph_ids)):
        raise RuntimeError("canonical schedule contains duplicate graph IDs")
    if config is FULL_EXPERIMENT_CONFIG:
        if len(entries) != 360:
            raise RuntimeError("full schedule must contain exactly 360 graphs")
        if entries[0].graph_id != "er_n0100_m04_rep000":
            raise RuntimeError("full schedule first graph identity changed")
        if entries[-1].graph_id != "ba_n1000_m16_rep019":
            raise RuntimeError("full schedule last graph identity changed")
    return tuple(entries)


def _requirements_pins(root: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        if requirement.count("==") != 1:
            raise RuntimeError(f"dependency is not exactly pinned: {requirement}")
        name, expected = requirement.split("==", 1)
        pins[name] = expected
    return pins


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError(f"no existing parent for output root: {path}")
        candidate = candidate.parent
    return candidate


def _run_directory_name(commit_hash: str, *, profile: str) -> str:
    prefix = "final" if profile == "full" else "development"
    return (
        f"{prefix}_{COMBINED_FREEZE_HASH[:12]}_{commit_hash[:12]}"
    )


def build_experiment_run_manifest(
    *,
    output_root: Path,
    schedule: Sequence[GraphScheduleEntry],
    execution_profile: str,
    require_final_scientific_source: bool,
    timestamp: datetime | None = None,
    config: ExperimentConfig = FULL_EXPERIMENT_CONFIG,
) -> dict[str, object]:
    """Build the immutable Step 14 run manifest in memory."""

    if execution_profile not in ("full", "development_fixture"):
        raise ValueError("unsupported execution profile")
    base = build_run_manifest(
        "full_experiment_run",
        repository_root=repository_root(),
        require_final_scientific_source=require_final_scientific_source,
        timestamp=timestamp,
    )
    run_name = _run_directory_name(
        str(base["git_commit_hash"]),
        profile=execution_profile,
    )
    base.update(
        {
            "manifest_schema": RUN_MANIFEST_SCHEMA,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "output_schema": {
                "id": OUTPUT_SCHEMA_ID,
                "version": RESULT_SCHEMA_VERSION,
                "float_encoding": "tagged_float64_hex_and_csv_float64_hex",
                "compression": "gzip_mtime_zero",
                "completion_marker": "COMPLETE.json_written_last",
            },
            "execution_profile": execution_profile,
            "execution_model": EXECUTION_MODEL,
            "scientific_status": (
                "final_scientific_run"
                if execution_profile == "full"
                else DEVELOPMENT_FIXTURE_LABEL
            ),
            "schedule": [entry.graph_id for entry in schedule],
            "schedule_definition": (
                "ER_then_BA; n_ascending; m_ascending; replicate_000_through_019"
                if execution_profile == "full"
                else "two_excluded_seed_fixture_graphs"
            ),
            "workload": (
                config.workload_estimate
                if execution_profile == "full"
                else {
                    "graph_replicates": len(schedule),
                    "sampled_ordered_pairs": 2 * len(schedule),
                    "coordinate_dependent_routing_executions": 30 * len(schedule),
                    "actual_dijkstra_executions": 2 * len(schedule),
                    "distortion_metric_conditions_per_graph": 7,
                }
            ),
            "output_root": str(output_root),
            "run_directory_name": run_name,
            "timer": SCIENTIFIC_TIMER,
            "timing_policy": {
                "scientific_components_exclude_checkpoint_serialization": True,
                "payload_serialization_recorded_separately": True,
                "prepublication_wall_is_not_end_to_end": True,
                "end_to_end_operational_record": (
                    "publication_timings/<graph_id>.json"
                ),
                "end_to_end_endpoint": (
                    "successful atomic checkpoint publication and final "
                    "published-checkpoint validation"
                ),
                "step15_runtime_field": "end_to_end_graph_wall_ns",
                "apsp_is_not_dijkstra": True,
            },
            "checkpoint_policy": {
                "atomic_directory_rename": True,
                "completion_marker_written_last": True,
                "publication_timing_record_required_for_resume": True,
                "existing_checkpoint_overwrite": False,
                "corrupt_or_incomplete_checkpoint": "stop",
            },
            "created_at_utc": base["timestamp_utc"],
        }
    )
    return base


def preflight(
    *,
    mode: str,
    confirmation: str | None,
    output_root: Path | str | None = None,
    timestamp: datetime | None = None,
) -> PreflightReport:
    """Perform a strictly read-only preflight for the production schedule."""

    declared_output = _resolved_output_root(output_root)
    schedule = build_full_schedule()
    manifest = build_experiment_run_manifest(
        output_root=declared_output,
        schedule=schedule,
        execution_profile="full",
        require_final_scientific_source=False,
        timestamp=timestamp,
    )
    current_volume = volume_identifier(declared_output)
    free_bytes = shutil.disk_usage(_nearest_existing_parent(declared_output)).free
    capacity_status: dict[str, object] = {
        "profile_present": False,
        "profile_valid": False,
        "benchmark_volume_identifier": None,
        "current_output_volume_identifier": current_volume,
        "projected_storage_bytes": None,
        "required_free_bytes": None,
        "current_available_bytes": free_bytes,
        "disk_space_pass": False,
        "nominal_projected_runtime_seconds": None,
        "conservative_projected_runtime_seconds": None,
        "error": None,
    }
    capacity_profile: dict[str, object] | None = None
    if mode == "full":
        try:
            capacity_profile = load_capacity_profile(
                expected_volume_identifier=current_volume,
                current_available_bytes=free_bytes,
            )
            capacity_status.update(
                {
                    "profile_present": True,
                    "profile_valid": True,
                    "benchmark_volume_identifier": capacity_profile[
                        "volume_identifier"
                    ],
                    "projected_storage_bytes": capacity_profile[
                        "projected_storage_bytes"
                    ],
                    "required_free_bytes": capacity_profile[
                        "required_free_bytes"
                    ],
                    "disk_space_pass": True,
                    "nominal_projected_runtime_seconds": capacity_profile[
                        "runtime_projection"
                    ]["nominal_projected_runtime_seconds"],
                    "conservative_projected_runtime_seconds": capacity_profile[
                        "runtime_projection"
                    ]["conservative_projected_runtime_seconds"],
                    "profile_sha256": capacity_profile["profile_sha256"],
                }
            )
            manifest["capacity_profile"] = {
                "profile_schema_version": capacity_profile[
                    "profile_schema_version"
                ],
                "profile_sha256": capacity_profile["profile_sha256"],
                "volume_identifier": capacity_profile["volume_identifier"],
                "required_free_bytes": capacity_profile["required_free_bytes"],
            }
        except CapacityProfileError as exc:
            capacity_status["profile_present"] = (
                "missing" not in str(exc).lower()
            )
            capacity_status["error"] = str(exc)
    run_root = _resolved_inside(
        declared_output,
        str(manifest["run_directory_name"]),
    )
    audit = audit_run_checkpoints(
        run_root,
        schedule_ids=tuple(entry.graph_id for entry in schedule),
        expected_run_manifest=manifest,
    )
    reasons: list[str] = []
    if mode != "full":
        reasons.append("production execution requires --mode full")
    if confirmation != COMBINED_FREEZE_HASH:
        reasons.append(
            "production execution requires the exact --confirm-full-run combined hash"
        )
    actual_hashes = (
        DATA_GENERATION_HASH,
        ANALYSIS_PLAN_HASH,
        COMBINED_FREEZE_HASH,
    )
    expected_hashes = (
        EXPECTED_DATA_GENERATION_HASH,
        EXPECTED_ANALYSIS_PLAN_HASH,
        EXPECTED_COMBINED_FREEZE_HASH,
    )
    if actual_hashes != expected_hashes:
        reasons.append("Step 13 freeze hashes do not match approved constants")
    if manifest["git_working_tree"] != "clean":
        reasons.append("final scientific source must have a clean Git working tree")
    if not str(manifest["git_commit_hash"]).strip():
        reasons.append("final scientific source must be committed")
    pins = _requirements_pins(repository_root())
    if manifest["dependency_versions"] != pins:
        reasons.append("installed dependency versions do not match requirements.txt")
    if mode == "full" and capacity_profile is None:
        reasons.append(
            "valid Step 15 capacity profile and sufficient output-volume "
            "space are required"
        )
    if audit.errors:
        reasons.append("run directory contains incompatible or corrupt state")
    return PreflightReport(
        authorized=not reasons,
        authorization_reasons=tuple(reasons),
        output_root=declared_output,
        run_root=run_root,
        schedule_ids=tuple(entry.graph_id for entry in schedule),
        run_manifest=manifest,
        checkpoint_audit=audit,
        free_disk_bytes=free_bytes,
        capacity_status=capacity_status,
    )


def _timed(function: Callable[..., Any], /, *args, **kwargs):
    start = perf_counter_ns()
    value = function(*args, **kwargs)
    return value, perf_counter_ns() - start


def _timed_stage(
    stage: str,
    function: Callable[..., Any],
    /,
    *args,
    **kwargs,
):
    try:
        return _timed(function, *args, **kwargs)
    except GraphExecutionStageError:
        raise
    except BaseException as exc:
        raise GraphExecutionStageError(stage, exc) from exc


def _call_stage(
    stage: str,
    function: Callable[..., Any],
    /,
    *args,
    **kwargs,
):
    try:
        return function(*args, **kwargs)
    except GraphExecutionStageError:
        raise
    except BaseException as exc:
        raise GraphExecutionStageError(stage, exc) from exc


def _basic_network_measurements(graph) -> dict[str, object]:
    n = graph.number_of_nodes()
    degrees = [degree for _, degree in graph.degree()]
    average_degree = sum(degrees) / n
    return {
        "number_of_vertices": n,
        "number_of_edges": graph.number_of_edges(),
        "average_degree": float(average_degree),
        "maximum_degree": int(max(degrees)),
        "population_degree_variance": float(
            sum((degree - average_degree) ** 2 for degree in degrees) / n
        ),
        "average_clustering_coefficient": float(
            nx.average_clustering(graph)
        ),
    }


def _complete_path_measurements(
    measurements: dict[str, object],
    shortest_paths,
) -> dict[str, object]:
    n = int(measurements["number_of_vertices"])
    result = dict(measurements)
    result["diameter"] = shortest_paths.diameter
    result["average_shortest_path_length"] = (
        float(shortest_paths.ordered_distance_sum) / (n * (n - 1))
    )
    return result


def _distortion_record(
    metric_condition_id: str,
    condition_id: str,
    metric: str,
    distortion,
    *,
    radius: float | None,
    runtime_ns: int,
) -> dict[str, object]:
    return {
        "metric_condition_id": metric_condition_id,
        "coordinate_condition_id": condition_id,
        "metric": metric,
        "mds_radius": radius,
        "fitted_scale_alpha": distortion.fitted_scale_alpha,
        "mean_absolute_relative_distortion": (
            distortion.mean_relative_distortion
        ),
        "relative_rmse": distortion.rmse_relative_distortion,
        "unordered_pair_count": distortion.unordered_pair_count,
        "runtime_ns": runtime_ns,
    }


def _routing_record(
    *,
    graph_id: str,
    pair_index: int,
    pair_id: str,
    condition_id: str,
    method_id: str,
    result: RoutingResult,
    dijkstra_length: int,
    runtime_ns: int,
) -> dict[str, object]:
    if result.success and result.route_length < dijkstra_length:
        raise RuntimeError("successful greedy route is shorter than Dijkstra")
    stretch = (
        result.route_length / dijkstra_length if result.success else None
    )
    return {
        "graph_id": graph_id,
        "pair_index": pair_index,
        "pair_id": pair_id,
        "source": result.source,
        "destination": result.destination,
        "coordinate_condition_id": condition_id,
        "method_id": method_id,
        "success": result.success,
        "initial_failure_type": result.initial_failure_type,
        "final_failure_type": result.final_failure_type,
        "repair_attempted": result.repair_attempted,
        "repair_succeeded": result.repair_succeeded,
        "repair_alternative_existed": result.repair_alternative_existed,
        "repair_attempt_count": result.repair_attempt_count,
        "route_length": result.route_length,
        "physical_hop_count": result.route_length,
        "dijkstra_length": dijkstra_length,
        "dijkstra_hop_count": dijkstra_length,
        "stretch": stretch,
        "runtime_ns": runtime_ns,
        "walk": list(result.walk),
        "forwarding_decisions": result.forwarding_decisions,
    }


def _execute_generated_graph(
    *,
    entry: GraphScheduleEntry,
    config: ExperimentConfig,
    generated: GeneratedGraph,
    generation_runtime_ns: int,
    pair_master_seed: int,
    pair_count: int,
    run_manifest: Mapping[str, object],
    graph_wall_start_ns: int,
) -> GraphCheckpointData:
    """Execute all frozen per-graph computations and retain raw records."""

    graph = generated.graph
    expected_nodes = set(range(entry.n))
    if set(graph.nodes) != expected_nodes:
        raise RuntimeError("generated graph node IDs are not exactly 0 through n-1")
    if not nx.is_connected(graph):
        raise RuntimeError("accepted graph is not connected")

    timings: dict[str, int] = {"graph_generation_ns": generation_runtime_ns}
    basic_metrics, timings["network_measurements_ns"] = _timed_stage(
        "network_measurements",
        _basic_network_measurements,
        graph,
    )
    shortest_paths, timings["apsp_preparation_ns"] = _timed_stage(
        "apsp_preparation",
        prepare_all_pairs_shortest_paths,
        graph,
    )
    metrics = _complete_path_measurements(basic_metrics, shortest_paths)
    embedding_input, timings["embedding_input_preparation_ns"] = _timed_stage(
        "embedding_input_preparation",
        prepare_embedding_input,
        graph,
        shortest_paths,
        configuration_fingerprint=config.configuration_fingerprint,
        tolerance=config.numerical_tolerance,
    )
    design = config.approved_embedding_design
    hydra, timings["hydra_embedding_ns"] = _timed_stage(
        "hydra_embedding",
        embed_hydra,
        embedding_input,
        dimension=design.hydra_dimension,
        kappa=design.hydra_kappa,
        centering_tolerance=design.hydra_centering_tolerance,
        centering_max_iterations=design.hydra_centering_max_iterations,
        eigenvalue_tolerance=design.hydra_eigenvalue_tolerance,
        pairwise_isometry_absolute_tolerance=(
            design.hydra_isometry_absolute_tolerance
        ),
        pairwise_isometry_tolerance=design.hydra_isometry_tolerance,
        boundary_roundoff_tolerance=design.hydra_boundary_roundoff_tolerance,
    )
    mds_base, timings["mds_base_embedding_ns"] = _timed_stage(
        "mds_base_embedding",
        classical_mds,
        embedding_input,
        dimension=design.mds_dimension,
        eigenvalue_relative_tolerance=design.mds_eigenvalue_relative_tolerance,
        centroid_tolerance=design.mds_centroid_tolerance,
    )

    conditions: list[_CoordinateCondition] = [
        _CoordinateCondition(
            condition_id="hydra",
            coordinates=hydra.coordinates,
            metadata=hydra.metadata,
            radius=None,
        )
    ]
    embedding_metadata: dict[str, object] = {
        "hydra": hydra.metadata,
        "mds_base": mds_base.metadata,
    }
    for radius, condition_id in zip(
        design.mds_maximum_radii,
        design.mds_condition_ids,
        strict=True,
    ):
        condition, runtime = _timed_stage(
            f"mds_radius_transformation_{condition_id}",
            transform_mds_to_radius,
            mds_base,
            maximum_radius=radius,
            tolerance=config.numerical_tolerance,
        )
        timings[f"mds_radius_{condition_id}_ns"] = runtime
        conditions.append(
            _CoordinateCondition(
                condition_id=condition_id,
                coordinates=condition.coordinates,
                metadata=condition.metadata,
                radius=radius,
            )
        )
        embedding_metadata[condition_id] = condition.metadata
    if tuple(condition.condition_id for condition in conditions) != (
        FULL_COORDINATE_CONDITION_IDS
    ):
        raise RuntimeError("coordinate condition IDs do not match the result schema")

    distortion_records: dict[str, dict[str, object]] = {}

    def measure_distortion(
        metric_id: str,
        condition_id: str,
        coordinates,
        metric: str,
        radius: float | None,
    ):
        distortion, runtime = _timed_stage(
            f"distortion_{metric_id}",
            calculate_embedding_distortion,
            graph,
            coordinates,
            tolerance=config.numerical_tolerance,
            shortest_paths=shortest_paths,
            metric=metric,
        )
        timings[f"distortion_{metric_id}_ns"] = runtime
        distortion_records[metric_id] = _distortion_record(
            metric_id,
            condition_id,
            metric,
            distortion,
            radius=radius,
            runtime_ns=runtime,
        )
        return distortion

    measure_distortion(
        "hydra_euclidean", "hydra", hydra.coordinates, "euclidean", None
    )
    measure_distortion(
        "hydra_poincare", "hydra", hydra.coordinates, "poincare", None
    )
    measure_distortion(
        "base_mds_euclidean",
        MDS_BASE_EMBEDDING_ID,
        mds_base.coordinates,
        "euclidean",
        None,
    )
    for condition in conditions[1:]:
        metric_id = f"mds_poincare_r{int(round(condition.radius * 100)):03d}"
        measure_distortion(
            metric_id,
            condition.condition_id,
            condition.coordinates,
            "poincare",
            condition.radius,
        )
    if tuple(distortion_records) != _DISTORTION_ORDER:
        raise RuntimeError("all seven distortions must execute in frozen order")

    pairs, timings["ordered_pair_sampling_ns"] = _timed_stage(
        "ordered_pair_sampling",
        sample_ordered_pairs,
        graph.nodes,
        pair_count,
        pair_master_seed,
        graph_identity=entry.canonical_pair_graph_identity,
    )
    routing_preparation_start = perf_counter_ns()
    contexts: dict[str, tuple[object, object, float, float]] = {}
    for condition in conditions:
        euclidean_context = _call_stage(
            f"routing_preparation_{condition.condition_id}_euclidean",
            prepare_routing_coordinates,
            graph,
            condition.coordinates,
            euclidean_distance,
            metric_name="euclidean",
        )
        poincare_context = _call_stage(
            f"routing_preparation_{condition.condition_id}_poincare",
            prepare_routing_coordinates,
            graph,
            condition.coordinates,
            poincare_distance,
            metric_name="poincare",
        )
        if tuple(euclidean_context.items()) != tuple(poincare_context.items()):
            raise RuntimeError("routing metrics received different coordinates")
        euclidean_tolerance = (
            config.numerical_tolerance
            if condition.radius is None
            else scale_equivariant_euclidean_routing_tolerance(
                config.numerical_tolerance,
                condition.radius,
            )
        )
        contexts[condition.condition_id] = (
            euclidean_context,
            poincare_context,
            euclidean_tolerance,
            config.numerical_tolerance,
        )
    timings["routing_coordinate_preparation_ns"] = (
        perf_counter_ns() - routing_preparation_start
    )

    dijkstra_records: list[dict[str, object]] = []
    route_records: list[dict[str, object]] = []
    record_construction_ns = 0
    aggregate = {
        "actual_dijkstra_ns": 0,
        "routing_euclidean_greedy_ns": 0,
        "routing_poincare_greedy_ns": 0,
        "routing_repaired_poincare_greedy_ns": 0,
    }
    for pair_index, (source, destination) in enumerate(pairs):
        pair_id = f"{entry.graph_id}:pair:{pair_index:04d}"
        benchmark, runtime = _timed_stage(
            f"dijkstra_pair_{pair_index:04d}",
            dijkstra_benchmark,
            graph,
            source,
            destination,
            expected_shortest_path_length=(
                shortest_paths.distances[source][destination]
            ),
        )
        aggregate["actual_dijkstra_ns"] += runtime
        record_start = perf_counter_ns()
        dijkstra_records.append(
            {
                "graph_id": entry.graph_id,
                "pair_index": pair_index,
                "pair_id": pair_id,
                "source": source,
                "destination": destination,
                "coordinate_condition_id": None,
                "method_id": "dijkstra",
                "success": True,
                "route_length": benchmark.route_length,
                "apsp_length": shortest_paths.distances[source][destination],
                "apsp_agreement": True,
                "runtime_ns": runtime,
                "walk": list(benchmark.walk),
            }
        )
        record_construction_ns += perf_counter_ns() - record_start

        mds_euclidean_reference: tuple[bool, tuple[int, ...], str | None] | None = None
        for condition in conditions:
            (
                euclidean_context,
                poincare_context,
                euclidean_tolerance,
                poincare_tolerance,
            ) = contexts[condition.condition_id]
            methods = (
                (
                    euclidean_greedy_route,
                    euclidean_context,
                    euclidean_tolerance,
                    "routing_euclidean_greedy_ns",
                ),
                (
                    hyperbolic_greedy_route,
                    poincare_context,
                    poincare_tolerance,
                    "routing_poincare_greedy_ns",
                ),
                (
                    repaired_hyperbolic_greedy_route,
                    poincare_context,
                    poincare_tolerance,
                    "routing_repaired_poincare_greedy_ns",
                ),
            )
            for method, context, tolerance, timing_key in methods:
                result, route_runtime = _timed_stage(
                    (
                        f"{timing_key.removeprefix('routing_').removesuffix('_ns')}"
                        f"_{condition.condition_id}_pair_{pair_index:04d}"
                    ),
                    method,
                    graph,
                    context,
                    source,
                    destination,
                    tolerance=tolerance,
                )
                aggregate[timing_key] += route_runtime
                method_id = _METHOD_IDS[result.method]
                if method_id == "euclidean_greedy" and condition.radius is not None:
                    route_identity = (
                        result.success,
                        result.walk,
                        result.final_failure_type,
                    )
                    if mds_euclidean_reference is None:
                        mds_euclidean_reference = route_identity
                    elif route_identity != mds_euclidean_reference:
                        raise RuntimeError(
                            "nested MDS scaling changed Euclidean routing"
                        )
                record_start = perf_counter_ns()
                route_records.append(
                    _routing_record(
                        graph_id=entry.graph_id,
                        pair_index=pair_index,
                        pair_id=pair_id,
                        condition_id=condition.condition_id,
                        method_id=method_id,
                        result=result,
                        dijkstra_length=benchmark.route_length,
                        runtime_ns=route_runtime,
                    )
                )
                record_construction_ns += perf_counter_ns() - record_start
    timings.update(aggregate)
    timings["record_construction_ns"] = record_construction_ns
    if len(dijkstra_records) != pair_count:
        raise RuntimeError("Dijkstra must execute exactly once per sampled pair")
    if len(route_records) != pair_count * 15:
        raise RuntimeError("each pair must execute 15 coordinate-dependent routes")
    timings["scientific_component_total_ns"] = sum(
        value for key, value in timings.items() if key.endswith("_ns")
    )

    generation_metadata = dict(generated.metadata)
    generation_metadata.update(
        {
            "graph_id": entry.graph_id,
            "schedule_index": entry.schedule_index,
            "pair_sampling_graph_identity": (
                entry.canonical_pair_graph_identity
            ),
            "pair_master_seed": pair_master_seed,
            "pair_count": pair_count,
            "embedding_input_fingerprint": embedding_input.input_fingerprint,
        }
    )
    return GraphCheckpointData(
        graph_id=entry.graph_id,
        generation_metadata=generation_metadata,
        edges=tuple(graph.edges()),
        network_metrics=metrics,
        pairs=pairs,
        coordinates={
            condition.condition_id: condition.coordinates
            for condition in conditions
        },
        embedding_metadata=embedding_metadata,
        distortions=tuple(
            distortion_records[metric_id] for metric_id in _DISTORTION_ORDER
        ),
        dijkstra_records=tuple(dijkstra_records),
        route_records=tuple(route_records),
        timings=timings,
        run_manifest=run_manifest,
    )


def execute_graph_entry(
    entry: GraphScheduleEntry,
    *,
    run_manifest: Mapping[str, object],
    config: ExperimentConfig = FULL_EXPERIMENT_CONFIG,
    pair_count: int | None = None,
    pair_master_seed: int | None = None,
    generated_override: GeneratedGraph | None = None,
    generation_runtime_override_ns: int = 0,
) -> tuple[GraphCheckpointData, int]:
    """Execute one graph in memory and return data plus its wall-clock start."""

    graph_wall_start_ns = perf_counter_ns()
    if generated_override is None:
        generated, generation_runtime = _timed_stage(
            "graph_generation",
            generate_graph,
            config,
            entry.setting_index,
            entry.model,
            entry.replicate_index,
        )
    else:
        generated = generated_override
        generation_runtime = generation_runtime_override_ns
    try:
        data = _execute_generated_graph(
            entry=entry,
            config=config,
            generated=generated,
            generation_runtime_ns=generation_runtime,
            pair_master_seed=(
                config.source_destination_sampling_master_seed
                if pair_master_seed is None
                else pair_master_seed
            ),
            pair_count=(
                config.source_destination_pairs_per_graph
                if pair_count is None
                else pair_count
            ),
            run_manifest=run_manifest,
            graph_wall_start_ns=graph_wall_start_ns,
        )
    except GraphExecutionStageError:
        raise
    except BaseException as exc:
        raise GraphExecutionStageError(
            "per_graph_invariant_validation",
            exc,
        ) from exc
    return data, graph_wall_start_ns


def execute_full_run(
    report: PreflightReport,
    *,
    resume: bool,
    graph_executor: Callable[..., tuple[GraphCheckpointData, int]] = (
        execute_graph_entry
    ),
) -> tuple[str, ...]:
    """Execute only missing graphs in canonical order after authorization."""

    if not report.authorized:
        raise FullRunAuthorizationError(
            "; ".join(report.authorization_reasons)
        )
    if report.run_root.exists() and not resume:
        raise FullRunAuthorizationError(
            "the run directory already exists; use --resume after a clean audit"
        )
    report.output_root.mkdir(parents=True, exist_ok=True)
    write_run_manifest_once(report.run_root, report.run_manifest)
    schedule = build_full_schedule()
    audit = audit_run_checkpoints(
        report.run_root,
        schedule_ids=report.schedule_ids,
        expected_run_manifest=report.run_manifest,
    )
    if not audit.resumable:
        raise CheckpointCompatibilityError("; ".join(audit.errors))
    completed = list(audit.complete_graph_ids)
    remaining = set(audit.remaining_graph_ids)
    write_progress(
        report.run_root,
        schedule_ids=report.schedule_ids,
        complete_graph_ids=completed,
    )
    for entry in schedule:
        if entry.graph_id not in remaining:
            continue
        try:
            data, wall_start = graph_executor(
                entry,
                run_manifest=report.run_manifest,
                config=FULL_EXPERIMENT_CONFIG,
            )
            publish_graph_checkpoint(
                report.run_root,
                data,
                graph_wall_start_ns=wall_start,
            )
        except BaseException as exc:
            graph_parent = report.run_root / CHECKPOINT_DIRECTORY
            already_preserved = (
                graph_parent.exists()
                and any(
                    entry.graph_id in child.name and ".tmp-" in child.name
                    for child in graph_parent.iterdir()
                )
            )
            if not already_preserved:
                preserve_graph_error(
                    report.run_root,
                    graph_id=entry.graph_id,
                    stage=getattr(exc, "stage", "per_graph_execution"),
                    exception=exc,
                    run_manifest=report.run_manifest,
                )
            raise
        finally:
            if "data" in locals():
                del data
            gc.collect()
        completed.append(entry.graph_id)
        write_progress(
            report.run_root,
            schedule_ids=report.schedule_ids,
            complete_graph_ids=completed,
        )
    return tuple(completed)


def _development_fixture_entries() -> tuple[GraphScheduleEntry, ...]:
    setting = DEVELOPMENT_CONFIG.parameter_settings[0]
    return (
        GraphScheduleEntry(
            schedule_index=0,
            setting_index=0,
            model=ERDOS_RENYI,
            n=setting.n,
            m=setting.ba_m,
            replicate_index=0,
            graph_id="dev_er_n0030_m02_rep000",
            configuration_name="excluded_step14_fixture",
            setting_label="excluded_fixture_n30_m2",
        ),
        GraphScheduleEntry(
            schedule_index=1,
            setting_index=0,
            model=BARABASI_ALBERT,
            n=setting.n,
            m=setting.ba_m,
            replicate_index=0,
            graph_id="dev_ba_n0030_m02_rep000",
            configuration_name="excluded_step14_fixture",
            setting_label="excluded_fixture_n30_m2",
        ),
    )


def run_development_fixture() -> dict[str, object]:
    """Run a disposable two-graph checkpoint/resume/corruption fixture."""

    entries = _development_fixture_entries()
    setting = DEVELOPMENT_CONFIG.parameter_settings[0]
    with TemporaryDirectory(prefix="greedy-routing-step14-") as temporary:
        output_root = Path(temporary).resolve()
        manifest = build_experiment_run_manifest(
            output_root=output_root,
            schedule=entries,
            execution_profile="development_fixture",
            require_final_scientific_source=False,
            config=DEVELOPMENT_CONFIG,
        )
        run_root = output_root / str(manifest["run_directory_name"])
        write_run_manifest_once(run_root, manifest)
        validations = []
        for index, entry in enumerate(entries):
            graph_seed = FEASIBILITY_PILOT_SEEDS[index]
            start = perf_counter_ns()
            if entry.model == ERDOS_RENYI:
                generated = generate_connected_erdos_renyi(
                    n=entry.n,
                    p=setting.er_p,
                    graph_seed=graph_seed,
                    replicate_index=0,
                    max_attempts=25,
                    p_exact_numerator=setting.er_probability_numerator,
                    p_exact_denominator=setting.er_probability_denominator,
                )
            else:
                generated = generate_connected_barabasi_albert(
                    n=entry.n,
                    m=entry.m,
                    graph_seed=graph_seed,
                    replicate_index=0,
                )
            generation_runtime = perf_counter_ns() - start
            data, wall_start = execute_graph_entry(
                entry,
                run_manifest=manifest,
                config=DEVELOPMENT_CONFIG,
                pair_count=2,
                pair_master_seed=FEASIBILITY_PILOT_SEEDS[index + 2],
                generated_override=generated,
                generation_runtime_override_ns=generation_runtime,
            )
            validations.append(
                publish_graph_checkpoint(
                    run_root,
                    data,
                    graph_wall_start_ns=wall_start,
                )
            )
        schedule_ids = tuple(entry.graph_id for entry in entries)
        audit = audit_run_checkpoints(
            run_root,
            schedule_ids=schedule_ids,
            expected_run_manifest=manifest,
        )
        if not audit.resumable or audit.remaining_graph_ids:
            raise RuntimeError("development fixture resume audit failed")

        # Exercise corruption detection only in a disposable sibling copy.
        corrupt_root = output_root / "corruption_fixture"
        shutil.copytree(run_root, corrupt_root)
        corrupt_file = (
            corrupt_root
            / CHECKPOINT_DIRECTORY
            / entries[0].graph_id
            / "routes.jsonl.gz"
        )
        with corrupt_file.open("ab") as stream:
            stream.write(b"corruption")
        corrupt_audit = audit_run_checkpoints(
            corrupt_root,
            schedule_ids=schedule_ids,
            expected_run_manifest=manifest,
        )
        if corrupt_audit.resumable:
            raise RuntimeError("development fixture did not detect corruption")

        mismatch = dict(manifest)
        mismatch["source_fingerprint"] = "0" * 64
        mismatch_detected = False
        try:
            validate_run_manifest_compatibility(manifest, mismatch)
        except CheckpointCompatibilityError:
            mismatch_detected = True
        if not mismatch_detected:
            raise RuntimeError("development fixture did not detect identity mismatch")

        simulated_error = preserve_graph_error(
            output_root / str(manifest["run_directory_name"]),
            graph_id="dev_er_n0030_m02_rep001",
            stage="simulated_mid_graph_failure",
            exception=RuntimeError("intentional development fixture failure"),
            run_manifest=manifest,
        )
        if (simulated_error / "COMPLETE.json").exists():
            raise RuntimeError("failed fixture checkpoint has a completion marker")
        return {
            "label": DEVELOPMENT_FIXTURE_LABEL,
            "excluded_seeds": list(FEASIBILITY_PILOT_SEEDS[:4]),
            "graph_count": len(validations),
            "pair_count": 2 * len(validations),
            "route_count": sum(
                validation.counts["route_records"] for validation in validations
            ),
            "resume_skipped_complete_graphs": len(audit.complete_graph_ids),
            "corruption_detected": bool(corrupt_audit.errors),
            "identity_mismatch_detected": mismatch_detected,
            "mid_graph_failure_preserved_without_complete": True,
            "temporary_output_cleaned_on_return": True,
        }


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 14 production full-experiment runner"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    for name in ("preflight", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--mode", choices=("development", "full"), required=True)
        command.add_argument("--confirm-full-run")
        command.add_argument("--output-root")
        if name == "run":
            command.add_argument(
                "--resume",
                action="store_true",
                help="resume only after complete checkpoint validation",
            )

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output-root")
    subparsers.add_parser("development-fixture")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "development-fixture":
        _print_json(run_development_fixture())
        return 0
    if args.operation == "audit":
        report = preflight(
            mode="development",
            confirmation=None,
            output_root=args.output_root,
        )
        _print_json(
            {
                "operation": "read_only_checkpoint_audit",
                "run_root": str(report.run_root),
                "complete_graph_ids": list(
                    report.checkpoint_audit.complete_graph_ids
                ),
                "remaining_graph_ids": list(
                    report.checkpoint_audit.remaining_graph_ids
                ),
                "errors": list(report.checkpoint_audit.errors),
                "resumable": report.checkpoint_audit.resumable,
            }
        )
        return 0 if report.checkpoint_audit.resumable else 1
    report = preflight(
        mode=args.mode,
        confirmation=args.confirm_full_run,
        output_root=args.output_root,
    )
    if args.operation == "preflight":
        _print_json(report.as_dict())
        return 0 if report.authorized else 2
    if not report.authorized:
        _print_json(report.as_dict())
        return 2
    completed = execute_full_run(report, resume=args.resume)
    _print_json(
        {
            "status": "complete",
            "run_root": str(report.run_root),
            "complete_graph_count": len(completed),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
