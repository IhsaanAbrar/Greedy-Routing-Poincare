"""Excluded Step 15 runtime-and-storage capacity benchmark.

This module measures operational capacity only.  It executes six explicitly
excluded graph fixtures through the frozen Step 14 per-graph pipeline, removes
all raw checkpoints after successful validation, and writes one concise
machine-readable profile.  It never schedules a final experiment graph.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from math import ceil, isfinite
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from time import monotonic, perf_counter_ns
from typing import Callable, Mapping, Sequence

from experiment_checkpoint import (
    CHECKPOINT_DIRECTORY,
    PROGRESS_FILENAME,
    PUBLICATION_TIMING_DIRECTORY,
    RESULT_SCHEMA_VERSION,
    RUN_MANIFEST_FILENAME,
    decode_json_value,
    publish_graph_checkpoint,
    validate_graph_checkpoint,
    validate_publication_timing_record,
    write_progress,
    write_run_manifest_once,
)
from experiment_config import (
    ANALYSIS_PLAN_HASH,
    BARABASI_ALBERT,
    COMBINED_FREEZE_HASH,
    DATA_GENERATION_HASH,
    ERDOS_RENYI,
    FEASIBILITY_PILOT_SEEDS,
    FULL_EXPERIMENT_CONFIG,
)
from graph_generation import (
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
)


CAPACITY_PROFILE_SCHEMA_VERSION = 1
CAPACITY_PROFILE_FILENAME = "step15_capacity_profile.json"
CAPACITY_PROFILE_LABEL = (
    "NON-SCIENTIFIC STEP 15 CAPACITY BENCHMARK - EXCLUDED FROM ANALYSIS"
)
BENCHMARK_PAIR_COUNT = 1_000
BENCHMARK_REPETITIONS = 3
WARMUP_REPETITION = 0
MEASURED_REPETITIONS = (1, 2)
FINAL_GRAPHS_PER_MODEL_N_CELL = 60
GIB_BYTES = 1_073_741_824
METADATA_ALLOWANCE_NUMERATOR = 1
METADATA_ALLOWANCE_DENOMINATOR = 100
CONSERVATIVE_RUNTIME_NUMERATOR = 3
CONSERVATIVE_RUNTIME_DENOMINATOR = 2
WATCHDOG_SECONDS_BY_N = {100: 120, 300: 300, 1_000: 600}
PERFORMANCE_SOURCE_FINGERPRINT_SCHEMA = (
    "step15_performance_source_content_v1"
)
PERFORMANCE_SOURCE_PATHS = (
    "requirements.txt",
    "code/benchmark_experiment_capacity.py",
    "code/embedding.py",
    "code/experiment_checkpoint.py",
    "code/experiment_config.py",
    "code/experiment_protocol.py",
    "code/graph_generation.py",
    "code/hydra_embedding.py",
    "code/mds_embedding.py",
    "code/network_metrics.py",
    "code/poincare_distance.py",
    "code/routing.py",
    "code/run_full_experiment.py",
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

_FORBIDDEN_SCIENTIFIC_PROFILE_KEYS = {
    "routing_success",
    "routing_successes",
    "success_rate",
    "stretch",
    "routing_failure",
    "routing_failures",
    "failure_rate",
    "scientific_outcomes",
}


class CapacityProfileError(RuntimeError):
    """Raised when a capacity profile or benchmark invariant is invalid."""


@dataclass(frozen=True)
class CapacityGraphSpec:
    """One excluded high-density capacity fixture."""

    model: str
    n: int
    m: int
    seed: int
    setting_index: int
    graph_id: str

    @property
    def cell_id(self) -> str:
        prefix = "er" if self.model == ERDOS_RENYI else "ba"
        return f"{prefix}_n{self.n:04d}"


BENCHMARK_GRAPH_SPECS = (
    CapacityGraphSpec(
        ERDOS_RENYI,
        100,
        16,
        4_000_003,
        2,
        "capacity_er_n0100_m16_seed4000003",
    ),
    CapacityGraphSpec(
        ERDOS_RENYI,
        300,
        16,
        4_000_019,
        5,
        "capacity_er_n0300_m16_seed4000019",
    ),
    CapacityGraphSpec(
        ERDOS_RENYI,
        1_000,
        16,
        4_000_037,
        8,
        "capacity_er_n1000_m16_seed4000037",
    ),
    CapacityGraphSpec(
        BARABASI_ALBERT,
        100,
        16,
        4_000_063,
        2,
        "capacity_ba_n0100_m16_seed4000063",
    ),
    CapacityGraphSpec(
        BARABASI_ALBERT,
        300,
        16,
        4_000_099,
        5,
        "capacity_ba_n0300_m16_seed4000099",
    ),
    CapacityGraphSpec(
        BARABASI_ALBERT,
        1_000,
        16,
        4_000_121,
        8,
        "capacity_ba_n1000_m16_seed4000121",
    ),
)
CANARY_GRAPH_SPECS_BY_N = {
    100: BENCHMARK_GRAPH_SPECS[0],
    1_000: BENCHMARK_GRAPH_SPECS[2],
}
CANARY_GRAPH_SPEC = CANARY_GRAPH_SPECS_BY_N[100]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_profile_path() -> Path:
    return Path(__file__).resolve().with_name(CAPACITY_PROFILE_FILENAME)


def nearest_existing_parent(path: Path | str) -> Path:
    candidate = Path(path).resolve(strict=False)
    while not candidate.exists():
        if candidate.parent == candidate:
            raise CapacityProfileError(f"no existing parent for path: {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate.resolve(strict=True)


def volume_identifier(path: Path | str) -> str:
    """Return a stable, non-sensitive identifier for the containing volume."""

    existing = nearest_existing_parent(path)
    if os.name == "nt":
        drive = existing.drive.upper()
        if not drive:
            raise CapacityProfileError("Windows volume has no drive identifier")
        return f"windows_drive:{drive}"
    return f"posix_device:{existing.stat().st_dev}"


def filesystem_type(path: Path | str) -> str | None:
    """Return the filesystem type when the standard library can obtain it."""

    existing = nearest_existing_parent(path)
    if os.name != "nt":
        return None
    try:
        import ctypes

        root = f"{existing.drive}\\"
        filesystem_name = ctypes.create_unicode_buffer(261)
        result = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(root),
            None,
            0,
            None,
            None,
            None,
            filesystem_name,
            len(filesystem_name),
        )
        return filesystem_name.value if result else None
    except (AttributeError, OSError):
        return None


def available_bytes(path: Path | str) -> int:
    return int(shutil.disk_usage(nearest_existing_parent(path)).free)


def _canonical_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("capacity profiles reject NaN and infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("capacity profile object keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise ValueError(f"unsupported capacity profile value: {type(value).__name__}")


def canonical_profile_bytes(profile: Mapping[str, object]) -> bytes:
    return json.dumps(
        _canonical_value(dict(profile)),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def profile_sha256(profile: Mapping[str, object]) -> str:
    payload = dict(profile)
    payload.pop("profile_sha256", None)
    return sha256(canonical_profile_bytes(payload)).hexdigest()


def performance_source_manifest(
    root: Path | str | None = None,
) -> dict[str, object]:
    """Describe performance-relevant contents without using Git identity."""

    project_root = (
        repository_root()
        if root is None
        else Path(root).resolve(strict=True)
    )
    files: dict[str, str] = {}
    for relative in PERFORMANCE_SOURCE_PATHS:
        path = (project_root / relative).resolve(strict=True)
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise CapacityProfileError(
                f"performance source path escapes repository: {relative}"
            ) from exc
        if not path.is_file():
            raise CapacityProfileError(
                f"performance source file is missing: {relative}"
            )
        files[relative] = sha256(path.read_bytes()).hexdigest()
    return {
        "schema": PERFORMANCE_SOURCE_FINGERPRINT_SCHEMA,
        "files": files,
        "dependency_versions": _dependency_versions(),
    }


def performance_source_fingerprint(
    root: Path | str | None = None,
) -> str:
    return sha256(
        canonical_profile_bytes(performance_source_manifest(root))
    ).hexdigest()


def benchmark_code_content_fingerprint(
    root: Path | str | None = None,
) -> str:
    manifest = performance_source_manifest(root)
    return str(
        manifest["files"]["code/benchmark_experiment_capacity.py"]
    )


def _profile_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key).lower())
            keys.update(_profile_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_profile_keys(item))
    return keys


def _profile_strings(value: object) -> tuple[str, ...]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            strings.extend(_profile_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            strings.extend(_profile_strings(item))
    return tuple(strings)


def validate_capacity_profile(
    profile: Mapping[str, object],
    *,
    expected_volume_identifier: str | None = None,
    current_available_bytes: int | None = None,
) -> dict[str, object]:
    """Validate identity, integrity, volume, and disk requirements."""

    if not isinstance(profile, Mapping):
        raise CapacityProfileError("capacity profile must be an object")
    canonical_profile_bytes(profile)
    if profile.get("profile_schema_version") != CAPACITY_PROFILE_SCHEMA_VERSION:
        raise CapacityProfileError("capacity profile schema mismatch")
    if profile.get("non_scientific") is not True:
        raise CapacityProfileError("capacity profile must be non-scientific")
    if profile.get("excluded_from_analysis") is not True:
        raise CapacityProfileError("capacity profile must be excluded from analysis")
    if profile.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise CapacityProfileError("capacity result-schema mismatch")
    hashes = profile.get("step13_hashes")
    expected_hashes = {
        "data_generation": DATA_GENERATION_HASH,
        "analysis_plan": ANALYSIS_PLAN_HASH,
        "combined": COMBINED_FREEZE_HASH,
    }
    if hashes != expected_hashes:
        raise CapacityProfileError("capacity profile Step 13 hashes mismatch")
    current_source = performance_source_manifest()
    if profile.get("performance_source_manifest") != current_source:
        raise CapacityProfileError(
            "capacity performance-source content fingerprint mismatch"
        )
    current_fingerprint = performance_source_fingerprint()
    if (
        profile.get("performance_source_content_fingerprint")
        != current_fingerprint
    ):
        raise CapacityProfileError(
            "capacity performance-source content fingerprint mismatch"
        )
    if (
        profile.get("benchmark_code_content_fingerprint")
        != benchmark_code_content_fingerprint()
    ):
        raise CapacityProfileError(
            "capacity benchmark-code content fingerprint mismatch"
        )
    _validate_capacity_profile_workload(profile)
    stored_hash = profile.get("profile_sha256")
    if not isinstance(stored_hash, str) or stored_hash != profile_sha256(profile):
        raise CapacityProfileError("capacity profile SHA-256 mismatch")
    benchmark_volume = profile.get("volume_identifier")
    if not isinstance(benchmark_volume, str) or not benchmark_volume:
        raise CapacityProfileError("capacity profile volume identifier is invalid")
    if (
        expected_volume_identifier is not None
        and benchmark_volume != expected_volume_identifier
    ):
        raise CapacityProfileError(
            "output volume differs from the benchmarked capacity volume"
        )
    required = profile.get("required_free_bytes")
    if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
        raise CapacityProfileError("capacity required_free_bytes is invalid")
    if (
        current_available_bytes is not None
        and current_available_bytes < required
    ):
        raise CapacityProfileError(
            "available output-volume space is below required_free_bytes"
        )
    if _profile_keys(profile) & _FORBIDDEN_SCIENTIFIC_PROFILE_KEYS:
        raise CapacityProfileError(
            "capacity profile contains forbidden scientific outcomes"
        )
    home = str(Path.home().resolve(strict=False))
    if home and any(
        home.lower() in value.lower() for value in _profile_strings(profile)
    ):
        raise CapacityProfileError(
            "capacity profile contains a full home-directory path"
        )
    return dict(profile)


def load_capacity_profile(
    path: Path | str | None = None,
    *,
    expected_volume_identifier: str | None = None,
    current_available_bytes: int | None = None,
) -> dict[str, object]:
    profile_path = default_profile_path() if path is None else Path(path)
    if not profile_path.is_file():
        raise CapacityProfileError(f"capacity profile is missing: {profile_path}")
    try:
        value = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapacityProfileError("capacity profile JSON is invalid") from exc
    return validate_capacity_profile(
        value,
        expected_volume_identifier=expected_volume_identifier,
        current_available_bytes=current_available_bytes,
    )


def write_capacity_profile(
    profile: Mapping[str, object],
    path: Path | str | None = None,
) -> Path:
    destination = (
        default_profile_path() if path is None else Path(path).resolve(strict=False)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(profile)
    payload["profile_sha256"] = profile_sha256(payload)
    validate_capacity_profile(payload)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def validate_benchmark_specs(
    final_schedule_ids: Sequence[str],
) -> tuple[CapacityGraphSpec, ...]:
    specs = BENCHMARK_GRAPH_SPECS
    if len(specs) != 6:
        raise CapacityProfileError("capacity benchmark requires exactly six graphs")
    if tuple(spec.seed for spec in specs) != FEASIBILITY_PILOT_SEEDS:
        raise CapacityProfileError("capacity fixtures must use all six excluded seeds")
    if any(spec.m != 16 for spec in specs):
        raise CapacityProfileError("capacity fixtures must all use m=16")
    if len({spec.graph_id for spec in specs}) != len(specs):
        raise CapacityProfileError("capacity graph identities must be unique")
    overlap = set(final_schedule_ids) & {spec.graph_id for spec in specs}
    if overlap:
        raise CapacityProfileError(
            f"capacity graph identity overlaps final schedule: {sorted(overlap)}"
        )
    for spec in specs:
        setting = FULL_EXPERIMENT_CONFIG.parameter_settings[spec.setting_index]
        if (setting.n, setting.ba_m) != (spec.n, spec.m):
            raise CapacityProfileError("capacity setting index is inconsistent")
    return specs


def summarise_capacity_cells(
    repetitions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for spec in BENCHMARK_GRAPH_SPECS:
        records = [
            record
            for record in repetitions
            if record.get("graph_id") == spec.graph_id
        ]
        if tuple(record.get("repetition") for record in records) != (0, 1, 2):
            raise CapacityProfileError(
                f"capacity cell {spec.cell_id} lacks repetitions 0, 1, 2"
            )
        measured = [
            record
            for record in records
            if record.get("repetition") in MEASURED_REPETITIONS
            and record.get("included_in_projection") is True
        ]
        if len(measured) != 2 or records[0].get("included_in_projection") is not False:
            raise CapacityProfileError(
                f"capacity cell {spec.cell_id} repetition policy is invalid"
            )
        runtimes = sorted(
            int(record["end_to_end_graph_wall_ns"]) for record in measured
        )
        median_runtime = (runtimes[0] + runtimes[1]) / 2
        checkpoint_sizes = [
            int(record["complete_graph_checkpoint_bytes"]) for record in measured
        ]
        publication_timing_sizes = [
            int(record["run_level_overhead_breakdown_bytes"][
                "publication_timing"
            ])
            for record in measured
        ]
        fixed_overhead_sizes = [
            int(record["run_level_overhead_breakdown_bytes"]["run_manifest"])
            + int(record["run_level_overhead_breakdown_bytes"]["progress"])
            for record in measured
        ]
        cells.append(
            {
                "cell_id": spec.cell_id,
                "model": spec.model,
                "n": spec.n,
                "m": spec.m,
                "graph_id": spec.graph_id,
                "measured_repetitions": list(MEASURED_REPETITIONS),
                "measured_end_to_end_graph_wall_ns": runtimes,
                "median_end_to_end_graph_wall_ns": median_runtime,
                "maximum_end_to_end_graph_wall_ns": max(runtimes),
                "minimum_end_to_end_graph_wall_ns": min(runtimes),
                "range_end_to_end_graph_wall_ns": max(runtimes) - min(runtimes),
                "relative_range_over_median": (
                    (max(runtimes) - min(runtimes)) / median_runtime
                    if median_runtime
                    else 0.0
                ),
                "maximum_complete_graph_checkpoint_bytes": max(checkpoint_sizes),
                "maximum_publication_timing_bytes": max(
                    publication_timing_sizes
                ),
                "maximum_fixed_run_overhead_bytes": max(fixed_overhead_sizes),
            }
        )
    return cells


def project_runtime(cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(cells) != 6:
        raise CapacityProfileError("runtime projection requires six cells")
    contributions = []
    for cell in cells:
        contribution = (
            float(cell["median_end_to_end_graph_wall_ns"])
            * FINAL_GRAPHS_PER_MODEL_N_CELL
        )
        contributions.append(
            {
                "cell_id": cell["cell_id"],
                "graphs_represented": FINAL_GRAPHS_PER_MODEL_N_CELL,
                "projected_runtime_ns": contribution,
                "projected_runtime_seconds": contribution / 1_000_000_000,
            }
        )
    nominal_ns = sum(
        float(item["projected_runtime_ns"]) for item in contributions
    )
    conservative_ns = (
        nominal_ns
        * CONSERVATIVE_RUNTIME_NUMERATOR
        / CONSERVATIVE_RUNTIME_DENOMINATOR
    )
    return {
        "nominal_projection_formula": (
            "sum(cell_median_end_to_end_graph_wall_ns*60)"
        ),
        "cell_contributions": contributions,
        "nominal_projected_runtime_ns": nominal_ns,
        "nominal_projected_runtime_seconds": nominal_ns / 1_000_000_000,
        "nominal_projected_runtime_hours": nominal_ns / 3_600_000_000_000,
        "nominal_projected_runtime_days": nominal_ns / 86_400_000_000_000,
        "conservative_multiplier": 1.5,
        "conservative_projected_runtime_ns": conservative_ns,
        "conservative_projected_runtime_seconds": conservative_ns / 1_000_000_000,
        "conservative_projected_runtime_hours": conservative_ns
        / 3_600_000_000_000,
        "conservative_projected_runtime_days": conservative_ns
        / 86_400_000_000_000,
    }


def project_storage(cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(cells) != 6:
        raise CapacityProfileError("storage projection requires six cells")
    contributions = []
    for cell in cells:
        checkpoint_bytes = (
            int(cell["maximum_complete_graph_checkpoint_bytes"])
            * FINAL_GRAPHS_PER_MODEL_N_CELL
        )
        timing_bytes = (
            int(cell["maximum_publication_timing_bytes"])
            * FINAL_GRAPHS_PER_MODEL_N_CELL
        )
        contributions.append(
            {
                "cell_id": cell["cell_id"],
                "graphs_represented": FINAL_GRAPHS_PER_MODEL_N_CELL,
                "projected_checkpoint_bytes": checkpoint_bytes,
                "projected_publication_timing_bytes": timing_bytes,
            }
        )
    projected_checkpoint_bytes = sum(
        int(item["projected_checkpoint_bytes"]) for item in contributions
    )
    projected_timing_bytes = sum(
        int(item["projected_publication_timing_bytes"])
        for item in contributions
    )
    fixed_overhead = max(
        int(cell["maximum_fixed_run_overhead_bytes"]) for cell in cells
    )
    run_level_overhead = projected_timing_bytes + fixed_overhead
    subtotal = projected_checkpoint_bytes + run_level_overhead
    metadata_allowance = ceil(
        subtotal
        * METADATA_ALLOWANCE_NUMERATOR
        / METADATA_ALLOWANCE_DENOMINATOR
    )
    projected_storage = subtotal + metadata_allowance
    required_free = max(
        2 * projected_storage,
        projected_storage + 5 * GIB_BYTES,
    )
    return {
        "cell_contributions": contributions,
        "projected_checkpoint_bytes": projected_checkpoint_bytes,
        "measured_projected_run_level_overhead_bytes": run_level_overhead,
        "metadata_allowance_rate": 0.01,
        "metadata_allowance_bytes": metadata_allowance,
        "projected_storage_bytes": projected_storage,
        "projected_storage_gib": projected_storage / GIB_BYTES,
        "required_free_space_formula": (
            "max(2*projected_storage_bytes,"
            "projected_storage_bytes+5*1073741824)"
        ),
        "required_free_bytes": required_free,
        "required_free_gib": required_free / GIB_BYTES,
    }


def _validate_capacity_profile_workload(
    profile: Mapping[str, object],
) -> None:
    expected_graphs = [
        {
            "graph_id": spec.graph_id,
            "model": spec.model,
            "n": spec.n,
            "m": spec.m,
            "excluded_seed": spec.seed,
        }
        for spec in BENCHMARK_GRAPH_SPECS
    ]
    if profile.get("benchmark_graphs") != expected_graphs:
        raise CapacityProfileError("capacity benchmark fixture grid mismatch")
    expected_policy = {
        "total_sequential_repetitions_per_graph": BENCHMARK_REPETITIONS,
        "warmup_repetition": WARMUP_REPETITION,
        "measured_repetitions": list(MEASURED_REPETITIONS),
        "parallel_execution": False,
    }
    if (
        profile.get("pair_count_per_graph") != BENCHMARK_PAIR_COUNT
        or profile.get("repetition_policy") != expected_policy
    ):
        raise CapacityProfileError("capacity repetition workload mismatch")
    raw = profile.get("raw_repetitions")
    if not isinstance(raw, list) or len(raw) != 18:
        raise CapacityProfileError(
            "capacity profile must contain exactly 18 repetitions"
        )
    expected_identities = [
        (spec.graph_id, repetition)
        for spec in BENCHMARK_GRAPH_SPECS
        for repetition in range(BENCHMARK_REPETITIONS)
    ]
    actual_identities = [
        (record.get("graph_id"), record.get("repetition"))
        if isinstance(record, Mapping)
        else (None, None)
        for record in raw
    ]
    if actual_identities != expected_identities:
        raise CapacityProfileError("capacity repetition identity order mismatch")
    expected_counts = {
        "pairs": 1_000,
        "dijkstra_records": 1_000,
        "route_records": 15_000,
        "distortion_records": 7,
    }
    for spec, repetition in (
        (spec, repetition)
        for spec in BENCHMARK_GRAPH_SPECS
        for repetition in range(BENCHMARK_REPETITIONS)
    ):
        index = BENCHMARK_GRAPH_SPECS.index(spec) * 3 + repetition
        record = raw[index]
        if not isinstance(record, Mapping):
            raise CapacityProfileError("capacity repetition record is invalid")
        if (
            record.get("model") != spec.model
            or record.get("n") != spec.n
            or record.get("m") != spec.m
            or record.get("excluded_seed") != spec.seed
            or record.get("row_counts") != expected_counts
            or record.get("published_checkpoint_integrity_validated") is not True
            or record.get("warmup") is not (repetition == 0)
            or record.get("included_in_projection")
            is not (repetition in MEASURED_REPETITIONS)
        ):
            raise CapacityProfileError("capacity repetition record mismatch")
    cells = summarise_capacity_cells(raw)
    runtime = project_runtime(cells)
    storage = project_storage(cells)
    if profile.get("cell_summaries") != cells:
        raise CapacityProfileError("capacity cell summaries mismatch")
    if profile.get("runtime_projection") != runtime:
        raise CapacityProfileError("capacity runtime projection mismatch")
    if profile.get("storage_projection") != storage:
        raise CapacityProfileError("capacity storage projection mismatch")
    for key in (
        "projected_storage_bytes",
        "projected_storage_gib",
        "required_free_bytes",
        "required_free_gib",
    ):
        if profile.get(key) != storage[key]:
            raise CapacityProfileError(f"capacity {key} mismatch")


def _tree_file_sizes(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree_bytes(root: Path) -> int:
    return sum(_tree_file_sizes(root).values()) if root.exists() else 0


def safe_remove_tree(containment_root: Path | str, target: Path | str) -> None:
    root = Path(containment_root).resolve(strict=True)
    resolved_target = Path(target).resolve(strict=True)
    try:
        relative = resolved_target.relative_to(root)
    except ValueError as exc:
        raise CapacityProfileError(
            f"cleanup target escapes benchmark root: {resolved_target}"
        ) from exc
    if not relative.parts:
        raise CapacityProfileError("refusing to remove the containment root itself")
    shutil.rmtree(resolved_target)
    if resolved_target.exists():
        raise CapacityProfileError(
            f"benchmark cleanup did not remove: {resolved_target}"
        )


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("networkx", "numpy"):
        versions[name] = importlib.metadata.version(name)
    return versions


def _read_json(path: Path) -> object:
    return decode_json_value(json.loads(path.read_text(encoding="utf-8")))


def _generate_excluded_graph(spec: CapacityGraphSpec, repetition: int):
    setting = FULL_EXPERIMENT_CONFIG.parameter_settings[spec.setting_index]
    if spec.model == ERDOS_RENYI:
        return generate_connected_erdos_renyi(
            n=spec.n,
            p=setting.er_p,
            graph_seed=spec.seed,
            replicate_index=0,
            max_attempts=FULL_EXPERIMENT_CONFIG.max_connected_graph_generation_attempts,
            setting_index=spec.setting_index,
            p_exact_numerator=setting.er_probability_numerator,
            p_exact_denominator=setting.er_probability_denominator,
        )
    if spec.model == BARABASI_ALBERT:
        return generate_connected_barabasi_albert(
            n=spec.n,
            m=spec.m,
            graph_seed=spec.seed,
            replicate_index=0,
            setting_index=spec.setting_index,
        )
    raise CapacityProfileError(f"unsupported capacity model: {spec.model}")


def _measure_repetition(
    *,
    benchmark_root: Path,
    spec: CapacityGraphSpec,
    repetition: int,
    disk_probe_path: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], int]:
    import run_full_experiment as runner

    def progress(stage: str) -> None:
        if progress_callback is not None:
            progress_callback(stage)

    repetition_root = (
        benchmark_root / f"{spec.graph_id}_repetition_{repetition}"
    ).resolve(strict=False)
    repetition_root.relative_to(benchmark_root)
    repetition_root.mkdir()
    progress("repetition_directory_created")
    setting = FULL_EXPERIMENT_CONFIG.parameter_settings[spec.setting_index]
    entry = runner.GraphScheduleEntry(
        schedule_index=0,
        setting_index=spec.setting_index,
        model=spec.model,
        n=spec.n,
        m=spec.m,
        replicate_index=0,
        graph_id=spec.graph_id,
        configuration_name="excluded_step15_capacity",
        setting_label=setting.label,
    )
    manifest = runner.build_experiment_run_manifest(
        output_root=benchmark_root,
        schedule=(entry,),
        execution_profile="development_fixture",
        require_final_scientific_source=False,
        config=FULL_EXPERIMENT_CONFIG,
    )
    manifest.update(
        {
            "execution_profile": "capacity_benchmark",
            "scientific_status": CAPACITY_PROFILE_LABEL,
            "non_scientific": True,
            "excluded_from_analysis": True,
            "capacity_repetition": repetition,
            "run_directory_name": repetition_root.name,
        }
    )
    write_run_manifest_once(repetition_root, manifest)
    progress("run_manifest_written")
    minimum_available = available_bytes(disk_probe_path)
    peak_temporary_plus_final = 0

    def observe(event: str, path: Path) -> None:
        nonlocal minimum_available, peak_temporary_plus_final
        progress(f"checkpoint_{event}")
        graph_parent = repetition_root / CHECKPOINT_DIRECTORY
        temporary_and_final = sum(
            _tree_bytes(child)
            for child in graph_parent.iterdir()
            if child.is_dir()
        ) if graph_parent.exists() else _tree_bytes(path)
        peak_temporary_plus_final = max(
            peak_temporary_plus_final,
            temporary_and_final,
        )
        minimum_available = min(
            minimum_available,
            available_bytes(disk_probe_path),
        )

    graph_start = perf_counter_ns()
    progress("graph_generation_started")
    generation_start = perf_counter_ns()
    generated = _generate_excluded_graph(spec, repetition)
    generation_runtime = perf_counter_ns() - generation_start
    progress("graph_generation_completed")
    try:
        progress("scientific_pipeline_started")
        data = runner._execute_generated_graph(
            entry=entry,
            config=FULL_EXPERIMENT_CONFIG,
            generated=generated,
            generation_runtime_ns=generation_runtime,
            pair_master_seed=spec.seed,
            pair_count=BENCHMARK_PAIR_COUNT,
            run_manifest=manifest,
            graph_wall_start_ns=graph_start,
        )
        progress("scientific_pipeline_completed")
        progress("checkpoint_publication_started")
        publication = publish_graph_checkpoint(
            repetition_root,
            data,
            graph_wall_start_ns=graph_start,
            event_callback=observe,
        )
        progress("checkpoint_publication_completed")
        write_progress(
            repetition_root,
            schedule_ids=(spec.graph_id,),
            complete_graph_ids=(spec.graph_id,),
        )
        progress("checkpoint_validation_started")
        validation = validate_graph_checkpoint(
            publication.path,
            expected_run_manifest=manifest,
            expected_graph_id=spec.graph_id,
        )
        timing_record = validate_publication_timing_record(
            repetition_root,
            graph_id=spec.graph_id,
            expected_run_manifest=manifest,
        )
        progress("checkpoint_validation_completed")
        expected_counts = {
            "pairs": 1_000,
            "dijkstra_records": 1_000,
            "route_records": 15_000,
            "distortion_records": 7,
        }
        for key, expected in expected_counts.items():
            if validation.counts.get(key) != expected:
                raise CapacityProfileError(
                    f"capacity checkpoint {key} count is not {expected}"
                )
        checkpoint_files = _tree_file_sizes(publication.path)
        checkpoint_bytes = sum(checkpoint_files.values())
        compressed_files = {
            name: size
            for name, size in checkpoint_files.items()
            if name.endswith(".gz")
        }
        timing_path = (
            repetition_root
            / PUBLICATION_TIMING_DIRECTORY
            / f"{spec.graph_id}.json"
        )
        overhead = {
            "run_manifest": (
                repetition_root / RUN_MANIFEST_FILENAME
            ).stat().st_size,
            "progress": (repetition_root / PROGRESS_FILENAME).stat().st_size,
            "publication_timing": timing_path.stat().st_size,
        }
        payload_timings = _read_json(publication.path / "timings.json")
        if not isinstance(payload_timings, dict):
            raise CapacityProfileError("checkpoint timings payload is invalid")
        minimum_available = min(
            minimum_available,
            available_bytes(disk_probe_path),
        )
        result = {
            "graph_id": spec.graph_id,
            "cell_id": spec.cell_id,
            "model": spec.model,
            "n": spec.n,
            "m": spec.m,
            "excluded_seed": spec.seed,
            "repetition": repetition,
            "warmup": repetition == WARMUP_REPETITION,
            "included_in_projection": repetition in MEASURED_REPETITIONS,
            "end_to_end_graph_wall_ns": int(
                timing_record["end_to_end_graph_wall_ns"]
            ),
            "component_timings_ns": {
                **{
                    key: int(value)
                    for key, value in payload_timings.items()
                    if key.endswith("_ns")
                },
                "payload_serialization_ns": int(
                    timing_record["payload_serialization_ns"]
                ),
                "atomic_publication_and_final_validation_ns": int(
                    timing_record[
                        "atomic_publication_and_final_validation_ns"
                    ]
                ),
            },
            "complete_graph_checkpoint_bytes": checkpoint_bytes,
            "checkpoint_files_bytes": checkpoint_files,
            "compressed_checkpoint_files_bytes": compressed_files,
            "run_level_overhead_bytes": sum(overhead.values()),
            "run_level_overhead_breakdown_bytes": overhead,
            "peak_temporary_plus_final_bytes": peak_temporary_plus_final,
            "row_counts": {
                key: int(validation.counts[key]) for key in expected_counts
            },
            "published_checkpoint_integrity_validated": True,
        }
        return result, minimum_available
    except BaseException as exc:
        raise CapacityProfileError(
            "capacity benchmark failed; preserved temporary evidence at "
            f"{repetition_root}: {type(exc).__name__}: {exc}"
        ) from exc


def run_capacity_canary(
    *,
    n: int = 100,
    output_root: Path | str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run exactly one approved excluded ER capacity canary and remove it."""

    import run_full_experiment as runner

    def progress(stage: str) -> None:
        if progress_callback is not None:
            progress_callback(stage)

    if (
        DATA_GENERATION_HASH,
        ANALYSIS_PLAN_HASH,
        COMBINED_FREEZE_HASH,
    ) != (
        EXPECTED_DATA_GENERATION_HASH,
        EXPECTED_ANALYSIS_PLAN_HASH,
        EXPECTED_COMBINED_FREEZE_HASH,
    ):
        raise CapacityProfileError("Step 13 hashes changed; canary refused")
    final_ids = tuple(entry.graph_id for entry in runner.build_full_schedule())
    specs = validate_benchmark_specs(final_ids)
    if n not in CANARY_GRAPH_SPECS_BY_N:
        raise CapacityProfileError("canary n must be 100 or 1000")
    spec = CANARY_GRAPH_SPECS_BY_N[n]
    if spec not in specs or spec.graph_id in set(final_ids):
        raise CapacityProfileError(
            "canary identity is not an approved excluded graph identity"
        )
    expected_seed = 4_000_003 if n == 100 else 4_000_037
    if (spec.model, spec.n, spec.m, spec.seed) != (
        ERDOS_RENYI,
        n,
        16,
        expected_seed,
    ):
        raise CapacityProfileError("canary fixture definition changed")

    creation_parent = (
        Path(tempfile.gettempdir()).resolve(strict=True)
        if output_root is None
        else nearest_existing_parent(Path(output_root))
    )
    benchmark_root = Path(
        tempfile.mkdtemp(prefix="step15-canary-", dir=creation_parent)
    ).resolve(strict=True)
    progress(f"canary_root_created:{benchmark_root}")
    repetition_root = (
        benchmark_root / f"{spec.graph_id}_repetition_0"
    ).resolve(strict=False)
    try:
        record, _minimum_available = _measure_repetition(
            benchmark_root=benchmark_root,
            spec=spec,
            repetition=0,
            disk_probe_path=creation_parent,
            progress_callback=progress_callback,
        )
        progress("canary_measurement_validated")
        safe_remove_tree(benchmark_root, repetition_root)
        progress("temporary_checkpoint_removed")
        benchmark_root.rmdir()
        if benchmark_root.exists():
            raise CapacityProfileError("canary root cleanup failed")
        progress("canary_root_removed")
    except BaseException as exc:
        raise CapacityProfileError(
            "capacity canary failed; preserved temporary evidence at "
            f"{benchmark_root}: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "canary_graph_id": spec.graph_id,
        "model": spec.model,
        "n": spec.n,
        "m": spec.m,
        "excluded_seed": spec.seed,
        "pair_count": BENCHMARK_PAIR_COUNT,
        "repetitions": 1,
        "row_counts": record["row_counts"],
        "end_to_end_graph_wall_ns": record["end_to_end_graph_wall_ns"],
        "component_timings_ns": record["component_timings_ns"],
        "complete_graph_checkpoint_bytes": record[
            "complete_graph_checkpoint_bytes"
        ],
        "peak_temporary_plus_final_bytes": record[
            "peak_temporary_plus_final_bytes"
        ],
        "compressed_checkpoint_files_bytes": record[
            "compressed_checkpoint_files_bytes"
        ],
        "published_checkpoint_integrity_validated": record[
            "published_checkpoint_integrity_validated"
        ],
        "publication_timing_integrity_validated": True,
        "temporary_checkpoint_cleanup_completed": True,
        "temporary_root_removed": True,
        "step13_hashes": {
            "data_generation": DATA_GENERATION_HASH,
            "analysis_plan": ANALYSIS_PLAN_HASH,
            "combined": COMBINED_FREEZE_HASH,
        },
        "non_scientific": True,
        "excluded_from_analysis": True,
    }


def watchdog_seconds_for_n(n: int) -> int:
    try:
        return WATCHDOG_SECONDS_BY_N[n]
    except KeyError as exc:
        raise CapacityProfileError(
            f"no Step 15 watchdog is defined for n={n}"
        ) from exc


def _run_one_repetition_process(
    *,
    spec_index: int,
    repetition: int,
    output_root: Path | str,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Child-process entry point for exactly one excluded repetition."""

    import run_full_experiment as runner

    def progress(stage: str) -> None:
        if progress_callback is not None:
            progress_callback(stage)

    final_ids = tuple(entry.graph_id for entry in runner.build_full_schedule())
    specs = validate_benchmark_specs(final_ids)
    if isinstance(spec_index, bool) or not 0 <= spec_index < len(specs):
        raise CapacityProfileError("internal capacity spec index is invalid")
    if repetition not in range(BENCHMARK_REPETITIONS):
        raise CapacityProfileError("internal capacity repetition is invalid")
    spec = specs[spec_index]
    creation_parent = nearest_existing_parent(Path(output_root))
    benchmark_root = Path(
        tempfile.mkdtemp(
            prefix=(
                f"step15-repetition-{spec.cell_id}-rep{repetition}-"
            ),
            dir=creation_parent,
        )
    ).resolve(strict=True)
    progress(f"repetition_root_created:{benchmark_root}")
    repetition_root = (
        benchmark_root / f"{spec.graph_id}_repetition_{repetition}"
    ).resolve(strict=False)
    try:
        record, minimum_available = _measure_repetition(
            benchmark_root=benchmark_root,
            spec=spec,
            repetition=repetition,
            disk_probe_path=creation_parent,
            progress_callback=progress_callback,
        )
        progress("repetition_measurement_validated")
        safe_remove_tree(benchmark_root, repetition_root)
        progress("temporary_repetition_removed")
        benchmark_root.rmdir()
        if benchmark_root.exists():
            raise CapacityProfileError("repetition root cleanup failed")
        progress("repetition_root_removed")
    except BaseException as exc:
        raise CapacityProfileError(
            "capacity repetition failed; preserved temporary evidence at "
            f"{benchmark_root}: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "record": record,
        "minimum_available_bytes": minimum_available,
        "temporary_checkpoint_cleanup_completed": True,
        "temporary_root_removed": True,
    }


def _child_python_executable() -> str:
    if os.name == "nt":
        candidate = Path(sys.prefix) / "Scripts" / "python.exe"
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _terminate_child_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_repetition_with_watchdog(
    *,
    spec_index: int,
    spec: CapacityGraphSpec,
    repetition: int,
    output_root: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], int]:
    """Run one excluded repetition in a child under its frozen hard limit."""

    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    command = [
        _child_python_executable(),
        "-B",
        str(Path(__file__).resolve()),
        "--internal-spec-index",
        str(spec_index),
        "--internal-repetition",
        str(repetition),
        "--output-root",
        str(output_root),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    timeout_seconds = watchdog_seconds_for_n(spec.n)
    started = monotonic()
    deadline = started + timeout_seconds
    next_heartbeat = started + 30.0
    reader_done = False
    last_stage = "child_started"
    preserved_root: str | None = None
    payload: dict[str, object] | None = None
    while True:
        now = monotonic()
        if now >= deadline and process.poll() is None:
            _terminate_child_process(process)
            raise CapacityProfileError(
                "capacity repetition watchdog timeout; "
                f"graph_id={spec.graph_id} repetition={repetition} "
                f"limit_seconds={timeout_seconds} last_stage={last_stage} "
                f"preserved_root={preserved_root}"
            )
        if now >= next_heartbeat and process.poll() is None:
            progress(
                "BENCHMARK_HEARTBEAT "
                f"graph_id={spec.graph_id} repetition={repetition} "
                f"elapsed_seconds={now - started:.1f} "
                f"last_stage={last_stage}"
            )
            next_heartbeat += 30.0
        try:
            item = lines.get(
                timeout=min(0.25, max(deadline - now, 0.01))
            )
        except queue.Empty:
            item = ...
        if item is None:
            reader_done = True
        elif item is not ...:
            progress(item)
            if item.startswith("BENCHMARK_STAGE "):
                last_stage = item.removeprefix("BENCHMARK_STAGE ")
                marker = "repetition_root_created:"
                if marker in last_stage:
                    preserved_root = last_stage.split(marker, 1)[1]
            elif item.startswith("BENCHMARK_RECORD "):
                decoded = json.loads(item.removeprefix("BENCHMARK_RECORD "))
                if not isinstance(decoded, dict):
                    raise CapacityProfileError(
                        "capacity child record is not an object"
                    )
                payload = decoded
        if process.poll() is not None and reader_done:
            break
    return_code = process.wait()
    if return_code != 0 or payload is None:
        raise CapacityProfileError(
            "capacity repetition child failed; "
            f"graph_id={spec.graph_id} repetition={repetition} "
            f"return_code={return_code} last_stage={last_stage} "
            f"preserved_root={preserved_root}"
        )
    record = payload.get("record")
    minimum_available = payload.get("minimum_available_bytes")
    if not isinstance(record, dict) or (
        isinstance(minimum_available, bool)
        or not isinstance(minimum_available, int)
        or minimum_available <= 0
    ):
        raise CapacityProfileError("capacity child result schema is invalid")
    if (
        record.get("graph_id") != spec.graph_id
        or record.get("repetition") != repetition
    ):
        raise CapacityProfileError("capacity child identity mismatch")
    expected_counts = {
        "pairs": 1_000,
        "dijkstra_records": 1_000,
        "route_records": 15_000,
        "distortion_records": 7,
    }
    if record.get("row_counts") != expected_counts:
        raise CapacityProfileError("capacity child row counts are invalid")
    if (
        payload.get("temporary_checkpoint_cleanup_completed") is not True
        or payload.get("temporary_root_removed") is not True
    ):
        raise CapacityProfileError("capacity child cleanup was not confirmed")
    return record, minimum_available


def build_capacity_profile(
    *,
    repetitions: Sequence[Mapping[str, object]],
    source_commit: str,
    benchmark_volume_identifier: str,
    benchmark_filesystem_type: str | None,
    available_before_bytes: int,
    available_during_peak_bytes: int,
    available_after_cleanup_bytes: int,
    cleanup_restored_temporary_usage: bool,
    benchmark_timestamp_utc: str,
) -> dict[str, object]:
    cells = summarise_capacity_cells(repetitions)
    runtime = project_runtime(cells)
    storage = project_storage(cells)
    profile: dict[str, object] = {
        "profile_schema_version": CAPACITY_PROFILE_SCHEMA_VERSION,
        "profile_label": CAPACITY_PROFILE_LABEL,
        "non_scientific": True,
        "excluded_from_analysis": True,
        "engineering_estimate_warning": (
            "Operational engineering estimate only; not a scientific outcome."
        ),
        "step13_hashes": {
            "data_generation": DATA_GENERATION_HASH,
            "analysis_plan": ANALYSIS_PLAN_HASH,
            "combined": COMBINED_FREEZE_HASH,
        },
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "base_commit": source_commit,
        "source_commit": source_commit,
        "performance_source_manifest": performance_source_manifest(),
        "performance_source_content_fingerprint": (
            performance_source_fingerprint()
        ),
        "benchmark_code_content_fingerprint": (
            benchmark_code_content_fingerprint()
        ),
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "volume_identifier": benchmark_volume_identifier,
        "filesystem_type": benchmark_filesystem_type,
        "available_before_benchmark_bytes": available_before_bytes,
        "available_during_peak_storage_bytes": available_during_peak_bytes,
        "available_after_cleanup_bytes": available_after_cleanup_bytes,
        "cleanup_restored_temporary_usage": cleanup_restored_temporary_usage,
        "benchmark_graphs": [
            {
                "graph_id": spec.graph_id,
                "model": spec.model,
                "n": spec.n,
                "m": spec.m,
                "excluded_seed": spec.seed,
            }
            for spec in BENCHMARK_GRAPH_SPECS
        ],
        "pair_count_per_graph": BENCHMARK_PAIR_COUNT,
        "repetition_policy": {
            "total_sequential_repetitions_per_graph": BENCHMARK_REPETITIONS,
            "warmup_repetition": WARMUP_REPETITION,
            "measured_repetitions": list(MEASURED_REPETITIONS),
            "parallel_execution": False,
        },
        "raw_repetitions": list(repetitions),
        "cell_summaries": cells,
        "runtime_projection": runtime,
        "storage_projection": storage,
        "projected_storage_bytes": storage["projected_storage_bytes"],
        "projected_storage_gib": storage["projected_storage_gib"],
        "required_free_bytes": storage["required_free_bytes"],
        "required_free_gib": storage["required_free_gib"],
        "benchmark_timestamp_utc": benchmark_timestamp_utc,
        "profile_sha256": "",
    }
    profile["profile_sha256"] = profile_sha256(profile)
    validate_capacity_profile(profile)
    return profile


def run_capacity_benchmark(
    *,
    output_root: Path | str | None = None,
    profile_path: Path | str | None = None,
    repetition_runner: (
        Callable[..., tuple[dict[str, object], int]] | None
    ) = None,
) -> dict[str, object]:
    """Execute all 18 excluded repetitions sequentially and write the profile."""

    import run_full_experiment as runner

    if (
        DATA_GENERATION_HASH,
        ANALYSIS_PLAN_HASH,
        COMBINED_FREEZE_HASH,
    ) != (
        EXPECTED_DATA_GENERATION_HASH,
        EXPECTED_ANALYSIS_PLAN_HASH,
        EXPECTED_COMBINED_FREEZE_HASH,
    ):
        raise CapacityProfileError("Step 13 hashes changed; benchmark refused")
    final_ids = tuple(entry.graph_id for entry in runner.build_full_schedule())
    specs = validate_benchmark_specs(final_ids)
    if output_root is None:
        creation_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    else:
        creation_parent = nearest_existing_parent(Path(output_root))
    benchmark_volume = volume_identifier(creation_parent)
    benchmark_filesystem = filesystem_type(creation_parent)
    available_before = available_bytes(creation_parent)
    repetitions: list[dict[str, object]] = []
    minimum_available = available_before
    child_runner = (
        _run_repetition_with_watchdog
        if repetition_runner is None
        else repetition_runner
    )
    source_commit = runner.build_experiment_run_manifest(
        output_root=creation_parent,
        schedule=(),
        execution_profile="development_fixture",
        require_final_scientific_source=False,
        config=FULL_EXPERIMENT_CONFIG,
    )["git_commit_hash"]
    for spec_index, spec in enumerate(specs):
        for repetition in range(BENCHMARK_REPETITIONS):
            print(
                "starting excluded capacity fixture "
                f"{spec.graph_id} repetition={repetition} "
                f"watchdog_seconds={watchdog_seconds_for_n(spec.n)}",
                flush=True,
            )
            record, repetition_minimum = child_runner(
                spec_index=spec_index,
                spec=spec,
                repetition=repetition,
                output_root=creation_parent,
                progress_callback=lambda message: print(
                    message,
                    flush=True,
                ),
            )
            repetitions.append(record)
            minimum_available = min(minimum_available, repetition_minimum)
            print(
                "completed excluded capacity fixture "
                f"{spec.graph_id} repetition={repetition} "
                f"end_to_end_seconds="
                f"{record['end_to_end_graph_wall_ns'] / 1_000_000_000:.6f} "
                f"checkpoint_bytes="
                f"{record['complete_graph_checkpoint_bytes']}",
                flush=True,
            )
    available_after = available_bytes(creation_parent)
    profile = build_capacity_profile(
        repetitions=repetitions,
        source_commit=str(source_commit),
        benchmark_volume_identifier=benchmark_volume,
        benchmark_filesystem_type=benchmark_filesystem,
        available_before_bytes=available_before,
        available_during_peak_bytes=minimum_available,
        available_after_cleanup_bytes=available_after,
        cleanup_restored_temporary_usage=True,
        benchmark_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    written = write_capacity_profile(profile, profile_path)
    validated = load_capacity_profile(
        written,
        expected_volume_identifier=benchmark_volume,
    )
    return {
        "profile_path": str(written),
        "profile_sha256": validated["profile_sha256"],
        "benchmark_graph_count": len(specs),
        "completed_repetitions": len(repetitions),
        "benchmark_volume_identifier": benchmark_volume,
        "available_after_cleanup_bytes": available_after,
        "required_free_bytes": validated["required_free_bytes"],
        "drive_passes": available_after >= int(validated["required_free_bytes"]),
        "nominal_projected_runtime_seconds": validated["runtime_projection"][
            "nominal_projected_runtime_seconds"
        ],
        "conservative_projected_runtime_seconds": validated[
            "runtime_projection"
        ]["conservative_projected_runtime_seconds"],
        "projected_storage_bytes": validated["projected_storage_bytes"],
        "raw_checkpoints_removed": True,
        "non_scientific": True,
        "excluded_from_analysis": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the excluded non-scientific Step 15 capacity benchmark"
        )
    )
    parser.add_argument("--output-root")
    parser.add_argument(
        "--canary",
        action="store_true",
        help="run exactly one approved excluded ER canary",
    )
    parser.add_argument(
        "--canary-n",
        type=int,
        choices=tuple(CANARY_GRAPH_SPECS_BY_N),
        default=100,
        help="approved excluded canary size (default: 100)",
    )
    parser.add_argument(
        "--internal-spec-index",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-repetition",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    internal_requested = (
        args.internal_spec_index is not None
        or args.internal_repetition is not None
    )
    if internal_requested:
        if (
            args.canary
            or args.internal_spec_index is None
            or args.internal_repetition is None
            or args.output_root is None
            or not 0 <= args.internal_spec_index < len(BENCHMARK_GRAPH_SPECS)
        ):
            raise CapacityProfileError(
                "internal repetition arguments are incomplete or conflicting"
            )
        spec = BENCHMARK_GRAPH_SPECS[args.internal_spec_index]
        result = _run_one_repetition_process(
            spec_index=args.internal_spec_index,
            repetition=args.internal_repetition,
            output_root=args.output_root,
            progress_callback=lambda stage: print(
                "BENCHMARK_STAGE "
                f"graph_id={spec.graph_id} "
                f"repetition={args.internal_repetition} "
                f"stage={stage}",
                flush=True,
            ),
        )
        print(
            "BENCHMARK_RECORD "
            + canonical_profile_bytes(result).decode("utf-8"),
            flush=True,
        )
        return 0
    if args.canary:
        result = run_capacity_canary(
            n=args.canary_n,
            output_root=args.output_root,
            progress_callback=lambda stage: print(
                f"CANARY_STAGE {stage}",
                flush=True,
            ),
        )
    else:
        result = run_capacity_benchmark(output_root=args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
