"""Excluded operational-capacity benchmark for the frozen Iteration 2 workload.

The benchmark uses identities and seeds disjoint from the scientific schedule,
writes graph checkpoints only below a verified system temporary directory, and
removes successful temporary outputs.  Its profile contains timings and byte
counts only; routing outcomes are neither retained nor interpreted.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
from hashlib import blake2s, sha256
from importlib.metadata import PackageNotFoundError, version
import inspect
import io
import json
from math import isfinite
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import gettempdir, mkdtemp
from time import perf_counter, perf_counter_ns, sleep
from typing import Mapping, Sequence
from uuid import uuid4

from graph_generation import (
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
)
from iteration2_config import (
    COMBINED_PROTOCOL_HASH,
    FULL_ORACLE_M_VALUES,
    FULL_ORACLE_REPLICATE_INDICES,
    GRAPH_MODELS,
    GRAPH_REPETITIONS,
    ITERATION2_CAPACITY_PROFILE_SCHEMA,
    ITERATION2_GRAPH_COUNT,
    ITERATION2_OUTPUT_SCHEMA,
    ITERATION2_RESULT_SCHEMA,
    ITERATION2_RUN_IDENTITY,
    MATCHED_CONDITIONS,
    N_VALUES,
    OUTPUT_SCHEMA_HASH,
    PAIRS_PER_GRAPH,
    full_schedule,
    seeds_for_graph,
)
from iteration2_excluded import ExcludedAnalysisFixtureContract
from iteration2_experiment import execute_iteration2_graph
from iteration2_runtime_guard import (
    PREFLIGHT_READ_ONLY,
    require_zero_scientific_operations,
    scientific_operation_context,
)
from iteration2_reporting import (
    FIGURE_FILES,
    WORKBOOK_SHEETS,
    build_reporting_bundle,
)
from validate_iteration2 import validate_iteration2_graph_result


PROFILE_FILENAME = "iteration2_capacity_profile.json"
PROFILE_LABEL = (
    "NON-SCIENTIFIC ITERATION 2 CAPACITY BENCHMARK - EXCLUDED FROM ANALYSIS"
)
BENCHMARK_SCHEMA = "greedy_routing_iteration2_capacity_benchmark_v2"
SOURCE_MANIFEST_SCHEMA = "greedy_routing_iteration2_capacity_source_v2"
ENVIRONMENT_IDENTITY_SCHEMA = "iteration2_capacity_environment_v1"
RECOVERY_EVIDENCE_SCHEMA = "iteration2_capacity_recovered_log_extract_v1"
PROFILE_INTERNAL_HASH_FIELD = "profile_sha256"
BENCHMARK_M = 16
WARMUP_ROLE = "warmup_standard"
MEASURED_STANDARD_ROLE = "measured_standard"
MEASURED_ORACLE_ROLE = "measured_full_oracle"
REPETITION_ROLES = (
    WARMUP_ROLE,
    MEASURED_STANDARD_ROLE,
    MEASURED_ORACLE_ROLE,
)
CONSERVATIVE_RUNTIME_FACTOR = 2.0
GIB_BYTES = 1024**3
FIXED_FREE_SPACE_RESERVE_BYTES = 10 * GIB_BYTES
DERIVED_MACHINE_READABLE_FLOOR_BYTES = 64 * 1024**2
WORKBOOK_FLOOR_BYTES = 32 * 1024**2
FIGURE_FLOOR_BYTES = 16 * 1024**2
RUN_LEVEL_FLOOR_BYTES = 16 * 1024**2
PUBLICATION_MACHINE_READABLE_FACTOR = 512
PUBLICATION_WORKBOOK_FACTOR = 64
PUBLICATION_FIGURE_FACTOR = 2
PRIMARY_COORDINATE_ROUTES_PER_PAIR = 27
NATIVE_MDS_REFERENCE_ROUTES_PER_PAIR = 1
ROUTE_RECORDS_PER_PAIR = 28
EMBEDDING_ARTIFACTS_PER_GRAPH = 10
FULL_ORACLE_GRAPH_COUNT = (
    len(GRAPH_MODELS)
    * len(N_VALUES)
    * len(FULL_ORACLE_M_VALUES)
    * len(FULL_ORACLE_REPLICATE_INDICES)
)
STANDARD_SENTINEL_GRAPH_COUNT = ITERATION2_GRAPH_COUNT - FULL_ORACLE_GRAPH_COUNT

SOURCE_ENTRYPOINTS = (
    "code/analyze_iteration2.py",
    "code/benchmark_iteration2_capacity.py",
    "code/iteration2_experiment.py",
    "code/run_iteration2.py",
    "code/validate_iteration2.py",
)

_FORBIDDEN_OUTCOME_KEYS = {
    "success",
    "failure",
    "stretch",
    "interaction",
    "estimate",
    "route_walk",
    "routing_outcome",
}


class Iteration2CapacityError(ValueError):
    """Raised when an Iteration 2 capacity invariant is not satisfied."""


@dataclass(frozen=True)
class BenchmarkSpec:
    cell_id: str
    graph_id: str
    model: str
    n: int
    m: int
    graph_seed: int
    pair_seed: int

    def identity(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "graph_id": self.graph_id,
            "model": self.model,
            "n": self.n,
            "m": self.m,
            "graph_seed": self.graph_seed,
            "pair_seed": self.pair_seed,
        }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_profile_path() -> Path:
    return Path(__file__).resolve().with_name(PROFILE_FILENAME)


def nearest_existing_parent(path: Path | str) -> Path:
    candidate = Path(path).resolve(strict=False)
    while not candidate.exists():
        if candidate.parent == candidate:
            raise Iteration2CapacityError(f"no existing parent for path: {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate.resolve(strict=True)


def volume_identifier(path: Path | str) -> str:
    existing = nearest_existing_parent(path)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        drive = existing.drive.upper()
        if not drive:
            raise Iteration2CapacityError("Windows volume has no drive identifier")
        serial = wintypes.DWORD()
        maximum_component = wintypes.DWORD()
        flags = wintypes.DWORD()
        volume_name = ctypes.create_unicode_buffer(261)
        file_system = ctypes.create_unicode_buffer(261)
        root = f"{drive}\\"
        if not ctypes.windll.kernel32.GetVolumeInformationW(
            root,
            volume_name,
            len(volume_name),
            ctypes.byref(serial),
            ctypes.byref(maximum_component),
            ctypes.byref(flags),
            file_system,
            len(file_system),
        ):
            raise Iteration2CapacityError(
                "Windows volume serial could not be resolved"
            )
        return f"windows_volume:{drive}:serial:{serial.value:08X}"
    return f"posix_device:{existing.stat().st_dev}"


def available_bytes(path: Path | str) -> int:
    return int(shutil.disk_usage(nearest_existing_parent(path)).free)


def _physical_processor_count() -> int | None:
    if os.name != "nt":
        return None
    process = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "(Get-CimInstance Win32_Processor | "
                "Measure-Object -Property NumberOfCores -Sum).Sum"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        value = int(process.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def environment_identity() -> dict[str, object]:
    """Return hard compatibility fields separately from information-only data."""

    architecture = platform.architecture()
    return {
        "schema": ENVIRONMENT_IDENTITY_SCHEMA,
        "hard_requirements": {
            "python_implementation": platform.python_implementation(),
            "python_full_version": platform.python_version(),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "operating_system_version": platform.version(),
            "machine_architecture": platform.machine(),
            "architecture_bits": architecture[0],
            "architecture_linkage": architecture[1],
            "processor_identity": platform.processor(),
            "logical_processor_count": os.cpu_count(),
            "physical_processor_count": _physical_processor_count(),
        },
        "informational": {
            "platform_string": platform.platform(),
            "python_build": list(platform.python_build()),
            "python_compiler": platform.python_compiler(),
        },
    }


def _canonical_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise Iteration2CapacityError(
                "capacity profiles reject NaN and infinity"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Iteration2CapacityError(
                    "capacity profile keys must be strings"
                )
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise Iteration2CapacityError(
        f"unsupported capacity profile value: {type(value).__name__}"
    )


def canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        _canonical_value(dict(value)),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def profile_sha256(profile: Mapping[str, object]) -> str:
    payload = dict(profile)
    payload.pop(PROFILE_INTERNAL_HASH_FIELD, None)
    return sha256(canonical_bytes(payload)).hexdigest()


def recovery_evidence_payload(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Canonical path-free extract of the preserved completion-log evidence."""

    return {
        "schema": RECOVERY_EVIDENCE_SCHEMA,
        "records": [
            {
                "cell_id": record["cell_id"],
                "repetition_role": record["repetition_role"],
                "end_to_end_ns": record["end_to_end_ns"],
                "checkpoint_bytes": record["checkpoint_bytes"],
            }
            for record in records
        ],
        "post_measurement_failure": (
            "ValueError: reporting target must use the Iteration 2 identity"
        ),
    }


def bind_recovery_evidence(
    recovery: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload = recovery_evidence_payload(records)
    encoded = canonical_bytes(payload)
    return {
        **dict(recovery),
        "evidence_schema": RECOVERY_EVIDENCE_SCHEMA,
        "evidence_sha256": sha256(encoded).hexdigest(),
        "evidence_byte_size": len(encoded),
        "evidence_record_count": len(records),
        "evidence_path_recorded": False,
    }


def file_sha256(path: Path | str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def dependency_fingerprint(
    root: Path | str | None = None,
) -> dict[str, object]:
    project = repository_root() if root is None else Path(root).resolve()
    versions: dict[str, str] = {}
    for line in (project / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        package, expected = requirement.split("==", 1)
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise Iteration2CapacityError(
                f"missing required dependency: {package}"
            ) from exc
        if installed != expected:
            raise Iteration2CapacityError(
                f"dependency mismatch for {package}: {installed} != {expected}"
            )
        versions[package] = installed
    return {
        "versions": versions,
        "sha256": sha256(canonical_bytes(versions)).hexdigest(),
    }


def performance_source_paths(
    root: Path | str | None = None,
) -> tuple[str, ...]:
    """Return the deterministic transitive closure of local runtime imports."""

    project = repository_root() if root is None else Path(root).resolve()
    code_root = project / "code"
    pending = list(SOURCE_ENTRYPOINTS)
    included: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in included:
            continue
        path = (project / relative).resolve(strict=True)
        try:
            path.relative_to(project)
        except ValueError as exc:
            raise Iteration2CapacityError(
                f"capacity source path escapes repository: {relative}"
            ) from exc
        if not path.is_file() or path.suffix != ".py":
            raise Iteration2CapacityError(
                f"capacity source entry is not Python: {relative}"
            )
        included.add(relative)
        try:
            syntax = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (SyntaxError, UnicodeError) as exc:
            raise Iteration2CapacityError(
                f"cannot parse capacity source dependency: {relative}"
            ) from exc
        modules: set[str] = set()
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".", 1)[0])
        for module in sorted(modules):
            candidate = code_root / f"{module}.py"
            if candidate.is_file():
                pending.append(f"code/{module}.py")
    included.add("requirements.txt")
    return tuple(sorted(included))


def performance_source_manifest(
    root: Path | str | None = None,
) -> dict[str, object]:
    project = repository_root() if root is None else Path(root).resolve()
    files: dict[str, str] = {}
    paths = performance_source_paths(project)
    for relative in paths:
        path = (project / relative).resolve(strict=True)
        try:
            path.relative_to(project)
        except ValueError as exc:
            raise Iteration2CapacityError(
                f"capacity source path escapes repository: {relative}"
            ) from exc
        if not path.is_file():
            raise Iteration2CapacityError(
                f"capacity source file is missing: {relative}"
            )
        files[relative] = file_sha256(path)
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "entrypoints": list(SOURCE_ENTRYPOINTS),
        "closure_rule": "recursive_top_level_local_python_imports_plus_requirements",
        "files": files,
    }


def verify_committed_source_manifest(
    root: Path | str,
    commit: str,
    manifest: Mapping[str, object],
) -> None:
    """Verify a stored source manifest against immutable Git object bytes."""

    project = Path(root).resolve(strict=True)
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise Iteration2CapacityError("stored source manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise Iteration2CapacityError("stored source manifest is empty")
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise Iteration2CapacityError("stored source manifest is malformed")
        process = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=project,
            check=False,
            capture_output=True,
        )
        if process.returncode != 0:
            raise Iteration2CapacityError(
                f"source manifest path is absent from commit: {relative}"
            )
        if sha256(process.stdout).hexdigest() != expected:
            raise Iteration2CapacityError(
                f"committed source differs from manifest: {relative}"
            )


def performance_source_fingerprint(
    root: Path | str | None = None,
) -> str:
    return sha256(canonical_bytes(performance_source_manifest(root))).hexdigest()


def benchmark_code_fingerprint(
    root: Path | str | None = None,
) -> str:
    project = repository_root() if root is None else Path(root).resolve()
    return file_sha256(project / "code" / "benchmark_iteration2_capacity.py")


def runner_source_fingerprint(
    root: Path | str | None = None,
) -> str:
    project = repository_root() if root is None else Path(root).resolve()
    return file_sha256(project / "code" / "run_iteration2.py")


def measurement_worker_fingerprint() -> str:
    worker_source = inspect.getsource(_worker_record)
    execution_source = worker_source.split("    record = {", 1)[0]
    payload = "".join(
        (
            inspect.getsource(_generate_graph),
            inspect.getsource(_json_bytes),
            inspect.getsource(_gzip_bytes),
            execution_source,
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def physical_profile_sha256(path: Path | str | None = None) -> str:
    profile_path = default_profile_path() if path is None else Path(path)
    return file_sha256(profile_path)


def _derived_seed(domain: str, model: str, n: int) -> int:
    digest = blake2s(
        f"excluded_iteration2_capacity_v1|{domain}|{model}|{n}".encode("ascii"),
        digest_size=4,
        person=b"i2capv1",
    ).digest()
    return int.from_bytes(digest, "big")


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    specs = []
    for model in GRAPH_MODELS:
        short = "er" if model == "erdos_renyi" else "ba"
        for n in N_VALUES:
            specs.append(
                BenchmarkSpec(
                    cell_id=f"{short}_n{n:04d}",
                    graph_id=(
                        f"excluded_i2_capacity_{short}_n{n:04d}_m{BENCHMARK_M:02d}"
                    ),
                    model=model,
                    n=n,
                    m=BENCHMARK_M,
                    graph_seed=_derived_seed("graph", model, n),
                    pair_seed=_derived_seed("pairs", model, n),
                )
            )
    return tuple(specs)


def excluded_capacity_contract() -> ExcludedAnalysisFixtureContract:
    """Return the canonical non-scientific contract for this benchmark."""

    specs = benchmark_specs()
    return ExcludedAnalysisFixtureContract(
        fixture_tag="capacity_benchmark",
        expected_graph_ids=tuple(spec.graph_id for spec in specs),
        excluded_seeds=tuple(
            seed
            for spec in specs
            for seed in (spec.graph_seed, spec.pair_seed)
        ),
        pair_count=PAIRS_PER_GRAPH,
        bootstrap_replicates=1,
        property_resampling_replicates=1,
        permutation_replicates=1,
    )


def excluded_capacity_identity() -> dict[str, object]:
    """Return the canonical non-scientific identity for this benchmark."""

    contract = excluded_capacity_contract()
    return {
        "payload": dict(contract.payload),
        "payload_sha256": contract.payload_hash,
        "raw_identity": contract.raw_identity,
        "analysis_identity": contract.analysis_identity,
        "scientific_status": "excluded_non_scientific",
        "production_compatible": False,
    }


def validate_benchmark_domains() -> None:
    schedule = full_schedule()
    scientific_ids = {spec.graph_id for spec in schedule}
    scientific_seeds = {
        value
        for spec in schedule
        for value in (
            seeds_for_graph(spec).graph,
            seeds_for_graph(spec).embedding_provenance,
            seeds_for_graph(spec).pairs,
        )
    }
    specs = benchmark_specs()
    identities = [spec.graph_id for spec in specs]
    seeds = [value for spec in specs for value in (spec.graph_seed, spec.pair_seed)]
    if len(specs) != 6 or len(set(identities)) != 6 or len(set(seeds)) != 12:
        raise Iteration2CapacityError("capacity benchmark identities are not unique")
    if scientific_ids.intersection(identities):
        raise Iteration2CapacityError(
            "capacity benchmark identity overlaps the scientific schedule"
        )
    if scientific_seeds.intersection(seeds):
        raise Iteration2CapacityError(
            "capacity benchmark seed overlaps the scientific schedule"
        )
    if any(not graph_id.startswith("excluded_i2_capacity_") for graph_id in identities):
        raise Iteration2CapacityError("capacity benchmark identity is not excluded")
    excluded_identity = excluded_capacity_identity()
    if (
        excluded_identity["raw_identity"] == ITERATION2_RUN_IDENTITY
        or not str(excluded_identity["raw_identity"]).startswith(
            "iteration2_excluded_raw_"
        )
        or excluded_identity["scientific_status"] != "excluded_non_scientific"
        or excluded_identity["production_compatible"] is not False
    ):
        raise Iteration2CapacityError(
            "capacity benchmark raw identity is not excluded"
        )


def frozen_workload() -> dict[str, object]:
    coordinate_routes = (
        ITERATION2_GRAPH_COUNT
        * PAIRS_PER_GRAPH
        * PRIMARY_COORDINATE_ROUTES_PER_PAIR
    )
    route_records = (
        ITERATION2_GRAPH_COUNT * PAIRS_PER_GRAPH * ROUTE_RECORDS_PER_PAIR
    )
    full_oracle_pairs = FULL_ORACLE_GRAPH_COUNT * PAIRS_PER_GRAPH
    sentinel_pairs = STANDARD_SENTINEL_GRAPH_COUNT * 5
    return {
        "graph_count": ITERATION2_GRAPH_COUNT,
        "pairs_per_graph": PAIRS_PER_GRAPH,
        "total_pairs": ITERATION2_GRAPH_COUNT * PAIRS_PER_GRAPH,
        "routable_coordinate_condition_count": 9,
        "embedding_artifact_count_per_graph": EMBEDDING_ARTIFACTS_PER_GRAPH,
        "primary_coordinate_routes_per_pair": PRIMARY_COORDINATE_ROUTES_PER_PAIR,
        "native_mds_reference_routes_per_pair": (
            NATIVE_MDS_REFERENCE_ROUTES_PER_PAIR
        ),
        "route_records_per_pair": ROUTE_RECORDS_PER_PAIR,
        "total_primary_coordinate_routes": coordinate_routes,
        "total_route_records": route_records,
        "full_oracle_graph_count": FULL_ORACLE_GRAPH_COUNT,
        "standard_sentinel_graph_count": STANDARD_SENTINEL_GRAPH_COUNT,
        "full_oracle_pairs": full_oracle_pairs,
        "standard_sentinel_pairs": sentinel_pairs,
        "total_independently_checked_pairs": full_oracle_pairs + sentinel_pairs,
        "full_oracle_route_decisions": full_oracle_pairs * ROUTE_RECORDS_PER_PAIR,
        "standard_sentinel_route_decisions": (
            sentinel_pairs * ROUTE_RECORDS_PER_PAIR
        ),
        "matched_condition_ids": list(MATCHED_CONDITIONS),
        "benchmark_m": BENCHMARK_M,
        "schedule_graphs_per_model_n_cell": GRAPH_REPETITIONS * 3,
        "full_oracle_graphs_per_model_n_cell": 2,
        "standard_graphs_per_model_n_cell": 58,
    }


def _forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_OUTCOME_KEYS:
                found.add(lowered)
            found.update(_forbidden_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _records_by_cell(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, Mapping[str, object]]]:
    expected = {spec.cell_id: spec for spec in benchmark_specs()}
    grouped: dict[str, dict[str, Mapping[str, object]]] = {}
    if len(records) != len(expected) * len(REPETITION_ROLES):
        raise Iteration2CapacityError(
            "capacity profile must contain exactly 18 benchmark repetitions"
        )
    for record in records:
        cell = str(record.get("cell_id", ""))
        role = str(record.get("repetition_role", ""))
        if cell not in expected or role not in REPETITION_ROLES:
            raise Iteration2CapacityError("capacity repetition identity is invalid")
        if role in grouped.setdefault(cell, {}):
            raise Iteration2CapacityError("duplicate capacity repetition role")
        spec = expected[cell]
        expected_identity = spec.identity()
        for key, value in expected_identity.items():
            if record.get(key) != value:
                raise Iteration2CapacityError(
                    f"capacity repetition {key} differs from excluded specification"
                )
        expected_oracle = role == MEASURED_ORACLE_ROLE
        expected_pairs_checked = PAIRS_PER_GRAPH if expected_oracle else 5
        excluded_identity = excluded_capacity_identity()
        if (
            record.get("pair_count") != PAIRS_PER_GRAPH
            or record.get("route_record_count")
            != PAIRS_PER_GRAPH * ROUTE_RECORDS_PER_PAIR
            or record.get("primary_coordinate_route_count")
            != PAIRS_PER_GRAPH * PRIMARY_COORDINATE_ROUTES_PER_PAIR
            or record.get("embedding_artifact_count")
            != EMBEDDING_ARTIFACTS_PER_GRAPH
            or record.get("independently_checked_pair_count")
            != expected_pairs_checked
            or record.get("independently_checked_route_decisions")
            != expected_pairs_checked * ROUTE_RECORDS_PER_PAIR
            or record.get("full_oracle") is not expected_oracle
            or record.get("checkpoint_validation_passed") is not True
            or record.get("temporary_output_removed") is not True
            or record.get("scientific_result_created") is not False
            or record.get("run_identity") != excluded_identity["raw_identity"]
            or record.get("scientific_status")
            != excluded_identity["scientific_status"]
        ):
            raise Iteration2CapacityError(
                "capacity repetition did not exercise the frozen workload"
            )
        for key in ("end_to_end_ns", "checkpoint_bytes"):
            if not isinstance(record.get(key), int) or int(record[key]) <= 0:
                raise Iteration2CapacityError(
                    f"capacity repetition has invalid {key}"
                )
        for key in (
            "graph_execution_ns",
            "serialization_ns",
            "atomic_publication_ns",
            "checkpoint_validation_ns",
            "uncompressed_result_bytes",
        ):
            if key in record and (
                not isinstance(record[key], int) or int(record[key]) <= 0
            ):
                raise Iteration2CapacityError(
                    f"capacity repetition has invalid optional {key}"
                )
        if (
            record.get("measurement_worker_sha256")
            != measurement_worker_fingerprint()
            or not isinstance(record.get("measurement_resolution_ns"), int)
            or int(record["measurement_resolution_ns"]) <= 0
            or not isinstance(record.get("component_timings_retained"), bool)
        ):
            raise Iteration2CapacityError(
                "capacity repetition measurement provenance is invalid"
            )
        grouped[cell][role] = record
    if any(set(items) != set(REPETITION_ROLES) for items in grouped.values()):
        raise Iteration2CapacityError(
            "capacity profile lacks warm-up or measured repetitions"
        )
    return grouped


def project_runtime(
    records: Sequence[Mapping[str, object]],
    publication_proxy: Mapping[str, object],
) -> dict[str, object]:
    grouped = _records_by_cell(records)
    contributions = []
    nominal = 0
    for spec in benchmark_specs():
        standard = int(
            grouped[spec.cell_id][MEASURED_STANDARD_ROLE]["end_to_end_ns"]
        )
        oracle = int(
            grouped[spec.cell_id][MEASURED_ORACLE_ROLE]["end_to_end_ns"]
        )
        contribution = 58 * standard + 2 * oracle
        nominal += contribution
        contributions.append(
            {
                "cell_id": spec.cell_id,
                "standard_graph_count": 58,
                "full_oracle_graph_count": 2,
                "measured_standard_end_to_end_ns": standard,
                "measured_full_oracle_end_to_end_ns": oracle,
                "projected_cell_runtime_ns": contribution,
            }
        )
    publication_ns = int(publication_proxy["end_to_end_ns"])
    nominal_with_publication = nominal + publication_ns
    conservative = int(
        nominal_with_publication * CONSERVATIVE_RUNTIME_FACTOR
    )
    return {
        "projection_formula": (
            "sum_over_model_n(58*measured_standard_ns"
            "+2*measured_full_oracle_ns)+publication_proxy_ns"
        ),
        "conservative_runtime_factor": CONSERVATIVE_RUNTIME_FACTOR,
        "cell_contributions": contributions,
        "publication_proxy_ns": publication_ns,
        "nominal_projected_runtime_ns": nominal_with_publication,
        "nominal_projected_runtime_seconds": nominal_with_publication / 1e9,
        "nominal_projected_runtime_hours": nominal_with_publication / 3.6e12,
        "conservative_projected_runtime_ns": conservative,
        "conservative_projected_runtime_seconds": conservative / 1e9,
        "conservative_projected_runtime_hours": conservative / 3.6e12,
    }


def project_storage(
    records: Sequence[Mapping[str, object]],
    publication_proxy: Mapping[str, object],
) -> dict[str, object]:
    grouped = _records_by_cell(records)
    contributions = []
    checkpoints = 0
    largest_checkpoint = 0
    largest_uncompressed = 0
    for spec in benchmark_specs():
        standard = grouped[spec.cell_id][MEASURED_STANDARD_ROLE]
        oracle = grouped[spec.cell_id][MEASURED_ORACLE_ROLE]
        standard_bytes = int(standard["checkpoint_bytes"])
        oracle_bytes = int(oracle["checkpoint_bytes"])
        contribution = 58 * standard_bytes + 2 * oracle_bytes
        checkpoints += contribution
        largest_checkpoint = max(
            largest_checkpoint,
            standard_bytes,
            oracle_bytes,
        )
        largest_uncompressed = max(
            largest_uncompressed,
            int(standard.get("uncompressed_result_bytes", 0)),
            int(oracle.get("uncompressed_result_bytes", 0)),
        )
        contributions.append(
            {
                "cell_id": spec.cell_id,
                "standard_graph_count": 58,
                "full_oracle_graph_count": 2,
                "measured_standard_checkpoint_bytes": standard_bytes,
                "measured_full_oracle_checkpoint_bytes": oracle_bytes,
                "projected_cell_checkpoint_bytes": contribution,
            }
        )
    proxy_machine = int(publication_proxy["machine_readable_bytes"])
    proxy_workbook = int(publication_proxy["workbook_bytes"])
    proxy_figures = int(publication_proxy["figure_bytes"])
    machine = max(
        proxy_machine * PUBLICATION_MACHINE_READABLE_FACTOR,
        DERIVED_MACHINE_READABLE_FLOOR_BYTES,
    )
    workbook = max(
        proxy_workbook * PUBLICATION_WORKBOOK_FACTOR,
        WORKBOOK_FLOOR_BYTES,
    )
    figures = max(
        proxy_figures * PUBLICATION_FIGURE_FACTOR,
        FIGURE_FLOOR_BYTES,
    )
    run_level = RUN_LEVEL_FLOOR_BYTES
    raw = checkpoints + run_level
    derived = machine + workbook + figures
    final_storage = raw + derived
    safe_resume = raw
    atomic_peak = largest_checkpoint
    required = (
        final_storage
        + safe_resume
        + atomic_peak
        + FIXED_FREE_SPACE_RESERVE_BYTES
    )
    return {
        "checkpoint_projection_formula": (
            "sum_over_model_n(58*standard_checkpoint_bytes"
            "+2*full_oracle_checkpoint_bytes)"
        ),
        "checkpoint_contributions": contributions,
        "projected_raw_checkpoint_bytes": checkpoints,
        "projected_run_level_bytes": run_level,
        "projected_raw_storage_bytes": raw,
        "publication_proxy_machine_readable_bytes": proxy_machine,
        "publication_proxy_workbook_bytes": proxy_workbook,
        "publication_proxy_figure_bytes": proxy_figures,
        "projected_derived_analysis_bytes": machine,
        "projected_workbook_bytes": workbook,
        "projected_figure_bytes": figures,
        "projected_derived_storage_bytes": derived,
        "projected_final_storage_bytes": final_storage,
        "safe_resume_overhead_bytes": safe_resume,
        "atomic_checkpoint_peak_overhead_bytes": atomic_peak,
        "largest_uncompressed_result_bytes": largest_uncompressed,
        "fixed_free_space_reserve_bytes": FIXED_FREE_SPACE_RESERVE_BYTES,
        "required_free_bytes_formula": (
            "projected_final_storage_bytes+safe_resume_overhead_bytes"
            "+atomic_checkpoint_peak_overhead_bytes"
            "+fixed_free_space_reserve_bytes"
        ),
        "required_free_bytes": required,
    }


def build_profile(
    *,
    records: Sequence[Mapping[str, object]],
    publication_proxy: Mapping[str, object],
    benchmark_volume_identifier: str,
    available_before_bytes: int | None,
    available_after_bytes: int,
    root: Path | str | None = None,
    measurement_recovery: Mapping[str, object] | None = None,
) -> dict[str, object]:
    project = repository_root() if root is None else Path(root).resolve()
    runtime = project_runtime(records, publication_proxy)
    storage = project_storage(records, publication_proxy)
    source_manifest = performance_source_manifest(project)
    dependencies = dependency_fingerprint(project)
    profile: dict[str, object] = {
        "profile_schema": ITERATION2_CAPACITY_PROFILE_SCHEMA,
        "benchmark_schema": BENCHMARK_SCHEMA,
        "profile_label": PROFILE_LABEL,
        "non_scientific": True,
        "excluded_from_analysis": True,
        "scientific_results_created": False,
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": COMBINED_PROTOCOL_HASH,
        "result_schema": ITERATION2_RESULT_SCHEMA,
        "output_schema": ITERATION2_OUTPUT_SCHEMA,
        "output_schema_hash": OUTPUT_SCHEMA_HASH,
        "runner_source_sha256": runner_source_fingerprint(project),
        "benchmark_code_sha256": benchmark_code_fingerprint(project),
        "measurement_worker_sha256": measurement_worker_fingerprint(),
        "performance_source_manifest": source_manifest,
        "performance_source_fingerprint": sha256(
            canonical_bytes(source_manifest)
        ).hexdigest(),
        "dependency_fingerprint": dependencies,
        "volume_identifier": benchmark_volume_identifier,
        "benchmark_volume_identifier": benchmark_volume_identifier,
        "future_output_volume_identifier": benchmark_volume_identifier,
        "available_before_bytes": available_before_bytes,
        "available_after_bytes": available_after_bytes,
        "environment_compatibility": environment_identity(),
        "workload": frozen_workload(),
        "benchmark_graphs": [spec.identity() for spec in benchmark_specs()],
        "excluded_benchmark_identity": excluded_capacity_identity(),
        "repetition_policy": {
            "roles": list(REPETITION_ROLES),
            "warmup_in_projection": False,
            "measured_standard_in_projection": True,
            "measured_full_oracle_in_projection": True,
            "pairs_per_repetition": PAIRS_PER_GRAPH,
        },
        "raw_repetitions": [dict(record) for record in records],
        "publication_proxy": dict(publication_proxy),
        "measurement_recovery": (
            None
            if measurement_recovery is None
            else bind_recovery_evidence(measurement_recovery, records)
        ),
        "runtime_projection": runtime,
        "storage_projection": storage,
        PROFILE_INTERNAL_HASH_FIELD: "",
    }
    profile[PROFILE_INTERNAL_HASH_FIELD] = profile_sha256(profile)
    validate_capacity_profile(
        profile,
        root=project,
        expected_volume_identifier=benchmark_volume_identifier,
        current_available_bytes=available_after_bytes,
    )
    return profile


def validate_capacity_profile(
    profile: Mapping[str, object],
    *,
    root: Path | str | None = None,
    expected_volume_identifier: str | None = None,
    current_available_bytes: int | None = None,
) -> dict[str, object]:
    project = repository_root() if root is None else Path(root).resolve()
    canonical_bytes(profile)
    if profile.get("profile_schema") != ITERATION2_CAPACITY_PROFILE_SCHEMA:
        raise Iteration2CapacityError(
            "Iteration 2 capacity profile schema mismatch"
        )
    if profile.get("benchmark_schema") != BENCHMARK_SCHEMA:
        raise Iteration2CapacityError("capacity benchmark schema mismatch")
    if (
        profile.get("non_scientific") is not True
        or profile.get("excluded_from_analysis") is not True
        or profile.get("scientific_results_created") is not False
    ):
        raise Iteration2CapacityError(
            "capacity profile is not marked excluded and non-scientific"
        )
    expected_identities = {
        "protocol hash": (profile.get("protocol_hash"), COMBINED_PROTOCOL_HASH),
        "result schema": (profile.get("result_schema"), ITERATION2_RESULT_SCHEMA),
        "output schema": (profile.get("output_schema"), ITERATION2_OUTPUT_SCHEMA),
        "output-schema hash": (
            profile.get("output_schema_hash"),
            OUTPUT_SCHEMA_HASH,
        ),
        "runner source": (
            profile.get("runner_source_sha256"),
            runner_source_fingerprint(project),
        ),
        "benchmark code": (
            profile.get("benchmark_code_sha256"),
            benchmark_code_fingerprint(project),
        ),
        "measurement worker": (
            profile.get("measurement_worker_sha256"),
            measurement_worker_fingerprint(),
        ),
        "performance source manifest": (
            profile.get("performance_source_manifest"),
            performance_source_manifest(project),
        ),
        "performance source fingerprint": (
            profile.get("performance_source_fingerprint"),
            performance_source_fingerprint(project),
        ),
        "dependency fingerprint": (
            profile.get("dependency_fingerprint"),
            dependency_fingerprint(project),
        ),
        "frozen workload": (profile.get("workload"), frozen_workload()),
        "benchmark identities": (
            profile.get("benchmark_graphs"),
            [spec.identity() for spec in benchmark_specs()],
        ),
        "excluded benchmark identity": (
            profile.get("excluded_benchmark_identity"),
            excluded_capacity_identity(),
        ),
    }
    for label, (observed, expected) in expected_identities.items():
        if observed != expected:
            raise Iteration2CapacityError(f"capacity profile {label} mismatch")
    recorded_environment = profile.get("environment_compatibility")
    current_environment = environment_identity()
    if (
        not isinstance(recorded_environment, Mapping)
        or recorded_environment.get("schema") != ENVIRONMENT_IDENTITY_SCHEMA
        or recorded_environment.get("hard_requirements")
        != current_environment["hard_requirements"]
        or not isinstance(recorded_environment.get("informational"), Mapping)
    ):
        raise Iteration2CapacityError(
            "capacity profile hard environment compatibility mismatch"
        )
    records = profile.get("raw_repetitions")
    publication = profile.get("publication_proxy")
    if not isinstance(records, list) or not isinstance(publication, Mapping):
        raise Iteration2CapacityError(
            "capacity profile measurements are missing or malformed"
        )
    _records_by_cell(records)
    if (
        publication.get("temporary_output_removed") is not True
        or publication.get("scientific_result_created") is not False
        or publication.get("run_identity")
        != excluded_capacity_identity()["raw_identity"]
        or publication.get("analysis_identity")
        != excluded_capacity_identity()["analysis_identity"]
        or publication.get("scientific_status") != "excluded_non_scientific"
        or publication.get("production_compatible") is not False
        or publication.get("workbook_sheet_count") != len(WORKBOOK_SHEETS)
        or publication.get("figure_count") != len(FIGURE_FILES)
    ):
        raise Iteration2CapacityError(
            "capacity publication proxy is incomplete"
        )
    ledger_snapshot = publication.get("scientific_operation_ledger")
    if not isinstance(ledger_snapshot, Mapping):
        raise Iteration2CapacityError(
            "capacity publication proxy ledger is missing"
        )
    require_zero_scientific_operations(
        ledger_snapshot,
        context="Iteration 2 capacity publication proxy",
    )
    for key in (
        "end_to_end_ns",
        "machine_readable_bytes",
        "workbook_bytes",
        "figure_bytes",
    ):
        if not isinstance(publication.get(key), int) or int(publication[key]) <= 0:
            raise Iteration2CapacityError(
                f"capacity publication proxy has invalid {key}"
            )
    if profile.get("runtime_projection") != project_runtime(records, publication):
        raise Iteration2CapacityError("capacity runtime projection mismatch")
    storage = project_storage(records, publication)
    if profile.get("storage_projection") != storage:
        raise Iteration2CapacityError("capacity storage projection mismatch")
    if profile.get(PROFILE_INTERNAL_HASH_FIELD) != profile_sha256(profile):
        raise Iteration2CapacityError("capacity profile SHA-256 mismatch")
    recovery = profile.get("measurement_recovery")
    if recovery is not None:
        if not isinstance(recovery, Mapping):
            raise Iteration2CapacityError(
                "capacity recovery evidence is malformed"
            )
        evidence = recovery_evidence_payload(records)
        encoded_evidence = canonical_bytes(evidence)
        if (
            recovery.get("evidence_schema") != RECOVERY_EVIDENCE_SCHEMA
            or recovery.get("evidence_sha256")
            != sha256(encoded_evidence).hexdigest()
            or recovery.get("evidence_byte_size") != len(encoded_evidence)
            or recovery.get("evidence_record_count") != len(records)
            or recovery.get("evidence_path_recorded") is not False
            or recovery.get("all_18_graph_workers_completed") is not True
            or recovery.get("checkpoint_sizes_exact") is not True
            or recovery.get("component_subtimings_retained") is not False
            or recovery.get("graph_workloads_rerun") is not False
        ):
            raise Iteration2CapacityError(
                "capacity recovery evidence does not match measurements"
            )
    if _forbidden_keys(profile):
        raise Iteration2CapacityError(
            "capacity profile contains forbidden routing outcomes"
        )
    benchmark_volume = profile.get("volume_identifier")
    if not isinstance(benchmark_volume, str) or not benchmark_volume:
        raise Iteration2CapacityError("capacity profile volume is invalid")
    if (
        expected_volume_identifier is not None
        and benchmark_volume != expected_volume_identifier
    ):
        raise Iteration2CapacityError(
            "output volume differs from Iteration 2 benchmark volume"
        )
    if (
        profile.get("benchmark_volume_identifier") != benchmark_volume
        or profile.get("future_output_volume_identifier") != benchmark_volume
    ):
        raise Iteration2CapacityError(
            "benchmark and future output volume identities are inconsistent"
        )
    required = int(storage["required_free_bytes"])
    if current_available_bytes is not None and current_available_bytes < required:
        raise Iteration2CapacityError(
            "available output-volume space is below Iteration 2 "
            "required_free_bytes"
        )
    return dict(profile)


def load_capacity_profile(
    path: Path | str | None = None,
    *,
    root: Path | str | None = None,
    expected_volume_identifier: str | None = None,
    current_available_bytes: int | None = None,
) -> dict[str, object]:
    profile_path = default_profile_path() if path is None else Path(path)
    if not profile_path.is_file():
        raise Iteration2CapacityError(
            f"Iteration 2 capacity profile is missing: {profile_path}"
        )
    try:
        value = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Iteration2CapacityError(
            "Iteration 2 capacity profile JSON is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise Iteration2CapacityError(
            "Iteration 2 capacity profile must be an object"
        )
    return validate_capacity_profile(
        value,
        root=root,
        expected_volume_identifier=expected_volume_identifier,
        current_available_bytes=current_available_bytes,
    )


def write_capacity_profile(
    profile: Mapping[str, object],
    path: Path | str | None = None,
) -> Path:
    target = (
        default_profile_path()
        if path is None
        else Path(path).resolve(strict=False)
    )
    if target.exists():
        raise FileExistsError(f"capacity profile already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(profile),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _gzip_bytes(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        mtime=0,
    ) as stream:
        stream.write(payload)
    return output.getvalue()


def _verified_worker_root(path: Path | str) -> Path:
    root = Path(path).resolve(strict=True)
    temporary_parent = Path(gettempdir()).resolve(strict=True)
    try:
        root.relative_to(temporary_parent)
    except ValueError as exc:
        raise Iteration2CapacityError(
            "benchmark worker directory is outside the system temporary root"
        ) from exc
    if root == temporary_parent or root.is_symlink():
        raise Iteration2CapacityError("benchmark worker directory is unsafe")
    if any(root.iterdir()):
        raise Iteration2CapacityError("benchmark worker directory is not empty")
    return root


def _generate_graph(spec: BenchmarkSpec):
    if spec.model == "erdos_renyi":
        numerator = 2 * spec.m * (spec.n - spec.m)
        denominator = spec.n * (spec.n - 1)
        return generate_connected_erdos_renyi(
            n=spec.n,
            p=numerator / denominator,
            graph_seed=spec.graph_seed,
            replicate_index=0,
            max_attempts=50,
            p_exact_numerator=numerator,
            p_exact_denominator=denominator,
        )
    if spec.model == "barabasi_albert":
        return generate_connected_barabasi_albert(
            n=spec.n,
            m=spec.m,
            graph_seed=spec.graph_seed,
            replicate_index=0,
        )
    raise Iteration2CapacityError(f"unknown benchmark model: {spec.model}")


def _worker_record(
    spec: BenchmarkSpec,
    role: str,
    temporary_root: Path | str,
) -> dict[str, object]:
    if role not in REPETITION_ROLES:
        raise Iteration2CapacityError("unknown benchmark repetition role")
    root = _verified_worker_root(temporary_root)
    full_oracle = role == MEASURED_ORACLE_ROLE
    excluded_identity = excluded_capacity_identity()
    total_started = perf_counter_ns()
    print("CAPACITY_WORKER graph_generation_started", flush=True)
    generated = _generate_graph(spec)
    print("CAPACITY_WORKER graph_execution_started", flush=True)
    execution_started = perf_counter_ns()
    result = execute_iteration2_graph(
        generated.graph,
        graph_id=spec.graph_id,
        model=spec.model,
        n=spec.n,
        m=spec.m,
        replicate_index=0,
        pair_seed=spec.pair_seed,
        pair_count=PAIRS_PER_GRAPH,
        graph_seed=spec.graph_seed,
        embedding_provenance_seed=None,
        generation_metadata=generated.metadata,
        audit_all_pairs=full_oracle,
        run_identity=str(excluded_identity["raw_identity"]),
    )
    graph_execution_ns = perf_counter_ns() - execution_started
    validate_iteration2_graph_result(result)
    if (
        result.get("run_identity") != excluded_identity["raw_identity"]
        or result.get("scientific_status")
        != excluded_identity["scientific_status"]
    ):
        raise Iteration2CapacityError(
            "benchmark execution did not retain its excluded identity"
        )
    print("CAPACITY_WORKER serialization_started", flush=True)
    serialization_started = perf_counter_ns()
    plain = _json_bytes(result)
    compressed = _gzip_bytes(plain)
    serialization_ns = perf_counter_ns() - serialization_started
    final_path = root / "excluded_checkpoint.json.gz"
    temporary = root / f".checkpoint.tmp-{uuid4().hex}"
    publication_started = perf_counter_ns()
    with temporary.open("xb") as stream:
        stream.write(compressed)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, final_path)
    atomic_publication_ns = perf_counter_ns() - publication_started
    print("CAPACITY_WORKER checkpoint_validation_started", flush=True)
    validation_started = perf_counter_ns()
    with gzip.open(final_path, "rt", encoding="utf-8") as stream:
        restored = json.load(stream)
    validate_iteration2_graph_result(restored)
    checkpoint_validation_ns = perf_counter_ns() - validation_started
    sentinel = result["high_precision_sentinel"]
    expected_checked_pairs = PAIRS_PER_GRAPH if full_oracle else 5
    if (
        len(result["coordinates"]) != EMBEDDING_ARTIFACTS_PER_GRAPH
        or len(result["route_records"])
        != PAIRS_PER_GRAPH * ROUTE_RECORDS_PER_PAIR
        or len(sentinel["pair_indices"]) != expected_checked_pairs
        or sentinel["route_decisions_checked"]
        != expected_checked_pairs * ROUTE_RECORDS_PER_PAIR
    ):
        raise Iteration2CapacityError(
            "benchmark execution did not cover the Iteration 2 workload"
        )
    checkpoint_bytes = final_path.stat().st_size
    end_to_end_ns = perf_counter_ns() - total_started
    record = {
        **spec.identity(),
        "run_identity": result["run_identity"],
        "scientific_status": result["scientific_status"],
        "repetition_role": role,
        "warmup": role == WARMUP_ROLE,
        "included_in_runtime_projection": role != WARMUP_ROLE,
        "full_oracle": full_oracle,
        "pair_count": PAIRS_PER_GRAPH,
        "primary_coordinate_route_count": (
            PAIRS_PER_GRAPH * PRIMARY_COORDINATE_ROUTES_PER_PAIR
        ),
        "route_record_count": PAIRS_PER_GRAPH * ROUTE_RECORDS_PER_PAIR,
        "embedding_artifact_count": len(result["coordinates"]),
        "independent_embedding_validation_exercised": True,
        "independently_checked_pair_count": expected_checked_pairs,
        "independently_checked_route_decisions": sentinel[
            "route_decisions_checked"
        ],
        "graph_execution_ns": graph_execution_ns,
        "serialization_ns": serialization_ns,
        "atomic_publication_ns": atomic_publication_ns,
        "checkpoint_validation_ns": checkpoint_validation_ns,
        "end_to_end_ns": end_to_end_ns,
        "checkpoint_bytes": checkpoint_bytes,
        "uncompressed_result_bytes": len(plain),
        "measurement_worker_sha256": measurement_worker_fingerprint(),
        "measurement_resolution_ns": 1,
        "component_timings_retained": True,
        "checkpoint_validation_passed": True,
        "temporary_output_removed": True,
        "scientific_result_created": False,
    }
    shutil.rmtree(root)
    if root.exists():
        raise Iteration2CapacityError(
            "successful benchmark temporary output was not removed"
        )
    return record


def _write_failure_evidence(root: Path, message: str) -> None:
    try:
        (root / "capacity_failure.txt").write_text(
            message + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _worker_main(args: argparse.Namespace) -> int:
    specs = {spec.cell_id: spec for spec in benchmark_specs()}
    spec = specs[args.cell]
    root = Path(args.temporary_root)
    try:
        record = _worker_record(spec, args.role, root)
    except Exception as exc:
        _write_failure_evidence(
            root,
            f"{type(exc).__name__}: {exc}",
        )
        raise
    print("CAPACITY_RECORD " + _json_bytes(record).decode("utf-8"), flush=True)
    return 0


def watchdog_seconds(spec: BenchmarkSpec, role: str) -> int:
    standard = {100: 600, 300: 1_200, 1_000: 3_600}
    oracle = {100: 1_200, 300: 2_400, 1_000: 7_200}
    return (oracle if role == MEASURED_ORACLE_ROLE else standard)[spec.n]


def _run_worker(spec: BenchmarkSpec, role: str) -> dict[str, object]:
    root = Path(
        mkdtemp(
            prefix=f"excluded-i2-capacity-{spec.cell_id}-{role}-",
            dir=gettempdir(),
        )
    ).resolve(strict=True)
    if volume_identifier(root) != volume_identifier(repository_root()):
        _write_failure_evidence(root, "temporary and output volumes differ")
        raise Iteration2CapacityError(
            f"capacity benchmark temporary volume mismatch; preserved {root}"
        )
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "_worker",
        "--cell",
        spec.cell_id,
        "--role",
        role,
        "--temporary-root",
        str(root),
    ]
    timeout_seconds = watchdog_seconds(spec, role)
    process = subprocess.Popen(
        command,
        cwd=repository_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    started = perf_counter()
    next_progress = 15.0
    while process.poll() is None:
        elapsed = perf_counter() - started
        if elapsed > timeout_seconds:
            process.kill()
            output, _ = process.communicate(timeout=30)
            _write_failure_evidence(
                root,
                f"watchdog timeout after {timeout_seconds}s\n{output}",
            )
            raise Iteration2CapacityError(
                "capacity watchdog timeout; diagnostic evidence preserved at "
                f"{root}"
            )
        if elapsed >= next_progress:
            print(
                "ITERATION2_CAPACITY_PROGRESS "
                f"cell={spec.cell_id} role={role} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
            next_progress += 15.0
        sleep(0.25)
    output, _ = process.communicate(timeout=30)
    if process.returncode != 0:
        _write_failure_evidence(
            root,
            f"worker exit={process.returncode}\n{output}",
        )
        raise Iteration2CapacityError(
            "capacity worker failed; diagnostic evidence preserved at "
            f"{root}"
        )
    prefix = "CAPACITY_RECORD "
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise Iteration2CapacityError(
            "capacity worker returned no unique measurement record"
        )
    if root.exists():
        raise Iteration2CapacityError(
            "successful capacity worker left temporary output"
        )
    record = json.loads(lines[0][len(prefix) :])
    print(
        "ITERATION2_CAPACITY_COMPLETED "
        f"cell={spec.cell_id} role={role} "
        f"seconds={int(record['end_to_end_ns']) / 1e9:.6f} "
        f"checkpoint_bytes={record['checkpoint_bytes']}",
        flush=True,
    )
    return record


def _publication_tables() -> dict[str, list[dict[str, object]]]:
    return {name: [] for name in WORKBOOK_SHEETS}


def _tree_sizes(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_publication_proxy() -> dict[str, object]:
    root = Path(
        mkdtemp(prefix="excluded-i2-capacity-publication-", dir=gettempdir())
    ).resolve(strict=True)
    temporary_parent = Path(gettempdir()).resolve(strict=True)
    try:
        root.relative_to(temporary_parent)
    except ValueError as exc:
        raise Iteration2CapacityError(
            "publication proxy is outside the temporary root"
        ) from exc
    excluded_identity = excluded_capacity_identity()
    output = root / str(excluded_identity["analysis_identity"])
    started = perf_counter_ns()
    try:
        with scientific_operation_context(PREFLIGHT_READ_ONLY) as ledger:
            initial_ledger = ledger.snapshot()
            manifest = build_reporting_bundle(
                output,
                tables=_publication_tables(),
                source_commit="0" * 40,
                raw_location="excluded/non-scientific/capacity",
                raw_file_hashes={"excluded-capacity.json.gz": "0" * 64},
                raw_generation_identity={
                    "run_identity": excluded_identity["raw_identity"],
                    "scientific_status": excluded_identity[
                        "scientific_status"
                    ],
                    "production_compatible": False,
                },
                analysis_validation_evidence={
                    "scientific_operation_ledger": initial_ledger,
                },
                excluded_fixture_payload=excluded_identity["payload"],
                limitations=(
                    "Excluded structural capacity proxy; no routing outcomes.",
                ),
            )
            final_ledger = ledger.snapshot()
            require_zero_scientific_operations(
                final_ledger,
                context="Iteration 2 capacity publication proxy",
            )
        sizes = _tree_sizes(output)
        workbook = sizes.get("iteration2_results.xlsx", 0)
        figures = sum(
            sizes.get(f"figures/{name}", 0) for name in FIGURE_FILES
        )
        machine = sum(sizes.values()) - workbook - figures
        if (
            len(manifest.get("workbook_sheets", ())) != len(WORKBOOK_SHEETS)
            or len(manifest.get("figures", ())) != len(FIGURE_FILES)
            or workbook <= 0
            or figures <= 0
            or machine <= 0
        ):
            raise Iteration2CapacityError(
                "publication proxy did not create the complete reporting bundle"
            )
        record = {
            "run_identity": excluded_identity["raw_identity"],
            "analysis_identity": excluded_identity["analysis_identity"],
            "scientific_status": excluded_identity["scientific_status"],
            "production_compatible": False,
            "scientific_operation_ledger": final_ledger,
            "end_to_end_ns": perf_counter_ns() - started,
            "file_count": len(sizes),
            "machine_readable_bytes": machine,
            "workbook_bytes": workbook,
            "figure_bytes": figures,
            "workbook_sheet_count": len(WORKBOOK_SHEETS),
            "figure_count": len(FIGURE_FILES),
            "temporary_output_removed": True,
            "scientific_result_created": False,
        }
    except Exception as exc:
        _write_failure_evidence(root, f"{type(exc).__name__}: {exc}")
        raise Iteration2CapacityError(
            f"publication proxy failed; diagnostic evidence preserved at {root}"
        ) from exc
    shutil.rmtree(root)
    if root.exists():
        raise Iteration2CapacityError(
            "successful publication proxy left temporary output"
        )
    print(
        "ITERATION2_CAPACITY_PUBLICATION "
        f"seconds={record['end_to_end_ns'] / 1e9:.6f} "
        f"workbook_bytes={record['workbook_bytes']} "
        f"figure_bytes={record['figure_bytes']}",
        flush=True,
    )
    return record


def run_benchmark(
    *,
    profile_path: Path | str | None = None,
) -> dict[str, object]:
    validate_benchmark_domains()
    target = default_profile_path() if profile_path is None else Path(profile_path)
    if target.exists():
        raise FileExistsError(f"capacity profile already exists: {target}")
    benchmark_volume = volume_identifier(repository_root() / "results")
    before = available_bytes(repository_root() / "results")
    records = []
    for spec in benchmark_specs():
        for role in REPETITION_ROLES:
            print(
                "ITERATION2_CAPACITY_START "
                f"cell={spec.cell_id} role={role} "
                f"watchdog_seconds={watchdog_seconds(spec, role)}",
                flush=True,
            )
            records.append(_run_worker(spec, role))
    publication = run_publication_proxy()
    after = available_bytes(repository_root() / "results")
    profile = build_profile(
        records=records,
        publication_proxy=publication,
        benchmark_volume_identifier=benchmark_volume,
        available_before_bytes=before,
        available_after_bytes=after,
    )
    written = write_capacity_profile(profile, target)
    validated = load_capacity_profile(
        written,
        expected_volume_identifier=benchmark_volume,
        current_available_bytes=after,
    )
    return {
        "profile_path": str(written),
        "physical_sha256": physical_profile_sha256(written),
        "internal_sha256": validated[PROFILE_INTERNAL_HASH_FIELD],
        "benchmark_code_sha256": validated["benchmark_code_sha256"],
        "nominal_projected_runtime_seconds": validated["runtime_projection"][
            "nominal_projected_runtime_seconds"
        ],
        "conservative_projected_runtime_seconds": validated[
            "runtime_projection"
        ]["conservative_projected_runtime_seconds"],
        "projected_final_storage_bytes": validated["storage_projection"][
            "projected_final_storage_bytes"
        ],
        "required_free_bytes": validated["storage_projection"][
            "required_free_bytes"
        ],
        "successful_temporary_outputs_removed": True,
        "scientific_results_created": False,
    }


def integrity_report(path: Path | str | None = None) -> dict[str, object]:
    profile_path = default_profile_path() if path is None else Path(path)
    output = repository_root() / "results"
    profile = load_capacity_profile(
        profile_path,
        expected_volume_identifier=volume_identifier(output),
        current_available_bytes=available_bytes(output),
    )
    return {
        "profile_path": str(profile_path.resolve()),
        "physical_sha256": physical_profile_sha256(profile_path),
        "internal_sha256": profile[PROFILE_INTERNAL_HASH_FIELD],
        "benchmark_code_sha256": profile["benchmark_code_sha256"],
        "performance_source_fingerprint": profile[
            "performance_source_fingerprint"
        ],
        "dependency_fingerprint": profile["dependency_fingerprint"]["sha256"],
        "protocol_hash": profile["protocol_hash"],
        "result_schema": profile["result_schema"],
        "output_schema_hash": profile["output_schema_hash"],
        "workload": profile["workload"],
        "runtime_projection": profile["runtime_projection"],
        "storage_projection": profile["storage_projection"],
        "current_available_bytes": available_bytes(output),
        "valid": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Excluded Iteration 2 operational-capacity benchmark"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--profile")
    integrity_parser = subparsers.add_parser("integrity")
    integrity_parser.add_argument("--profile")
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--cell", required=True)
    worker.add_argument("--role", choices=REPETITION_ROLES, required=True)
    worker.add_argument("--temporary-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "_worker":
        return _worker_main(args)
    if args.operation == "run":
        result = run_benchmark(profile_path=args.profile)
    else:
        result = integrity_report(args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
