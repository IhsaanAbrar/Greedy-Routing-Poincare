"""Guarded Iteration 2 runner and excluded in-memory feasibility fixture."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
import errno
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
from threading import Lock
from typing import Iterator, Mapping, Sequence
from uuid import uuid4

from benchmark_iteration2_capacity import (
    available_bytes,
    dependency_fingerprint,
    load_capacity_profile,
    performance_source_manifest,
    performance_source_fingerprint,
    physical_profile_sha256,
    volume_identifier,
)
from graph_generation import (
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
)
from iteration2_config import (
    ANALYSIS_PLAN_HASH,
    COMBINED_PROTOCOL_HASH,
    DATA_GENERATION_HASH,
    EXCLUDED_FIXTURE_SEEDS,
    EQUIVALENCE_MARGIN_APPROVED,
    FULL_RUN_CONFIRMATION_TOKEN,
    GRAPH_REPETITIONS_APPROVED,
    ITERATION2_GRAPH_COUNT,
    ITERATION2_GRAPH_CHECKPOINT_FILES_PER_DIRECTORY,
    ITERATION2_OUTPUT_SCHEMA,
    ITERATION2_RAW_GRAPH_FILE_COUNT,
    ITERATION2_RAW_TOTAL_FILE_COUNT,
    ITERATION2_RUN_IDENTITY,
    OUTPUT_SCHEMA_HASH,
    PAIRS_PER_GRAPH,
    full_schedule,
    resolve_iteration2_output,
)
from iteration2_experiment import (
    execute_iteration2_graph,
    execute_scheduled_graph,
)
from iteration2_excluded import ExcludedAnalysisFixtureContract
from iteration2_runtime_guard import (
    PREFLIGHT_READ_ONLY,
    SCIENTIFIC_REGENERATION_AUDIT,
    current_scientific_ledger,
    require_zero_scientific_operations,
    scientific_operation_boundary,
    scientific_operation_context,
    validate_scientific_boundary_registry,
)
from validate_iteration2 import (
    scheduled_specifications,
    validate_scheduled_iteration2_graph_result,
    validate_iteration2_graph_result,
    verify_iteration1_immutable,
)


MANIFEST_SCHEMA = "greedy_routing_iteration2_run_manifest_v3"
RUN_COMPLETION_SCHEMA = "greedy_routing_iteration2_run_completion_v2"
GRAPH_CHECKPOINT_MANIFEST_SCHEMA = (
    "greedy_routing_iteration2_graph_checkpoint_manifest_v1"
)
GRAPH_CHECKPOINT_COMPLETION_SCHEMA = (
    "greedy_routing_iteration2_graph_checkpoint_completion_v1"
)
GRAPH_RESULT_FILENAME = "result.json.gz"
GRAPH_MANIFEST_FILENAME = "checkpoint_manifest.json"
GRAPH_COMPLETION_FILENAME = "complete.json"
GRAPH_CHECKPOINT_FILENAMES = frozenset(
    {
        GRAPH_RESULT_FILENAME,
        GRAPH_MANIFEST_FILENAME,
        GRAPH_COMPLETION_FILENAME,
    }
)
GRAPH_CHECKPOINT_FILE_COUNT = len(GRAPH_CHECKPOINT_FILENAMES)
if GRAPH_CHECKPOINT_FILE_COUNT != ITERATION2_GRAPH_CHECKPOINT_FILES_PER_DIRECTORY:
    raise RuntimeError("checkpoint filenames disagree with frozen file count")
RAW_RUN_FILE_COUNT = ITERATION2_RAW_TOTAL_FILE_COUNT
FIXTURE_LABEL = (
    "NON-SCIENTIFIC ITERATION 2 FEASIBILITY FIXTURE - EXCLUDED FROM RESULTS"
)
CAPACITY_PROFILE_RELATIVE_PATH = Path("code/iteration2_capacity_profile.json")
RUN_LEASE_SCHEMA = "greedy_routing_iteration2_run_lease"
RUN_LEASE_VERSION = 1
RUN_LEASE_ERROR_CODE = "iteration2_run_already_active"
RUN_LEASE_FILENAME_SUFFIX = ".lease"
_RUN_LEASE_REGISTRY_GUARD = Lock()
_ACTIVE_RUN_LEASES: set[str] = set()


class Iteration2RunAlreadyActive(RuntimeError):
    """Raised when another thread or process owns the requested run lease."""

    def __init__(self, *, run_identity: str, lock_path: Path) -> None:
        self.run_identity = run_identity
        self.lock_path = lock_path
        super().__init__(f"Iteration 2 run already active: {run_identity}")

    def as_dict(self) -> dict[str, object]:
        return {
            "code": RUN_LEASE_ERROR_CODE,
            "message": str(self),
            "run_identity": self.run_identity,
            "lock_path": str(self.lock_path),
        }


def _path_is_reparse_point(path: Path) -> bool:
    """Return true for symbolic links, junctions, and other Windows reparses."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except FileNotFoundError:
        return False
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(reparse_flag and attributes & reparse_flag)


def iteration2_run_lease_path(
    root: Path | str,
    run_identity: str,
) -> Path:
    """Return the contained persistent lock-file path for one run identity."""

    if (
        not run_identity
        or len(run_identity) > 200
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in run_identity
        )
    ):
        raise ValueError("run identity is unsafe for an Iteration 2 lease path")
    project = Path(root).resolve(strict=True)
    results_root = project / "results"
    lock_path = results_root / f".{run_identity}{RUN_LEASE_FILENAME_SUFFIX}"
    if lock_path.parent != results_root:
        raise RuntimeError("Iteration 2 lease path escaped the results root")
    return lock_path


def _prepare_run_lease_path(lock_path: Path) -> None:
    results_root = lock_path.parent
    if results_root.exists():
        if _path_is_reparse_point(results_root) or not results_root.is_dir():
            raise RuntimeError("Iteration 2 lease parent is unsafe")
    else:
        results_root.mkdir(mode=0o700, exist_ok=True)
    if _path_is_reparse_point(results_root) or not results_root.is_dir():
        raise RuntimeError("Iteration 2 lease parent is unsafe")
    if _path_is_reparse_point(lock_path) or (
        lock_path.exists() and not lock_path.is_file()
    ):
        raise RuntimeError("Iteration 2 lease file is unsafe")


def _open_run_lease_file(lock_path: Path):
    flags = os.O_RDWR | os.O_CREAT
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOINHERIT", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        path_status = lock_path.lstat()
        descriptor_status = os.fstat(descriptor)
        if (
            _path_is_reparse_point(lock_path)
            or not stat.S_ISREG(path_status.st_mode)
            or not stat.S_ISREG(descriptor_status.st_mode)
            or not os.path.samestat(path_status, descriptor_status)
        ):
            raise RuntimeError("Iteration 2 lease file is unsafe")
        return os.fdopen(descriptor, "r+b", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def _try_os_lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_os_lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _is_lock_contention(exc: OSError) -> bool:
    return exc.errno in (errno.EACCES, errno.EAGAIN) or getattr(
        exc, "winerror", None
    ) in (32, 33, 36)


class _Iteration2RunLease:
    """Keep one OS lock and its in-process registry claim alive."""

    def __init__(
        self,
        *,
        root: Path | str,
        run_identity: str,
        source_commit: str,
        resume: bool,
    ) -> None:
        self.run_identity = run_identity
        self.path = iteration2_run_lease_path(root, run_identity)
        self.metadata = {
            "schema": RUN_LEASE_SCHEMA,
            "version": RUN_LEASE_VERSION,
            "run_identity": run_identity,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "source_commit": str(source_commit),
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "resume": bool(resume),
        }
        self._registry_key = (
            str(self.path).casefold() if os.name == "nt" else str(self.path)
        )
        self._handle = None
        self._registry_claimed = False

    def acquire(self) -> "_Iteration2RunLease":
        with _RUN_LEASE_REGISTRY_GUARD:
            if self._registry_key in _ACTIVE_RUN_LEASES:
                raise Iteration2RunAlreadyActive(
                    run_identity=self.run_identity,
                    lock_path=self.path,
                )
            _ACTIVE_RUN_LEASES.add(self._registry_key)
            self._registry_claimed = True
        try:
            _prepare_run_lease_path(self.path)
            handle = _open_run_lease_file(self.path)
            try:
                if (
                    _path_is_reparse_point(self.path.parent)
                    or not self.path.parent.is_dir()
                ):
                    raise RuntimeError("Iteration 2 lease parent became unsafe")
                _try_os_lock(handle)
            except OSError as exc:
                handle.close()
                if _is_lock_contention(exc):
                    raise Iteration2RunAlreadyActive(
                        run_identity=self.run_identity,
                        lock_path=self.path,
                    ) from exc
                raise
            except BaseException:
                handle.close()
                raise
            self._handle = handle
            try:
                handle.seek(0)
                handle.truncate()
                handle.write(_json_bytes(self.metadata) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                self.release()
                raise
            return self
        except BaseException:
            if self._registry_claimed:
                with _RUN_LEASE_REGISTRY_GUARD:
                    _ACTIVE_RUN_LEASES.discard(self._registry_key)
                self._registry_claimed = False
            raise

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        try:
            if handle is not None:
                try:
                    _release_os_lock(handle)
                finally:
                    handle.close()
        finally:
            if self._registry_claimed:
                with _RUN_LEASE_REGISTRY_GUARD:
                    _ACTIVE_RUN_LEASES.discard(self._registry_key)
                self._registry_claimed = False


@contextmanager
def acquire_iteration2_run_lease(
    *,
    root: Path | str,
    run_identity: str,
    source_commit: str,
    resume: bool,
) -> Iterator[_Iteration2RunLease]:
    """Acquire a non-blocking process-wide and run-wide exclusive lease."""

    lease = _Iteration2RunLease(
        root=root,
        run_identity=run_identity,
        source_commit=source_commit,
        resume=resume,
    ).acquire()
    try:
        yield lease
    finally:
        lease.release()


class ResumeValidationPolicy(Enum):
    """Explicitly separate structural reads from scientific regeneration."""

    READ_ONLY_STRUCTURAL = "read_only_structural"


SCIENTIFIC_REGENERATION_AUDIT_CONFIRMATION = (
    "I UNDERSTAND THIS REGENERATES ONE ITERATION 2 GRAPH FOR AUDIT ONLY"
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return commit, dirty


def _source_fingerprint(root: Path) -> str:
    return performance_source_fingerprint(root)


def _dependency_fingerprint(root: Path) -> dict[str, object]:
    return dependency_fingerprint(root)


def _capacity_profile_fingerprint(root: Path) -> str:
    return physical_profile_sha256(root / CAPACITY_PROFILE_RELATIVE_PATH)


def _capacity_profile_identity(root: Path) -> dict[str, object]:
    path = root / CAPACITY_PROFILE_RELATIVE_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "physical_sha256": physical_profile_sha256(path),
        "internal_sha256": value.get("profile_sha256"),
        "profile_schema": value.get("profile_schema"),
        "benchmark_schema": value.get("benchmark_schema"),
        "benchmark_code_sha256": value.get("benchmark_code_sha256"),
        "measurement_worker_sha256": value.get("measurement_worker_sha256"),
        "performance_source_fingerprint": value.get(
            "performance_source_fingerprint"
        ),
        "dependency_fingerprint": value.get("dependency_fingerprint"),
        "environment_compatibility": value.get(
            "environment_compatibility"
        ),
        "volume_identifier": value.get("volume_identifier"),
    }


def _capacity_status(root: Path, output: Path) -> dict[str, object]:
    """Validate the Iteration 2-specific benchmark and disk requirement."""

    profile_path = root / CAPACITY_PROFILE_RELATIVE_PATH
    output_volume = volume_identifier(output)
    free_bytes = available_bytes(output)
    profile = load_capacity_profile(
        profile_path,
        root=root,
        expected_volume_identifier=output_volume,
        current_available_bytes=free_bytes,
    )
    storage = profile["storage_projection"]
    runtime = profile["runtime_projection"]
    return {
        "profile_path": CAPACITY_PROFILE_RELATIVE_PATH.as_posix(),
        "profile_sha256": _capacity_profile_fingerprint(root),
        "profile_internal_sha256": profile["profile_sha256"],
        "benchmark_code_sha256": profile["benchmark_code_sha256"],
        "performance_source_fingerprint": profile[
            "performance_source_fingerprint"
        ],
        "protocol_hash": profile["protocol_hash"],
        "result_schema": profile["result_schema"],
        "output_schema_hash": profile["output_schema_hash"],
        "profile_volume_identifier": profile["volume_identifier"],
        "output_volume_identifier": output_volume,
        "current_available_bytes": free_bytes,
        "projected_nominal_runtime_seconds": runtime[
            "nominal_projected_runtime_seconds"
        ],
        "projected_conservative_runtime_seconds": runtime[
            "conservative_projected_runtime_seconds"
        ],
        "projected_raw_storage_bytes": storage[
            "projected_raw_storage_bytes"
        ],
        "projected_derived_storage_bytes": storage[
            "projected_derived_storage_bytes"
        ],
        "projected_final_storage_bytes": storage[
            "projected_final_storage_bytes"
        ],
        "safe_resume_overhead_bytes": storage[
            "safe_resume_overhead_bytes"
        ],
        "fixed_free_space_reserve_bytes": storage[
            "fixed_free_space_reserve_bytes"
        ],
        "required_free_bytes": storage["required_free_bytes"],
        "profile_valid": True,
        "output_volume_matches_profile": True,
        "disk_space_pass": True,
    }


def build_manifest(root: Path | None = None) -> dict[str, object]:
    root = repository_root() if root is None else Path(root).resolve()
    commit, dirty = _git_state(root)
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "run_identity": ITERATION2_RUN_IDENTITY,
        "protocol_hash": COMBINED_PROTOCOL_HASH,
        "data_generation_hash": DATA_GENERATION_HASH,
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "output_schema_hash": OUTPUT_SCHEMA_HASH,
        "output_schema": ITERATION2_OUTPUT_SCHEMA,
        "source_commit": commit,
        "source_worktree": "dirty" if dirty else "clean",
        "source_fingerprint": _source_fingerprint(root),
        "source_manifest": performance_source_manifest(root),
        "dependency_fingerprint": _dependency_fingerprint(root),
        "capacity_profile_sha256": _capacity_profile_fingerprint(root),
        "capacity_profile_identity": _capacity_profile_identity(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schedule": [spec.graph_id for spec in full_schedule()],
        "graph_count": ITERATION2_GRAPH_COUNT,
        "pairs_per_graph": PAIRS_PER_GRAPH,
        "raw_graph_file_count": ITERATION2_RAW_GRAPH_FILE_COUNT,
        "raw_graph_checkpoint_count": ITERATION2_RAW_GRAPH_FILE_COUNT,
        "raw_files_per_graph_checkpoint": GRAPH_CHECKPOINT_FILE_COUNT,
        "raw_total_file_count": RAW_RUN_FILE_COUNT,
        "checkpoint_layout": {
            "schema": GRAPH_CHECKPOINT_MANIFEST_SCHEMA,
            "directory": "graphs/<graph_id>",
            "result": GRAPH_RESULT_FILENAME,
            "manifest": GRAPH_MANIFEST_FILENAME,
            "completion": GRAPH_COMPLETION_FILENAME,
            "publication": "same_parent_atomic_directory_rename",
            "completion_written_last": True,
        },
        "equivalence_margin_human_approved": EQUIVALENCE_MARGIN_APPROVED,
        "graph_repetitions_human_approved": GRAPH_REPETITIONS_APPROVED,
        "scientific_status": "iteration2_prespecified_scientific_run",
        "production_compatible": True,
        "iteration1_protected": True,
        "adaptive_greedy_embedding_included": False,
    }


def _preflight_without_guard(
    *,
    mode: str,
    confirmation: str | None,
    expected_source_commit: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_dependency_fingerprint: str | None = None,
    expected_capacity_profile: str | None = None,
    expected_protocol_hash: str | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Return a read-only authorization report."""

    if mode not in ("development", "full"):
        raise ValueError("mode must be development or full")
    root = repository_root()
    output = resolve_iteration2_output(root, ITERATION2_RUN_IDENTITY)
    manifest = build_manifest(root)
    reasons = []
    if mode != "full":
        reasons.append("full mode was not requested")
    if confirmation != FULL_RUN_CONFIRMATION_TOKEN:
        reasons.append("exact Iteration 2 confirmation token was not supplied")
    if manifest["source_worktree"] != "clean":
        reasons.append("full scientific run requires clean committed source")
    expected_identities = {
        "source commit": (
            expected_source_commit,
            manifest["source_commit"],
        ),
        "source fingerprint": (
            expected_source_fingerprint,
            manifest["source_fingerprint"],
        ),
        "dependency fingerprint": (
            expected_dependency_fingerprint,
            manifest["dependency_fingerprint"]["sha256"],
        ),
        "capacity profile": (
            expected_capacity_profile,
            manifest["capacity_profile_sha256"],
        ),
        "protocol": (
            expected_protocol_hash,
            COMBINED_PROTOCOL_HASH,
        ),
    }
    for label, (expected, observed) in expected_identities.items():
        if expected != observed:
            reasons.append(f"expected {label} does not match current {label}")
    if not EQUIVALENCE_MARGIN_APPROVED:
        reasons.append("provisional equivalence margin requires human approval")
    if not GRAPH_REPETITIONS_APPROVED:
        reasons.append("graph replicate count requires human approval")
    capacity: dict[str, object]
    try:
        capacity = _capacity_status(root, output)
    except (OSError, RuntimeError, ValueError) as exc:
        capacity = {
            "profile_valid": False,
            "disk_space_pass": False,
            "error": str(exc),
        }
        reasons.append(f"capacity validation failed: {exc}")
    checkpoint_validation: dict[str, object] = {
        "resume_requested": resume,
        "deep_validation": False,
        "validated_graph_files": 0,
    }
    if output.exists():
        if not resume:
            reasons.append(
                "Iteration 2 run directory exists; explicit resume is required"
            )
        else:
            try:
                checkpoint_validation = _validate_resume_directory(
                    output,
                    manifest,
                    validation_policy=ResumeValidationPolicy.READ_ONLY_STRUCTURAL,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                reasons.append(f"resume checkpoint validation failed: {exc}")
                checkpoint_validation = {
                    "resume_requested": True,
                    "deep_validation": False,
                    "validated_graph_files": 0,
                    "error": str(exc),
                }
    elif resume:
        reasons.append("resume requested but Iteration 2 run directory is absent")
    iteration1 = verify_iteration1_immutable(root, deep=mode == "full")
    return {
        "operation": "read_only_iteration2_preflight",
        "authorized": not reasons,
        "authorization_reasons": reasons,
        "output_directory": str(output),
        "manifest": manifest,
        "capacity": capacity,
        "checkpoint_validation": checkpoint_validation,
        "iteration1_protection": iteration1,
    }


def preflight(
    *,
    mode: str,
    confirmation: str | None,
    expected_source_commit: str | None = None,
    expected_source_fingerprint: str | None = None,
    expected_dependency_fingerprint: str | None = None,
    expected_capacity_profile: str | None = None,
    expected_protocol_hash: str | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Return a measured, structurally read-only authorization report."""

    validate_scientific_boundary_registry()
    with scientific_operation_context(PREFLIGHT_READ_ONLY) as ledger:
        report = _preflight_without_guard(
            mode=mode,
            confirmation=confirmation,
            expected_source_commit=expected_source_commit,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_dependency_fingerprint=expected_dependency_fingerprint,
            expected_capacity_profile=expected_capacity_profile,
            expected_protocol_hash=expected_protocol_hash,
            resume=resume,
        )
        snapshot = ledger.snapshot()
        require_zero_scientific_operations(snapshot, context="Iteration 2 preflight")
        report["scientific_operation_ledger"] = snapshot
        return report


def _canonical_json_value(value: object) -> object:
    """Normalize JSON object keys before sorting and reject non-finite values."""

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if isinstance(raw_key, str):
                key = raw_key
            elif isinstance(raw_key, int) and not isinstance(raw_key, bool):
                key = str(raw_key)
            else:
                raise TypeError("canonical JSON object keys must be strings or integers")
            if key in normalized:
                raise ValueError(
                    "canonical JSON object keys collide after string conversion"
                )
            normalized[key] = _canonical_json_value(raw_value)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON values must be finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _gzip_bytes(payload: bytes) -> bytes:
    import io

    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        mtime=0,
    ) as stream:
        stream.write(payload)
    return output.getvalue()


def _manifest_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(manifest, Mapping):
        raise RuntimeError("run manifest must be a mapping")
    required = (
        "manifest_schema",
        "run_identity",
        "protocol_hash",
        "data_generation_hash",
        "analysis_plan_hash",
        "output_schema_hash",
        "output_schema",
        "source_commit",
        "source_worktree",
        "source_fingerprint",
        "source_manifest",
        "dependency_fingerprint",
        "capacity_profile_sha256",
        "capacity_profile_identity",
        "graph_count",
        "pairs_per_graph",
        "raw_graph_file_count",
        "raw_graph_checkpoint_count",
        "raw_files_per_graph_checkpoint",
        "raw_total_file_count",
        "checkpoint_layout",
        "equivalence_margin_human_approved",
        "graph_repetitions_human_approved",
        "schedule",
        "scientific_status",
        "production_compatible",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise RuntimeError(
            "run manifest is missing required field(s): " + ", ".join(missing)
        )
    identity = {key: manifest[key] for key in required}
    if manifest["scientific_status"] == "excluded_non_scientific":
        for key in (
            "excluded_fixture_payload",
            "excluded_fixture_payload_sha256",
        ):
            if key not in manifest:
                raise RuntimeError(
                    "excluded run manifest is missing identity field: " + key
                )
            identity[key] = manifest[key]
    elif "excluded_fixture_payload" in manifest:
        raise RuntimeError("production run manifest contains excluded fixture identity")
    if not isinstance(identity["source_manifest"], Mapping):
        raise RuntimeError("run manifest source_manifest must be a mapping")
    return identity


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON file must contain an object: {path.name}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    from hashlib import sha256

    return sha256(payload).hexdigest()


def _manifest_identity_sha256(manifest: Mapping[str, object]) -> str:
    return _sha256_bytes(_json_bytes(_manifest_identity(manifest)))


def _load_gzip_result(path: Path) -> tuple[dict[str, object], bytes]:
    physical = path.read_bytes()
    try:
        plain = gzip.decompress(physical)
        value = json.loads(plain.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"corrupt Iteration 2 checkpoint: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid Iteration 2 checkpoint: {path.name}")
    if _gzip_bytes(_json_bytes(value)) != physical:
        raise RuntimeError(
            "checkpoint gzip bytes are non-deterministic, stale, or non-canonical"
        )
    return value, physical


def _checkpoint_row_counts(result: Mapping[str, object]) -> dict[str, int]:
    fields = (
        "pair_records",
        "dijkstra_records",
        "route_records",
        "graph_level_rows",
        "graph_level_interactions",
    )
    counts: dict[str, int] = {}
    for field in fields:
        rows = result.get(field)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise RuntimeError(f"checkpoint result lacks sequence field: {field}")
        counts[field] = len(rows)
    return counts


def _fsync_directory(path: Path) -> None:
    """Flush directory metadata where the platform exposes directory handles."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_owned_checkpoint_staging(
    temporary: Path,
    *,
    root: Path,
    graph_id: str,
) -> None:
    """Remove only the uniquely named same-parent staging directory we own."""

    expected_prefix = f".{graph_id}.tmp-"
    if (
        temporary.parent != root
        or not temporary.name.startswith(expected_prefix)
        or len(temporary.name) != len(expected_prefix) + 32
    ):
        raise RuntimeError("refusing to clean an unowned checkpoint staging path")
    if _path_is_reparse_point(temporary):
        raise RuntimeError("refusing to clean a reparse-point checkpoint staging path")
    if temporary.exists():
        if not temporary.is_dir():
            raise RuntimeError("checkpoint staging path changed type")
        shutil.rmtree(temporary)


@scientific_operation_boundary("raw_checkpoint_construction")
def _construct_raw_checkpoint_payload(
    result: Mapping[str, object],
) -> tuple[bytes, bytes]:
    payload_json = _json_bytes(result)
    return payload_json, _gzip_bytes(payload_json)


@scientific_operation_boundary("raw_checkpoint_publication")
def publish_graph_checkpoint(
    graph_root: Path | str,
    result: Mapping[str, object],
    run_manifest: Mapping[str, object],
) -> Path:
    """Publish one complete checkpoint by same-parent atomic directory rename."""

    graph_root_path = Path(graph_root)
    if _path_is_reparse_point(graph_root_path) or not graph_root_path.is_dir():
        raise RuntimeError("graph checkpoint root is unsafe")
    root = graph_root_path.resolve(strict=True)
    if (
        _path_is_reparse_point(graph_root_path)
        or not root.is_dir()
        or not os.path.samestat(graph_root_path.stat(), root.stat())
    ):
        raise RuntimeError("graph checkpoint root became unsafe")
    identity = result.get("graph_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("checkpoint result graph identity is missing")
    graph_id = str(identity.get("graph_id", ""))
    if (
        not graph_id
        or graph_id not in run_manifest.get("schedule", ())
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in graph_id)
    ):
        raise RuntimeError("checkpoint graph identity is outside the run schedule")
    validate_iteration2_graph_result(result)
    target = root / graph_id
    if target.exists() or _path_is_reparse_point(target):
        raise FileExistsError(f"graph checkpoint already exists: {graph_id}")
    temporary = root / f".{graph_id}.tmp-{uuid4().hex}"
    temporary_created = False
    published = False
    try:
        temporary.mkdir()
        temporary_created = True
        if temporary.parent != target.parent:
            raise RuntimeError(
                "checkpoint staging directory is not a same-parent sibling"
            )
        payload_json, payload_gzip = _construct_raw_checkpoint_payload(result)
        payload_path = temporary / GRAPH_RESULT_FILENAME
        manifest_path = temporary / GRAPH_MANIFEST_FILENAME
        completion_path = temporary / GRAPH_COMPLETION_FILENAME
        _write_new(payload_path, payload_gzip)
        checkpoint_manifest = {
            "schema": GRAPH_CHECKPOINT_MANIFEST_SCHEMA,
            "graph_id": graph_id,
            "run_identity": run_manifest["run_identity"],
            "data_generation_hash": run_manifest["data_generation_hash"],
            "analysis_plan_hash": run_manifest["analysis_plan_hash"],
            "output_schema_hash": run_manifest["output_schema_hash"],
            "protocol_hash": run_manifest["protocol_hash"],
            "result_schema": result["result_schema"],
            "scientific_status": result["scientific_status"],
            "run_manifest_identity_sha256": _manifest_identity_sha256(
                run_manifest
            ),
            "payload": {
                "filename": GRAPH_RESULT_FILENAME,
                "sha256": _sha256_bytes(payload_gzip),
                "size_bytes": len(payload_gzip),
                "uncompressed_sha256": _sha256_bytes(payload_json),
                "uncompressed_size_bytes": len(payload_json),
                "serialization": "canonical_sorted_finite_json_utf8",
                "compression": "gzip_mtime_0_empty_filename",
            },
            "row_counts": _checkpoint_row_counts(result),
            "routes_per_pair": result["routes_per_pair"],
            "dijkstra_execution_count": result["dijkstra_execution_count"],
            "completion_filename": GRAPH_COMPLETION_FILENAME,
            "completion_written_last": True,
        }
        manifest_bytes = _json_bytes(checkpoint_manifest)
        _write_new(manifest_path, manifest_bytes)
        completion = {
            "schema": GRAPH_CHECKPOINT_COMPLETION_SCHEMA,
            "status": "complete",
            "graph_id": graph_id,
            "checkpoint_manifest_sha256": _sha256_bytes(manifest_bytes),
            "payload_sha256": _sha256_bytes(payload_gzip),
            "payload_size_bytes": len(payload_gzip),
            "files_before_completion": [
                GRAPH_RESULT_FILENAME,
                GRAPH_MANIFEST_FILENAME,
            ],
            "completion_written_last": True,
            "atomic_publication": "same_parent_directory_rename",
        }
        _write_new(completion_path, _json_bytes(completion))
        if {item.name for item in temporary.iterdir()} != GRAPH_CHECKPOINT_FILENAMES:
            raise RuntimeError("staged checkpoint file inventory is invalid")
        _fsync_directory(temporary)
        os.replace(temporary, target)
        published = True
        _fsync_directory(root)
    finally:
        if temporary_created and not published and temporary.exists():
            _cleanup_owned_checkpoint_staging(
                temporary,
                root=root,
                graph_id=graph_id,
            )
    return target


def validate_checkpoint_directory(
    checkpoint: Path | str,
    *,
    run_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Read and structurally validate one atomically published checkpoint."""

    checkpoint_path = Path(checkpoint)
    if _path_is_reparse_point(checkpoint_path) or not checkpoint_path.is_dir():
        raise RuntimeError("graph checkpoint must be a non-symlink directory")
    directory = checkpoint_path.resolve(strict=True)
    if (
        _path_is_reparse_point(checkpoint_path)
        or not directory.is_dir()
        or not os.path.samestat(checkpoint_path.stat(), directory.stat())
    ):
        raise RuntimeError("graph checkpoint became unsafe")
    entries = tuple(directory.iterdir())
    if any(_path_is_reparse_point(entry) for entry in entries):
        raise RuntimeError("graph checkpoint contains a link or reparse point")
    if any(not entry.is_file() for entry in entries):
        raise RuntimeError("graph checkpoint contains a nested directory")
    observed_files = {entry.name for entry in entries}
    if observed_files != GRAPH_CHECKPOINT_FILENAMES:
        raise RuntimeError(
            "graph checkpoint is incomplete or contains extra files: "
            f"{sorted(observed_files)}"
        )
    result_path = directory / GRAPH_RESULT_FILENAME
    manifest_path = directory / GRAPH_MANIFEST_FILENAME
    completion_path = directory / GRAPH_COMPLETION_FILENAME
    checkpoint_manifest = _load_manifest(manifest_path)
    completion = _load_manifest(completion_path)
    result, physical = _load_gzip_result(result_path)
    counts = validate_iteration2_graph_result(result)
    identity = result.get("graph_identity")
    graph_id = (
        str(identity.get("graph_id", ""))
        if isinstance(identity, Mapping)
        else ""
    )
    if graph_id != directory.name:
        raise RuntimeError("checkpoint directory and graph identity mismatch")
    manifest_required = {
        "schema",
        "graph_id",
        "run_identity",
        "data_generation_hash",
        "analysis_plan_hash",
        "output_schema_hash",
        "protocol_hash",
        "result_schema",
        "scientific_status",
        "run_manifest_identity_sha256",
        "payload",
        "row_counts",
        "routes_per_pair",
        "dijkstra_execution_count",
        "completion_filename",
        "completion_written_last",
    }
    missing = sorted(manifest_required - set(checkpoint_manifest))
    if missing:
        raise RuntimeError(
            "checkpoint manifest is missing required field(s): "
            + ", ".join(missing)
        )
    payload = checkpoint_manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint payload manifest is invalid")
    payload_json = _json_bytes(result)
    expected_payload = {
        "filename": GRAPH_RESULT_FILENAME,
        "sha256": _sha256_bytes(physical),
        "size_bytes": len(physical),
        "uncompressed_sha256": _sha256_bytes(payload_json),
        "uncompressed_size_bytes": len(payload_json),
        "serialization": "canonical_sorted_finite_json_utf8",
        "compression": "gzip_mtime_0_empty_filename",
    }
    if dict(payload) != expected_payload:
        raise RuntimeError("checkpoint payload hash, size, or serialization mismatch")
    expected_manifest_fields = {
        "schema": GRAPH_CHECKPOINT_MANIFEST_SCHEMA,
        "graph_id": graph_id,
        "run_identity": result.get("run_identity"),
        "data_generation_hash": result.get("data_generation_hash"),
        "analysis_plan_hash": result.get("analysis_plan_hash"),
        "output_schema_hash": result.get("output_schema_hash"),
        "protocol_hash": result.get("protocol_hash"),
        "result_schema": result.get("result_schema"),
        "scientific_status": result.get("scientific_status"),
        "row_counts": _checkpoint_row_counts(result),
        "routes_per_pair": result.get("routes_per_pair"),
        "dijkstra_execution_count": result.get("dijkstra_execution_count"),
        "completion_filename": GRAPH_COMPLETION_FILENAME,
        "completion_written_last": True,
    }
    if any(
        checkpoint_manifest.get(key) != value
        for key, value in expected_manifest_fields.items()
    ):
        raise RuntimeError("checkpoint manifest identity or count mismatch")
    if run_manifest is not None:
        run_identity = _manifest_identity(run_manifest)
        if graph_id not in run_identity["schedule"]:
            raise RuntimeError("checkpoint graph is absent from the run schedule")
        for key in (
            "run_identity",
            "data_generation_hash",
            "analysis_plan_hash",
            "output_schema_hash",
            "protocol_hash",
        ):
            if checkpoint_manifest.get(key) != run_identity[key]:
                raise RuntimeError("checkpoint is stale for the run manifest")
        if checkpoint_manifest.get(
            "run_manifest_identity_sha256"
        ) != _manifest_identity_sha256(run_manifest):
            raise RuntimeError("checkpoint run-manifest binding mismatch")
    manifest_bytes = manifest_path.read_bytes()
    expected_completion = {
        "schema": GRAPH_CHECKPOINT_COMPLETION_SCHEMA,
        "status": "complete",
        "graph_id": graph_id,
        "checkpoint_manifest_sha256": _sha256_bytes(manifest_bytes),
        "payload_sha256": _sha256_bytes(physical),
        "payload_size_bytes": len(physical),
        "files_before_completion": [
            GRAPH_RESULT_FILENAME,
            GRAPH_MANIFEST_FILENAME,
        ],
        "completion_written_last": True,
        "atomic_publication": "same_parent_directory_rename",
    }
    if completion != expected_completion:
        raise RuntimeError("checkpoint completion marker is invalid")
    return {
        "graph_id": graph_id,
        "result": result,
        "checkpoint_manifest": checkpoint_manifest,
        "completion": completion,
        "row_counts": expected_manifest_fields["row_counts"],
        "payload_sha256": _sha256_bytes(physical),
        "payload_size_bytes": len(physical),
        "structural_validation": counts,
        "scientific_validation": None,
    }


@scientific_operation_boundary("scientific_regeneration_audit")
def _regenerate_and_validate_checkpoint_scientific_result(
    checkpoint: Path | str,
    *,
    run_manifest: Mapping[str, object],
    specification: object,
) -> dict[str, object]:
    validated = validate_checkpoint_directory(
        checkpoint,
        run_manifest=run_manifest,
    )
    scientific_validation = validate_scheduled_iteration2_graph_result(
        validated["result"],
        specification,
    )
    return {**validated, "scientific_validation": scientific_validation}


def regenerate_and_validate_checkpoint_scientific_result(
    checkpoint: Path | str,
    *,
    run_manifest: Mapping[str, object],
    specification: object,
    confirmation: str,
) -> dict[str, object]:
    """Run one explicit, measured, non-publishing regeneration audit."""

    if confirmation != SCIENTIFIC_REGENERATION_AUDIT_CONFIRMATION:
        raise RuntimeError("exact scientific regeneration audit confirmation required")
    if current_scientific_ledger() is not None:
        return _regenerate_and_validate_checkpoint_scientific_result(
            checkpoint,
            run_manifest=run_manifest,
            specification=specification,
        )
    with scientific_operation_context(SCIENTIFIC_REGENERATION_AUDIT) as ledger:
        result = _regenerate_and_validate_checkpoint_scientific_result(
            checkpoint,
            run_manifest=run_manifest,
            specification=specification,
        )
        return {
            **result,
            "audit_status": "explicit_non_publishing_scientific_regeneration_audit",
            "scientific_operation_ledger": ledger.snapshot(),
        }


def _validate_resume_directory(
    output: Path,
    current_manifest: Mapping[str, object],
    *,
    validation_policy: ResumeValidationPolicy,
) -> dict[str, object]:
    """Validate every checkpoint against the frozen schedule and reject strays."""

    if validation_policy is not ResumeValidationPolicy.READ_ONLY_STRUCTURAL:
        raise ValueError("resume validation must be structurally read-only")
    output_path = Path(output)
    if _path_is_reparse_point(output_path) or not output_path.is_dir():
        raise RuntimeError("resume output must be a non-symlink directory")
    output = output_path.resolve(strict=True)
    if (
        _path_is_reparse_point(output_path)
        or not output.is_dir()
        or not os.path.samestat(output_path.stat(), output.stat())
    ):
        raise RuntimeError("resume output became unsafe")
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("resume manifest is missing")
    graph_root = output / "graphs"
    if not graph_root.is_dir() or _path_is_reparse_point(graph_root):
        raise RuntimeError("resume graph checkpoint directory is missing or unsafe")
    direct_graph_files = [path.name for path in graph_root.iterdir() if path.is_file()]
    if direct_graph_files:
        raise RuntimeError(
            "corrupt or legacy flat graph checkpoint files are forbidden: "
            f"{sorted(direct_graph_files)}"
        )
    stored_manifest = _load_manifest(manifest_path)
    if _manifest_identity(stored_manifest) != _manifest_identity(
        current_manifest
    ):
        raise RuntimeError("resume manifest identity mismatch")
    specifications = scheduled_specifications()
    schedule = tuple(str(item) for item in current_manifest["schedule"])
    if set(specifications) != set(schedule) or len(schedule) != len(specifications):
        raise RuntimeError("resume manifest schedule is not the frozen schedule")
    if any(
        _path_is_reparse_point(path) for path in (output, *output.rglob("*"))
    ):
        raise RuntimeError("resume directory contains a link or reparse point")
    output_files = {path.name for path in output.iterdir() if path.is_file()}
    unexpected_root_files = output_files - {"run_manifest.json", "run_complete.json"}
    if unexpected_root_files:
        raise RuntimeError(
            "resume directory contains unexpected root files: "
            f"{sorted(unexpected_root_files)}"
        )
    output_directories = {path.name for path in output.iterdir() if path.is_dir()}
    if output_directories != {"graphs"}:
        raise RuntimeError(
            "resume directory contains unexpected root directories: "
            f"{sorted(output_directories - {'graphs'})}"
        )
    checkpoint_directories = sorted(
        (path for path in graph_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    observed_ids = {path.name for path in checkpoint_directories}
    orphaned = sorted(
        graph_id
        for graph_id in observed_ids
        if graph_id.startswith(".") or graph_id not in specifications
    )
    if orphaned:
        raise RuntimeError(
            f"resume directory contains orphan or stale checkpoints: {orphaned}"
        )
    ordered_completed = [graph_id for graph_id in schedule if graph_id in observed_ids]
    if tuple(ordered_completed) != schedule[: len(ordered_completed)]:
        raise RuntimeError("resume checkpoints are not a contiguous schedule prefix")
    validated_ids: list[str] = []
    validation_seconds = 0.0
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    for graph_id in ordered_completed:
        path = graph_root / graph_id
        validated = validate_checkpoint_directory(
            path,
            run_manifest=current_manifest,
        )
        scientific = validated.get("scientific_validation")
        if isinstance(scientific, Mapping):
            validation_seconds += float(
                scientific.get("resume_validation_seconds", 0.0)
            )
        validated_ids.append(graph_id)
        checkpoint_hashes[graph_id] = {
            filename: _sha256_path(path / filename)
            for filename in sorted(GRAPH_CHECKPOINT_FILENAMES)
        }
    completion_path = output / "run_complete.json"
    completion_valid = False
    if completion_path.exists():
        if len(validated_ids) != ITERATION2_RAW_GRAPH_FILE_COUNT:
            raise RuntimeError(
                "run completion marker exists before all checkpoints"
            )
        completion = _load_manifest(completion_path)
        expected_completion = {
            "schema": RUN_COMPLETION_SCHEMA,
            "status": "complete",
            "run_identity": ITERATION2_RUN_IDENTITY,
            "data_generation_hash": DATA_GENERATION_HASH,
            "analysis_plan_hash": ANALYSIS_PLAN_HASH,
            "output_schema_hash": OUTPUT_SCHEMA_HASH,
            "protocol_hash": COMBINED_PROTOCOL_HASH,
            "run_manifest_sha256": _sha256_path(manifest_path),
            "graph_checkpoint_count": ITERATION2_RAW_GRAPH_FILE_COUNT,
            "files_per_graph_checkpoint": GRAPH_CHECKPOINT_FILE_COUNT,
            "raw_file_count": RAW_RUN_FILE_COUNT,
            "graph_checkpoint_file_sha256": checkpoint_hashes,
            "schedule": list(schedule),
            "completion_written_last": True,
        }
        if completion != expected_completion:
            raise RuntimeError("run completion marker is invalid")
        completion_valid = True
    return {
        "resume_requested": True,
        "validation_policy": validation_policy.value,
        "deep_validation": False,
        "scientific_regeneration_performed": False,
        "validated_graph_files": len(validated_ids),
        "validated_graph_checkpoints": len(validated_ids),
        "validated_graph_ids": validated_ids,
        "remaining_graph_files": (
            ITERATION2_RAW_GRAPH_FILE_COUNT - len(validated_ids)
        ),
        "resume_validation_seconds": validation_seconds,
        "unexpected_files": [],
        "orphaned_checkpoints": [],
        "manifest_identity_matches": True,
        "completion_marker_valid": completion_valid,
    }


def _recheck_execution_authorization(
    report: Mapping[str, object],
    *,
    confirmation: str | None,
    expected_source_commit: str | None,
    expected_source_fingerprint: str | None,
    expected_dependency_fingerprint: str | None,
    expected_capacity_profile: str | None,
    expected_protocol_hash: str | None,
) -> None:
    """Fail if any authorization identity changed after preflight."""

    root = repository_root()
    output = resolve_iteration2_output(root, ITERATION2_RUN_IDENTITY)
    current_manifest = build_manifest(root)
    if report.get("authorized") is not True:
        raise RuntimeError("Iteration 2 full run is not authorized")
    if confirmation != FULL_RUN_CONFIRMATION_TOKEN:
        raise RuntimeError("confirmation token changed after preflight")
    if current_manifest["source_worktree"] != "clean":
        raise RuntimeError("source became dirty after preflight")
    expected = {
        "source_commit": expected_source_commit,
        "source_fingerprint": expected_source_fingerprint,
        "dependency_fingerprint": expected_dependency_fingerprint,
        "capacity_profile_sha256": expected_capacity_profile,
        "protocol_hash": expected_protocol_hash,
    }
    observed = {
        "source_commit": current_manifest["source_commit"],
        "source_fingerprint": current_manifest["source_fingerprint"],
        "dependency_fingerprint": current_manifest[
            "dependency_fingerprint"
        ]["sha256"],
        "capacity_profile_sha256": current_manifest[
            "capacity_profile_sha256"
        ],
        "protocol_hash": COMBINED_PROTOCOL_HASH,
    }
    if expected != observed:
        raise RuntimeError("authorization identity changed after preflight")
    if _manifest_identity(current_manifest) != _manifest_identity(
        report["manifest"]
    ):
        raise RuntimeError("run manifest changed after preflight")
    capacity = _capacity_status(root, output)
    if (
        capacity.get("profile_valid") is not True
        or capacity.get("disk_space_pass") is not True
        or capacity.get("profile_sha256") != expected_capacity_profile
    ):
        raise RuntimeError("capacity authorization changed after preflight")
    verify_iteration1_immutable(root, deep=True)


def _execute_full_run_with_lease_held(
    *,
    mode: str,
    confirmation: str | None,
    expected_source_commit: str | None,
    expected_source_fingerprint: str | None,
    expected_dependency_fingerprint: str | None,
    expected_capacity_profile: str | None,
    expected_protocol_hash: str | None,
    resume: bool,
) -> int:
    """Freshly authorize and execute; caller-provided reports are never trusted."""

    report = preflight(
        mode=mode,
        confirmation=confirmation,
        expected_source_commit=expected_source_commit,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_dependency_fingerprint=expected_dependency_fingerprint,
        expected_capacity_profile=expected_capacity_profile,
        expected_protocol_hash=expected_protocol_hash,
        resume=resume,
    )
    if report.get("authorized") is not True:
        raise RuntimeError(
            "Iteration 2 full run is not authorized: "
            + "; ".join(report.get("authorization_reasons", ()))
        )
    _recheck_execution_authorization(
        report,
        confirmation=confirmation,
        expected_source_commit=expected_source_commit,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_dependency_fingerprint=expected_dependency_fingerprint,
        expected_capacity_profile=expected_capacity_profile,
        expected_protocol_hash=expected_protocol_hash,
    )
    root = repository_root()
    output = resolve_iteration2_output(root, ITERATION2_RUN_IDENTITY)
    manifest = dict(report["manifest"])
    manifest_path = output / "run_manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError("run directory exists; explicit resume required")
        stored = _load_manifest(manifest_path)
        if _manifest_identity(stored) != _manifest_identity(manifest):
            raise RuntimeError("resume manifest identity mismatch")
    else:
        _recheck_execution_authorization(
            report,
            confirmation=confirmation,
            expected_source_commit=expected_source_commit,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_dependency_fingerprint=expected_dependency_fingerprint,
            expected_capacity_profile=expected_capacity_profile,
            expected_protocol_hash=expected_protocol_hash,
        )
        output.mkdir(parents=True)
        _write_new(manifest_path, _json_bytes(manifest))
    graph_root = output / "graphs"
    graph_root.mkdir(exist_ok=True)
    completed = 0
    validated_resume_ids = set(
        report.get("checkpoint_validation", {}).get(
            "validated_graph_ids",
            (),
        )
    )
    if (output / "run_complete.json").exists():
        if (
            report["checkpoint_validation"].get("completion_marker_valid")
            is not True
            or report["checkpoint_validation"].get("remaining_graph_files") != 0
        ):
            raise RuntimeError("existing run completion marker is invalid")
        return ITERATION2_RAW_GRAPH_FILE_COUNT
    _recheck_execution_authorization(
        report,
        confirmation=confirmation,
        expected_source_commit=expected_source_commit,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_dependency_fingerprint=expected_dependency_fingerprint,
        expected_capacity_profile=expected_capacity_profile,
        expected_protocol_hash=expected_protocol_hash,
    )
    for spec in full_schedule():
        target = graph_root / spec.graph_id
        if target.exists():
            if spec.graph_id not in validated_resume_ids:
                raise RuntimeError(
                    "checkpoint was not scientifically validated for resume"
                )
            completed += 1
            continue
        result = execute_scheduled_graph(spec, pair_count=PAIRS_PER_GRAPH)
        validate_iteration2_graph_result(result)
        published = publish_graph_checkpoint(graph_root, result, manifest)
        validate_checkpoint_directory(
            published,
            run_manifest=manifest,
        )
        completed += 1
        print(
            f"ITERATION2_PROGRESS {completed}/{ITERATION2_GRAPH_COUNT} "
            f"{spec.graph_id}",
            flush=True,
        )
    final_validation = _validate_resume_directory(
        output,
        manifest,
        validation_policy=ResumeValidationPolicy.READ_ONLY_STRUCTURAL,
    )
    if (
        completed != ITERATION2_RAW_GRAPH_FILE_COUNT
        or final_validation["remaining_graph_files"] != 0
    ):
        raise RuntimeError("Iteration 2 completed with the wrong output volume")
    checkpoint_hashes = {
        spec.graph_id: {
            filename: _sha256_path(graph_root / spec.graph_id / filename)
            for filename in sorted(GRAPH_CHECKPOINT_FILENAMES)
        }
        for spec in full_schedule()
    }
    completion = {
        "schema": RUN_COMPLETION_SCHEMA,
        "status": "complete",
        "run_identity": ITERATION2_RUN_IDENTITY,
        "data_generation_hash": DATA_GENERATION_HASH,
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "output_schema_hash": OUTPUT_SCHEMA_HASH,
        "protocol_hash": COMBINED_PROTOCOL_HASH,
        "run_manifest_sha256": _sha256_path(manifest_path),
        "graph_checkpoint_count": len(checkpoint_hashes),
        "files_per_graph_checkpoint": GRAPH_CHECKPOINT_FILE_COUNT,
        "raw_file_count": RAW_RUN_FILE_COUNT,
        "graph_checkpoint_file_sha256": checkpoint_hashes,
        "schedule": [spec.graph_id for spec in full_schedule()],
        "completion_written_last": True,
    }
    completion_path = output / "run_complete.json"
    temporary_completion = (
        output / f".run_complete.tmp-{uuid4().hex}"
    )
    try:
        _write_new(temporary_completion, _json_bytes(completion))
        if completion_path.exists():
            raise FileExistsError("run completion marker already exists")
        os.replace(temporary_completion, completion_path)
    finally:
        if temporary_completion.exists():
            temporary_completion.unlink()
    completed_validation = _validate_resume_directory(
        output,
        manifest,
        validation_policy=ResumeValidationPolicy.READ_ONLY_STRUCTURAL,
    )
    if (
        completed_validation["completion_marker_valid"] is not True
        or len(
            [path for path in output.rglob("*") if path.is_file()]
        )
        != RAW_RUN_FILE_COUNT
    ):
        raise RuntimeError("Iteration 2 final completion validation failed")
    return completed


def execute_full_run(
    *,
    mode: str,
    confirmation: str | None,
    expected_source_commit: str | None,
    expected_source_fingerprint: str | None,
    expected_dependency_fingerprint: str | None,
    expected_capacity_profile: str | None,
    expected_protocol_hash: str | None,
    resume: bool,
) -> int:
    """Own the run-wide lease for authorization, execution, and completion."""

    root = repository_root()
    source_commit = _git_state(root)[0]
    with acquire_iteration2_run_lease(
        root=root,
        run_identity=ITERATION2_RUN_IDENTITY,
        source_commit=source_commit,
        resume=resume,
    ):
        return _execute_full_run_with_lease_held(
            mode=mode,
            confirmation=confirmation,
            expected_source_commit=expected_source_commit,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_dependency_fingerprint=expected_dependency_fingerprint,
            expected_capacity_profile=expected_capacity_profile,
            expected_protocol_hash=expected_protocol_hash,
            resume=resume,
        )


@scientific_operation_boundary("excluded_fixture_execution")
def excluded_feasibility_results() -> tuple[dict[str, object], ...]:
    """Return two in-memory results from explicitly excluded small graphs."""

    contract = excluded_feasibility_contract()
    n = 14
    m = 2
    probability = 2 * m * (n - m) / (n * (n - 1))
    er = generate_connected_erdos_renyi(
        n=n,
        p=probability,
        graph_seed=EXCLUDED_FIXTURE_SEEDS[0],
        replicate_index=0,
        max_attempts=50,
        p_exact_numerator=2 * m * (n - m),
        p_exact_denominator=n * (n - 1),
    )
    ba = generate_connected_barabasi_albert(
        n=n,
        m=m,
        graph_seed=EXCLUDED_FIXTURE_SEEDS[1],
        replicate_index=0,
    )
    results: list[dict[str, object]] = []
    for index, (model, generated) in enumerate(
        (
            ("erdos_renyi", er),
            ("barabasi_albert", ba),
        )
    ):
        results.append(execute_iteration2_graph(
            generated.graph,
            graph_id=f"excluded_fixture_{model}",
            model=model,
            n=n,
            m=m,
            replicate_index=0,
            pair_seed=EXCLUDED_FIXTURE_SEEDS[index + 2],
            pair_count=12,
            graph_seed=EXCLUDED_FIXTURE_SEEDS[index],
            embedding_provenance_seed=None,
            generation_metadata=generated.metadata,
            audit_all_pairs=True,
            run_identity=contract.raw_identity,
        ))
    return tuple(results)


def excluded_feasibility_contract() -> ExcludedAnalysisFixtureContract:
    return ExcludedAnalysisFixtureContract(
        fixture_tag="feasibility_e2e",
        expected_graph_ids=(
            "excluded_fixture_erdos_renyi",
            "excluded_fixture_barabasi_albert",
        ),
        excluded_seeds=tuple(EXCLUDED_FIXTURE_SEEDS),
        pair_count=12,
        bootstrap_replicates=2,
        property_resampling_replicates=2,
        permutation_replicates=2,
    )


def run_excluded_feasibility_fixture() -> dict[str, object]:
    """Run two small graphs in memory using only new non-scientific seeds."""

    results = excluded_feasibility_results()
    validations = []
    diagnostic_summaries = []
    for result in results:
        validations.append(validate_iteration2_graph_result(result))
        diagnostic_summaries.append(
            {
                "graph_id": result["graph_identity"]["graph_id"],
                "oracle_route_decisions_checked": result[
                    "high_precision_sentinel"
                ]["route_decisions_checked"],
                "oracle_pair_count": len(
                    result["high_precision_sentinel"]["pair_indices"]
                ),
                "oracle_selection_mode": result[
                    "high_precision_sentinel"
                ]["selection_mode"],
                "pair_reuse_across_all_conditions": result[
                    "graph_and_pair_diagnostics"
                ]["pair_sampling"]["same_pairs_all_conditions_and_methods"],
                "accepted_generation_seed": result[
                    "graph_and_pair_diagnostics"
                ]["accepted_seed"],
                "hydra_poincare_gauge_changed_pairs": result[
                    "gauge_and_centering_diagnostics"
                ]["hydra_centering"]["poincare_routing_changed_pairs"],
            }
        )
    return {
        "label": FIXTURE_LABEL,
        "excluded_from_scientific_analysis": True,
        "seeds": list(EXCLUDED_FIXTURE_SEEDS),
        "graph_count": 2,
        "pair_count": sum(len(result["pairs"]) for result in results),
        "validation_counts": validations,
        "diagnostic_summaries": diagnostic_summaries,
        "wrote_scientific_results": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iteration 2 experiment runner")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--mode",
        choices=("development", "full"),
        required=True,
    )
    preflight_parser.add_argument("--confirm-full-run")
    _add_identity_arguments(preflight_parser)
    preflight_parser.add_argument("--resume", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=("full",), required=True)
    run_parser.add_argument("--confirm-full-run", required=True)
    _add_identity_arguments(run_parser)
    run_parser.add_argument("--resume", action="store_true")
    subparsers.add_parser("development-fixture")
    subparsers.add_parser("protocol")
    return parser


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-fingerprint")
    parser.add_argument("--expected-dependency-fingerprint")
    parser.add_argument("--expected-capacity-profile")
    parser.add_argument("--expected-protocol-hash")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "protocol":
        root = repository_root()
        manifest = build_manifest(root)
        capacity = _capacity_status(
            root,
            resolve_iteration2_output(root, ITERATION2_RUN_IDENTITY),
        )
        print(
            json.dumps(
                {
                    "run_identity": ITERATION2_RUN_IDENTITY,
                    "protocol_hash": COMBINED_PROTOCOL_HASH,
                    "data_generation_hash": DATA_GENERATION_HASH,
                    "analysis_plan_hash": ANALYSIS_PLAN_HASH,
                    "output_schema_hash": OUTPUT_SCHEMA_HASH,
                    "full_run_confirmation_token": FULL_RUN_CONFIRMATION_TOKEN,
                    "source_commit": manifest["source_commit"],
                    "source_worktree": manifest["source_worktree"],
                    "source_fingerprint": manifest["source_fingerprint"],
                    "dependency_fingerprint": (
                        manifest["dependency_fingerprint"]["sha256"]
                    ),
                    "capacity_profile_sha256": (
                        manifest["capacity_profile_sha256"]
                    ),
                    "capacity": capacity,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.operation == "development-fixture":
        print(
            json.dumps(
                run_excluded_feasibility_fixture(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.operation == "preflight":
        report = preflight(
            mode=args.mode,
            confirmation=args.confirm_full_run,
            expected_source_commit=args.expected_source_commit,
            expected_source_fingerprint=args.expected_source_fingerprint,
            expected_dependency_fingerprint=(
                args.expected_dependency_fingerprint
            ),
            expected_capacity_profile=args.expected_capacity_profile,
            expected_protocol_hash=args.expected_protocol_hash,
            resume=args.resume,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["authorized"] else 2
    try:
        completed = execute_full_run(
            mode=args.mode,
            confirmation=args.confirm_full_run,
            expected_source_commit=args.expected_source_commit,
            expected_source_fingerprint=args.expected_source_fingerprint,
            expected_dependency_fingerprint=(
                args.expected_dependency_fingerprint
            ),
            expected_capacity_profile=args.expected_capacity_profile,
            expected_protocol_hash=args.expected_protocol_hash,
            resume=args.resume,
        )
    except Iteration2RunAlreadyActive as exc:
        print(
            json.dumps(
                {"authorized": False, "error": exc.as_dict()},
                sort_keys=True,
            )
        )
        return 3
    except RuntimeError as exc:
        print(json.dumps({"authorized": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"completed_graphs": completed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
