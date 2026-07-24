"""Atomic, resumable per-graph checkpoints for the full experiment.

This module performs no work at import time.  Callers must explicitly create a
run directory and publish graph data.  Scientific payloads are written in
deterministic formats; runtime and timestamp fields are intentionally excluded
from byte-for-byte determinism claims.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import csv
import gzip
import io
import json
from math import isfinite
import os
from pathlib import Path
import re
from time import perf_counter_ns
from uuid import uuid4


RESULT_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "run_manifest.json"
PROGRESS_FILENAME = "progress.json"
CHECKPOINT_DIRECTORY = "graphs"
PUBLICATION_TIMING_DIRECTORY = "publication_timings"
COMPLETE_MARKER_FILENAME = "COMPLETE.json"
GRAPH_MANIFEST_FILENAME = "graph_manifest.json"
ERROR_REPORT_FILENAME = "ERROR.json"
TEMPORARY_DIRECTORY_TOKEN = ".tmp-"
PUBLICATION_TIMING_SCHEMA = "graph_publication_timing_v1"
PUBLICATION_TIMING_ENDPOINT = (
    "successful atomic directory rename followed by validation of the "
    "published checkpoint; this record write is excluded"
)

PUBLICATION_TIMING_DEFINITIONS = {
    "payload_serialization_ns": (
        "Elapsed nanoseconds from payload-serialization start through all "
        "scientific payload and run-manifest files, excluding timings.json, "
        "graph_manifest.json, COMPLETE.json, validation, and atomic rename."
    ),
    "prepublication_wall_ns": (
        "Elapsed nanoseconds from graph-generation start through payload "
        "serialization, sampled before timings.json and publication finalization."
    ),
    "atomic_publication_and_final_validation_ns": (
        "Elapsed nanoseconds from immediately before atomic directory rename "
        "through successful validation of the published checkpoint."
    ),
    "end_to_end_graph_wall_ns": (
        "Elapsed nanoseconds from immediately before graph generation through "
        "successful atomic publication and validation of the published checkpoint; "
        "writing this operational timing record is excluded."
    ),
}

FULL_COORDINATE_CONDITION_IDS = (
    "hydra",
    "mds_r050",
    "mds_r070",
    "mds_r085",
    "mds_r095",
)
FULL_DISTORTION_CONDITION_IDS = (
    "hydra_euclidean",
    "hydra_poincare",
    "base_mds_euclidean",
    "mds_poincare_r050",
    "mds_poincare_r070",
    "mds_poincare_r085",
    "mds_poincare_r095",
)
FULL_PAIR_COUNT = 1_000
FULL_DIJKSTRA_RECORD_COUNT = 1_000
FULL_ROUTE_RECORD_COUNT = 15_000
FULL_DISTORTION_RECORD_COUNT = 7

DATA_FILENAMES = (
    "generation.json",
    "network_metrics.json",
    "edges.csv.gz",
    "coordinates.csv.gz",
    "pairs.csv.gz",
    "embedding_metadata.json",
    "distortions.json",
    "dijkstra.jsonl.gz",
    "routes.jsonl.gz",
    "run_manifest.json",
    "timings.json",
)

_GRAPH_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_FLOAT_TAG = "__float64__"
_RUN_IDENTITY_FIELDS = (
    "result_schema_version",
    "configuration_schema_version",
    "seed_identity_version",
    "data_generation_hash",
    "analysis_plan_hash",
    "combined_freeze_hash",
    "git_commit_hash",
    "git_working_tree",
    "source_fingerprint",
    "python_version",
    "dependency_versions",
    "operating_system",
    "hardware",
    "output_schema",
    "execution_profile",
    "execution_model",
    "run_directory_name",
)


class CheckpointError(RuntimeError):
    """Base class for checkpoint publication and validation errors."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint is missing, incomplete, or fails integrity."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when checkpoint/run provenance does not match the current run."""


@dataclass(frozen=True)
class GraphCheckpointData:
    """Complete raw data required to publish one graph checkpoint."""

    graph_id: str
    generation_metadata: Mapping[str, object]
    edges: Sequence[tuple[int, int]]
    network_metrics: Mapping[str, object]
    pairs: Sequence[tuple[int, int]]
    coordinates: Mapping[str, Mapping[int, Sequence[float]]]
    embedding_metadata: Mapping[str, object]
    distortions: Sequence[Mapping[str, object]]
    dijkstra_records: Sequence[Mapping[str, object]]
    route_records: Sequence[Mapping[str, object]]
    timings: Mapping[str, int]
    run_manifest: Mapping[str, object]


@dataclass(frozen=True)
class CheckpointValidation:
    """Validated identity and counts for one complete checkpoint."""

    graph_id: str
    path: Path
    counts: Mapping[str, int]
    payload_sha256: Mapping[str, str]


@dataclass(frozen=True)
class CheckpointAudit:
    """Read-only assessment of a run directory and its graph checkpoints."""

    run_root: Path
    run_manifest_present: bool
    complete_graph_ids: tuple[str, ...]
    remaining_graph_ids: tuple[str, ...]
    errors: tuple[str, ...]
    resumable: bool


def _validate_graph_id(graph_id: str) -> str:
    if not isinstance(graph_id, str) or not _GRAPH_ID_PATTERN.fullmatch(graph_id):
        raise ValueError(
            "graph_id must contain only lowercase ASCII letters, digits, and underscores"
        )
    return graph_id


def _resolved_child(root: Path | str, *parts: str) -> Path:
    resolved_root = Path(root).resolve(strict=False)
    candidate = resolved_root.joinpath(*parts).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes declared root: {candidate}") from exc
    return candidate


def _json_value(value: object) -> object:
    """Return deterministic JSON data with exact tagged finite float64 values."""

    if is_dataclass(value):
        value = asdict(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("checkpoint data rejects NaN and infinity")
        return {_FLOAT_TAG: value.hex()}
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("checkpoint JSON object keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    raise ValueError(f"unsupported checkpoint value: {type(value).__name__}")


def decode_json_value(value: object) -> object:
    """Decode values emitted by :func:`_json_value`."""

    if isinstance(value, list):
        return [decode_json_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            encoded = value[_FLOAT_TAG]
            if not isinstance(encoded, str):
                raise CheckpointCorruptionError("invalid tagged float64 value")
            decoded = float.fromhex(encoded)
            if not isfinite(decoded):
                raise CheckpointCorruptionError("non-finite tagged float64 value")
            return decoded
        return {key: decode_json_value(item) for key, item in value.items()}
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def run_manifest_sha256(manifest: Mapping[str, object]) -> str:
    """Hash the exact deterministic JSON representation used in checkpoint files."""

    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    return sha256(_json_bytes(manifest)).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, _json_bytes(value))


def _gzip_bytes(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        fileobj=output,
        mode="wb",
        compresslevel=9,
        mtime=0,
    ) as compressed:
        compressed.write(payload)
    return output.getvalue()


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return _gzip_bytes(text.getvalue().encode("utf-8"))


def _jsonl_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    payload = b"".join(_json_bytes(record) + b"\n" for record in records)
    return _gzip_bytes(payload)


def _read_json(path: Path) -> object:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError(f"invalid JSON file: {path}") from exc
    return decode_json_value(value)


def _file_digest(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _run_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    missing = [key for key in _RUN_IDENTITY_FIELDS if key not in manifest]
    if missing:
        raise CheckpointCompatibilityError(
            f"run manifest is missing immutable identity fields: {missing}"
        )
    return {key: manifest[key] for key in _RUN_IDENTITY_FIELDS}


def validate_run_manifest_compatibility(
    existing: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Reject a resume when any immutable run identity differs."""

    _run_identity(existing)
    _run_identity(expected)

    def comparison_payload(manifest: Mapping[str, object]) -> dict[str, object]:
        payload = dict(manifest)
        timestamp = payload.pop("timestamp_utc", None)
        created_at = payload.pop("created_at_utc", timestamp)
        if (
            not isinstance(timestamp, str)
            or not timestamp
            or not isinstance(created_at, str)
            or created_at != timestamp
        ):
            raise CheckpointCompatibilityError(
                "run manifest creation timestamps are invalid"
            )
        return payload

    existing_payload = comparison_payload(existing)
    expected_payload = comparison_payload(expected)
    if _json_bytes(existing_payload) != _json_bytes(expected_payload):
        differing = sorted(
            key
            for key in set(existing_payload) | set(expected_payload)
            if existing_payload.get(key) != expected_payload.get(key)
        )
        raise CheckpointCompatibilityError(
            "run manifest identity mismatch: " + ", ".join(differing)
        )


def write_run_manifest_once(
    run_root: Path | str,
    manifest: Mapping[str, object],
) -> Path:
    """Create one immutable run manifest, or validate the existing manifest."""

    root = Path(run_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = _resolved_child(root, RUN_MANIFEST_FILENAME)
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if not isinstance(existing, dict):
            raise CheckpointCorruptionError("run manifest must be a JSON object")
        validate_run_manifest_compatibility(existing, manifest)
        return manifest_path
    temporary = _resolved_child(root, f".{RUN_MANIFEST_FILENAME}.{uuid4().hex}.tmp")
    _write_json(temporary, manifest)
    os.replace(temporary, manifest_path)
    return manifest_path


def write_progress(
    run_root: Path | str,
    *,
    schedule_ids: Sequence[str],
    complete_graph_ids: Sequence[str],
) -> Path:
    """Atomically replace the derived, mutable progress summary."""

    root = Path(run_root).resolve(strict=False)
    if not root.is_dir():
        raise CheckpointError("run root does not exist")
    scheduled = tuple(_validate_graph_id(item) for item in schedule_ids)
    complete = tuple(_validate_graph_id(item) for item in complete_graph_ids)
    if any(item not in scheduled for item in complete):
        raise CheckpointError("progress contains a graph outside the schedule")
    progress_path = _resolved_child(root, PROGRESS_FILENAME)
    temporary = _resolved_child(root, f".{PROGRESS_FILENAME}.{uuid4().hex}.tmp")
    _write_json(
        temporary,
        {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "scheduled_graph_count": len(scheduled),
            "complete_graph_count": len(complete),
            "remaining_graph_count": len(scheduled) - len(complete),
            "complete_graph_ids": list(complete),
        },
    )
    os.replace(temporary, progress_path)
    return progress_path


def _validate_duration(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointCorruptionError(
            f"publication timing {name} must be a non-negative integer"
        )
    return value


def _publication_timing_path(run_root: Path, graph_id: str) -> Path:
    timing_root = _resolved_child(run_root, PUBLICATION_TIMING_DIRECTORY)
    return _resolved_child(timing_root, f"{_validate_graph_id(graph_id)}.json")


def validate_publication_timing_record(
    run_root: Path | str,
    *,
    graph_id: str,
    expected_run_manifest: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate one required post-publication operational timing record."""

    root = Path(run_root).resolve(strict=False)
    timing_path = _publication_timing_path(root, graph_id)
    if not timing_path.is_file():
        raise CheckpointCorruptionError(
            "operational-integrity error: publication timing record is missing: "
            f"{timing_path}"
        )
    record = _read_json(timing_path)
    if not isinstance(record, dict):
        raise CheckpointCorruptionError(
            "operational-integrity error: publication timing record must be an object"
        )
    if (
        record.get("operational_timing_schema") != PUBLICATION_TIMING_SCHEMA
        or record.get("result_schema_version") != RESULT_SCHEMA_VERSION
    ):
        raise CheckpointCompatibilityError(
            "operational-integrity error: publication timing schema mismatch"
        )
    if record.get("graph_id") != graph_id:
        raise CheckpointCompatibilityError(
            "operational-integrity error: publication timing graph mismatch"
        )
    recorded_manifest = record.get("run_manifest")
    if not isinstance(recorded_manifest, dict):
        raise CheckpointCorruptionError(
            "operational-integrity error: publication timing run manifest is invalid"
        )
    if record.get("run_manifest_sha256") != run_manifest_sha256(recorded_manifest):
        raise CheckpointCorruptionError(
            "operational-integrity error: publication timing manifest hash mismatch"
        )
    validate_run_manifest_compatibility(
        recorded_manifest,
        expected_run_manifest,
    )
    if record.get("run_identity") != _run_identity(recorded_manifest):
        raise CheckpointCompatibilityError(
            "operational-integrity error: publication timing run identity mismatch"
        )
    if record.get("definitions") != PUBLICATION_TIMING_DEFINITIONS:
        raise CheckpointCompatibilityError(
            "operational-integrity error: publication timing definitions mismatch"
        )
    if record.get("endpoint") != PUBLICATION_TIMING_ENDPOINT:
        raise CheckpointCompatibilityError(
            "operational-integrity error: publication timing endpoint mismatch"
        )
    payload = _validate_duration(
        "payload_serialization_ns",
        record.get("payload_serialization_ns"),
    )
    prepublication = _validate_duration(
        "prepublication_wall_ns",
        record.get("prepublication_wall_ns"),
    )
    publication = _validate_duration(
        "atomic_publication_and_final_validation_ns",
        record.get("atomic_publication_and_final_validation_ns"),
    )
    end_to_end = _validate_duration(
        "end_to_end_graph_wall_ns",
        record.get("end_to_end_graph_wall_ns"),
    )
    if prepublication < payload:
        raise CheckpointCorruptionError(
            "operational-integrity error: prepublication wall time is too small"
        )
    if end_to_end < prepublication or end_to_end < publication:
        raise CheckpointCorruptionError(
            "operational-integrity error: end-to-end graph time is inconsistent"
        )
    if not isinstance(record.get("recorded_at_utc"), str) or not record[
        "recorded_at_utc"
    ]:
        raise CheckpointCorruptionError(
            "operational-integrity error: publication timing timestamp is invalid"
        )
    return record


def write_publication_timing_record(
    run_root: Path | str,
    *,
    graph_id: str,
    run_manifest: Mapping[str, object],
    payload_serialization_ns: int,
    prepublication_wall_ns: int,
    atomic_publication_and_final_validation_ns: int,
    end_to_end_graph_wall_ns: int,
) -> Path:
    """Atomically write the operational timing captured after publication."""

    root = Path(run_root).resolve(strict=False)
    if not root.is_dir():
        raise CheckpointError("run root does not exist")
    timing_root = _resolved_child(root, PUBLICATION_TIMING_DIRECTORY)
    timing_root.mkdir(exist_ok=True)
    final = _publication_timing_path(root, graph_id)
    if final.exists():
        raise CheckpointError(
            f"publication timing record already exists: {final}"
        )
    temporary = _resolved_child(
        timing_root,
        f".{_validate_graph_id(graph_id)}.{uuid4().hex}.tmp",
    )
    record = {
        "operational_timing_schema": PUBLICATION_TIMING_SCHEMA,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "graph_id": graph_id,
        "run_identity": _run_identity(run_manifest),
        "run_manifest": dict(run_manifest),
        "run_manifest_sha256": run_manifest_sha256(run_manifest),
        "payload_serialization_ns": payload_serialization_ns,
        "prepublication_wall_ns": prepublication_wall_ns,
        "atomic_publication_and_final_validation_ns": (
            atomic_publication_and_final_validation_ns
        ),
        "end_to_end_graph_wall_ns": end_to_end_graph_wall_ns,
        "endpoint": PUBLICATION_TIMING_ENDPOINT,
        "definitions": PUBLICATION_TIMING_DEFINITIONS,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _validate_duration("payload_serialization_ns", payload_serialization_ns)
    _validate_duration("prepublication_wall_ns", prepublication_wall_ns)
    _validate_duration(
        "atomic_publication_and_final_validation_ns",
        atomic_publication_and_final_validation_ns,
    )
    _validate_duration("end_to_end_graph_wall_ns", end_to_end_graph_wall_ns)
    _write_json(temporary, record)
    os.replace(temporary, final)
    validate_publication_timing_record(
        root,
        graph_id=graph_id,
        expected_run_manifest=run_manifest,
    )
    return final


def _required_route_fields(record: Mapping[str, object]) -> None:
    required = {
        "graph_id",
        "pair_index",
        "pair_id",
        "source",
        "destination",
        "coordinate_condition_id",
        "method_id",
        "success",
        "initial_failure_type",
        "final_failure_type",
        "repair_attempted",
        "repair_succeeded",
        "route_length",
        "physical_hop_count",
        "dijkstra_length",
        "dijkstra_hop_count",
        "stretch",
        "runtime_ns",
        "walk",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"route record is missing fields: {sorted(missing)}")


def _validate_record_common(
    record: Mapping[str, object],
    *,
    graph_id: str,
    pair_lookup: Mapping[str, tuple[int, int]],
    edge_set: set[tuple[int, int]],
) -> str:
    pair_id = record.get("pair_id")
    if not isinstance(pair_id, str) or pair_id not in pair_lookup:
        raise ValueError("record pair_id is outside the sampled pair list")
    if record.get("graph_id") != graph_id:
        raise ValueError("record graph_id does not match checkpoint")
    source, destination = pair_lookup[pair_id]
    expected_pair_index = int(pair_id.rsplit(":", 1)[1])
    if record.get("pair_index") != expected_pair_index:
        raise ValueError("record pair_index does not match pair_id")
    if record.get("source") != source or record.get("destination") != destination:
        raise ValueError("record endpoints do not match the sampled pair")
    runtime = record.get("runtime_ns")
    if isinstance(runtime, bool) or not isinstance(runtime, int) or runtime < 0:
        raise ValueError("record runtime_ns must be a non-negative integer")
    walk = record.get("walk")
    if (
        not isinstance(walk, (list, tuple))
        or not walk
        or walk[0] != source
        or any(isinstance(node, bool) or not isinstance(node, int) for node in walk)
    ):
        raise ValueError("record walk must be an integer walk beginning at source")
    if any(
        tuple(sorted((left, right))) not in edge_set
        for left, right in zip(walk, walk[1:])
    ):
        raise ValueError("record walk contains an edge outside the accepted graph")
    route_length = record.get("route_length")
    if (
        isinstance(route_length, bool)
        or not isinstance(route_length, int)
        or route_length != len(walk) - 1
    ):
        raise ValueError("record route_length must equal len(walk)-1")
    physical_hops = record.get("physical_hop_count", route_length)
    if physical_hops != route_length:
        raise ValueError("record physical_hop_count is inconsistent")
    return pair_id


def _validate_data(data: GraphCheckpointData) -> dict[str, int]:
    graph_id = _validate_graph_id(data.graph_id)
    if not isinstance(data.run_manifest, Mapping):
        raise ValueError("run_manifest must be a mapping")
    _run_identity(data.run_manifest)

    n_value = data.network_metrics.get("number_of_vertices")
    if isinstance(n_value, bool) or not isinstance(n_value, int) or n_value < 2:
        raise ValueError("network_metrics.number_of_vertices must be at least 2")
    n = n_value
    expected_nodes = set(range(n))
    if any(
        isinstance(node, bool)
        or not isinstance(node, int)
        or node not in expected_nodes
        for edge in data.edges
        for node in edge
    ):
        raise ValueError("edges must use integer node IDs 0 through n-1")
    canonical_edges = [tuple(sorted(edge)) for edge in data.edges]
    if any(len(edge) != 2 or edge[0] == edge[1] for edge in canonical_edges):
        raise ValueError("edges must be two distinct vertices")
    if len(set(canonical_edges)) != len(canonical_edges):
        raise ValueError("edges must be unique")
    edge_set = set(canonical_edges)

    pairs = tuple(data.pairs)
    if any(
        len(pair) != 2
        or isinstance(pair[0], bool)
        or isinstance(pair[1], bool)
        or not isinstance(pair[0], int)
        or not isinstance(pair[1], int)
        or pair[0] not in expected_nodes
        or pair[1] not in expected_nodes
        or pair[0] == pair[1]
        for pair in pairs
    ):
        raise ValueError("pairs must contain valid distinct integer vertices")
    if len(set(pairs)) != len(pairs):
        raise ValueError("ordered pairs must be unique")
    pair_lookup = {
        f"{graph_id}:pair:{index:04d}": pair
        for index, pair in enumerate(pairs)
    }

    coordinate_ids = tuple(data.coordinates)
    if not coordinate_ids:
        raise ValueError("at least one coordinate condition is required")
    for condition_id, coordinate_map in data.coordinates.items():
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError("coordinate condition IDs must be non-empty strings")
        if set(coordinate_map) != expected_nodes:
            raise ValueError(
                f"coordinate condition {condition_id} does not cover nodes 0 through n-1"
            )
        for point in coordinate_map.values():
            if len(point) != 2:
                raise ValueError("coordinates must be two-dimensional")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                for value in point
            ):
                raise ValueError("coordinates must be finite numeric pairs")
    if set(data.embedding_metadata) != set(coordinate_ids) | {"mds_base"}:
        raise ValueError(
            "embedding metadata must cover every coordinate condition and mds_base"
        )

    dijkstra_by_pair: dict[str, int] = {}
    for record in data.dijkstra_records:
        pair_id = _validate_record_common(
            record,
            graph_id=graph_id,
            pair_lookup=pair_lookup,
            edge_set=edge_set,
        )
        if record.get("method_id") != "dijkstra" or record.get("success") is not True:
            raise ValueError("invalid Dijkstra method identity or outcome")
        if record.get("coordinate_condition_id") is not None:
            raise ValueError("Dijkstra must not have a coordinate condition")
        if (
            record.get("apsp_length") != record.get("route_length")
            or record.get("apsp_agreement") is not True
        ):
            raise ValueError("Dijkstra/APSP agreement metadata is invalid")
        if pair_id in dijkstra_by_pair:
            raise ValueError("Dijkstra record pair IDs must be unique")
        dijkstra_by_pair[pair_id] = int(record["route_length"])
    route_identity_counts: dict[str, set[tuple[str, str]]] = {
        pair_id: set() for pair_id in pair_lookup
    }
    for record in data.route_records:
        _required_route_fields(record)
        pair_id = _validate_record_common(
            record,
            graph_id=graph_id,
            pair_lookup=pair_lookup,
            edge_set=edge_set,
        )
        condition_id = record.get("coordinate_condition_id")
        method_id = record.get("method_id")
        if (
            not isinstance(condition_id, str)
            or condition_id not in data.coordinates
            or method_id
            not in (
                "euclidean_greedy",
                "poincare_greedy",
                "repaired_poincare_greedy",
            )
        ):
            raise ValueError("route condition or method identity is invalid")
        route_identity = (condition_id, method_id)
        if route_identity in route_identity_counts[pair_id]:
            raise ValueError("duplicate route condition/method for sampled pair")
        route_identity_counts[pair_id].add(route_identity)
        success = record.get("success")
        if not isinstance(success, bool):
            raise ValueError("route success must be a boolean")
        walk = record["walk"]
        if success and walk[-1] != record["destination"]:
            raise ValueError("successful route walk must end at destination")
        if not success and walk[-1] == record["destination"]:
            raise ValueError("failed route walk cannot end at destination")
        if pair_id not in dijkstra_by_pair:
            raise ValueError("route has no corresponding Dijkstra record")
        dijkstra_length = record.get("dijkstra_length")
        if dijkstra_length != dijkstra_by_pair[pair_id] or dijkstra_length <= 0:
            raise ValueError("route Dijkstra length is inconsistent")
        if record.get("dijkstra_hop_count") != dijkstra_length:
            raise ValueError("route dijkstra_hop_count is inconsistent")
        stretch = record.get("stretch")
        if success:
            expected_stretch = record["route_length"] / dijkstra_length
            if (
                isinstance(stretch, bool)
                or not isinstance(stretch, (int, float))
                or not isfinite(float(stretch))
                or float(stretch) != expected_stretch
            ):
                raise ValueError("successful route stretch is inconsistent")
        elif stretch is not None:
            raise ValueError("failed routes must record null stretch")

    counts = {
        "vertices": n,
        "edges": len(data.edges),
        "coordinate_conditions": len(coordinate_ids),
        "coordinate_rows": n * len(coordinate_ids),
        "pairs": len(pairs),
        "dijkstra_records": len(data.dijkstra_records),
        "route_records": len(data.route_records),
        "distortion_records": len(data.distortions),
    }
    if data.run_manifest.get("execution_profile") == "full":
        expected = {
            "coordinate_conditions": len(FULL_COORDINATE_CONDITION_IDS),
            "pairs": FULL_PAIR_COUNT,
            "dijkstra_records": FULL_DIJKSTRA_RECORD_COUNT,
            "route_records": FULL_ROUTE_RECORD_COUNT,
            "distortion_records": FULL_DISTORTION_RECORD_COUNT,
        }
        for key, value in expected.items():
            if counts[key] != value:
                raise ValueError(
                    f"full checkpoint requires {value} {key}, got {counts[key]}"
                )
        if coordinate_ids != FULL_COORDINATE_CONDITION_IDS:
            raise ValueError("full checkpoint coordinate condition order is invalid")
        distortion_ids = tuple(
            record.get("metric_condition_id") for record in data.distortions
        )
        if distortion_ids != FULL_DISTORTION_CONDITION_IDS:
            raise ValueError("full checkpoint distortion condition order is invalid")
        required_routes = {
            (condition_id, method_id)
            for condition_id in FULL_COORDINATE_CONDITION_IDS
            for method_id in (
                "euclidean_greedy",
                "poincare_greedy",
                "repaired_poincare_greedy",
            )
        }
        if set(dijkstra_by_pair) != set(pair_lookup):
            raise ValueError("full checkpoint requires one Dijkstra record per pair")
        if any(
            identities != required_routes
            for identities in route_identity_counts.values()
        ):
            raise ValueError(
                "full checkpoint requires all 15 condition/method routes per pair"
            )
        metadata = data.generation_metadata
        required_generation = {
            "graph_model",
            "n",
            "graph_seed",
            "generation_attempt_count",
            "generation_attempt_index",
            "generation_attempt_seed",
            "generation_attempt_seeds",
            "rejected_disconnected_count",
            "realised_edge_count",
            "realised_average_degree",
            "setting_index",
        }
        missing_generation = required_generation - set(metadata)
        if missing_generation:
            raise ValueError(
                "full generation metadata is missing fields: "
                f"{sorted(missing_generation)}"
            )
        attempts = metadata["generation_attempt_seeds"]
        if (
            not isinstance(attempts, (list, tuple))
            or len(attempts) != metadata["generation_attempt_count"]
            or metadata["generation_attempt_index"] != len(attempts) - 1
            or metadata["generation_attempt_seed"] != attempts[-1]
        ):
            raise ValueError("generation attempt metadata is inconsistent")
        if metadata["realised_edge_count"] != counts["edges"]:
            raise ValueError("generation metadata edge count is inconsistent")
        if metadata["graph_model"] == "erdos_renyi":
            if (
                metadata["rejected_disconnected_count"] != len(attempts) - 1
                or not {
                    "p",
                    "p_float64_hex",
                    "p_exact_numerator",
                    "p_exact_denominator",
                }.issubset(metadata)
            ):
                raise ValueError("connected ER attempt/probability metadata is invalid")
        elif metadata["graph_model"] == "barabasi_albert":
            if (
                metadata["rejected_disconnected_count"] != 0
                or "m" not in metadata
                or metadata.get("ba_initial_graph") != "networkx.star_graph(m)"
            ):
                raise ValueError("BA generation metadata is invalid")
        else:
            raise ValueError("full generation graph model is invalid")

    for name, value in data.timings.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError("timings must map names to non-negative integer nanoseconds")
    _json_value(data.generation_metadata)
    _json_value(data.network_metrics)
    _json_value(data.embedding_metadata)
    _json_value(data.distortions)
    _json_value(data.dijkstra_records)
    _json_value(data.route_records)
    return counts


def _write_payload_files(
    directory: Path,
    data: GraphCheckpointData,
    *,
    serialization_start_ns: int,
    graph_wall_start_ns: int | None,
) -> dict[str, int]:
    _write_json(directory / "generation.json", data.generation_metadata)
    _write_json(directory / "network_metrics.json", data.network_metrics)
    edges = sorted(tuple(sorted((int(left), int(right)))) for left, right in data.edges)
    _write_bytes(
        directory / "edges.csv.gz",
        _csv_bytes(("source", "destination"), edges),
    )

    coordinate_rows: list[tuple[object, ...]] = []
    for condition_id, coordinate_map in data.coordinates.items():
        for node in sorted(coordinate_map):
            point = coordinate_map[node]
            coordinate_rows.append(
                (
                    condition_id,
                    node,
                    float(point[0]).hex(),
                    float(point[1]).hex(),
                )
            )
    _write_bytes(
        directory / "coordinates.csv.gz",
        _csv_bytes(("condition_id", "node", "x_float64_hex", "y_float64_hex"), coordinate_rows),
    )
    pair_rows = [
        (index, source, destination)
        for index, (source, destination) in enumerate(data.pairs)
    ]
    _write_bytes(
        directory / "pairs.csv.gz",
        _csv_bytes(("pair_index", "source", "destination"), pair_rows),
    )
    _write_json(directory / "embedding_metadata.json", data.embedding_metadata)
    _write_json(directory / "distortions.json", list(data.distortions))
    _write_bytes(
        directory / "dijkstra.jsonl.gz",
        _jsonl_bytes(data.dijkstra_records),
    )
    _write_bytes(
        directory / "routes.jsonl.gz",
        _jsonl_bytes(data.route_records),
    )
    _write_json(directory / "run_manifest.json", data.run_manifest)
    payload_serialization_ns = perf_counter_ns() - serialization_start_ns
    effective_graph_start_ns = (
        graph_wall_start_ns
        if graph_wall_start_ns is not None
        else serialization_start_ns
    )
    timings = dict(data.timings)
    timings["payload_serialization_ns"] = payload_serialization_ns
    timings["prepublication_wall_ns"] = (
        perf_counter_ns() - effective_graph_start_ns
    )
    _write_json(directory / "timings.json", timings)
    return timings


def _payload_file_records(directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for filename in DATA_FILENAMES:
        path = directory / filename
        if not path.is_file():
            raise CheckpointCorruptionError(f"checkpoint payload is missing {filename}")
        digest, size = _file_digest(path)
        records.append({"path": filename, "sha256": digest, "size_bytes": size})
    return records


def _write_error_report(
    temporary: Path,
    *,
    graph_id: str,
    stage: str,
    exception: BaseException,
    run_manifest: Mapping[str, object],
    final_checkpoint_path: Path | None,
) -> None:
    report = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "graph_id": graph_id,
        "stage": stage,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "temporary_checkpoint_path": str(temporary.resolve(strict=False)),
        "final_checkpoint_path": (
            str(final_checkpoint_path.resolve(strict=False))
            if final_checkpoint_path is not None
            else None
        ),
        "run_identity": _run_identity(run_manifest),
        "run_manifest": dict(run_manifest),
        "run_manifest_sha256": run_manifest_sha256(run_manifest),
    }
    path = temporary / ERROR_REPORT_FILENAME
    if not path.exists():
        try:
            _write_json(path, report)
        except OSError:
            pass


def preserve_graph_error(
    run_root: Path | str,
    *,
    graph_id: str,
    stage: str,
    exception: BaseException,
    run_manifest: Mapping[str, object],
) -> Path:
    """Preserve a non-complete per-graph error directory for diagnosis."""

    validated_id = _validate_graph_id(graph_id)
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    root = Path(run_root).resolve(strict=False)
    if not root.is_dir():
        raise CheckpointError("run root must exist before preserving an error")
    graph_parent = _resolved_child(root, CHECKPOINT_DIRECTORY)
    graph_parent.mkdir(exist_ok=True)
    temporary = _resolved_child(
        graph_parent,
        f".{validated_id}{TEMPORARY_DIRECTORY_TOKEN}{uuid4().hex}",
    )
    temporary.mkdir()
    _write_error_report(
        temporary,
        graph_id=validated_id,
        stage=stage,
        exception=exception,
        run_manifest=run_manifest,
        final_checkpoint_path=_resolved_child(graph_parent, validated_id),
    )
    return temporary


def publish_graph_checkpoint(
    run_root: Path | str,
    data: GraphCheckpointData,
    *,
    graph_wall_start_ns: int | None = None,
    event_callback: Callable[[str, Path], None] | None = None,
) -> CheckpointValidation:
    """Write, validate, and atomically publish exactly one graph checkpoint.

    Existing complete checkpoints are never overwritten.  On any error the
    temporary directory and an error report are retained for diagnosis.
    """

    counts = _validate_data(data)
    root = Path(run_root).resolve(strict=False)
    if not root.is_dir():
        raise CheckpointError("run root must exist before checkpoint publication")
    graph_parent = _resolved_child(root, CHECKPOINT_DIRECTORY)
    graph_parent.mkdir(exist_ok=True)
    final = _resolved_child(graph_parent, data.graph_id)
    if final.exists():
        raise CheckpointError(f"checkpoint already exists: {final}")
    temporary = _resolved_child(
        graph_parent,
        f".{data.graph_id}{TEMPORARY_DIRECTORY_TOKEN}{uuid4().hex}",
    )
    temporary.mkdir()
    serialization_start = perf_counter_ns()
    try:
        if event_callback is not None:
            event_callback("temporary_directory_created", temporary)
        payload_timings = _write_payload_files(
            temporary,
            data,
            serialization_start_ns=serialization_start,
            graph_wall_start_ns=graph_wall_start_ns,
        )
        file_records = _payload_file_records(temporary)
        graph_manifest = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "graph_id": data.graph_id,
            "run_identity": _run_identity(data.run_manifest),
            "counts": counts,
            "files": file_records,
        }
        _write_json(temporary / GRAPH_MANIFEST_FILENAME, graph_manifest)
        if event_callback is not None:
            event_callback("before_complete_marker", temporary)
        manifest_hash, manifest_size = _file_digest(
            temporary / GRAPH_MANIFEST_FILENAME
        )
        _write_json(
            temporary / COMPLETE_MARKER_FILENAME,
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "graph_id": data.graph_id,
                "graph_manifest_sha256": manifest_hash,
                "graph_manifest_size_bytes": manifest_size,
            },
        )
        if event_callback is not None:
            event_callback("complete_marker_written", temporary)
        validate_graph_checkpoint(
            temporary,
            expected_run_manifest=data.run_manifest,
            expected_graph_id=data.graph_id,
        )
        atomic_publication_start = perf_counter_ns()
        if event_callback is not None:
            event_callback("before_atomic_publication", temporary)
        os.rename(temporary, final)
        if event_callback is not None:
            event_callback("checkpoint_renamed", final)
        validation = validate_graph_checkpoint(
            final,
            expected_run_manifest=data.run_manifest,
            expected_graph_id=data.graph_id,
        )
        publication_completed = perf_counter_ns()
        if event_callback is not None:
            event_callback("published_checkpoint_validated", final)
        effective_graph_start = (
            graph_wall_start_ns
            if graph_wall_start_ns is not None
            else serialization_start
        )
        write_publication_timing_record(
            root,
            graph_id=data.graph_id,
            run_manifest=data.run_manifest,
            payload_serialization_ns=payload_timings[
                "payload_serialization_ns"
            ],
            prepublication_wall_ns=payload_timings["prepublication_wall_ns"],
            atomic_publication_and_final_validation_ns=(
                publication_completed - atomic_publication_start
            ),
            end_to_end_graph_wall_ns=publication_completed - effective_graph_start,
        )
        if event_callback is not None:
            event_callback("publication_timing_record_written", final)
            event_callback("checkpoint_published", final)
        return CheckpointValidation(
            graph_id=validation.graph_id,
            path=final,
            counts=validation.counts,
            payload_sha256=validation.payload_sha256,
        )
    except BaseException as exc:
        error_directory = temporary
        if not error_directory.exists():
            error_directory = _resolved_child(
                graph_parent,
                f".{data.graph_id}{TEMPORARY_DIRECTORY_TOKEN}{uuid4().hex}",
            )
            error_directory.mkdir()
        _write_error_report(
            error_directory,
            graph_id=data.graph_id,
            stage="checkpoint_publication",
            exception=exc,
            run_manifest=data.run_manifest,
            final_checkpoint_path=final,
        )
        raise


def _validate_csv_row_count(path: Path, expected: int) -> None:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            count = sum(1 for _ in csv.reader(stream)) - 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise CheckpointCorruptionError(f"invalid gzip CSV file: {path}") from exc
    if count != expected:
        raise CheckpointCorruptionError(
            f"{path.name} row count is {count}, expected {expected}"
        )


def _validate_jsonl_count(path: Path, expected: int) -> None:
    count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                decode_json_value(value)
                count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError(f"invalid gzip JSONL file: {path}") from exc
    if count != expected:
        raise CheckpointCorruptionError(
            f"{path.name} row count is {count}, expected {expected}"
        )


def validate_graph_checkpoint(
    checkpoint_path: Path | str,
    *,
    expected_run_manifest: Mapping[str, object] | None = None,
    expected_graph_id: str | None = None,
) -> CheckpointValidation:
    """Fully validate a complete checkpoint without modifying it."""

    path = Path(checkpoint_path).resolve(strict=False)
    if not path.is_dir():
        raise CheckpointCorruptionError(f"checkpoint directory is missing: {path}")
    manifest_path = path / GRAPH_MANIFEST_FILENAME
    marker_path = path / COMPLETE_MARKER_FILENAME
    if not marker_path.is_file():
        raise CheckpointCorruptionError(f"completion marker is missing: {path}")
    marker = _read_json(marker_path)
    manifest = _read_json(manifest_path)
    if not isinstance(marker, dict) or not isinstance(manifest, dict):
        raise CheckpointCorruptionError("checkpoint marker and manifest must be objects")
    graph_id = manifest.get("graph_id")
    if not isinstance(graph_id, str):
        raise CheckpointCorruptionError("checkpoint graph_id is invalid")
    _validate_graph_id(graph_id)
    if expected_graph_id is not None and graph_id != expected_graph_id:
        raise CheckpointCompatibilityError(
            f"checkpoint graph_id {graph_id} does not match {expected_graph_id}"
        )
    if marker.get("graph_id") != graph_id:
        raise CheckpointCorruptionError("completion marker graph identity mismatch")
    if (
        marker.get("result_schema_version") != RESULT_SCHEMA_VERSION
        or manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION
    ):
        raise CheckpointCompatibilityError("unsupported result schema version")
    manifest_hash, manifest_size = _file_digest(manifest_path)
    if (
        marker.get("graph_manifest_sha256") != manifest_hash
        or marker.get("graph_manifest_size_bytes") != manifest_size
    ):
        raise CheckpointCorruptionError("graph manifest integrity mismatch")

    run_identity = manifest.get("run_identity")
    if not isinstance(run_identity, dict):
        raise CheckpointCorruptionError("checkpoint run identity is invalid")
    if expected_run_manifest is not None:
        if _json_bytes(run_identity) != _json_bytes(
            _run_identity(expected_run_manifest)
        ):
            raise CheckpointCompatibilityError(
                f"checkpoint {graph_id} belongs to a different run identity"
            )

    counts = manifest.get("counts")
    files = manifest.get("files")
    if not isinstance(counts, dict) or not isinstance(files, list):
        raise CheckpointCorruptionError("checkpoint manifest schema is invalid")
    payload_hashes: dict[str, str] = {}
    expected_names = set(DATA_FILENAMES)
    seen_names: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise CheckpointCorruptionError("invalid checkpoint file record")
        filename = record.get("path")
        if not isinstance(filename, str) or filename not in expected_names:
            raise CheckpointCorruptionError("unexpected checkpoint payload path")
        if filename in seen_names:
            raise CheckpointCorruptionError("duplicate checkpoint payload path")
        seen_names.add(filename)
        payload_path = path / filename
        if not payload_path.is_file():
            raise CheckpointCorruptionError(
                f"checkpoint payload is missing: {payload_path}"
            )
        digest, size = _file_digest(payload_path)
        if digest != record.get("sha256") or size != record.get("size_bytes"):
            raise CheckpointCorruptionError(
                f"payload integrity mismatch: {payload_path}"
            )
        payload_hashes[filename] = digest
    if seen_names != expected_names:
        raise CheckpointCorruptionError("checkpoint payload file set is incomplete")

    _validate_csv_row_count(path / "edges.csv.gz", int(counts["edges"]))
    _validate_csv_row_count(
        path / "coordinates.csv.gz", int(counts["coordinate_rows"])
    )
    _validate_csv_row_count(path / "pairs.csv.gz", int(counts["pairs"]))
    _validate_jsonl_count(
        path / "dijkstra.jsonl.gz", int(counts["dijkstra_records"])
    )
    _validate_jsonl_count(path / "routes.jsonl.gz", int(counts["route_records"]))
    distortions = _read_json(path / "distortions.json")
    if not isinstance(distortions, list) or len(distortions) != int(
        counts["distortion_records"]
    ):
        raise CheckpointCorruptionError("distortion record count mismatch")
    for filename in (
        "generation.json",
        "network_metrics.json",
        "embedding_metadata.json",
        "timings.json",
    ):
        if not isinstance(_read_json(path / filename), dict):
            raise CheckpointCorruptionError(f"{filename} must contain an object")
    checkpoint_run_manifest = _read_json(path / "run_manifest.json")
    if not isinstance(checkpoint_run_manifest, dict):
        raise CheckpointCorruptionError(
            "checkpoint run_manifest.json must contain an object"
        )
    if _json_bytes(_run_identity(checkpoint_run_manifest)) != _json_bytes(
        run_identity
    ):
        raise CheckpointCompatibilityError(
            "checkpoint run manifest does not match graph manifest identity"
        )
    if expected_run_manifest is not None:
        validate_run_manifest_compatibility(
            checkpoint_run_manifest,
            expected_run_manifest,
        )
    return CheckpointValidation(
        graph_id=graph_id,
        path=path,
        counts={key: int(value) for key, value in counts.items()},
        payload_sha256=payload_hashes,
    )


def audit_run_checkpoints(
    run_root: Path | str,
    *,
    schedule_ids: Sequence[str],
    expected_run_manifest: Mapping[str, object] | None = None,
) -> CheckpointAudit:
    """Read-only validation of run provenance and every published checkpoint."""

    root = Path(run_root).resolve(strict=False)
    scheduled = tuple(_validate_graph_id(item) for item in schedule_ids)
    errors: list[str] = []
    complete: list[str] = []
    manifest: Mapping[str, object] | None = None
    manifest_path = root / RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            loaded = _read_json(manifest_path)
            if not isinstance(loaded, dict):
                raise CheckpointCorruptionError("run manifest must be an object")
            manifest = loaded
            if expected_run_manifest is not None:
                validate_run_manifest_compatibility(loaded, expected_run_manifest)
            manifest_schedule = tuple(loaded.get("schedule", ()))
            if manifest_schedule != scheduled:
                raise CheckpointCompatibilityError(
                    "run manifest schedule does not match the canonical schedule"
                )
        except (CheckpointError, ValueError) as exc:
            errors.append(str(exc))
    elif root.exists():
        errors.append(f"run manifest is missing: {manifest_path}")

    graph_parent = root / CHECKPOINT_DIRECTORY
    if graph_parent.exists():
        for child in sorted(graph_parent.iterdir(), key=lambda item: item.name):
            if TEMPORARY_DIRECTORY_TOKEN in child.name:
                errors.append(f"incomplete temporary checkpoint exists: {child}")
                continue
            if not child.is_dir():
                errors.append(f"unexpected checkpoint entry: {child}")
                continue
            if child.name not in scheduled:
                errors.append(f"checkpoint is outside the canonical schedule: {child}")
                continue
            try:
                validate_graph_checkpoint(
                    child,
                    expected_run_manifest=manifest or expected_run_manifest,
                    expected_graph_id=child.name,
                )
                timing_manifest = manifest or expected_run_manifest
                if timing_manifest is None:
                    raise CheckpointCompatibilityError(
                        "operational-integrity error: no run manifest is available "
                        "to validate publication timing"
                    )
                validate_publication_timing_record(
                    root,
                    graph_id=child.name,
                    expected_run_manifest=timing_manifest,
                )
                complete.append(child.name)
            except (CheckpointError, ValueError, OSError, KeyError) as exc:
                errors.append(f"{child}: {exc}")
    timing_parent = root / PUBLICATION_TIMING_DIRECTORY
    if timing_parent.exists():
        for child in sorted(timing_parent.iterdir(), key=lambda item: item.name):
            if child.name.startswith(".") or child.suffix != ".json":
                errors.append(
                    f"unexpected publication timing entry: {child}"
                )
                continue
            if not child.is_file():
                errors.append(
                    f"unexpected publication timing entry: {child}"
                )
                continue
            graph_id = child.stem
            if graph_id not in scheduled:
                errors.append(
                    "publication timing is outside the canonical schedule: "
                    f"{child}"
                )
            elif graph_id not in complete:
                errors.append(
                    "publication timing exists without a validated checkpoint: "
                    f"{child}"
                )
    complete_set = set(complete)
    remaining = tuple(item for item in scheduled if item not in complete_set)
    return CheckpointAudit(
        run_root=root,
        run_manifest_present=manifest_path.is_file(),
        complete_graph_ids=tuple(item for item in scheduled if item in complete_set),
        remaining_graph_ids=remaining,
        errors=tuple(errors),
        resumable=not errors,
    )


def deterministic_payload_hashes(
    checkpoint: Path | str,
) -> Mapping[str, str]:
    """Return scientific payload hashes, excluding timing/provenance wrappers."""

    validation = validate_graph_checkpoint(checkpoint)
    excluded = {"timings.json", "run_manifest.json"}
    return {
        name: digest
        for name, digest in validation.payload_sha256.items()
        if name not in excluded
    }
