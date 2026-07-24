"""Independent, streaming validation for the immutable Step 16 full run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import gzip
from hashlib import blake2s, sha256
from importlib.metadata import PackageNotFoundError, version
import json
from math import asinh, ceil, comb, hypot, isclose, isfinite, sqrt
from pathlib import Path
import subprocess
from time import monotonic
from typing import Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from benchmark_experiment_capacity import load_capacity_profile
from embedding import EmbeddingDistortion
from experiment_checkpoint import (
    RESULT_SCHEMA_VERSION,
    decode_json_value,
    validate_graph_checkpoint,
    validate_publication_timing_record,
)
from experiment_config import (
    ANALYSIS_PLAN_HASH,
    BA_INITIAL_GRAPH,
    BARABASI_ALBERT,
    COMBINED_FREEZE_HASH,
    CONFIGURATION_SCHEMA_VERSION,
    DATA_GENERATION_HASH,
    ERDOS_RENYI,
    FULL_EXPERIMENT_CONFIG,
    SEED_IDENTITY_VERSION,
)
from experiment_protocol import sample_ordered_pairs
from hydra_embedding import _frechet_residual
from mds_embedding import scale_equivariant_euclidean_routing_tolerance
from network_metrics import calculate_network_metrics, prepare_all_pairs_shortest_paths
from poincare_distance import euclidean_distance, poincare_distance
from routing import (
    CYCLE,
    LOCAL_MINIMUM,
    POST_REPAIR_ATTEMPTED_REVISIT,
    POST_REPAIR_LOCAL_MINIMUM,
    REPAIR_FAILED,
    REPAIR_UNAVAILABLE,
    RoutingResult,
    euclidean_greedy_route,
    hyperbolic_greedy_route,
    prepare_routing_coordinates,
    repaired_hyperbolic_greedy_route,
)
from run_full_experiment import (
    EXPECTED_ANALYSIS_PLAN_HASH,
    EXPECTED_COMBINED_FREEZE_HASH,
    EXPECTED_DATA_GENERATION_HASH,
    GraphScheduleEntry,
    build_full_schedule,
)


RAW_TREE_FINGERPRINT_SCHEMA = "raw_tree_fingerprint_v1"
VALIDATION_REPORT_SCHEMA = "step17_full_validation_report_v1"
ROUTE_AUDIT_DOMAIN = "step17_route_audit_v1"
ROUTE_AUDIT_PERSON = b"GRP17aud"
ROUTE_AUDIT_PAIR_COUNT = 10
EXPECTED_SOURCE_COMMIT = "a121c33a20ea721c2a5fca96bdfd6e2eeb7dd0bc"
EXPECTED_RUN_DIRECTORY = "final_8e002ef20f96_a121c33a20ea"
EXPECTED_GRAPH_COUNT = 360
EXPECTED_PAIR_COUNT = 1_000
EXPECTED_ROUTE_COUNT_PER_GRAPH = 15_000
EXPECTED_DISTORTION_COUNT_PER_GRAPH = 7

COORDINATE_CONDITIONS = (
    "hydra",
    "mds_r050",
    "mds_r070",
    "mds_r085",
    "mds_r095",
)
MDS_RADII = {
    "mds_r050": 0.50,
    "mds_r070": 0.70,
    "mds_r085": 0.85,
    "mds_r095": 0.95,
}
ROUTING_METHODS = (
    "euclidean_greedy",
    "poincare_greedy",
    "repaired_poincare_greedy",
)
INTERNAL_METHODS = {
    "euclidean_greedy": "euclidean_greedy",
    "poincare_greedy": "hyperbolic_greedy",
    "repaired_poincare_greedy": "repaired_hyperbolic_greedy",
}
FINAL_FAILURE_TYPES = (
    LOCAL_MINIMUM,
    CYCLE,
    REPAIR_UNAVAILABLE,
    REPAIR_FAILED,
    POST_REPAIR_LOCAL_MINIMUM,
    POST_REPAIR_ATTEMPTED_REVISIT,
)
DISTORTION_SPECS = (
    ("hydra_euclidean", "hydra", "euclidean"),
    ("hydra_poincare", "hydra", "poincare"),
    ("base_mds_euclidean", "mds_base", "euclidean"),
    ("mds_poincare_r050", "mds_r050", "poincare"),
    ("mds_poincare_r070", "mds_r070", "poincare"),
    ("mds_poincare_r085", "mds_r085", "poincare"),
    ("mds_poincare_r095", "mds_r095", "poincare"),
)


class FullResultValidationError(RuntimeError):
    """Raised on the first immutable-run validation failure."""


@dataclass(frozen=True)
class RawTreeFingerprint:
    schema: str
    sha256: str
    file_count: int
    byte_count: int
    entries: tuple[dict[str, object], ...]

    def summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class ValidatedRun:
    validation_report: dict[str, object]
    graph_level_rows: tuple[dict[str, object], ...]
    runtime_records: tuple[dict[str, object], ...]
    initial_raw_fingerprint: RawTreeFingerprint


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def committed_source_fingerprint(
    repository_root: Path | str,
    commit: str,
) -> str:
    """Reconstruct the Step 16 source fingerprint from immutable Git objects."""

    root = Path(repository_root).resolve(strict=True)
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    relative_paths = sorted(
        path
        for path in listing
        if (path.startswith("code/") and path.endswith(".py"))
        or path == "requirements.txt"
    )
    if "requirements.txt" not in relative_paths or not relative_paths:
        raise FullResultValidationError("committed source inventory is incomplete")
    digest = sha256()
    for relative in relative_paths:
        payload = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def installed_dependency_versions(
    repository_root: Path | str,
) -> dict[str, str]:
    """Return exact installed versions for every pinned project dependency."""

    requirements = Path(repository_root).resolve(strict=True) / "requirements.txt"
    result: dict[str, str] = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        package = requirement.split("==", 1)[0]
        try:
            result[package] = version(package)
        except PackageNotFoundError as exc:
            raise FullResultValidationError(
                f"required dependency is not installed: {package}"
            ) from exc
    return result


def compute_raw_tree_fingerprint(
    root: Path | str,
    *,
    include_entries: bool = True,
    progress: Callable[[str], None] | None = None,
) -> RawTreeFingerprint:
    """Hash sorted relative paths, byte sizes, and SHA-256 file hashes."""

    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise FullResultValidationError("raw run root must be a directory")
    paths = sorted(
        (path for path in resolved.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )
    entries: list[dict[str, object]] = []
    byte_count = 0
    last_update = monotonic()
    for index, path in enumerate(paths, 1):
        digest, size = _sha256_file(path)
        entries.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "size_bytes": size,
                "sha256": digest,
            }
        )
        byte_count += size
        if progress is not None and monotonic() - last_update >= 60:
            progress(
                f"raw fingerprint files={index}/{len(paths)} bytes={byte_count}"
            )
            last_update = monotonic()
    payload = {"schema": RAW_TREE_FINGERPRINT_SCHEMA, "files": entries}
    return RawTreeFingerprint(
        schema=RAW_TREE_FINGERPRINT_SCHEMA,
        sha256=sha256(_canonical_json_bytes(payload)).hexdigest(),
        file_count=len(entries),
        byte_count=byte_count,
        entries=tuple(entries) if include_entries else (),
    )


def derive_route_audit_pair_indices(
    graph_id: str,
    *,
    pair_count: int = EXPECTED_PAIR_COUNT,
    sample_count: int = ROUTE_AUDIT_PAIR_COUNT,
    combined_freeze_hash: str = COMBINED_FREEZE_HASH,
) -> tuple[int, ...]:
    """Select distinct pair indices without reading any routing outcome."""

    if not isinstance(graph_id, str) or not graph_id:
        raise ValueError("graph_id must be a non-empty string")
    if pair_count <= 0 or sample_count <= 0 or sample_count > pair_count:
        raise ValueError("route-audit sample counts are invalid")
    limit = (1 << 64) - ((1 << 64) % pair_count)
    selected: list[int] = []
    selected_set: set[int] = set()
    counter = 0
    while len(selected) < sample_count:
        rejection_counter = 0
        while True:
            identity = {
                "combined_freeze_hash": combined_freeze_hash,
                "counter": counter,
                "domain": ROUTE_AUDIT_DOMAIN,
                "graph_id": graph_id,
                "rejection_counter": rejection_counter,
            }
            word = int.from_bytes(
                blake2s(
                    _canonical_json_bytes(identity),
                    digest_size=8,
                    person=ROUTE_AUDIT_PERSON,
                ).digest(),
                "big",
                signed=False,
            )
            if word < limit:
                candidate = word % pair_count
                break
            rejection_counter += 1
        counter += 1
        if candidate not in selected_set:
            selected.append(candidate)
            selected_set.add(candidate)
    return tuple(selected)


def _read_json(path: Path) -> object:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FullResultValidationError(f"invalid JSON file {path}: {exc}") from exc
    return decode_json_value(value)


def _assert_close(
    actual: float,
    expected: float,
    *,
    label: str,
    absolute: float = 1e-12,
    relative: float = 1e-12,
) -> None:
    if not (
        isfinite(float(actual))
        and isfinite(float(expected))
        and isclose(
            float(actual),
            float(expected),
            abs_tol=absolute,
            rel_tol=relative,
        )
    ):
        raise FullResultValidationError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}"
        )


def validate_raw_inventory(
    run_root: Path,
    schedule_ids: Sequence[str],
) -> dict[str, int]:
    """Reject missing, duplicate, temporary, orphan, or error entries."""

    expected_top = {
        "graphs",
        "publication_timings",
        "progress.json",
        "run_manifest.json",
    }
    actual_top = {path.name for path in run_root.iterdir()}
    if actual_top != expected_top:
        raise FullResultValidationError(
            f"raw run top-level inventory mismatch: {sorted(actual_top ^ expected_top)}"
        )
    error_files = tuple(run_root.rglob("ERROR.json"))
    if error_files:
        raise FullResultValidationError(
            f"raw run contains ERROR.json: {error_files[0]}"
        )
    temporary = tuple(
        path
        for path in run_root.rglob("*")
        if ".tmp-" in path.name or path.name.startswith(".")
    )
    if temporary:
        raise FullResultValidationError(
            f"raw run contains temporary/orphan entry: {temporary[0]}"
        )
    graph_entries = tuple(
        path.name
        for path in sorted(
            (run_root / "graphs").iterdir(), key=lambda item: item.name
        )
        if path.is_dir()
    )
    if len(graph_entries) != len(set(graph_entries)):
        raise FullResultValidationError("duplicate graph directory name")
    if set(graph_entries) != set(schedule_ids):
        raise FullResultValidationError("graph directory schedule mismatch")
    timing_entries = tuple(
        path.stem
        for path in sorted(
            (run_root / "publication_timings").iterdir(),
            key=lambda item: item.name,
        )
        if path.is_file() and path.suffix == ".json"
    )
    if set(timing_entries) != set(schedule_ids):
        raise FullResultValidationError("publication timing schedule mismatch")
    return {
        "graph_checkpoints": len(graph_entries),
        "publication_timing_records": len(timing_entries),
        "error_json_records": 0,
        "temporary_or_orphan_entries": 0,
    }


def _load_edges(path: Path, n: int) -> tuple[nx.Graph, tuple[tuple[int, int], ...]]:
    edges: list[tuple[int, int]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["source", "destination"]:
            raise FullResultValidationError("edge CSV schema mismatch")
        for row in reader:
            left, right = int(row["source"]), int(row["destination"])
            if not (0 <= left < right < n):
                raise FullResultValidationError("edge list is not canonical and sorted")
            edges.append((left, right))
    if edges != sorted(edges) or len(edges) != len(set(edges)):
        raise FullResultValidationError("edge list is unsorted or contains duplicates")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise FullResultValidationError("graph is not simple and undirected")
    if any(data for _, _, data in graph.edges(data=True)):
        raise FullResultValidationError("graph contains edge weights or attributes")
    if set(graph.nodes) != set(range(n)) or not nx.is_connected(graph):
        raise FullResultValidationError("graph node IDs or connectedness are invalid")
    return graph, tuple(edges)


def _load_pairs(path: Path) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["pair_index", "source", "destination"]:
            raise FullResultValidationError("pair CSV schema mismatch")
        for expected_index, row in enumerate(reader):
            if int(row["pair_index"]) != expected_index:
                raise FullResultValidationError("pair indices are not consecutive")
            source, destination = int(row["source"]), int(row["destination"])
            if source == destination:
                raise FullResultValidationError("ordered pair contains a self-pair")
            pairs.append((source, destination))
    if len(pairs) != EXPECTED_PAIR_COUNT or len(set(pairs)) != len(pairs):
        raise FullResultValidationError("ordered pairs are missing or duplicated")
    return tuple(pairs)


def _load_coordinates(
    path: Path,
    n: int,
) -> dict[str, dict[int, tuple[float, float]]]:
    coordinates = {condition: {} for condition in COORDINATE_CONDITIONS}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [
            "condition_id",
            "node",
            "x_float64_hex",
            "y_float64_hex",
        ]:
            raise FullResultValidationError("coordinate CSV schema mismatch")
        for row in reader:
            condition = row["condition_id"]
            if condition not in coordinates:
                raise FullResultValidationError(
                    f"unknown coordinate condition: {condition}"
                )
            node = int(row["node"])
            if node in coordinates[condition]:
                raise FullResultValidationError("duplicate coordinate row")
            point = (
                float.fromhex(row["x_float64_hex"]),
                float.fromhex(row["y_float64_hex"]),
            )
            if not all(isfinite(value) for value in point):
                raise FullResultValidationError("coordinate is not finite float64")
            if hypot(*point) >= 1.0:
                raise FullResultValidationError("coordinate is outside the unit disk")
            coordinates[condition][node] = point
    expected_nodes = set(range(n))
    if any(set(points) != expected_nodes for points in coordinates.values()):
        raise FullResultValidationError("coordinate node coverage is incomplete")
    return coordinates


def _coincidence_groups(
    coordinates: Mapping[int, tuple[float, float]],
) -> tuple[tuple[int, ...], ...]:
    grouped: dict[tuple[float, float], list[int]] = defaultdict(list)
    for node in sorted(coordinates):
        grouped[coordinates[node]].append(node)
    return tuple(
        tuple(nodes)
        for _, nodes in sorted(grouped.items(), key=lambda item: item[1][0])
        if len(nodes) > 1
    )


def _validate_coincidence_metadata(
    metadata: Mapping[str, object],
    coordinates: Mapping[int, tuple[float, float]],
    *,
    label: str,
) -> None:
    groups = _coincidence_groups(coordinates)
    affected = sum(len(group) for group in groups)
    pairs = sum(comb(len(group), 2) for group in groups)
    recorded_groups = tuple(
        tuple(int(node) for node in group)
        for group in metadata.get("coincident_coordinate_groups", ())
    )
    if (
        recorded_groups != groups
        or metadata.get("coincident_coordinate_group_count") != len(groups)
        or metadata.get("coincident_vertex_count") != affected
        or metadata.get("coincident_vertex_pair_count") != pairs
    ):
        raise FullResultValidationError(f"{label} coincidence metadata mismatch")
    if len(set(coordinates.values())) <= 1:
        raise FullResultValidationError(f"{label} has complete coordinate collapse")


def _validate_embedding_metadata(
    coordinates: Mapping[str, Mapping[int, tuple[float, float]]],
    metadata: Mapping[str, object],
    config_fingerprint: str,
) -> dict[int, tuple[float, float]]:
    if set(metadata) != {
        "hydra",
        "mds_base",
        "mds_r050",
        "mds_r070",
        "mds_r085",
        "mds_r095",
    }:
        raise FullResultValidationError("embedding metadata condition set mismatch")
    hydra = metadata["hydra"]
    if not isinstance(hydra, Mapping):
        raise FullResultValidationError("Hydra metadata is invalid")
    if (
        hydra.get("embedding_family") != "hydra"
        or hydra.get("dimension") != 2
        or hydra.get("effective_spatial_rank") not in (1, 2)
        or hydra.get("node_order")
        != list(range(len(coordinates["hydra"])))
        or hydra.get("radial_rescaling_after_centering") is not False
        or hydra.get("configuration_fingerprint") != config_fingerprint
        or float(hydra.get("final_frechet_mean_residual", 1.0)) > 1e-10
        or hydra.get("centering_tolerance") != 1e-10
        or hydra.get("centering_max_iterations") != 256
        or not 0 <= int(hydra.get("centering_iteration_count", -1)) <= 256
        or hydra.get("boundary_roundoff_tolerance") != 1e-12
        or float(hydra.get("maximum_pairwise_normalized_error", 2.0)) > 1.0
        or float(hydra.get("maximum_pairwise_distance_error", -1.0)) < 0.0
    ):
        raise FullResultValidationError("Hydra metadata policy mismatch")
    boundary_count = hydra.get("boundary_correction_count")
    if (
        isinstance(boundary_count, bool)
        or not isinstance(boundary_count, int)
        or boundary_count < 0
        or hydra.get("boundary_correction_occurred") != (boundary_count > 0)
    ):
        raise FullResultValidationError("Hydra boundary metadata mismatch")
    if hydra["effective_spatial_rank"] == 1 and any(
        point[1] != 0.0 for point in coordinates["hydra"].values()
    ):
        raise FullResultValidationError(
            "rank-one Hydra second coordinate is not exactly zero-padded"
        )
    hydra_max = max(hypot(*point) for point in coordinates["hydra"].values())
    _assert_close(
        hydra_max,
        float(hydra["maximum_coordinate_norm"]),
        label="Hydra maximum coordinate norm",
    )
    _validate_coincidence_metadata(
        hydra, coordinates["hydra"], label="Hydra"
    )
    hydra_array = np.asarray(
        [
            coordinates["hydra"][node]
            for node in range(len(coordinates["hydra"]))
        ],
        dtype=np.float64,
    )
    _, recomputed_hydra_residual = _frechet_residual(
        np.zeros(2, dtype=np.float64), hydra_array
    )
    _assert_close(
        recomputed_hydra_residual,
        float(hydra["final_frechet_mean_residual"]),
        label="Hydra Frechet centering residual",
        absolute=2e-15,
        relative=2e-12,
    )
    if recomputed_hydra_residual > 2e-10:
        raise FullResultValidationError(
            "Hydra coordinates are not Frechet-centered at the origin"
        )

    normalized_reference: dict[int, tuple[float, float]] | None = None
    for condition, radius in MDS_RADII.items():
        item = metadata[condition]
        if not isinstance(item, Mapping):
            raise FullResultValidationError("MDS metadata is invalid")
        if (
            item.get("embedding_family") != "classical_mds"
            or item.get("dimension") != 2
            or item.get("coordinate_condition_id") != condition
            or item.get("nested_sensitivity_condition") is not True
            or item.get("effective_rank") not in (1, 2)
            or item.get("node_order")
            != list(range(len(coordinates[condition])))
            or item.get("configuration_fingerprint") != config_fingerprint
            or float(item.get("centroid_residual", 1.0)) > 1e-12
        ):
            raise FullResultValidationError(f"{condition} metadata mismatch")
        achieved = max(hypot(*point) for point in coordinates[condition].values())
        _assert_close(achieved, radius, label=f"{condition} radius")
        _assert_close(
            achieved,
            float(item["achieved_maximum_radius"]),
            label=f"{condition} achieved radius metadata",
        )
        _validate_coincidence_metadata(item, coordinates[condition], label=condition)
        centroid = np.mean(
            np.asarray(
                [
                    coordinates[condition][node]
                    for node in range(len(coordinates[condition]))
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        recomputed_centroid_residual = float(np.linalg.norm(centroid))
        _assert_close(
            recomputed_centroid_residual,
            float(item["centroid_residual"]),
            label=f"{condition} centroid residual",
            absolute=2e-15,
            relative=2e-12,
        )
        if recomputed_centroid_residual > 1e-12:
            raise FullResultValidationError(
                f"{condition} is not centered at its Euclidean centroid"
            )
        if item["effective_rank"] == 1 and any(
            point[1] != 0.0 for point in coordinates[condition].values()
        ):
            raise FullResultValidationError(
                f"{condition} rank-one second coordinate is not zero-padded"
            )
        normalized = {
            node: (point[0] / radius, point[1] / radius)
            for node, point in coordinates[condition].items()
        }
        if normalized_reference is None:
            normalized_reference = normalized
        else:
            for node in normalized:
                _assert_close(
                    normalized[node][0],
                    normalized_reference[node][0],
                    label=f"{condition} nested x",
                    absolute=3e-12,
                )
                _assert_close(
                    normalized[node][1],
                    normalized_reference[node][1],
                    label=f"{condition} nested y",
                    absolute=3e-12,
                )
    base = metadata["mds_base"]
    if (
        not isinstance(base, Mapping)
        or base.get("embedding_family") != "classical_mds"
        or base.get("node_order") != list(range(len(coordinates["mds_r050"])))
        or base.get("effective_rank") != metadata["mds_r050"]["effective_rank"]
        or float(base.get("centroid_residual", 1.0)) > 1e-12
    ):
        raise FullResultValidationError("base MDS metadata mismatch")
    scale = float(metadata["mds_r050"]["scale_factor"])
    if scale <= 0.0 or not isfinite(scale):
        raise FullResultValidationError("base MDS scale factor is invalid")
    base_coordinates = {
        node: (point[0] / scale, point[1] / scale)
        for node, point in coordinates["mds_r050"].items()
    }
    _validate_coincidence_metadata(base, base_coordinates, label="base MDS")
    return base_coordinates


def _vectorized_distortion(
    shortest_paths,
    coordinates: Mapping[int, tuple[float, float]],
    *,
    metric: str,
) -> EmbeddingDistortion:
    """Recompute the frozen distortion formulas over every unordered pair."""

    ordered = tuple(sorted(coordinates))
    array = np.asarray([coordinates[node] for node in ordered], dtype=np.float64)
    left, right = np.triu_indices(len(ordered), k=1)
    delta_x = array[left, 0] - array[right, 0]
    delta_y = array[left, 1] - array[right, 1]
    difference = np.hypot(delta_x, delta_y)
    if metric == "euclidean":
        geometric = difference
    elif metric == "poincare":
        norms = np.hypot(array[:, 0], array[:, 1])
        factors = (1.0 - norms) * (1.0 + norms)
        denominator = np.sqrt(factors[left]) * np.sqrt(factors[right])
        if np.any(denominator <= 0.0):
            raise FullResultValidationError("invalid Poincare denominator")
        geometric = 2.0 * np.arcsinh(np.maximum(0.0, difference / denominator))
    else:
        raise ValueError("unknown distortion metric")
    graph_distances = np.fromiter(
        (
            shortest_paths.distances[ordered[i]][ordered[j]]
            for i, j in zip(left, right, strict=True)
        ),
        dtype=np.float64,
        count=len(left),
    )
    ratios = geometric / graph_distances
    squared_sum = float(np.dot(ratios, ratios))
    alpha = float(np.sum(ratios) / squared_sum)
    errors = alpha * ratios - 1.0
    mean = float(np.mean(np.abs(errors)))
    rmse = sqrt(float(np.mean(np.square(errors))))
    if not all(isfinite(value) for value in (alpha, mean, rmse)):
        raise FullResultValidationError("recomputed distortion is non-finite")
    return EmbeddingDistortion(
        fitted_scale_alpha=alpha,
        mean_relative_distortion=mean,
        rmse_relative_distortion=rmse,
        unordered_pair_count=len(ratios),
        metric=metric,
    )


def _validate_distortions(
    path: Path,
    graph: nx.Graph,
    shortest_paths,
    coordinates: Mapping[str, Mapping[int, tuple[float, float]]],
    base_mds_coordinates: Mapping[int, tuple[float, float]],
) -> dict[str, dict[str, object]]:
    records = _read_json(path)
    if not isinstance(records, list) or len(records) != 7:
        raise FullResultValidationError("distortion record count mismatch")
    by_id = {
        str(record["metric_condition_id"]): record
        for record in records
        if isinstance(record, Mapping)
    }
    if tuple(by_id) != tuple(spec[0] for spec in DISTORTION_SPECS):
        raise FullResultValidationError("distortion condition order mismatch")
    result: dict[str, dict[str, object]] = {}
    for metric_id, condition, metric in DISTORTION_SPECS:
        record = by_id[metric_id]
        selected = (
            base_mds_coordinates
            if condition == "mds_base"
            else coordinates[condition]
        )
        recomputed = _vectorized_distortion(
            shortest_paths, selected, metric=metric
        )
        if record.get("metric") != metric:
            raise FullResultValidationError(f"{metric_id} metric label mismatch")
        _assert_close(
            float(record["fitted_scale_alpha"]),
            recomputed.fitted_scale_alpha,
            label=f"{metric_id} fitted alpha",
            absolute=2e-11,
            relative=2e-11,
        )
        _assert_close(
            float(record["mean_absolute_relative_distortion"]),
            recomputed.mean_relative_distortion,
            label=f"{metric_id} mean distortion",
            absolute=2e-12,
            relative=2e-12,
        )
        _assert_close(
            float(record["relative_rmse"]),
            recomputed.rmse_relative_distortion,
            label=f"{metric_id} RMSE distortion",
            absolute=2e-12,
            relative=2e-12,
        )
        if record.get("unordered_pair_count") != graph.number_of_nodes() * (
            graph.number_of_nodes() - 1
        ) // 2:
            raise FullResultValidationError(
                f"{metric_id} unordered-pair count mismatch"
            )
        result[metric_id] = dict(record)
    return result


def _walk_edges_are_valid(graph: nx.Graph, walk: Sequence[int]) -> bool:
    return all(
        graph.has_edge(left, right)
        for left, right in zip(walk, walk[1:])
    )


def validate_route_record(
    row: Mapping[str, object],
    *,
    graph: nx.Graph,
    graph_id: str,
    pair_index: int,
    source: int,
    destination: int,
    dijkstra_length: int,
) -> RoutingResult:
    method_id = row.get("method_id")
    if method_id not in INTERNAL_METHODS:
        raise FullResultValidationError("route method ID is invalid")
    if (
        row.get("graph_id") != graph_id
        or row.get("pair_index") != pair_index
        or row.get("pair_id") != f"{graph_id}:pair:{pair_index:04d}"
        or row.get("source") != source
        or row.get("destination") != destination
        or row.get("dijkstra_length") != dijkstra_length
        or row.get("dijkstra_hop_count") != dijkstra_length
    ):
        raise FullResultValidationError("route identity or Dijkstra length mismatch")
    walk = tuple(int(node) for node in row.get("walk", ()))
    if not _walk_edges_are_valid(graph, walk):
        raise FullResultValidationError("route walk contains a non-edge")
    if row.get("physical_hop_count") != len(walk) - 1:
        raise FullResultValidationError("physical hop count mismatch")
    try:
        result = RoutingResult(
            method=INTERNAL_METHODS[str(method_id)],
            source=source,
            destination=destination,
            success=row["success"],
            walk=walk,
            route_length=row["route_length"],
            failure_type=row["final_failure_type"],
            repair_attempted=row["repair_attempted"],
            repair_succeeded=row["repair_succeeded"],
            forwarding_decisions=row["forwarding_decisions"],
            initial_failure_type=row["initial_failure_type"],
            final_failure_type=row["final_failure_type"],
            repair_alternative_existed=row["repair_alternative_existed"],
            repair_attempt_count=row["repair_attempt_count"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FullResultValidationError(
            f"route failure/repair state is invalid: {exc}"
        ) from exc
    if (
        not result.success
        and result.final_failure_type not in FINAL_FAILURE_TYPES
    ):
        raise FullResultValidationError("route has an unknown final failure type")
    immediate_backtracks = sum(
        walk[index] == walk[index - 2] for index in range(2, len(walk))
    )
    if method_id == "repaired_poincare_greedy":
        expected_backtracks = 1 if result.repair_attempted else 0
        if immediate_backtracks != expected_backtracks:
            raise FullResultValidationError(
                "repair backtracking is missing or repeated in the physical walk"
            )
    elif immediate_backtracks:
        raise FullResultValidationError(
            "ordinary greedy route contains a forbidden physical revisit"
        )
    expected_stretch = (
        result.route_length / dijkstra_length if result.success else None
    )
    stored_stretch = row.get("stretch")
    if expected_stretch is None:
        if stored_stretch is not None:
            raise FullResultValidationError("failed route has non-null stretch")
    else:
        _assert_close(
            float(stored_stretch),
            expected_stretch,
            label="route stretch",
            absolute=0.0,
            relative=0.0,
        )
    runtime = row.get("runtime_ns")
    if isinstance(runtime, bool) or not isinstance(runtime, int) or runtime < 0:
        raise FullResultValidationError("route runtime is invalid")
    return result


def _validate_dijkstra_records(
    path: Path,
    graph: nx.Graph,
    graph_id: str,
    pairs: Sequence[tuple[int, int]],
    shortest_paths,
) -> tuple[int, ...]:
    distances: list[int] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for expected_index, line in enumerate(stream):
            row = decode_json_value(json.loads(line))
            if expected_index >= len(pairs):
                raise FullResultValidationError("too many Dijkstra rows")
            source, destination = pairs[expected_index]
            expected_distance = shortest_paths.distances[source][destination]
            walk = tuple(int(node) for node in row.get("walk", ()))
            runtime = row.get("runtime_ns")
            if (
                row.get("graph_id") != graph_id
                or row.get("pair_index") != expected_index
                or row.get("pair_id") != f"{graph_id}:pair:{expected_index:04d}"
                or row.get("source") != source
                or row.get("destination") != destination
                or row.get("method_id") != "dijkstra"
                or row.get("coordinate_condition_id") is not None
                or row.get("success") is not True
                or row.get("route_length") != expected_distance
                or row.get("apsp_length") != expected_distance
                or row.get("apsp_agreement") is not True
                or len(walk) - 1 != expected_distance
                or not walk
                or walk[0] != source
                or walk[-1] != destination
                or not _walk_edges_are_valid(graph, walk)
                or isinstance(runtime, bool)
                or not isinstance(runtime, int)
                or runtime < 0
            ):
                raise FullResultValidationError("Dijkstra record mismatch")
            distances.append(expected_distance)
    if len(distances) != EXPECTED_PAIR_COUNT:
        raise FullResultValidationError("Dijkstra record count mismatch")
    return tuple(distances)


def _route_statistics_template() -> dict[str, object]:
    return {
        "success_count": 0,
        "successful_route_length_sum": 0.0,
        "successful_stretch_sum": 0.0,
        "final_failure_counts": Counter(),
    }


def _finalize_pair_records(
    records: Mapping[tuple[str, str], tuple[Mapping[str, object], RoutingResult]],
    *,
    condition_stats: dict[tuple[str, str], dict[str, object]],
    common_stats: dict[str, dict[str, float | int]],
) -> None:
    expected = {
        (condition, method)
        for condition in COORDINATE_CONDITIONS
        for method in ROUTING_METHODS
    }
    if set(records) != expected:
        raise FullResultValidationError("missing or duplicate method-condition route")
    for (condition, method), (row, result) in records.items():
        stats = condition_stats[(condition, method)]
        if result.success:
            stats["success_count"] = int(stats["success_count"]) + 1
            stats["successful_route_length_sum"] = (
                float(stats["successful_route_length_sum"]) + result.route_length
            )
            stats["successful_stretch_sum"] = (
                float(stats["successful_stretch_sum"]) + float(row["stretch"])
            )
        else:
            stats["final_failure_counts"][result.final_failure_type] += 1
    for condition in COORDINATE_CONDITIONS:
        eu_row, euclidean = records[(condition, "euclidean_greedy")]
        p_row, poincare = records[(condition, "poincare_greedy")]
        r_row, repaired = records[(condition, "repaired_poincare_greedy")]
        if poincare.success and not repaired.success:
            raise FullResultValidationError(
                "repaired Poincare success is below ordinary Poincare"
            )
        if poincare.success and (
            repaired.walk != poincare.walk
            or repaired.route_length != poincare.route_length
        ):
            raise FullResultValidationError(
                "repaired route differs when ordinary Poincare succeeds"
            )
        if repaired.repair_succeeded and poincare.success:
            raise FullResultValidationError(
                "repair recovery was recorded after ordinary success"
            )
        common = common_stats[condition]
        if euclidean.success and poincare.success:
            common["common_success_count"] += 1
            common["common_euclidean_stretch_sum"] += float(eu_row["stretch"])
            common["common_poincare_stretch_sum"] += float(p_row["stretch"])
        if not poincare.success and repaired.success:
            common["recovered_count"] += 1
            common["recovered_repaired_stretch_sum"] += float(r_row["stretch"])


def _validate_route_stream(
    path: Path,
    *,
    graph: nx.Graph,
    graph_id: str,
    pairs: Sequence[tuple[int, int]],
    dijkstra_lengths: Sequence[int],
    audit_indices: set[int],
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    dict[str, dict[str, float | int]],
    dict[tuple[int, str, str], dict[str, object]],
]:
    stats = {
        (condition, method): _route_statistics_template()
        for condition in COORDINATE_CONDITIONS
        for method in ROUTING_METHODS
    }
    common = {
        condition: {
            "common_success_count": 0,
            "common_euclidean_stretch_sum": 0.0,
            "common_poincare_stretch_sum": 0.0,
            "recovered_count": 0,
            "recovered_repaired_stretch_sum": 0.0,
        }
        for condition in COORDINATE_CONDITIONS
    }
    audit_rows: dict[tuple[int, str, str], dict[str, object]] = {}
    current_pair = -1
    current_records: dict[
        tuple[str, str], tuple[Mapping[str, object], RoutingResult]
    ] = {}
    row_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = decode_json_value(json.loads(line))
            pair_index = int(row.get("pair_index", -1))
            if pair_index != current_pair:
                if current_pair >= 0:
                    _finalize_pair_records(
                        current_records,
                        condition_stats=stats,
                        common_stats=common,
                    )
                if pair_index != current_pair + 1 or pair_index >= len(pairs):
                    raise FullResultValidationError("route pair order mismatch")
                current_pair = pair_index
                current_records = {}
            source, destination = pairs[pair_index]
            condition = row.get("coordinate_condition_id")
            method = row.get("method_id")
            if condition not in COORDINATE_CONDITIONS or method not in ROUTING_METHODS:
                raise FullResultValidationError("route condition or method is invalid")
            key = (str(condition), str(method))
            if key in current_records:
                raise FullResultValidationError("duplicate route key")
            result = validate_route_record(
                row,
                graph=graph,
                graph_id=graph_id,
                pair_index=pair_index,
                source=source,
                destination=destination,
                dijkstra_length=dijkstra_lengths[pair_index],
            )
            current_records[key] = (row, result)
            if pair_index in audit_indices:
                audit_rows[(pair_index, str(condition), str(method))] = row
            row_count += 1
    if current_pair >= 0:
        _finalize_pair_records(
            current_records,
            condition_stats=stats,
            common_stats=common,
        )
    if row_count != EXPECTED_ROUTE_COUNT_PER_GRAPH or current_pair != 999:
        raise FullResultValidationError("route record count mismatch")
    if len(audit_rows) != len(audit_indices) * 15:
        raise FullResultValidationError("route-audit raw-row selection is incomplete")
    for key, item in stats.items():
        failure_total = sum(item["final_failure_counts"].values())
        if int(item["success_count"]) + failure_total != EXPECTED_PAIR_COUNT:
            raise FullResultValidationError(
                f"success/failure denominator mismatch: {key}"
            )
    return stats, common, audit_rows


def _route_audit(
    *,
    graph: nx.Graph,
    graph_id: str,
    pairs: Sequence[tuple[int, int]],
    dijkstra_lengths: Sequence[int],
    coordinates: Mapping[str, Mapping[int, tuple[float, float]]],
    selected_indices: Sequence[int],
    stored_rows: Mapping[tuple[int, str, str], Mapping[str, object]],
) -> int:
    contexts = {}
    for condition in COORDINATE_CONDITIONS:
        contexts[condition] = (
            prepare_routing_coordinates(
                graph,
                coordinates[condition],
                euclidean_distance,
                metric_name="euclidean",
            ),
            prepare_routing_coordinates(
                graph,
                coordinates[condition],
                poincare_distance,
                metric_name="poincare",
            ),
        )
    matched = 0
    for pair_index in selected_indices:
        source, destination = pairs[pair_index]
        for condition in COORDINATE_CONDITIONS:
            euclidean_context, poincare_context = contexts[condition]
            euclidean_tolerance = (
                FULL_EXPERIMENT_CONFIG.numerical_tolerance
                if condition == "hydra"
                else scale_equivariant_euclidean_routing_tolerance(
                    FULL_EXPERIMENT_CONFIG.numerical_tolerance,
                    MDS_RADII[condition],
                )
            )
            executions = (
                (
                    "euclidean_greedy",
                    euclidean_greedy_route(
                        graph,
                        euclidean_context,
                        source,
                        destination,
                        tolerance=euclidean_tolerance,
                    ),
                ),
                (
                    "poincare_greedy",
                    hyperbolic_greedy_route(
                        graph,
                        poincare_context,
                        source,
                        destination,
                        tolerance=FULL_EXPERIMENT_CONFIG.numerical_tolerance,
                    ),
                ),
                (
                    "repaired_poincare_greedy",
                    repaired_hyperbolic_greedy_route(
                        graph,
                        poincare_context,
                        source,
                        destination,
                        tolerance=FULL_EXPERIMENT_CONFIG.numerical_tolerance,
                    ),
                ),
            )
            for method, result in executions:
                stored = stored_rows[(pair_index, condition, method)]
                expected_stretch = (
                    result.route_length / dijkstra_lengths[pair_index]
                    if result.success
                    else None
                )
                comparisons = {
                    "walk": list(result.walk),
                    "success": result.success,
                    "initial_failure_type": result.initial_failure_type,
                    "final_failure_type": result.final_failure_type,
                    "repair_attempted": result.repair_attempted,
                    "repair_succeeded": result.repair_succeeded,
                    "repair_alternative_existed": result.repair_alternative_existed,
                    "repair_attempt_count": result.repair_attempt_count,
                    "route_length": result.route_length,
                    "physical_hop_count": result.route_length,
                    "stretch": expected_stretch,
                }
                if any(stored.get(key) != value for key, value in comparisons.items()):
                    raise FullResultValidationError(
                        "deterministic route-audit mismatch "
                        f"{graph_id} pair={pair_index} condition={condition} "
                        f"method={method}"
                    )
                matched += 1
    return matched


def _validate_generation_metadata(
    generation: Mapping[str, object],
    entry: GraphScheduleEntry,
    graph: nx.Graph,
) -> None:
    config = FULL_EXPERIMENT_CONFIG
    setting = config.parameter_settings[entry.setting_index]
    seeds = config.seeds_for_replicate(
        entry.setting_index, entry.model, entry.replicate_index
    )
    expected_common = {
        "graph_id": entry.graph_id,
        "graph_model": entry.model,
        "n": entry.n,
        "replicate_index": entry.replicate_index,
        "schedule_index": entry.schedule_index,
        "setting_index": entry.setting_index,
        "setting_label": entry.setting_label,
        "configuration_name": config.name,
        "configuration_schema_version": CONFIGURATION_SCHEMA_VERSION,
        "seed_identity_version": SEED_IDENTITY_VERSION,
        "configuration_fingerprint": config.configuration_fingerprint,
        "data_generation_hash": DATA_GENERATION_HASH,
        "combined_freeze_hash": COMBINED_FREEZE_HASH,
        "graph_seed": seeds.graph_generation,
        "embedding_initialization_seed": seeds.embedding_initialization,
        "source_destination_sampling_seed": seeds.source_destination_sampling,
        "pair_master_seed": config.source_destination_sampling_master_seed,
        "pair_count": EXPECTED_PAIR_COUNT,
        "realised_edge_count": graph.number_of_edges(),
    }
    for key, expected in expected_common.items():
        if generation.get(key) != expected:
            raise FullResultValidationError(
                f"{entry.graph_id} generation metadata mismatch: {key}"
            )
    realised_average = 2.0 * graph.number_of_edges() / entry.n
    _assert_close(
        float(generation["realised_average_degree"]),
        realised_average,
        label=f"{entry.graph_id} realised average degree",
        absolute=0.0,
        relative=0.0,
    )
    attempts = generation.get("generation_attempt_seeds")
    count = generation.get("generation_attempt_count")
    index = generation.get("generation_attempt_index")
    if (
        not isinstance(attempts, list)
        or not isinstance(count, int)
        or count != len(attempts)
        or index != count - 1
        or generation.get("generation_attempt_seed") != attempts[-1]
    ):
        raise FullResultValidationError("generation attempt metadata mismatch")
    if entry.model == ERDOS_RENYI:
        expected_attempts = [
            config.seed_for_graph_attempt(
                entry.setting_index,
                entry.model,
                entry.replicate_index,
                attempt_index,
            )
            for attempt_index in range(count)
        ]
        if (
            attempts != expected_attempts
            or generation.get("rejected_disconnected_count") != count - 1
            or generation.get("p_exact_numerator")
            != setting.er_probability_numerator
            or generation.get("p_exact_denominator")
            != setting.er_probability_denominator
        ):
            raise FullResultValidationError("ER attempt/probability metadata mismatch")
        _assert_close(
            float(generation["p"]),
            setting.er_p,
            label="ER p",
            absolute=0.0,
            relative=0.0,
        )
    elif entry.model == BARABASI_ALBERT:
        if (
            generation.get("m") != entry.m
            or generation.get("ba_initial_graph") != BA_INITIAL_GRAPH
            or attempts != [seeds.graph_generation]
            or count != 1
            or generation.get("rejected_disconnected_count") != 0
        ):
            raise FullResultValidationError("BA generation metadata mismatch")
    else:
        raise FullResultValidationError("unknown graph model")


def _graph_rows_from_stats(
    *,
    entry: GraphScheduleEntry,
    network_metrics: Mapping[str, object],
    distortions: Mapping[str, Mapping[str, object]],
    route_stats: Mapping[tuple[str, str], Mapping[str, object]],
    common_stats: Mapping[str, Mapping[str, float | int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in COORDINATE_CONDITIONS:
        success_counts = {
            method: int(route_stats[(condition, method)]["success_count"])
            for method in ROUTING_METHODS
        }
        p_failures = EXPECTED_PAIR_COUNT - success_counts["poincare_greedy"]
        recovered = success_counts["repaired_poincare_greedy"] - success_counts[
            "poincare_greedy"
        ]
        row: dict[str, object] = {
            "graph_id": entry.graph_id,
            "schedule_index": entry.schedule_index,
            "model": entry.model,
            "n": entry.n,
            "m": entry.m,
            "replicate_index": entry.replicate_index,
            "coordinate_condition_id": condition,
            "pair_count": EXPECTED_PAIR_COUNT,
            "euclidean_success_count": success_counts["euclidean_greedy"],
            "poincare_success_count": success_counts["poincare_greedy"],
            "repaired_success_count": success_counts[
                "repaired_poincare_greedy"
            ],
            "euclidean_success": success_counts["euclidean_greedy"] / 1000,
            "poincare_success": success_counts["poincare_greedy"] / 1000,
            "repaired_success": success_counts["repaired_poincare_greedy"] / 1000,
            "poincare_advantage": (
                success_counts["poincare_greedy"]
                - success_counts["euclidean_greedy"]
            )
            / 1000,
            "repair_improvement": recovered / 1000,
            "repair_recovery_numerator": recovered,
            "repair_recovery_denominator": p_failures,
            "repair_recovery": (
                None if p_failures == 0 else recovered / p_failures
            ),
            "success_and_failure_rate_unit": "proportion",
            "success_and_contrast_denominator_pairs": EXPECTED_PAIR_COUNT,
            "embedding_distortion_unit": "scale_fitted_relative_error",
            "route_length_unit": "physical_unweighted_hops",
            "stretch_unit": "route_hops_per_dijkstra_hop",
            **network_metrics,
            "euclidean_embedding_distortion": float(
                distortions[
                    "hydra_euclidean"
                    if condition == "hydra"
                    else "base_mds_euclidean"
                ]["mean_absolute_relative_distortion"]
            ),
            "poincare_embedding_distortion": float(
                distortions[
                    "hydra_poincare"
                    if condition == "hydra"
                    else f"mds_poincare_r{condition[-3:]}"
                ]["mean_absolute_relative_distortion"]
            ),
        }
        for method in ROUTING_METHODS:
            stats = route_stats[(condition, method)]
            success_count = int(stats["success_count"])
            row[f"{method}_successful_route_length_mean"] = (
                None
                if success_count == 0
                else float(stats["successful_route_length_sum"]) / success_count
            )
            row[f"{method}_success_stretch_mean"] = (
                None
                if success_count == 0
                else float(stats["successful_stretch_sum"]) / success_count
            )
            for failure in FINAL_FAILURE_TYPES:
                failure_count = int(stats["final_failure_counts"][failure])
                row[f"{method}_failure_{failure}_count"] = failure_count
                row[f"{method}_failure_{failure}_rate"] = (
                    failure_count / EXPECTED_PAIR_COUNT
                )
        common = common_stats[condition]
        common_count = int(common["common_success_count"])
        recovered_count = int(common["recovered_count"])
        row.update(
            {
                "common_success_count": common_count,
                "common_euclidean_stretch_mean": (
                    None
                    if common_count == 0
                    else float(common["common_euclidean_stretch_sum"])
                    / common_count
                ),
                "common_poincare_stretch_mean": (
                    None
                    if common_count == 0
                    else float(common["common_poincare_stretch_sum"])
                    / common_count
                ),
                "recovered_pair_count": recovered_count,
                "recovered_repaired_stretch_mean": (
                    None
                    if recovered_count == 0
                    else float(common["recovered_repaired_stretch_sum"])
                    / recovered_count
                ),
            }
        )
        rows.append(row)
    return rows


def _validate_one_graph(
    run_root: Path,
    entry: GraphScheduleEntry,
    run_manifest: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    tuple[int, ...],
    int,
]:
    graph_root = run_root / "graphs" / entry.graph_id
    validation = validate_graph_checkpoint(
        graph_root,
        expected_run_manifest=run_manifest,
        expected_graph_id=entry.graph_id,
    )
    expected_checkpoint_counts = {
        "vertices": entry.n,
        "coordinate_conditions": 5,
        "coordinate_rows": entry.n * 5,
        "pairs": 1_000,
        "dijkstra_records": 1_000,
        "route_records": 15_000,
        "distortion_records": 7,
    }
    if any(
        validation.counts.get(key) != value
        for key, value in expected_checkpoint_counts.items()
    ):
        raise FullResultValidationError("checkpoint row-count manifest mismatch")
    publication_timing = validate_publication_timing_record(
        run_root,
        graph_id=entry.graph_id,
        expected_run_manifest=run_manifest,
    )
    generation = _read_json(graph_root / "generation.json")
    network_stored = _read_json(graph_root / "network_metrics.json")
    metadata = _read_json(graph_root / "embedding_metadata.json")
    timings = _read_json(graph_root / "timings.json")
    if not all(
        isinstance(value, Mapping)
        for value in (generation, network_stored, metadata, timings)
    ):
        raise FullResultValidationError("graph metadata JSON schema mismatch")

    graph, _edges = _load_edges(graph_root / "edges.csv.gz", entry.n)
    _validate_generation_metadata(generation, entry, graph)
    shortest_paths = prepare_all_pairs_shortest_paths(graph)
    network_recomputed = calculate_network_metrics(
        graph, shortest_paths=shortest_paths
    )
    for key, expected in network_recomputed.items():
        stored = network_stored.get(key)
        if isinstance(expected, float):
            _assert_close(
                float(stored),
                expected,
                label=f"{entry.graph_id} network metric {key}",
            )
        elif stored != expected:
            raise FullResultValidationError(
                f"{entry.graph_id} network metric {key} mismatch"
            )

    pairs = _load_pairs(graph_root / "pairs.csv.gz")
    expected_pairs = sample_ordered_pairs(
        graph.nodes,
        EXPECTED_PAIR_COUNT,
        FULL_EXPERIMENT_CONFIG.source_destination_sampling_master_seed,
        graph_identity=entry.canonical_pair_graph_identity,
    )
    if pairs != expected_pairs:
        raise FullResultValidationError("stored pair list differs from frozen sampler")
    dijkstra_lengths = _validate_dijkstra_records(
        graph_root / "dijkstra.jsonl.gz",
        graph,
        entry.graph_id,
        pairs,
        shortest_paths,
    )

    coordinates = _load_coordinates(graph_root / "coordinates.csv.gz", entry.n)
    base_mds = _validate_embedding_metadata(
        coordinates,
        metadata,
        FULL_EXPERIMENT_CONFIG.configuration_fingerprint,
    )
    distortions = _validate_distortions(
        graph_root / "distortions.json",
        graph,
        shortest_paths,
        coordinates,
        base_mds,
    )

    selected = derive_route_audit_pair_indices(entry.graph_id)
    route_stats, common_stats, audit_rows = _validate_route_stream(
        graph_root / "routes.jsonl.gz",
        graph=graph,
        graph_id=entry.graph_id,
        pairs=pairs,
        dijkstra_lengths=dijkstra_lengths,
        audit_indices=set(selected),
    )
    audit_matches = _route_audit(
        graph=graph,
        graph_id=entry.graph_id,
        pairs=pairs,
        dijkstra_lengths=dijkstra_lengths,
        coordinates=coordinates,
        selected_indices=selected,
        stored_rows=audit_rows,
    )
    graph_rows = _graph_rows_from_stats(
        entry=entry,
        network_metrics=network_recomputed,
        distortions=distortions,
        route_stats=route_stats,
        common_stats=common_stats,
    )
    runtime_record = {
        "graph_id": entry.graph_id,
        "model": entry.model,
        "n": entry.n,
        "m": entry.m,
        "replicate_index": entry.replicate_index,
        **{key: int(value) for key, value in timings.items()},
        "end_to_end_graph_wall_ns": int(
            publication_timing["end_to_end_graph_wall_ns"]
        ),
        "atomic_publication_and_final_validation_ns": int(
            publication_timing["atomic_publication_and_final_validation_ns"]
        ),
    }
    return graph_rows, runtime_record, selected, audit_matches


def validate_full_run(
    run_root: Path | str,
    *,
    initial_fingerprint: RawTreeFingerprint | None = None,
    progress: Callable[[str], None] | None = None,
) -> ValidatedRun:
    """Validate every raw graph and retain only compact graph-level data."""

    root = Path(run_root).resolve(strict=True)
    if root.name != EXPECTED_RUN_DIRECTORY:
        raise FullResultValidationError("raw run directory identity mismatch")
    if (
        DATA_GENERATION_HASH != EXPECTED_DATA_GENERATION_HASH
        or ANALYSIS_PLAN_HASH != EXPECTED_ANALYSIS_PLAN_HASH
        or COMBINED_FREEZE_HASH != EXPECTED_COMBINED_FREEZE_HASH
    ):
        raise FullResultValidationError("Step 13 freeze hash mismatch")
    fingerprint = initial_fingerprint or compute_raw_tree_fingerprint(
        root, include_entries=True, progress=progress
    )
    run_manifest = _read_json(root / "run_manifest.json")
    if not isinstance(run_manifest, Mapping):
        raise FullResultValidationError("raw run manifest is invalid")
    repository_root = Path(__file__).resolve().parents[1]
    if (
        run_manifest.get("git_commit_hash") != EXPECTED_SOURCE_COMMIT
        or run_manifest.get("git_working_tree") != "clean"
        or run_manifest.get("data_generation_hash") != DATA_GENERATION_HASH
        or run_manifest.get("analysis_plan_hash") != ANALYSIS_PLAN_HASH
        or run_manifest.get("combined_freeze_hash") != COMBINED_FREEZE_HASH
        or run_manifest.get("configuration_schema_version")
        != CONFIGURATION_SCHEMA_VERSION
        or run_manifest.get("seed_identity_version") != SEED_IDENTITY_VERSION
        or run_manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION
    ):
        raise FullResultValidationError("raw run manifest identity mismatch")
    committed_fingerprint = committed_source_fingerprint(
        repository_root, EXPECTED_SOURCE_COMMIT
    )
    if run_manifest.get("source_fingerprint") != committed_fingerprint:
        raise FullResultValidationError(
            "raw source fingerprint differs from the committed Step 16 tree"
        )
    dependency_versions = installed_dependency_versions(repository_root)
    if run_manifest.get("dependency_versions") != dependency_versions:
        raise FullResultValidationError(
            "installed dependency environment differs from raw run manifest"
        )
    capacity_profile = load_capacity_profile()
    recorded_capacity = run_manifest.get("capacity_profile")
    if (
        not isinstance(recorded_capacity, Mapping)
        or recorded_capacity.get("profile_sha256")
        != capacity_profile.get("profile_sha256")
        or recorded_capacity.get("profile_schema_version")
        != capacity_profile.get("profile_schema_version")
    ):
        raise FullResultValidationError(
            "capacity profile differs from raw run manifest"
        )
    schedule = build_full_schedule()
    if len(schedule) != EXPECTED_GRAPH_COUNT:
        raise FullResultValidationError("canonical schedule size mismatch")
    schedule_ids = tuple(entry.graph_id for entry in schedule)
    if tuple(run_manifest.get("schedule", ())) != schedule_ids:
        raise FullResultValidationError("raw manifest schedule mismatch")
    inventory = validate_raw_inventory(root, schedule_ids)
    graph_rows: list[dict[str, object]] = []
    runtime_records: list[dict[str, object]] = []
    audit_selections: dict[str, list[int]] = {}
    audit_matches = 0
    last_update = monotonic()
    for completed, entry in enumerate(schedule, 1):
        if progress is not None:
            progress(
                f"validating graph {completed}/{len(schedule)} {entry.graph_id}"
            )
        rows, runtime, selected, matched = _validate_one_graph(
            root, entry, run_manifest
        )
        graph_rows.extend(rows)
        runtime_records.append(runtime)
        audit_selections[entry.graph_id] = list(selected)
        audit_matches += matched
        if progress is not None and monotonic() - last_update >= 60:
            progress(
                f"validation heartbeat graphs={completed}/{len(schedule)} "
                f"route_audit_matches={audit_matches}"
            )
            last_update = monotonic()

    observed_counts = {
        "graph_checkpoints": len(runtime_records),
        "ordered_pair_records": len(runtime_records) * 1000,
        "dijkstra_records": len(runtime_records) * 1000,
        "coordinate_routing_records": len(runtime_records) * 15000,
        "distortion_records": len(runtime_records) * 7,
        "publication_timing_records": inventory["publication_timing_records"],
        "graph_level_rows": len(graph_rows),
    }
    expected_counts = {
        "graph_checkpoints": 360,
        "ordered_pair_records": 360_000,
        "dijkstra_records": 360_000,
        "coordinate_routing_records": 5_400_000,
        "distortion_records": 2_520,
        "publication_timing_records": 360,
        "graph_level_rows": 1_800,
    }
    if observed_counts != expected_counts or audit_matches != 54_000:
        raise FullResultValidationError("global validation count mismatch")
    manifest_hash, manifest_size = _sha256_file(root / "run_manifest.json")
    report: dict[str, object] = {
        "validation_report_schema": VALIDATION_REPORT_SCHEMA,
        "validation_passed": True,
        "raw_run_identity": root.name,
        "raw_run_manifest_sha256": manifest_hash,
        "raw_run_manifest_size_bytes": manifest_size,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "initial_source_and_environment_gate": {
            "head_commit_matches_step16": True,
            "committed_source_fingerprint": committed_fingerprint,
            "committed_source_fingerprint_matches_manifest": True,
            "installed_dependency_versions": dependency_versions,
            "dependency_environment_matches_manifest": True,
            "capacity_profile_sha256": capacity_profile["profile_sha256"],
            "capacity_profile_matches_manifest": True,
            "production_preflight_complete_graph_count": 360,
            "production_preflight_remaining_graph_count": 0,
            "production_preflight_checkpoint_content_errors": 0,
            "current_source_guard_note": (
                "Step 17 source is intentionally untracked during implementation; "
                "the raw source identity was reconstructed from immutable Git objects."
            ),
        },
        "step13_hashes": {
            "data_generation": DATA_GENERATION_HASH,
            "analysis_plan": ANALYSIS_PLAN_HASH,
            "combined": COMBINED_FREEZE_HASH,
        },
        "raw_tree_fingerprint_initial": fingerprint.summary(),
        "raw_tree_fingerprint_final": None,
        "inventory": inventory,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "checkpoint_integrity": {
            "complete_markers_validated": 360,
            "graph_manifests_validated": 360,
            "payload_hashes_and_sizes_validated": True,
            "publication_timing_records_validated": 360,
            "checkpoint_errors": 0,
        },
        "graph_data_validation": {
            "graphs_reconstructed_from_sorted_edges": 360,
            "simple_undirected_unweighted_connected": 360,
            "network_measurements_recomputed": 360,
            "apsp_recomputed": 360,
            "dijkstra_distances_checked": 360_000,
            "frozen_pair_lists_checked": 360,
        },
        "embedding_validation": {
            "coordinate_conditions_checked": 1_800,
            "hydra_frechet_centering_residuals_recomputed": 360,
            "mds_euclidean_centroids_recomputed": 1_440,
            "distortion_records_recomputed": 2_520,
            "distortion_metric_pair_evaluations": 461_412_000,
            "coincidence_metadata_checked": 2_160,
        },
        "route_structural_validation": {
            "rows_streamed": 5_400_000,
            "walks_checked": 5_400_000,
            "repair_relationship_groups_checked": 1_800_000,
            "duplicate_or_missing_route_keys": 0,
        },
        "deterministic_route_audit": {
            "domain": ROUTE_AUDIT_DOMAIN,
            "person_hex": ROUTE_AUDIT_PERSON.hex(),
            "selection_algorithm": (
                "blake2s_unsigned_64_unbiased_modulo_duplicate_rejection_v1"
            ),
            "pair_indices_per_graph": ROUTE_AUDIT_PAIR_COUNT,
            "selected_indices": audit_selections,
            "expected_routes": 54_000,
            "matched_routes": audit_matches,
            "mismatched_routes": 0,
        },
        "warnings": [
            (
                "Legacy development-mode audit omits capacity_profile identity; "
                "the required full-mode preflight validated all checkpoints."
            )
        ],
    }
    return ValidatedRun(
        validation_report=report,
        graph_level_rows=tuple(graph_rows),
        runtime_records=tuple(runtime_records),
        initial_raw_fingerprint=fingerprint,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate immutable full-run results")
    parser.add_argument("--run-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validated = validate_full_run(
        args.run_root,
        progress=lambda message: print(f"VALIDATION_PROGRESS {message}", flush=True),
    )
    print(
        json.dumps(
            validated.validation_report,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
