"""Executable utilities for the frozen Step 13 experiment protocol.

The scientific constants live in :mod:`experiment_config`.  This module
implements the versioned samplers, graph-level estimands, clustered bootstrap,
and in-memory run manifest without introducing a second configuration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import blake2s, sha256
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
from math import ceil
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

from experiment_config import (
    ANALYSIS_BOOTSTRAP_DOMAIN,
    ANALYSIS_BOOTSTRAP_REPLICATES,
    BARABASI_ALBERT,
    COMBINED_FREEZE_HASH,
    CONFIGURATION_SCHEMA_VERSION,
    DATA_GENERATION_HASH,
    FEASIBILITY_PILOT_SEEDS,
    FULL_EXPERIMENT_CONFIG,
    GRAPH_MODELS,
    MAX_SEED,
    ORDERED_PAIR_SAMPLER_ALGORITHM,
    ORDERED_PAIR_SAMPLER_DOMAIN,
    SEED_IDENTITY_VERSION,
    ANALYSIS_PLAN_HASH,
)


UINT64_SPACE = 1 << 64
PAIR_SAMPLER_PERSON = b"GRPpair1"
BOOTSTRAP_PERSON = b"GRPboot1"


def _require_int(name: str, value: int, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _canonical_identity_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def ordered_pair_from_index(n: int, index: int) -> tuple[int, int]:
    """Map an index in ``[0, n*(n-1))`` to the frozen ordered pair."""

    n = _require_int("n", n, minimum=2)
    index = _require_int("index", index, minimum=0)
    pair_space = n * (n - 1)
    if index >= pair_space:
        raise ValueError(f"index must be smaller than {pair_space}")
    source = index // (n - 1)
    remainder = index % (n - 1)
    destination = remainder if remainder < source else remainder + 1
    return source, destination


def sample_ordered_pairs(
    nodes: Iterable[int],
    pair_count: int,
    pair_master_seed: int,
    *,
    graph_identity: str,
) -> tuple[tuple[int, int], ...]:
    """Sample frozen ordered pairs by BLAKE2s rejection, without replacement."""

    supplied_nodes = tuple(nodes)
    if any(
        isinstance(node, bool) or not isinstance(node, int)
        for node in supplied_nodes
    ) or set(supplied_nodes) != set(range(len(supplied_nodes))):
        raise ValueError("nodes must be integer IDs exactly 0 through n-1")
    if len(supplied_nodes) < 2:
        raise ValueError("at least two nodes are required")
    pair_count = _require_int("pair_count", pair_count, minimum=0)
    pair_master_seed = _require_int(
        "pair_master_seed", pair_master_seed, minimum=0
    )
    if pair_master_seed > MAX_SEED:
        raise ValueError(f"pair_master_seed must be at most {MAX_SEED}")
    if not isinstance(graph_identity, str) or not graph_identity.strip():
        raise ValueError("graph_identity must be a non-empty string")

    n = len(supplied_nodes)
    pair_space = n * (n - 1)
    if pair_count > pair_space:
        raise ValueError(
            f"pair_count exceeds the {pair_space} available ordered pairs"
        )
    unbiased_limit = UINT64_SPACE - (UINT64_SPACE % pair_space)
    accepted_indices: set[int] = set()
    accepted_pairs: list[tuple[int, int]] = []
    counter = 0
    while len(accepted_pairs) < pair_count:
        payload = {
            "algorithm": ORDERED_PAIR_SAMPLER_ALGORITHM,
            "counter": counter,
            "domain": ORDERED_PAIR_SAMPLER_DOMAIN,
            "graph_identity": graph_identity,
            "pair_master_seed": pair_master_seed,
            "seed_identity_version": SEED_IDENTITY_VERSION,
        }
        word = int.from_bytes(
            blake2s(
                _canonical_identity_bytes(payload),
                digest_size=8,
                person=PAIR_SAMPLER_PERSON,
            ).digest(),
            "big",
            signed=False,
        )
        counter += 1
        if word >= unbiased_limit:
            continue
        index = word % pair_space
        if index in accepted_indices:
            continue
        accepted_indices.add(index)
        accepted_pairs.append(ordered_pair_from_index(n, index))
    return tuple(accepted_pairs)


def graph_identity(
    *,
    configuration_name: str,
    setting_index: int,
    setting_label: str,
    model: str,
    replicate_index: int,
) -> str:
    """Return the canonical graph identity used by deterministic streams."""

    if model not in GRAPH_MODELS:
        raise ValueError(f"model must be one of {GRAPH_MODELS}")
    for name, value in (
        ("configuration_name", configuration_name),
        ("setting_label", setting_label),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    _require_int("setting_index", setting_index, minimum=0)
    _require_int("replicate_index", replicate_index, minimum=0)
    return json.dumps(
        {
            "configuration_name": configuration_name,
            "model": model,
            "replicate_index": replicate_index,
            "seed_identity_version": SEED_IDENTITY_VERSION,
            "setting_index": setting_index,
            "setting_label": setting_label,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def bootstrap_graph_indices(
    *,
    bootstrap_replicate: int,
    model: str,
    n: int,
    m: int,
    combined_freeze_hash: str = COMBINED_FREEZE_HASH,
    master_seed: int = 3_000_003,
    graph_count: int = 20,
) -> tuple[int, ...]:
    """Return one exact graph-cluster bootstrap resample for a stratum."""

    bootstrap_replicate = _require_int(
        "bootstrap_replicate", bootstrap_replicate, minimum=0
    )
    if bootstrap_replicate >= ANALYSIS_BOOTSTRAP_REPLICATES:
        raise ValueError(
            "bootstrap_replicate is outside the frozen 10,000-replicate range"
        )
    if model not in GRAPH_MODELS:
        raise ValueError(f"model must be one of {GRAPH_MODELS}")
    n = _require_int("n", n, minimum=2)
    m = _require_int("m", m, minimum=1)
    if m >= n:
        raise ValueError("m must be smaller than n")
    graph_count = _require_int("graph_count", graph_count, minimum=1)
    master_seed = _require_int("master_seed", master_seed, minimum=0)
    if master_seed > MAX_SEED:
        raise ValueError(f"master_seed must be at most {MAX_SEED}")
    if (
        not isinstance(combined_freeze_hash, str)
        or len(combined_freeze_hash) != 64
        or any(character not in "0123456789abcdef" for character in combined_freeze_hash)
    ):
        raise ValueError("combined_freeze_hash must be lowercase SHA-256 hex")

    unbiased_limit = UINT64_SPACE - (UINT64_SPACE % graph_count)
    indices: list[int] = []
    for draw_position in range(graph_count):
        rejection_counter = 0
        while True:
            payload = {
                "bootstrap_replicate": bootstrap_replicate,
                "combined_freeze_hash": combined_freeze_hash,
                "domain": ANALYSIS_BOOTSTRAP_DOMAIN,
                "draw_position": draw_position,
                "m": m,
                "master_seed": master_seed,
                "model": model,
                "n": n,
                "rejection_counter": rejection_counter,
            }
            word = int.from_bytes(
                blake2s(
                    _canonical_identity_bytes(payload),
                    digest_size=8,
                    person=BOOTSTRAP_PERSON,
                ).digest(),
                "big",
                signed=False,
            )
            if word < unbiased_limit:
                indices.append(word % graph_count)
                break
            rejection_counter += 1
    return tuple(indices)


@dataclass(frozen=True)
class RepairRecovery:
    value: float | None
    numerator: int
    denominator: int

    @property
    def display(self) -> str | float:
        return "N/A" if self.value is None else self.value


@dataclass(frozen=True)
class GraphLevelEstimands:
    euclidean_success: float
    poincare_success: float
    repaired_success: float
    poincare_advantage: float
    repair_improvement: float
    repair_recovery: RepairRecovery
    pair_count: int


@dataclass(frozen=True)
class BootstrapInterval:
    lower: float
    upper: float
    confidence_level: float
    bootstrap_replicates: int
    quantile_rule: str


def percentile_bootstrap_interval(
    bootstrap_estimates: Sequence[float],
) -> BootstrapInterval:
    """Return the frozen non-interpolated two-sided 95% percentile interval."""

    values = np.asarray(tuple(bootstrap_estimates), dtype=np.float64)
    if values.shape != (ANALYSIS_BOOTSTRAP_REPLICATES,):
        raise ValueError("the frozen interval requires exactly 10,000 estimates")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap estimates must be finite")
    ordered = np.sort(values, kind="stable")
    lower_rank = ceil(0.025 * ANALYSIS_BOOTSTRAP_REPLICATES)
    upper_rank = ceil(0.975 * ANALYSIS_BOOTSTRAP_REPLICATES)
    return BootstrapInterval(
        lower=float(ordered[lower_rank - 1]),
        upper=float(ordered[upper_rank - 1]),
        confidence_level=0.95,
        bootstrap_replicates=ANALYSIS_BOOTSTRAP_REPLICATES,
        quantile_rule="noninterpolated_nearest_rank_order_statistics_v1",
    )


def calculate_graph_level_estimands(
    euclidean_successes: Sequence[bool],
    poincare_successes: Sequence[bool],
    repaired_successes: Sequence[bool],
    *,
    expected_pair_count: int = 1_000,
) -> GraphLevelEstimands:
    """Calculate the frozen within-graph success and repair estimands."""

    expected_pair_count = _require_int(
        "expected_pair_count", expected_pair_count, minimum=1
    )
    sequences = tuple(
        tuple(values)
        for values in (
            euclidean_successes,
            poincare_successes,
            repaired_successes,
        )
    )
    if any(len(values) != expected_pair_count for values in sequences):
        raise ValueError("every method must contain exactly the expected pairs")
    if any(any(not isinstance(value, bool) for value in values) for values in sequences):
        raise ValueError("success sequences must contain booleans")
    euclidean, poincare, repaired = sequences
    if any(
        poincare_ok and not repaired_ok
        for poincare_ok, repaired_ok in zip(poincare, repaired, strict=True)
    ):
        raise ValueError(
            "repaired Poincare routing cannot fail when ordinary Poincare succeeds"
        )
    fractions = tuple(
        sum(values) / expected_pair_count
        for values in (euclidean, poincare, repaired)
    )
    failures = tuple(not value for value in poincare)
    recovery_denominator = sum(failures)
    recovery_numerator = sum(
        failed and repaired_ok
        for failed, repaired_ok in zip(failures, repaired, strict=True)
    )
    recovery = RepairRecovery(
        value=(
            None
            if recovery_denominator == 0
            else recovery_numerator / recovery_denominator
        ),
        numerator=recovery_numerator,
        denominator=recovery_denominator,
    )
    return GraphLevelEstimands(
        euclidean_success=fractions[0],
        poincare_success=fractions[1],
        repaired_success=fractions[2],
        poincare_advantage=fractions[1] - fractions[0],
        repair_improvement=fractions[2] - fractions[1],
        repair_recovery=recovery,
        pair_count=expected_pair_count,
    )


def equally_weighted_cell_mean(
    graph_values: Sequence[float],
    *,
    expected_graph_count: int = 20,
) -> float:
    """Mean graph-level values with exactly equal graph weights."""

    expected_graph_count = _require_int(
        "expected_graph_count", expected_graph_count, minimum=1
    )
    values = tuple(float(value) for value in graph_values)
    if len(values) != expected_graph_count:
        raise ValueError("cell estimate requires the expected graph count")
    if not all(isfinite(value) for value in values):
        raise ValueError("graph values must be finite")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def equally_weighted_n_m_marginal(
    stratum_estimates: Mapping[tuple[int, int], float],
) -> float:
    """Average the frozen nine n-m strata with equal weights."""

    expected = {
        (n, m)
        for n in (100, 300, 1_000)
        for m in (4, 8, 16)
    }
    if set(stratum_estimates) != expected:
        raise ValueError("marginal summaries require exactly the nine n-m strata")
    values = np.asarray(
        [float(stratum_estimates[key]) for key in sorted(expected)],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("stratum estimates must be finite")
    return float(np.mean(values))


def descriptive_unpaired_model_contrast(
    ba_graph_values: Sequence[float],
    er_graph_values: Sequence[float],
) -> float:
    """Return the frozen descriptive BA-minus-ER graph-level contrast."""

    return equally_weighted_cell_mean(
        ba_graph_values
    ) - equally_weighted_cell_mean(er_graph_values)


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("association variables must be finite one-dimensional data")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def partial_spearman_by_n_m_stratum(
    first: Sequence[float],
    second: Sequence[float],
    strata: Sequence[tuple[int, int]],
) -> float:
    """Correlate rank residuals after categorical n-m stratum adjustment."""

    if len(first) != len(second) or len(first) != len(strata):
        raise ValueError("variables and strata must have equal lengths")
    categories = tuple(sorted(set(strata)))
    if len(categories) < 2:
        raise ValueError("at least two n-m strata are required")
    design = np.asarray(
        [[1.0 if stratum == category else 0.0 for category in categories]
         for stratum in strata],
        dtype=np.float64,
    )
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_residual = first_rank - design @ np.linalg.lstsq(
        design, first_rank, rcond=None
    )[0]
    second_residual = second_rank - design @ np.linalg.lstsq(
        design, second_rank, rcond=None
    )[0]
    denominator = float(
        np.linalg.norm(first_residual) * np.linalg.norm(second_residual)
    )
    if denominator == 0.0:
        raise ValueError("rank residual variance is zero")
    return float(np.dot(first_residual, second_residual) / denominator)


def _git_state(repository_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return commit, dirty


def _source_fingerprint(repository_root: Path) -> str:
    digest = sha256()
    paths = sorted(
        [
            *repository_root.joinpath("code").glob("*.py"),
            repository_root / "requirements.txt",
        ],
        key=lambda path: path.relative_to(repository_root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(repository_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dependency_versions(repository_root: Path) -> dict[str, str]:
    dependencies: dict[str, str] = {}
    for line in (repository_root / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        package = requirement.split("==", 1)[0]
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"required dependency is not installed: {package}") from exc
    return dependencies


def build_run_manifest(
    graph_identity_value: str,
    *,
    repository_root: Path | str | None = None,
    require_final_scientific_source: bool = False,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build a complete in-memory provenance manifest without writing files."""

    if not isinstance(graph_identity_value, str) or not graph_identity_value:
        raise ValueError("graph_identity_value must be a non-empty string")
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    commit, dirty = _git_state(root)
    if require_final_scientific_source and dirty:
        raise RuntimeError(
            "a final scientific run requires a clean committed source state"
        )
    instant = datetime.now(timezone.utc) if timestamp is None else timestamp
    if instant.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return {
        "manifest_schema": "greedy_routing_run_manifest_v1",
        "configuration_schema_version": CONFIGURATION_SCHEMA_VERSION,
        "seed_identity_version": SEED_IDENTITY_VERSION,
        "data_generation_hash": DATA_GENERATION_HASH,
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "combined_freeze_hash": COMBINED_FREEZE_HASH,
        "git_commit_hash": commit,
        "git_working_tree": "dirty" if dirty else "clean",
        "source_fingerprint": _source_fingerprint(root),
        "python_version": sys.version,
        "dependency_versions": _dependency_versions(root),
        "operating_system": platform.platform(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "timestamp_utc": instant.astimezone(timezone.utc).isoformat(),
        "graph_identity": graph_identity_value,
        "timer": "time.perf_counter_ns",
        "excluded_feasibility_seeds": list(FEASIBILITY_PILOT_SEEDS),
        "final_scientific_source_required": require_final_scientific_source,
    }
