"""Frozen Step 17 graph-level analysis and atomic derived-output publication."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import io
import json
from math import ceil, isfinite, sqrt
import os
from pathlib import Path
import platform
import shutil
import sys
from time import monotonic
from typing import Callable, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from experiment_config import (
    ANALYSIS_BOOTSTRAP_DOMAIN,
    ANALYSIS_BOOTSTRAP_REPLICATES,
    ANALYSIS_PLAN_HASH,
    BARABASI_ALBERT,
    COMBINED_FREEZE_HASH,
    DATA_GENERATION_HASH,
    ERDOS_RENYI,
    FULL_EXPERIMENT_CONFIG,
)
from experiment_protocol import (
    bootstrap_graph_indices,
    percentile_bootstrap_interval,
)
from validate_full_experiment import (
    COORDINATE_CONDITIONS,
    EXPECTED_RUN_DIRECTORY,
    EXPECTED_SOURCE_COMMIT,
    FINAL_FAILURE_TYPES,
    ROUTING_METHODS,
    FullResultValidationError,
    RawTreeFingerprint,
    ValidatedRun,
    _canonical_json_bytes,
    _sha256_file,
    compute_raw_tree_fingerprint,
    validate_full_run,
)


ANALYSIS_DIRECTORY = "analysis_8e002ef20f96_a121c33a20ea"
ANALYSIS_SCHEMA = "step17_frozen_analysis_v1"
MANIFEST_SCHEMA = "step17_analysis_manifest_v1"
BOOTSTRAP_MASTER_SEED = 3_000_003
N_VALUES = (100, 300, 1_000)
M_VALUES = (4, 8, 16)
MODELS = (ERDOS_RENYI, BARABASI_ALBERT)
PRIMARY_METRICS = (
    "euclidean_success",
    "poincare_success",
    "repaired_success",
    "poincare_advantage",
    "repair_improvement",
)
CONTRAST_METRICS = ("poincare_advantage", "repair_improvement")
NETWORK_PROPERTIES = (
    "average_degree",
    "maximum_degree",
    "population_degree_variance",
    "average_clustering_coefficient",
    "diameter",
    "average_shortest_path_length",
    "euclidean_embedding_distortion",
    "poincare_embedding_distortion",
)
OUTPUT_FILES = (
    "validation_report.json",
    "graph_level_metrics.csv.gz",
    "cell_estimates.csv",
    "success_contrasts.csv",
    "embedding_interactions.csv",
    "model_contrasts.csv",
    "failure_summaries.csv",
    "stretch_summaries.csv",
    "property_correlations.csv",
    "runtime_summaries.csv",
    "analysis_summary.json",
)


class StatisticalAnalysisError(RuntimeError):
    """Raised when validated inputs cannot satisfy the frozen analysis."""


def _require_finite(value: object, *, path: str = "value") -> None:
    if isinstance(value, float) and not isfinite(value):
        raise StatisticalAnalysisError(f"{path} contains NaN or infinity")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, path=f"{path}[{index}]")


def deterministic_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    """Serialize finite JSON deterministically with no trailing newline."""

    _require_finite(value)
    if pretty:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
    return _canonical_json_bytes(value)


def deterministic_csv_bytes(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> bytes:
    """Serialize stable-column CSV, using empty fields only for null/N/A."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        normalized: dict[str, object] = {}
        for field in fieldnames:
            value = row.get(field)
            _require_finite(value, path=field)
            normalized[field] = "" if value is None else value
        writer.writerow(normalized)
    return output.getvalue().encode("utf-8")


def deterministic_gzip_bytes(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        mtime=0,
    ) as stream:
        stream.write(payload)
    return output.getvalue()


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def analysis_content_fingerprint(repository_root: Path | str) -> str:
    """Hash analysis source/specification bytes, independent of a commit SHA."""

    root = Path(repository_root).resolve()
    relative_paths = (
        "code/analyze_full_experiment.py",
        "code/validate_full_experiment.py",
        "code/experiment_protocol.py",
        "code/experiment_config.py",
        "requirements.txt",
    )
    digest = sha256()
    for relative in relative_paths:
        payload = (root / relative).read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    digest.update(COMBINED_FREEZE_HASH.encode("ascii"))
    return digest.hexdigest()


def _rows_by_stratum(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int, int, str], tuple[Mapping[str, object], ...]]:
    grouped: dict[
        tuple[str, int, int, str], list[Mapping[str, object]]
    ] = {}
    for row in rows:
        key = (
            str(row["model"]),
            int(row["n"]),
            int(row["m"]),
            str(row["coordinate_condition_id"]),
        )
        grouped.setdefault(key, []).append(row)
    expected = {
        (model, n, m, condition)
        for model in MODELS
        for n in N_VALUES
        for m in M_VALUES
        for condition in COORDINATE_CONDITIONS
    }
    if set(grouped) != expected:
        raise StatisticalAnalysisError("graph-level stratum coverage mismatch")
    result = {}
    for key, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda row: int(row["replicate_index"])))
        if (
            len(ordered) != 20
            or tuple(int(row["replicate_index"]) for row in ordered)
            != tuple(range(20))
        ):
            raise StatisticalAnalysisError(f"invalid replicate coverage: {key}")
        result[key] = ordered
    return result


def build_bootstrap_draws(
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[tuple[str, int, int], np.ndarray]:
    """Build the exact frozen graph-index stream once for all comparisons."""

    draws: dict[tuple[str, int, int], np.ndarray] = {}
    last_update = monotonic()
    for model in MODELS:
        for n in N_VALUES:
            for m in M_VALUES:
                key = (model, n, m)
                matrix = np.empty(
                    (ANALYSIS_BOOTSTRAP_REPLICATES, 20), dtype=np.uint8
                )
                for replicate in range(ANALYSIS_BOOTSTRAP_REPLICATES):
                    matrix[replicate] = bootstrap_graph_indices(
                        bootstrap_replicate=replicate,
                        model=model,
                        n=n,
                        m=m,
                    )
                    if progress is not None and monotonic() - last_update >= 60:
                        progress(
                            "bootstrap stream heartbeat "
                            f"stratum={model}/{n}/{m} replicate={replicate}"
                        )
                        last_update = monotonic()
                draws[key] = matrix
                if progress is not None:
                    progress(f"bootstrap stream complete {model}/{n}/{m}")
    return draws


def _interval(values: np.ndarray) -> tuple[float, float]:
    interval = percentile_bootstrap_interval(values)
    return interval.lower, interval.upper


def _values(
    grouped: Mapping[
        tuple[str, int, int, str], Sequence[Mapping[str, object]]
    ],
    key: tuple[str, int, int, str],
    metric: str,
) -> np.ndarray:
    result = np.asarray(
        [float(row[metric]) for row in grouped[key]], dtype=np.float64
    )
    if result.shape != (20,) or not np.isfinite(result).all():
        raise StatisticalAnalysisError(f"non-finite graph values: {key}/{metric}")
    return result


def cell_estimates(
    grouped: Mapping[
        tuple[str, int, int, str], Sequence[Mapping[str, object]]
    ],
    draws: Mapping[tuple[str, int, int], np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for n in N_VALUES:
            for m in M_VALUES:
                sample = draws[(model, n, m)]
                for condition in COORDINATE_CONDITIONS:
                    key = (model, n, m, condition)
                    row: dict[str, object] = {
                        "model": model,
                        "n": n,
                        "m": m,
                        "coordinate_condition_id": condition,
                        "graph_count": 20,
                        "pairs_per_graph": 1_000,
                        "graph_weighting": "equal",
                    }
                    for metric in PRIMARY_METRICS:
                        values = _values(grouped, key, metric)
                        boot = values[sample].mean(axis=1)
                        lower, upper = _interval(boot)
                        scale = (
                            100.0 if metric in CONTRAST_METRICS else 1.0
                        )
                        prefix = metric
                        row[f"{prefix}_estimate"] = float(values.mean()) * scale
                        row[f"{prefix}_ci_lower"] = lower * scale
                        row[f"{prefix}_ci_upper"] = upper * scale
                        row[f"{prefix}_unit"] = (
                            "percentage_points"
                            if metric in CONTRAST_METRICS
                            else "proportion"
                        )
                    recovery_values = [
                        item["repair_recovery"] for item in grouped[key]
                    ]
                    defined_recovery = [
                        float(value)
                        for value in recovery_values
                        if value is not None
                    ]
                    zero_denominator_count = 20 - len(defined_recovery)
                    row["repair_recovery_defined_graph_count"] = len(
                        defined_recovery
                    )
                    row["repair_recovery_zero_denominator_graph_count"] = (
                        zero_denominator_count
                    )
                    row["repair_recovery_numerator_sum"] = sum(
                        int(item["repair_recovery_numerator"])
                        for item in grouped[key]
                    )
                    row["repair_recovery_denominator_sum"] = sum(
                        int(item["repair_recovery_denominator"])
                        for item in grouped[key]
                    )
                    row["repair_recovery_unit"] = "proportion"
                    if zero_denominator_count:
                        row["repair_recovery_estimate"] = None
                        row["repair_recovery_ci_lower"] = None
                        row["repair_recovery_ci_upper"] = None
                        row["repair_recovery_status"] = (
                            "N/A_one_or_more_graph_zero_denominator"
                        )
                    else:
                        recovery = np.asarray(
                            defined_recovery, dtype=np.float64
                        )
                        recovery_boot = recovery[sample].mean(axis=1)
                        lower, upper = _interval(recovery_boot)
                        row["repair_recovery_estimate"] = float(
                            recovery.mean()
                        )
                        row["repair_recovery_ci_lower"] = lower
                        row["repair_recovery_ci_upper"] = upper
                        row["repair_recovery_status"] = "defined_all_20_graphs"
                    rows.append(row)
    return rows


def success_contrasts(
    grouped: Mapping[
        tuple[str, int, int, str], Sequence[Mapping[str, object]]
    ],
    draws: Mapping[tuple[str, int, int], np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for condition in COORDINATE_CONDITIONS:
            marginal_boot: dict[str, list[np.ndarray]] = {
                metric: [] for metric in CONTRAST_METRICS
            }
            marginal_point: dict[str, list[float]] = {
                metric: [] for metric in CONTRAST_METRICS
            }
            for n in N_VALUES:
                for m in M_VALUES:
                    key = (model, n, m, condition)
                    for metric in CONTRAST_METRICS:
                        values = _values(grouped, key, metric)
                        boot = values[draws[(model, n, m)]].mean(axis=1)
                        lower, upper = _interval(boot)
                        rows.append(
                            {
                                "scope": "model_n_m_coordinate_cell",
                                "model": model,
                                "n": n,
                                "m": m,
                                "coordinate_condition_id": condition,
                                "contrast_id": (
                                    "poincare_minus_euclidean"
                                    if metric == "poincare_advantage"
                                    else "repaired_minus_poincare"
                                ),
                                "estimate": float(values.mean()) * 100.0,
                                "ci_lower": lower * 100.0,
                                "ci_upper": upper * 100.0,
                                "unit": "percentage_points",
                                "graph_count": 20,
                                "stratum_count": 1,
                                "pairs_per_graph": 1_000,
                                "weighting": "equal_graph",
                            }
                        )
                        marginal_boot[metric].append(boot)
                        marginal_point[metric].append(float(values.mean()))
            for metric in CONTRAST_METRICS:
                boot = np.mean(np.stack(marginal_boot[metric]), axis=0)
                lower, upper = _interval(boot)
                rows.append(
                    {
                        "scope": "model_coordinate_n_m_marginal",
                        "model": model,
                        "n": None,
                        "m": None,
                        "coordinate_condition_id": condition,
                        "contrast_id": (
                            "poincare_minus_euclidean"
                            if metric == "poincare_advantage"
                            else "repaired_minus_poincare"
                        ),
                        "estimate": float(np.mean(marginal_point[metric])) * 100.0,
                        "ci_lower": lower * 100.0,
                        "ci_upper": upper * 100.0,
                        "unit": "percentage_points",
                        "graph_count": 180,
                        "stratum_count": 9,
                        "pairs_per_graph": 1_000,
                        "weighting": "equal_n_m_strata_then_equal_graph",
                    }
                )
    return rows


def embedding_interactions(
    grouped: Mapping[
        tuple[str, int, int, str], Sequence[Mapping[str, object]]
    ],
    draws: Mapping[tuple[str, int, int], np.ndarray],
) -> list[dict[str, object]]:
    """Hydra-minus-MDS differences in the two prespecified success contrasts."""

    rows: list[dict[str, object]] = []
    for model in MODELS:
        for n in N_VALUES:
            for m in M_VALUES:
                sample = draws[(model, n, m)]
                for mds_condition in COORDINATE_CONDITIONS[1:]:
                    for metric in CONTRAST_METRICS:
                        hydra = _values(
                            grouped, (model, n, m, "hydra"), metric
                        )
                        mds = _values(
                            grouped, (model, n, m, mds_condition), metric
                        )
                        paired = hydra - mds
                        boot = paired[sample].mean(axis=1)
                        lower, upper = _interval(boot)
                        rows.append(
                            {
                                "model": model,
                                "n": n,
                                "m": m,
                                "hydra_condition_id": "hydra",
                                "mds_condition_id": mds_condition,
                                "interaction_id": (
                                    "hydra_minus_mds_poincare_advantage"
                                    if metric == "poincare_advantage"
                                    else "hydra_minus_mds_repair_improvement"
                                ),
                                "estimate": float(paired.mean()) * 100.0,
                                "ci_lower": lower * 100.0,
                                "ci_upper": upper * 100.0,
                                "unit": "percentage_points",
                                "graph_count": 20,
                                "pairs_per_graph": 1_000,
                                "pairing": "within_graph",
                            }
                        )
    return rows


def model_contrasts(
    grouped: Mapping[
        tuple[str, int, int, str], Sequence[Mapping[str, object]]
    ],
    draws: Mapping[tuple[str, int, int], np.ndarray],
) -> list[dict[str, object]]:
    """Descriptive BA-minus-ER contrasts with independent model streams."""

    rows: list[dict[str, object]] = []
    for n in N_VALUES:
        for m in M_VALUES:
            er_draw = draws[(ERDOS_RENYI, n, m)]
            ba_draw = draws[(BARABASI_ALBERT, n, m)]
            for condition in COORDINATE_CONDITIONS:
                for metric in CONTRAST_METRICS:
                    er = _values(
                        grouped, (ERDOS_RENYI, n, m, condition), metric
                    )
                    ba = _values(
                        grouped, (BARABASI_ALBERT, n, m, condition), metric
                    )
                    boot = (
                        ba[ba_draw].mean(axis=1)
                        - er[er_draw].mean(axis=1)
                    )
                    lower, upper = _interval(boot)
                    rows.append(
                        {
                            "n": n,
                            "m": m,
                            "coordinate_condition_id": condition,
                            "contrast_id": (
                                "ba_minus_er_poincare_advantage"
                                if metric == "poincare_advantage"
                                else "ba_minus_er_repair_improvement"
                            ),
                            "estimate": float(ba.mean() - er.mean()) * 100.0,
                            "ci_lower": lower * 100.0,
                            "ci_upper": upper * 100.0,
                            "unit": "percentage_points",
                            "ba_graph_count": 20,
                            "er_graph_count": 20,
                            "pairs_per_graph": 1_000,
                            "resampling": "independent_er_and_ba_strata",
                        }
                    )
    return rows


def failure_summaries(
    grouped: Mapping[
        tuple[str, int, int, str], Sequence[Mapping[str, object]]
    ],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for n in N_VALUES:
            for m in M_VALUES:
                for condition in COORDINATE_CONDITIONS:
                    values = grouped[(model, n, m, condition)]
                    for method in ROUTING_METHODS:
                        failure_counts = {
                            failure: sum(
                                int(
                                    row[
                                        f"{method}_failure_{failure}_count"
                                    ]
                                )
                                for row in values
                            )
                            for failure in FINAL_FAILURE_TYPES
                        }
                        total_failures = sum(failure_counts.values())
                        for failure in FINAL_FAILURE_TYPES:
                            count = failure_counts[failure]
                            rows.append(
                                {
                                    "model": model,
                                    "n": n,
                                    "m": m,
                                    "coordinate_condition_id": condition,
                                    "method_id": method,
                                    "failure_type": failure,
                                    "failure_count": count,
                                    "primary_denominator_all_pairs": 20_000,
                                    "failure_rate_all_pairs": count / 20_000,
                                    "conditional_failure_denominator": total_failures,
                                    "conditional_failure_composition": (
                                        None
                                        if total_failures == 0
                                        else count / total_failures
                                    ),
                                    "unit": "proportion",
                                    "graph_count": 20,
                                }
                            )
    return rows


def _mean_non_null(
    rows: Sequence[Mapping[str, object]], field: str
) -> tuple[float | None, int]:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return (
        (None if not values else float(np.mean(values))),
        len(values),
    )


def stretch_summaries(
    grouped: Mapping[
        tuple[str, int, int, str], Sequence[Mapping[str, object]]
    ],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for n in N_VALUES:
            for m in M_VALUES:
                for condition in COORDINATE_CONDITIONS:
                    values = grouped[(model, n, m, condition)]
                    definitions = [
                        (
                            "method_specific_euclidean_success",
                            "euclidean_greedy_success_stretch_mean",
                            "euclidean_success_count",
                        ),
                        (
                            "method_specific_poincare_success",
                            "poincare_greedy_success_stretch_mean",
                            "poincare_success_count",
                        ),
                        (
                            "method_specific_repaired_success",
                            "repaired_poincare_greedy_success_stretch_mean",
                            "repaired_success_count",
                        ),
                        (
                            "common_success_euclidean",
                            "common_euclidean_stretch_mean",
                            "common_success_count",
                        ),
                        (
                            "common_success_poincare",
                            "common_poincare_stretch_mean",
                            "common_success_count",
                        ),
                        (
                            "newly_recovered_repaired",
                            "recovered_repaired_stretch_mean",
                            "recovered_pair_count",
                        ),
                    ]
                    for summary_id, field, denominator_field in definitions:
                        graph_mean, contributing_graphs = _mean_non_null(
                            values, field
                        )
                        denominator = sum(
                            int(row[denominator_field]) for row in values
                        )
                        rows.append(
                            {
                                "model": model,
                                "n": n,
                                "m": m,
                                "coordinate_condition_id": condition,
                                "stretch_summary_id": summary_id,
                                "mean_of_graph_success_conditioned_means": graph_mean,
                                "contributing_graph_count": contributing_graphs,
                                "pair_denominator": denominator,
                                "total_graph_count": 20,
                                "unit": "route_hops_per_dijkstra_hop",
                                "weighting": (
                                    "equal_contributing_graphs_not_route_rows"
                                ),
                            }
                        )
    return rows


def _batch_rank_residuals(
    sampled_values: np.ndarray,
    *,
    stratum_size: int = 20,
) -> np.ndarray:
    """Average-rank rows and subtract each categorical-stratum rank mean."""

    values = np.asarray(sampled_values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise StatisticalAnalysisError("bootstrap association values are invalid")
    row_count, width = values.shape
    if width % stratum_size:
        raise StatisticalAnalysisError("association strata have invalid width")
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    positions = np.arange(width, dtype=np.int32)[None, :]
    starts = np.empty((row_count, width), dtype=bool)
    starts[:, 0] = True
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    start_index = np.maximum.accumulate(
        np.where(starts, positions, 0), axis=1
    )
    ends = np.empty((row_count, width), dtype=bool)
    ends[:, -1] = True
    ends[:, :-1] = sorted_values[:, :-1] != sorted_values[:, 1:]
    end_seed = np.where(ends, positions, width - 1)
    end_index = np.minimum.accumulate(end_seed[:, ::-1], axis=1)[:, ::-1]
    sorted_ranks = (start_index + end_index) / 2.0 + 1.0
    ranks = np.empty_like(sorted_ranks, dtype=np.float64)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    shaped = ranks.reshape(row_count, width // stratum_size, stratum_size)
    shaped -= shaped.mean(axis=2, keepdims=True)
    return shaped.reshape(row_count, width)


def _batch_residual_correlation(
    first_residual: np.ndarray, second_residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.sum(first_residual * second_residual, axis=1)
    denominator = np.sqrt(
        np.sum(first_residual * first_residual, axis=1)
        * np.sum(second_residual * second_residual, axis=1)
    )
    defined = denominator > 0.0
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=defined)
    if not np.isfinite(result[defined]).all():
        raise StatisticalAnalysisError(
            "bootstrap adjusted rank correlation is non-finite"
        )
    return result, defined


def property_correlations(
    grouped: Mapping[
        tuple[str, int, int, str], Sequence[Mapping[str, object]]
    ],
    draws: Mapping[tuple[str, int, int], np.ndarray],
    *,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Frozen partial-Spearman-style association analysis."""

    rows: list[dict[str, object]] = []
    for model in MODELS:
        sample_indices = np.concatenate(
            [
                draws[(model, n, m)].astype(np.int32) + stratum * 20
                for stratum, (n, m) in enumerate(
                    (pair for pair in ((n, m) for n in N_VALUES for m in M_VALUES))
                )
            ],
            axis=1,
        )
        for condition in COORDINATE_CONDITIONS:
            source_rows = tuple(
                row
                for n in N_VALUES
                for m in M_VALUES
                for row in grouped[(model, n, m, condition)]
            )
            outcome_residuals: dict[str, np.ndarray] = {}
            for outcome in CONTRAST_METRICS:
                original = np.asarray(
                    [float(row[outcome]) for row in source_rows],
                    dtype=np.float64,
                )
                outcome_residuals[outcome] = _batch_rank_residuals(
                    original[sample_indices]
                )
            for property_id in NETWORK_PROPERTIES:
                original_property = np.asarray(
                    [float(row[property_id]) for row in source_rows],
                    dtype=np.float64,
                )
                property_residual = _batch_rank_residuals(
                    original_property[sample_indices]
                )
                for outcome in CONTRAST_METRICS:
                    original_outcome = [
                        float(row[outcome]) for row in source_rows
                    ]
                    point_first = _batch_rank_residuals(
                        original_property[None, :]
                    )
                    point_second = _batch_rank_residuals(
                        np.asarray(original_outcome, dtype=np.float64)[None, :]
                    )
                    point_values, point_defined = _batch_residual_correlation(
                        point_first, point_second
                    )
                    boot, boot_defined = _batch_residual_correlation(
                        property_residual, outcome_residuals[outcome]
                    )
                    undefined_bootstrap = int(
                        ANALYSIS_BOOTSTRAP_REPLICATES
                        - int(np.count_nonzero(boot_defined))
                    )
                    if not bool(point_defined[0]):
                        point: float | None = None
                        lower: float | None = None
                        upper: float | None = None
                        status = "N/A_zero_residual_variance"
                    elif undefined_bootstrap:
                        point = float(point_values[0])
                        lower = None
                        upper = None
                        status = (
                            "estimate_defined_CI_N/A_bootstrap_zero_residual_variance"
                        )
                    else:
                        point = float(point_values[0])
                        lower, upper = _interval(boot)
                        status = "defined"
                    rows.append(
                        {
                            "model": model,
                            "coordinate_condition_id": condition,
                            "property_id": property_id,
                            "outcome_id": outcome,
                            "estimate": point,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "unit": "adjusted_rank_correlation",
                            "graph_count": 180,
                            "stratum_count": 9,
                            "bootstrap_replicates": 10_000,
                            "defined_bootstrap_replicates": int(
                                np.count_nonzero(boot_defined)
                            ),
                            "undefined_bootstrap_replicates": undefined_bootstrap,
                            "status": status,
                            "label": (
                                "exploratory_adjusted_partial_spearman_style_noncausal"
                            ),
                        }
                    )
            if progress is not None:
                progress(f"property associations complete {model}/{condition}")
    return rows


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise StatisticalAnalysisError("runtime summary has no values")
    rank = max(1, ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _integer_median(values: Sequence[int]) -> float:
    ordered = sorted(int(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def runtime_summaries(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Descriptive component and end-to-end runtime summaries only."""

    if len(records) != 360:
        raise StatisticalAnalysisError("runtime analysis requires 360 graphs")
    excluded = {"graph_id", "model", "n", "m", "replicate_index"}
    components = tuple(
        sorted(set(records[0]) - excluded)
    )
    if (
        "end_to_end_graph_wall_ns" not in components
        or "prepublication_wall_ns" not in components
        or "payload_serialization_ns" not in components
        or "atomic_publication_and_final_validation_ns" not in components
    ):
        raise StatisticalAnalysisError("truthful runtime fields are incomplete")
    scopes: list[tuple[str, str | None, int | None, Sequence[Mapping[str, object]]]]
    scopes = [("all_graphs", None, None, records)]
    scopes.extend(
        (
            "model_n",
            model,
            n,
            tuple(
                row
                for row in records
                if row["model"] == model and int(row["n"]) == n
            ),
        )
        for model in MODELS
        for n in N_VALUES
    )
    rows: list[dict[str, object]] = []
    for scope, model, n, selected in scopes:
        expected_count = 360 if scope == "all_graphs" else 60
        if len(selected) != expected_count:
            raise StatisticalAnalysisError("runtime scope graph count mismatch")
        for component in components:
            values = [int(row[component]) for row in selected]
            if any(value < 0 for value in values):
                raise StatisticalAnalysisError("negative runtime encountered")
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "n": n,
                    "component": component,
                    "graph_count": len(values),
                    "total_ns": sum(values),
                    "median_ns": _integer_median(values),
                    "q1_ns": _nearest_rank(values, 0.25),
                    "q3_ns": _nearest_rank(values, 0.75),
                    "unit": "nanoseconds",
                    "runtime_role": (
                        "end_to_end"
                        if component == "end_to_end_graph_wall_ns"
                        else "descriptive_component"
                    ),
                    "quartile_rule": "noninterpolated_nearest_rank",
                    "hypothesis_tests": "none",
                }
            )
    return rows


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    if not rows:
        raise StatisticalAnalysisError("cannot serialize an empty table")
    expected = tuple(rows[0])
    if any(tuple(row) != expected for row in rows):
        raise StatisticalAnalysisError("table column order is unstable")
    return expected


def _write_table(
    directory: Path,
    name: str,
    rows: Sequence[Mapping[str, object]],
    *,
    compressed: bool = False,
) -> None:
    payload = deterministic_csv_bytes(rows, _fieldnames(rows))
    if compressed:
        payload = deterministic_gzip_bytes(payload)
    _write_new_file(directory / name, payload)


def _manifest_file_records(directory: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name in OUTPUT_FILES:
        digest, size = _sha256_file(directory / name)
        records[name] = {"sha256": digest, "size_bytes": size}
    return records


def validate_derived_directory(
    directory: Path | str,
    *,
    expected_raw_fingerprint: RawTreeFingerprint | None = None,
) -> dict[str, object]:
    """Validate the published derived directory and its manifest."""

    root = Path(directory).resolve(strict=True)
    expected_names = set(OUTPUT_FILES) | {"analysis_manifest.json"}
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != expected_names or any(path.is_dir() for path in root.iterdir()):
        raise StatisticalAnalysisError("derived output inventory mismatch")
    manifest_bytes = (root / "analysis_manifest.json").read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StatisticalAnalysisError("analysis manifest is invalid") from exc
    _require_finite(manifest, path="analysis_manifest")
    if (
        manifest.get("manifest_schema") != MANIFEST_SCHEMA
        or manifest.get("analysis_schema") != ANALYSIS_SCHEMA
        or manifest.get("raw_run_identity") != EXPECTED_RUN_DIRECTORY
        or manifest.get("step13_hashes", {}).get("combined")
        != COMBINED_FREEZE_HASH
        or manifest.get("validation_passed") is not True
    ):
        raise StatisticalAnalysisError("analysis manifest identity mismatch")
    if manifest.get("files") != _manifest_file_records(root):
        raise StatisticalAnalysisError("derived file hash/size mismatch")
    if expected_raw_fingerprint is not None:
        if (
            manifest.get("raw_tree_fingerprint")
            != expected_raw_fingerprint.summary()
        ):
            raise StatisticalAnalysisError("manifest raw fingerprint mismatch")
    expected_rows = {
        "graph_level_metrics.csv.gz": 1_800,
        "cell_estimates.csv": 90,
        "success_contrasts.csv": 200,
        "embedding_interactions.csv": 144,
        "model_contrasts.csv": 90,
        "failure_summaries.csv": 1_620,
        "stretch_summaries.csv": 540,
        "property_correlations.csv": 160,
    }
    if {
        key: int(manifest["row_counts"][key]) for key in expected_rows
    } != expected_rows:
        raise StatisticalAnalysisError("derived row-count manifest mismatch")
    return manifest


def _analysis_summary(
    contrasts: Sequence[Mapping[str, object]],
    interactions: Sequence[Mapping[str, object]],
    model_rows: Sequence[Mapping[str, object]],
    correlations: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    marginal = [
        dict(row)
        for row in contrasts
        if row["scope"] == "model_coordinate_n_m_marginal"
    ]
    end_to_end = next(
        row
        for row in runtime_rows
        if row["scope"] == "all_graphs"
        and row["component"] == "end_to_end_graph_wall_ns"
    )
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "scientific_interpretation_status": (
            "descriptive_frozen_estimates_not_paper_claims"
        ),
        "units": {
            "success_contrasts": "percentage_points",
            "correlations": "adjusted_rank_correlation",
            "runtime": "nanoseconds",
        },
        "primary_marginal_success_contrasts": marginal,
        "embedding_interaction_row_count": len(interactions),
        "model_contrast_row_count": len(model_rows),
        "exploratory_property_association_row_count": len(correlations),
        "runtime": {
            "total_end_to_end_ns": end_to_end["total_ns"],
            "median_end_to_end_graph_ns": end_to_end["median_ns"],
            "definition": (
                "graph start through successful atomic publication and final "
                "published-checkpoint validation"
            ),
        },
        "bootstrap": {
            "domain": ANALYSIS_BOOTSTRAP_DOMAIN,
            "master_seed": BOOTSTRAP_MASTER_SEED,
            "replicates": ANALYSIS_BOOTSTRAP_REPLICATES,
            "confidence_level": 0.95,
            "interval_rule": (
                "noninterpolated_nearest_rank_order_statistics_v1"
            ),
            "p_values": "not_calculated",
            "holm_correction": "not_calculated",
        },
    }


def publish_analysis(
    *,
    raw_run_root: Path | str,
    output_root: Path | str,
    validated: ValidatedRun,
    repository_root: Path | str,
    progress: Callable[[str], None] | None = None,
    timestamp: datetime | None = None,
) -> tuple[Path, dict[str, object], str]:
    """Analyze validated data and atomically publish the complete directory."""

    if validated.validation_report.get("validation_passed") is not True:
        raise StatisticalAnalysisError("analysis requires validation_passed=true")
    raw_root = Path(raw_run_root).resolve(strict=True)
    output_parent = Path(output_root).resolve(strict=True)
    if raw_root.parent != output_parent:
        raise StatisticalAnalysisError("raw run must be a child of output root")
    target = output_parent / ANALYSIS_DIRECTORY
    if target.exists():
        raise FileExistsError(f"derived analysis already exists: {target}")
    temporary = output_parent / f".{ANALYSIS_DIRECTORY}.tmp-{uuid4().hex}"
    if temporary.exists():
        raise FileExistsError(f"temporary analysis path exists: {temporary}")
    grouped = _rows_by_stratum(validated.graph_level_rows)
    draws = build_bootstrap_draws(progress=progress)
    if progress is not None:
        progress("computing cell estimates and prespecified contrasts")
    cells = cell_estimates(grouped, draws)
    contrasts = success_contrasts(grouped, draws)
    interactions = embedding_interactions(grouped, draws)
    model_rows = model_contrasts(grouped, draws)
    failures = failure_summaries(grouped)
    stretches = stretch_summaries(grouped)
    correlations = property_correlations(grouped, draws, progress=progress)
    runtimes = runtime_summaries(validated.runtime_records)
    summary = _analysis_summary(
        contrasts, interactions, model_rows, correlations, runtimes
    )
    analysis_fingerprint = analysis_content_fingerprint(repository_root)
    validation_report = dict(validated.validation_report)
    final_raw_fingerprint = compute_raw_tree_fingerprint(
        raw_root, include_entries=False, progress=progress
    )
    if final_raw_fingerprint.summary() != validated.initial_raw_fingerprint.summary():
        raise FullResultValidationError(
            "raw tree changed between initial validation and analysis publication"
        )
    validation_report["raw_tree_fingerprint_final"] = (
        final_raw_fingerprint.summary()
    )
    validation_report["raw_tree_unchanged_before_publication"] = True

    table_rows: dict[str, Sequence[Mapping[str, object]]] = {
        "graph_level_metrics.csv.gz": validated.graph_level_rows,
        "cell_estimates.csv": cells,
        "success_contrasts.csv": contrasts,
        "embedding_interactions.csv": interactions,
        "model_contrasts.csv": model_rows,
        "failure_summaries.csv": failures,
        "stretch_summaries.csv": stretches,
        "property_correlations.csv": correlations,
        "runtime_summaries.csv": runtimes,
    }
    try:
        temporary.mkdir()
        _write_new_file(
            temporary / "validation_report.json",
            deterministic_json_bytes(validation_report, pretty=True),
        )
        for name, rows in table_rows.items():
            _write_table(
                temporary,
                name,
                rows,
                compressed=name.endswith(".gz"),
            )
        _write_new_file(
            temporary / "analysis_summary.json",
            deterministic_json_bytes(summary, pretty=True),
        )
        files = _manifest_file_records(temporary)
        created = (
            datetime.now(timezone.utc)
            if timestamp is None
            else timestamp.astimezone(timezone.utc)
        )
        run_manifest = json.loads(
            (raw_root / "run_manifest.json").read_text(encoding="utf-8")
        )
        manifest = {
            "manifest_schema": MANIFEST_SCHEMA,
            "analysis_schema": ANALYSIS_SCHEMA,
            "raw_run_identity": raw_root.name,
            "raw_run_manifest_sha256": validation_report[
                "raw_run_manifest_sha256"
            ],
            "raw_tree_fingerprint": final_raw_fingerprint.summary(),
            "step13_hashes": {
                "data_generation": DATA_GENERATION_HASH,
                "analysis_plan": ANALYSIS_PLAN_HASH,
                "combined": COMBINED_FREEZE_HASH,
            },
            "step16_source_commit": EXPECTED_SOURCE_COMMIT,
            "analysis_content_fingerprint": analysis_fingerprint,
            "dependency_versions": run_manifest["dependency_versions"],
            "python_version": sys.version,
            "operating_system": platform.platform(),
            "validation_passed": True,
            "bootstrap": {
                "master_seed": BOOTSTRAP_MASTER_SEED,
                "domain": ANALYSIS_BOOTSTRAP_DOMAIN,
                "replicates": ANALYSIS_BOOTSTRAP_REPLICATES,
                "resampling_unit": "whole_graph_within_model_n_m_stratum",
                "paired_draw_reuse": True,
                "er_ba_streams_independent": True,
                "interval": (
                    "two_sided_95_percent_noninterpolated_nearest_rank"
                ),
            },
            "output_schemas": {
                name: (
                    "stable_column_csv_gzip_mtime_zero"
                    if name.endswith(".csv.gz")
                    else (
                        "stable_column_csv"
                        if name.endswith(".csv")
                        else "finite_sorted_key_json"
                    )
                )
                for name in OUTPUT_FILES
            },
            "row_counts": {
                name: len(rows) for name, rows in table_rows.items()
            },
            "creation_timestamp_utc": created.isoformat(),
            "files": files,
        }
        _write_new_file(
            temporary / "analysis_manifest.json",
            deterministic_json_bytes(manifest, pretty=True),
        )
        os.replace(temporary, target)
        published_manifest = validate_derived_directory(
            target, expected_raw_fingerprint=final_raw_fingerprint
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    final_check = compute_raw_tree_fingerprint(
        raw_root, include_entries=False, progress=progress
    )
    if final_check.summary() != final_raw_fingerprint.summary():
        raise FullResultValidationError(
            "raw tree changed during atomic analysis publication"
        )
    manifest_hash = sha256(
        (target / "analysis_manifest.json").read_bytes()
    ).hexdigest()
    return target, published_manifest, manifest_hash


def execute_step17(
    raw_run_root: Path | str,
    *,
    output_root: Path | str,
    repository_root: Path | str,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, object], str]:
    """Run full independent validation before any derived output is created."""

    initial = compute_raw_tree_fingerprint(
        raw_run_root, include_entries=True, progress=progress
    )
    validated = validate_full_run(
        raw_run_root,
        initial_fingerprint=initial,
        progress=progress,
    )
    return publish_analysis(
        raw_run_root=raw_run_root,
        output_root=output_root,
        validated=validated,
        repository_root=repository_root,
        progress=progress,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and analyze the immutable full experiment"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    target, manifest, manifest_hash = execute_step17(
        args.run_root,
        output_root=args.output_root,
        repository_root=repository_root,
        progress=lambda message: print(
            f"STEP17_PROGRESS {message}", flush=True
        ),
    )
    print(
        json.dumps(
            {
                "analysis_directory": str(target),
                "analysis_manifest_sha256": manifest_hash,
                "row_counts": manifest["row_counts"],
                "validation_passed": manifest["validation_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
