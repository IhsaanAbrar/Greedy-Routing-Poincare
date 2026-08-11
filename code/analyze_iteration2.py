"""Guarded graph-level analysis and reporting entry point for Iteration 2."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import numpy as np

from benchmark_iteration2_capacity import (
    canonical_bytes as capacity_canonical_bytes,
    dependency_fingerprint,
    performance_source_fingerprint,
    performance_source_manifest,
    verify_committed_source_manifest,
)
from iteration2_analysis import (
    aggregate_graph_metrics,
    aggregate_interactions,
    bootstrap_indices,
    descriptive_model_contrasts,
    graph_level_complete_system_comparisons,
    graph_level_native_interactions,
    percentile_interval,
    property_associations,
    simultaneous_radius_bands,
)
from iteration2_config import (
    ANALYSIS_PLAN_HASH,
    BOOTSTRAP_REPLICATES,
    COMBINED_PROTOCOL_HASH,
    DATA_GENERATION_HASH,
    ITERATION2_ANALYSIS_IDENTITY,
    ITERATION2_GRAPH_COUNT,
    ITERATION2_RAW_TOTAL_FILE_COUNT,
    ITERATION2_RUN_IDENTITY,
    MIN_DEFINED_BOOTSTRAP_FRACTION,
    OUTPUT_SCHEMA_HASH,
    PROPERTY_ASSOCIATION_REPLICATES,
    full_schedule,
    resolve_iteration2_output,
)
from iteration2_excluded import (
    EXCLUDED_SCIENTIFIC_STATUS,
    ExcludedAnalysisFixtureContract,
    validate_excluded_fixture_payload,
)
from iteration2_reporting import (
    WORKBOOK_SHEETS,
    build_reporting_bundle,
    validate_reporting_bundle,
)
from iteration2_runtime_guard import (
    ANALYSIS_READ_ONLY,
    ScientificOperationLedger,
    current_scientific_ledger,
    require_zero_scientific_operations,
    scientific_operation_context,
    validate_scientific_boundary_registry,
)
from run_iteration2 import (
    GRAPH_CHECKPOINT_FILENAMES,
    GRAPH_CHECKPOINT_FILE_COUNT,
    RAW_RUN_FILE_COUNT,
    RUN_COMPLETION_SCHEMA,
    validate_checkpoint_directory,
)
from validate_iteration2 import (
    validate_iteration2_graph_result,
    verify_iteration1_immutable,
)
from validate_full_experiment import compute_raw_tree_fingerprint


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_complete_raw_run_for_analysis(
    raw: Path,
    manifest: Mapping[str, object],
    *,
    expected_graph_ids: Sequence[str],
    expected_total_file_count: int,
    operation_ledger: ScientificOperationLedger,
) -> tuple[list[dict[str, object]], list[Path], dict[str, object]]:
    """Validate a completed raw tree without any scientific execution path."""

    before = compute_raw_tree_fingerprint(raw, include_entries=False)
    if not raw.is_dir() or raw.is_symlink():
        raise RuntimeError("Iteration 2 raw output must be a non-symlink directory")
    if any(path.is_symlink() for path in raw.rglob("*")):
        raise RuntimeError("Iteration 2 raw output contains a symbolic link")
    expected_ids = tuple(expected_graph_ids)
    if tuple(str(value) for value in manifest.get("schedule", ())) != expected_ids:
        raise RuntimeError("Iteration 2 raw manifest schedule mismatch")
    graph_root = raw / "graphs"
    if not graph_root.is_dir() or graph_root.is_symlink():
        raise RuntimeError("Iteration 2 raw graph directory is missing or unsafe")
    root_files = {path.name for path in raw.iterdir() if path.is_file()}
    if root_files != {"run_manifest.json", "run_complete.json"}:
        raise RuntimeError("Iteration 2 raw run-level file inventory is incorrect")
    root_directories = {path.name for path in raw.iterdir() if path.is_dir()}
    if root_directories != {"graphs"}:
        raise RuntimeError("Iteration 2 raw root directory inventory is incorrect")
    graph_directories = sorted(
        (path for path in graph_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if (
        len(graph_directories) != len(expected_ids)
        or {path.name for path in graph_directories} != set(expected_ids)
        or any(path.is_file() for path in graph_root.iterdir())
    ):
        raise RuntimeError("Iteration 2 analysis requires the exact graph schedule")

    results: list[dict[str, object]] = []
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    for graph_id in expected_ids:
        checkpoint = graph_root / graph_id
        validated = validate_checkpoint_directory(
            checkpoint,
            run_manifest=manifest,
        )
        value = validated.get("result")
        if not isinstance(value, dict):
            raise RuntimeError("validated checkpoint did not contain a graph result")
        results.append(value)
        checkpoint_hashes[graph_id] = {
            filename: _sha256(checkpoint / filename)
            for filename in sorted(GRAPH_CHECKPOINT_FILENAMES)
        }

    manifest_path = raw / "run_manifest.json"
    completion_path = raw / "run_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_completion = {
        "schema": RUN_COMPLETION_SCHEMA,
        "status": "complete",
        "run_identity": manifest["run_identity"],
        "data_generation_hash": manifest["data_generation_hash"],
        "analysis_plan_hash": manifest["analysis_plan_hash"],
        "output_schema_hash": manifest["output_schema_hash"],
        "protocol_hash": manifest["protocol_hash"],
        "run_manifest_sha256": _sha256(manifest_path),
        "graph_checkpoint_count": len(expected_ids),
        "files_per_graph_checkpoint": GRAPH_CHECKPOINT_FILE_COUNT,
        "raw_file_count": expected_total_file_count,
        "graph_checkpoint_file_sha256": checkpoint_hashes,
        "schedule": list(expected_ids),
        "completion_written_last": True,
    }
    if completion != expected_completion:
        raise RuntimeError("Iteration 2 raw completion marker is invalid")
    raw_files = sorted(
        (path for path in raw.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(raw).as_posix(),
    )
    if len(raw_files) != expected_total_file_count:
        raise RuntimeError("Iteration 2 raw output file volume is incorrect")
    after = compute_raw_tree_fingerprint(raw, include_entries=False)
    if before != after:
        raise RuntimeError("analysis validation modified the raw-run tree")
    ledger_snapshot = operation_ledger.snapshot()
    attempted = ledger_snapshot["attempted_operation_counts"]
    executed = ledger_snapshot["executed_operation_counts"]
    assert isinstance(attempted, Mapping)
    assert isinstance(executed, Mapping)
    evidence = {
        "validation_mode": "read_only_analysis_consumer",
        "regeneration_requested": (
            int(attempted["scientific_regeneration_audit"]) != 0
        ),
        "scientific_graphs_executed_during_analysis": (
            int(executed["graph_generation"])
            + int(executed["scheduled_graph_execution"])
            + int(executed["graph_workload_execution"])
            + int(executed["excluded_fixture_execution"])
        ),
        "dijkstra_executions_during_analysis": int(executed["dijkstra"]),
        "routing_executions_during_analysis": sum(
            int(executed[name])
            for name in (
                "euclidean_greedy_routing",
                "poincare_greedy_routing",
                "repaired_poincare_routing",
            )
        ),
        "raw_checkpoints_written_during_analysis": (
            int(executed["raw_checkpoint_publication"])
        ),
        "raw_tree_before": before.summary(),
        "raw_tree_after": after.summary(),
        "raw_tree_unchanged": before == after,
        "scientific_operation_ledger": ledger_snapshot,
    }
    return results, raw_files, evidence


def _load_complete_raw_run_in_context(
    repository_root: Path | str,
    *,
    excluded_fixture: ExcludedAnalysisFixtureContract | None = None,
) -> tuple[
    list[dict[str, object]],
    dict[str, str],
    dict[str, object],
]:
    """Read a completed raw run through the non-regenerating analysis API."""

    operation_ledger = current_scientific_ledger()
    if operation_ledger is None or operation_ledger.mode != ANALYSIS_READ_ONLY:
        raise RuntimeError("raw analysis loading requires analysis-read-only mode")
    root = Path(repository_root).resolve(strict=True)
    expected_run_identity = (
        ITERATION2_RUN_IDENTITY
        if excluded_fixture is None
        else excluded_fixture.raw_identity
    )
    raw = resolve_iteration2_output(root, expected_run_identity)
    manifest_path = raw / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Iteration 2 raw manifest does not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("data_generation_hash") != DATA_GENERATION_HASH
        or manifest.get("analysis_plan_hash") != ANALYSIS_PLAN_HASH
        or manifest.get("output_schema_hash") != OUTPUT_SCHEMA_HASH
        or manifest.get("protocol_hash") != COMBINED_PROTOCOL_HASH
        or manifest.get("run_identity") != expected_run_identity
    ):
        raise RuntimeError("Iteration 2 raw manifest identity mismatch")
    expected_graph_ids = (
        tuple(spec.graph_id for spec in full_schedule())
        if excluded_fixture is None
        else excluded_fixture.expected_graph_ids
    )
    expected_graph_count = len(expected_graph_ids)
    expected_total_file_count = (
        RAW_RUN_FILE_COUNT
        if excluded_fixture is None
        else 2 + expected_graph_count * GRAPH_CHECKPOINT_FILE_COUNT
    )
    if excluded_fixture is None:
        if (
            manifest.get("scientific_status")
            != "iteration2_prespecified_scientific_run"
            or manifest.get("production_compatible") is not True
            or "excluded_fixture_payload" in manifest
        ):
            raise RuntimeError("production analysis rejects excluded raw data")
    else:
        payload = manifest.get("excluded_fixture_payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("excluded raw fixture payload is missing")
        normalized_payload = validate_excluded_fixture_payload(payload)
        if (
            normalized_payload != dict(excluded_fixture.payload)
            or manifest.get("excluded_fixture_payload_sha256")
            != excluded_fixture.payload_hash
            or manifest.get("scientific_status") != EXCLUDED_SCIENTIFIC_STATUS
            or manifest.get("production_compatible") is not False
        ):
            raise RuntimeError("excluded analysis contract identity mismatch")
    source_manifest = manifest.get("source_manifest")
    source_commit = str(manifest.get("source_commit", ""))
    if not isinstance(source_manifest, Mapping) or len(source_commit) != 40:
        raise RuntimeError("raw generation source provenance is incomplete")
    if excluded_fixture is None:
        verify_committed_source_manifest(root, source_commit, source_manifest)
    source_fingerprint = sha256(
        capacity_canonical_bytes(source_manifest)
    ).hexdigest()
    if source_fingerprint != manifest.get("source_fingerprint"):
        raise RuntimeError("raw generation source fingerprint mismatch")
    dependency_identity = manifest.get("dependency_fingerprint")
    capacity_identity = manifest.get("capacity_profile_identity")
    if (
        not isinstance(dependency_identity, Mapping)
        or not isinstance(dependency_identity.get("sha256"), str)
        or not isinstance(capacity_identity, Mapping)
        or capacity_identity.get("physical_sha256")
        != manifest.get("capacity_profile_sha256")
    ):
        raise RuntimeError("raw dependency/capacity provenance is incomplete")
    results, raw_files, analysis_validation = (
        _validate_complete_raw_run_for_analysis(
            raw,
            manifest,
            expected_graph_ids=expected_graph_ids,
            expected_total_file_count=expected_total_file_count,
            operation_ledger=operation_ledger,
        )
    )
    if (
        len(results) != expected_graph_count
        or analysis_validation.get("regeneration_requested") is not False
        or analysis_validation.get("raw_tree_unchanged") is not True
    ):
        raise RuntimeError("Iteration 2 raw run did not pass complete validation")
    graph_ids = {
        str(result["graph_identity"]["graph_id"]) for result in results
    }
    schedule = set(str(value) for value in manifest.get("schedule", ()))
    if (
        graph_ids != schedule
        or len(graph_ids) != expected_graph_count
    ):
        raise RuntimeError("raw graph schedule is incomplete or duplicated")
    hashes = {
        path.relative_to(raw).as_posix(): _sha256(path)
        for path in raw_files
    }
    tree = analysis_validation["raw_tree_after"]
    provenance = {
        "raw_generation": {
            "source_commit": source_commit,
            "source_fingerprint": source_fingerprint,
            "source_manifest": dict(source_manifest),
            "dependency_fingerprint": dict(dependency_identity),
            "capacity_profile_identity": dict(capacity_identity),
            "data_generation_hash": manifest["data_generation_hash"],
            "analysis_plan_hash": manifest["analysis_plan_hash"],
            "protocol_hash": manifest["protocol_hash"],
            "output_schema_hash": manifest["output_schema_hash"],
            "run_identity": manifest["run_identity"],
            "scientific_status": manifest["scientific_status"],
            "production_compatible": manifest["production_compatible"],
        },
        "raw_manifest_sha256": hashes["run_manifest.json"],
        "raw_completion_sha256": hashes["run_complete.json"],
        "raw_tree_fingerprint": tree,
        "analysis_validation": analysis_validation,
    }
    return results, hashes, provenance


def load_complete_raw_run(
    repository_root: Path | str,
    *,
    excluded_fixture: ExcludedAnalysisFixtureContract | None = None,
) -> tuple[
    list[dict[str, object]],
    dict[str, str],
    dict[str, object],
]:
    """Load raw results under an isolated measured analysis-read-only guard."""

    active = current_scientific_ledger()
    if active is not None:
        return _load_complete_raw_run_in_context(
            repository_root,
            excluded_fixture=excluded_fixture,
        )
    validate_scientific_boundary_registry()
    with scientific_operation_context(ANALYSIS_READ_ONLY):
        return _load_complete_raw_run_in_context(
            repository_root,
            excluded_fixture=excluded_fixture,
        )


def _identity(result: Mapping[str, object]) -> dict[str, object]:
    identity = result["graph_identity"]
    return {
        "graph_id": identity["graph_id"],
        "model": identity["model"],
        "n": identity["n"],
        "m": identity["m"],
        "replicate_index": identity["replicate_index"],
    }


def _flatten_failure_and_stretch(
    graph_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    stretches: list[dict[str, object]] = []
    for row in graph_rows:
        identity = {
            key: row[key]
            for key in ("graph_id", "model", "n", "m", "replicate_index")
        }
        condition = row["coordinate_condition_id"]
        for method, categories in row["failure_summaries"].items():
            for failure, summary in categories.items():
                failures.append(
                    {
                        **identity,
                        "coordinate_condition_id": condition,
                        "method_id": method,
                        "failure_type": failure,
                        **summary,
                        "independent_unit": "graph",
                    }
                )
        for method, summary in row["stretch_summaries"].items():
            if not isinstance(summary, Mapping):
                continue
            stretches.append(
                {
                    **identity,
                    "coordinate_condition_id": condition,
                    "stretch_scope": method,
                    **summary,
                    "independent_unit": "graph",
                    "conditioning": "successful_delivery",
                }
            )
    return failures, stretches


def _bootstrap_cell_summaries(
    rows: Sequence[Mapping[str, object]],
    *,
    identity_fields: Sequence[str],
    value_fields: Sequence[str],
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in identity_fields)].append(row)
    output: list[dict[str, object]] = []
    for key, selected_rows in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        selected = sorted(
            selected_rows,
            key=lambda row: int(row["replicate_index"]),
        )
        identity = dict(zip(identity_fields, key, strict=True))
        model = str(selected[0]["model"])
        n = int(selected[0]["n"])
        m = int(selected[0]["m"])
        selected_pair_count = sum(
            int(row.get("pair_count", 0) or 0) for row in selected
        )
        base_draw = np.asarray(
            [
                bootstrap_indices(
                    model=model,
                    n=n,
                    m=m,
                    replicate=replicate,
                    graph_count=len(selected),
                )
                for replicate in range(bootstrap_replicates)
            ],
            dtype=np.int64,
        )
        minimum_defined = max(
            2,
            int(np.ceil(
                bootstrap_replicates * MIN_DEFINED_BOOTSTRAP_FRACTION
            )),
        )
        for field in value_fields:
            raw = [row.get(field) for row in selected]
            available = [value for value in raw if value is not None]
            if not available:
                output.append(
                    {
                        **identity,
                        "metric": field,
                        "status": "all_graphs_na",
                        "applicability": "not_applicable",
                        "na_reason": "all_graph_values_undefined",
                        "estimate": None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "total_graph_count": len(selected),
                        "graph_count": len(selected),
                        "pair_count": selected_pair_count,
                        "numerator": None,
                        "denominator": 0,
                        "contributing_graph_count": 0,
                        "na_graph_count": len(selected),
                        "bootstrap_unit": "whole_graph",
                        "bootstrap_replicates": bootstrap_replicates,
                        "defined_bootstrap_replicates": 0,
                        "minimum_defined_bootstrap_replicates": (
                            minimum_defined
                        ),
                    }
                )
                continue
            boot_values = []
            for indices in base_draw:
                sampled = [
                    float(raw[int(index)])
                    for index in indices
                    if raw[int(index)] is not None
                ]
                if sampled:
                    boot_values.append(float(np.mean(sampled)))
            interval_defined = len(boot_values) >= minimum_defined
            if interval_defined:
                lower, upper = percentile_interval(boot_values)
                status = "defined"
                reason = None
            else:
                lower, upper = None, None
                status = "estimate_defined_interval_unavailable"
                reason = "insufficient_defined_bootstrap_replicates"
            values = np.asarray(available, dtype=np.float64)
            output.append(
                {
                    **identity,
                    "metric": field,
                    "status": status,
                    "applicability": "applicable",
                    "na_reason": reason,
                    "estimate": float(values.mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "total_graph_count": len(selected),
                    "graph_count": len(selected),
                    "pair_count": selected_pair_count,
                    "numerator": None,
                    "denominator": len(values),
                    "contributing_graph_count": len(values),
                    "na_graph_count": len(selected) - len(values),
                    "bootstrap_unit": "whole_contributing_graph",
                    "bootstrap_replicates": bootstrap_replicates,
                    "defined_bootstrap_replicates": len(boot_values),
                    "minimum_defined_bootstrap_replicates": minimum_defined,
                }
            )
    return output


def _embedding_and_graph_tables(
    results: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    graph_metrics: list[dict[str, object]] = []
    embeddings: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    for result in results:
        identity = _identity(result)
        graph_diagnostics = result["graph_and_pair_diagnostics"]
        graph_metrics.append(
            {
                **identity,
                **{
                    key: value
                    for key, value in graph_diagnostics.items()
                    if key not in ("network_metrics", "pair_sampling")
                },
                **graph_diagnostics["network_metrics"],
                "pair_sampling": graph_diagnostics["pair_sampling"],
                "er_ba_comparison_is_causal": False,
            }
        )
        for row in result["embedding_metadata"]["embedding_diagnostics"]:
            embeddings.append({**identity, **row})
        runtimes.append(
            {
                **identity,
                **result["timing"],
                "runtime_used_to_change_scientific_output": False,
            }
        )
    return graph_metrics, embeddings, runtimes


def _distortion_table(
    embeddings: Sequence[Mapping[str, object]],
    graph_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    success = {
        (str(row["graph_id"]), str(row["coordinate_condition_id"])): row
        for row in graph_rows
    }
    output: list[dict[str, object]] = []
    for row in embeddings:
        for geometry in ("euclidean", "poincare"):
            diagnostic = row.get(f"{geometry}_distortion")
            route = success[(str(row["graph_id"]), str(row["condition_id"]))]
            output.append(
                {
                    "graph_id": row["graph_id"],
                    "model": row["model"],
                    "n": row["n"],
                    "m": row["m"],
                    "replicate_index": row["replicate_index"],
                    "coordinate_condition_id": row["condition_id"],
                    "geometry": geometry,
                    "applicability": (
                        "applicable" if diagnostic is not None else "not_applicable"
                    ),
                    "diagnostic": diagnostic,
                    "fitted_scale": (
                        None if diagnostic is None else diagnostic["fitted_scale"]
                    ),
                    "scale_fitted_mean_relative_error": (
                        None
                        if diagnostic is None
                        else diagnostic["scale_fitted_mean_relative_error"]
                    ),
                    "scale_fitted_rmse_relative_error": (
                        None
                        if diagnostic is None
                        else diagnostic["scale_fitted_rmse_relative_error"]
                    ),
                    "graph_distance_rank_correlation": (
                        None
                        if diagnostic is None
                        else diagnostic["graph_distance_rank_correlation"]
                    ),
                    "normalized_distance_stress": (
                        None
                        if diagnostic is None
                        else diagnostic["normalized_distance_stress"]
                    ),
                    "euclidean_success": route["euclidean_success"],
                    "poincare_success": route["poincare_success"],
                    "cross_geometry_comparability": (
                        "not_directly_commensurate"
                    ),
                    "association_with_routing": "exploratory_noncausal",
                }
            )
    return output


def _graph_output_tables(
    graph_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Flatten nested graph estimands without losing conditioning metadata."""

    output = {
        "Graph-Level Estimates": [],
        "Repair Denominators": [],
        "Failure Composition": [],
        "Distance Bands": [],
        "Method Stretch": [],
        "Common-Success Stretch": [],
        "Recovered Stretch": [],
        "Routing Resources": [],
    }
    failure_order = {
        name: index
        for index, name in enumerate(
            (
                "local_minimum",
                "repair_unavailable_at_source",
                "no_alternative_after_backtracking",
                "post_repair_local_minimum",
                "post_repair_attempted_revisit",
            )
        )
    }
    for row in graph_rows:
        identity = {
            key: row.get(key)
            for key in (
                "graph_id",
                "model",
                "n",
                "m",
                "replicate_index",
                "coordinate_condition_id",
                "embedding_family",
                "condition_kind",
                "matched_radius",
            )
        }
        pair_count = int(row["pair_count"])
        poincare_applicable = row.get("poincare_success") is not None
        output["Graph-Level Estimates"].append(
            {
                **identity,
                "independent_unit": "graph",
                "numerator": None,
                "denominator": pair_count,
                "graph_count": 1,
                "pair_count": pair_count,
                "euclidean_success_count": row.get("euclidean_success_count"),
                "euclidean_success": row.get("euclidean_success"),
                "poincare_success_count": row.get("poincare_success_count"),
                "poincare_success": row.get("poincare_success"),
                "repaired_poincare_success_count": row.get(
                    "repaired_poincare_success_count"
                ),
                "repaired_poincare_success": row.get(
                    "repaired_poincare_success"
                ),
                "poincare_minus_euclidean": row.get(
                    "poincare_minus_euclidean"
                ),
                "repaired_minus_unrepaired_poincare": row.get(
                    "repaired_minus_unrepaired_poincare"
                ),
                "common_success_pair_count": row.get(
                    "common_success_pair_count"
                ),
                "common_success_poincare_minus_euclidean_stretch": row.get(
                    "common_success_poincare_minus_euclidean_stretch"
                ),
                "recovered_pair_count": row.get("recovered_pair_count"),
                "recovered_forwarding_stretch": row.get(
                    "recovered_route_stretch"
                ),
                "physical_recovered_stretch": row.get(
                    "physical_recovered_route_stretch"
                ),
                "poincare_status": (
                    "defined" if poincare_applicable else "not_applicable"
                ),
                "poincare_applicability": (
                    "applicable" if poincare_applicable else "not_applicable"
                ),
                "poincare_na_reason": (
                    None
                    if poincare_applicable
                    else "native_mds_has_euclidean_method_only"
                ),
                "repaired_poincare_status": (
                    "defined" if poincare_applicable else "not_applicable"
                ),
                "repaired_poincare_applicability": (
                    "applicable" if poincare_applicable else "not_applicable"
                ),
                "repaired_poincare_na_reason": (
                    None
                    if poincare_applicable
                    else "native_mds_has_euclidean_method_only"
                ),
                "status": (
                    "defined"
                    if poincare_applicable
                    else "defined_with_method_not_applicable"
                ),
                "applicability": "applicable",
                "na_reason": None,
                "unit": "mixed_estimands_see_named_fields",
            }
        )
        for estimand, summary in row["repair_estimands"].items():
            output["Repair Denominators"].append(
                {
                    **identity,
                    "estimand": estimand,
                    "estimate": summary.get("estimate"),
                    "numerator": summary.get("numerator"),
                    "denominator": summary.get("denominator"),
                    "graph_count": summary.get("graph_count", 1),
                    "pair_count": summary.get("pair_count", 0),
                    "status": summary.get("status"),
                    "applicability": summary.get("applicability"),
                    "na_reason": summary.get("na_reason"),
                    "unit": "proportion",
                    "independent_unit": "graph",
                }
            )
        for method, categories in row["failure_summaries"].items():
            for failure, summary in categories.items():
                for stage in ("initial", "final"):
                    prefix = "initial_" if stage == "initial" else ""
                    applicability = summary.get(f"{prefix}applicability")
                    if applicability is None:
                        applicability = summary.get(f"{prefix}status")
                    count = summary.get(f"{prefix}count")
                    rate = summary.get(f"{prefix}rate_all_pairs")
                    status = summary.get(f"{prefix}status")
                    na_reason = summary.get(f"{prefix}na_reason")
                    conditional = summary.get(
                        f"{prefix}conditional_failure_composition"
                    )
                    conditional_denominator = summary.get(
                        f"{prefix}conditional_failure_denominator"
                    )
                    conditional_status = summary.get(
                        f"{prefix}conditional_failure_status"
                    )
                    conditional_na_reason = summary.get(
                        f"{prefix}conditional_failure_na_reason"
                    )
                    for (
                        estimand,
                        estimate,
                        denominator,
                        estimand_status,
                        estimand_na_reason,
                    ) in (
                        (
                            "failure_rate_all_sampled_pairs",
                            rate,
                            pair_count if count is not None else None,
                            status,
                            na_reason,
                        ),
                        (
                            "composition_among_failures_at_stage",
                            conditional,
                            conditional_denominator,
                            conditional_status,
                            conditional_na_reason,
                        ),
                    ):
                        output["Failure Composition"].append(
                            {
                                **identity,
                                "method_id": method,
                                "failure_stage": stage,
                                "failure_type": failure,
                                "category_index": failure_order[failure],
                                "estimand": estimand,
                                "estimate": estimate,
                                "numerator": count,
                                "denominator": denominator,
                                "graph_count": 1,
                                "pair_count": pair_count,
                                "status": estimand_status,
                                "applicability": applicability,
                                "na_reason": estimand_na_reason,
                                "unit": "proportion",
                                "invariant_error_category": False,
                                "independent_unit": "graph",
                            }
                        )
        for method, bands in row["distance_band_summaries"].items():
            for band, summary in bands.items():
                output["Distance Bands"].append(
                    {
                        **identity,
                        "method_id": method,
                        "distance_band": band,
                        "estimate": summary.get("success_rate"),
                        "successful_stretch_mean": summary.get(
                            "successful_stretch_mean"
                        ),
                        "numerator": summary.get("numerator"),
                        "denominator": summary.get("denominator"),
                        "graph_count": summary.get("graph_count", 1),
                        "pair_count": summary.get("pair_count", 0),
                        "status": summary.get("status"),
                        "applicability": summary.get("applicability"),
                        "na_reason": summary.get("na_reason"),
                        "unit": "proportion",
                        "independent_unit": "graph",
                    }
                )
        stretch = row["stretch_summaries"]
        for method in (
            "euclidean_greedy",
            "poincare_greedy",
            "repaired_poincare_greedy",
        ):
            if method not in stretch:
                continue
            summary = stretch[method]
            output["Method Stretch"].append(
                {
                    **identity,
                    "method_id": method,
                    "estimate": summary.get("success_conditioned_mean"),
                    "numerator": summary.get("numerator"),
                    "denominator": summary.get("denominator"),
                    "graph_count": summary.get("graph_count", 1),
                    "pair_count": summary.get("pair_count", pair_count),
                    "status": summary.get("status"),
                    "applicability": summary.get("applicability"),
                    "na_reason": summary.get("na_reason"),
                    "conditioning": "method_success",
                    "unit": "hop_stretch_ratio",
                    "independent_unit": "graph",
                }
            )
        common = stretch.get("common_success")
        if isinstance(common, Mapping):
            output["Common-Success Stretch"].append(
                {
                    **identity,
                    "estimate": common.get("paired_difference_mean"),
                    "euclidean_mean": common.get("euclidean_mean"),
                    "poincare_mean": common.get("poincare_mean"),
                    "numerator": common.get("numerator"),
                    "denominator": common.get("denominator"),
                    "graph_count": common.get("graph_count", 1),
                    "pair_count": common.get("pair_count", 0),
                    "status": common.get("status"),
                    "applicability": common.get("applicability"),
                    "na_reason": common.get("na_reason"),
                    "conditioning": "both_ordinary_methods_succeeded",
                    "unit": "hop_stretch_ratio",
                    "independent_unit": "graph",
                }
            )
        recovered = stretch.get("newly_recovered")
        if isinstance(recovered, Mapping):
            denominator = recovered.get("denominator")
            for metric, estimate in (
                (
                    "recovered_forwarding_stretch_excluding_backtrack",
                    recovered.get("repaired_mean"),
                ),
                (
                    "physical_repaired_stretch_including_backtrack",
                    recovered.get("physical_repaired_mean"),
                ),
            ):
                output["Recovered Stretch"].append(
                    {
                        **identity,
                        "metric": metric,
                        "estimate": estimate,
                        "numerator": (
                            None
                            if estimate is None or denominator is None
                            else float(estimate) * int(denominator)
                        ),
                        "denominator": denominator,
                        "graph_count": recovered.get("graph_count", 1),
                        "pair_count": recovered.get("pair_count", 0),
                        "status": recovered.get("status"),
                        "applicability": recovered.get("applicability"),
                        "na_reason": recovered.get("na_reason"),
                        "conditioning": "newly_recovered_after_ordinary_poincare_failure",
                        "unit": "hop_stretch_ratio",
                        "independent_unit": "graph",
                    }
                )
        for method, summary in row["resource_summaries"].items():
            for resource, unit in (
                ("physical_hops", "hop_count"),
                ("forwarding_decisions", "decision_count"),
                ("logical_distance_evaluations", "logical_distance_requests"),
                ("peak_history_vertices", "vertex_count"),
            ):
                output["Routing Resources"].append(
                    {
                        **identity,
                        "method_id": method,
                        "resource": resource,
                        "estimate": summary.get(f"{resource}_mean"),
                        "numerator": summary.get(f"{resource}_sum"),
                        "denominator": summary.get("pair_count"),
                        "graph_count": summary.get("graph_count", 1),
                        "pair_count": summary.get("pair_count", pair_count),
                        "maximum": summary.get(f"{resource}_maximum"),
                        "status": summary.get("status"),
                        "applicability": summary.get("applicability"),
                        "na_reason": summary.get("na_reason"),
                        "unit": unit,
                        "independent_unit": "graph",
                    }
                )
    return output


def _evidence_rows(
    interactions: Sequence[Mapping[str, object]],
    marginal_estimates: Sequence[Mapping[str, object]],
    resources: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Assemble prespecified evidence without making deployment claims."""

    lookup = {
        (
            row.get("model"),
            row.get("coordinate_condition_id"),
            row.get("metric"),
        ): row
        for row in marginal_estimates
        if row.get("scope") == "model_condition_n_m_marginal"
    }
    resource_means: dict[tuple[object, object, object, object], list[float]] = (
        defaultdict(list)
    )
    for row in resources:
        if row.get("estimate") is not None:
            resource_means[
                (
                    row.get("model"),
                    row.get("coordinate_condition_id"),
                    row.get("method_id"),
                    row.get("resource"),
                )
            ].append(float(row["estimate"]))
    rows: list[dict[str, object]] = []
    selected = [
        row
        for row in interactions
        if row.get("ci_type") == "simultaneous_95_percent_familywise"
        and row.get("stratum_count") == 9
    ]
    by_model: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        by_model[str(row["model"])].append(row)
    for model, model_rows in sorted(by_model.items()):
        signs = {
            1 if float(row["estimate"]) > 0 else -1 if float(row["estimate"]) < 0 else 0
            for row in model_rows
        }
        consistent = len(signs) == 1
        for row in sorted(model_rows, key=lambda item: float(item["matched_radius"])):
            radius = float(row["matched_radius"])
            label = f"r{int(round(radius * 100)):03d}"
            stretch = {}
            repair = {}
            overhead = {}
            for family in ("hydra", "mds"):
                condition = f"{family}_scaled_{label}"
                stretch_row = lookup.get(
                    (
                        model,
                        condition,
                        "common_success_poincare_minus_euclidean_stretch",
                    )
                )
                repair_row = lookup.get(
                    (model, condition, "repaired_minus_unrepaired_poincare")
                )
                stretch[family] = None if stretch_row is None else stretch_row.get("estimate")
                repair[family] = None if repair_row is None else repair_row.get("estimate")
                ordinary_key = (model, condition, "poincare_greedy", "physical_hops")
                repaired_key = (
                    model,
                    condition,
                    "repaired_poincare_greedy",
                    "physical_hops",
                )
                ordinary_values = resource_means.get(ordinary_key, ())
                repaired_values = resource_means.get(repaired_key, ())
                overhead[family] = (
                    None
                    if not ordinary_values or not repaired_values
                    else float(np.mean(repaired_values) - np.mean(ordinary_values))
                )
            rows.append(
                {
                    "model": model,
                    "matched_radius": radius,
                    "primary_effect_size_percentage_points": row.get("estimate"),
                    "simultaneous_ci_lower_percentage_points": row.get("ci_lower"),
                    "simultaneous_ci_upper_percentage_points": row.get("ci_upper"),
                    "consistent_interaction_direction_across_four_radii": consistent,
                    "common_success_stretch_difference_by_embedding": stretch,
                    "repair_recovery_improvement_by_embedding": repair,
                    "mean_physical_hop_overhead_by_embedding": overhead,
                    "synthetic_graph_scope_only": True,
                    "deployment_recommendation_generated": False,
                    "causal_interpretation": False,
                    "hyperbolic_specific_repair_superiority_claim": False,
                    "status": "defined",
                    "applicability": "applicable",
                    "na_reason": None,
                    "graph_count": row.get("graph_count"),
                    "pair_count": None,
                    "numerator": None,
                    "denominator": None,
                }
            )
    return rows


def create_analysis_tables(
    results: Sequence[Mapping[str, object]],
    *,
    analysis_validation_evidence: Mapping[str, object],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    require_complete_design: bool = True,
) -> dict[str, list[dict[str, object]]]:
    """Build the reporting tables from validated graph-level records."""

    if not results:
        raise ValueError("analysis requires at least one validated graph result")
    required_validation_evidence = {
        "validation_mode",
        "regeneration_requested",
        "scientific_graphs_executed_during_analysis",
        "dijkstra_executions_during_analysis",
        "routing_executions_during_analysis",
        "raw_checkpoints_written_during_analysis",
        "raw_tree_before",
        "raw_tree_after",
        "raw_tree_unchanged",
        "scientific_operation_ledger",
    }
    if not required_validation_evidence <= set(analysis_validation_evidence):
        raise ValueError("analysis validation execution evidence is incomplete")
    scientific_counts = tuple(
        int(analysis_validation_evidence[key])
        for key in (
            "scientific_graphs_executed_during_analysis",
            "dijkstra_executions_during_analysis",
            "routing_executions_during_analysis",
        )
    )
    raw_checkpoint_writes = int(
        analysis_validation_evidence["raw_checkpoints_written_during_analysis"]
    )
    ledger_snapshot = analysis_validation_evidence["scientific_operation_ledger"]
    if not isinstance(ledger_snapshot, Mapping):
        raise ValueError("analysis scientific-operation ledger is invalid")
    require_zero_scientific_operations(
        ledger_snapshot,
        context="Iteration 2 analysis publication",
    )
    scientific_execution_performed = any(count != 0 for count in scientific_counts)
    if (
        analysis_validation_evidence["validation_mode"]
        != "read_only_analysis_consumer"
        or analysis_validation_evidence["regeneration_requested"] is not False
        or any(count < 0 for count in scientific_counts)
        or raw_checkpoint_writes != 0
        or analysis_validation_evidence["raw_tree_unchanged"] is not True
        or analysis_validation_evidence["raw_tree_before"]
        != analysis_validation_evidence["raw_tree_after"]
        or scientific_execution_performed
    ):
        raise RuntimeError(
            "analysis publication requires read-only validation with zero "
            "scientific execution and an unchanged raw tree"
        )
    expected_ids = {spec.graph_id for spec in full_schedule()}
    observed_ids = {
        str(result["graph_identity"]["graph_id"]) for result in results
    }
    complete_design = (
        len(results) == ITERATION2_GRAPH_COUNT
        and len(observed_ids) == ITERATION2_GRAPH_COUNT
        and observed_ids == expected_ids
    )
    if require_complete_design and not complete_design:
        raise ValueError("scientific analysis requires the exact 360-graph design")
    graph_rows = [
        row for result in results for row in result["graph_level_rows"]
    ]
    matched_graph = [
        row
        for result in results
        for row in result["graph_level_interactions"]
    ]
    complete_system_graph = [
        row
        for result in results
        for row in graph_level_complete_system_comparisons(
            result["graph_level_rows"]
        )
    ]
    native_graph = [
        row
        for result in results
        for row in graph_level_native_interactions(result["graph_level_rows"])
    ]
    conditional = _graph_output_tables(graph_rows)
    graph_metrics, embeddings, runtimes = _embedding_and_graph_tables(results)
    distortion_graph = _distortion_table(embeddings, graph_rows)

    estimates: list[dict[str, object]] = []
    pointwise: list[dict[str, object]] = []
    simultaneous: list[dict[str, object]] = []
    complete_system: list[dict[str, object]] = []
    native_estimates: list[dict[str, object]] = []
    distortion_estimates: list[dict[str, object]] = []
    associations: list[dict[str, object]] = []
    model_contrasts: list[dict[str, object]] = []
    if complete_design:
        estimates = aggregate_graph_metrics(
            graph_rows,
            metrics=(
                "euclidean_success",
                "poincare_success",
                "repaired_poincare_success",
                "poincare_minus_euclidean",
                "repaired_minus_unrepaired_poincare",
                "conditional_repair_recovery",
                "common_success_poincare_minus_euclidean_stretch",
                "recovered_route_stretch",
                "physical_recovered_route_stretch",
            ),
            bootstrap_replicates=bootstrap_replicates,
        )
        pointwise = aggregate_interactions(
            matched_graph,
            bootstrap_replicates=bootstrap_replicates,
        )
        simultaneous = simultaneous_radius_bands(
            matched_graph,
            bootstrap_replicates=bootstrap_replicates,
        )
        complete_for_aggregation = [
            {
                **row,
                "interaction": row["complete_system_difference"],
            }
            for row in complete_system_graph
        ]
        complete_system = aggregate_interactions(
            complete_for_aggregation,
            bootstrap_replicates=bootstrap_replicates,
        )
        for row in complete_system:
            row["comparison_type"] = "matched_complete_system_comparison"
            row["interaction_definition"] = (
                "Hydra-Poincare_minus_MDS-Euclidean_at_same_radius"
            )
            row["metric_only_effect"] = False
            row["interpretation"] = (
                "coordinate_construction_and_metric_are_jointly_contrasted"
            )
        native_estimates = _bootstrap_cell_summaries(
            native_graph,
            identity_fields=("model", "n", "m", "comparison_type"),
            value_fields=("native_reference_difference",),
            bootstrap_replicates=bootstrap_replicates,
        )
        for row in native_estimates:
            row.update(
                {
                    "scope": "model_n_m_native_reference_cell",
                    "unit": "proportion",
                    "independent_unit": "graph",
                    "metric_only_effect": False,
                    "interpretation": (
                        "secondary_native_complete_system_reference"
                    ),
                }
            )
        distortion_estimates = _bootstrap_cell_summaries(
            distortion_graph,
            identity_fields=(
                "model",
                "n",
                "m",
                "coordinate_condition_id",
                "geometry",
            ),
            value_fields=(
                "scale_fitted_mean_relative_error",
                "scale_fitted_rmse_relative_error",
                "graph_distance_rank_correlation",
                "normalized_distance_stress",
                "euclidean_success",
                "poincare_success",
            ),
            bootstrap_replicates=bootstrap_replicates,
        )
        for row in distortion_estimates:
            metric = str(row["metric"])
            row["unit"] = (
                "correlation"
                if "correlation" in metric
                else "proportion"
                if metric.endswith("success")
                else "distance_diagnostic"
            )
            row["pair_count"] = None
            row["numerator"] = None
            row["denominator"] = row.get("contributing_graph_count")
            row["independent_unit"] = "graph"
        associations = property_associations(
            graph_metrics,
            graph_rows,
            inference_replicates=bootstrap_replicates,
        )
        model_contrasts = descriptive_model_contrasts(
            graph_rows,
            bootstrap_replicates=bootstrap_replicates,
        )
    cell = [row for row in estimates if row["stratum_count"] == 1]
    marginal = [row for row in estimates if row["stratum_count"] == 9]
    contrasts = [
        row
        for row in estimates
        if row["metric"]
        in (
            "poincare_minus_euclidean",
            "repaired_minus_unrepaired_poincare",
            "conditional_repair_recovery",
        )
        and "_scaled_" in str(row["coordinate_condition_id"])
    ]
    common_success_stretch = [
        {
            **row,
            "conditioning": "both_ordinary_methods_succeeded",
        }
        for row in estimates
        if row.get("metric")
        == "common_success_poincare_minus_euclidean_stretch"
    ]
    interactions = [*pointwise, *simultaneous]
    evidence = _evidence_rows(
        interactions,
        marginal,
        conditional["Routing Resources"],
    )
    dashboard = [
        {
            **row,
            "interpretation_warning": (
                "Synthetic associative evidence; inspect effect size, uncertainty, "
                "stretch, recovery, and resource overhead together."
            ),
        }
        for row in evidence
    ]
    runtime_rows = [
        {
            **row,
            "graph_count": 1,
            "pair_count": None,
            "numerator": None,
            "denominator": None,
            "status": "defined",
            "applicability": "operational_descriptive_only",
            "na_reason": None,
            "unit": "seconds",
            "method_specific_runtime_claim": False,
        }
        for row in runtimes
    ]
    native_condition_estimates = [
        row
        for row in [*cell, *marginal]
        if row.get("coordinate_condition_id") in {"hydra_native", "mds_native"}
    ]
    tables = {name: [] for name in WORKBOOK_SHEETS}
    tables.update(
        {
            "Results Dashboard": dashboard,
            "Graph Metrics": graph_metrics,
            "Graph-Level Estimates": conditional["Graph-Level Estimates"],
            "Cell Estimates": cell,
            "Model Marginals": marginal,
            "Matched Success Contrasts": contrasts,
            "Matched Embedding Interactions": interactions,
            "Complete-System Comparisons": complete_system,
            "Native References": [*native_condition_estimates, *native_estimates],
            "Equivalence Sensitivity": interactions,
            "Repair Denominators": conditional["Repair Denominators"],
            "Failure Composition": conditional["Failure Composition"],
            "Distance Bands": conditional["Distance Bands"],
            "Method Stretch": conditional["Method Stretch"],
            "Common-Success Stretch": common_success_stretch,
            "Recovered Stretch": conditional["Recovered Stretch"],
            "Distortion Diagnostics": distortion_estimates,
            "Property Associations": associations,
            "Model Contrasts": model_contrasts,
            "Routing Resources": conditional["Routing Resources"],
            "Operational Runtime": runtime_rows,
            "Evidence Table": evidence,
            "Validation Summary": [
                {
                    "graph_count": len(results),
                    "pair_count": sum(len(result["pairs"]) for result in results),
                    "numerator": None,
                    "denominator": None,
                    "raw_graphs_validated": len(results),
                    "oracle_disagreements": sum(
                        int(result["high_precision_sentinel"]["disagreements"])
                        for result in results
                    ),
                    "analysis_validation_mode": analysis_validation_evidence[
                        "validation_mode"
                    ],
                    "scientific_regeneration_requested_by_analysis": (
                        analysis_validation_evidence["regeneration_requested"]
                    ),
                    "scientific_graphs_executed_during_analysis": scientific_counts[0],
                    "dijkstra_executions_during_analysis": scientific_counts[1],
                    "routing_executions_during_analysis": scientific_counts[2],
                    "raw_checkpoints_written_during_analysis": raw_checkpoint_writes,
                    "raw_tree_unchanged_during_analysis": (
                        analysis_validation_evidence["raw_tree_unchanged"]
                    ),
                    "full_scientific_run_was_performed_by_analysis": (
                        scientific_execution_performed
                    ),
                    "complete_frozen_design": complete_design,
                    "bootstrap_replicates": bootstrap_replicates,
                    "status": "validated",
                    "applicability": "applicable",
                    "na_reason": None,
                }
            ],
        }
    )
    return tables


def _analysis_source_identity(root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("analysis requires clean committed source")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = performance_source_manifest(root)
    verify_committed_source_manifest(root, commit, manifest)
    return {
        "source_commit": commit,
        "source_fingerprint": performance_source_fingerprint(root),
        "analysis_content_fingerprint": performance_source_fingerprint(root),
        "source_manifest": manifest,
        "dependency_fingerprint": dependency_fingerprint(root),
        "worktree": "clean",
    }


def _analyze_in_context(
    repository_root: Path | str,
    *,
    excluded_fixture: ExcludedAnalysisFixtureContract | None = None,
) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    if excluded_fixture is None:
        verify_iteration1_immutable(root, deep=True)
    analysis_identity = _analysis_source_identity(root)
    results, raw_hashes, provenance = load_complete_raw_run(
        root,
        excluded_fixture=excluded_fixture,
    )
    tables = create_analysis_tables(
        results,
        analysis_validation_evidence=provenance["analysis_validation"],
        bootstrap_replicates=(
            BOOTSTRAP_REPLICATES
            if excluded_fixture is None
            else excluded_fixture.bootstrap_replicates
        ),
        require_complete_design=excluded_fixture is None,
    )
    output_identity = (
        ITERATION2_ANALYSIS_IDENTITY
        if excluded_fixture is None
        else excluded_fixture.analysis_identity
    )
    raw_identity = (
        ITERATION2_RUN_IDENTITY
        if excluded_fixture is None
        else excluded_fixture.raw_identity
    )
    output = resolve_iteration2_output(root, output_identity)
    return build_reporting_bundle(
        output,
        tables=tables,
        raw_generation_identity=provenance["raw_generation"],
        analysis_source_identity=analysis_identity,
        raw_tree_fingerprint=provenance["raw_tree_fingerprint"],
        raw_manifest_sha256=provenance["raw_manifest_sha256"],
        raw_completion_sha256=provenance["raw_completion_sha256"],
        analysis_validation_evidence=provenance["analysis_validation"],
        raw_location=str(
            resolve_iteration2_output(root, raw_identity)
        ),
        raw_file_hashes=raw_hashes,
        limitations=(
            "Hydra and MDS optimize different geometric objectives.",
            "Hydra Euclidean routing is bounded to the frozen centering gauge.",
            "MDS Poincare routing depends on the prespecified radial mapping.",
            "Scaled Hydra is a non-isometric sensitivity transformation.",
            "ER and BA are synthetic ensembles and differ in multiple properties.",
        ),
        excluded_fixture_payload=(
            None if excluded_fixture is None else excluded_fixture.payload
        ),
    )


def analyze(
    repository_root: Path | str,
    *,
    excluded_fixture: ExcludedAnalysisFixtureContract | None = None,
) -> dict[str, object]:
    """Analyze a complete raw run under one isolated read-only ledger."""

    validate_scientific_boundary_registry()
    with scientific_operation_context(ANALYSIS_READ_ONLY):
        return _analyze_in_context(
            repository_root,
            excluded_fixture=excluded_fixture,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Iteration 2 raw results")
    parser.add_argument("--repository-root", type=Path, default=_repository_root())
    args = parser.parse_args(argv)
    manifest = analyze(args.repository_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
