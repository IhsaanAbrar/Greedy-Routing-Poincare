from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from benchmark_iteration2_capacity import canonical_bytes

from iteration2_config import (
    ANALYSIS_PLAN_HASH,
    COMBINED_PROTOCOL_HASH,
    DATA_GENERATION_HASH,
    EQUIVALENCE_MARGIN_APPROVED,
    EXCLUDED_FIXTURE_SEEDS,
    GRAPH_REPETITIONS_APPROVED,
    ITERATION2_OUTPUT_SCHEMA,
    MATCHED_RADII,
    MATCHED_RADIUS_LABELS,
    OUTPUT_SCHEMA_HASH,
)
from iteration2_excluded import (
    EXCLUDED_SCIENTIFIC_STATUS,
    ExcludedAnalysisFixtureContract,
)
from iteration2_runtime_guard import (
    ANALYSIS_READ_ONLY,
    scientific_operation_context,
)
from run_iteration2 import (
    GRAPH_CHECKPOINT_FILE_COUNT,
    GRAPH_CHECKPOINT_FILENAMES,
    GRAPH_CHECKPOINT_MANIFEST_SCHEMA,
    MANIFEST_SCHEMA,
    RUN_COMPLETION_SCHEMA,
    _json_bytes,
    publish_graph_checkpoint,
)


def condition_ids() -> tuple[str, ...]:
    return (
        "hydra_native",
        "mds_native",
        *(
            f"{family}_scaled_{label}"
            for family in ("hydra", "mds")
            for label in MATCHED_RADIUS_LABELS
        ),
    )


def deterministic_bootstrap_provider(
    *,
    model: str,
    n: int,
    m: int,
    replicate: int,
    graph_count: int,
) -> tuple[int, ...]:
    del model, n, m
    ordered = tuple(range(graph_count))
    return ordered if replicate % 2 == 0 else tuple(reversed(ordered))


def synthetic_inference_rows(
    *,
    replicates_per_stratum: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Build a complete excluded identity grid without production graph IDs."""

    graph_metrics: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    interactions: list[dict[str, object]] = []
    for model_index, model in enumerate(("erdos_renyi", "barabasi_albert")):
        for n_index, n in enumerate((100, 300, 1000)):
            for m_index, m in enumerate((4, 8, 16)):
                for replicate in range(replicates_per_stratum):
                    graph_id = (
                        f"excluded_inference_{model}_{n}_{m}_{replicate}"
                    )
                    within = replicate / max(1, replicates_per_stratum - 1)
                    structural = (
                        model_index * 0.2
                        + n_index * 0.03
                        + m_index * 0.01
                        + within * 0.005
                    )
                    graph_metrics.append(
                        {
                            "graph_id": graph_id,
                            "model": model,
                            "n": n,
                            "m": m,
                            "replicate_index": replicate,
                            "average_degree": 8.0 + structural,
                            "maximum_degree": 12.0 + 2.0 * structural,
                            "population_degree_variance": 3.0 + 3.0 * structural,
                            "average_clustering_coefficient": 0.1 + structural / 10.0,
                            "diameter": 4.0 + structural,
                            "average_shortest_path_length": 2.0 + structural / 2.0,
                        }
                    )
                    for condition_index, condition in enumerate(condition_ids()):
                        mds_native = condition == "mds_native"
                        base = (
                            0.55
                            + model_index * 0.04
                            + n_index * 0.005
                            + m_index * 0.003
                            + condition_index * 0.001
                        )
                        euclidean = base + within * 0.002
                        poincare = (
                            None
                            if mds_native
                            else euclidean + 0.01 + within * 0.001
                        )
                        repaired = (
                            None if mds_native else poincare + 0.02
                        )
                        common_stretch = (
                            None
                            if mds_native
                            else 0.02 + condition_index * 0.001 + within * 0.002
                        )
                        pair_count = 1_000
                        graph_rows.append(
                            {
                                "graph_id": graph_id,
                                "model": model,
                                "n": n,
                                "m": m,
                                "replicate_index": replicate,
                                "coordinate_condition_id": condition,
                                "pair_count": pair_count,
                                "euclidean_success_count": round(euclidean * pair_count),
                                "euclidean_success": euclidean,
                                "poincare_success_count": (
                                    None
                                    if poincare is None
                                    else round(poincare * pair_count)
                                ),
                                "poincare_success": poincare,
                                "repaired_poincare_success_count": (
                                    None
                                    if repaired is None
                                    else round(repaired * pair_count)
                                ),
                                "repaired_poincare_success": repaired,
                                "poincare_minus_euclidean": (
                                    None
                                    if poincare is None
                                    else poincare - euclidean
                                ),
                                "repaired_minus_unrepaired_poincare": (
                                    None
                                    if repaired is None or poincare is None
                                    else repaired - poincare
                                ),
                                "conditional_repair_recovery": (
                                    None if mds_native else 0.2 + within * 0.01
                                ),
                                "repair_recovery_numerator": (
                                    None if mds_native else 20 + replicate
                                ),
                                "repair_recovery_denominator": (
                                    None if mds_native else 100
                                ),
                                "common_success_poincare_minus_euclidean_stretch": common_stretch,
                                "recovered_route_stretch": (
                                    None if mds_native else 1.2 + within * 0.01
                                ),
                                "physical_recovered_route_stretch": (
                                    None if mds_native else 1.3 + within * 0.01
                                ),
                                "stretch_summaries": {
                                    "common_success": {
                                        "numerator": (
                                            None
                                            if common_stretch is None
                                            else common_stretch * 500
                                        ),
                                        "denominator": (
                                            None if common_stretch is None else 500
                                        ),
                                    },
                                    "newly_recovered": {
                                        "numerator": (
                                            None if mds_native else 24.0
                                        ),
                                        "denominator": (
                                            None if mds_native else 20
                                        ),
                                    },
                                },
                            }
                        )
                    for radius_index, radius in enumerate(MATCHED_RADII):
                        interactions.append(
                            {
                                "graph_id": graph_id,
                                "model": model,
                                "n": n,
                                "m": m,
                                "replicate_index": replicate,
                                "matched_radius": radius,
                                "interaction": (
                                    0.002 * (radius_index + 1)
                                    + model_index * 0.001
                                    + within * 0.0005
                                ),
                                "pair_count": 1_000,
                            }
                        )
    return graph_metrics, graph_rows, interactions


def excluded_fixture_contract(
    results: Sequence[Mapping[str, object]],
    *,
    fixture_tag: str = "feasibility_e2e",
) -> ExcludedAnalysisFixtureContract:
    return ExcludedAnalysisFixtureContract(
        fixture_tag=fixture_tag,
        expected_graph_ids=tuple(
            str(result["graph_identity"]["graph_id"]) for result in results
        ),
        excluded_seeds=tuple(EXCLUDED_FIXTURE_SEEDS),
        pair_count=len(results[0]["pairs"]),
        bootstrap_replicates=2,
        property_resampling_replicates=2,
        permutation_replicates=2,
    )


def excluded_reporting_contract(
    fixture_tag: str = "reporting_fixture",
) -> ExcludedAnalysisFixtureContract:
    return ExcludedAnalysisFixtureContract(
        fixture_tag=fixture_tag,
        expected_graph_ids=("excluded_reporting_fixture",),
        excluded_seeds=tuple(EXCLUDED_FIXTURE_SEEDS),
        pair_count=12,
        bootstrap_replicates=2,
        property_resampling_replicates=2,
        permutation_replicates=2,
    )


def excluded_run_manifest(
    results: Sequence[Mapping[str, object]],
    contract: ExcludedAnalysisFixtureContract | None = None,
) -> dict[str, object]:
    contract = contract or excluded_fixture_contract(results)
    schedule = [str(result["graph_identity"]["graph_id"]) for result in results]
    if any(graph_id.startswith("i2_") for graph_id in schedule):
        raise ValueError("fixture manifest cannot contain production graph IDs")
    source_manifest = {"excluded_fixture.py": "3" * 64}
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "run_identity": contract.raw_identity,
        "protocol_hash": COMBINED_PROTOCOL_HASH,
        "data_generation_hash": DATA_GENERATION_HASH,
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "output_schema_hash": OUTPUT_SCHEMA_HASH,
        "output_schema": ITERATION2_OUTPUT_SCHEMA,
        "source_commit": "1" * 40,
        "source_worktree": "clean",
        "source_fingerprint": sha256(canonical_bytes(source_manifest)).hexdigest(),
        "source_manifest": source_manifest,
        "dependency_fingerprint": {"sha256": "4" * 64},
        "capacity_profile_sha256": "5" * 64,
        "capacity_profile_identity": {
            "physical_sha256": "5" * 64,
            "profile_schema": "excluded_fixture_only",
        },
        "graph_count": len(schedule),
        "pairs_per_graph": len(results[0]["pairs"]),
        "raw_graph_file_count": len(schedule),
        "raw_graph_checkpoint_count": len(schedule),
        "raw_files_per_graph_checkpoint": GRAPH_CHECKPOINT_FILE_COUNT,
        "raw_total_file_count": 2 + len(schedule) * GRAPH_CHECKPOINT_FILE_COUNT,
        "checkpoint_layout": {
            "schema": GRAPH_CHECKPOINT_MANIFEST_SCHEMA,
            "directory": "graphs/<graph_id>",
            "publication": "same_parent_atomic_directory_rename",
        },
        "equivalence_margin_human_approved": EQUIVALENCE_MARGIN_APPROVED,
        "graph_repetitions_human_approved": GRAPH_REPETITIONS_APPROVED,
        "schedule": schedule,
        "scientific_status": EXCLUDED_SCIENTIFIC_STATUS,
        "production_compatible": False,
        "excluded_fixture_payload": dict(contract.payload),
        "excluded_fixture_payload_sha256": contract.payload_hash,
    }


def excluded_analysis_validation_evidence() -> dict[str, object]:
    fingerprint = {
        "schema": "raw_tree_fingerprint_v1",
        "file_count": 0,
        "byte_count": 0,
        "sha256": "0" * 64,
    }
    with scientific_operation_context(ANALYSIS_READ_ONLY) as ledger:
        ledger_snapshot = ledger.snapshot()
    return {
        "validation_mode": "read_only_analysis_consumer",
        "regeneration_requested": False,
        "scientific_graphs_executed_during_analysis": 0,
        "dijkstra_executions_during_analysis": 0,
        "routing_executions_during_analysis": 0,
        "raw_checkpoints_written_during_analysis": 0,
        "raw_tree_before": fingerprint,
        "raw_tree_after": dict(fingerprint),
        "raw_tree_unchanged": True,
        "scientific_operation_ledger": ledger_snapshot,
    }


def publish_completed_excluded_raw_run(
    run_root: Path,
    results: Sequence[Mapping[str, object]],
    contract: ExcludedAnalysisFixtureContract | None = None,
) -> dict[str, object]:
    """Publish a small completed raw tree before analysis instrumentation starts."""

    contract = contract or excluded_fixture_contract(results)
    if run_root.name != contract.raw_identity:
        raise ValueError("excluded raw output path does not match fixture identity")
    manifest = excluded_run_manifest(results, contract)
    graph_root = run_root / "graphs"
    graph_root.mkdir(parents=True)
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    for result in results:
        publish_graph_checkpoint(graph_root, result, manifest)
    schedule = [str(result["graph_identity"]["graph_id"]) for result in results]
    checkpoint_hashes = {
        graph_id: {
            filename: sha256((graph_root / graph_id / filename).read_bytes()).hexdigest()
            for filename in sorted(GRAPH_CHECKPOINT_FILENAMES)
        }
        for graph_id in schedule
    }
    completion = {
        "schema": RUN_COMPLETION_SCHEMA,
        "status": "complete",
        "run_identity": manifest["run_identity"],
        "data_generation_hash": manifest["data_generation_hash"],
        "analysis_plan_hash": manifest["analysis_plan_hash"],
        "output_schema_hash": manifest["output_schema_hash"],
        "protocol_hash": manifest["protocol_hash"],
        "run_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "graph_checkpoint_count": len(schedule),
        "files_per_graph_checkpoint": GRAPH_CHECKPOINT_FILE_COUNT,
        "raw_file_count": 2 + len(schedule) * GRAPH_CHECKPOINT_FILE_COUNT,
        "graph_checkpoint_file_sha256": checkpoint_hashes,
        "schedule": schedule,
        "completion_written_last": True,
    }
    (run_root / "run_complete.json").write_bytes(_json_bytes(completion))
    return manifest
