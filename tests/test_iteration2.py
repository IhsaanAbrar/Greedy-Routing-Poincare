from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import networkx as nx
import matplotlib.image as mpimg
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from embedding import prepare_embedding_input  # noqa: E402
from iteration2_analysis import (  # noqa: E402
    aggregate_graph_metrics,
    equivalence_classification,
    equivalence_status_fields,
    graph_level_interactions,
    graph_level_native_interactions,
    graph_level_rows,
    simultaneous_radius_bands,
)
from iteration2_config import (  # noqa: E402
    ANALYSIS_PLAN_HASH,
    COMBINED_PROTOCOL_HASH,
    DATA_GENERATION_HASH,
    EQUIVALENCE_MARGIN_APPROVED,
    EXCLUDED_FIXTURE_SEEDS,
    FULL_RUN_CONFIRMATION_TOKEN,
    GRAPH_REPETITIONS_APPROVED,
    ITERATION2_OUTPUT_SCHEMA,
    MATCHED_RADII,
    MATCHED_RADIUS_LABELS,
    OUTPUT_SCHEMA_HASH,
    analysis_payload,
    canonical_json_bytes,
    data_generation_payload,
    fingerprint,
    full_schedule,
    is_full_oracle_graph,
    output_schema_payload,
    sample_ordered_pairs,
    seeds_for_graph,
)
from iteration2_runtime_guard import (  # noqa: E402
    PREFLIGHT_READ_ONLY,
    ScientificOperationLedger,
)
from iteration2_excluded import ExcludedAnalysisFixtureContract  # noqa: E402
from iteration2_v2_support import (  # noqa: E402
    excluded_analysis_validation_evidence,
)
from iteration2_coordinates import (  # noqa: E402
    create_iteration2_embeddings,
    euclidean_routing_tolerance,
    maximum_radius,
    native_condition,
    uniformly_map_to_radius,
)
from iteration2_embedding_oracle import (  # noqa: E402
    audit_embedding,
    independent_classical_mds,
    independent_hydra,
    pairwise_euclidean,
    pairwise_poincare,
)
from iteration2_oracle import (  # noqa: E402
    audit_production_result,
    decimal_euclidean_distance,
    decimal_poincare_distance,
    independent_euclidean_distance,
    independent_poincare_distance,
)
from iteration2_precision import projected_precision  # noqa: E402
from iteration2_reporting import (  # noqa: E402
    EXCEL_MAXIMUM_ROWS,
    FIGURE_FILES,
    WORKBOOK_SHEETS,
    _worksheet_xml,
    build_reporting_bundle,
    data_dictionary,
    generate_required_figures,
    write_xlsx,
)
from iteration2_routing import (  # noqa: E402
    LOCAL_MINIMUM,
    RoutingPriorityContext,
    euclidean_greedy_route_v2,
    poincare_greedy_route_v2,
    repaired_poincare_greedy_route_v2,
)
from network_metrics import prepare_all_pairs_shortest_paths  # noqa: E402
import run_iteration2 as iteration2_runner  # noqa: E402
from run_iteration2 import run_excluded_feasibility_fixture  # noqa: E402
from validate_iteration2 import verify_iteration1_immutable  # noqa: E402
from analyze_iteration2 import _bootstrap_cell_summaries  # noqa: E402
import benchmark_iteration2_capacity as iteration2_capacity  # noqa: E402


def _analysis_route_record(
    *,
    graph_id: str,
    pair_index: int,
    condition: str,
    method: str,
    success: bool,
    stretch: float | None = None,
    dijkstra_length: int = 1,
    ordinary_failed: bool = False,
) -> dict[str, object]:
    """Build one explicit route row for graph-level analysis fixtures."""

    source = pair_index * 2
    destination = source + 1
    if success:
        resolved_stretch = 1.0 if stretch is None else float(stretch)
        physical_hops = int(round(resolved_stretch * dijkstra_length))
        if not np.isclose(
            physical_hops / dijkstra_length,
            resolved_stretch,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("fixture stretch must resolve to whole physical hops")
        walk = tuple(
            [source]
            + [source + 2] * max(0, physical_hops - 1)
            + [destination]
        )
    else:
        resolved_stretch = None
        physical_hops = 0
        walk = (source,)
    repaired = method == "repaired_poincare_greedy"
    repair_attempted = repaired and ordinary_failed
    repair_succeeded = repair_attempted and success
    final_failure = None
    if not success:
        final_failure = (
            "repair_unavailable_at_source" if repaired else "local_minimum"
        )
    return {
        "graph_id": graph_id,
        "pair_index": pair_index,
        "source": source,
        "destination": destination,
        "coordinate_condition_id": condition,
        "method_id": method,
        "success": success,
        "walk": list(walk),
        "route_length": physical_hops,
        "physical_hops": physical_hops,
        "dijkstra_length": dijkstra_length,
        "stretch": resolved_stretch,
        "physical_stretch": resolved_stretch,
        "initial_failure_type": "local_minimum" if ordinary_failed else None,
        "final_failure_type": final_failure,
        "repair_attempted": repair_attempted,
        "repair_succeeded": repair_succeeded,
        "repair_backtrackable": repair_attempted,
        "repair_eligible": repair_attempted,
        "repair_alternative_existed": True if repair_attempted else None,
        "repair_alternative_selected": repair_attempted,
        "repair_selected_alternative": source + 2 if repair_attempted else None,
        "repair_attempt_count": 1 if repair_attempted else 0,
        "forwarding_decisions": physical_hops,
        "logical_distance_evaluations": physical_hops + 1,
        "peak_history_vertices": max(1, len(set(walk))),
    }


class Iteration2ConfigurationTests(unittest.TestCase):
    def test_schedule_and_seed_domains_are_unique(self):
        schedule = full_schedule()
        self.assertEqual(len(schedule), 360)
        seeds = [seeds_for_graph(spec) for spec in schedule]
        graph_seeds = {seed.graph for seed in seeds}
        pair_seeds = {seed.pairs for seed in seeds}
        self.assertEqual(len(graph_seeds), 360)
        self.assertEqual(len(pair_seeds), 360)
        self.assertTrue(graph_seeds.isdisjoint(pair_seeds))

    def test_pairs_are_unique_ordered_and_repeatable(self):
        first = sample_ordered_pairs(
            range(40), 1_000, graph_id="fixture", pair_seed=123
        )
        second = sample_ordered_pairs(
            range(40), 1_000, graph_id="fixture", pair_seed=123
        )
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 1_000)
        self.assertTrue(all(left != right for left, right in first))

    def test_canonical_float_serialization_is_unambiguous(self):
        payload = canonical_json_bytes({"x": 0.5, "items": (1, True)})
        self.assertIn(b"0x1.0000000000000p-1", payload)
        self.assertEqual(fingerprint({"a": 1}), fingerprint({"a": 1}))
        with self.assertRaises(ValueError):
            canonical_json_bytes({"x": float("nan")})

    def test_human_approvals_are_frozen_in_canonical_payloads(self):
        self.assertTrue(EQUIVALENCE_MARGIN_APPROVED)
        self.assertTrue(GRAPH_REPETITIONS_APPROVED)
        self.assertTrue(
            analysis_payload()["equivalence"]["human_approved"]
        )
        self.assertTrue(
            data_generation_payload()["graph_design"][
                "replicates_human_approved"
            ]
        )

    def test_data_analysis_output_and_combined_hashes_are_separate(self):
        self.assertEqual(
            {
                DATA_GENERATION_HASH,
                ANALYSIS_PLAN_HASH,
                OUTPUT_SCHEMA_HASH,
                COMBINED_PROTOCOL_HASH,
            },
            {
                fingerprint(data_generation_payload()),
                fingerprint(analysis_payload()),
                fingerprint(output_schema_payload()),
                fingerprint(
                    {
                        "schema": data_generation_payload()["schema"],
                        "data_generation_hash": DATA_GENERATION_HASH,
                        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
                        "output_schema_hash": OUTPUT_SCHEMA_HASH,
                    }
                ),
            },
        )
        self.assertEqual(
            output_schema_payload()["schema"],
            ITERATION2_OUTPUT_SCHEMA,
        )

    def test_full_oracle_graphs_are_prespecified_sparse_and_dense(self):
        selected = [spec for spec in full_schedule() if is_full_oracle_graph(spec)]
        self.assertEqual(len(selected), 12)
        self.assertEqual({spec.m for spec in selected}, {4, 16})
        self.assertEqual({spec.n for spec in selected}, {100, 300, 1000})
        self.assertEqual(
            {spec.model for spec in selected},
            {"erdos_renyi", "barabasi_albert"},
        )

    def test_configuration_payload_forbids_result_dependent_selection(self):
        encoded = canonical_json_bytes(
            {
                "data": data_generation_payload(),
                "analysis": analysis_payload(),
            }
        )
        self.assertIn(b"no_outcome_dependent_tuning", encoded)
        self.assertIn(b"forbid_results_as_configuration_inputs", encoded)

    def test_iteration1_manifests_and_derived_inventory_are_immutable(self):
        evidence = verify_iteration1_immutable(PROJECT_ROOT, deep=False)
        self.assertFalse(evidence["deep_raw_tree_checked"])
        self.assertEqual(len(evidence["step17_files"]), 11)


class Iteration2PreflightTests(unittest.TestCase):
    @staticmethod
    def _manifest(*, dirty: bool = False) -> dict[str, object]:
        return {
            "source_commit": "1" * 40,
            "source_worktree": "dirty" if dirty else "clean",
            "source_fingerprint": "2" * 64,
            "dependency_fingerprint": {"sha256": "3" * 64},
            "capacity_profile_sha256": "4" * 64,
        }

    def _preflight(self, **overrides):
        arguments = {
            "mode": "full",
            "confirmation": FULL_RUN_CONFIRMATION_TOKEN,
            "expected_source_commit": "1" * 40,
            "expected_source_fingerprint": "2" * 64,
            "expected_dependency_fingerprint": "3" * 64,
            "expected_capacity_profile": "4" * 64,
            "expected_protocol_hash": COMBINED_PROTOCOL_HASH,
            "resume": False,
        }
        arguments.update(overrides)
        with (
            TemporaryDirectory() as temporary,
            patch.object(
                iteration2_runner,
                "repository_root",
                return_value=Path(temporary),
            ),
            patch.object(
                iteration2_runner,
                "resolve_iteration2_output",
                return_value=Path(temporary) / "absent-output",
            ),
            patch.object(
                iteration2_runner,
                "build_manifest",
                return_value=self._manifest(),
            ),
            patch.object(
                iteration2_runner,
                "_capacity_status",
                return_value={
                    "profile_valid": True,
                    "disk_space_pass": True,
                },
            ),
            patch.object(
                iteration2_runner,
                "verify_iteration1_immutable",
                return_value={"unchanged": True},
            ),
        ):
            output = Path(temporary) / "absent-output"
            report = iteration2_runner.preflight(**arguments)
            self.assertFalse(output.exists())
            return report

    def test_full_preflight_accepts_only_all_matching_identities(self):
        self.assertTrue(self._preflight()["authorized"])
        mismatches = (
            ("confirmation", "wrong"),
            ("expected_source_commit", "0" * 40),
            ("expected_source_fingerprint", "0" * 64),
            ("expected_dependency_fingerprint", "0" * 64),
            ("expected_capacity_profile", "0" * 64),
            ("expected_protocol_hash", "0" * 64),
        )
        for name, value in mismatches:
            with self.subTest(name=name):
                report = self._preflight(**{name: value})
                self.assertFalse(report["authorized"])

    def test_preflight_rejects_dirty_source_missing_approval_and_capacity(self):
        with (
            TemporaryDirectory() as temporary,
            patch.object(
                iteration2_runner,
                "repository_root",
                return_value=Path(temporary),
            ),
            patch.object(
                iteration2_runner,
                "resolve_iteration2_output",
                return_value=Path(temporary) / "absent-output",
            ),
            patch.object(
                iteration2_runner,
                "build_manifest",
                return_value=self._manifest(dirty=True),
            ),
            patch.object(
                iteration2_runner,
                "_capacity_status",
                side_effect=RuntimeError("wrong volume or inadequate disk"),
            ),
            patch.object(
                iteration2_runner,
                "verify_iteration1_immutable",
                return_value={"unchanged": True},
            ),
            patch.object(
                iteration2_runner,
                "EQUIVALENCE_MARGIN_APPROVED",
                False,
            ),
        ):
            report = iteration2_runner.preflight(
                mode="full",
                confirmation=FULL_RUN_CONFIRMATION_TOKEN,
                expected_source_commit="1" * 40,
                expected_source_fingerprint="2" * 64,
                expected_dependency_fingerprint="3" * 64,
                expected_capacity_profile="4" * 64,
                expected_protocol_hash=COMBINED_PROTOCOL_HASH,
            )
        self.assertFalse(report["authorized"])
        joined = " ".join(report["authorization_reasons"])
        self.assertIn("clean committed source", joined)
        self.assertIn("human approval", joined)
        self.assertIn("capacity validation failed", joined)

    def test_resume_structural_validation_rejects_corrupt_checkpoint(self):
        manifest = {
            **self._manifest(),
            "manifest_schema": iteration2_runner.MANIFEST_SCHEMA,
            "run_identity": "test",
            "protocol_hash": COMBINED_PROTOCOL_HASH,
            "data_generation_hash": DATA_GENERATION_HASH,
            "analysis_plan_hash": ANALYSIS_PLAN_HASH,
            "output_schema_hash": OUTPUT_SCHEMA_HASH,
            "output_schema": ITERATION2_OUTPUT_SCHEMA,
            "graph_count": 360,
            "pairs_per_graph": 1000,
            "raw_graph_file_count": 360,
            "raw_total_file_count": 361,
            "equivalence_margin_human_approved": True,
            "graph_repetitions_human_approved": True,
            "schedule": ["g"],
            "scientific_status": "iteration2_prespecified_scientific_run",
            "production_compatible": True,
        }
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "graphs").mkdir()
            (output / "run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            (output / "graphs" / "g.json.gz").write_bytes(b"not-gzip")
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                iteration2_runner._validate_resume_directory(
                    output,
                    manifest,
                    validation_policy=iteration2_runner.ResumeValidationPolicy.READ_ONLY_STRUCTURAL,
                )


class Iteration2CapacityProfileTests(unittest.TestCase):
    @staticmethod
    def _records() -> list[dict[str, object]]:
        records = []
        for cell_index, spec in enumerate(iteration2_capacity.benchmark_specs()):
            for role_index, role in enumerate(
                iteration2_capacity.REPETITION_ROLES
            ):
                full_oracle = (
                    role == iteration2_capacity.MEASURED_ORACLE_ROLE
                )
                checked_pairs = 1_000 if full_oracle else 5
                records.append(
                    {
                        **spec.identity(),
                        "run_identity": iteration2_capacity.excluded_capacity_identity()[
                            "raw_identity"
                        ],
                        "scientific_status": "excluded_non_scientific",
                        "repetition_role": role,
                        "warmup": role == iteration2_capacity.WARMUP_ROLE,
                        "included_in_runtime_projection": (
                            role != iteration2_capacity.WARMUP_ROLE
                        ),
                        "full_oracle": full_oracle,
                        "pair_count": 1_000,
                        "primary_coordinate_route_count": 27_000,
                        "route_record_count": 28_000,
                        "embedding_artifact_count": 10,
                        "independent_embedding_validation_exercised": True,
                        "independently_checked_pair_count": checked_pairs,
                        "independently_checked_route_decisions": (
                            checked_pairs * 28
                        ),
                        "graph_execution_ns": (
                            1_000_000 + cell_index * 10_000 + role_index * 1_000
                        ),
                        "serialization_ns": 10_000,
                        "atomic_publication_ns": 20_000,
                        "checkpoint_validation_ns": 30_000,
                        "end_to_end_ns": (
                            2_000_000 + cell_index * 20_000 + role_index * 2_000
                        ),
                        "checkpoint_bytes": (
                            100_000 + cell_index * 1_000 + role_index * 100
                        ),
                        "uncompressed_result_bytes": (
                            500_000 + cell_index * 2_000 + role_index * 200
                        ),
                        "measurement_worker_sha256": (
                            iteration2_capacity.measurement_worker_fingerprint()
                        ),
                        "measurement_resolution_ns": 1,
                        "component_timings_retained": True,
                        "checkpoint_validation_passed": True,
                        "temporary_output_removed": True,
                        "scientific_result_created": False,
                    }
                )
        return records

    @staticmethod
    def _publication() -> dict[str, object]:
        excluded_identity = iteration2_capacity.excluded_capacity_identity()
        ledger = ScientificOperationLedger(mode=PREFLIGHT_READ_ONLY).snapshot()
        return {
            "run_identity": excluded_identity["raw_identity"],
            "analysis_identity": excluded_identity["analysis_identity"],
            "scientific_status": "excluded_non_scientific",
            "production_compatible": False,
            "scientific_operation_ledger": ledger,
            "end_to_end_ns": 5_000_000,
            "file_count": 30,
            "machine_readable_bytes": 100_000,
            "workbook_bytes": 200_000,
            "figure_bytes": 300_000,
            "workbook_sheet_count": 26,
            "figure_count": 9,
            "temporary_output_removed": True,
            "scientific_result_created": False,
        }

    def _profile(self) -> dict[str, object]:
        return iteration2_capacity.build_profile(
            records=self._records(),
            publication_proxy=self._publication(),
            benchmark_volume_identifier="test-volume",
            available_before_bytes=20 * 1024**3,
            available_after_bytes=20 * 1024**3,
            root=PROJECT_ROOT,
        )

    @staticmethod
    def _clone(value):
        return json.loads(json.dumps(value))

    def test_iteration1_capacity_profile_is_rejected_as_stale(self):
        with self.assertRaisesRegex(
            iteration2_capacity.Iteration2CapacityError,
            "Iteration 2 capacity profile schema mismatch",
        ):
            iteration2_capacity.load_capacity_profile(
                PROJECT_ROOT / "code" / "step15_capacity_profile.json",
                root=PROJECT_ROOT,
            )

    def test_missing_and_malformed_iteration2_profiles_are_rejected(self):
        with TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            with self.assertRaisesRegex(
                iteration2_capacity.Iteration2CapacityError,
                "missing",
            ):
                iteration2_capacity.load_capacity_profile(
                    missing,
                    root=PROJECT_ROOT,
                )
            malformed = Path(temporary) / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(
                iteration2_capacity.Iteration2CapacityError,
                "JSON is invalid",
            ):
                iteration2_capacity.load_capacity_profile(
                    malformed,
                    root=PROJECT_ROOT,
                )

    def test_profile_rejects_all_bound_identity_mismatches(self):
        mutations = {
            "protocol hash": ("protocol_hash", "0" * 64),
            "benchmark code": ("benchmark_code_sha256", "0" * 64),
            "runner source": ("runner_source_sha256", "0" * 64),
            "dependency fingerprint": (
                "dependency_fingerprint",
                {"versions": {}, "sha256": "0" * 64},
            ),
            "output-schema hash": ("output_schema_hash", "0" * 64),
            "excluded benchmark identity": (
                "excluded_benchmark_identity",
                {"raw_identity": "iteration2_raw_invalid"},
            ),
        }
        for expected_message, (key, replacement) in mutations.items():
            with self.subTest(identity=key):
                profile = self._clone(self._profile())
                profile[key] = replacement
                with self.assertRaisesRegex(
                    iteration2_capacity.Iteration2CapacityError,
                    expected_message,
                ):
                    iteration2_capacity.validate_capacity_profile(
                        profile,
                        root=PROJECT_ROOT,
                    )

    def test_wrong_volume_and_insufficient_space_are_rejected(self):
        profile = self._profile()
        with self.assertRaisesRegex(
            iteration2_capacity.Iteration2CapacityError,
            "output volume differs",
        ):
            iteration2_capacity.validate_capacity_profile(
                profile,
                root=PROJECT_ROOT,
                expected_volume_identifier="different-volume",
            )
        required = profile["storage_projection"]["required_free_bytes"]
        with self.assertRaisesRegex(
            iteration2_capacity.Iteration2CapacityError,
            "below Iteration 2 required_free_bytes",
        ):
            iteration2_capacity.validate_capacity_profile(
                profile,
                root=PROJECT_ROOT,
                expected_volume_identifier="test-volume",
                current_available_bytes=required - 1,
            )

    def test_projected_workload_and_runtime_arithmetic_are_exact(self):
        workload = iteration2_capacity.frozen_workload()
        self.assertEqual(workload["total_primary_coordinate_routes"], 9_720_000)
        self.assertEqual(workload["total_route_records"], 10_080_000)
        self.assertEqual(workload["full_oracle_graph_count"], 12)
        self.assertEqual(workload["standard_sentinel_graph_count"], 348)
        records = self._records()
        runtime = iteration2_capacity.project_runtime(
            records,
            self._publication(),
        )
        expected = self._publication()["end_to_end_ns"]
        by_cell = {
            (row["cell_id"], row["repetition_role"]): row for row in records
        }
        for spec in iteration2_capacity.benchmark_specs():
            expected += (
                58
                * by_cell[
                    (
                        spec.cell_id,
                        iteration2_capacity.MEASURED_STANDARD_ROLE,
                    )
                ]["end_to_end_ns"]
                + 2
                * by_cell[
                    (
                        spec.cell_id,
                        iteration2_capacity.MEASURED_ORACLE_ROLE,
                    )
                ]["end_to_end_ns"]
            )
        self.assertEqual(runtime["nominal_projected_runtime_ns"], expected)
        self.assertEqual(
            runtime["conservative_projected_runtime_ns"],
            int(expected * iteration2_capacity.CONSERVATIVE_RUNTIME_FACTOR),
        )
        storage = iteration2_capacity.project_storage(
            records,
            self._publication(),
        )
        self.assertEqual(
            storage["required_free_bytes"],
            storage["projected_final_storage_bytes"]
            + storage["safe_resume_overhead_bytes"]
            + storage["atomic_checkpoint_peak_overhead_bytes"]
            + storage["fixed_free_space_reserve_bytes"],
        )

    def test_excluded_benchmark_domains_and_profile_forbid_scientific_results(self):
        iteration2_capacity.validate_benchmark_domains()
        excluded_identity = iteration2_capacity.excluded_capacity_identity()
        self.assertTrue(
            str(excluded_identity["raw_identity"]).startswith(
                "iteration2_excluded_raw_"
            )
        )
        self.assertEqual(
            excluded_identity["scientific_status"],
            "excluded_non_scientific",
        )
        self.assertFalse(excluded_identity["production_compatible"])
        scientific_ids = {spec.graph_id for spec in full_schedule()}
        self.assertTrue(
            scientific_ids.isdisjoint(
                spec.graph_id for spec in iteration2_capacity.benchmark_specs()
            )
        )
        profile = self._profile()
        self.assertFalse(profile["scientific_results_created"])
        self.assertFalse(
            iteration2_capacity._forbidden_keys(profile)
        )

    def test_worker_passes_canonical_excluded_raw_identity_to_execution(self):
        spec = iteration2_capacity.benchmark_specs()[0]
        generated = SimpleNamespace(graph=object(), metadata={})
        with TemporaryDirectory() as temporary:
            with (
                patch.object(
                    iteration2_capacity,
                    "_generate_graph",
                    return_value=generated,
                ),
                patch.object(
                    iteration2_capacity,
                    "execute_iteration2_graph",
                    side_effect=RuntimeError("identity captured"),
                ) as execute,
            ):
                with self.assertRaisesRegex(RuntimeError, "identity captured"):
                    iteration2_capacity._worker_record(
                        spec,
                        iteration2_capacity.WARMUP_ROLE,
                        temporary,
                    )
        self.assertEqual(
            execute.call_args.kwargs["run_identity"],
            iteration2_capacity.excluded_capacity_identity()["raw_identity"],
        )

    def test_real_publication_proxy_is_excluded_measured_and_cleaned(self):
        record = iteration2_capacity.run_publication_proxy()
        excluded_identity = iteration2_capacity.excluded_capacity_identity()
        self.assertEqual(record["run_identity"], excluded_identity["raw_identity"])
        self.assertEqual(
            record["analysis_identity"],
            excluded_identity["analysis_identity"],
        )
        self.assertEqual(record["scientific_status"], "excluded_non_scientific")
        self.assertFalse(record["production_compatible"])
        self.assertEqual(
            record["scientific_operation_ledger"]["total_attempted"],
            0,
        )
        self.assertTrue(record["temporary_output_removed"])
        self.assertFalse(record["scientific_result_created"])

    def test_mocked_benchmark_creates_only_profile_not_scientific_results(self):
        records = iter(self._records())
        before = {
            path.relative_to(PROJECT_ROOT / "results").as_posix()
            for path in (PROJECT_ROOT / "results").rglob("*")
            if path.is_file()
        }
        with (
            TemporaryDirectory() as temporary,
            patch.object(
                iteration2_capacity,
                "_run_worker",
                side_effect=lambda _spec, _role: next(records),
            ),
            patch.object(
                iteration2_capacity,
                "run_publication_proxy",
                return_value=self._publication(),
            ),
        ):
            target = Path(temporary) / "capacity.json"
            report = iteration2_capacity.run_benchmark(profile_path=target)
            self.assertTrue(target.is_file())
            self.assertFalse(report["scientific_results_created"])
        after = {
            path.relative_to(PROJECT_ROOT / "results").as_posix()
            for path in (PROJECT_ROOT / "results").rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)


class Iteration2RoutingTests(unittest.TestCase):
    def test_independent_euclidean_oracle_matches_decimal_reference(self):
        fixtures = (
            ((0.0, 0.0), (0.0, 0.0)),
            ((1e-300, -1e-300), (2e-300, 3e-300)),
            ((-0.75, 0.25), (0.625, -0.5)),
        )
        for left, right in fixtures:
            observed = independent_euclidean_distance(left, right)
            reference = float(decimal_euclidean_distance(left, right))
            self.assertTrue(
                np.isclose(observed, reference, rtol=2e-15, atol=0.0)
                or observed == reference == 0.0
            )

    def test_zero_coordinate_distance_is_exact(self):
        point = (0.125, -0.375)
        self.assertEqual(independent_euclidean_distance(point, point), 0.0)
        self.assertEqual(independent_poincare_distance(point, point), 0.0)

    def test_strict_progress_is_classified_before_revisit(self):
        graph = nx.Graph([(0, 1), (1, 3), (3, 2)])
        coordinates = {
            0: (0.30, 0.0),
            1: (0.20, 0.0),
            2: (0.00, 0.0),
            3: (0.50, 0.0),
        }
        result = euclidean_greedy_route_v2(
            graph, coordinates, 0, 2, tolerance=1e-14
        )
        self.assertFalse(result.success)
        self.assertEqual(result.walk, (0, 1))
        self.assertEqual(result.final_failure_type, LOCAL_MINIMUM)
        self.assertLessEqual(
            result.final_failure_diagnostic.progress_gap,
            result.final_failure_diagnostic.distance_tolerance,
        )

    def test_exact_tie_uses_frozen_keyed_priority(self):
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        coordinates = {
            0: (-0.5, 0.0),
            1: (0.0, 0.2),
            2: (0.0, -0.2),
            3: (0.5, 0.0),
        }
        priority = RoutingPriorityContext(
            DATA_GENERATION_HASH, "legacy-exact-tie-fixture", 0, 0, 3
        )
        result = euclidean_greedy_route_v2(
            graph,
            coordinates,
            0,
            3,
            tolerance=1e-14,
            priority_context=priority,
        )
        expected = min(
            (priority.priority(0, candidate), candidate)
            for candidate in (1, 2)
        )[1]
        self.assertEqual(result.walk[1], expected)

    def test_near_tie_within_tolerance_uses_frozen_keyed_priority(self):
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        coordinates = {
            0: (-0.8, 0.0),
            1: (0.200000000000005, 0.0),
            2: (0.2, 0.0),
            3: (0.0, 0.0),
        }
        priority = RoutingPriorityContext(
            DATA_GENERATION_HASH, "legacy-near-tie-fixture", 0, 0, 3
        )
        result = euclidean_greedy_route_v2(
            graph,
            coordinates,
            0,
            3,
            tolerance=1e-14,
            priority_context=priority,
        )
        expected = min(
            (priority.priority(0, candidate), candidate)
            for candidate in (1, 2)
        )[1]
        self.assertEqual(result.walk, (0, expected, 3))
        agreement = audit_production_result(
            result,
            graph=graph,
            coordinates=coordinates,
            source=0,
            destination=3,
            metric="euclidean",
            tolerance=1e-14,
            repaired=False,
        )
        self.assertTrue(agreement.float64_matches_high_precision)

    def test_duplicate_coordinates_are_not_jittered(self):
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        coordinates = {
            0: (-0.5, 0.0),
            1: (0.0, 0.0),
            2: (0.0, 0.0),
            3: (0.5, 0.0),
        }
        result = euclidean_greedy_route_v2(
            graph, coordinates, 0, 3, tolerance=1e-14
        )
        self.assertTrue(result.success)
        self.assertEqual(result.walk, (0, 1, 3))

    def test_one_step_repair_records_complete_provenance(self):
        graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.2, 0.0),
            2: (0.65, 0.0),
            3: (-0.5, 0.4),
            4: (0.85, 0.0),
        }
        result = repaired_poincare_greedy_route_v2(
            graph, coordinates, 0, 4, tolerance=1e-14
        )
        self.assertTrue(result.success)
        self.assertEqual(result.walk, (0, 1, 2, 1, 3, 4))
        self.assertEqual(result.repair_backtracked_vertex, 1)
        self.assertEqual(result.repair_excluded_branch, 2)
        self.assertEqual(result.repair_selected_alternative, 3)
        self.assertEqual(result.repair_attempt_count, 1)
        agreement = audit_production_result(
            result,
            graph=graph,
            coordinates=coordinates,
            source=0,
            destination=4,
            metric="poincare",
            tolerance=1e-14,
            repaired=True,
        )
        self.assertTrue(agreement.float64_matches_production)
        self.assertTrue(agreement.high_precision_matches_production)

    def test_independent_oracle_matches_all_methods(self):
        graph = nx.path_graph(5)
        coordinates = {
            node: (-0.8 + 0.4 * node, 0.0) for node in graph.nodes
        }
        methods = (
            (
                euclidean_greedy_route_v2,
                "euclidean",
                False,
            ),
            (
                poincare_greedy_route_v2,
                "poincare",
                False,
            ),
            (
                repaired_poincare_greedy_route_v2,
                "poincare",
                True,
            ),
        )
        for source in graph.nodes:
            for destination in graph.nodes:
                if source == destination:
                    continue
                for function, metric, repaired in methods:
                    result = function(
                        graph,
                        coordinates,
                        source,
                        destination,
                        tolerance=1e-14,
                    )
                    agreement = audit_production_result(
                        result,
                        graph=graph,
                        coordinates=coordinates,
                        source=source,
                        destination=destination,
                        metric=metric,
                        tolerance=1e-14,
                        repaired=repaired,
                    )
                    self.assertTrue(agreement.float64_matches_production)
                    self.assertTrue(agreement.high_precision_matches_production)

    def test_high_precision_boundary_distance_agrees(self):
        left = (np.nextafter(1.0, 0.0), 0.0)
        right = (-np.nextafter(1.0, 0.0), 0.0)
        value = independent_poincare_distance(left, right)
        precise = float(decimal_poincare_distance(left, right))
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, precise, places=12)

    def test_poincare_reference_handles_close_and_antipodal_points(self):
        fixtures = (
            ((0.1, 0.2), np.nextafter((0.1, 0.2), (1.0, 1.0))),
            ((0.999999999, 0.0), (-0.999999999, 0.0)),
            ((0.0, 0.0), (1e-15, -1e-15)),
        )
        for left, right in fixtures:
            observed = independent_poincare_distance(left, right)
            reference = float(decimal_poincare_distance(left, right))
            self.assertTrue(np.isfinite(observed))
            self.assertTrue(
                np.isclose(observed, reference, rtol=5e-14, atol=1e-28)
            )

    def test_outside_disk_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            independent_poincare_distance((1.0, 0.0), (0.0, 0.0))

    def test_uniform_radius_can_change_poincare_route_as_prespecified(self):
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        raw = {
            0: (-0.8, -0.2),
            1: (0.1079515859954221, 0.8690451426229069),
            2: (0.05114376335588244, -0.26655514605035746),
            3: (-0.44231931848785133, 0.5442924158530003),
        }
        maximum = max(np.hypot(*point) for point in raw.values())
        scaled = {
            radius: {
                node: tuple(value * radius / maximum for value in point)
                for node, point in raw.items()
            }
            for radius in (0.5, 0.95)
        }
        low = poincare_greedy_route_v2(
            graph, scaled[0.5], 0, 3, tolerance=1e-14
        )
        high = poincare_greedy_route_v2(
            graph, scaled[0.95], 0, 3, tolerance=1e-14
        )
        self.assertEqual(low.walk, (0, 1, 3))
        self.assertEqual(high.walk, (0, 2, 3))


class Iteration2EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = nx.path_graph(6)
        cls.paths = prepare_all_pairs_shortest_paths(cls.graph)
        cls.embedding_input = prepare_embedding_input(
            cls.graph,
            cls.paths,
            configuration_fingerprint="iteration2-test",
            tolerance=1e-12,
        )
        cls.embeddings = create_iteration2_embeddings(cls.embedding_input)

    def test_all_native_and_matched_conditions_exist(self):
        conditions = self.embeddings.routable_conditions
        self.assertEqual(len(conditions), 9)
        self.assertEqual(
            {condition.condition_id for condition in conditions},
            {
                "hydra_native",
                *(
                    f"{family}_scaled_{label}"
                    for family in ("hydra", "mds")
                    for label in MATCHED_RADIUS_LABELS
                ),
            },
        )

    def test_matched_radii_are_exact_within_frozen_roundoff(self):
        for condition in self.embeddings.matched_conditions:
            self.assertAlmostEqual(
                maximum_radius(condition.coordinates),
                condition.target_maximum_radius,
                places=14,
            )

    def test_uniform_scaling_preserves_euclidean_pair_order(self):
        base = self.embeddings.mds_native
        first = uniformly_map_to_radius(
            embedding_family="mds",
            condition_id="mds_scaled_r050",
            coordinates=base.coordinates,
            node_order=self.embedding_input.node_order,
            target_radius=0.5,
        )
        before = pairwise_euclidean(
            base.coordinates, self.embedding_input.node_order
        )
        after = pairwise_euclidean(
            first.coordinates, self.embedding_input.node_order
        )
        nonzero = before > 0.0
        ratios = after[nonzero] / before[nonzero]
        self.assertTrue(
            np.allclose(
                ratios,
                first.scale_factor,
                rtol=1e-14,
                atol=1e-15,
            )
        )

    def test_scaled_hydra_is_not_mislabelled_as_native(self):
        scaled = next(
            condition
            for condition in self.embeddings.matched_conditions
            if condition.condition_id == "hydra_scaled_r050"
        )
        self.assertFalse(scaled.standard_native_embedding)
        self.assertTrue(scaled.sensitivity_transformation)
        self.assertNotEqual(
            pairwise_poincare(
                scaled.coordinates, self.embedding_input.node_order
            ).tolist(),
            pairwise_poincare(
                self.embeddings.hydra_native.coordinates,
                self.embedding_input.node_order,
            ).tolist(),
        )

    def test_independent_hydra_reconstruction_agrees(self):
        agreement = audit_embedding(
            family="hydra",
            distance_matrix=self.embedding_input.distance_matrix,
            node_order=self.embedding_input.node_order,
            production_coordinates=self.embeddings.hydra_result.coordinates,
            production_effective_rank=(
                self.embeddings.hydra_result.metadata.effective_spatial_rank
            ),
        )
        self.assertLess(agreement.maximum_pairwise_distance_error, 1e-8)

    def test_independent_mds_reconstruction_agrees(self):
        agreement = audit_embedding(
            family="mds",
            distance_matrix=self.embedding_input.distance_matrix,
            node_order=self.embedding_input.node_order,
            production_coordinates=self.embeddings.mds_result.coordinates,
            production_effective_rank=self.embeddings.mds_result.metadata.effective_rank,
        )
        self.assertLess(agreement.maximum_pairwise_distance_error, 1e-8)

    def test_independent_rank_one_fallbacks(self):
        distance = np.asarray(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
        )
        mds = independent_classical_mds(distance, (0, 1, 2))
        hydra = independent_hydra(distance, (0, 1, 2))
        self.assertEqual(mds.effective_rank, 1)
        self.assertIn(hydra.effective_rank, (1, 2))

    def test_diagnostics_cover_every_condition(self):
        diagnostics = self.embeddings.diagnostics
        self.assertEqual(len(diagnostics), 10)
        self.assertTrue(all(row["finite_coordinates"] for row in diagnostics))
        self.assertTrue(
            all("rejected_eigenvalues" in row for row in diagnostics)
        )

    def test_complete_collapse_and_mds_rank_zero_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "collapse"):
            native_condition(
                "collapsed",
                "mds",
                {0: (0.0, 0.0), 1: (0.0, 0.0)},
                (0, 1),
            )
        with self.assertRaisesRegex(ValueError, "rank zero"):
            independent_classical_mds(np.zeros((3, 3)), (0, 1, 2))

    def test_independent_reconstructions_cover_mathematical_graph_fixtures(self):
        twin = nx.Graph(
            [(0, 1), (0, 4), (4, 5), (1, 2), (1, 3)]
        )
        fixtures = (
            nx.path_graph(5),
            nx.cycle_graph(6),
            nx.star_graph(5),
            nx.complete_graph(5),
            twin,
        )
        for graph in fixtures:
            paths = prepare_all_pairs_shortest_paths(graph)
            embedding_input = prepare_embedding_input(
                graph,
                paths,
                configuration_fingerprint="iteration2-fixtures",
                tolerance=1e-12,
            )
            embeddings = create_iteration2_embeddings(embedding_input)
            for audit in embeddings.independent_validation:
                self.assertLess(
                    audit["maximum_pairwise_distance_error"],
                    1e-8,
                )

    def test_degenerate_hydra_truncation_mismatch_fails_closed(self):
        graph = nx.Graph(
            [(0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)]
        )
        paths = prepare_all_pairs_shortest_paths(graph)
        embedding_input = prepare_embedding_input(
            graph,
            paths,
            configuration_fingerprint="degenerate-hydra",
            tolerance=1e-12,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "independent embedding reconstruction disagreed",
        ):
            create_iteration2_embeddings(embedding_input)

    def test_frechet_convergence_and_centering_isometry_are_recorded(self):
        metadata = self.embeddings.hydra_result.metadata
        self.assertLessEqual(
            metadata.final_frechet_mean_residual,
            2.0 * metadata.centering_tolerance,
        )
        self.assertLessEqual(
            metadata.centering_iteration_count,
            metadata.centering_max_iterations,
        )
        before = pairwise_poincare(
            self.embeddings.hydra_uncentered_reference,
            self.embedding_input.node_order,
        )
        after = pairwise_poincare(
            self.embeddings.hydra_native.coordinates,
            self.embedding_input.node_order,
        )
        self.assertTrue(np.allclose(before, after, rtol=1e-9, atol=1e-10))

    def test_mds_radii_preserve_every_euclidean_route(self):
        conditions = [
            condition
            for condition in self.embeddings.matched_conditions
            if condition.embedding_family == "mds"
        ]
        for source in self.graph:
            for destination in self.graph:
                if source == destination:
                    continue
                walks = {
                    euclidean_greedy_route_v2(
                        self.graph,
                        condition.coordinates,
                        source,
                        destination,
                        tolerance=euclidean_routing_tolerance(condition),
                    ).walk
                    for condition in conditions
                }
                self.assertEqual(len(walks), 1)

    def test_rotation_and_reflection_preserve_both_metric_routes(self):
        coordinates = self.embeddings.hydra_native.coordinates
        transformed = {
            node: (-point[1], point[0]) for node, point in coordinates.items()
        }
        for function in (
            euclidean_greedy_route_v2,
            poincare_greedy_route_v2,
        ):
            for source in self.graph:
                for destination in self.graph:
                    if source == destination:
                        continue
                    first = function(
                        self.graph,
                        coordinates,
                        source,
                        destination,
                        tolerance=1e-13,
                    )
                    second = function(
                        self.graph,
                        transformed,
                        source,
                        destination,
                        tolerance=1e-13,
                    )
                    self.assertEqual(first.walk, second.walk)
                    self.assertEqual(first.success, second.success)

    def test_adjacency_insertion_order_does_not_change_embeddings(self):
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
        first = nx.Graph()
        first.add_edges_from(edges)
        second = nx.Graph()
        second.add_edges_from(reversed(edges))
        outputs = []
        for graph in (first, second):
            paths = prepare_all_pairs_shortest_paths(graph)
            item = prepare_embedding_input(
                graph,
                paths,
                configuration_fingerprint="insertion-order",
                tolerance=1e-12,
            )
            outputs.append(create_iteration2_embeddings(item))
        self.assertTrue(
            np.allclose(
                pairwise_poincare(
                    outputs[0].hydra_native.coordinates, tuple(range(5))
                ),
                pairwise_poincare(
                    outputs[1].hydra_native.coordinates, tuple(range(5))
                ),
                rtol=1e-10,
                atol=1e-10,
            )
        )
        self.assertTrue(
            np.allclose(
                pairwise_euclidean(
                    outputs[0].mds_native.coordinates, tuple(range(5))
                ),
                pairwise_euclidean(
                    outputs[1].mds_native.coordinates, tuple(range(5))
                ),
                rtol=1e-10,
                atol=1e-10,
            )
        )

    def test_node_relabelling_preserves_pairwise_geometries(self):
        graph = nx.Graph(
            [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
        )
        relabel = {0: 3, 1: 0, 2: 5, 3: 2, 4: 1, 5: 4}
        embeddings = []
        for item in (graph, nx.relabel_nodes(graph, relabel)):
            paths = prepare_all_pairs_shortest_paths(item)
            prepared = prepare_embedding_input(
                item,
                paths,
                configuration_fingerprint="node-relabel",
                tolerance=1e-12,
            )
            embeddings.append(create_iteration2_embeddings(prepared))
        for family, distance in (
            ("hydra_native", independent_poincare_distance),
            ("mds_native", independent_euclidean_distance),
        ):
            left = getattr(embeddings[0], family).coordinates
            right = getattr(embeddings[1], family).coordinates
            for first in graph:
                for second in graph:
                    self.assertAlmostEqual(
                        distance(left[first], left[second]),
                        distance(
                            right[relabel[first]],
                            right[relabel[second]],
                        ),
                        places=9,
                    )


class Iteration2AnalysisTests(unittest.TestCase):
    def test_equivalence_classification_boundaries(self):
        cases = (
            ((-1.0, 1.0), "ci_wholly_inside_margin"),
            ((1.01, 2.0), "ci_wholly_positive_beyond_margin"),
            ((-2.0, -1.01), "ci_wholly_negative_beyond_margin"),
            ((0.01, 1.2), "practical_magnitude_unresolved"),
            ((-2.0, 0.5), "practical_magnitude_unresolved"),
        )
        for interval, expected in cases:
            self.assertEqual(
                equivalence_classification(*interval), expected
            )

    def test_equivalence_statuses_remain_separate_and_exhaustive(self):
        statuses = equivalence_status_fields(0.01, 1.2)
        self.assertTrue(statuses["ci_excludes_zero"])
        self.assertFalse(statuses["ci_wholly_inside_margin"])
        self.assertFalse(statuses["ci_wholly_positive_beyond_margin"])
        self.assertFalse(statuses["ci_wholly_negative_beyond_margin"])
        self.assertTrue(statuses["practical_magnitude_unresolved"])

    def test_graph_level_rows_and_interactions(self):
        records = []
        methods = {
            "hydra_native": (
                "euclidean_greedy",
                "poincare_greedy",
                "repaired_poincare_greedy",
            ),
            "mds_native": ("euclidean_greedy",),
        }
        for family in ("hydra", "mds"):
            for label in MATCHED_RADIUS_LABELS:
                methods[f"{family}_scaled_{label}"] = (
                    "euclidean_greedy",
                    "poincare_greedy",
                    "repaired_poincare_greedy",
                )
        for pair in range(2):
            for condition, condition_methods in methods.items():
                for method in condition_methods:
                    ordinary_failed = pair == 1
                    success = not (
                        ordinary_failed and method == "poincare_greedy"
                    )
                    records.append(
                        _analysis_route_record(
                            graph_id="g",
                            pair_index=pair,
                            condition=condition,
                            method=method,
                            success=success,
                            ordinary_failed=(
                                ordinary_failed
                                and method == "repaired_poincare_greedy"
                            ),
                        )
                    )
        rows = graph_level_rows(
            records,
            graph_id="g",
            model="erdos_renyi",
            n=100,
            m=4,
            replicate_index=0,
            pair_count=2,
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(graph_level_interactions(rows)), 4)
        self.assertEqual(len(graph_level_native_interactions(rows)), 1)
        ordinary = rows[0]["failure_summaries"]["euclidean_greedy"]
        self.assertEqual(
            ordinary["repair_unavailable_at_source"]["applicability"],
            "not_applicable",
        )

    def test_simultaneous_bands_use_paired_graphs(self):
        rows = []
        for model in ("erdos_renyi", "barabasi_albert"):
            for n in (100, 300, 1000):
                for m in (4, 8, 16):
                    for replicate in range(3):
                        for radius in MATCHED_RADII:
                            rows.append(
                                {
                                    "graph_id": f"{model}-{n}-{m}-{replicate}",
                                    "model": model,
                                    "n": n,
                                    "m": m,
                                    "replicate_index": replicate,
                                    "matched_radius": radius,
                                    "interaction": (replicate - 1) * 0.001,
                                }
                            )

        def provider(**kwargs):
            count = kwargs["graph_count"]
            replicate = kwargs["replicate"]
            return tuple((replicate + index) % count for index in range(count))

        bands = simultaneous_radius_bands(
            rows, bootstrap_replicates=10, bootstrap_provider=provider
        )
        self.assertEqual(len(bands), 80)
        self.assertTrue(
            all(row["all_radius_ci_wholly_inside_margin"] for row in bands)
        )
        self.assertTrue(
            all(
                row["ci_lower"] == row["simultaneous_ci_lower"]
                and row["ci_upper"] == row["simultaneous_ci_upper"]
                and row["equivalence_margin_lower"] == -1.0
                and row["equivalence_margin_upper"] == 1.0
                for row in bands
            )
        )

    def test_graph_aggregation_and_equal_stratum_marginal(self):
        rows = []
        for model_index, model in enumerate(
            ("erdos_renyi", "barabasi_albert")
        ):
            for n_index, n in enumerate((100, 300, 1000)):
                for m_index, m in enumerate((4, 8, 16)):
                    for replicate in range(2):
                        rows.append(
                            {
                                "graph_id": (
                                    f"{model}-{n}-{m}-{replicate}"
                                ),
                                "model": model,
                                "n": n,
                                "m": m,
                                "replicate_index": replicate,
                                "coordinate_condition_id": "hydra_native",
                                "euclidean_success": (
                                    0.2
                                    + model_index * 0.1
                                    + n_index * 0.01
                                    + m_index * 0.001
                                    + replicate * 0.0001
                                ),
                            }
                        )

        def provider(**kwargs):
            count = kwargs["graph_count"]
            replicate = kwargs["replicate"]
            return tuple((replicate + index) % count for index in range(count))

        estimates = aggregate_graph_metrics(
            rows,
            metrics=("euclidean_success",),
            bootstrap_replicates=10,
            bootstrap_provider=provider,
        )
        marginals = [
            row
            for row in estimates
            if row["scope"] == "model_condition_n_m_marginal"
        ]
        self.assertEqual(len(estimates), 20)
        self.assertEqual(len(marginals), 2)
        expected_er = np.mean(
            [
                row["euclidean_success"]
                for row in rows
                if row["model"] == "erdos_renyi"
            ]
        )
        self.assertAlmostEqual(marginals[0]["estimate"], expected_er)
        self.assertEqual(marginals[0]["independent_unit"], "graph")

    def test_failure_and_distortion_bootstrap_preserve_applicability(self):
        rows = [
            {
                "model": "erdos_renyi",
                "n": 100,
                "m": 4,
                "replicate_index": replicate,
                "coordinate_condition_id": "hydra_native",
                "failure_type": "local_minimum",
                "rate_all_pairs": value,
                "repair_only_rate": None,
                "scale_fitted_mean_relative_error": 0.2 + value,
            }
            for replicate, value in enumerate((0.1, 0.2, 0.3))
        ]
        estimates = _bootstrap_cell_summaries(
            rows,
            identity_fields=(
                "model",
                "n",
                "m",
                "coordinate_condition_id",
                "failure_type",
            ),
            value_fields=(
                "rate_all_pairs",
                "repair_only_rate",
                "scale_fitted_mean_relative_error",
            ),
            bootstrap_replicates=10,
        )
        by_metric = {row["metric"]: row for row in estimates}
        self.assertEqual(
            by_metric["repair_only_rate"]["applicability"],
            "not_applicable",
        )
        self.assertIsNone(by_metric["repair_only_rate"]["estimate"])
        self.assertEqual(
            by_metric["rate_all_pairs"]["contributing_graph_count"],
            3,
        )
        self.assertLessEqual(
            by_metric["scale_fitted_mean_relative_error"]["ci_lower"],
            by_metric["scale_fitted_mean_relative_error"]["estimate"],
        )

    def test_success_conditioned_stretch_records_common_and_recovered_sets(self):
        records = []
        condition_methods = {
            "hydra_native": (
                "euclidean_greedy",
                "poincare_greedy",
                "repaired_poincare_greedy",
            ),
            "mds_native": ("euclidean_greedy",),
            **{
                f"{family}_scaled_{label}": (
                    "euclidean_greedy",
                    "poincare_greedy",
                    "repaired_poincare_greedy",
                )
                for family in ("hydra", "mds")
                for label in MATCHED_RADIUS_LABELS
            },
        }
        for pair in range(3):
            for condition, methods in condition_methods.items():
                for method in methods:
                    success = True
                    stretch = 1.0
                    if condition == "hydra_native":
                        if method == "euclidean_greedy":
                            success, stretch = pair != 2, 1.0 + pair
                        elif method == "poincare_greedy":
                            success, stretch = pair != 1, 1.5 + pair
                        else:
                            success, stretch = True, 1.5 + pair
                    records.append(
                        _analysis_route_record(
                            graph_id="stretch",
                            pair_index=pair,
                            condition=condition,
                            method=method,
                            success=success,
                            stretch=stretch if success else None,
                            dijkstra_length=2,
                            ordinary_failed=(
                                condition == "hydra_native"
                                and method == "repaired_poincare_greedy"
                                and pair == 1
                            ),
                        )
                    )
        rows = graph_level_rows(
            records,
            graph_id="stretch",
            model="erdos_renyi",
            n=100,
            m=4,
            replicate_index=0,
            pair_count=3,
        )
        summaries = rows[0]["stretch_summaries"]
        self.assertEqual(
            summaries["common_success"]["pair_count"],
            1,
        )
        self.assertEqual(
            summaries["newly_recovered"]["pair_count"],
            1,
        )


class Iteration2PrecisionAndReportingTests(unittest.TestCase):
    def test_precision_projection_uses_graph_variance(self):
        rows = []
        for model in ("erdos_renyi", "barabasi_albert"):
            prefix = "er" if model == "erdos_renyi" else "ba"
            for n in (100, 300, 1000):
                for m in (4, 8, 16):
                    for replicate in range(20):
                        graph_id = f"{prefix}-{n}-{m}-{replicate}"
                        for condition in (
                            "hydra",
                            "mds_r050",
                            "mds_r070",
                            "mds_r085",
                            "mds_r095",
                        ):
                            rows.append(
                                {
                                    "graph_id": graph_id,
                                    "model": model,
                                    "n": str(n),
                                    "m": str(m),
                                    "coordinate_condition_id": condition,
                                    "poincare_advantage": str(
                                        (replicate - 9.5) / 10_000
                                        if condition == "hydra"
                                        else 0.0
                                    ),
                                }
                            )
        report = projected_precision(rows)
        self.assertEqual(len(report["projections"]), 32)
        widths = [
            row["projected_model_marginal_full_95_ci_width_pp"]
            for row in report["projections"]
            if row["model"] == "erdos_renyi"
            and row["radius_label"] == "r050"
        ]
        self.assertGreater(widths[0], widths[-1])
        self.assertFalse(report["confirmatory_equivalence_power_claim"])
        self.assertFalse(report["design_changed_from_frozen_protocol"])

    def test_workbook_has_exact_formula_free_sheet_set(self):
        sheets = {
            name: [
                {
                    "status": "ok",
                    "estimate": 0.5,
                    "unit": "proportion",
                },
                {
                    "status": "ok",
                    "estimate": 0.25,
                    "unit": "percentage_points",
                },
            ]
            for name in WORKBOOK_SHEETS
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "iteration2.xlsx"
            write_xlsx(path, sheets)
            with zipfile.ZipFile(path) as archive:
                workbook = archive.read("xl/workbook.xml").decode("utf-8")
                all_xml = b"".join(
                    archive.read(name)
                    for name in archive.namelist()
                    if name.endswith(".xml")
                )
            self.assertTrue(all(name in workbook for name in WORKBOOK_SHEETS))
            self.assertNotIn(b"<f>", all_xml)
            self.assertNotIn(b"<f ", all_xml)
            self.assertIn(b'state="frozen"', all_xml)
            self.assertNotIn(b"<filterColumn", all_xml)
            self.assertNotIn(b'state="hidden"', all_xml)
            self.assertIn(b's="3"', all_xml)
            self.assertIn(b's="1"', all_xml)

    def test_data_dictionary_covers_every_machine_table_column(self):
        tables = {
            "Cell Estimates": [
                {
                    "estimate": 0.5,
                    "unit": "proportion",
                    "applicability": "applicable",
                }
            ],
            "Property Associations": [
                {
                    "association_estimate": None,
                    "na_reason": "zero_within_stratum_residual_variance",
                }
            ],
        }
        dictionary = data_dictionary(tables)
        covered = {(row["table"], row["column"]) for row in dictionary}
        expected = {
            (table, column)
            for table, rows in tables.items()
            for column in rows[0]
        }
        self.assertTrue(expected <= covered)
        self.assertTrue(
            all(
                row["definition"]
                and row["unit"]
                and row["denominator"]
                and row["applicability"]
                and row["missing_value_rule"]
                for row in dictionary
            )
        )

    def test_excel_row_limit_is_enforced_before_serialization(self):
        class OversizedRows:
            def __len__(self):
                return EXCEL_MAXIMUM_ROWS

            def __iter__(self):
                return iter(())

        with self.assertRaisesRegex(ValueError, "row limit"):
            _worksheet_xml(OversizedRows())

    def test_required_figure_files_are_generated(self):
        with TemporaryDirectory() as temporary:
            paths = generate_required_figures(
                temporary,
                {
                    "Success Contrasts": (),
                    "Matched-Radius Interactions": (),
                },
            )
            self.assertEqual(tuple(path.name for path in paths), FIGURE_FILES)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_excluded_synthetic_reporting_bundle_is_complete_and_visual(self):
        tables = {name: [] for name in WORKBOOK_SHEETS}
        tables.update(
            {
                "Cell Estimates": [
                    {
                        "model": "erdos_renyi",
                        "n": 100,
                        "m": 4,
                        "coordinate_condition_id": "hydra_scaled_r050",
                        "metric": "poincare_success",
                        "estimate": 0.6,
                        "ci_lower": 0.5,
                        "ci_upper": 0.7,
                        "unit": "proportion",
                        "applicability": "applicable",
                    }
                ],
                "Matched Success Contrasts": [
                    {
                        "metric": "poincare_minus_euclidean",
                        "estimate": 0.25,
                        "ci_lower": -0.1,
                        "ci_upper": 0.6,
                        "unit": "percentage_points",
                    },
                    {
                        "metric": "repaired_minus_unrepaired_poincare",
                        "estimate": 5.0,
                        "ci_lower": 4.0,
                        "ci_upper": 6.0,
                        "unit": "percentage_points",
                    },
                ],
                "Matched Embedding Interactions": [
                    {
                        "matched_radius": 0.5,
                        "estimate": 0.2,
                        "ci_lower": -0.2,
                        "ci_upper": 0.6,
                        "simultaneous_ci_lower": -0.4,
                        "simultaneous_ci_upper": 0.8,
                        "equivalence_margin_lower": -1.0,
                        "equivalence_margin_upper": 1.0,
                        "equivalence_classification": "ci_wholly_inside_margin",
                        "all_radius_ci_wholly_inside_margin": True,
                        "unit": "percentage_points",
                    }
                ],
                "Equivalence Sensitivity": [
                    {
                        "estimate": 0.2,
                        "ci_lower": -0.4,
                        "ci_upper": 0.8,
                        "equivalence_margin_lower": -1.0,
                        "equivalence_margin_upper": 1.0,
                        "equivalence_classification": "ci_wholly_inside_margin",
                        "simultaneous_result_applicability": "applicable",
                        "unit": "percentage_points",
                    }
                ],
                "Failure Composition": [
                    {
                        "failure_type": "local_minimum",
                        "failure_stage": "final",
                        "method_id": "poincare_greedy",
                        "category_index": 0,
                        "estimate": 0.4,
                        "applicability": "applicable",
                        "unit": "proportion",
                    }
                ],
                "Common-Success Stretch": [
                    {
                        "estimate": 1.25,
                        "graph_count": 2,
                        "pair_count": 10,
                        "conditioning": "both_ordinary_methods_succeeded",
                    }
                ],
                "Distortion Diagnostics": [
                    {
                        "model": "erdos_renyi",
                        "n": 100,
                        "m": 4,
                        "coordinate_condition_id": "hydra_scaled_r050",
                        "geometry": "poincare",
                        "metric": "scale_fitted_mean_relative_error",
                        "estimate": 0.2,
                    },
                    {
                        "model": "erdos_renyi",
                        "n": 100,
                        "m": 4,
                        "coordinate_condition_id": "hydra_scaled_r050",
                        "geometry": "poincare",
                        "metric": "poincare_success",
                        "estimate": 0.6,
                    },
                ],
                "Property Associations": [
                    {
                        "association_estimate": None,
                        "applicability": "not_applicable",
                        "na_reason": "zero_within_stratum_residual_variance",
                    }
                ],
                "Operational Runtime": [
                    {"n": 100, "total_seconds": 0.1}
                ],
            }
        )
        with TemporaryDirectory() as temporary:
            contract = ExcludedAnalysisFixtureContract(
                fixture_tag="legacy_reporting_visual",
                expected_graph_ids=("excluded_synthetic_reporting",),
                excluded_seeds=tuple(EXCLUDED_FIXTURE_SEEDS),
                pair_count=10,
                bootstrap_replicates=2,
                property_resampling_replicates=2,
                permutation_replicates=2,
            )
            output = Path(temporary) / contract.analysis_identity
            manifest = build_reporting_bundle(
                output,
                tables=tables,
                source_commit="1" * 40,
                raw_location="excluded/synthetic/raw",
                raw_file_hashes={"synthetic.json.gz": "2" * 64},
                limitations=("Synthetic fixture; no scientific conclusion.",),
                raw_generation_identity={
                    "run_identity": contract.raw_identity,
                    "scientific_status": "excluded_non_scientific",
                    "production_compatible": False,
                },
                excluded_fixture_payload=contract.payload,
                analysis_validation_evidence=(
                    excluded_analysis_validation_evidence()
                ),
            )
            workbook_path = output / "iteration2_results.xlsx"
            with zipfile.ZipFile(workbook_path) as archive:
                workbook = archive.read("xl/workbook.xml")
                xml = b"".join(
                    archive.read(name)
                    for name in archive.namelist()
                    if name.endswith(".xml")
                )
            self.assertEqual(len(manifest["workbook_sheets"]), 26)
            self.assertEqual(manifest["figures"], list(FIGURE_FILES))
            self.assertEqual(
                manifest["output_schema_hash"],
                OUTPUT_SCHEMA_HASH,
            )
            self.assertEqual(workbook.count(b"<sheet "), 26)
            self.assertNotIn(b"<f>", xml)
            self.assertNotIn(b"<filterColumn", xml)
            self.assertNotIn(b'state="hidden"', xml)
            dictionary_csv = (
                output / "data_dictionary.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("na_reason", dictionary_csv)
            self.assertIn("equivalence_margin_lower", dictionary_csv)
            readme_csv = (output / "readme.csv").read_text(encoding="utf-8")
            self.assertIn(DATA_GENERATION_HASH, readme_csv)
            self.assertIn(ANALYSIS_PLAN_HASH, readme_csv)
            self.assertIn(OUTPUT_SCHEMA_HASH, readme_csv)
            for name in FIGURE_FILES:
                image = mpimg.imread(output / "figures" / name)
                self.assertGreater(image.shape[0], 100)
                self.assertGreater(image.shape[1], 100)
                self.assertGreater(float(np.std(image)), 0.0)


class Iteration2IntegrationTests(unittest.TestCase):
    def test_excluded_fixture_exercises_every_new_condition_without_writes(self):
        report = run_excluded_feasibility_fixture()
        self.assertTrue(report["excluded_from_scientific_analysis"])
        self.assertFalse(report["wrote_scientific_results"])
        self.assertEqual(report["graph_count"], 2)
        self.assertEqual(
            {row["route_records"] for row in report["validation_counts"]},
            {336},
        )
        self.assertTrue(
            all(
                row["oracle_pair_count"] == 12
                and row["oracle_route_decisions_checked"] == 336
                and row["oracle_selection_mode"]
                == "prespecified_full_pair_oracle_graph"
                and row["pair_reuse_across_all_conditions"]
                and row["hydra_poincare_gauge_changed_pairs"] == 0
                for row in report["diagnostic_summaries"]
            )
        )

    def test_hash_seed_does_not_change_protocol(self):
        command = [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(PROJECT_ROOT / 'code')!r});"
                "from iteration2_config import COMBINED_PROTOCOL_HASH;"
                "print(COMBINED_PROTOCOL_HASH)"
            ),
        ]
        outputs = []
        for seed in ("1", "987654"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            outputs.append(
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_embedding_subprocess_is_exactly_deterministic_across_hash_seeds(self):
        command = [
            sys.executable,
            "-B",
            "-c",
            (
                "import json,sys,networkx as nx;"
                f"sys.path.insert(0,{str(PROJECT_ROOT / 'code')!r});"
                "from embedding import prepare_embedding_input;"
                "from network_metrics import prepare_all_pairs_shortest_paths;"
                "from iteration2_coordinates import create_iteration2_embeddings;"
                "g=nx.Graph();"
                "g.add_edges_from([(3,4),(0,2),(2,3),(0,1),(1,3)]);"
                "p=prepare_all_pairs_shortest_paths(g);"
                "i=prepare_embedding_input(g,p,"
                "configuration_fingerprint='subprocess',tolerance=1e-12);"
                "e=create_iteration2_embeddings(i);"
                "print(json.dumps({"
                "'h':dict(e.hydra_native.coordinates),"
                "'m':dict(e.mds_native.coordinates)},"
                "sort_keys=True,separators=(',',':')))"
            ),
        ]
        outputs = []
        for seed in ("3", "7001"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            outputs.append(
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
