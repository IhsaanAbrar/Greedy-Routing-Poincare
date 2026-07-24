"""Safety, schedule, orchestration, and fixture tests for Step 14."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_full_experiment as runner  # noqa: E402
from experiment_checkpoint import (  # noqa: E402
    CheckpointAudit,
    GraphCheckpointData,
    RESULT_SCHEMA_VERSION,
    audit_run_checkpoints,
    validate_graph_checkpoint,
)
from experiment_config import (  # noqa: E402
    BARABASI_ALBERT,
    ERDOS_RENYI,
    FULL_EXPERIMENT_CONFIG,
)


def small_entries() -> tuple[runner.GraphScheduleEntry, ...]:
    setting = FULL_EXPERIMENT_CONFIG.parameter_settings[0]
    return tuple(
        runner.GraphScheduleEntry(
            schedule_index=index,
            setting_index=0,
            model=model,
            n=3,
            m=1,
            replicate_index=0,
            graph_id=graph_id,
            configuration_name="test",
            setting_label=setting.label,
        )
        for index, (model, graph_id) in enumerate(
            (
                (ERDOS_RENYI, "test_er_graph"),
                (BARABASI_ALBERT, "test_ba_graph"),
            )
        )
    )


def small_manifest(
    output_root: Path,
    entries: tuple[runner.GraphScheduleEntry, ...],
) -> dict[str, object]:
    return {
        "manifest_schema": "test",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "configuration_schema_version": 4,
        "seed_identity_version": 3,
        "data_generation_hash": runner.DATA_GENERATION_HASH,
        "analysis_plan_hash": runner.ANALYSIS_PLAN_HASH,
        "combined_freeze_hash": runner.COMBINED_FREEZE_HASH,
        "git_commit_hash": "a" * 40,
        "git_working_tree": "clean",
        "source_fingerprint": "b" * 64,
        "python_version": "3.14.0",
        "dependency_versions": {"networkx": "3.6.1"},
        "operating_system": "test-os",
        "hardware": {"machine": "test", "processor": "test", "cpu_count": 1},
        "output_schema": {"id": runner.OUTPUT_SCHEMA_ID, "version": 1},
        "execution_profile": "development_fixture",
        "execution_model": runner.EXECUTION_MODEL,
        "run_directory_name": "test_run",
        "schedule": [entry.graph_id for entry in entries],
        "workload": {"graph_replicates": len(entries)},
        "output_root": str(output_root.resolve()),
        "timer": "time.perf_counter_ns",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
    }


def small_checkpoint_data(
    graph_id: str,
    manifest: dict[str, object],
) -> GraphCheckpointData:
    pair = (0, 1)
    route = {
        "graph_id": graph_id,
        "pair_index": 0,
        "pair_id": f"{graph_id}:pair:0000",
        "source": 0,
        "destination": 1,
        "coordinate_condition_id": "hydra",
        "method_id": "euclidean_greedy",
        "success": True,
        "initial_failure_type": None,
        "final_failure_type": None,
        "repair_attempted": False,
        "repair_succeeded": False,
        "repair_alternative_existed": None,
        "repair_attempt_count": 0,
        "route_length": 1,
        "physical_hop_count": 1,
        "dijkstra_length": 1,
        "dijkstra_hop_count": 1,
        "stretch": 1.0,
        "runtime_ns": 1,
        "walk": [0, 1],
        "forwarding_decisions": 1,
    }
    return GraphCheckpointData(
        graph_id=graph_id,
        generation_metadata={"graph_id": graph_id, "graph_seed": 1},
        edges=((0, 1), (1, 2)),
        network_metrics={"number_of_vertices": 3, "number_of_edges": 2},
        pairs=(pair,),
        coordinates={
            "hydra": {0: (0.0, 0.0), 1: (0.2, 0.0), 2: (0.0, 0.2)}
        },
        embedding_metadata={"hydra": {"rank": 2}, "mds_base": {"rank": 2}},
        distortions=(
            {
                "metric_condition_id": "hydra_euclidean",
                "mean_absolute_relative_distortion": 0.1,
            },
        ),
        dijkstra_records=(
            {
                "graph_id": graph_id,
                "pair_index": 0,
                "pair_id": f"{graph_id}:pair:0000",
                "source": 0,
                "destination": 1,
                "method_id": "dijkstra",
                "success": True,
                "route_length": 1,
                "apsp_length": 1,
                "apsp_agreement": True,
                "runtime_ns": 1,
                "walk": [0, 1],
            },
        ),
        route_records=(route,),
        timings={"graph_generation_ns": 1},
        run_manifest=manifest,
    )


def authorized_report(
    output_root: Path,
    entries: tuple[runner.GraphScheduleEntry, ...],
) -> runner.PreflightReport:
    manifest = small_manifest(output_root, entries)
    run_root = output_root / "test_run"
    audit = CheckpointAudit(
        run_root=run_root,
        run_manifest_present=False,
        complete_graph_ids=(),
        remaining_graph_ids=tuple(entry.graph_id for entry in entries),
        errors=(),
        resumable=True,
    )
    return runner.PreflightReport(
        authorized=True,
        authorization_reasons=(),
        output_root=output_root,
        run_root=run_root,
        schedule_ids=tuple(entry.graph_id for entry in entries),
        run_manifest=manifest,
        checkpoint_audit=audit,
        free_disk_bytes=1,
    )


class ScheduleAndManifestTests(unittest.TestCase):
    def test_full_schedule_has_exact_stable_order_and_ids(self):
        schedule = runner.build_full_schedule()
        self.assertEqual(len(schedule), 360)
        self.assertEqual(len({entry.graph_id for entry in schedule}), 360)
        self.assertEqual(schedule[0].graph_id, "er_n0100_m04_rep000")
        self.assertEqual(schedule[19].graph_id, "er_n0100_m04_rep019")
        self.assertEqual(schedule[20].graph_id, "er_n0100_m08_rep000")
        self.assertEqual(schedule[179].graph_id, "er_n1000_m16_rep019")
        self.assertEqual(schedule[180].graph_id, "ba_n0100_m04_rep000")
        self.assertEqual(schedule[-1].graph_id, "ba_n1000_m16_rep019")
        self.assertTrue(
            all(entry.model == ERDOS_RENYI for entry in schedule[:180])
        )
        self.assertTrue(
            all(entry.model == BARABASI_ALBERT for entry in schedule[180:])
        )

    def test_run_manifest_records_frozen_identity_schema_and_workload(self):
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            manifest = runner.build_experiment_run_manifest(
                output_root=output_root,
                schedule=runner.build_full_schedule(),
                execution_profile="full",
                require_final_scientific_source=False,
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            self.assertEqual(manifest["result_schema_version"], 1)
            self.assertEqual(
                manifest["data_generation_hash"],
                runner.EXPECTED_DATA_GENERATION_HASH,
            )
            self.assertEqual(
                manifest["analysis_plan_hash"],
                runner.EXPECTED_ANALYSIS_PLAN_HASH,
            )
            self.assertEqual(
                manifest["combined_freeze_hash"],
                runner.EXPECTED_COMBINED_FREEZE_HASH,
            )
            self.assertEqual(manifest["execution_model"], runner.EXECUTION_MODEL)
            self.assertEqual(len(manifest["schedule"]), 360)
            self.assertEqual(
                manifest["workload"]["total_routing_and_benchmark_executions"],
                5_760_000,
            )
            self.assertEqual(
                manifest["workload"]["distortion_metric_pair_evaluations"],
                461_412_000,
            )
            self.assertEqual(
                manifest["timing_policy"]["step15_runtime_field"],
                "end_to_end_graph_wall_ns",
            )
            self.assertTrue(
                manifest["timing_policy"][
                    "prepublication_wall_is_not_end_to_end"
                ]
            )
            self.assertTrue(
                manifest["checkpoint_policy"][
                    "publication_timing_record_required_for_resume"
                ]
            )
            self.assertEqual(manifest["created_at_utc"], manifest["timestamp_utc"])
            self.assertEqual(runner.EXECUTION_MODEL, "single_process_sequential_per_graph")

    def test_import_and_schedule_planning_create_no_output(self):
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "declared-results"
            runner.build_full_schedule()
            self.assertFalse(output_root.exists())

    def test_production_parser_has_no_subset_or_skip_switch(self):
        parser = runner._parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "run",
                        "--mode",
                        "full",
                        "--confirm-full-run",
                        runner.COMBINED_FREEZE_HASH,
                        "--max-graphs",
                        "1",
                    ]
                )
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "run",
                        "--mode",
                        "full",
                        "--confirm-full-run",
                        runner.COMBINED_FREEZE_HASH,
                        "--skip",
                        "3",
                    ]
                )

    def test_output_path_containment_rejects_escape(self):
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "escapes"):
                runner._resolved_inside(output_root, "..", "outside")

    def test_frozen_hashes_and_workloads_remain_exact(self):
        self.assertEqual(
            runner.DATA_GENERATION_HASH,
            runner.EXPECTED_DATA_GENERATION_HASH,
        )
        self.assertEqual(
            runner.ANALYSIS_PLAN_HASH,
            runner.EXPECTED_ANALYSIS_PLAN_HASH,
        )
        self.assertEqual(
            runner.COMBINED_FREEZE_HASH,
            runner.EXPECTED_COMBINED_FREEZE_HASH,
        )
        workload = FULL_EXPERIMENT_CONFIG.workload_estimate
        self.assertEqual(workload["graph_replicates"], 360)
        self.assertEqual(workload["sampled_ordered_pairs"], 360_000)
        self.assertEqual(
            workload["coordinate_dependent_routing_executions"], 5_400_000
        )
        self.assertEqual(workload["actual_dijkstra_executions"], 360_000)
        self.assertEqual(
            workload["total_routing_and_benchmark_executions"], 5_760_000
        )
        self.assertEqual(
            workload["distortion_metric_pair_evaluations"], 461_412_000
        )

    def test_results_are_ignored_and_step14_tests_do_not_create_them(self):
        ignore_text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("results/", ignore_text.splitlines())
        self.assertFalse((PROJECT_ROOT / "results").exists())


class PreflightSafetyTests(unittest.TestCase):
    def test_preflight_is_read_only_and_wrong_confirmation_is_refused(self):
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "results"
            report = runner.preflight(
                mode="full",
                confirmation="wrong",
                output_root=output_root,
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            self.assertFalse(report.authorized)
            self.assertIn("exact --confirm-full-run", " ".join(report.authorization_reasons))
            self.assertEqual(len(report.schedule_ids), 360)
            self.assertFalse(output_root.exists())

    def test_development_mode_cannot_authorize_production(self):
        with TemporaryDirectory() as temporary:
            report = runner.preflight(
                mode="development",
                confirmation=runner.COMBINED_FREEZE_HASH,
                output_root=Path(temporary) / "results",
            )
            self.assertFalse(report.authorized)
            self.assertIn("--mode full", " ".join(report.authorization_reasons))

    def test_dirty_source_tree_is_refused_for_final_mode(self):
        with TemporaryDirectory() as temporary:
            original_builder = runner.build_experiment_run_manifest

            def dirty_manifest(**kwargs):
                manifest = original_builder(**kwargs)
                manifest["git_working_tree"] = "dirty"
                return manifest

            with patch.object(
                runner,
                "build_experiment_run_manifest",
                side_effect=dirty_manifest,
            ):
                report = runner.preflight(
                    mode="full",
                    confirmation=runner.COMBINED_FREEZE_HASH,
                    output_root=Path(temporary) / "results",
                )
            self.assertEqual(report.run_manifest["git_working_tree"], "dirty")
            self.assertIn(
                "clean Git working tree",
                " ".join(report.authorization_reasons),
            )

    def test_missing_confirmation_is_refused_without_writes(self):
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "results"
            report = runner.preflight(
                mode="full",
                confirmation=None,
                output_root=output_root,
            )
            self.assertFalse(report.authorized)
            self.assertFalse(output_root.exists())

    def test_dependency_mismatch_is_a_hard_authorization_failure(self):
        with TemporaryDirectory() as temporary:
            with patch.object(runner, "_requirements_pins", return_value={}):
                report = runner.preflight(
                    mode="full",
                    confirmation=runner.COMBINED_FREEZE_HASH,
                    output_root=Path(temporary) / "results",
                )
            self.assertIn(
                "dependency versions",
                " ".join(report.authorization_reasons),
            )

    def test_wrong_confirmation_never_invokes_execution_or_creates_output(self):
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "results"
            with patch.object(runner, "execute_full_run") as execute:
                with redirect_stdout(io.StringIO()):
                    exit_code = runner.main(
                        [
                            "run",
                            "--mode",
                            "full",
                            "--confirm-full-run",
                            "incorrect",
                            "--output-root",
                            str(output_root),
                        ]
                    )
            self.assertEqual(exit_code, 2)
            execute.assert_not_called()
            self.assertFalse(output_root.exists())

    def test_preflight_reports_workload_disk_and_every_graph_id(self):
        with TemporaryDirectory() as temporary:
            report = runner.preflight(
                mode="full",
                confirmation=runner.COMBINED_FREEZE_HASH,
                output_root=Path(temporary) / "results",
            )
            payload = report.as_dict()
            self.assertGreater(payload["free_disk_bytes"], 0)
            self.assertEqual(payload["scheduled_graph_count"], 360)
            self.assertEqual(
                payload["workload"]["actual_dijkstra_executions"], 360_000
            )
            self.assertEqual(len(payload["schedule_ids"]), 360)
            self.assertEqual(
                payload["disk_space_policy"],
                "reported_only_no_unvalidated_minimum_threshold",
            )


class ResumeOrchestrationTests(unittest.TestCase):
    def test_resume_skips_complete_graphs_and_executes_only_missing_in_order(self):
        entries = small_entries()
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "output"
            report = authorized_report(output_root, entries)
            calls: list[str] = []

            def executor(entry, **_kwargs):
                calls.append(entry.graph_id)
                return small_checkpoint_data(entry.graph_id, report.run_manifest), 0

            with patch.object(runner, "build_full_schedule", return_value=entries):
                completed = runner.execute_full_run(
                    report,
                    resume=False,
                    graph_executor=executor,
                )
            self.assertEqual(calls, [entry.graph_id for entry in entries])
            self.assertEqual(completed, tuple(calls))
            for entry in entries:
                validate_graph_checkpoint(
                    report.run_root / "graphs" / entry.graph_id,
                    expected_run_manifest=report.run_manifest,
                )

            calls.clear()
            resumed_audit = audit_run_checkpoints(
                report.run_root,
                schedule_ids=report.schedule_ids,
                expected_run_manifest=report.run_manifest,
            )
            resumed_report = runner.PreflightReport(
                authorized=True,
                authorization_reasons=(),
                output_root=report.output_root,
                run_root=report.run_root,
                schedule_ids=report.schedule_ids,
                run_manifest=report.run_manifest,
                checkpoint_audit=resumed_audit,
                free_disk_bytes=1,
            )
            with patch.object(runner, "build_full_schedule", return_value=entries):
                completed_again = runner.execute_full_run(
                    resumed_report,
                    resume=True,
                    graph_executor=executor,
                )
            self.assertEqual(calls, [])
            self.assertEqual(completed_again, report.schedule_ids)

    def test_existing_run_requires_explicit_resume(self):
        entries = small_entries()
        with TemporaryDirectory() as temporary:
            report = authorized_report(Path(temporary) / "output", entries)
            report.run_root.mkdir(parents=True)
            with self.assertRaisesRegex(
                runner.FullRunAuthorizationError, "--resume"
            ):
                runner.execute_full_run(report, resume=False)

    def test_mid_graph_failure_stops_schedule_and_preserves_no_complete_marker(self):
        entries = small_entries()
        with TemporaryDirectory() as temporary:
            report = authorized_report(Path(temporary) / "output", entries)
            calls: list[str] = []

            def failing_executor(entry, **_kwargs):
                calls.append(entry.graph_id)
                raise RuntimeError("intentional graph failure")

            with patch.object(runner, "build_full_schedule", return_value=entries):
                with self.assertRaisesRegex(RuntimeError, "intentional graph failure"):
                    runner.execute_full_run(
                        report,
                        resume=False,
                        graph_executor=failing_executor,
                    )
            self.assertEqual(calls, [entries[0].graph_id])
            temporary_errors = tuple(
                (report.run_root / "graphs").glob("*.tmp-*")
            )
            self.assertEqual(len(temporary_errors), 1)
            self.assertTrue((temporary_errors[0] / "ERROR.json").is_file())
            self.assertFalse((temporary_errors[0] / "COMPLETE.json").exists())
            self.assertFalse(
                (report.run_root / "graphs" / entries[1].graph_id).exists()
            )

    def test_unauthorized_report_cannot_enter_execution(self):
        entries = small_entries()
        with TemporaryDirectory() as temporary:
            report = authorized_report(Path(temporary) / "output", entries)
            blocked = runner.PreflightReport(
                authorized=False,
                authorization_reasons=("blocked",),
                output_root=report.output_root,
                run_root=report.run_root,
                schedule_ids=report.schedule_ids,
                run_manifest=report.run_manifest,
                checkpoint_audit=report.checkpoint_audit,
                free_disk_bytes=1,
            )
            executor = Mock()
            with self.assertRaises(runner.FullRunAuthorizationError):
                runner.execute_full_run(
                    blocked,
                    resume=False,
                    graph_executor=executor,
                )
            executor.assert_not_called()
            self.assertFalse(report.output_root.exists())


class DevelopmentFixtureTests(unittest.TestCase):
    def test_one_excluded_graph_produces_five_conditions_and_three_methods(self):
        entry = runner._development_fixture_entries()[1]
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            manifest = runner.build_experiment_run_manifest(
                output_root=output_root,
                schedule=(entry,),
                execution_profile="development_fixture",
                require_final_scientific_source=False,
                config=runner.DEVELOPMENT_CONFIG,
            )
            generated = runner.generate_connected_barabasi_albert(
                n=entry.n,
                m=entry.m,
                graph_seed=runner.FEASIBILITY_PILOT_SEEDS[0],
                replicate_index=0,
            )
            data, _ = runner.execute_graph_entry(
                entry,
                run_manifest=manifest,
                config=runner.DEVELOPMENT_CONFIG,
                pair_count=1,
                pair_master_seed=runner.FEASIBILITY_PILOT_SEEDS[1],
                generated_override=generated,
            )
            self.assertEqual(
                tuple(data.coordinates),
                ("hydra", "mds_r050", "mds_r070", "mds_r085", "mds_r095"),
            )
            self.assertEqual(
                {
                    record["method_id"] for record in data.route_records
                },
                {
                    "euclidean_greedy",
                    "poincare_greedy",
                    "repaired_poincare_greedy",
                },
            )
            self.assertEqual(len(data.route_records), 15)
            self.assertEqual(len(data.dijkstra_records), 1)
            self.assertEqual(len(data.distortions), 7)
            self.assertEqual(
                data.dijkstra_records[0]["route_length"],
                data.route_records[0]["dijkstra_length"],
            )
            self.assertIn("actual_dijkstra_ns", data.timings)
            self.assertIn("routing_repaired_poincare_greedy_ns", data.timings)

    def test_disposable_fixture_exercises_both_models_and_safety_paths(self):
        report = runner.run_development_fixture()
        self.assertEqual(report["label"], runner.DEVELOPMENT_FIXTURE_LABEL)
        self.assertEqual(report["graph_count"], 2)
        self.assertEqual(report["pair_count"], 4)
        self.assertEqual(report["route_count"], 60)
        self.assertEqual(report["resume_skipped_complete_graphs"], 2)
        self.assertTrue(report["corruption_detected"])
        self.assertTrue(report["identity_mismatch_detected"])
        self.assertTrue(report["mid_graph_failure_preserved_without_complete"])
        self.assertTrue(report["temporary_output_cleaned_on_return"])


if __name__ == "__main__":
    unittest.main()
