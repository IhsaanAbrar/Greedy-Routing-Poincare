from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from benchmark_iteration2_capacity import (  # noqa: E402
    Iteration2CapacityError,
    load_capacity_profile,
)
from iteration2_v2_support import excluded_run_manifest  # noqa: E402
from iteration2_runtime_guard import (  # noqa: E402
    ANALYSIS_READ_ONLY,
    ScientificOperationBlocked,
    scientific_operation_context,
)
import run_iteration2 as runner  # noqa: E402
from run_iteration2 import (  # noqa: E402
    GRAPH_CHECKPOINT_FILENAMES,
    GRAPH_COMPLETION_FILENAME,
    GRAPH_MANIFEST_FILENAME,
    GRAPH_RESULT_FILENAME,
    ResumeValidationPolicy,
    _gzip_bytes,
    _json_bytes,
    _parser,
    _validate_resume_directory,
    excluded_feasibility_results,
    publish_graph_checkpoint,
    validate_checkpoint_directory,
)


class Iteration2CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = excluded_feasibility_results()
        cls.manifest = excluded_run_manifest(cls.results)

    def test_gzip_is_canonical_deterministic_and_has_zero_mtime(self):
        payload = _json_bytes({"finite": 1.25, "stable": [3, 2, 1]})
        first = _gzip_bytes(payload)
        second = _gzip_bytes(payload)
        self.assertEqual(first, second)
        self.assertEqual(first[4:8], b"\x00\x00\x00\x00")
        self.assertEqual(first[3], 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            _json_bytes({"invalid": float("nan")})
        with self.assertRaisesRegex(ValueError, "collide"):
            _json_bytes({1: "integer", "1": "string"})

    def test_atomic_checkpoint_has_exact_inventory_and_validates(self):
        with TemporaryDirectory(prefix="iteration2-excluded-checkpoint-") as temporary:
            root = Path(temporary)
            first_root = root / "first" / "graphs"
            second_root = root / "second" / "graphs"
            first_root.mkdir(parents=True)
            second_root.mkdir(parents=True)
            first = publish_graph_checkpoint(
                first_root,
                self.results[0],
                self.manifest,
            )
            second = publish_graph_checkpoint(
                second_root,
                self.results[0],
                self.manifest,
            )
            self.assertEqual(
                {path.name for path in first.iterdir()},
                GRAPH_CHECKPOINT_FILENAMES,
            )
            validated = validate_checkpoint_directory(
                first,
                run_manifest=self.manifest,
            )
            self.assertEqual(validated["graph_id"], first.name)
            self.assertEqual(validated["row_counts"]["route_records"], 12 * 28)
            for filename in GRAPH_CHECKPOINT_FILENAMES:
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )
            completion = json.loads(
                (first / GRAPH_COMPLETION_FILENAME).read_text(encoding="utf-8")
            )
            self.assertTrue(completion["completion_written_last"])
            self.assertEqual(
                completion["files_before_completion"],
                [GRAPH_RESULT_FILENAME, GRAPH_MANIFEST_FILENAME],
            )

    def test_checkpoint_corruption_and_stale_manifest_fail_closed(self):
        with TemporaryDirectory(prefix="iteration2-excluded-corrupt-") as temporary:
            graph_root = Path(temporary) / "graphs"
            graph_root.mkdir()
            checkpoint = publish_graph_checkpoint(
                graph_root,
                self.results[0],
                self.manifest,
            )
            stale = deepcopy(self.manifest)
            stale["output_schema_hash"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "stale|binding"):
                validate_checkpoint_directory(
                    checkpoint,
                    run_manifest=stale,
                )
            payload = checkpoint / GRAPH_RESULT_FILENAME
            payload.write_bytes(payload.read_bytes() + b"corruption")
            with self.assertRaisesRegex(
                RuntimeError,
                "corrupt|non-deterministic|mismatch",
            ):
                validate_checkpoint_directory(
                    checkpoint,
                    run_manifest=self.manifest,
                )

    def test_fixture_schedule_cannot_be_resumed_as_scientific_schedule(self):
        with TemporaryDirectory(prefix="iteration2-excluded-resume-") as temporary:
            output = Path(temporary) / "run"
            (output / "graphs").mkdir(parents=True)
            (output / "run_manifest.json").write_bytes(
                _json_bytes(self.manifest)
            )
            with self.assertRaisesRegex(RuntimeError, "frozen schedule"):
                _validate_resume_directory(
                    output,
                    self.manifest,
                    validation_policy=ResumeValidationPolicy.READ_ONLY_STRUCTURAL,
                )


class Iteration2FutureRunGuardTests(unittest.TestCase):
    def test_regeneration_audit_requires_exact_confirmation_before_entry(self):
        with patch.object(
            runner,
            "_regenerate_and_validate_checkpoint_scientific_result",
        ) as regenerate:
            with self.assertRaisesRegex(
                RuntimeError,
                "exact scientific regeneration audit confirmation required",
            ):
                runner.regenerate_and_validate_checkpoint_scientific_result(
                    Path("unused-checkpoint"),
                    run_manifest={},
                    specification=object(),
                    confirmation="wrong",
                )
        regenerate.assert_not_called()

    def test_regeneration_audit_is_blocked_during_analysis(self):
        with scientific_operation_context(ANALYSIS_READ_ONLY) as ledger:
            with self.assertRaisesRegex(
                ScientificOperationBlocked,
                "scientific_regeneration_audit",
            ):
                runner.regenerate_and_validate_checkpoint_scientific_result(
                    Path("unused-checkpoint"),
                    run_manifest={},
                    specification=object(),
                    confirmation=(
                        runner.SCIENTIFIC_REGENERATION_AUDIT_CONFIRMATION
                    ),
                )
        snapshot = ledger.snapshot()
        self.assertEqual(
            snapshot["attempted_operation_counts"][
                "scientific_regeneration_audit"
            ],
            1,
        )
        self.assertEqual(
            snapshot["executed_operation_counts"][
                "scientific_regeneration_audit"
            ],
            0,
        )
        self.assertEqual(
            snapshot["blocked_operation_counts"][
                "scientific_regeneration_audit"
            ],
            1,
        )

    def test_resume_preflight_is_structural_and_executes_zero_science(self):
        results = excluded_feasibility_results()
        manifest = excluded_run_manifest(results)
        with TemporaryDirectory(prefix="iteration2-read-only-preflight-") as temporary:
            root = Path(temporary)
            output = root / "results" / str(manifest["run_identity"])
            graph_root = output / "graphs"
            graph_root.mkdir(parents=True)
            (output / "run_manifest.json").write_bytes(_json_bytes(manifest))
            for result in results:
                publish_graph_checkpoint(graph_root, result, manifest)
            before = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            specifications = {
                str(result["graph_identity"]["graph_id"]): object()
                for result in results
            }
            sentinel = AssertionError("preflight executed scientific work")
            sentinels = (
                patch.object(runner, "execute_scheduled_graph", side_effect=sentinel),
                patch.object(
                    runner,
                    "regenerate_and_validate_checkpoint_scientific_result",
                    side_effect=sentinel,
                ),
                patch("iteration2_experiment.generate_iteration2_graph", side_effect=sentinel),
                patch("iteration2_experiment.create_iteration2_embeddings", side_effect=sentinel),
                patch("iteration2_experiment.sample_ordered_pairs", side_effect=sentinel),
                patch("iteration2_experiment._execute_dijkstra_benchmark", side_effect=sentinel),
                patch("iteration2_experiment.euclidean_greedy_route_v2", side_effect=sentinel),
                patch("iteration2_experiment.poincare_greedy_route_v2", side_effect=sentinel),
                patch("iteration2_experiment.repaired_poincare_greedy_route_v2", side_effect=sentinel),
                patch.object(runner, "publish_graph_checkpoint", side_effect=sentinel),
            )
            with (
                patch.object(runner, "repository_root", return_value=root),
                patch.object(runner, "resolve_iteration2_output", return_value=output),
                patch.object(runner, "build_manifest", return_value=manifest),
                patch.object(
                    runner,
                    "_capacity_status",
                    return_value={"profile_valid": True, "disk_space_pass": True},
                ),
                patch.object(
                    runner,
                    "verify_iteration1_immutable",
                    return_value={"verified": True},
                ),
                patch.object(
                    runner,
                    "scheduled_specifications",
                    return_value=specifications,
                ),
            ):
                started = [sentinel_patch.start() for sentinel_patch in sentinels]
                try:
                    report = runner.preflight(
                        mode="full",
                        confirmation=runner.FULL_RUN_CONFIRMATION_TOKEN,
                        expected_source_commit=str(manifest["source_commit"]),
                        expected_source_fingerprint=str(manifest["source_fingerprint"]),
                        expected_dependency_fingerprint=str(
                            manifest["dependency_fingerprint"]["sha256"]
                        ),
                        expected_capacity_profile=str(
                            manifest["capacity_profile_sha256"]
                        ),
                        expected_protocol_hash=str(manifest["protocol_hash"]),
                        resume=True,
                    )
                finally:
                    for sentinel_patch in reversed(sentinels):
                        sentinel_patch.stop()
            self.assertTrue(report["authorized"])
            self.assertEqual(
                report["checkpoint_validation"]["validation_policy"],
                ResumeValidationPolicy.READ_ONLY_STRUCTURAL.value,
            )
            self.assertFalse(
                report["checkpoint_validation"]["scientific_regeneration_performed"]
            )
            ledger = report["scientific_operation_ledger"]
            self.assertEqual(ledger["total_attempted"], 0)
            self.assertEqual(ledger["total_executed"], 0)
            self.assertEqual(ledger["total_blocked"], 0)
            for mocked in started:
                mocked.assert_not_called()
            after = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_production_cli_has_no_maximum_graphs_subset_or_skip_switch(self):
        parser = _parser()
        help_text = parser.format_help().lower()
        self.assertNotIn("max-graphs", help_text)
        self.assertNotIn("subset", help_text)
        self.assertNotIn("skip", help_text)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "run",
                        "--mode",
                        "full",
                        "--confirm-full-run",
                        "invalid-on-purpose",
                        "--max-graphs",
                        "1",
                    ]
                )

    def test_repository_profile_is_untracked_and_missing_or_stale_profiles_fail(self):
        tracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                "--",
                "code/iteration2_capacity_profile.json",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
        )
        self.assertNotEqual(tracked.returncode, 0)
        with TemporaryDirectory(prefix="iteration2-capacity-guards-") as temporary:
            fixture_root = Path(temporary)
            stale = fixture_root / "stale-v1.json"
            stale.write_text(
                json.dumps(
                    {
                        "profile_schema": (
                            "greedy_routing_iteration2_capacity_profile_v1"
                        )
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                Iteration2CapacityError,
                "schema mismatch",
            ):
                load_capacity_profile(stale, root=PROJECT_ROOT)
            with self.assertRaisesRegex(
                Iteration2CapacityError,
                "profile is missing",
            ):
                load_capacity_profile(
                    fixture_root / "missing.json",
                    root=PROJECT_ROOT,
                )


if __name__ == "__main__":
    unittest.main()
