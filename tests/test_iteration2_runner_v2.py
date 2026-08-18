from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from benchmark_iteration2_capacity import (  # noqa: E402
    Iteration2CapacityError,
    dependency_fingerprint,
    integrity_report,
    load_capacity_profile,
    performance_source_fingerprint,
    performance_source_manifest,
    physical_profile_sha256,
    profile_sha256,
)
from iteration2_config import COMBINED_PROTOCOL_HASH  # noqa: E402
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


def _create_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise unittest.SkipTest(
                "Windows directory junctions are unavailable: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        raise unittest.SkipTest(f"directory symlinks are unavailable: {exc}") from exc


def _remove_directory_link(link: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif getattr(link, "is_junction", lambda: False)():
        link.rmdir()


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

    def test_truncated_corrupt_and_noncanonical_gzip_and_completion_are_rejected(self):
        mutations = ("truncated", "corrupt", "noncanonical")
        for mutation in mutations:
            with self.subTest(mutation=mutation), TemporaryDirectory(
                prefix=f"iteration2-excluded-{mutation}-"
            ) as temporary:
                graph_root = Path(temporary) / "graphs"
                graph_root.mkdir()
                checkpoint = publish_graph_checkpoint(
                    graph_root,
                    self.results[0],
                    self.manifest,
                )
                payload = checkpoint / GRAPH_RESULT_FILENAME
                physical = payload.read_bytes()
                if mutation == "truncated":
                    payload.write_bytes(physical[:-7])
                elif mutation == "corrupt":
                    changed = bytearray(physical)
                    changed[len(changed) // 2] ^= 0xFF
                    payload.write_bytes(changed)
                else:
                    payload.write_bytes(
                        runner.gzip.compress(
                            runner._json_bytes(self.results[0]),
                            mtime=1,
                        )
                    )
                with self.assertRaisesRegex(
                    (EOFError, OSError, RuntimeError),
                    "[Cc]ompress|corrupt|gzip|deterministic|mismatch|CRC|end-of-stream",
                ):
                    validate_checkpoint_directory(
                        checkpoint,
                        run_manifest=self.manifest,
                    )

        with TemporaryDirectory(prefix="iteration2-excluded-completion-") as temporary:
            graph_root = Path(temporary) / "graphs"
            graph_root.mkdir()
            checkpoint = publish_graph_checkpoint(
                graph_root,
                self.results[0],
                self.manifest,
            )
            (checkpoint / GRAPH_COMPLETION_FILENAME).write_bytes(b"{")
            with self.assertRaisesRegex(RuntimeError, "JSON|completion|invalid"):
                validate_checkpoint_directory(
                    checkpoint,
                    run_manifest=self.manifest,
                )

    def test_concurrent_publication_collision_cleans_losing_staging_directory(self):
        with TemporaryDirectory(prefix="iteration2-excluded-collision-") as temporary:
            graph_root = Path(temporary) / "graphs"
            graph_root.mkdir()
            unrelated = graph_root / "unrelated.txt"
            unrelated.write_text("must remain untouched", encoding="utf-8")
            barrier = Barrier(2)
            real_replace = runner.os.replace
            outcomes: list[object] = []

            def synchronized_replace(source: Path, target: Path) -> None:
                barrier.wait(timeout=10.0)
                real_replace(source, target)

            def publish() -> None:
                try:
                    outcomes.append(
                        publish_graph_checkpoint(
                            graph_root,
                            self.results[0],
                            self.manifest,
                        )
                    )
                except BaseException as exc:  # Capture the losing publisher.
                    outcomes.append(exc)

            with patch.object(runner.os, "replace", side_effect=synchronized_replace):
                threads = [Thread(target=publish) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=15.0)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(sum(isinstance(item, Path) for item in outcomes), 1)
            self.assertEqual(sum(isinstance(item, BaseException) for item in outcomes), 1)
            checkpoint = graph_root / str(
                self.results[0]["graph_identity"]["graph_id"]
            )
            validate_checkpoint_directory(checkpoint, run_manifest=self.manifest)
            self.assertEqual(
                unrelated.read_text(encoding="utf-8"),
                "must remain untouched",
            )
            self.assertEqual(
                [path.name for path in graph_root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_serialization_failure_cleans_only_owned_staging_directory(self):
        with TemporaryDirectory(prefix="iteration2-excluded-cleanup-") as temporary:
            graph_root = Path(temporary) / "graphs"
            graph_root.mkdir()
            unrelated = graph_root / "unrelated.txt"
            unrelated.write_text("preserved", encoding="utf-8")
            with patch.object(
                runner,
                "_construct_raw_checkpoint_payload",
                side_effect=RuntimeError("forced serialization failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced serialization"):
                    publish_graph_checkpoint(
                        graph_root,
                        self.results[0],
                        self.manifest,
                    )
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "preserved")
            self.assertEqual(
                [path.name for path in graph_root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_each_publication_phase_failure_is_clean_or_atomically_recoverable(self):
        for failing_write in (1, 2, 3):
            with self.subTest(failing_write=failing_write), TemporaryDirectory(
                prefix="iteration2-excluded-phase-write-"
            ) as temporary:
                graph_root = Path(temporary) / "graphs"
                graph_root.mkdir()
                unrelated = graph_root / "unrelated.txt"
                unrelated.write_text("preserved", encoding="utf-8")
                real_write = runner._write_new
                write_count = 0

                def interrupt_after_write(path: Path, payload: bytes) -> None:
                    nonlocal write_count
                    real_write(path, payload)
                    write_count += 1
                    if write_count == failing_write:
                        raise KeyboardInterrupt

                with patch.object(
                    runner,
                    "_write_new",
                    side_effect=interrupt_after_write,
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        publish_graph_checkpoint(
                            graph_root,
                            self.results[0],
                            self.manifest,
                        )
                self.assertEqual(unrelated.read_text(encoding="utf-8"), "preserved")
                self.assertEqual(
                    [path.name for path in graph_root.iterdir() if path.name.startswith(".")],
                    [],
                )
                published = publish_graph_checkpoint(
                    graph_root,
                    self.results[0],
                    self.manifest,
                )
                validate_checkpoint_directory(published, run_manifest=self.manifest)

        with TemporaryDirectory(prefix="iteration2-excluded-rename-failure-") as temporary:
            graph_root = Path(temporary) / "graphs"
            graph_root.mkdir()
            unrelated = graph_root / "unrelated.txt"
            unrelated.write_text("preserved", encoding="utf-8")
            with patch.object(
                runner.os,
                "replace",
                side_effect=PermissionError("forced atomic rename failure"),
            ):
                with self.assertRaisesRegex(PermissionError, "forced atomic rename"):
                    publish_graph_checkpoint(
                        graph_root,
                        self.results[0],
                        self.manifest,
                    )
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "preserved")
            self.assertEqual(
                [path.name for path in graph_root.iterdir() if path.name.startswith(".")],
                [],
            )
            self.assertFalse(
                (graph_root / str(self.results[0]["graph_identity"]["graph_id"])).exists()
            )

        with TemporaryDirectory(prefix="iteration2-excluded-post-rename-") as temporary:
            graph_root = Path(temporary) / "graphs"
            graph_root.mkdir()
            real_fsync = runner._fsync_directory
            fsync_count = 0

            def interrupt_after_rename(path: Path) -> None:
                nonlocal fsync_count
                fsync_count += 1
                if fsync_count == 2:
                    raise KeyboardInterrupt
                real_fsync(path)

            with patch.object(
                runner,
                "_fsync_directory",
                side_effect=interrupt_after_rename,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    publish_graph_checkpoint(
                        graph_root,
                        self.results[0],
                        self.manifest,
                    )
            checkpoint = graph_root / str(
                self.results[0]["graph_identity"]["graph_id"]
            )
            validate_checkpoint_directory(checkpoint, run_manifest=self.manifest)
            self.assertEqual(
                [path.name for path in graph_root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_resume_rejects_noncontiguous_prefix_and_premature_completion(self):
        with TemporaryDirectory(prefix="iteration2-excluded-prefix-gap-") as temporary:
            output = Path(temporary) / "run"
            graph_root = output / "graphs"
            graph_root.mkdir(parents=True)
            (output / "run_manifest.json").write_bytes(_json_bytes(self.manifest))
            publish_graph_checkpoint(graph_root, self.results[1], self.manifest)
            with (
                patch.object(
                    runner,
                    "scheduled_specifications",
                    return_value={graph_id: object() for graph_id in self.manifest["schedule"]},
                ),
                self.assertRaisesRegex(RuntimeError, "contiguous schedule prefix"),
            ):
                _validate_resume_directory(
                    output,
                    self.manifest,
                    validation_policy=ResumeValidationPolicy.READ_ONLY_STRUCTURAL,
                )

        with TemporaryDirectory(prefix="iteration2-excluded-premature-complete-") as temporary:
            output = Path(temporary) / "run"
            (output / "graphs").mkdir(parents=True)
            (output / "run_manifest.json").write_bytes(_json_bytes(self.manifest))
            (output / "run_complete.json").write_bytes(b"{")
            with (
                patch.object(
                    runner,
                    "scheduled_specifications",
                    return_value={graph_id: object() for graph_id in self.manifest["schedule"]},
                ),
                self.assertRaisesRegex(RuntimeError, "completion marker"),
            ):
                _validate_resume_directory(
                    output,
                    self.manifest,
                    validation_policy=ResumeValidationPolicy.READ_ONLY_STRUCTURAL,
                )

    def test_execution_resumes_at_first_missing_graph_without_duplicate_execution(self):
        with TemporaryDirectory(prefix="iteration2-excluded-resume-next-") as temporary:
            root = Path(temporary)
            output = root / "results" / str(self.manifest["run_identity"])
            graph_root = output / "graphs"
            graph_root.mkdir(parents=True)
            (output / "run_manifest.json").write_bytes(_json_bytes(self.manifest))
            first_id, second_id = [str(item) for item in self.manifest["schedule"]]
            (graph_root / first_id).mkdir()
            specifications = (
                SimpleNamespace(graph_id=first_id),
                SimpleNamespace(graph_id=second_id),
            )
            report = {
                "authorized": True,
                "manifest": self.manifest,
                "checkpoint_validation": {
                    "validated_graph_ids": [first_id],
                },
            }
            sentinel = RuntimeError("stop after proving next graph")
            with (
                patch.object(runner, "preflight", return_value=report),
                patch.object(runner, "_recheck_execution_authorization"),
                patch.object(runner, "repository_root", return_value=root),
                patch.object(runner, "resolve_iteration2_output", return_value=output),
                patch.object(runner, "full_schedule", return_value=specifications),
                patch.object(
                    runner,
                    "execute_scheduled_graph",
                    side_effect=sentinel,
                ) as execute,
                patch.object(runner, "publish_graph_checkpoint") as publish,
                self.assertRaisesRegex(RuntimeError, "proving next graph"),
            ):
                runner._execute_full_run_with_lease_held(
                    mode="full",
                    confirmation="confirmed",
                    expected_source_commit="commit",
                    expected_source_fingerprint="source",
                    expected_dependency_fingerprint="dependency",
                    expected_capacity_profile="capacity",
                    expected_protocol_hash="protocol",
                    resume=True,
                )
            execute.assert_called_once_with(specifications[1], pair_count=runner.PAIRS_PER_GRAPH)
            publish.assert_not_called()

    def test_publication_and_validation_reject_directory_links_without_escape(self):
        with TemporaryDirectory(prefix="iteration2-excluded-link-escape-") as temporary:
            root = Path(temporary)
            physical_graph_root = root / "physical-graphs"
            physical_graph_root.mkdir()
            linked_graph_root = root / "linked-graphs"
            _create_directory_link(linked_graph_root, physical_graph_root)
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "unsafe|reparse|symlink|junction",
                ):
                    publish_graph_checkpoint(
                        linked_graph_root,
                        self.results[0],
                        self.manifest,
                    )
                self.assertEqual(list(physical_graph_root.iterdir()), [])
            finally:
                _remove_directory_link(linked_graph_root)

            checkpoint = publish_graph_checkpoint(
                physical_graph_root,
                self.results[0],
                self.manifest,
            )
            linked_checkpoint = root / "linked-checkpoint"
            _create_directory_link(linked_checkpoint, checkpoint)
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "non-symlink|reparse|junction",
                ):
                    validate_checkpoint_directory(
                        linked_checkpoint,
                        run_manifest=self.manifest,
                    )
            finally:
                _remove_directory_link(linked_checkpoint)

    def test_output_resolution_and_resume_reject_directory_links(self):
        with TemporaryDirectory(prefix="iteration2-excluded-resume-link-") as temporary:
            root = Path(temporary)
            results_root = root / "results"
            results_root.mkdir()
            physical_output = root / "physical-output"
            graph_root = physical_output / "graphs"
            graph_root.mkdir(parents=True)
            (physical_output / "run_manifest.json").write_bytes(
                _json_bytes(self.manifest)
            )
            for result in self.results:
                publish_graph_checkpoint(graph_root, result, self.manifest)
            linked_output = results_root / runner.ITERATION2_RUN_IDENTITY
            _create_directory_link(linked_output, physical_output)
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "unsafe|reparse|symlink|junction",
                ):
                    runner.resolve_iteration2_output(
                        root,
                        runner.ITERATION2_RUN_IDENTITY,
                    )
                with (
                    patch.object(
                        runner,
                        "scheduled_specifications",
                        return_value={graph_id: object() for graph_id in self.manifest["schedule"]},
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "non-symlink|reparse|junction",
                    ),
                ):
                    _validate_resume_directory(
                        linked_output,
                        self.manifest,
                        validation_policy=ResumeValidationPolicy.READ_ONLY_STRUCTURAL,
                    )
            finally:
                _remove_directory_link(linked_output)

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

    def test_repository_profile_is_tracked_valid_and_invalid_fixtures_fail(self):
        profile_path = PROJECT_ROOT / "code" / "iteration2_capacity_profile.json"
        self.assertTrue(profile_path.is_file())
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
        self.assertEqual(tracked.returncode, 0, tracked.stderr.decode())
        self.assertEqual(
            tracked.stdout.decode().strip().replace("\\", "/"),
            "code/iteration2_capacity_profile.json",
        )
        profile = load_capacity_profile(profile_path, root=PROJECT_ROOT)
        report = integrity_report(profile_path)
        self.assertTrue(report["valid"])
        self.assertEqual(profile["profile_sha256"], profile_sha256(profile))
        self.assertEqual(report["internal_sha256"], profile["profile_sha256"])
        self.assertEqual(
            report["physical_sha256"],
            physical_profile_sha256(profile_path),
        )
        self.assertEqual(profile["protocol_hash"], COMBINED_PROTOCOL_HASH)
        self.assertEqual(
            profile["dependency_fingerprint"],
            dependency_fingerprint(PROJECT_ROOT),
        )
        self.assertEqual(
            profile["performance_source_manifest"],
            performance_source_manifest(PROJECT_ROOT),
        )
        self.assertEqual(
            profile["performance_source_fingerprint"],
            performance_source_fingerprint(PROJECT_ROOT),
        )

        with TemporaryDirectory(prefix="iteration2-capacity-guards-") as temporary:
            fixture_root = Path(temporary)
            stale = fixture_root / "stale-source-manifest.json"
            stale_profile = deepcopy(profile)
            stale_profile["performance_source_manifest"]["files"][
                "requirements.txt"
            ] = "0" * 64
            stale_profile["profile_sha256"] = profile_sha256(stale_profile)
            stale.write_text(
                json.dumps(
                    stale_profile,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                Iteration2CapacityError,
                "performance source manifest mismatch",
            ):
                load_capacity_profile(stale, root=PROJECT_ROOT)

            corrupt = fixture_root / "corrupt.json"
            corrupt.write_text('{"profile_schema":', encoding="utf-8")
            with self.assertRaisesRegex(
                Iteration2CapacityError,
                "profile JSON is invalid",
            ):
                load_capacity_profile(corrupt, root=PROJECT_ROOT)

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
