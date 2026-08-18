from __future__ import annotations

import ast
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from iteration2_v2_support import excluded_run_manifest  # noqa: E402
import run_iteration2 as runner  # noqa: E402


SUBPROCESS_HELPER = PROJECT_ROOT / "tests" / "iteration2_lease_subprocess.py"


def _wait_for_files(paths: list[Path], processes: list[subprocess.Popen[str]]) -> None:
    deadline = time.monotonic() + 60.0
    while not all(path.exists() for path in paths):
        failed = [
            process.returncode
            for path, process in zip(paths, processes, strict=True)
            if process.poll() is not None and not path.exists()
        ]
        if failed:
            raise AssertionError(f"lease subprocess exited before signaling: {failed}")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for lease subprocesses")
        time.sleep(0.02)


class Iteration2RunLeaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = runner.excluded_feasibility_results()
        cls.manifest = excluded_run_manifest(cls.results)
        cls.run_identity = str(cls.manifest["run_identity"])

    def _run_process_contest(
        self,
        root: Path,
        *,
        hash_seeds: tuple[str, str],
        crash_loser: bool = False,
    ) -> dict[str, bytes]:
        self.assertFalse((root / "results").exists())
        start = root / "start"
        release = root / "release"
        role_paths = [root / f"role-{index}" for index in range(2)]
        status_paths = [root / f"status-{index}.json" for index in range(2)]
        processes: list[subprocess.Popen[str]] = []
        for index, seed in enumerate(hash_seeds):
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONHASHSEED"] = seed
            command = [
                sys.executable,
                "-B",
                str(SUBPROCESS_HELPER),
                "contest",
                "--root",
                str(root),
                "--run-identity",
                self.run_identity,
                "--start",
                str(start),
                "--release",
                str(release),
                "--role",
                str(role_paths[index]),
                "--status",
                str(status_paths[index]),
            ]
            if crash_loser:
                command.append("--crash-on-loss")
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        try:
            start.write_text("go", encoding="utf-8")
            _wait_for_files(role_paths, processes)
            roles = [path.read_text(encoding="utf-8") for path in role_paths]
            self.assertEqual(sorted(roles), ["acquired", "lost"])
            release.write_text("release", encoding="utf-8")
            outputs = [process.communicate(timeout=60.0) for process in processes]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10.0)
        return_codes = [process.returncode for process in processes]
        self.assertEqual(
            sorted(return_codes),
            [0, 42] if crash_loser else [0, 23],
            outputs,
        )
        statuses = [
            json.loads(path.read_text(encoding="utf-8")) for path in status_paths
        ]
        owner = next(item for item in statuses if item["outcome"] == "acquired")
        loser = next(item for item in statuses if item["outcome"] == "lost")
        self.assertFalse(owner["output_existed_at_acquisition"])
        self.assertEqual(
            owner["ledger"]["executed_operation_counts"][
                "excluded_fixture_execution"
            ],
            1,
        )
        self.assertEqual(loser["ledger"]["total_attempted"], 0)
        self.assertEqual(loser["ledger"]["total_executed"], 0)
        self.assertEqual(loser["ledger"]["total_blocked"], 0)
        self.assertEqual(
            loser["error"]["code"],
            runner.RUN_LEASE_ERROR_CODE,
        )
        graph_root = root / "results" / self.run_identity / "graphs"
        self.assertEqual(
            {path.name for path in graph_root.iterdir()},
            set(self.manifest["schedule"]),
        )
        self.assertFalse(
            any(path.name.startswith(".") for path in graph_root.iterdir())
        )
        for graph_id in self.manifest["schedule"]:
            runner.validate_checkpoint_directory(
                graph_root / str(graph_id),
                run_manifest=self.manifest,
            )
        return {
            path.relative_to(graph_root).as_posix(): path.read_bytes()
            for path in graph_root.rglob("*")
            if path.is_file()
        }

    def test_two_processes_have_one_owner_and_zero_science_loser_across_hash_seeds(self):
        with TemporaryDirectory(prefix="iteration2-lease-process-a-") as first:
            first_files = self._run_process_contest(
                Path(first),
                hash_seeds=("1", "987654321"),
            )
        with TemporaryDirectory(prefix="iteration2-lease-process-b-") as second:
            second_files = self._run_process_contest(
                Path(second),
                hash_seeds=("987654321", "1"),
            )
        self.assertEqual(set(first_files), set(second_files))

    def test_losing_process_crash_does_not_disturb_owner_or_later_acquisition(self):
        with TemporaryDirectory(prefix="iteration2-lease-loser-crash-") as temporary:
            root = Path(temporary)
            self.assertFalse((root / "results").exists())
            self._run_process_contest(
                root,
                hash_seeds=("17", "9001"),
                crash_loser=True,
            )
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="post-loser-crash-owner",
                resume=True,
            ):
                pass

    def test_threads_cannot_bypass_in_process_registry(self):
        with TemporaryDirectory(prefix="iteration2-lease-thread-") as temporary:
            root = Path(temporary)
            outcomes: list[object] = []
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="thread-owner",
                resume=False,
            ):
                def contend() -> None:
                    try:
                        with runner.acquire_iteration2_run_lease(
                            root=root,
                            run_identity=self.run_identity,
                            source_commit="thread-loser",
                            resume=True,
                        ):
                            outcomes.append("unexpected-acquisition")
                    except BaseException as exc:
                        outcomes.append(exc)

                thread = Thread(target=contend)
                thread.start()
                thread.join(timeout=10.0)
                self.assertFalse(thread.is_alive())
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], runner.Iteration2RunAlreadyActive)

    def test_normal_and_exceptional_release_allow_later_owner(self):
        with TemporaryDirectory(prefix="iteration2-lease-release-") as temporary:
            root = Path(temporary)
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="normal-owner",
                resume=False,
            ):
                pass
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="later-owner",
                resume=True,
            ):
                pass
            with self.assertRaisesRegex(RuntimeError, "forced owner failure"):
                with runner.acquire_iteration2_run_lease(
                    root=root,
                    run_identity=self.run_identity,
                    source_commit="exception-owner",
                    resume=False,
                ):
                    raise RuntimeError("forced owner failure")
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="post-exception-owner",
                resume=True,
            ):
                pass
            with self.assertRaises(KeyboardInterrupt):
                with runner.acquire_iteration2_run_lease(
                    root=root,
                    run_identity=self.run_identity,
                    source_commit="interrupt-owner",
                    resume=False,
                ):
                    raise KeyboardInterrupt
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="post-interrupt-owner",
                resume=True,
            ):
                pass

    def test_forced_process_termination_releases_os_lock_and_replaces_metadata(self):
        with TemporaryDirectory(prefix="iteration2-lease-termination-") as temporary:
            root = Path(temporary)
            ready = root / "ready"
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(SUBPROCESS_HELPER),
                    "hold",
                    "--root",
                    str(root),
                    "--run-identity",
                    self.run_identity,
                    "--ready",
                    str(ready),
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                _wait_for_files([ready], [process])
                terminated_pid = int(ready.read_text(encoding="utf-8"))
                process.kill()
                process.communicate(timeout=20.0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10.0)
            self.assertNotEqual(process.returncode, 0)
            lock_path = runner.iteration2_run_lease_path(root, self.run_identity)
            stale = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(stale["pid"], terminated_pid)
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="post-termination-owner",
                resume=True,
            ) as lease:
                replacement = lease.metadata
                self.assertEqual(replacement["pid"], os.getpid())
                self.assertTrue(replacement["resume"])
            self.assertEqual(
                json.loads(lock_path.read_text(encoding="utf-8")),
                replacement,
            )
            self.assertFalse(
                (root / "results" / runner.ITERATION2_RUN_IDENTITY).exists()
            )

    def test_malformed_and_stale_metadata_are_replaced_only_after_lock(self):
        with TemporaryDirectory(prefix="iteration2-lease-metadata-") as temporary:
            root = Path(temporary)
            lock_path = runner.iteration2_run_lease_path(root, self.run_identity)
            lock_path.parent.mkdir()
            lock_path.write_bytes(b"not-json")
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="metadata-owner",
                resume=False,
            ) as lease:
                metadata = lease.metadata
                self.assertEqual(
                    set(metadata),
                    {
                        "schema",
                        "version",
                        "run_identity",
                        "pid",
                        "hostname",
                        "source_commit",
                        "acquired_at_utc",
                        "resume",
                    },
                )
            self.assertEqual(
                json.loads(lock_path.read_text(encoding="utf-8")),
                metadata,
            )
            stale = dict(metadata)
            stale["pid"] = os.getpid()
            lock_path.write_text(json.dumps(stale), encoding="utf-8")
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="stale-replacement-owner",
                resume=True,
            ) as lease:
                replacement = lease.metadata
                self.assertEqual(replacement["pid"], os.getpid())
                self.assertEqual(
                    replacement["source_commit"],
                    "stale-replacement-owner",
                )
            self.assertEqual(
                json.loads(lock_path.read_text(encoding="utf-8")),
                replacement,
            )

    def test_permission_failure_and_dangling_reparse_release_registry_claim(self):
        with TemporaryDirectory(prefix="iteration2-lease-open-failure-") as temporary:
            root = Path(temporary)
            lock_path = runner.iteration2_run_lease_path(root, self.run_identity)
            with patch.object(
                runner,
                "_open_run_lease_file",
                side_effect=PermissionError("forced lease permission failure"),
            ):
                with self.assertRaisesRegex(PermissionError, "forced lease permission"):
                    with runner.acquire_iteration2_run_lease(
                        root=root,
                        run_identity=self.run_identity,
                        source_commit="permission-failure",
                        resume=False,
                    ):
                        self.fail("permission failure acquired a lease")

            with (
                patch.object(
                    runner,
                    "_path_is_reparse_point",
                    side_effect=lambda path: path == lock_path,
                ),
                patch.object(runner, "_open_run_lease_file") as open_lease,
            ):
                with self.assertRaisesRegex(RuntimeError, "lease file is unsafe"):
                    with runner.acquire_iteration2_run_lease(
                        root=root,
                        run_identity=self.run_identity,
                        source_commit="dangling-reparse",
                        resume=False,
                    ):
                        self.fail("dangling reparse acquired a lease")
                open_lease.assert_not_called()
            self.assertFalse(lock_path.exists())

            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=self.run_identity,
                source_commit="post-failure-owner",
                resume=True,
            ):
                pass

    def test_initial_and_resume_use_identical_lease_before_execution_body(self):
        with TemporaryDirectory(prefix="iteration2-lease-order-") as temporary:
            root = Path(temporary)
            events: list[tuple[object, ...]] = []

            @contextmanager
            def fake_lease(**kwargs):
                events.append(
                    (
                        "acquire",
                        kwargs["root"],
                        kwargs["run_identity"],
                        kwargs["resume"],
                    )
                )
                try:
                    yield object()
                finally:
                    events.append(("release", kwargs["resume"]))

            def body(**kwargs):
                self.assertEqual(events[-1][0], "acquire")
                events.append(("body", kwargs["resume"]))
                return 360

            with (
                patch.object(runner, "repository_root", return_value=root),
                patch.object(runner, "_git_state", return_value=("commit", True)),
                patch.object(
                    runner,
                    "acquire_iteration2_run_lease",
                    side_effect=fake_lease,
                ) as acquire,
                patch.object(
                    runner,
                    "_execute_full_run_with_lease_held",
                    side_effect=body,
                ),
            ):
                for resume in (False, True):
                    self.assertEqual(
                        runner.execute_full_run(
                            mode="full",
                            confirmation="confirmation",
                            expected_source_commit="commit",
                            expected_source_fingerprint="source",
                            expected_dependency_fingerprint="dependency",
                            expected_capacity_profile="capacity",
                            expected_protocol_hash="protocol",
                            resume=resume,
                        ),
                        360,
                    )
            self.assertEqual(acquire.call_count, 2)
            first, second = [call.kwargs for call in acquire.call_args_list]
            self.assertEqual(first["root"], second["root"])
            self.assertEqual(first["run_identity"], second["run_identity"])
            self.assertFalse(first["resume"])
            self.assertTrue(second["resume"])
            self.assertEqual(
                events,
                [
                    ("acquire", root, runner.ITERATION2_RUN_IDENTITY, False),
                    ("body", False),
                    ("release", False),
                    ("acquire", root, runner.ITERATION2_RUN_IDENTITY, True),
                    ("body", True),
                    ("release", True),
                ],
            )

    def test_private_production_body_has_only_the_leased_public_caller(self):
        module = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        callers = []
        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_execute_full_run_with_lease_held"
                for call in ast.walk(node)
            ):
                callers.append(node.name)
        self.assertEqual(callers, ["execute_full_run"])

    def test_preflight_creates_no_results_or_lease_and_executes_zero_science(self):
        with TemporaryDirectory(prefix="iteration2-lease-preflight-") as temporary:
            root = Path(temporary)
            output = root / "results" / runner.ITERATION2_RUN_IDENTITY
            manifest = self.manifest
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
            ):
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
                    resume=False,
                )
            self.assertTrue(report["authorized"])
            self.assertEqual(
                report["scientific_operation_ledger"]["total_attempted"],
                0,
            )
            self.assertFalse((root / "results").exists())
            self.assertFalse(output.exists())

    def test_lock_path_containment_and_reparse_rejection(self):
        with TemporaryDirectory(prefix="iteration2-lease-path-") as temporary:
            root = Path(temporary)
            for unsafe in ("../escape", "nested/run", "nested\\run", "."):
                with self.assertRaises(ValueError):
                    runner.iteration2_run_lease_path(root, unsafe)
            results = root / "results"
            results.mkdir()
            with patch.object(
                runner,
                "_path_is_reparse_point",
                side_effect=lambda path: path == results,
            ):
                with self.assertRaisesRegex(RuntimeError, "parent is unsafe"):
                    with runner.acquire_iteration2_run_lease(
                        root=root,
                        run_identity=self.run_identity,
                        source_commit="unsafe-parent",
                        resume=False,
                    ):
                        self.fail("unsafe reparse-point parent acquired a lease")
            lock_path = runner.iteration2_run_lease_path(root, self.run_identity)
            lock_path.write_text("ordinary-file-before-mock", encoding="utf-8")
            with patch.object(
                runner,
                "_path_is_reparse_point",
                side_effect=lambda path: path == lock_path,
            ):
                with self.assertRaisesRegex(RuntimeError, "lease file is unsafe"):
                    with runner.acquire_iteration2_run_lease(
                        root=root,
                        run_identity=self.run_identity,
                        source_commit="unsafe-file",
                        resume=False,
                    ):
                        self.fail("unsafe reparse-point file acquired a lease")

    def test_physical_lock_file_symlink_is_rejected_when_supported(self):
        with TemporaryDirectory(prefix="iteration2-lease-symlink-") as temporary:
            root = Path(temporary)
            lock_path = runner.iteration2_run_lease_path(root, self.run_identity)
            lock_path.parent.mkdir()
            target = root / "unrelated-target"
            target.write_text("unchanged", encoding="utf-8")
            try:
                lock_path.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"file symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(RuntimeError, "lease file is unsafe"):
                with runner.acquire_iteration2_run_lease(
                    root=root,
                    run_identity=self.run_identity,
                    source_commit="symlink-rejection",
                    resume=False,
                ):
                    self.fail("symlink lock file acquired a lease")
            self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")

    def test_cli_run_active_error_is_structured_and_nonzero(self):
        lock_path = Path("results") / ".excluded.lease"
        error = runner.Iteration2RunAlreadyActive(
            run_identity=self.run_identity,
            lock_path=lock_path,
        )
        output = io.StringIO()
        with (
            patch.object(runner, "execute_full_run", side_effect=error),
            redirect_stdout(output),
        ):
            return_code = runner.main(
                [
                    "run",
                    "--mode",
                    "full",
                    "--confirm-full-run",
                    "intentionally-not-executed",
                ]
            )
        self.assertEqual(return_code, 3)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["authorized"])
        self.assertEqual(payload["error"]["code"], runner.RUN_LEASE_ERROR_CODE)


if __name__ == "__main__":
    unittest.main()
