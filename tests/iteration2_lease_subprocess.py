"""Excluded subprocess helper for Iteration 2 lease regression tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from iteration2_runtime_guard import (  # noqa: E402
    SCIENTIFIC_EXECUTION,
    scientific_operation_context,
)
from iteration2_v2_support import excluded_run_manifest  # noqa: E402
import run_iteration2 as runner  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _wait_for(path: Path, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.01)


def _contest(args: argparse.Namespace) -> int:
    root = Path(args.root)
    role_path = Path(args.role)
    status_path = Path(args.status)
    _wait_for(Path(args.start))
    with scientific_operation_context(SCIENTIFIC_EXECUTION) as ledger:
        try:
            with runner.acquire_iteration2_run_lease(
                root=root,
                run_identity=args.run_identity,
                source_commit="excluded-subprocess-fixture",
                resume=False,
            ):
                output_existed_at_acquisition = (
                    root / "results" / args.run_identity
                ).exists()
                role_path.write_text("acquired", encoding="utf-8")
                results = runner.excluded_feasibility_results()
                manifest = excluded_run_manifest(results)
                if manifest["run_identity"] != args.run_identity:
                    raise RuntimeError("excluded lease run identity mismatch")
                graph_root = root / "results" / args.run_identity / "graphs"
                graph_root.mkdir(parents=True)
                for result in results:
                    published = runner.publish_graph_checkpoint(
                        graph_root,
                        result,
                        manifest,
                    )
                    runner.validate_checkpoint_directory(
                        published,
                        run_manifest=manifest,
                    )
                _wait_for(Path(args.release))
                _write_json(
                    status_path,
                    {
                        "outcome": "acquired",
                        "pid": os.getpid(),
                        "output_existed_at_acquisition": (
                            output_existed_at_acquisition
                        ),
                        "ledger": ledger.snapshot(),
                    },
                )
                return 0
        except runner.Iteration2RunAlreadyActive as exc:
            role_path.write_text("lost", encoding="utf-8")
            _write_json(
                status_path,
                {
                    "outcome": "lost",
                    "pid": os.getpid(),
                    "error": exc.as_dict(),
                    "ledger": ledger.snapshot(),
                },
            )
            if args.crash_on_loss:
                os._exit(42)
            return 23


def _hold(args: argparse.Namespace) -> int:
    with runner.acquire_iteration2_run_lease(
        root=Path(args.root),
        run_identity=args.run_identity,
        source_commit="excluded-forced-termination-fixture",
        resume=True,
    ):
        Path(args.ready).write_text(str(os.getpid()), encoding="utf-8")
        while True:
            time.sleep(1.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    contest = subparsers.add_parser("contest")
    contest.add_argument("--root", required=True)
    contest.add_argument("--run-identity", required=True)
    contest.add_argument("--start", required=True)
    contest.add_argument("--release", required=True)
    contest.add_argument("--role", required=True)
    contest.add_argument("--status", required=True)
    contest.add_argument("--crash-on-loss", action="store_true")
    hold = subparsers.add_parser("hold")
    hold.add_argument("--root", required=True)
    hold.add_argument("--run-identity", required=True)
    hold.add_argument("--ready", required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    if arguments.operation == "contest":
        raise SystemExit(_contest(arguments))
    raise SystemExit(_hold(arguments))
