"""Focused tests for deterministic atomic Step 14 checkpoints."""

from __future__ import annotations

from copy import deepcopy
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import experiment_checkpoint as checkpoint_module  # noqa: E402
from experiment_checkpoint import (  # noqa: E402
    CHECKPOINT_DIRECTORY,
    COMPLETE_MARKER_FILENAME,
    ERROR_REPORT_FILENAME,
    FULL_COORDINATE_CONDITION_IDS,
    FULL_DISTORTION_CONDITION_IDS,
    PUBLICATION_TIMING_DEFINITIONS,
    PUBLICATION_TIMING_DIRECTORY,
    RESULT_SCHEMA_VERSION,
    RUN_MANIFEST_FILENAME,
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointError,
    GraphCheckpointData,
    audit_run_checkpoints,
    deterministic_payload_hashes,
    decode_json_value,
    preserve_graph_error,
    publish_graph_checkpoint,
    run_manifest_sha256,
    validate_graph_checkpoint,
    validate_publication_timing_record,
    validate_run_manifest_compatibility,
    write_run_manifest_once,
)


def fixture_manifest(
    output_root: Path,
    *,
    graph_ids: tuple[str, ...] = ("fixture_graph",),
    profile: str = "development_fixture",
) -> dict[str, object]:
    return {
        "manifest_schema": "test_manifest",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "configuration_schema_version": 4,
        "seed_identity_version": 3,
        "data_generation_hash": "1" * 64,
        "analysis_plan_hash": "2" * 64,
        "combined_freeze_hash": "3" * 64,
        "git_commit_hash": "4" * 40,
        "git_working_tree": "clean",
        "source_fingerprint": "5" * 64,
        "python_version": "3.14.0",
        "dependency_versions": {"networkx": "3.6.1", "numpy": "2.4.6"},
        "operating_system": "test-os",
        "hardware": {"machine": "test", "processor": "test", "cpu_count": 1},
        "output_schema": {"id": "test", "version": RESULT_SCHEMA_VERSION},
        "execution_profile": profile,
        "execution_model": "single_process_sequential_per_graph",
        "run_directory_name": "fixture_run",
        "schedule": list(graph_ids),
        "workload": {"graph_replicates": len(graph_ids)},
        "output_root": str(output_root.resolve()),
        "timer": "time.perf_counter_ns",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
    }


def route_record(
    graph_id: str,
    pair_index: int,
    source: int,
    destination: int,
) -> dict[str, object]:
    return {
        "graph_id": graph_id,
        "pair_index": pair_index,
        "pair_id": f"{graph_id}:pair:{pair_index:04d}",
        "source": source,
        "destination": destination,
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
        "runtime_ns": 17,
        "walk": [source, destination],
        "forwarding_decisions": 1,
    }


def fixture_data(
    manifest: dict[str, object],
    *,
    graph_id: str = "fixture_graph",
    timing: int = 10,
) -> GraphCheckpointData:
    pairs = ((0, 1), (1, 2))
    return GraphCheckpointData(
        graph_id=graph_id,
        generation_metadata={
            "graph_id": graph_id,
            "graph_seed": 7,
            "generation_attempt_seeds": [7],
            "p": 0.5,
        },
        edges=((0, 1), (1, 2)),
        network_metrics={
            "number_of_vertices": 3,
            "number_of_edges": 2,
            "average_degree": 4.0 / 3.0,
        },
        pairs=pairs,
        coordinates={
            "hydra": {
                0: (0.0, 0.0),
                1: (0.25, 0.0),
                2: (0.0, 0.25),
            }
        },
        embedding_metadata={
            "hydra": {"effective_rank": 2},
            "mds_base": {"effective_rank": 2},
        },
        distortions=(
            {
                "metric_condition_id": "hydra_euclidean",
                "fitted_scale_alpha": 1.25,
            },
        ),
        dijkstra_records=tuple(
            {
                "graph_id": graph_id,
                "pair_index": index,
                "pair_id": f"{graph_id}:pair:{index:04d}",
                "source": source,
                "destination": destination,
                "method_id": "dijkstra",
                "success": True,
                "route_length": 1,
                "apsp_length": 1,
                "apsp_agreement": True,
                "runtime_ns": 11,
                "walk": [source, destination],
            }
            for index, (source, destination) in enumerate(pairs)
        ),
        route_records=tuple(
            route_record(graph_id, index, source, destination)
            for index, (source, destination) in enumerate(pairs)
        ),
        timings={"graph_generation_ns": timing},
        run_manifest=manifest,
    )


class AtomicCheckpointTests(unittest.TestCase):
    def test_completion_marker_is_written_last_then_directory_is_published(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            events: list[str] = []

            def observe(event: str, path: Path) -> None:
                events.append(event)
                if event == "before_complete_marker":
                    self.assertFalse((path / COMPLETE_MARKER_FILENAME).exists())
                if event == "complete_marker_written":
                    self.assertTrue((path / COMPLETE_MARKER_FILENAME).is_file())

            validation = publish_graph_checkpoint(
                run_root,
                fixture_data(manifest),
                event_callback=observe,
            )
            self.assertEqual(
                events,
                [
                    "temporary_directory_created",
                    "before_complete_marker",
                    "complete_marker_written",
                    "before_atomic_publication",
                    "checkpoint_renamed",
                    "published_checkpoint_validated",
                    "publication_timing_record_written",
                    "checkpoint_published",
                ],
            )
            self.assertEqual(validation.graph_id, "fixture_graph")
            self.assertTrue(validation.path.is_dir())
            self.assertFalse(
                any(".tmp-" in item.name for item in validation.path.parent.iterdir())
            )

    def test_operational_timer_includes_publication_and_final_validation(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            original_validate = checkpoint_module.validate_graph_checkpoint
            events: list[str] = []

            def observe(event: str, _path: Path) -> None:
                events.append(event)
                if event == "before_atomic_publication":
                    time.sleep(0.025)

            def delayed_final_validation(path, **kwargs):
                result = original_validate(path, **kwargs)
                if Path(path).name == "fixture_graph":
                    time.sleep(0.025)
                return result

            graph_start = time.perf_counter_ns()
            with patch.object(
                checkpoint_module,
                "validate_graph_checkpoint",
                side_effect=delayed_final_validation,
            ):
                published = publish_graph_checkpoint(
                    run_root,
                    fixture_data(manifest),
                    graph_wall_start_ns=graph_start,
                    event_callback=observe,
                )
            record = validate_publication_timing_record(
                run_root,
                graph_id="fixture_graph",
                expected_run_manifest=manifest,
            )
            self.assertGreaterEqual(
                record["atomic_publication_and_final_validation_ns"],
                45_000_000,
            )
            self.assertGreaterEqual(
                record["end_to_end_graph_wall_ns"],
                record["atomic_publication_and_final_validation_ns"],
            )
            self.assertLess(
                events.index("checkpoint_renamed"),
                events.index("published_checkpoint_validated"),
            )
            self.assertLess(
                events.index("published_checkpoint_validated"),
                events.index("publication_timing_record_written"),
            )
            with (published.path / "timings.json").open(
                "r", encoding="utf-8"
            ) as stream:
                payload_timings = decode_json_value(json.load(stream))
            self.assertIn("payload_serialization_ns", payload_timings)
            self.assertIn("prepublication_wall_ns", payload_timings)
            self.assertNotIn("checkpoint_serialization_ns", payload_timings)
            self.assertNotIn("total_graph_wall_time_ns", payload_timings)
            self.assertNotIn("end_to_end_graph_wall_ns", payload_timings)
            self.assertEqual(
                record["definitions"],
                PUBLICATION_TIMING_DEFINITIONS,
            )
            self.assertIn("validation", record["endpoint"])

    def test_publication_timing_record_is_atomic_and_path_contained(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            publish_graph_checkpoint(run_root, fixture_data(manifest))
            timing_root = run_root / PUBLICATION_TIMING_DIRECTORY
            entries = tuple(timing_root.iterdir())
            self.assertEqual(
                entries,
                (timing_root / "fixture_graph.json",),
            )
            self.assertTrue(entries[0].is_file())
            entries[0].resolve().relative_to(run_root.resolve())
            self.assertFalse(
                any(item.name.startswith(".") for item in timing_root.iterdir())
            )

    def test_complete_checkpoint_round_trips_and_has_exact_counts(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            published = publish_graph_checkpoint(run_root, fixture_data(manifest))
            validated = validate_graph_checkpoint(
                published.path,
                expected_run_manifest=manifest,
                expected_graph_id="fixture_graph",
            )
            self.assertEqual(validated.counts["vertices"], 3)
            self.assertEqual(validated.counts["pairs"], 2)
            self.assertEqual(validated.counts["dijkstra_records"], 2)
            self.assertEqual(validated.counts["route_records"], 2)
            self.assertEqual(validated.counts["coordinate_rows"], 3)

    def test_compressed_route_walk_is_losslessly_recoverable(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            published = publish_graph_checkpoint(run_root, fixture_data(manifest))
            with gzip.open(
                published.path / "routes.jsonl.gz",
                "rt",
                encoding="utf-8",
            ) as stream:
                records = [
                    decode_json_value(json.loads(line)) for line in stream
                ]
            self.assertEqual(records[0]["walk"], [0, 1])
            self.assertEqual(records[0]["physical_hop_count"], 1)
            self.assertEqual(records[0]["dijkstra_hop_count"], 1)
            self.assertEqual(records[0]["stretch"], 1.0)
            self.assertIn("initial_failure_type", records[0])
            self.assertIn("repair_attempted", records[0])

    def test_scientific_payload_bytes_are_deterministic_and_gzip_is_stable(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            roots = (Path(first) / "run", Path(second) / "run")
            hashes = []
            for index, run_root in enumerate(roots):
                manifest = fixture_manifest(run_root.parent)
                write_run_manifest_once(run_root, manifest)
                published = publish_graph_checkpoint(
                    run_root,
                    fixture_data(manifest, timing=10 + index),
                )
                hashes.append(deterministic_payload_hashes(published.path))
            self.assertEqual(hashes[0], hashes[1])
            self.assertIn("coordinates.csv.gz", hashes[0])
            self.assertNotIn("timings.json", hashes[0])

    def test_payload_hashes_match_across_python_hash_seeds(self):
        probe = (
            "import json;"
            "from pathlib import Path;"
            "from tempfile import TemporaryDirectory;"
            "from tests.test_experiment_checkpoint import "
            "fixture_manifest,fixture_data;"
            "from experiment_checkpoint import "
            "write_run_manifest_once,publish_graph_checkpoint,"
            "deterministic_payload_hashes;"
            "temporary=TemporaryDirectory();"
            "root=Path(temporary.name)/'run';"
            "manifest=fixture_manifest(root.parent);"
            "write_run_manifest_once(root,manifest);"
            "published=publish_graph_checkpoint(root,fixture_data(manifest));"
            "print(json.dumps(deterministic_payload_hashes(published.path),"
            "sort_keys=True))"
        )
        outputs = []
        for hash_seed in ("1", "8675309"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONPATH"] = str(CODE_DIR)
            completed = subprocess.run(
                [sys.executable, "-B", "-c", probe],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(json.loads(completed.stdout))
        self.assertEqual(outputs[0], outputs[1])

    def test_existing_checkpoint_is_never_overwritten(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            first = publish_graph_checkpoint(run_root, fixture_data(manifest))
            before = deterministic_payload_hashes(first.path)
            with self.assertRaisesRegex(CheckpointError, "already exists"):
                publish_graph_checkpoint(run_root, fixture_data(manifest))
            self.assertEqual(before, deterministic_payload_hashes(first.path))

    def test_publication_failure_preserves_error_without_complete_marker(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)

            def fail_before_marker(event: str, _path: Path) -> None:
                if event == "before_complete_marker":
                    raise RuntimeError("intentional failure")

            with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                publish_graph_checkpoint(
                    run_root,
                    fixture_data(manifest),
                    event_callback=fail_before_marker,
                )
            temporary_dirs = tuple(
                (run_root / CHECKPOINT_DIRECTORY).glob("*.tmp-*")
            )
            self.assertEqual(len(temporary_dirs), 1)
            self.assertTrue((temporary_dirs[0] / ERROR_REPORT_FILENAME).is_file())
            self.assertFalse(
                (temporary_dirs[0] / COMPLETE_MARKER_FILENAME).exists()
            )
            with (temporary_dirs[0] / ERROR_REPORT_FILENAME).open(
                "r", encoding="utf-8"
            ) as stream:
                report = decode_json_value(json.load(stream))
            self.assertEqual(report["run_manifest"], manifest)
            self.assertEqual(
                report["run_manifest_sha256"],
                run_manifest_sha256(manifest),
            )
            self.assertEqual(
                report["run_manifest_sha256"],
                sha256(
                    (run_root / RUN_MANIFEST_FILENAME).read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(report["graph_id"], "fixture_graph")
            self.assertEqual(report["stage"], "checkpoint_publication")
            self.assertEqual(report["exception_type"], "RuntimeError")
            self.assertEqual(report["exception_message"], "intentional failure")
            self.assertTrue(report["timestamp_utc"])
            self.assertEqual(
                Path(report["temporary_checkpoint_path"]),
                temporary_dirs[0].resolve(),
            )
            self.assertEqual(
                Path(report["final_checkpoint_path"]),
                (run_root / CHECKPOINT_DIRECTORY / "fixture_graph").resolve(),
            )

    def test_non_checkpoint_failure_is_preserved_without_completion(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            error_path = preserve_graph_error(
                run_root,
                graph_id="fixture_graph",
                stage="hydra_embedding",
                exception=RuntimeError("failed"),
                run_manifest=manifest,
            )
            self.assertTrue((error_path / ERROR_REPORT_FILENAME).is_file())
            self.assertFalse((error_path / COMPLETE_MARKER_FILENAME).exists())
            with (error_path / ERROR_REPORT_FILENAME).open(
                "r", encoding="utf-8"
            ) as stream:
                report = decode_json_value(json.load(stream))
            self.assertEqual(report["run_manifest"], manifest)
            self.assertEqual(
                report["run_manifest_sha256"],
                run_manifest_sha256(manifest),
            )
            self.assertEqual(report["stage"], "hydra_embedding")
            self.assertEqual(report["exception_type"], "RuntimeError")

    def test_hostile_graph_identity_is_rejected_before_any_path_creation(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            hostile = fixture_data(manifest)
            object.__setattr__(hostile, "graph_id", "../escape")
            with self.assertRaisesRegex(ValueError, "graph_id"):
                publish_graph_checkpoint(run_root, hostile)
            self.assertFalse((Path(temporary) / "escape").exists())


class CheckpointAuditTests(unittest.TestCase):
    def test_audit_is_read_only_when_run_directory_does_not_exist(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "absent"
            audit = audit_run_checkpoints(
                run_root,
                schedule_ids=("fixture_graph",),
            )
            self.assertTrue(audit.resumable)
            self.assertEqual(audit.remaining_graph_ids, ("fixture_graph",))
            self.assertFalse(run_root.exists())

    def test_audit_accepts_complete_and_reports_remaining_in_schedule_order(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            schedule = ("fixture_graph", "second_graph")
            manifest = fixture_manifest(run_root.parent, graph_ids=schedule)
            write_run_manifest_once(run_root, manifest)
            publish_graph_checkpoint(run_root, fixture_data(manifest))
            audit = audit_run_checkpoints(
                run_root,
                schedule_ids=schedule,
                expected_run_manifest=manifest,
            )
            self.assertTrue(audit.resumable)
            self.assertEqual(audit.complete_graph_ids, ("fixture_graph",))
            self.assertEqual(audit.remaining_graph_ids, ("second_graph",))

    def test_corruption_is_detected_and_resume_is_refused(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            published = publish_graph_checkpoint(run_root, fixture_data(manifest))
            with (published.path / "routes.jsonl.gz").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(CheckpointCorruptionError):
                validate_graph_checkpoint(published.path)
            audit = audit_run_checkpoints(
                run_root,
                schedule_ids=("fixture_graph",),
                expected_run_manifest=manifest,
            )
            self.assertFalse(audit.resumable)
            self.assertTrue(audit.errors)

    def test_missing_publication_timing_record_blocks_resume(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            publish_graph_checkpoint(run_root, fixture_data(manifest))
            (
                run_root
                / PUBLICATION_TIMING_DIRECTORY
                / "fixture_graph.json"
            ).unlink()
            audit = audit_run_checkpoints(
                run_root,
                schedule_ids=("fixture_graph",),
                expected_run_manifest=manifest,
            )
            self.assertFalse(audit.resumable)
            self.assertEqual(audit.complete_graph_ids, ())
            self.assertTrue(
                any("operational-integrity error" in error for error in audit.errors)
            )

    def test_corrupt_publication_timing_record_blocks_resume(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            publish_graph_checkpoint(run_root, fixture_data(manifest))
            timing_path = (
                run_root
                / PUBLICATION_TIMING_DIRECTORY
                / "fixture_graph.json"
            )
            timing_path.write_bytes(b"{not-json")
            audit = audit_run_checkpoints(
                run_root,
                schedule_ids=("fixture_graph",),
                expected_run_manifest=manifest,
            )
            self.assertFalse(audit.resumable)
            self.assertEqual(audit.complete_graph_ids, ())

    def test_mismatched_publication_timing_record_blocks_resume(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            publish_graph_checkpoint(run_root, fixture_data(manifest))
            timing_path = (
                run_root
                / PUBLICATION_TIMING_DIRECTORY
                / "fixture_graph.json"
            )
            with timing_path.open("r", encoding="utf-8") as stream:
                record = json.load(stream)
            record["graph_id"] = "wrong_graph"
            timing_path.write_text(
                json.dumps(record, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            audit = audit_run_checkpoints(
                run_root,
                schedule_ids=("fixture_graph",),
                expected_run_manifest=manifest,
            )
            self.assertFalse(audit.resumable)
            self.assertEqual(audit.complete_graph_ids, ())
            self.assertTrue(
                any("graph mismatch" in error for error in audit.errors)
            )

    def test_missing_completion_marker_is_detected(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            incomplete = run_root / CHECKPOINT_DIRECTORY / "fixture_graph"
            incomplete.mkdir(parents=True)
            with self.assertRaisesRegex(
                CheckpointCorruptionError, "completion marker"
            ):
                validate_graph_checkpoint(incomplete)

    def test_missing_payload_file_is_detected(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            published = publish_graph_checkpoint(run_root, fixture_data(manifest))
            (published.path / "pairs.csv.gz").unlink()
            with self.assertRaisesRegex(CheckpointCorruptionError, "pairs.csv.gz"):
                validate_graph_checkpoint(published.path)

    def test_incomplete_temporary_directory_blocks_resume(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            (
                run_root
                / CHECKPOINT_DIRECTORY
                / ".fixture_graph.tmp-interrupted"
            ).mkdir(parents=True)
            audit = audit_run_checkpoints(
                run_root,
                schedule_ids=("fixture_graph",),
                expected_run_manifest=manifest,
            )
            self.assertFalse(audit.resumable)
            self.assertIn("incomplete temporary", audit.errors[0])

    def test_source_or_configuration_identity_mismatch_blocks_resume(self):
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            existing = fixture_manifest(output_root)
            mismatched = deepcopy(existing)
            mismatched["source_fingerprint"] = "f" * 64
            with self.assertRaisesRegex(
                CheckpointCompatibilityError, "source_fingerprint"
            ):
                validate_run_manifest_compatibility(existing, mismatched)
            mismatched = deepcopy(existing)
            mismatched["combined_freeze_hash"] = "e" * 64
            with self.assertRaisesRegex(
                CheckpointCompatibilityError, "combined_freeze_hash"
            ):
                validate_run_manifest_compatibility(existing, mismatched)
            mismatched = deepcopy(existing)
            mismatched["dependency_versions"] = {"networkx": "0.0"}
            with self.assertRaisesRegex(
                CheckpointCompatibilityError, "dependency_versions"
            ):
                validate_run_manifest_compatibility(existing, mismatched)

    def test_full_schema_rejects_non_production_row_counts(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(
                run_root.parent,
                profile="full",
            )
            write_run_manifest_once(run_root, manifest)
            with self.assertRaisesRegex(ValueError, "full checkpoint requires"):
                publish_graph_checkpoint(run_root, fixture_data(manifest))

    def test_full_schema_constants_are_exact(self):
        self.assertEqual(RESULT_SCHEMA_VERSION, 1)
        self.assertEqual(
            FULL_COORDINATE_CONDITION_IDS,
            ("hydra", "mds_r050", "mds_r070", "mds_r085", "mds_r095"),
        )
        self.assertEqual(len(FULL_DISTORTION_CONDITION_IDS), 7)

    def test_nan_and_infinity_are_rejected_before_publication(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            for non_finite in (float("nan"), float("inf"), float("-inf")):
                data = fixture_data(manifest)
                object.__setattr__(
                    data,
                    "network_metrics",
                    {
                        "number_of_vertices": 3,
                        "number_of_edges": 2,
                        "invalid": non_finite,
                    },
                )
                with self.assertRaisesRegex(ValueError, "NaN and infinity"):
                    publish_graph_checkpoint(run_root, data)

    def test_dijkstra_apsp_disagreement_is_rejected_before_publication(self):
        with TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            manifest = fixture_manifest(run_root.parent)
            write_run_manifest_once(run_root, manifest)
            data = fixture_data(manifest)
            records = [dict(record) for record in data.dijkstra_records]
            records[0]["apsp_length"] = 2
            object.__setattr__(data, "dijkstra_records", tuple(records))
            with self.assertRaisesRegex(ValueError, "Dijkstra/APSP"):
                publish_graph_checkpoint(run_root, data)


if __name__ == "__main__":
    unittest.main()
