"""Focused tests for the excluded Step 15 capacity benchmark."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import benchmark_experiment_capacity as capacity  # noqa: E402
import run_full_experiment as runner  # noqa: E402
from experiment_config import (  # noqa: E402
    ANALYSIS_PLAN_HASH,
    BARABASI_ALBERT,
    COMBINED_FREEZE_HASH,
    DATA_GENERATION_HASH,
    ERDOS_RENYI,
    FEASIBILITY_PILOT_SEEDS,
)


def repetition_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for cell_index, spec in enumerate(capacity.BENCHMARK_GRAPH_SPECS):
        for repetition in range(3):
            runtime = (cell_index + 1) * 1_000_000_000 + repetition * 100
            checkpoint = (cell_index + 1) * 10_000 + repetition
            records.append(
                {
                    "graph_id": spec.graph_id,
                    "cell_id": spec.cell_id,
                    "model": spec.model,
                    "n": spec.n,
                    "m": spec.m,
                    "excluded_seed": spec.seed,
                    "repetition": repetition,
                    "warmup": repetition == 0,
                    "included_in_projection": repetition in (1, 2),
                    "end_to_end_graph_wall_ns": runtime,
                    "component_timings_ns": {
                        "graph_generation_ns": 1,
                        "payload_serialization_ns": 2,
                        "atomic_publication_and_final_validation_ns": 3,
                    },
                    "complete_graph_checkpoint_bytes": checkpoint,
                    "checkpoint_files_bytes": {"routes.jsonl.gz": checkpoint},
                    "compressed_checkpoint_files_bytes": {
                        "routes.jsonl.gz": checkpoint
                    },
                    "run_level_overhead_bytes": 60,
                    "run_level_overhead_breakdown_bytes": {
                        "run_manifest": 10,
                        "progress": 20,
                        "publication_timing": 30 + repetition,
                    },
                    "peak_temporary_plus_final_bytes": checkpoint,
                    "row_counts": {
                        "pairs": 1_000,
                        "dijkstra_records": 1_000,
                        "route_records": 15_000,
                        "distortion_records": 7,
                    },
                    "published_checkpoint_integrity_validated": True,
                }
            )
    return records


def valid_profile() -> dict[str, object]:
    return capacity.build_capacity_profile(
        repetitions=repetition_records(),
        source_commit="1" * 40,
        benchmark_volume_identifier="test_volume",
        benchmark_filesystem_type="testfs",
        available_before_bytes=100_000_000_000,
        available_during_peak_bytes=99_000_000_000,
        available_after_cleanup_bytes=100_000_000_000,
        cleanup_restored_temporary_usage=True,
        benchmark_timestamp_utc="2026-01-01T00:00:00+00:00",
    )


def rehash(profile: dict[str, object]) -> dict[str, object]:
    profile["profile_sha256"] = capacity.profile_sha256(profile)
    return profile


class BenchmarkDesignTests(unittest.TestCase):
    def test_exact_six_excluded_graph_identities(self):
        specs = capacity.validate_benchmark_specs(
            tuple(entry.graph_id for entry in runner.build_full_schedule())
        )
        self.assertEqual(len(specs), 6)
        self.assertEqual(tuple(spec.seed for spec in specs), FEASIBILITY_PILOT_SEEDS)
        self.assertEqual(
            tuple((spec.model, spec.n, spec.m) for spec in specs),
            (
                (ERDOS_RENYI, 100, 16),
                (ERDOS_RENYI, 300, 16),
                (ERDOS_RENYI, 1_000, 16),
                (BARABASI_ALBERT, 100, 16),
                (BARABASI_ALBERT, 300, 16),
                (BARABASI_ALBERT, 1_000, 16),
            ),
        )

    def test_final_schedule_identity_is_rejected(self):
        with self.assertRaisesRegex(capacity.CapacityProfileError, "overlaps"):
            capacity.validate_benchmark_specs(
                (capacity.BENCHMARK_GRAPH_SPECS[0].graph_id,)
            )
        final_ids = {entry.graph_id for entry in runner.build_full_schedule()}
        self.assertFalse(
            final_ids
            & {spec.graph_id for spec in capacity.BENCHMARK_GRAPH_SPECS}
        )

    def test_full_pair_and_repetition_policy_is_exact(self):
        self.assertEqual(capacity.BENCHMARK_PAIR_COUNT, 1_000)
        self.assertEqual(capacity.BENCHMARK_REPETITIONS, 3)
        self.assertEqual(capacity.WARMUP_REPETITION, 0)
        self.assertEqual(capacity.MEASURED_REPETITIONS, (1, 2))
        self.assertEqual(
            capacity.WATCHDOG_SECONDS_BY_N,
            {100: 120, 300: 300, 1_000: 600},
        )

    def test_canary_cli_is_excluded_only_and_absent_from_production_runner(self):
        parsed = capacity._parser().parse_args(
            ["--canary", "--canary-n", "1000"]
        )
        self.assertTrue(parsed.canary)
        self.assertEqual(parsed.canary_n, 1_000)
        production_options = {
            option
            for action in runner._parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--canary", production_options)

    def test_only_repetitions_one_and_two_enter_cell_summaries(self):
        cells = capacity.summarise_capacity_cells(repetition_records())
        self.assertEqual(len(cells), 6)
        self.assertTrue(
            all(cell["measured_repetitions"] == [1, 2] for cell in cells)
        )
        first = cells[0]
        self.assertEqual(
            first["measured_end_to_end_graph_wall_ns"],
            [1_000_000_100, 1_000_000_200],
        )
        self.assertEqual(
            first["median_end_to_end_graph_wall_ns"],
            1_000_000_150,
        )
        self.assertEqual(
            first["maximum_end_to_end_graph_wall_ns"],
            1_000_000_200,
        )
        self.assertEqual(first["range_end_to_end_graph_wall_ns"], 100)

    def test_invalid_warmup_inclusion_is_rejected(self):
        records = repetition_records()
        records[0]["included_in_projection"] = True
        with self.assertRaisesRegex(capacity.CapacityProfileError, "policy"):
            capacity.summarise_capacity_cells(records)


class ProjectionTests(unittest.TestCase):
    def test_runtime_projection_is_exact_across_six_cells_times_sixty(self):
        cells = capacity.summarise_capacity_cells(repetition_records())
        projection = capacity.project_runtime(cells)
        expected = sum(
            ((index + 1) * 1_000_000_000 + 150) * 60
            for index in range(6)
        )
        self.assertEqual(projection["nominal_projected_runtime_ns"], expected)
        self.assertEqual(
            projection["conservative_projected_runtime_ns"],
            expected * 1.5,
        )
        self.assertEqual(len(projection["cell_contributions"]), 6)
        self.assertTrue(
            all(
                item["graphs_represented"] == 60
                for item in projection["cell_contributions"]
            )
        )

    def test_storage_projection_allowance_and_required_space_are_exact(self):
        cells = capacity.summarise_capacity_cells(repetition_records())
        projection = capacity.project_storage(cells)
        checkpoint = sum(
            ((index + 1) * 10_000 + 2) * 60 for index in range(6)
        )
        publication = 32 * 60 * 6
        fixed = 30
        subtotal = checkpoint + publication + fixed
        allowance = math.ceil(subtotal * 0.01)
        projected = subtotal + allowance
        required = max(
            2 * projected,
            projected + 5 * capacity.GIB_BYTES,
        )
        self.assertEqual(projection["projected_checkpoint_bytes"], checkpoint)
        self.assertEqual(
            projection["measured_projected_run_level_overhead_bytes"],
            publication + fixed,
        )
        self.assertEqual(projection["metadata_allowance_bytes"], allowance)
        self.assertEqual(projection["projected_storage_bytes"], projected)
        self.assertEqual(projection["required_free_bytes"], required)
        self.assertEqual(
            projection["required_free_gib"],
            required / 1_073_741_824,
        )


class CapacityProfileTests(unittest.TestCase):
    def test_profile_canonicalization_and_hash_round_trip(self):
        profile = valid_profile()
        self.assertEqual(profile["profile_sha256"], capacity.profile_sha256(profile))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            capacity.write_capacity_profile(profile, path)
            loaded = capacity.load_capacity_profile(
                path,
                expected_volume_identifier="test_volume",
                current_available_bytes=10**15,
            )
        self.assertEqual(loaded, profile)

    def test_nan_and_infinity_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            profile = valid_profile()
            profile["invalid_seconds"] = value
            with self.assertRaisesRegex(ValueError, "NaN and infinity"):
                capacity.profile_sha256(profile)

    def test_scientific_routing_outcomes_are_forbidden(self):
        profile = valid_profile()
        profile["routing_success"] = 1.0
        rehash(profile)
        with self.assertRaisesRegex(capacity.CapacityProfileError, "scientific"):
            capacity.validate_capacity_profile(profile)

    def test_full_home_path_is_forbidden(self):
        profile = valid_profile()
        profile["unsafe_path"] = str(Path.home().resolve())
        rehash(profile)
        with self.assertRaisesRegex(capacity.CapacityProfileError, "home"):
            capacity.validate_capacity_profile(profile)

    def test_volume_and_insufficient_disk_are_rejected(self):
        profile = valid_profile()
        with self.assertRaisesRegex(capacity.CapacityProfileError, "volume differs"):
            capacity.validate_capacity_profile(
                profile,
                expected_volume_identifier="different_volume",
            )
        with self.assertRaisesRegex(capacity.CapacityProfileError, "below"):
            capacity.validate_capacity_profile(
                profile,
                expected_volume_identifier="test_volume",
                current_available_bytes=int(profile["required_free_bytes"]) - 1,
            )

    def test_hash_configuration_and_schema_mismatches_are_rejected(self):
        profile = valid_profile()
        profile["profile_sha256"] = "0" * 64
        with self.assertRaisesRegex(capacity.CapacityProfileError, "SHA-256"):
            capacity.validate_capacity_profile(profile)

        profile = valid_profile()
        profile["step13_hashes"]["combined"] = "0" * 64
        rehash(profile)
        with self.assertRaisesRegex(capacity.CapacityProfileError, "Step 13"):
            capacity.validate_capacity_profile(profile)

        profile = valid_profile()
        profile["result_schema_version"] = 999
        rehash(profile)
        with self.assertRaisesRegex(capacity.CapacityProfileError, "result-schema"):
            capacity.validate_capacity_profile(profile)

    def test_profile_contains_no_scientific_outcomes_or_absolute_home_path(self):
        profile = valid_profile()
        encoded = json.dumps(profile, sort_keys=True)
        self.assertNotIn(str(Path.home().resolve()), encoded)
        self.assertFalse(
            capacity._profile_keys(profile)
            & capacity._FORBIDDEN_SCIENTIFIC_PROFILE_KEYS
        )
        self.assertTrue(profile["non_scientific"])
        self.assertTrue(profile["excluded_from_analysis"])

    def test_step13_hashes_remain_exact(self):
        self.assertEqual(
            DATA_GENERATION_HASH,
            capacity.EXPECTED_DATA_GENERATION_HASH,
        )
        self.assertEqual(
            ANALYSIS_PLAN_HASH,
            capacity.EXPECTED_ANALYSIS_PLAN_HASH,
        )
        self.assertEqual(
            COMBINED_FREEZE_HASH,
            capacity.EXPECTED_COMBINED_FREEZE_HASH,
        )

    def test_performance_content_fingerprint_is_commit_independent(self):
        profile = valid_profile()
        expected = capacity.performance_source_fingerprint()
        self.assertEqual(
            profile["performance_source_content_fingerprint"],
            expected,
        )
        profile["base_commit"] = "2" * 40
        profile["source_commit"] = "2" * 40
        rehash(profile)
        validated = capacity.validate_capacity_profile(profile)
        self.assertEqual(
            validated["performance_source_content_fingerprint"],
            expected,
        )

    def test_performance_content_fingerprint_mismatch_is_rejected(self):
        profile = valid_profile()
        profile["performance_source_content_fingerprint"] = "0" * 64
        rehash(profile)
        with self.assertRaisesRegex(
            capacity.CapacityProfileError,
            "performance-source",
        ):
            capacity.validate_capacity_profile(profile)

        profile = valid_profile()
        profile["performance_source_manifest"]["files"][
            "code/routing.py"
        ] = "0" * 64
        rehash(profile)
        with self.assertRaisesRegex(
            capacity.CapacityProfileError,
            "performance-source",
        ):
            capacity.validate_capacity_profile(profile)


class PreflightAndCleanupTests(unittest.TestCase):
    def test_canary_runs_exactly_one_approved_fixture_and_cleans_up(self):
        stages: list[str] = []
        calls: list[dict[str, object]] = []

        def measure_once(**kwargs):
            calls.append(kwargs)
            spec = kwargs["spec"]
            repetition_root = (
                kwargs["benchmark_root"]
                / f"{spec.graph_id}_repetition_{kwargs['repetition']}"
            )
            repetition_root.mkdir()
            return repetition_records()[0], 1

        with TemporaryDirectory() as temporary:
            with patch.object(
                capacity,
                "_measure_repetition",
                side_effect=measure_once,
            ):
                result = capacity.run_capacity_canary(
                    output_root=temporary,
                    progress_callback=stages.append,
                )
            self.assertEqual(list(Path(temporary).iterdir()), [])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["spec"], capacity.CANARY_GRAPH_SPEC)
        self.assertEqual(calls[0]["repetition"], 0)
        self.assertEqual(result["repetitions"], 1)
        self.assertEqual(result["pair_count"], 1_000)
        self.assertEqual(
            result["row_counts"],
            {
                "pairs": 1_000,
                "dijkstra_records": 1_000,
                "route_records": 15_000,
                "distortion_records": 7,
            },
        )
        self.assertTrue(result["temporary_checkpoint_cleanup_completed"])
        self.assertIn("temporary_checkpoint_removed", stages)

    def test_largest_canary_selects_only_the_excluded_er_n1000_fixture(self):
        calls: list[dict[str, object]] = []

        def measure_once(**kwargs):
            calls.append(kwargs)
            spec = kwargs["spec"]
            repetition_root = (
                kwargs["benchmark_root"]
                / f"{spec.graph_id}_repetition_{kwargs['repetition']}"
            )
            repetition_root.mkdir()
            record = deepcopy(repetition_records()[6])
            record.update(
                {
                    "graph_id": spec.graph_id,
                    "cell_id": spec.cell_id,
                    "model": spec.model,
                    "n": spec.n,
                    "m": spec.m,
                    "excluded_seed": spec.seed,
                    "repetition": 0,
                }
            )
            return record, 1

        with TemporaryDirectory() as temporary:
            with patch.object(
                capacity,
                "_measure_repetition",
                side_effect=measure_once,
            ):
                result = capacity.run_capacity_canary(
                    n=1_000,
                    output_root=temporary,
                )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["spec"].model, ERDOS_RENYI)
        self.assertEqual(calls[0]["spec"].n, 1_000)
        self.assertEqual(calls[0]["spec"].seed, 4_000_037)
        self.assertEqual(result["repetitions"], 1)

    def test_read_only_full_preflight_uses_profile_without_creating_output(self):
        profile = valid_profile()
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "uncreated-results"
            with (
                patch.object(runner, "load_capacity_profile", return_value=profile),
                patch.object(
                    runner,
                    "volume_identifier",
                    return_value="test_volume",
                ),
            ):
                report = runner.preflight(
                    mode="full",
                    confirmation=runner.COMBINED_FREEZE_HASH,
                    output_root=output_root,
                )
            self.assertFalse(output_root.exists())
            self.assertTrue(report.capacity_status["profile_valid"])
            self.assertTrue(report.capacity_status["disk_space_pass"])

    def test_safe_cleanup_is_contained(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "specific-benchmark-repetition"
            target.mkdir()
            (target / "payload").write_text("x", encoding="utf-8")
            capacity.safe_remove_tree(root, target)
            self.assertFalse(target.exists())
            outside = root.parent
            with self.assertRaisesRegex(capacity.CapacityProfileError, "escapes"):
                capacity.safe_remove_tree(root, outside)

    def test_failed_benchmark_directory_is_preserved(self):
        preserved: list[Path] = []

        def fail_measurement(*, output_root, **_kwargs):
            evidence = Path(output_root) / "failed-evidence"
            evidence.mkdir()
            preserved.append(evidence)
            raise capacity.CapacityProfileError("intentional capacity failure")

        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                capacity.CapacityProfileError,
                "intentional",
            ):
                capacity.run_capacity_benchmark(
                    output_root=temporary,
                    profile_path=Path(temporary) / "unused.json",
                    repetition_runner=fail_measurement,
                )
            self.assertEqual(len(preserved), 1)
            self.assertTrue(preserved[0].is_dir())

    def test_benchmark_runs_exactly_18_repetitions_in_frozen_order(self):
        calls: list[tuple[str, int]] = []
        records = repetition_records()

        def run_one(*, spec, repetition, **_kwargs):
            calls.append((spec.graph_id, repetition))
            record = next(
                deepcopy(item)
                for item in records
                if item["graph_id"] == spec.graph_id
                and item["repetition"] == repetition
            )
            return record, 90_000_000_000

        with TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "profile.json"
            result = capacity.run_capacity_benchmark(
                output_root=temporary,
                profile_path=profile_path,
                repetition_runner=run_one,
            )

        self.assertEqual(
            calls,
            [
                (spec.graph_id, repetition)
                for spec in capacity.BENCHMARK_GRAPH_SPECS
                for repetition in range(3)
            ],
        )
        self.assertEqual(result["completed_repetitions"], 18)

    def test_repetition_failure_stops_before_later_repetitions(self):
        calls: list[tuple[str, int]] = []
        records = repetition_records()

        def stop_second(*, spec, repetition, **_kwargs):
            calls.append((spec.graph_id, repetition))
            if len(calls) == 2:
                raise capacity.CapacityProfileError("watchdog timeout")
            return deepcopy(records[0]), 90_000_000_000

        with TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "profile.json"
            with self.assertRaisesRegex(
                capacity.CapacityProfileError,
                "watchdog timeout",
            ):
                capacity.run_capacity_benchmark(
                    output_root=temporary,
                    profile_path=profile_path,
                    repetition_runner=stop_second,
                )
            self.assertFalse(profile_path.exists())

        self.assertEqual(
            calls,
            [
                (capacity.BENCHMARK_GRAPH_SPECS[0].graph_id, 0),
                (capacity.BENCHMARK_GRAPH_SPECS[0].graph_id, 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
