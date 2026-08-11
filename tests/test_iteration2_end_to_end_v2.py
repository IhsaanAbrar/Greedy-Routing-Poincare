from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import random
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import analyze_iteration2 as analysis_entry  # noqa: E402
from analyze_iteration2 import (  # noqa: E402
    create_analysis_tables,
    load_complete_raw_run,
)
from iteration2_analysis import (  # noqa: E402
    aggregate_graph_metrics,
    aggregate_interactions,
    descriptive_model_contrasts,
    property_associations,
    simultaneous_radius_bands,
)
from iteration2_config import ITERATION2_RUN_IDENTITY  # noqa: E402
from iteration2_excluded import ExcludedAnalysisFixtureContract  # noqa: E402
from iteration2_reporting import (  # noqa: E402
    FIGURE_FILES,
    WORKBOOK_SHEETS,
    build_reporting_bundle,
    validate_reporting_bundle,
)
from iteration2_v2_support import (  # noqa: E402
    deterministic_bootstrap_provider,
    excluded_analysis_validation_evidence,
    excluded_fixture_contract,
    excluded_run_manifest,
    publish_completed_excluded_raw_run,
    synthetic_inference_rows,
)
from run_iteration2 import (  # noqa: E402
    GRAPH_RESULT_FILENAME,
    excluded_feasibility_results,
    publish_graph_checkpoint,
    validate_checkpoint_directory,
)
from validate_iteration2 import (  # noqa: E402
    Iteration2ValidationError,
    _scientific_payload_bytes,
    validate_iteration2_graph_result,
)
from validate_full_experiment import compute_raw_tree_fingerprint  # noqa: E402


def _add_reduced_inference_tables(
    tables: dict[str, list[dict[str, object]]],
) -> None:
    graph_metrics, graph_rows, interactions = synthetic_inference_rows()
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
        bootstrap_replicates=2,
        bootstrap_provider=deterministic_bootstrap_provider,
    )
    pointwise = aggregate_interactions(
        interactions,
        bootstrap_replicates=2,
        bootstrap_provider=deterministic_bootstrap_provider,
    )
    simultaneous = simultaneous_radius_bands(
        interactions,
        bootstrap_replicates=2,
        bootstrap_provider=deterministic_bootstrap_provider,
    )
    interaction_rows = [*pointwise, *simultaneous]
    tables["Cell Estimates"] = [
        row for row in estimates if row["stratum_count"] == 1
    ]
    tables["Model Marginals"] = [
        row for row in estimates if row["stratum_count"] == 9
    ]
    tables["Matched Success Contrasts"] = [
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
    tables["Common-Success Stretch"] = [
        {
            **row,
            "conditioning": "both_ordinary_methods_succeeded",
        }
        for row in estimates
        if row["metric"]
        == "common_success_poincare_minus_euclidean_stretch"
    ]
    tables["Matched Embedding Interactions"] = interaction_rows
    tables["Equivalence Sensitivity"] = [dict(row) for row in interaction_rows]
    tables["Property Associations"] = property_associations(
        graph_metrics,
        graph_rows,
        inference_replicates=2,
        bootstrap_provider=deterministic_bootstrap_provider,
    )
    tables["Model Contrasts"] = descriptive_model_contrasts(
        graph_rows,
        bootstrap_replicates=2,
        bootstrap_provider=deterministic_bootstrap_provider,
    )


class Iteration2ExcludedEndToEndTests(unittest.TestCase):
    def test_complete_excluded_pipeline_and_resume_validation(self):
        temporary_path: Path | None = None
        with TemporaryDirectory(prefix="iteration2-excluded-e2e-") as temporary:
            temporary_path = Path(temporary)
            results = excluded_feasibility_results()
            contract = excluded_fixture_contract(results)
            run_root = temporary_path / "results" / contract.raw_identity
            graph_root = run_root / "graphs"
            graph_root.mkdir(parents=True)

            repeated = excluded_feasibility_results()
            self.assertEqual(len(results), 2)
            for first, second in zip(results, repeated, strict=True):
                self.assertEqual(
                    _scientific_payload_bytes(first),
                    _scientific_payload_bytes(second),
                )
                counts = validate_iteration2_graph_result(first)
                self.assertEqual(counts["pairs"], 12)
                self.assertEqual(counts["dijkstra_records"], 12)
                self.assertEqual(counts["route_records"], 12 * 28)
                by_pair: dict[int, int] = {}
                for row in first["route_records"]:
                    pair_index = int(row["pair_index"])
                    by_pair[pair_index] = by_pair.get(pair_index, 0) + 1
                self.assertEqual(set(by_pair.values()), {28})

            shuffled = deepcopy(results[0])
            random.Random(8675309).shuffle(shuffled["route_records"])
            validate_iteration2_graph_result(shuffled)

            missing = deepcopy(results[0])
            missing["route_records"].pop()
            with self.assertRaises(Iteration2ValidationError):
                validate_iteration2_graph_result(missing)
            wrong_count = deepcopy(results[0])
            wrong_count["routes_per_pair"] = 27
            with self.assertRaises(Iteration2ValidationError):
                validate_iteration2_graph_result(wrong_count)
            stale = deepcopy(results[0])
            stale["data_generation_hash"] = "0" * 64
            with self.assertRaises(Iteration2ValidationError):
                validate_iteration2_graph_result(stale)

            run_manifest = excluded_run_manifest(results, contract)
            (run_root / "run_manifest.json").write_text(
                json.dumps(run_manifest, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            checkpoint_paths = [
                publish_graph_checkpoint(graph_root, result, run_manifest)
                for result in results
            ]
            first_validation = [
                validate_checkpoint_directory(
                    path,
                    run_manifest=run_manifest,
                )
                for path in checkpoint_paths
            ]
            resumed_validation = [
                validate_checkpoint_directory(
                    path,
                    run_manifest=run_manifest,
                )
                for path in checkpoint_paths
            ]
            self.assertEqual(
                [row["payload_sha256"] for row in first_validation],
                [row["payload_sha256"] for row in resumed_validation],
            )
            loaded_results = [row["result"] for row in resumed_validation]

            tables = create_analysis_tables(
                loaded_results,
                analysis_validation_evidence=(
                    excluded_analysis_validation_evidence()
                ),
                bootstrap_replicates=2,
                require_complete_design=False,
            )
            self.assertEqual(set(tables), set(WORKBOOK_SHEETS))
            _add_reduced_inference_tables(tables)
            analysis_output = (
                temporary_path / "results" / contract.analysis_identity
            )
            raw_hashes = {
                f"graphs/{row['graph_id']}/{GRAPH_RESULT_FILENAME}": str(
                    row["payload_sha256"]
                )
                for row in resumed_validation
            }
            manifest = build_reporting_bundle(
                analysis_output,
                tables=tables,
                source_commit="1" * 40,
                raw_location="excluded/raw",
                raw_file_hashes=raw_hashes,
                limitations=(
                    "Excluded deterministic fixture; not scientific evidence.",
                ),
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
            self.assertEqual(validate_reporting_bundle(analysis_output), manifest)
            self.assertEqual(manifest["workbook_sheets"], list(WORKBOOK_SHEETS))
            self.assertEqual(manifest["figures"], list(FIGURE_FILES))
            self.assertTrue((analysis_output / "iteration2_results.xlsx").is_file())
            self.assertTrue((analysis_output / "analysis_complete.json").is_file())
            self.assertGreater(
                manifest["table_row_counts"]["Property Associations"],
                0,
            )
            self.assertGreater(
                manifest["table_row_counts"]["Matched Embedding Interactions"],
                0,
            )

            corrupt = analysis_output / "graph_metrics.csv"
            corrupt.write_bytes(corrupt.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_reporting_bundle(analysis_output)

        self.assertIsNotNone(temporary_path)
        self.assertFalse(temporary_path.exists())


class Iteration2ReadOnlyAnalysisIntegrationTests(unittest.TestCase):
    @staticmethod
    def _publish_fixture(root: Path) -> tuple[Path, ExcludedAnalysisFixtureContract]:
        results = excluded_feasibility_results()
        contract = excluded_fixture_contract(results)
        raw = root / "results" / contract.raw_identity
        publish_completed_excluded_raw_run(raw, results, contract)
        return raw, contract

    @staticmethod
    def _scientific_boundary_patches() -> tuple[object, ...]:
        sentinel = AssertionError("analysis reached a scientific execution boundary")
        return (
            patch("run_iteration2.execute_scheduled_graph", side_effect=sentinel),
            patch(
                "run_iteration2.regenerate_and_validate_checkpoint_scientific_result",
                side_effect=sentinel,
            ),
            patch(
                "iteration2_experiment.generate_connected_erdos_renyi",
                side_effect=sentinel,
            ),
            patch(
                "iteration2_experiment.generate_connected_barabasi_albert",
                side_effect=sentinel,
            ),
            patch("iteration2_experiment.sample_ordered_pairs", side_effect=sentinel),
            patch("iteration2_experiment.dijkstra_benchmark", side_effect=sentinel),
            patch(
                "iteration2_experiment.euclidean_greedy_route_v2",
                side_effect=sentinel,
            ),
            patch(
                "iteration2_experiment.poincare_greedy_route_v2",
                side_effect=sentinel,
            ),
            patch(
                "iteration2_experiment.repaired_poincare_greedy_route_v2",
                side_effect=sentinel,
            ),
            patch("run_iteration2.publish_graph_checkpoint", side_effect=sentinel),
        )

    def test_real_analysis_entry_is_read_only_and_executes_no_science(self):
        with TemporaryDirectory(prefix="iteration2-real-loader-") as temporary:
            root = Path(temporary)
            raw, contract = self._publish_fixture(root)
            before = compute_raw_tree_fingerprint(raw, include_entries=False)
            boundary_patches = self._scientific_boundary_patches()
            started = [item.start() for item in boundary_patches]
            try:
                with patch.object(
                    analysis_entry,
                    "_analysis_source_identity",
                    return_value={"source_commit": "1" * 40},
                ):
                    manifest = analysis_entry.analyze(
                        root,
                        excluded_fixture=contract,
                    )
            finally:
                for item in reversed(boundary_patches):
                    item.stop()
            after = compute_raw_tree_fingerprint(raw, include_entries=False)
            self.assertEqual(before, after)
            for mocked_boundary in started:
                mocked_boundary.assert_not_called()

            evidence = manifest["analysis_validation_evidence"]
            self.assertFalse(evidence["regeneration_requested"])
            self.assertEqual(
                evidence["scientific_graphs_executed_during_analysis"], 0
            )
            self.assertEqual(evidence["dijkstra_executions_during_analysis"], 0)
            self.assertEqual(evidence["routing_executions_during_analysis"], 0)
            self.assertEqual(evidence["raw_checkpoints_written_during_analysis"], 0)
            self.assertTrue(evidence["raw_tree_unchanged"])
            self.assertEqual(evidence["raw_tree_before"], before.summary())
            self.assertEqual(evidence["raw_tree_after"], after.summary())
            ledger = evidence["scientific_operation_ledger"]
            self.assertEqual(ledger["mode"], "analysis_read_only")
            self.assertEqual(ledger["total_attempted"], 0)
            self.assertEqual(ledger["total_executed"], 0)
            self.assertEqual(ledger["total_blocked"], 0)
            self.assertTrue(
                all(
                    value == 0
                    for value in ledger["attempted_operation_counts"].values()
                )
            )

            output = root / "results" / contract.analysis_identity
            with (output / "validation_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                validation = next(csv.DictReader(stream))
            self.assertEqual(
                validation["full_scientific_run_was_performed_by_analysis"],
                "False",
            )
            self.assertEqual(
                validation["scientific_graphs_executed_during_analysis"], "0"
            )
            self.assertEqual(
                validation["raw_checkpoints_written_during_analysis"], "0"
            )
            self.assertEqual(validate_reporting_bundle(output), manifest)

    def test_corrupt_checkpoint_fails_without_regeneration_or_rewrite(self):
        with TemporaryDirectory(prefix="iteration2-corrupt-loader-") as temporary:
            root = Path(temporary)
            raw, contract = self._publish_fixture(root)
            payload = next((raw / "graphs").glob("*/result.json.gz"))
            payload.write_bytes(payload.read_bytes() + b"corrupt")
            before = compute_raw_tree_fingerprint(raw, include_entries=False)
            boundary_patches = self._scientific_boundary_patches()
            started = [item.start() for item in boundary_patches]
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "corrupt Iteration 2 checkpoint",
                ):
                    load_complete_raw_run(root, excluded_fixture=contract)
            finally:
                for item in reversed(boundary_patches):
                    item.stop()
            self.assertEqual(
                before,
                compute_raw_tree_fingerprint(raw, include_entries=False),
            )
            for mocked_boundary in started:
                mocked_boundary.assert_not_called()

    def test_missing_or_incompatible_raw_identity_fails_closed(self):
        with TemporaryDirectory(prefix="iteration2-identity-loader-") as temporary:
            root = Path(temporary)
            raw, contract = self._publish_fixture(root)
            manifest_path = raw / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["analysis_plan_hash"]
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            before = compute_raw_tree_fingerprint(raw, include_entries=False)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                load_complete_raw_run(root, excluded_fixture=contract)
            self.assertEqual(
                before,
                compute_raw_tree_fingerprint(raw, include_entries=False),
            )

    def test_production_and_excluded_raw_identities_cannot_cross_load(self):
        with TemporaryDirectory(prefix="iteration2-cross-identity-") as temporary:
            root = Path(temporary)
            raw, contract = self._publish_fixture(root)
            production_path = root / "results" / ITERATION2_RUN_IDENTITY
            shutil.copytree(raw, production_path)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                load_complete_raw_run(root)

            mismatched = ExcludedAnalysisFixtureContract(
                fixture_tag="different_fixture_meaning",
                expected_graph_ids=contract.expected_graph_ids,
                excluded_seeds=contract.excluded_seeds,
                pair_count=contract.pair_count,
                bootstrap_replicates=contract.bootstrap_replicates,
                property_resampling_replicates=(
                    contract.property_resampling_replicates
                ),
                permutation_replicates=contract.permutation_replicates,
            )
            mismatched_path = root / "results" / mismatched.raw_identity
            shutil.copytree(raw, mismatched_path)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                load_complete_raw_run(root, excluded_fixture=mismatched)

    def test_analysis_cli_exposes_no_regeneration_mode(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                analysis_entry.main(["--regenerate-scientific-results"])
        source = Path(analysis_entry.__file__).read_text(encoding="utf-8")
        loader_source = source[
            source.index("def _validate_complete_raw_run_for_analysis") :
            source.index("def _identity")
        ]
        self.assertNotIn("execute_scheduled_graph", loader_source)
        self.assertNotIn("regenerate_and_validate", loader_source)


if __name__ == "__main__":
    unittest.main()
