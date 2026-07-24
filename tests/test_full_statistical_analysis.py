"""Focused tests for the frozen Step 17 graph-level statistical analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import analyze_full_experiment as analysis  # noqa: E402
from experiment_config import (  # noqa: E402
    BARABASI_ALBERT,
    ERDOS_RENYI,
)
from experiment_protocol import (  # noqa: E402
    bootstrap_graph_indices,
    partial_spearman_by_n_m_stratum,
    percentile_bootstrap_interval,
)
from validate_full_experiment import (  # noqa: E402
    COORDINATE_CONDITIONS,
    EXPECTED_RUN_DIRECTORY,
    ROUTING_METHODS,
    RawTreeFingerprint,
    ValidatedRun,
    _graph_rows_from_stats,
    compute_raw_tree_fingerprint,
)


def synthetic_grouped():
    grouped = {}
    for model_index, model in enumerate(analysis.MODELS):
        for n_index, n in enumerate(analysis.N_VALUES):
            for m_index, m in enumerate(analysis.M_VALUES):
                for condition_index, condition in enumerate(
                    COORDINATE_CONDITIONS
                ):
                    values = []
                    for replicate in range(20):
                        base = (
                            0.001 * replicate
                            + 0.01 * n_index
                            + 0.02 * m_index
                            + 0.03 * condition_index
                            + 0.04 * model_index
                        )
                        values.append(
                            {
                                "replicate_index": replicate,
                                "euclidean_success": 0.50 + base,
                                "poincare_success": 0.55 + base,
                                "repaired_success": 0.57 + base,
                                "poincare_advantage": 0.05,
                                "repair_improvement": 0.02,
                                "repair_recovery": 0.20,
                                "repair_recovery_numerator": 20,
                                "repair_recovery_denominator": 100,
                            }
                        )
                    grouped[(model, n, m, condition)] = tuple(values)
    return grouped


def cyclic_draws():
    row = np.arange(20, dtype=np.uint8)
    matrix = np.tile(row, (10_000, 1))
    return {
        (model, n, m): matrix
        for model in analysis.MODELS
        for n in analysis.N_VALUES
        for m in analysis.M_VALUES
    }


class GraphLevelAnalysisTests(unittest.TestCase):
    def test_cell_estimates_use_equal_graph_weights_and_percentage_points(self):
        rows = analysis.cell_estimates(synthetic_grouped(), cyclic_draws())
        self.assertEqual(len(rows), 90)
        first = rows[0]
        self.assertAlmostEqual(first["euclidean_success_estimate"], 0.5095)
        self.assertAlmostEqual(first["poincare_advantage_estimate"], 5.0)
        self.assertEqual(
            first["poincare_advantage_unit"], "percentage_points"
        )
        self.assertEqual(first["graph_count"], 20)

    def test_marginal_contrasts_weight_nine_strata_equally(self):
        rows = analysis.success_contrasts(
            synthetic_grouped(), cyclic_draws()
        )
        self.assertEqual(len(rows), 200)
        marginal = [
            row
            for row in rows
            if row["scope"] == "model_coordinate_n_m_marginal"
        ]
        self.assertEqual(len(marginal), 20)
        self.assertTrue(all(row["stratum_count"] == 9 for row in marginal))
        self.assertTrue(
            all(
                row["weighting"] == "equal_n_m_strata_then_equal_graph"
                for row in marginal
            )
        )

    def test_interaction_formula_is_within_graph_hydra_minus_mds(self):
        grouped = synthetic_grouped()
        for key, rows in list(grouped.items()):
            model, n, m, condition = key
            adjusted = []
            for row in rows:
                item = dict(row)
                item["poincare_advantage"] = (
                    0.08 if condition == "hydra" else 0.03
                )
                adjusted.append(item)
            grouped[key] = tuple(adjusted)
        rows = analysis.embedding_interactions(grouped, cyclic_draws())
        self.assertEqual(len(rows), 144)
        metric_rows = [
            row
            for row in rows
            if row["interaction_id"].endswith("poincare_advantage")
        ]
        self.assertTrue(
            all(abs(float(row["estimate"]) - 5.0) < 1e-12 for row in metric_rows)
        )
        self.assertTrue(all(row["pairing"] == "within_graph" for row in rows))

    def test_model_contrasts_use_independent_model_draws(self):
        rows = analysis.model_contrasts(
            synthetic_grouped(), cyclic_draws()
        )
        self.assertEqual(len(rows), 90)
        self.assertTrue(
            all(
                row["resampling"] == "independent_er_and_ba_strata"
                for row in rows
            )
        )
        self.assertNotEqual(
            bootstrap_graph_indices(
                bootstrap_replicate=7,
                model=ERDOS_RENYI,
                n=100,
                m=4,
            ),
            bootstrap_graph_indices(
                bootstrap_replicate=7,
                model=BARABASI_ALBERT,
                n=100,
                m=4,
            ),
        )

    def test_graph_recovery_null_and_stretch_denominators(self):
        route_stats = {}
        for condition in COORDINATE_CONDITIONS:
            for method in ROUTING_METHODS:
                route_stats[(condition, method)] = {
                    "success_count": 1_000,
                    "successful_route_length_sum": 1_100.0,
                    "successful_stretch_sum": 1_050.0,
                    "final_failure_counts": Counter(),
                }
        common = {
            condition: {
                "common_success_count": 1_000,
                "common_euclidean_stretch_sum": 1_050.0,
                "common_poincare_stretch_sum": 1_040.0,
                "recovered_count": 0,
                "recovered_repaired_stretch_sum": 0.0,
            }
            for condition in COORDINATE_CONDITIONS
        }
        entry = SimpleNamespace(
            graph_id="g",
            schedule_index=0,
            model=ERDOS_RENYI,
            n=100,
            m=4,
            replicate_index=0,
        )
        distortions = {
            "hydra_euclidean": {
                "mean_absolute_relative_distortion": 0.1
            },
            "hydra_poincare": {
                "mean_absolute_relative_distortion": 0.2
            },
            "base_mds_euclidean": {
                "mean_absolute_relative_distortion": 0.3
            },
            **{
                f"mds_poincare_r{suffix}": {
                    "mean_absolute_relative_distortion": 0.4
                }
                for suffix in ("050", "070", "085", "095")
            },
        }
        rows = _graph_rows_from_stats(
            entry=entry,
            network_metrics={
                "average_degree": 2.0,
                "maximum_degree": 3,
                "population_degree_variance": 1.0,
                "average_clustering_coefficient": 0.0,
                "diameter": 3,
                "average_shortest_path_length": 1.5,
                "number_of_edges": 100,
                "number_of_vertices": 100,
            },
            distortions=distortions,
            route_stats=route_stats,
            common_stats=common,
        )
        self.assertEqual(len(rows), 5)
        self.assertIsNone(rows[0]["repair_recovery"])
        self.assertEqual(rows[0]["repair_recovery_denominator"], 0)
        self.assertEqual(rows[0]["common_success_count"], 1_000)
        self.assertIsNone(rows[0]["recovered_repaired_stretch_mean"])


class BootstrapAndAssociationTests(unittest.TestCase):
    def test_nearest_rank_interval_is_noninterpolated(self):
        values = np.arange(10_000, dtype=np.float64)
        interval = percentile_bootstrap_interval(values)
        self.assertEqual(interval.lower, 249.0)
        self.assertEqual(interval.upper, 9749.0)
        self.assertIn("noninterpolated", interval.quantile_rule)

    def test_batch_adjusted_rank_correlations_match_scalar_definition(self):
        first = np.asarray([1, 2, 5, 4, 9, 7], dtype=np.float64)
        second = np.asarray([2, 1, 4, 5, 8, 6], dtype=np.float64)
        strata = [(100, 4), (100, 4), (300, 4), (300, 4), (1000, 4), (1000, 4)]
        expected = partial_spearman_by_n_m_stratum(first, second, strata)
        first_residual = analysis._batch_rank_residuals(
            first[None, :], stratum_size=2
        )
        second_residual = analysis._batch_rank_residuals(
            second[None, :], stratum_size=2
        )
        actual, defined = analysis._batch_residual_correlation(
            first_residual, second_residual
        )
        self.assertTrue(bool(defined[0]))
        self.assertAlmostEqual(float(actual[0]), expected, places=14)

    def test_zero_residual_variance_is_explicitly_undefined(self):
        constant_within_strata = np.asarray(
            [[1, 1, 2, 2, 3, 3]], dtype=np.float64
        )
        varying = np.asarray([[1, 2, 1, 2, 1, 2]], dtype=np.float64)
        first = analysis._batch_rank_residuals(
            constant_within_strata, stratum_size=2
        )
        second = analysis._batch_rank_residuals(varying, stratum_size=2)
        values, defined = analysis._batch_residual_correlation(first, second)
        self.assertFalse(bool(defined[0]))
        self.assertEqual(float(values[0]), 0.0)

    def test_bootstrap_stream_is_deterministic_and_paired_reusable(self):
        first = bootstrap_graph_indices(
            bootstrap_replicate=123,
            model=ERDOS_RENYI,
            n=300,
            m=8,
        )
        second = bootstrap_graph_indices(
            bootstrap_replicate=123,
            model=ERDOS_RENYI,
            n=300,
            m=8,
        )
        self.assertEqual(first, second)
        values_a = np.arange(20)
        values_b = values_a + 100
        self.assertTrue(
            np.array_equal(values_a[list(first)] + 100, values_b[list(second)])
        )


class RuntimeAndSerializationTests(unittest.TestCase):
    def test_runtime_summaries_use_truthful_end_to_end_field(self):
        records = []
        for model in analysis.MODELS:
            for n in analysis.N_VALUES:
                for index in range(60):
                    records.append(
                        {
                            "graph_id": f"{model}-{n}-{index}",
                            "model": model,
                            "n": n,
                            "m": analysis.M_VALUES[index % 3],
                            "replicate_index": index % 20,
                            "payload_serialization_ns": 10 + index,
                            "prepublication_wall_ns": 100 + index,
                            "atomic_publication_and_final_validation_ns": 20 + index,
                            "end_to_end_graph_wall_ns": 130 + index,
                            "actual_dijkstra_ns": 30 + index,
                        }
                    )
        rows = analysis.runtime_summaries(records)
        end_to_end = [
            row
            for row in rows
            if row["scope"] == "all_graphs"
            and row["component"] == "end_to_end_graph_wall_ns"
        ]
        self.assertEqual(len(end_to_end), 1)
        self.assertEqual(end_to_end[0]["runtime_role"], "end_to_end")
        prepublication = [
            row
            for row in rows
            if row["component"] == "prepublication_wall_ns"
        ]
        self.assertTrue(
            all(row["runtime_role"] != "end_to_end" for row in prepublication)
        )

    def test_finite_deterministic_json_csv_and_gzip(self):
        value = {"b": 2, "a": 1.5, "nullable": None}
        self.assertEqual(
            analysis.deterministic_json_bytes(value),
            analysis.deterministic_json_bytes(value),
        )
        rows = [{"a": 1.5, "b": None}]
        csv_bytes = analysis.deterministic_csv_bytes(rows, ("a", "b"))
        self.assertEqual(csv_bytes, b"a,b\n1.5,\n")
        self.assertEqual(
            analysis.deterministic_gzip_bytes(csv_bytes),
            analysis.deterministic_gzip_bytes(csv_bytes),
        )
        with self.assertRaises(analysis.StatisticalAnalysisError):
            analysis.deterministic_json_bytes({"bad": float("nan")})

    def test_atomic_publication_writes_only_separate_derived_directory(self):
        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            raw_root = output_root / EXPECTED_RUN_DIRECTORY
            raw_root.mkdir(parents=True)
            (raw_root / "run_manifest.json").write_text(
                json.dumps({"dependency_versions": {"numpy": "2.4.6"}}),
                encoding="utf-8",
            )
            raw_before = compute_raw_tree_fingerprint(raw_root)
            graph_rows = []
            for model in analysis.MODELS:
                for n in analysis.N_VALUES:
                    for m in analysis.M_VALUES:
                        for replicate in range(20):
                            for condition in COORDINATE_CONDITIONS:
                                graph_rows.append(
                                    {
                                        "model": model,
                                        "n": n,
                                        "m": m,
                                        "replicate_index": replicate,
                                        "coordinate_condition_id": condition,
                                    }
                                )
            validated = ValidatedRun(
                validation_report={
                    "validation_passed": True,
                    "raw_run_manifest_sha256": "a" * 64,
                },
                graph_level_rows=tuple(graph_rows),
                runtime_records=(),
                initial_raw_fingerprint=raw_before,
            )

            def rows(count):
                return [{"row_index": index} for index in range(count)]

            original_replace = analysis.os.replace
            replace_calls = []

            def recording_replace(source, destination):
                replace_calls.append((Path(source), Path(destination)))
                return original_replace(source, destination)

            with (
                patch.object(analysis, "build_bootstrap_draws", return_value={}),
                patch.object(analysis, "cell_estimates", return_value=rows(90)),
                patch.object(
                    analysis, "success_contrasts", return_value=rows(200)
                ),
                patch.object(
                    analysis, "embedding_interactions", return_value=rows(144)
                ),
                patch.object(
                    analysis, "model_contrasts", return_value=rows(90)
                ),
                patch.object(
                    analysis, "failure_summaries", return_value=rows(1_620)
                ),
                patch.object(
                    analysis, "stretch_summaries", return_value=rows(540)
                ),
                patch.object(
                    analysis, "property_correlations", return_value=rows(160)
                ),
                patch.object(
                    analysis, "runtime_summaries", return_value=rows(5)
                ),
                patch.object(
                    analysis,
                    "_analysis_summary",
                    return_value={"status": "synthetic_test"},
                ),
                patch.object(analysis.os, "replace", side_effect=recording_replace),
            ):
                target, manifest, manifest_hash = analysis.publish_analysis(
                    raw_run_root=raw_root,
                    output_root=output_root,
                    validated=validated,
                    repository_root=PROJECT_ROOT,
                    timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            self.assertTrue(target.is_dir())
            self.assertEqual(len(replace_calls), 1)
            self.assertTrue(replace_calls[0][0].name.startswith(".analysis_"))
            self.assertEqual(replace_calls[0][1], target)
            self.assertTrue(manifest["validation_passed"])
            self.assertEqual(len(manifest_hash), 64)
            self.assertEqual(raw_before, compute_raw_tree_fingerprint(raw_root))
            self.assertFalse(any(".tmp-" in path.name for path in output_root.iterdir()))


if __name__ == "__main__":
    unittest.main()
