from __future__ import annotations

import copy
from pathlib import Path
import random
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from analyze_iteration2 import _graph_output_tables  # noqa: E402
from iteration2_analysis import (  # noqa: E402
    _simultaneous_rows,
    aggregate_graph_metrics,
    aggregate_interactions,
    graph_level_rows,
    property_associations,
    simultaneous_radius_bands,
)
from iteration2_v2_support import (  # noqa: E402
    deterministic_bootstrap_provider,
    synthetic_inference_rows,
)
from run_iteration2 import excluded_feasibility_results  # noqa: E402
from validate_iteration2 import (  # noqa: E402
    Iteration2ValidationError,
    validate_iteration2_graph_result,
)


class Iteration2AnalysisIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = excluded_feasibility_results()[0]

    def test_route_shuffle_is_invariant_because_joins_use_explicit_identities(self):
        original = self.result
        shuffled_routes = copy.deepcopy(original["route_records"])
        random.Random(8675309).shuffle(shuffled_routes)
        identity = original["graph_identity"]
        recomputed = graph_level_rows(
            shuffled_routes,
            graph_id=identity["graph_id"],
            model=identity["model"],
            n=identity["n"],
            m=identity["m"],
            replicate_index=identity["replicate_index"],
            pair_count=len(original["pairs"]),
        )
        self.assertEqual(recomputed, original["graph_level_rows"])
        shuffled_result = copy.deepcopy(original)
        shuffled_result["route_records"] = shuffled_routes
        self.assertEqual(
            validate_iteration2_graph_result(shuffled_result)["route_records"],
            len(shuffled_routes),
        )

    def test_duplicate_missing_and_inconsistent_route_identities_fail_closed(self):
        duplicate = copy.deepcopy(self.result)
        duplicate["route_records"][1] = copy.deepcopy(
            duplicate["route_records"][0]
        )
        with self.assertRaisesRegex(
            Iteration2ValidationError, "missing or duplicated"
        ):
            validate_iteration2_graph_result(duplicate)

        inconsistent = copy.deepcopy(self.result)
        inconsistent["route_records"][0]["source"] = (
            inconsistent["route_records"][0]["destination"]
        )
        with self.assertRaisesRegex(
            Iteration2ValidationError, "route structural mismatch"
        ):
            validate_iteration2_graph_result(inconsistent)

    def test_graph_outputs_keep_failure_stages_and_repair_denominators_separate(self):
        tables = _graph_output_tables(self.result["graph_level_rows"])
        repair_names = {row["estimand"] for row in tables["Repair Denominators"]}
        self.assertEqual(
            repair_names,
            {
                "overall_repaired_minus_ordinary_success",
                "recovered_over_all_ordinary_failures",
                "backtrackable_over_all_ordinary_failures",
                "alternative_available_over_backtrackable_failures",
                "recovered_over_repair_eligible_failures",
                "recovered_over_selected_alternatives",
            },
        )
        self.assertEqual(
            {row["failure_stage"] for row in tables["Failure Composition"]},
            {"initial", "final"},
        )
        self.assertEqual(
            {row["estimand"] for row in tables["Failure Composition"]},
            {
                "failure_rate_all_sampled_pairs",
                "composition_among_failures_at_stage",
            },
        )
        for row in tables["Failure Composition"]:
            if row["status"] == "defined":
                self.assertEqual(
                    row["estimate"],
                    row["numerator"] / row["denominator"],
                )
        self.assertFalse(
            any(row["invariant_error_category"] for row in tables["Failure Composition"])
        )
        self.assertEqual(
            {row["distance_band"] for row in tables["Distance Bands"]},
            {"1", "2", "3", "4", ">=5"},
        )
        for table_name in (
            "Repair Denominators",
            "Failure Composition",
            "Distance Bands",
            "Method Stretch",
            "Common-Success Stretch",
            "Recovered Stretch",
            "Routing Resources",
        ):
            for row in tables[table_name]:
                self.assertTrue(
                    {
                        "numerator",
                        "denominator",
                        "graph_count",
                        "pair_count",
                        "status",
                        "applicability",
                        "na_reason",
                    }
                    <= set(row),
                    (table_name, row),
                )


class Iteration2ClusteredInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.graph_metrics,
            cls.graph_rows,
            cls.interactions,
        ) = synthetic_inference_rows(replicates_per_stratum=2)

    def test_whole_graph_bootstrap_and_equal_stratum_marginals(self):
        rows = aggregate_graph_metrics(
            self.graph_rows,
            metrics=(
                "poincare_minus_euclidean",
                "common_success_poincare_minus_euclidean_stretch",
            ),
            bootstrap_replicates=6,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        marginals = [
            row for row in rows if row["scope"] == "model_condition_n_m_marginal"
        ]
        self.assertEqual(len(marginals), 40)
        self.assertTrue(all(row["stratum_count"] == 9 for row in marginals))
        self.assertTrue(
            all(row["weighting"] == "equal_n_m_strata_then_equal_graph" for row in marginals)
        )
        self.assertTrue(all(row["independent_unit"] == "graph" for row in rows))

    def test_simultaneous_bands_pair_all_four_radii_by_graph(self):
        pointwise = aggregate_interactions(
            self.interactions,
            bootstrap_replicates=6,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        simultaneous = simultaneous_radius_bands(
            self.interactions,
            bootstrap_replicates=6,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        self.assertTrue(pointwise)
        marginal = [row for row in simultaneous if row["stratum_count"] == 9]
        self.assertEqual(len(marginal), 8)
        self.assertEqual({row["matched_radius"] for row in marginal}, {0.5, 0.7, 0.85, 0.95})
        self.assertTrue(
            all(row["ci_type"] == "simultaneous_95_percent_familywise" for row in marginal)
        )
        self.assertTrue(
            all(
                {
                    "ci_excludes_zero",
                    "ci_wholly_inside_margin",
                    "ci_wholly_positive_beyond_margin",
                    "ci_wholly_negative_beyond_margin",
                    "practical_magnitude_unresolved",
                }
                <= set(row)
                for row in marginal
            )
        )

    def test_simultaneous_critical_value_is_unstandardized_max_absolute_deviation(self):
        point = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
        magnitudes = np.arange(1, 21, dtype=np.float64) / 1_000.0
        signs = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        boot = np.stack(
            [point + magnitude * signs for magnitude in magnitudes]
        )
        independent_deviations = np.max(
            np.abs(boot - point[None, :]), axis=1
        )
        expected_critical = np.sort(
            independent_deviations, kind="stable"
        )[int(np.ceil(0.95 * len(independent_deviations))) - 1]
        rows = _simultaneous_rows(
            point,
            boot,
            scope="toy_four_radius_family",
            model="erdos_renyi",
            n=None,
            m=None,
            graph_count=20,
            pair_count=20_000,
            stratum_count=9,
            bootstrap_replicates=20,
        )
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(expected_critical, 0.019)
        for index, row in enumerate(rows):
            expected_point = point[index] * 100.0
            expected_critical_pp = expected_critical * 100.0
            self.assertAlmostEqual(
                row["simultaneous_critical_value"], expected_critical_pp
            )
            self.assertAlmostEqual(
                row["simultaneous_ci_lower"],
                expected_point - expected_critical_pp,
            )
            self.assertAlmostEqual(
                row["simultaneous_ci_upper"],
                expected_point + expected_critical_pp,
            )
    def test_property_inference_has_four_exact_shared_max_stat_families(self):
        rows = property_associations(
            self.graph_metrics,
            self.graph_rows,
            inference_replicates=6,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        self.assertEqual(len(rows), 216)
        families = {}
        for row in rows:
            families.setdefault(row["family_id"], []).append(row)
            self.assertFalse(row["causal_interpretation"])
            self.assertEqual(row["multiplicity_procedure"], "maximum_absolute_statistic_familywise")
            if row["raw_permutation_p_value"] is not None:
                self.assertGreaterEqual(
                    row["familywise_adjusted_p_value"],
                    row["raw_permutation_p_value"],
                )
        self.assertEqual(
            set(families),
            {
                "er_success_contrast",
                "ba_success_contrast",
                "er_common_success_stretch_contrast",
                "ba_common_success_stretch_contrast",
            },
        )
        for family in families.values():
            self.assertEqual(len(family), 54)
            self.assertEqual(
                len({row["multiplicity_family_fingerprint"] for row in family}),
                1,
            )
            self.assertTrue(
                all(row["permutation_mapping"] == "shared_across_all_54_family_hypotheses" for row in family)
            )

    def test_zero_within_stratum_property_variance_is_explicit_na(self):
        metrics = copy.deepcopy(self.graph_metrics)
        for row in metrics:
            row["average_degree"] = float(row["n"] + row["m"])
        associations = property_associations(
            metrics,
            self.graph_rows,
            inference_replicates=4,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        selected = [row for row in associations if row["property"] == "average_degree"]
        self.assertEqual(len(selected), 36)
        self.assertTrue(all(row["association_estimate"] is None for row in selected))
        self.assertTrue(
            all(
                row["na_reason"]
                == "zero_within_stratum_residual_variance_predictor"
                for row in selected
            )
        )


if __name__ == "__main__":
    unittest.main()
