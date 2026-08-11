from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import analyze_iteration2 as analysis_entry  # noqa: E402
from analyze_iteration2 import create_analysis_tables  # noqa: E402
from iteration2_analysis import (  # noqa: E402
    aggregate_graph_metrics,
    aggregate_interactions,
    descriptive_model_contrasts,
    property_associations,
    simultaneous_radius_bands,
)
from iteration2_config import (  # noqa: E402
    ITERATION2_ANALYSIS_IDENTITY,
    ITERATION2_RUN_IDENTITY,
    OUTPUT_FIGURE_FILES,
    OUTPUT_WORKBOOK_SHEETS,
    REPORTING_TABLE_SCHEMA_HASH,
    fingerprint,
    output_schema_payload,
)
from iteration2_reporting import (  # noqa: E402
    FIGURE_FILES,
    SEMANTIC_DEFINITION_REGISTRY,
    TABLE_SCHEMA_COLUMNS,
    WORKBOOK_SHEETS,
    _cell_xml,
    _annotation_text_colour,
    _contrast_ratio,
    _definition_is_tautological,
    _field_unit,
    _forest,
    _save_empty_or_faceted_points,
    build_reporting_bundle,
    data_dictionary,
    generate_required_figures,
    validate_semantic_definition_registry,
    validate_reporting_bundle,
)
from iteration2_v2_support import (  # noqa: E402
    deterministic_bootstrap_provider,
    excluded_analysis_validation_evidence,
    excluded_reporting_contract,
    synthetic_inference_rows,
)
from run_iteration2 import excluded_feasibility_results  # noqa: E402


class Iteration2ReportingSchemaTests(unittest.TestCase):
    def test_faceted_layout_reflows_under_inherited_large_font_state(self):
        rows = [
            {"model": model, "n": 14, "total_seconds": seconds}
            for model, seconds in (
                ("barabasi_albert", 0.151),
                ("erdos_renyi", 0.245),
            )
        ]
        with matplotlib.rc_context({"font.size": 14.0}):
            with TemporaryDirectory() as temporary:
                output = Path(temporary) / "runtime.png"
                layout = _save_empty_or_faceted_points(
                    rows,
                    output=output,
                    title="Operational total runtime by graph model",
                    group_fields=("model",),
                    x_field="n",
                    y_field="total_seconds",
                    x_label="n",
                    y_label="seconds",
                )
                self.assertTrue(output.is_file())
        self.assertGreater(layout["layout_expansion_attempts"], 0)
        self.assertTrue(layout["suptitle_inside_canvas"])
        self.assertTrue(layout["panel_titles_inside_canvas"])
        self.assertTrue(layout["labels_inside_canvas"])
        self.assertEqual(layout["suptitle_panel_title_intersection_count"], 0)
        self.assertEqual(layout["adjacent_panel_title_intersection_count"], 0)

    def test_populated_distortion_long_titles_and_labels_fit(self):
        rows = []
        diagnostic_index = 0
        for model in ("erdos_renyi", "barabasi_albert"):
            for geometry in ("euclidean", "poincare"):
                for metric in (
                    "scale_fitted_mean_relative_error",
                    "spectral_stress_with_complete_condition_identity",
                ):
                    for radius in (0.50, 0.70, 0.85, 0.95):
                        rows.append(
                            {
                                "model": model,
                                "geometry": geometry,
                                "metric": metric,
                                "diagnostic_index": diagnostic_index,
                                "estimate": 0.01 * (diagnostic_index + 1),
                                "coordinate_condition_id": (
                                    f"excluded_long_condition_radius_{radius:.2f}"
                                ),
                            }
                        )
                        diagnostic_index += 1
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "distortion.png"
            layout = _save_empty_or_faceted_points(
                rows,
                output=output,
                title=(
                    "Geometry-specific distortion diagnostics "
                    "(exploratory, noncausal)"
                ),
                group_fields=("model", "geometry", "metric"),
                x_field="diagnostic_index",
                y_field="estimate",
                x_label="condition-stratum index (see complete table)",
                y_label="diagnostic value on metric-specific natural scale",
            )
            self.assertTrue(output.is_file())
        self.assertEqual(layout["row_count"], len(rows))
        self.assertEqual(layout["panel_count"], 8)
        self.assertTrue(layout["suptitle_inside_canvas"])
        self.assertTrue(layout["panel_titles_inside_canvas"])
        self.assertTrue(layout["labels_inside_canvas"])
        self.assertEqual(layout["suptitle_panel_title_intersection_count"], 0)
        self.assertEqual(layout["adjacent_panel_title_intersection_count"], 0)

    def test_exact_reporting_schema_is_bound_into_the_output_identity(self):
        payload = output_schema_payload()
        self.assertEqual(payload["workbook"]["sheets"], WORKBOOK_SHEETS)
        self.assertEqual(payload["figures"]["files"], FIGURE_FILES)
        self.assertEqual(OUTPUT_WORKBOOK_SHEETS, WORKBOOK_SHEETS)
        self.assertEqual(OUTPUT_FIGURE_FILES, FIGURE_FILES)
        self.assertEqual(
            fingerprint(
                {
                    name: list(columns)
                    for name, columns in TABLE_SCHEMA_COLUMNS.items()
                }
            ),
            REPORTING_TABLE_SCHEMA_HASH,
        )

    def test_schema_dictionary_is_complete_and_substantive(self):
        graph_metrics, graph_rows, interactions = synthetic_inference_rows()
        aggregate = aggregate_graph_metrics(
            graph_rows,
            metrics=(
                "poincare_minus_euclidean",
                "common_success_poincare_minus_euclidean_stretch",
            ),
            bootstrap_replicates=2,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        interaction_rows = [
            *aggregate_interactions(
                interactions,
                bootstrap_replicates=2,
                bootstrap_provider=deterministic_bootstrap_provider,
            ),
            *simultaneous_radius_bands(
                interactions,
                bootstrap_replicates=2,
                bootstrap_provider=deterministic_bootstrap_provider,
            ),
        ]
        tables = {name: [] for name in WORKBOOK_SHEETS}
        tables["Cell Estimates"] = [
            row for row in aggregate if row["stratum_count"] == 1
        ]
        tables["Model Marginals"] = [
            row for row in aggregate if row["stratum_count"] == 9
        ]
        tables["Matched Success Contrasts"] = tables["Cell Estimates"]
        tables["Common-Success Stretch"] = [
            row
            for row in aggregate
            if row["metric"]
            == "common_success_poincare_minus_euclidean_stretch"
        ]
        tables["Matched Embedding Interactions"] = interaction_rows
        tables["Equivalence Sensitivity"] = interaction_rows
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
        dictionary = data_dictionary(tables)
        self.assertEqual(
            len(dictionary),
            sum(len(columns) for columns in TABLE_SCHEMA_COLUMNS.values()),
        )
        validate_semantic_definition_registry()
        covered = {(row["table"], row["column"]) for row in dictionary}
        for table, columns in TABLE_SCHEMA_COLUMNS.items():
            self.assertTrue(
                {(table, column) for column in columns} <= covered,
                table,
            )
        self.assertTrue(
            all(
                row["definition"]
                and row["unit"]
                and row["conditioning"]
                and row["denominator"]
                and row["applicability"]
                and row["missing_value_rule"]
                for row in dictionary
            )
        )
        self.assertTrue(
            all(
                not _definition_is_tautological(
                    str(row["column"]), str(row["definition"])
                )
                for row in dictionary
            )
        )

    def test_property_association_flag_has_only_noncausal_statistical_meaning(self):
        semantic = SEMANTIC_DEFINITION_REGISTRY[
            ("Property Associations", "associative")
        ]
        definition = semantic["definition"].lower()
        self.assertIn("statistical association only", definition)
        self.assertIn("prohibits causal interpretation", definition)
        self.assertIn("does not describe the mathematical associative law", definition)
        self.assertNotIn("combination operation", definition)
        self.assertNotIn("associativity test", definition)
        self.assertEqual(semantic["unit"], "boolean")
        self.assertIn("each emitted property-association", semantic["conditioning"].lower())
        self.assertIn("not applicable", semantic["denominator"].lower())
        self.assertIn("every property-association row", semantic["applicability"].lower())
        self.assertIn("never missing", semantic["missing_value_rule"].lower())

        graph_metrics, graph_rows, _ = synthetic_inference_rows()
        rows = property_associations(
            graph_metrics,
            graph_rows,
            inference_replicates=2,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        self.assertTrue(rows)
        self.assertTrue(all(row["associative"] is True for row in rows))
        self.assertTrue(all(row["exploratory"] is True for row in rows))
        self.assertTrue(all(row["causal_interpretation"] is False for row in rows))

    def test_simultaneous_critical_value_dictionary_is_unstandardized(self):
        for table in (
            "Matched Embedding Interactions",
            "Equivalence Sensitivity",
        ):
            semantic = SEMANTIC_DEFINITION_REGISTRY[
                (table, "simultaneous_critical_value")
            ]
            definition = semantic["definition"].lower()
            words = set(definition.replace("-", " ").split())
            self.assertIn("absolute unstandardized deviation", definition)
            self.assertIn("non-interpolated", definition)
            self.assertNotIn("standardized", words)
            self.assertNotIn("studentized", words)
            self.assertEqual(semantic["unit"], "percentage_points")
            self.assertIn("four matched radii", semantic["conditioning"].lower())
            self.assertIn("neither studentisation nor standardisation", semantic["conditioning"].lower())
            self.assertIn("not applicable", semantic["denominator"].lower())
            self.assertIn("simultaneous matched-radius", semantic["applicability"].lower())

    def test_failure_annotations_choose_accessible_contrast_for_zero_bright_and_na(self):
        colour_map = matplotlib.colormaps["viridis"].with_extremes(bad="#bdbdbd")
        zero_background = colour_map(0.0)
        bright_background = colour_map(1.0)
        na_background = colour_map(np.ma.masked)
        zero_colour, zero_ratio = _annotation_text_colour(zero_background)
        bright_colour, bright_ratio = _annotation_text_colour(bright_background)
        na_colour, na_ratio = _annotation_text_colour(na_background)
        self.assertEqual(zero_colour, "white")
        self.assertEqual(bright_colour, "#111111")
        self.assertEqual(na_colour, "#111111")
        self.assertGreaterEqual(zero_ratio, 4.5)
        self.assertGreaterEqual(bright_ratio, 4.5)
        self.assertGreaterEqual(na_ratio, 4.5)
        self.assertNotEqual(zero_background, na_background)
        self.assertGreaterEqual(
            _contrast_ratio(matplotlib.colors.to_rgba(zero_colour), zero_background),
            4.5,
        )
    def test_missing_generic_tautological_and_incomplete_semantics_fail(self):
        identity = ("Graph Metrics", "graph_model")
        original = SEMANTIC_DEFINITION_REGISTRY[identity]
        mutations = (
            ("missing definition", {**original, "definition": ""}, "empty"),
            (
                "generic fallback",
                {
                    **original,
                    "definition": (
                        "Graph model recorded for one independent graph and its "
                        "pair-sampling diagnostics."
                    ),
                },
                "restates",
            ),
            (
                "tautology",
                {**original, "definition": "Graph model."},
                "restates",
            ),
        )
        for label, replacement, message in mutations:
            with self.subTest(label=label), patch.dict(
                SEMANTIC_DEFINITION_REGISTRY,
                {identity: replacement},
            ):
                with self.assertRaisesRegex(ValueError, message):
                    data_dictionary({"Graph Metrics": []})

        conditional = ("Failure Composition", "estimate")
        original_conditional = SEMANTIC_DEFINITION_REGISTRY[conditional]
        for field in (
            "unit",
            "conditioning",
            "denominator",
            "applicability",
            "missing_value_rule",
        ):
            with self.subTest(field=field), patch.dict(
                SEMANTIC_DEFINITION_REGISTRY,
                {conditional: {**original_conditional, field: ""}},
            ):
                with self.assertRaisesRegex(ValueError, field):
                    data_dictionary({"Failure Composition": []})

    def test_unknown_output_columns_fail_instead_of_receiving_generic_text(self):
        with self.assertRaisesRegex(ValueError, "undocumented reporting column"):
            data_dictionary({"Cell Estimates": [{"silent_extra": 1}]})

    def test_percentage_points_are_converted_exactly_once_and_correlations_are_natural(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "forest.png"
            rows = (
                {
                    "estimate": 0.1,
                    "ci_lower": 0.05,
                    "ci_upper": 0.15,
                    "unit": "proportion",
                },
                {
                    "estimate": 10.0,
                    "ci_lower": 5.0,
                    "ci_upper": 15.0,
                    "unit": "percentage_points",
                },
            )
            with patch("matplotlib.axes.Axes.errorbar") as errorbar:
                _forest(
                    rows,
                    estimate="estimate",
                    lower="ci_lower",
                    upper="ci_upper",
                    output=output,
                    title="unit fixture",
                    axis_unit="percentage_points",
                )
            points = np.asarray(errorbar.call_args.args[0])
            self.assertTrue(np.array_equal(points, np.asarray([10.0, 10.0])))

            with patch("matplotlib.axes.Axes.errorbar") as errorbar:
                _forest(
                    (
                        {
                            "estimate": -0.4,
                            "ci_lower": -0.6,
                            "ci_upper": -0.2,
                            "unit": "correlation",
                        },
                    ),
                    estimate="estimate",
                    lower="ci_lower",
                    upper="ci_upper",
                    output=output,
                    title="correlation fixture",
                    axis_unit="correlation",
                )
            self.assertEqual(float(errorbar.call_args.args[0][0]), -0.4)

    def test_workbook_number_formats_follow_each_column_not_the_row_label(self):
        mixed = {"unit": "mixed_estimands_see_named_fields"}
        self.assertIn(
            's="3"',
            _cell_xml(
                2,
                1,
                0.5,
                header="euclidean_success",
                row=mixed,
            ),
        )
        self.assertNotIn(
            's="3"',
            _cell_xml(
                2,
                1,
                0.25,
                header="common_success_poincare_minus_euclidean_stretch",
                row=mixed,
            ),
        )
        self.assertNotIn(
            's="3"',
            _cell_xml(
                2,
                1,
                0.5,
                header="matched_radius",
                row={"unit": "proportion"},
            ),
        )
        self.assertIn(
            's="3"',
            _cell_xml(
                2,
                1,
                0.5,
                header="estimate",
                row={"unit": "proportion"},
            ),
        )
        self.assertNotIn(
            's="3"',
            _cell_xml(
                2,
                1,
                -0.4,
                header="correlation",
                row={"unit": "correlation"},
            ),
        )
        self.assertEqual(_field_unit("correlation"), "correlation_-1_to_1")
        self.assertEqual(_field_unit("matched_radius"), "disk_radius")

    def test_figure_coverage_has_no_thirty_row_or_first_condition_truncation(self):
        tables = {name: [] for name in WORKBOOK_SHEETS}
        tables["Matched Success Contrasts"] = [
            {
                "model": "erdos_renyi",
                "n": 100,
                "m": 4,
                "coordinate_condition_id": f"excluded_condition_{index}",
                "metric": "poincare_minus_euclidean",
                "estimate": float(index) / 10.0,
                "ci_lower": float(index) / 10.0 - 0.1,
                "ci_upper": float(index) / 10.0 + 0.1,
                "unit": "percentage_points",
            }
            for index in range(31)
        ]
        tables["Cell Estimates"] = [
            {
                "scope": "model_condition_n_m_cell",
                "model": model,
                "n": 100,
                "m": 4,
                "coordinate_condition_id": condition,
                "embedding_family": condition.split("_", 1)[0],
                "condition_kind": "matched_radius",
                "matched_radius": 0.95,
                "metric": "poincare_success",
                "estimate": 0.5,
            }
            for model in ("erdos_renyi", "barabasi_albert")
            for condition in ("hydra_scaled_r050", "mds_scaled_r050")
        ]
        with TemporaryDirectory() as temporary:
            paths = generate_required_figures(temporary, tables)
            coverage = json.loads(
                (Path(temporary) / "figure_coverage.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(tuple(path.name for path in paths), FIGURE_FILES)
        self.assertEqual(
            coverage["poincare_minus_euclidean_forest.png"]["row_count"],
            31,
        )
        self.assertEqual(
            coverage["n_m_success_heatmaps.png"]["panel_count"],
            4,
        )
        self.assertNotIn(
            "first",
            json.dumps(coverage, sort_keys=True).lower(),
        )

    def test_populated_heatmap_titles_are_inside_and_do_not_intersect(self):
        _, graph_rows, _ = synthetic_inference_rows()
        estimates = aggregate_graph_metrics(
            graph_rows,
            metrics=("poincare_success",),
            bootstrap_replicates=2,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        tables = {name: [] for name in WORKBOOK_SHEETS}
        tables["Cell Estimates"] = [
            row for row in estimates if row["stratum_count"] == 1
        ]
        with TemporaryDirectory(prefix="iteration2-populated-heatmap-") as temporary:
            generate_required_figures(temporary, tables)
            coverage = json.loads(
                (Path(temporary) / "figure_coverage.json").read_text(
                    encoding="utf-8"
                )
            )["n_m_success_heatmaps.png"]
            self.assertGreater(
                (Path(temporary) / "n_m_success_heatmaps.png").stat().st_size,
                0,
            )
        self.assertEqual(coverage["row_count"], 162)
        self.assertEqual(coverage["panel_count"], 18)
        self.assertEqual(len(coverage["panel_title_bboxes_pixels"]), 18)
        self.assertTrue(coverage["suptitle_inside_canvas"])
        self.assertTrue(coverage["panel_titles_inside_canvas"])
        self.assertEqual(
            coverage["suptitle_panel_title_intersection_count"],
            0,
        )

    def test_common_success_forest_long_labels_fit_and_preserve_full_identities(self):
        tables = {name: [] for name in WORKBOOK_SHEETS}
        tables["Common-Success Stretch"] = [
            {
                "model": model,
                "n": n,
                "m": m,
                "coordinate_condition_id": condition,
                "metric": (
                    "common_success_poincare_minus_euclidean_stretch"
                ),
                "estimate": 0.01,
                "ci_lower": -0.01,
                "ci_upper": 0.03,
                "unit": "hop_stretch_ratio",
            }
            for model in ("erdos_renyi", "barabasi_albert")
            for n in (100, 300, 1_000)
            for m in (4, 8, 16)
            for condition in (
                "hydra_scaled_r095",
                "mds_scaled_r095",
            )
        ]
        expected_rows = len(tables["Common-Success Stretch"])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_required_figures(root, tables)
            image_path = root / "success_stretch_paired.png"
            image = plt.imread(image_path)
            coverage = json.loads(
                (root / "figure_coverage.json").read_text(encoding="utf-8")
            )["success_stretch_paired.png"]
            self.assertGreater(image_path.stat().st_size, 0)
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[1], 0)

        self.assertEqual(coverage["row_count"], expected_rows)
        self.assertEqual(coverage["rendered_row_count"], expected_rows)
        self.assertEqual(len(coverage["display_labels"]), expected_rows)
        self.assertEqual(len(coverage["full_identities"]), expected_rows)
        self.assertTrue(coverage["title_inside_canvas"])
        self.assertTrue(coverage["ytick_labels_inside_canvas"])
        self.assertTrue(coverage["axes_do_not_overlap_title"])
        self.assertEqual(
            coverage["full_identity_fields"],
            [
                "scope",
                "model",
                "n",
                "m",
                "coordinate_condition_id",
                "embedding_family",
                "condition_kind",
                "matched_radius",
                "metric",
            ],
        )
        self.assertEqual(
            {row["coordinate_condition_id"] for row in coverage["full_identities"]},
            {"hydra_scaled_r095", "mds_scaled_r095"},
        )

    def test_absolute_success_and_failure_heatmap_encodings_are_explicit(self):
        graph_metrics, graph_rows, _ = synthetic_inference_rows()
        estimates = aggregate_graph_metrics(
            graph_rows,
            metrics=(
                "euclidean_success",
                "poincare_success",
                "repaired_poincare_success",
            ),
            bootstrap_replicates=2,
            bootstrap_provider=deterministic_bootstrap_provider,
        )
        tables = create_analysis_tables(
            excluded_feasibility_results(),
            analysis_validation_evidence=excluded_analysis_validation_evidence(),
            bootstrap_replicates=2,
            require_complete_design=False,
        )
        tables["Model Marginals"] = [
            row for row in estimates if row["stratum_count"] == 9
        ]
        with TemporaryDirectory() as temporary:
            generate_required_figures(temporary, tables)
            coverage = json.loads(
                (Path(temporary) / "figure_coverage.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreater(
                (Path(temporary) / "absolute_success.png").stat().st_size,
                0,
            )
            self.assertGreater(
                (Path(temporary) / "failure_composition.png").stat().st_size,
                0,
            )

        absolute = coverage["absolute_success.png"]
        self.assertEqual(absolute["model_facet_field"], "model")
        self.assertEqual(
            absolute["coordinate_condition_field"],
            "coordinate_condition_id",
        )
        self.assertEqual(absolute["embedding_family_field"], "embedding_family")
        self.assertEqual(absolute["radius_field"], "matched_radius")
        self.assertEqual(absolute["condition_role_field"], "condition_kind")
        self.assertEqual(absolute["estimate_field"], "estimate")
        self.assertEqual(absolute["interval_fields"], ["ci_lower", "ci_upper"])
        self.assertEqual(
            set(absolute["method_encodings"]),
            {
                "euclidean_success",
                "poincare_success",
                "repaired_poincare_success",
            },
        )
        self.assertEqual(absolute["applicable_point_count"], 56)
        self.assertEqual(absolute["na_annotation_count"], 4)
        self.assertIn("never_zero", absolute["na_policy"])
        self.assertEqual(len(absolute["deterministic_condition_order"]), 10)

        failure = coverage["failure_composition.png"]
        for field, expected in {
            "model_field": "model",
            "method_field": "method_id",
            "coordinate_condition_field": "coordinate_condition_id",
            "embedding_family_field": "embedding_family",
            "radius_field": "matched_radius",
            "condition_role_field": "condition_kind",
            "failure_stage_field": "failure_stage",
            "failure_category_field": "failure_type",
            "numerator_field": "numerator",
            "denominator_field": "denominator",
        }.items():
            self.assertEqual(failure[field], expected)
        self.assertEqual(failure["matrix_shape_per_panel"], [10, 6])
        self.assertEqual(failure["panel_count"], 6)
        self.assertGreater(failure["defined_cell_count"], 0)
        self.assertGreater(failure["na_cell_count"], 0)
        self.assertGreater(failure["zero_cell_count"], 0)
        self.assertGreaterEqual(failure["minimum_annotation_contrast_ratio"], 4.5)
        self.assertGreaterEqual(
            failure["minimum_zero_annotation_contrast_ratio"], 4.5
        )
        self.assertGreaterEqual(
            failure["minimum_na_annotation_contrast_ratio"], 4.5
        )
        self.assertIn("never_zero", failure["na_policy"])
        self.assertEqual(len(failure["deterministic_condition_order"]), 10)
        self.assertEqual(len(failure["deterministic_failure_order"]), 6)


class Iteration2ReportingPublicationTests(unittest.TestCase):
    def test_production_and_excluded_reporting_identities_cannot_cross_publish(self):
        tables = {name: [] for name in WORKBOOK_SHEETS}
        contract = excluded_reporting_contract("reporting_cross_identity")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "publication identity"):
                build_reporting_bundle(
                    root / contract.analysis_identity,
                    tables=tables,
                    source_commit="1" * 40,
                    raw_location="excluded/raw",
                    raw_file_hashes={},
                    limitations=("Excluded fixture.",),
                    raw_generation_identity={
                        "run_identity": ITERATION2_RUN_IDENTITY,
                        "scientific_status": (
                            "iteration2_prespecified_scientific_run"
                        ),
                        "production_compatible": True,
                    },
                    analysis_validation_evidence=(
                        excluded_analysis_validation_evidence()
                    ),
                )
            with self.assertRaisesRegex(ValueError, "publication identity"):
                build_reporting_bundle(
                    root / ITERATION2_ANALYSIS_IDENTITY,
                    tables=tables,
                    source_commit="1" * 40,
                    raw_location="excluded/raw",
                    raw_file_hashes={},
                    limitations=("Excluded fixture.",),
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

    def test_bundle_is_atomic_complete_and_detects_corruption(self):
        tables = create_analysis_tables(
            excluded_feasibility_results(),
            analysis_validation_evidence=excluded_analysis_validation_evidence(),
            bootstrap_replicates=2,
            require_complete_design=False,
        )
        required_conditioning_fields = {
            "numerator",
            "denominator",
            "graph_count",
            "pair_count",
            "applicability",
            "status",
            "na_reason",
        }
        for table in (
            "Graph-Level Estimates",
            "Repair Denominators",
            "Failure Composition",
            "Distance Bands",
            "Method Stretch",
            "Recovered Stretch",
            "Routing Resources",
            "Operational Runtime",
        ):
            self.assertTrue(tables[table], table)
            self.assertTrue(
                all(
                    required_conditioning_fields <= set(row)
                    for row in tables[table]
                ),
                table,
            )
        with TemporaryDirectory() as temporary:
            contract = excluded_reporting_contract("reporting_atomic_complete")
            output = Path(temporary) / contract.analysis_identity
            manifest = build_reporting_bundle(
                output,
                tables=tables,
                source_commit="1" * 40,
                raw_location="excluded/raw",
                raw_file_hashes={"excluded.json.gz": "2" * 64},
                limitations=("Excluded deterministic fixture.",),
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
            self.assertEqual(manifest["workbook_sheets"], list(WORKBOOK_SHEETS))
            self.assertEqual(validate_reporting_bundle(output), manifest)
            self.assertEqual(set(manifest["figure_coverage"]), set(FIGURE_FILES))
            for table, filename in (
                ("Graph Metrics", "graph_metrics.csv"),
                ("Cell Estimates", "cell_estimates.csv"),
            ):
                with (output / filename).open(
                    "r", encoding="utf-8", newline=""
                ) as stream:
                    fieldnames = csv.DictReader(stream).fieldnames
                self.assertEqual(
                    fieldnames,
                    list(TABLE_SCHEMA_COLUMNS[table]),
                )
            csv_path = output / "graph_metrics.csv"
            csv_path.write_bytes(csv_path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_reporting_bundle(output)

    def test_failed_publication_leaves_no_output_or_staging_directory(self):
        tables = {name: [] for name in WORKBOOK_SHEETS}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = excluded_reporting_contract("reporting_failed_publication")
            output = root / contract.analysis_identity
            with patch(
                "iteration2_reporting.generate_required_figures",
                side_effect=RuntimeError("synthetic figure failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic figure failure"):
                    build_reporting_bundle(
                        output,
                        tables=tables,
                        source_commit="1" * 40,
                        raw_location="excluded/raw",
                        raw_file_hashes={},
                        limitations=("Excluded deterministic fixture.",),
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
            self.assertFalse(output.exists())
            self.assertFalse(
                any(path.name.startswith(f".{output.name}.tmp-") for path in root.iterdir())
            )

    def test_real_analysis_entry_calls_the_real_reporting_api(self):
        tables = {name: [] for name in WORKBOOK_SHEETS}
        provenance = {
            "raw_generation": {
                "source_commit": "1" * 40,
                "run_identity": ITERATION2_RUN_IDENTITY,
                "scientific_status": "iteration2_prespecified_scientific_run",
                "production_compatible": True,
            },
            "raw_tree_fingerprint": {"sha256": "2" * 64},
            "raw_manifest_sha256": "3" * 64,
            "raw_completion_sha256": "4" * 64,
            "analysis_validation": excluded_analysis_validation_evidence(),
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(analysis_entry, "verify_iteration1_immutable"),
                patch.object(
                    analysis_entry,
                    "_analysis_source_identity",
                    return_value={"source_commit": "1" * 40},
                ),
                patch.object(
                    analysis_entry,
                    "load_complete_raw_run",
                    return_value=([{"excluded": True}], {"raw": "5" * 64}, provenance),
                ),
                patch.object(
                    analysis_entry,
                    "create_analysis_tables",
                    return_value=tables,
                ),
            ):
                manifest = analysis_entry.analyze(root)
            output = root / "results" / ITERATION2_ANALYSIS_IDENTITY
            self.assertTrue(output.is_dir())
            self.assertEqual(validate_reporting_bundle(output), manifest)


if __name__ == "__main__":
    unittest.main()
