"""Machine-readable tables, figures, and a dependency-free Iteration 2 XLSX."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tempfile
from xml.sax.saxutils import escape
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from iteration2_config import (
    ANALYSIS_PLAN_HASH,
    COMBINED_PROTOCOL_HASH,
    DATA_GENERATION_HASH,
    EQUIVALENCE_MARGIN_PERCENTAGE_POINTS,
    ITERATION2_ANALYSIS_IDENTITY,
    ITERATION2_ANALYSIS_SCHEMA,
    ITERATION2_RUN_IDENTITY,
    OUTPUT_EXCEL_MAXIMUM_ROWS as EXCEL_MAXIMUM_ROWS,
    OUTPUT_FIGURE_FILES as FIGURE_FILES,
    OUTPUT_FIGURE_SPECIFICATIONS,
    OUTPUT_SCHEMA_HASH,
    OUTPUT_WORKBOOK_SHEETS as WORKBOOK_SHEETS,
    REPORTING_SEMANTIC_REGISTRY_SCHEMA,
    REPORTING_TABLE_SCHEMA_HASH,
    fingerprint,
)
from iteration2_excluded import (
    EXCLUDED_SCIENTIFIC_STATUS,
    excluded_analysis_identity,
    excluded_fixture_payload_hash,
    excluded_raw_identity,
    validate_excluded_fixture_payload,
)
from iteration2_runtime_guard import require_zero_scientific_operations


_GRAPH_IDENTITY_COLUMNS = (
    "graph_id",
    "model",
    "n",
    "m",
    "replicate_index",
)
_CONDITION_COLUMNS = (
    "coordinate_condition_id",
    "embedding_family",
    "condition_kind",
    "matched_radius",
)
_CONDITIONAL_COLUMNS = (
    "estimate",
    "numerator",
    "denominator",
    "graph_count",
    "pair_count",
    "status",
    "applicability",
    "na_reason",
    "unit",
)
_INTERVAL_COLUMNS = (
    "estimate",
    "ci_lower",
    "ci_upper",
    "numerator",
    "denominator",
    "graph_count",
    "pair_count",
    "status",
    "applicability",
    "na_reason",
    "unit",
)


def _columns(*groups: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for group in groups:
        for name in group:
            if name not in ordered:
                ordered.append(name)
    return tuple(ordered)


TABLE_SCHEMA_COLUMNS: dict[str, tuple[str, ...]] = {
    "README": (
        "analysis_schema",
        "protocol_hash",
        "data_generation_hash",
        "analysis_plan_hash",
        "output_schema_hash",
        "source_commit",
        "independent_experimental_unit",
        "primary_estimand",
        "primary_conditions",
        "native_references_are_secondary",
        "scaled_hydra_is_non_isometric_sensitivity",
        "raw_data_location",
        "raw_data_schema",
        "limitations",
    ),
    "Results Dashboard": (
        "model",
        "matched_radius",
        "primary_effect_size_percentage_points",
        "simultaneous_ci_lower_percentage_points",
        "simultaneous_ci_upper_percentage_points",
        "consistent_interaction_direction_across_four_radii",
        "common_success_stretch_difference_by_embedding",
        "repair_recovery_improvement_by_embedding",
        "mean_physical_hop_overhead_by_embedding",
        "synthetic_graph_scope_only",
        "deployment_recommendation_generated",
        "causal_interpretation",
        "hyperbolic_specific_repair_superiority_claim",
        "graph_count",
        "pair_count",
        "numerator",
        "denominator",
        "status",
        "applicability",
        "na_reason",
        "interpretation_warning",
    ),
    "Data Dictionary": (
        "schema",
        "table",
        "column",
        "definition",
        "unit",
        "conditioning",
        "denominator",
        "applicability",
        "missing_value_rule",
    ),
    "Graph Metrics": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        (
            "graph_model",
            "er_nominal_p",
            "ba_exact_expected_average_degree",
            "er_generation_attempts",
            "accepted_seed",
            "rejection_seeds",
            "connectivity_conditioning",
            "connected_er_observed_acceptance_rate",
            "realised_edge_count",
            "realised_average_degree",
            "target_average_degree",
            "average_degree_deviation_from_target",
            "number_of_vertices",
            "number_of_edges",
            "average_degree",
            "maximum_degree",
            "population_degree_variance",
            "average_clustering_coefficient",
            "diameter",
            "average_shortest_path_length",
            "pair_sampling",
            "er_ba_comparison_is_causal",
        ),
    ),
    "Graph-Level Estimates": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        _CONDITION_COLUMNS,
        (
            "independent_unit",
            "numerator",
            "denominator",
            "graph_count",
            "pair_count",
            "euclidean_success_count",
            "euclidean_success",
            "poincare_success_count",
            "poincare_success",
            "repaired_poincare_success_count",
            "repaired_poincare_success",
            "poincare_minus_euclidean",
            "repaired_minus_unrepaired_poincare",
            "common_success_pair_count",
            "common_success_poincare_minus_euclidean_stretch",
            "recovered_pair_count",
            "recovered_forwarding_stretch",
            "physical_recovered_stretch",
            "poincare_status",
            "poincare_applicability",
            "poincare_na_reason",
            "repaired_poincare_status",
            "repaired_poincare_applicability",
            "repaired_poincare_na_reason",
            "status",
            "applicability",
            "na_reason",
            "unit",
        ),
    ),
    "Cell Estimates": _columns(
        (
            "scope",
            "model",
            "n",
            "m",
            "coordinate_condition_id",
            "embedding_family",
            "condition_kind",
            "matched_radius",
            "metric",
        ),
        _INTERVAL_COLUMNS,
        (
            "total_graph_count",
            "contributing_graph_count",
            "applicable_graph_count",
            "na_graph_count",
            "stratum_count",
            "bootstrap_unit",
            "bootstrap_replicates",
            "defined_bootstrap_replicates",
            "minimum_defined_bootstrap_replicates",
            "weighting",
            "independent_unit",
        ),
    ),
    "Model Marginals": _columns(
        (
            "scope",
            "model",
            "n",
            "m",
            "coordinate_condition_id",
            "embedding_family",
            "condition_kind",
            "matched_radius",
            "metric",
        ),
        _INTERVAL_COLUMNS,
        (
            "total_graph_count",
            "contributing_graph_count",
            "applicable_graph_count",
            "na_graph_count",
            "applicable_stratum_count",
            "stratum_count",
            "bootstrap_unit",
            "bootstrap_replicates",
            "defined_bootstrap_replicates",
            "minimum_defined_bootstrap_replicates",
            "weighting",
            "independent_unit",
        ),
    ),
    "Matched Success Contrasts": _columns(
        (
            "scope",
            "model",
            "n",
            "m",
            "coordinate_condition_id",
            "embedding_family",
            "condition_kind",
            "matched_radius",
            "metric",
        ),
        _INTERVAL_COLUMNS,
        (
            "stratum_count",
            "applicable_stratum_count",
            "bootstrap_unit",
            "bootstrap_replicates",
            "weighting",
            "total_graph_count",
            "applicable_graph_count",
            "na_graph_count",
            "defined_bootstrap_replicates",
            "minimum_defined_bootstrap_replicates",
            "independent_unit",
        ),
    ),
    "Matched Embedding Interactions": _columns(
        ("scope", "model", "n", "m", "matched_radius"),
        _INTERVAL_COLUMNS,
        (
            "ci_type",
            "simultaneous_ci_lower",
            "simultaneous_ci_upper",
            "simultaneous_critical_value",
            "stratum_count",
            "bootstrap_replicates",
            "procedure",
            "interaction_definition",
            "weighting",
            "bootstrap_unit",
            "equivalence_margin_lower",
            "equivalence_margin_upper",
            "ci_excludes_zero",
            "ci_wholly_inside_margin",
            "ci_wholly_positive_beyond_margin",
            "ci_wholly_negative_beyond_margin",
            "practical_magnitude_unresolved",
            "equivalence_classification",
            "simultaneous_result_applicability",
            "all_radius_ci_wholly_inside_margin",
        ),
    ),
    "Complete-System Comparisons": _columns(
        ("scope", "model", "n", "m", "matched_radius"),
        _INTERVAL_COLUMNS,
        (
            "stratum_count",
            "bootstrap_replicates",
            "comparison_type",
            "interaction_definition",
            "metric_only_effect",
            "interpretation",
            "weighting",
            "bootstrap_unit",
        ),
    ),
    "Native References": _columns(
        ("scope", "model", "n", "m", "coordinate_condition_id", "comparison_type", "metric"),
        _INTERVAL_COLUMNS,
        (
            "stratum_count",
            "bootstrap_replicates",
            "metric_only_effect",
            "interpretation",
            "total_graph_count",
            "contributing_graph_count",
            "na_graph_count",
            "defined_bootstrap_replicates",
            "minimum_defined_bootstrap_replicates",
            "bootstrap_unit",
            "independent_unit",
        ),
    ),
    "Equivalence Sensitivity": _columns(
        ("scope", "model", "n", "m", "matched_radius"),
        _INTERVAL_COLUMNS,
        (
            "ci_type",
            "simultaneous_ci_lower",
            "simultaneous_ci_upper",
            "simultaneous_critical_value",
            "interaction_definition",
            "stratum_count",
            "weighting",
            "bootstrap_unit",
            "bootstrap_replicates",
            "procedure",
            "equivalence_margin_lower",
            "equivalence_margin_upper",
            "ci_excludes_zero",
            "ci_wholly_inside_margin",
            "ci_wholly_positive_beyond_margin",
            "ci_wholly_negative_beyond_margin",
            "practical_magnitude_unresolved",
            "equivalence_classification",
            "simultaneous_result_applicability",
            "all_radius_ci_wholly_inside_margin",
        ),
    ),
    "Repair Denominators": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        _CONDITION_COLUMNS,
        ("estimand",),
        _CONDITIONAL_COLUMNS,
        ("independent_unit",),
    ),
    "Failure Composition": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        _CONDITION_COLUMNS,
        (
            "method_id",
            "failure_stage",
            "failure_type",
            "category_index",
            "estimand",
        ),
        _CONDITIONAL_COLUMNS,
        ("invariant_error_category", "independent_unit"),
    ),
    "Distance Bands": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        _CONDITION_COLUMNS,
        ("method_id", "distance_band", "successful_stretch_mean"),
        _CONDITIONAL_COLUMNS,
        ("independent_unit",),
    ),
    "Method Stretch": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        _CONDITION_COLUMNS,
        ("method_id",),
        _CONDITIONAL_COLUMNS,
        ("conditioning", "independent_unit"),
    ),
    "Common-Success Stretch": _columns(
        (
            "scope",
            "model",
            "n",
            "m",
            "coordinate_condition_id",
            "embedding_family",
            "condition_kind",
            "matched_radius",
            "metric",
        ),
        _INTERVAL_COLUMNS,
        (
            "total_graph_count",
            "applicable_graph_count",
            "na_graph_count",
            "applicable_stratum_count",
            "stratum_count",
            "weighting",
            "bootstrap_unit",
            "bootstrap_replicates",
            "defined_bootstrap_replicates",
            "minimum_defined_bootstrap_replicates",
            "conditioning",
            "independent_unit",
        ),
    ),
    "Recovered Stretch": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        _CONDITION_COLUMNS,
        ("metric",),
        _CONDITIONAL_COLUMNS,
        ("conditioning", "independent_unit"),
    ),
    "Distortion Diagnostics": _columns(
        ("model", "n", "m", "coordinate_condition_id", "geometry", "metric"),
        _INTERVAL_COLUMNS,
        (
            "total_graph_count",
            "contributing_graph_count",
            "na_graph_count",
            "bootstrap_unit",
            "bootstrap_replicates",
            "defined_bootstrap_replicates",
            "minimum_defined_bootstrap_replicates",
            "independent_unit",
        ),
    ),
    "Property Associations": (
        "model",
        "outcome",
        "coordinate_condition_id",
        "property",
        "family_id",
        "multiplicity_family_fingerprint",
        "hypothesis_family_size",
        "association_estimate",
        "correlation",
        "ci_lower",
        "ci_upper",
        "rank_sensitivity_estimate",
        "rank_sensitivity_ci_lower",
        "rank_sensitivity_ci_upper",
        "rank_sensitivity_na_reason",
        "status",
        "applicability",
        "na_reason",
        "graph_count",
        "defined_graph_count",
        "pair_count",
        "numerator",
        "denominator",
        "unit",
        "independent_unit",
        "stratification",
        "bootstrap_replicates",
        "defined_bootstrap_replicates",
        "permutation_replicates",
        "permutation_mapping",
        "multiplicity_procedure",
        "raw_permutation_p_value",
        "familywise_adjusted_p_value",
        "familywise_alpha",
        "defined_family_hypothesis_count",
        "exploratory",
        "associative",
        "causal_interpretation",
    ),
    "Model Contrasts": _columns(
        ("scope", "metric", "contrast", "coordinate_condition_id", "n", "m"),
        _INTERVAL_COLUMNS,
        (
            "stratum_count",
            "er_graph_count",
            "ba_graph_count",
            "weighting",
            "resampling",
            "bootstrap_replicates",
            "causal_interpretation",
            "models_differ_in_multiple_properties",
        ),
    ),
    "Routing Resources": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        _CONDITION_COLUMNS,
        ("method_id", "resource", "maximum"),
        _CONDITIONAL_COLUMNS,
        ("independent_unit",),
    ),
    "Operational Runtime": _columns(
        _GRAPH_IDENTITY_COLUMNS,
        (
            "setup_seconds",
            "route_execution_seconds",
            "total_seconds",
            "timing_is_scientific_outcome",
            "runtime_used_to_change_scientific_output",
            "method_specific_runtime_claim",
            "graph_count",
            "pair_count",
            "numerator",
            "denominator",
            "status",
            "applicability",
            "na_reason",
            "unit",
        ),
    ),
    "Evidence Table": (
        "model",
        "matched_radius",
        "primary_effect_size_percentage_points",
        "simultaneous_ci_lower_percentage_points",
        "simultaneous_ci_upper_percentage_points",
        "consistent_interaction_direction_across_four_radii",
        "common_success_stretch_difference_by_embedding",
        "repair_recovery_improvement_by_embedding",
        "mean_physical_hop_overhead_by_embedding",
        "synthetic_graph_scope_only",
        "deployment_recommendation_generated",
        "causal_interpretation",
        "hyperbolic_specific_repair_superiority_claim",
        "graph_count",
        "pair_count",
        "numerator",
        "denominator",
        "status",
        "applicability",
        "na_reason",
    ),
    "Validation Summary": (
        "graph_count",
        "pair_count",
        "numerator",
        "denominator",
        "raw_graphs_validated",
        "oracle_disagreements",
        "analysis_validation_mode",
        "scientific_regeneration_requested_by_analysis",
        "scientific_graphs_executed_during_analysis",
        "dijkstra_executions_during_analysis",
        "routing_executions_during_analysis",
        "raw_checkpoints_written_during_analysis",
        "raw_tree_unchanged_during_analysis",
        "full_scientific_run_was_performed_by_analysis",
        "complete_frozen_design",
        "bootstrap_replicates",
        "status",
        "applicability",
        "na_reason",
    ),
    "Provenance": (
        "protocol_hash",
        "data_generation_hash",
        "analysis_plan_hash",
        "output_schema_hash",
        "source_commit",
        "raw_location",
        "raw_generation_identity",
        "analysis_source_identity",
        "raw_tree_fingerprint",
        "raw_manifest_sha256",
        "raw_completion_sha256",
        "raw_file_hashes",
        "analysis_validation_evidence",
        "formula_cells",
        "hidden_outcome_dependent_filters",
    ),
}

if set(TABLE_SCHEMA_COLUMNS) != set(WORKBOOK_SHEETS):
    raise RuntimeError("reporting table schemas must cover every workbook sheet")
if (
    fingerprint(
        {
            name: list(columns)
            for name, columns in TABLE_SCHEMA_COLUMNS.items()
        }
    )
    != REPORTING_TABLE_SCHEMA_HASH
):
    raise RuntimeError("reporting table schemas differ from the frozen output identity")
_FIGURE_SPEC_BY_FILE = {
    str(specification["file"]): specification
    for specification in OUTPUT_FIGURE_SPECIFICATIONS
}
if tuple(_FIGURE_SPEC_BY_FILE) != FIGURE_FILES:
    raise RuntimeError("reporting figure specifications differ from the output identity")


def _column_name(index: int) -> str:
    result = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _cell_xml(
    row_index: int,
    column_index: int,
    value: object,
    *,
    header: str,
    row: Mapping[str, object] | None,
) -> str:
    reference = f"{_column_name(column_index)}{row_index}"
    if value is None:
        return f'<c r="{reference}" t="inlineStr"><is><t>NA</t></is></c>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, int):
        return f'<c r="{reference}" s="2"><v>{value}</v></c>'
    if isinstance(value, float):
        field_unit = _field_unit(header)
        unit = (
            ""
            if row is None
            else str(row.get("unit", ""))
            if field_unit == "declared_by_row_unit"
            else field_unit
        )
        style = "3" if unit in ("proportion", "probability_0_to_1") else "1"
        return f'<c r="{reference}" s="{style}"><v>{value:.17g}</v></c>'
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    text = escape(str(value))
    preserve = ' xml:space="preserve"' if text.strip() != text else ""
    return (
        f'<c r="{reference}" t="inlineStr"><is><t{preserve}>'
        f"{text}</t></is></c>"
    )


def _worksheet_xml(
    rows: Sequence[Mapping[str, object]],
    headers: Sequence[str] | None = None,
) -> str:
    if len(rows) + 1 > EXCEL_MAXIMUM_ROWS:
        raise ValueError("worksheet exceeds the Excel row limit")
    ordered_headers = list(headers or ())
    if not ordered_headers:
        for row in rows:
            for key in row:
                if key not in ordered_headers:
                    ordered_headers.append(key)
    if not ordered_headers:
        ordered_headers = ["status"]
        rows = ({"status": "no_rows"},)
    sheet_rows: list[str] = []
    header_cells = "".join(
        _cell_xml(1, index, header, header=header, row=None)
        for index, header in enumerate(ordered_headers, 1)
    )
    sheet_rows.append(f'<row r="1" s="4">{header_cells}</row>')
    for row_index, row in enumerate(rows, 2):
        cells = "".join(
            _cell_xml(
                row_index,
                column_index,
                row.get(header),
                header=header,
                row=row,
            )
            for column_index, header in enumerate(ordered_headers, 1)
        )
        sheet_rows.append(f'<row r="{row_index}">{cells}</row>')
    last_column = _column_name(len(ordered_headers))
    last_row = len(rows) + 1
    columns = "".join(
        f'<col min="{index}" max="{index}" width="18" customWidth="1"/>'
        for index in range(1, len(ordered_headers) + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane xSplit="1" ySplit="1" topLeftCell="B2" '
        'activePane="bottomRight" state="frozen"/>'
        '</sheetView></sheetViews>'
        f"<cols>{columns}</cols><sheetData>{''.join(sheet_rows)}</sheetData>"
        f'<autoFilter ref="A1:{last_column}{last_row}"/>'
        f'<dimension ref="A1:{last_column}{last_row}"/>'
        "</worksheet>"
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main">'
        '<numFmts count="2">'
        '<numFmt numFmtId="164" formatCode="0.0000"/>'
        '<numFmt numFmtId="165" formatCode="0.00%"/>'
        "</numFmts>"
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="5">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>'
        '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="1" fontId="0" fillId="0" borderId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="165" fontId="0" fillId="0" borderId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0"/>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" '
        'builtinId="0"/></cellStyles></styleSheet>'
    )


def write_xlsx(
    path: Path | str,
    sheets: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Write a formula-free workbook with stable headers and frozen panes."""

    if tuple(sheets) != WORKBOOK_SHEETS:
        raise ValueError("workbook sheets must match the frozen Iteration 2 order")
    target = Path(path)
    if target.exists():
        raise FileExistsError("Iteration 2 workbook target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" '
        f'r:id="rId{index}"/>'
        for index, name in enumerate(WORKBOOK_SHEETS, 1)
    )
    relationships = "".join(
        '<Relationship '
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/'
        f'2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(WORKBOOK_SHEETS) + 1)
    )
    relationships += (
        '<Relationship Id="rIdStyles" '
        'Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/styles" Target="styles.xml"/>'
    )
    overrides = "".join(
        '<Override '
        f'PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(WORKBOOK_SHEETS) + 1)
    )
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        def write_member(name: str, payload: str) -> None:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.encode("utf-8"))

        write_member(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxml'
            'formats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.'
            'openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{overrides}</Types>",
        )
        write_member(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        write_member(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.'
            f'org/officeDocument/2006/relationships"><sheets>{workbook_sheets}'
            "</sheets></workbook>",
        )
        write_member(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            f'package/2006/relationships">{relationships}</Relationships>',
        )
        write_member("xl/styles.xml", _styles_xml())
        for index, name in enumerate(WORKBOOK_SHEETS, 1):
            write_member(
                f"xl/worksheets/sheet{index}.xml",
                _worksheet_xml(sheets[name], TABLE_SCHEMA_COLUMNS[name]),
            )


_FIELD_DEFINITIONS = {
    "analysis_schema": "Versioned schema identifier for the analysed Iteration 2 artifacts.",
    "schema": "Versioned machine-readable schema identifier for this table.",
    "table": "Workbook sheet and companion CSV to which the dictionary row applies.",
    "column": "Exact machine-readable column name defined by the dictionary row.",
    "definition": "Scientific or operational meaning of the named output column.",
    "conditioning": "Population or event on which the output value is conditioned.",
    "missing_value_rule": "Rule that determines when the output is represented as NA.",
    "graph_id": "Versioned deterministic graph identity.",
    "model": "Synthetic graph model; Erdos-Renyi or Barabasi-Albert.",
    "n": "Number of vertices in the graph design stratum.",
    "m": "Frozen graph density parameter in the design stratum.",
    "replicate_index": "Zero-based independent graph replicate within model-n-m.",
    "coordinate_condition_id": "Exact native or matched-radius coordinate condition identity.",
    "method_id": "Exact routing method identity.",
    "matched_radius": "Prespecified maximum disk radius for a matched condition.",
    "embedding_family": "Coordinate-construction family: Hydra or classical MDS.",
    "condition_kind": "Native-reference or matched-radius-sensitivity condition class.",
    "scope": "Aggregation scope: design cell, equal-stratum marginal, or reference comparison.",
    "geometry": "Coordinate-distance geometry used for the diagnostic.",
    "metric": "Name of the estimand or diagnostic reported in this row.",
    "estimand": "Exact prespecified repair or routing estimand represented by the row.",
    "resource": "Routing-cost component summarized by the row.",
    "estimate": "Point estimate on the scale declared by unit.",
    "association_estimate": "Exploratory within-model and within-n-m residual Pearson correlation between the named graph property and outcome; it is descriptive and non-causal.",
    "ci_lower": "Lower endpoint of the declared 95 percent interval.",
    "ci_upper": "Upper endpoint of the declared 95 percent interval.",
    "ci_type": "Pointwise or simultaneous confidence-interval construction.",
    "simultaneous_ci_lower": "Lower endpoint of the familywise 95% band covering the four prespecified matched radii jointly within one model.",
    "simultaneous_ci_upper": "Upper endpoint of the familywise 95% band covering the four prespecified matched radii jointly within one model.",
    "simultaneous_critical_value": (
        "The non-interpolated bootstrap critical value obtained from the "
        "distribution, across bootstrap replicates, of the maximum over the "
        "four matched radii of the absolute unstandardized deviation between "
        "the bootstrap interaction estimate and the original graph-level "
        "interaction estimate. It is expressed in percentage points and is "
        "used to form the simultaneous band by adding and subtracting the "
        "critical value from each radius-specific interaction estimate."
    ),
    "equivalence_margin_lower": "Lower closed sensitivity margin in percentage points.",
    "equivalence_margin_upper": "Upper closed sensitivity margin in percentage points.",
    "ci_excludes_zero": "True when both interval endpoints are strictly positive or strictly negative; false when zero is contained or touched.",
    "ci_wholly_inside_margin": "True when both interval endpoints lie inside the closed sensitivity margin; false otherwise. This is a compatibility diagnostic, not proof of embedding neutrality.",
    "ci_wholly_positive_beyond_margin": "True when the entire interval is strictly above the positive margin; false otherwise.",
    "ci_wholly_negative_beyond_margin": "True when the entire interval is strictly below the negative margin; false otherwise.",
    "practical_magnitude_unresolved": "True when the interval is neither wholly inside nor wholly beyond either margin; false when one classification is resolved.",
    "equivalence_classification": "Compatibility label derived from the separate interval flags; it must not be interpreted as proof that embeddings are neutral or equivalent.",
    "all_radius_ci_wholly_inside_margin": "True when every one of the four simultaneous-radius intervals is inside the closed margin; false otherwise, without implying embedding neutrality in either case.",
    "numerator": "Exact numerator for the conditional proportion or ratio when defined.",
    "denominator": "Exact denominator for the conditional proportion or ratio when defined.",
    "graph_count": "Number of independent graphs represented by the row.",
    "pair_count": "Number of sampled ordered pairs represented by the row.",
    "status": "Defined, undefined, incomplete, or not-applicable state.",
    "applicability": "Whether the estimand is defined by design for this row.",
    "na_reason": "Machine-readable reason for an NA value; null when defined.",
    "unit": "Measurement scale for estimate and interval fields.",
    "independent_unit": "Independent resampling and inferential unit; graph for scientific estimates.",
    "independent_experimental_unit": "Independent experimental unit; graph for Iteration 2.",
    "property": "One of the six prespecified network-property predictors.",
    "outcome": "Prespecified graph-level success or common-success stretch outcome.",
    "raw_permutation_p_value": "Exploratory two-sided within-stratum shared-map permutation p-value; it does not support a causal interpretation.",
    "familywise_adjusted_p_value": "Exploratory max-absolute-statistic familywise adjusted p-value across the prespecified association family; it is non-causal.",
    "family_id": "One of four prespecified model-by-outcome multiplicity families.",
    "multiplicity_family_fingerprint": "SHA-256 identity of the exact 54-hypothesis family and resampling plan.",
    "hypothesis_family_size": "Number of hypotheses in the prespecified multiplicity family; exactly 54.",
    "defined_family_hypothesis_count": "Number of family hypotheses having a defined observed statistic.",
    "failure_stage": "Initial ordinary or final method-specific failure stage.",
    "failure_type": "Non-invariant scientific terminal category at the stated stage.",
    "distance_band": "Dijkstra hop band: 1, 2, 3, 4, or at least 5.",
    "physical_hops": "Count of every physically traversed edge, including repair backtrack and escape.",
    "forwarding_decisions": "Count of deliberate next-edge choices, including repair choices.",
    "logical_distance_evaluations": "Metric evaluations logically required regardless of cache hits.",
    "peak_history_vertices": "Maximum distinct vertex IDs retained simultaneously in route history.",
    "distance_band": "Dijkstra hop band: 1, 2, 3, 4, or at least 5.",
    "successful_stretch_mean": "Mean physical-hop stretch among successful routes in the stated distance band.",
    "euclidean_mean": "Mean Euclidean greedy stretch on the common-success pair set.",
    "poincare_mean": "Mean Poincare greedy stretch on the identical common-success pair set.",
    "maximum": "Maximum observed routing-resource value within the graph-method row when defined.",
    "source_commit": "Committed source revision used by the future scientific run or analysis.",
    "protocol_hash": "Combined identity referencing data, analysis, and output identities.",
    "data_generation_hash": "Frozen data-generation identity fingerprint.",
    "analysis_plan_hash": "Frozen statistical-analysis identity fingerprint.",
    "output_schema_hash": "Frozen reporting/output identity fingerprint.",
    "raw_location": "Repository-relative location of the validated raw Iteration 2 run.",
    "raw_data_location": "Repository-relative location of the validated raw Iteration 2 run.",
    "raw_data_schema": "Versioned raw-run identity expected by the analysis.",
    "raw_manifest_sha256": "SHA-256 digest of the validated raw run manifest.",
    "raw_completion_sha256": "SHA-256 digest of the validated raw completion marker.",
    "raw_file_hashes": "Complete relative-path to SHA-256 mapping for raw input files.",
    "raw_tree_fingerprint": "Canonical byte-level fingerprint summary for the complete raw tree.",
    "raw_generation_identity": "Validated source and protocol identity recorded by raw generation.",
    "analysis_source_identity": "Clean committed source and dependency identity used for analysis.",
    "primary_estimand": "Primary matched-radius embedding interaction within each graph model.",
    "primary_conditions": "Hydra and MDS compared only at the same prespecified maximum radius.",
    "limitations": "Explicit interpretation and generalizability limitations for the bundle.",
    "number_of_vertices": "Realised number of vertices in the validated simple graph.",
    "number_of_edges": "Realised number of undirected edges in the validated simple graph.",
    "realised_edge_count": "Realised number of undirected edges recorded with generation provenance.",
    "average_degree": "Realised arithmetic mean vertex degree.",
    "maximum_degree": "Largest realised vertex degree.",
    "population_degree_variance": "Population variance of vertex degrees using ddof=0.",
    "average_clustering_coefficient": "Mean NetworkX local clustering coefficient.",
    "diameter": "Maximum exact unweighted shortest-path distance.",
    "average_shortest_path_length": "Mean exact unweighted distance over distinct vertex pairs.",
    "pair_sampling": "Lossless pair-sampling provenance and distance-distribution summary.",
    "er_nominal_p": "Frozen Erdos-Renyi edge probability 2m(n-m)/[n(n-1)].",
    "ba_exact_expected_average_degree": "Exact finite-size BA expected average degree 2m(n-m)/n.",
    "er_generation_attempts": "Number of deterministic ER connectivity attempts through acceptance.",
    "accepted_seed": "Final accepted graph-generation seed.",
    "rejection_seeds": "Ordered ER generation seeds rejected for disconnected graphs.",
    "connectivity_conditioning": "Whether connectivity was conditioned by rejection or guaranteed by construction.",
    "connected_er_observed_acceptance_rate": "Reciprocal of ER attempts through the accepted connected graph.",
    "realised_average_degree": "Realised average degree recorded with graph-generation provenance.",
    "target_average_degree": "Frozen finite-size target average degree for the graph stratum.",
    "average_degree_deviation_from_target": "Realised minus target average degree.",
    "euclidean_success": "Graph-level proportion of pairs delivered by Euclidean greedy routing.",
    "poincare_success": "Graph-level proportion of pairs delivered by Poincare greedy routing.",
    "repaired_poincare_success": "Graph-level proportion delivered after at most one Poincare repair attempt.",
    "poincare_minus_euclidean": "Within-graph Poincare minus Euclidean success proportion.",
    "repaired_minus_unrepaired_poincare": "Within-graph repaired minus ordinary Poincare success proportion.",
    "common_success_poincare_minus_euclidean_stretch": "Paired Poincare minus Euclidean hop stretch among pairs delivered by both ordinary methods.",
    "recovered_forwarding_stretch": "Forwarding stretch for newly recovered routes with the one physical backtrack excluded.",
    "physical_recovered_stretch": "Physical stretch for newly recovered routes including backtrack and escape edges.",
    "comparison_type": "Prespecified within-condition, matched-system, or secondary native comparison class.",
    "interaction_definition": "Algebraic definition of the reported matched interaction.",
    "metric_only_effect": "True only when coordinate construction is held fixed and solely the distance metric changes; false when the comparison changes the complete coordinate-and-metric system.",
    "interpretation": "Machine-readable interpretation boundary for the comparison.",
    "bootstrap_unit": "Unit resampled by the bootstrap; whole graphs within model-n-m strata.",
    "bootstrap_replicates": "Number of deterministic whole-graph bootstrap maps used.",
    "permutation_replicates": "Number of deterministic shared within-stratum permutation maps used.",
    "permutation_mapping": "Rule identifying the shared permutation map across family hypotheses.",
    "multiplicity_procedure": "Maximum-absolute-statistic familywise multiplicity procedure.",
    "stratum_count": "Number of equally weighted n-m strata contributing to the row.",
    "weighting": "Rule used to combine graphs and design strata.",
    "runtime_used_to_change_scientific_output": "True would mean timing selected or altered a scientific result; false confirms runtime is operational only and never changes scientific output.",
    "method_specific_runtime_claim": "True would assert separately supported method-specific timing; false means only total operational runtime is reported and no method-speed claim is made.",
    "setup_seconds": "Operational elapsed seconds for graph setup, APSP, and embeddings.",
    "route_execution_seconds": "Operational elapsed seconds for all routing work on the graph.",
    "total_seconds": "Operational total graph execution seconds.",
    "applicable_graph_count": "Number of independent graphs for which the row's estimand is defined by the protocol, including graphs with a defined zero numerator.",
    "applicable_stratum_count": "Number of prespecified n-m strata for which the estimand is applicable and has sufficient graph-level support.",
    "associative": (
        "Boolean interpretation flag. True identifies the reported "
        "relationship as a statistical association only and explicitly "
        "prohibits causal interpretation; False would mean this "
        "associative-only interpretation flag is not asserted. This field "
        "does not describe the mathematical associative law."
    ),
    "ba_graph_count": "Number of independent Barabasi-Albert graphs contributing to the descriptive model contrast.",
    "category_index": "Zero-based deterministic display order of the named failure category; it is not a measured quantity.",
    "causal_interpretation": "True would claim that the reported contrast identifies a causal effect; false explicitly restricts the evidence to descriptive synthetic comparisons.",
    "common_success_pair_count": "Number of sampled ordered pairs delivered by both ordinary Euclidean and Poincare methods; this is the denominator population for common-success stretch.",
    "common_success_stretch_difference_by_embedding": "Mapping from embedding family to the Poincare-minus-Euclidean hop-stretch difference among the identical pairs delivered by both ordinary methods.",
    "complete_frozen_design": "True when all prespecified model-n-m-replicate-condition cells are present and validated; false when any required design element is absent.",
    "consistent_interaction_direction_across_four_radii": "True when all four matched-radius interaction point estimates have the same non-zero sign; false when signs differ or include zero.",
    "contrast": "Exact ordered estimand contrast, with the minuend and subtrahend identified by the row's comparison metadata.",
    "contributing_graph_count": "Number of independent graphs with a finite value that actually contributes to the point estimate.",
    "correlation": "Exploratory correlation on the natural -1 to 1 scale; it describes association and does not identify causation.",
    "defined_bootstrap_replicates": "Number of requested bootstrap maps producing a finite statistic for this estimand after applying the frozen missingness rules.",
    "defined_graph_count": "Number of independent graphs with both predictor and outcome defined for the exploratory association.",
    "deployment_recommendation_generated": "True would indicate that a deployment recommendation was produced; false confirms that synthetic Iteration 2 evidence is not converted into deployment advice.",
    "er_ba_comparison_is_causal": "True would assert a causal ER-versus-BA model effect; false correctly states that the models differ in multiple structural properties and the comparison is descriptive.",
    "er_graph_count": "Number of independent Erdos-Renyi graphs contributing to the descriptive model contrast.",
    "euclidean_success_count": "Exact number of sampled ordered pairs delivered successfully by Euclidean greedy routing in the stated graph and coordinate condition.",
    "exploratory": "True labels the analysis as exploratory and non-causal; false would identify a prespecified confirmatory analysis instead.",
    "familywise_alpha": "Prespecified familywise Type-I error probability used by the exploratory multiplicity procedure.",
    "formula_cells": "Number of workbook cells containing executable formulas; acceptance requires zero so published values are static.",
    "full_scientific_run_was_performed_by_analysis": "True would mean analysis executed the scientific experiment; false confirms analysis only consumed previously validated raw results.",
    "analysis_validation_mode": "Named validation policy used while loading raw results; ordinary publication requires read_only_analysis_consumer.",
    "scientific_regeneration_requested_by_analysis": "True would mean analysis requested scientific regeneration; publication requires false.",
    "scientific_graphs_executed_during_analysis": "Count of complete graph experiments executed after analysis began; publication requires zero.",
    "dijkstra_executions_during_analysis": "Count of shortest-path computations executed after analysis began; publication requires zero.",
    "routing_executions_during_analysis": "Count of routing attempts executed after analysis began; publication requires zero.",
    "raw_checkpoints_written_during_analysis": "Count of raw graph checkpoints written or replaced after analysis began; publication requires zero.",
    "raw_tree_unchanged_during_analysis": "True only when complete byte-level raw-tree fingerprints before and after validation are identical; publication requires true.",
    "analysis_validation_evidence": "Machine-readable read-only validation policy, zero-execution counters, zero-write counter, and complete raw-tree fingerprints observed before and after analysis loading.",
    "graph_model": "Generation mechanism used for the graph: connected-conditioned Erdos-Renyi or Barabasi-Albert preferential attachment.",
    "hidden_outcome_dependent_filters": "True would indicate hidden filtering based on observed outcomes; false confirms all published filters are explicit and outcome-independent.",
    "hyperbolic_specific_repair_superiority_claim": "True would claim repair superiority is specific to hyperbolic geometry; false acknowledges that the design does not identify such a geometry-specific causal claim.",
    "interpretation_warning": "Explicit restriction describing conclusions that the row does not support, including causal, deployment, or embedding-neutrality claims where relevant.",
    "invariant_error_category": "True identifies an implementation-invariant violation that invalidates the run rather than a scientific routing failure; false identifies an admissible scientific failure category.",
    "mean_physical_hop_overhead_by_embedding": "Mapping from embedding family to mean physical-hop overhead for newly recovered routes, including repair backtrack and escape traversal.",
    "minimum_defined_bootstrap_replicates": "Minimum finite bootstrap replicate count required by the frozen completeness threshold before an interval may be reported.",
    "models_differ_in_multiple_properties": "True states that ER and BA differ jointly in several graph properties so their contrast is descriptive; false would assert a single isolated model difference.",
    "na_graph_count": "Number of independent graphs whose estimand value is NA under the declared applicability or zero-denominator rule.",
    "native_references_are_secondary": "True means native-coordinate comparisons are labelled secondary references and excluded from the primary matched-radius estimand; false would incorrectly promote them to the primary comparison.",
    "oracle_disagreements": "Count of validation cases where the production result disagreed with the independent oracle; acceptance requires exactly zero.",
    "poincare_applicability": "Whether ordinary Poincare routing is defined for the coordinate condition; native MDS is not applicable by design.",
    "poincare_na_reason": "Machine-readable reason ordinary Poincare output is NA; null only when the method is applicable and its value is defined.",
    "poincare_status": "Validation state of ordinary Poincare output, distinguishing defined values from protocol-level non-applicability or undefined denominators.",
    "poincare_success_count": "Exact number of sampled ordered pairs delivered successfully by ordinary Poincare greedy routing in the stated graph and condition.",
    "primary_effect_size_percentage_points": "Primary matched-radius Hydra-versus-MDS embedding interaction expressed in percentage points within the named graph model.",
    "procedure": "Versioned resampling and interval-construction procedure used for the row, including the joint four-radius family when simultaneous.",
    "rank_sensitivity_ci_lower": "Lower endpoint of the exploratory non-causal interval for the within-stratum residual midrank Spearman sensitivity estimate.",
    "rank_sensitivity_ci_upper": "Upper endpoint of the exploratory non-causal interval for the within-stratum residual midrank Spearman sensitivity estimate.",
    "rank_sensitivity_estimate": "Exploratory non-causal within-stratum residual midrank Spearman association used as a rank-based sensitivity analysis.",
    "rank_sensitivity_na_reason": "Machine-readable reason the exploratory rank sensitivity estimate or interval is NA; null when fully defined.",
    "raw_graphs_validated": "Number of raw graph checkpoints that passed schema, identity, count, hash, and scientific-content validation.",
    "recovered_pair_count": "Exact number of pairs that failed ordinary Poincare routing but were delivered by the one-backtrack repair; denominator is the relevant ordinary-failure population.",
    "repair_recovery_improvement_by_embedding": "Mapping from embedding family to recovered-pair numerator divided by ordinary Poincare failures eligible under the named repair estimand.",
    "repaired_poincare_applicability": "Whether repaired Poincare routing is defined for the coordinate condition; native MDS is not applicable by design.",
    "repaired_poincare_na_reason": "Machine-readable reason repaired Poincare output is NA; null only when the method is applicable and its value is defined.",
    "repaired_poincare_status": "Validation state of repaired Poincare output, distinguishing defined values from protocol-level non-applicability or undefined denominators.",
    "repaired_poincare_success_count": "Exact number of sampled ordered pairs delivered after at most one Poincare repair attempt in the stated graph and condition.",
    "resampling": "Frozen whole-graph bootstrap or shared-map permutation design used to quantify uncertainty without treating route rows as independent.",
    "scaled_hydra_is_non_isometric_sensitivity": "True means radially scaled Hydra coordinates are treated as a non-isometric sensitivity condition; false would incorrectly treat scaling as geometry preserving.",
    "simultaneous_ci_lower_percentage_points": "Lower endpoint, in percentage points, of the familywise 95% band jointly covering the four matched-radius interactions.",
    "simultaneous_ci_upper_percentage_points": "Upper endpoint, in percentage points, of the familywise 95% band jointly covering the four matched-radius interactions.",
    "simultaneous_result_applicability": "Whether all four prespecified radius statistics required for the simultaneous family are defined; inapplicable rows must report NA with a reason.",
    "stratification": "Frozen model-by-n-by-m strata within which graph-level residualization, resampling, or equal-stratum weighting is performed.",
    "synthetic_graph_scope_only": "True restricts interpretation to the synthetic ER and BA design; false would incorrectly imply broader empirical-network coverage.",
    "timing_is_scientific_outcome": "True would classify elapsed time as a scientific endpoint; false confirms it is operational provenance only.",
    "total_graph_count": "Total number of independent graphs in the prespecified scope before applicability and finite-value filtering.",
}


_TABLE_CONDITIONING = {
    "README": "Unconditional bundle-level protocol, provenance, and interpretation declarations.",
    "Results Dashboard": "The model and matched radius named by each dashboard row; detailed scientific denominators remain explicit in the source tables.",
    "Data Dictionary": "The exact workbook table and column identified by the dictionary row.",
    "Graph Metrics": "One independently generated graph after the frozen connectivity rule and its presampled ordered-pair schedule.",
    "Graph-Level Estimates": "One independent graph and one exact native or matched coordinate condition.",
    "Cell Estimates": "Independent graphs within one model-n-m cell and coordinate condition.",
    "Model Marginals": "Nine equally weighted n-m strata within one graph model and coordinate condition.",
    "Matched Success Contrasts": "Paired method outcomes for the same sampled pairs, graph, and matched coordinate condition.",
    "Matched Embedding Interactions": "Paired Hydra and MDS estimates at the same radius, evaluated jointly across all four radii within a model when simultaneous.",
    "Complete-System Comparisons": "Paired complete coordinate-and-metric systems at the same matched radius.",
    "Native References": "Secondary native-coordinate comparisons; native MDS supports Euclidean routing only.",
    "Equivalence Sensitivity": "The four-radius simultaneous interaction family within one model; classifications are sensitivity diagnostics, not neutrality claims.",
    "Repair Denominators": "The precise repair-eligibility event named by estimand for one graph and coordinate condition.",
    "Method Stretch": "Successful routes for the named method only.",
    "Common-Success Stretch": "Identical pairs delivered by both ordinary methods.",
    "Recovered Stretch": "Pairs recovered after ordinary Poincare failure.",
    "Failure Composition": "Failures at the exact initial or final stage for one method, condition, and graph; stages and categories are never pooled.",
    "Distance Bands": "All sampled pairs in the stated Dijkstra-distance band.",
    "Distortion Diagnostics": "One geometry-specific diagnostic within the named graph and coordinate condition; cross-geometry objectives are not pooled.",
    "Property Associations": "Within-model, within-n-m residual association across whole graphs.",
    "Model Contrasts": "Independent whole graphs compared descriptively between BA and ER within matched n-m strata.",
    "Routing Resources": "All sampled pairs for one method, graph, and coordinate condition, with method applicability enforced.",
    "Operational Runtime": "Operational graph execution only; no method-speed inference.",
    "Evidence Table": "The exact evidence item and model/condition scope identified by the row.",
    "Validation Summary": "The complete candidate raw run and analysis bundle subjected to the named validation check.",
    "Provenance": "The complete raw-input, source, protocol, dependency, and published-artifact identity set.",
}


_TABLE_DENOMINATORS = {
    "README": "Not a scientific ratio; identity and interpretation declarations have no numerator or denominator.",
    "Results Dashboard": "Exact graph_count, pair_count, numerator, and denominator fields from the corresponding source estimand.",
    "Data Dictionary": "Not a scientific ratio; denominator text documents the named output field.",
    "Graph Metrics": "One independent graph; pair_sampling contains exact pair counts.",
    "Graph-Level Estimates": "Exact pair_count and method-specific success counts.",
    "Cell Estimates": "Contributing whole graphs within one model-n-m cell.",
    "Model Marginals": "Nine equally weighted n-m strata, then equal graphs within stratum.",
    "Matched Success Contrasts": "Paired sampled pairs within graphs; graphs drive uncertainty.",
    "Matched Embedding Interactions": "Paired graph identities across both embeddings and four radii.",
    "Complete-System Comparisons": "Paired sampled pairs and graph identities at the same radius.",
    "Native References": "Paired graph identities; native conditions are secondary.",
    "Equivalence Sensitivity": "Same graph count as the associated interaction interval.",
    "Repair Denominators": "Exact numerator and the estimand-specific denominator columns.",
    "Failure Composition": "All sampled pairs for rates; final failures only for conditional composition.",
    "Distance Bands": "Exact sampled-pair count in the named distance band.",
    "Method Stretch": "Successful routes for the named method.",
    "Common-Success Stretch": "Pairs delivered by both ordinary methods.",
    "Recovered Stretch": "Pairs newly recovered by one-backtrack repair.",
    "Distortion Diagnostics": "Contributing graphs; pairwise coordinate diagnostics are computed within graph.",
    "Property Associations": "Defined whole graphs across all nine n-m strata.",
    "Model Contrasts": "Equal n-m strata with independent whole-graph samples by model.",
    "Routing Resources": "All sampled pairs for the named method and condition.",
    "Operational Runtime": "One graph per row; elapsed seconds are not divided by routes to make method claims.",
    "Evidence Table": "Graph and pair counts are exposed; no substituted denominator.",
    "Validation Summary": "The complete expected inventory or oracle-check population named by each validation row; acceptance counts must match the frozen requirement.",
    "Provenance": "Not a scientific ratio; hashes and file maps cover the complete declared artifact set.",
}


def _field_unit(header: str) -> str:
    if header in {
        "n",
        "m",
        "replicate_index",
        "denominator",
        "graph_count",
        "pair_count",
    } or header.endswith("_count"):
        return "count"
    if header == "numerator":
        return "declared_by_estimand"
    if header.endswith("_seconds"):
        return "seconds"
    if header.endswith("_sha256") or header.endswith("_hash") or "fingerprint" in header:
        return "cryptographic_identifier"
    if header.endswith("_p_value") or header == "familywise_alpha":
        return "probability_0_to_1"
    if "percentage_points" in header or header.startswith("equivalence_margin"):
        return "percentage_points"
    if "correlation" in header or header in {
        "association_estimate",
        "rank_sensitivity_estimate",
        "rank_sensitivity_ci_lower",
        "rank_sensitivity_ci_upper",
    }:
        return "correlation_-1_to_1"
    if header == "common_success_stretch_difference_by_embedding":
        return "mapping_embedding_family_to_hop_stretch_ratio_difference"
    if "stretch" in header:
        return "hop_stretch_ratio"
    if header == "matched_radius":
        return "disk_radius"
    if header in {
        "er_nominal_p",
        "connected_er_observed_acceptance_rate",
        "euclidean_success",
        "poincare_success",
        "repaired_poincare_success",
        "poincare_minus_euclidean",
        "repaired_minus_unrepaired_poincare",
    } or header.endswith("_rate"):
        return "proportion"
    if header == "repair_recovery_improvement_by_embedding":
        return "mapping_embedding_family_to_proportion"
    if header == "mean_physical_hop_overhead_by_embedding":
        return "mapping_embedding_family_to_hop_count"
    if header in {
        "average_degree",
        "maximum_degree",
        "population_degree_variance",
        "average_clustering_coefficient",
        "diameter",
        "average_shortest_path_length",
    }:
        return "graph_metric_declared_by_column"
    if header in {
        "estimate",
        "ci_lower",
        "ci_upper",
        "simultaneous_ci_lower",
        "simultaneous_ci_upper",
        "simultaneous_critical_value",
        "maximum",
    }:
        return "declared_by_row_unit"
    if header == "unit":
        return "unit_identifier"
    return "text_or_source_schema_unit"


_TABLE_FIELD_DEFINITION_OVERRIDES = {
    ("Data Dictionary", "definition"): "Complete substantive meaning of the named table-column pair, including interpretation restrictions rather than a restatement of its name.",
    ("Data Dictionary", "unit"): "Measurement unit or scale assigned to the named table-column pair by the semantic registry.",
    ("Data Dictionary", "conditioning"): "Population, event, design stratum, or artifact scope on which the named table-column value is conditioned.",
    ("Data Dictionary", "denominator"): "Exact denominator population for the named table-column value, or an explicit statement that the field is not a ratio.",
    ("Data Dictionary", "applicability"): "Protocol rule specifying when the named table-column value is meaningful and when it is not applicable.",
    ("Repair Denominators", "estimate"): "Repair recovery numerator divided only by the precise eligibility denominator named by estimand; denominator populations are never substituted.",
    ("Repair Denominators", "numerator"): "Exact number of pairs satisfying the recovery or repair-stage event named by estimand.",
    ("Repair Denominators", "denominator"): "Exact ordinary-failure, backtrackable-failure, repair-eligible, or selected-alternative population named by estimand.",
    ("Failure Composition", "estimate"): "Exact failure-category numerator divided by failures at the stated stage for conditional composition, or by all sampled pairs for the all-pairs rate.",
    ("Failure Composition", "numerator"): "Exact count of failures having the stated method, stage, and scientific failure category.",
    ("Failure Composition", "denominator"): "Exact number of failures at the stated stage for conditional composition, or all sampled pairs for the all-pairs failure rate.",
    ("Method Stretch", "estimate"): "Mean physical-hop count divided by Dijkstra hops among routes successfully delivered by the named method only.",
    ("Method Stretch", "denominator"): "Exact number of successful routes for the named method; unsuccessful and inapplicable routes do not enter stretch.",
    ("Common-Success Stretch", "estimate"): "Paired Poincare-minus-Euclidean hop-stretch difference among the identical sampled pairs successfully delivered by both ordinary methods.",
    ("Common-Success Stretch", "denominator"): "Exact common-success pair population delivered by both ordinary Euclidean and Poincare routing.",
    ("Recovered Stretch", "estimate"): "Mean Dijkstra-normalized forwarding or physical hop stretch among pairs newly delivered by the repair after ordinary Poincare failure.",
    ("Recovered Stretch", "denominator"): "Exact newly recovered pair count; routes not recovered by repair are excluded from this conditional stretch.",
    ("Property Associations", "estimate"): "Exploratory, non-causal within-model and within-n-m residual association between the named graph property and routing outcome.",
    ("Property Associations", "ci_lower"): "Lower endpoint of the exploratory graph-resampling interval for the non-causal association estimate.",
    ("Property Associations", "ci_upper"): "Upper endpoint of the exploratory graph-resampling interval for the non-causal association estimate.",
    ("Validation Summary", "status"): "Acceptance state of the named completeness, integrity, oracle, or schema validation; accepted means every frozen requirement for that check passed.",
}

_TABLE_FIELD_SEMANTIC_OVERRIDES = {
    ("Property Associations", "associative"): {
        "unit": "boolean",
        "conditioning": "Each emitted property-association result.",
        "denominator": "Not applicable; this Boolean interpretation flag is not a ratio.",
        "applicability": "Applicable to every property-association row and interpreted as exploratory and non-causal.",
        "missing_value_rule": "Never missing for a valid property-association row.",
    },
    ("Matched Embedding Interactions", "simultaneous_critical_value"): {
        "unit": "percentage_points",
        "conditioning": (
            "The four matched radii jointly within one graph model; the "
            "maximum absolute deviation is unstandardized and neither "
            "studentisation nor standardisation is applied."
        ),
        "denominator": "Not applicable; the critical value is an absolute deviation, not a ratio.",
        "applicability": "Applicable only to simultaneous matched-radius interaction rows for a complete four-radius family.",
        "missing_value_rule": (
            "NA when any required radius statistic or the simultaneous "
            "bootstrap family is undefined; otherwise required."
        ),
    },
    ("Equivalence Sensitivity", "simultaneous_critical_value"): {
        "unit": "percentage_points",
        "conditioning": (
            "The four matched radii jointly within one graph model; the "
            "maximum absolute deviation is unstandardized and neither "
            "studentisation nor standardisation is applied."
        ),
        "denominator": "Not applicable; the critical value is an absolute deviation, not a ratio.",
        "applicability": "Applicable only to simultaneous matched-radius interaction rows for a complete four-radius family.",
        "missing_value_rule": (
            "NA when any required radius statistic or the simultaneous "
            "bootstrap family is undefined; otherwise required."
        ),
    },
}

_VALUE_FIELDS_WITH_APPLICABILITY = frozenset(
    {
        "estimate",
        "ci_lower",
        "ci_upper",
        "simultaneous_ci_lower",
        "simultaneous_ci_upper",
        "numerator",
        "denominator",
        "association_estimate",
        "rank_sensitivity_estimate",
        "rank_sensitivity_ci_lower",
        "rank_sensitivity_ci_upper",
        "successful_stretch_mean",
        "euclidean_mean",
        "poincare_mean",
        "maximum",
    }
)


def _definition_is_tautological(header: str, definition: str) -> bool:
    """Reject definitions that add no meaning beyond the machine name."""

    normalized_header = " ".join(header.lower().replace("_", " ").split())
    normalized_definition = " ".join(
        definition.lower().replace("_", " ").strip(" .:;").split()
    )
    if normalized_definition in {
        normalized_header,
        f"the {normalized_header}",
        f"value of {normalized_header}",
        f"the value of {normalized_header}",
    }:
        return True
    if " recorded for " in f" {normalized_definition} ":
        return True
    header_words = set(normalized_header.split())
    meaning_words = {
        word
        for word in normalized_definition.split()
        if word not in header_words
        and word not in {"a", "an", "and", "for", "of", "or", "the", "this", "to"}
    }
    return len(meaning_words) < 1


def _semantic_registry() -> dict[tuple[str, str], dict[str, str]]:
    registry: dict[tuple[str, str], dict[str, str]] = {}
    for table, headers in TABLE_SCHEMA_COLUMNS.items():
        if table not in _TABLE_CONDITIONING or table not in _TABLE_DENOMINATORS:
            raise RuntimeError(f"semantic table metadata is missing for {table}")
        for header in headers:
            definition = _TABLE_FIELD_DEFINITION_OVERRIDES.get(
                (table, header), _FIELD_DEFINITIONS.get(header)
            )
            if not definition:
                raise RuntimeError(
                    f"semantic definition is missing for {table}.{header}"
                )
            if _definition_is_tautological(header, definition):
                raise RuntimeError(
                    f"semantic definition merely restates field name for {table}.{header}"
                )
            conditional_value = header in _VALUE_FIELDS_WITH_APPLICABILITY
            semantic = {
                "definition": definition,
                "unit": _field_unit(header),
                "conditioning": _TABLE_CONDITIONING[table],
                "denominator": _TABLE_DENOMINATORS[table],
                "applicability": (
                    "Governed by the row's status, applicability, and na_reason fields; a defined zero remains applicable."
                    if conditional_value
                    else "Applicable whenever a row exists at the table scope declared by the protocol."
                ),
                "missing_value_rule": (
                    "NA only when the method or estimand is inapplicable, its denominator is zero, or the frozen completeness rule fails; na_reason is mandatory and zero is never recoded as NA."
                    if conditional_value
                    else "Required for every emitted row; absence is a reporting-validation error rather than an imputed value."
                ),
            }
            semantic.update(
                _TABLE_FIELD_SEMANTIC_OVERRIDES.get((table, header), {})
            )
            registry[(table, header)] = semantic
    return registry


SEMANTIC_DEFINITION_REGISTRY = _semantic_registry()


def validate_semantic_definition_registry() -> None:
    expected = {
        (table, header)
        for table, headers in TABLE_SCHEMA_COLUMNS.items()
        for header in headers
    }
    observed = set(SEMANTIC_DEFINITION_REGISTRY)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"semantic registry coverage mismatch; missing={missing}, extra={extra}"
        )
    required = (
        "definition",
        "unit",
        "conditioning",
        "denominator",
        "applicability",
        "missing_value_rule",
    )
    for (table, header), semantic in SEMANTIC_DEFINITION_REGISTRY.items():
        for field in required:
            value = semantic.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"semantic registry {field} is empty for {table}.{header}"
                )
        if _definition_is_tautological(header, semantic["definition"]):
            raise ValueError(
                f"semantic definition merely restates field name for {table}.{header}"
            )


def data_dictionary(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Build complete definitions from the validated output schema."""

    validate_semantic_definition_registry()
    rows: list[dict[str, object]] = []
    for table, values in tables.items():
        if table not in TABLE_SCHEMA_COLUMNS:
            raise ValueError(f"unknown reporting table {table}")
        headers: list[str] = list(TABLE_SCHEMA_COLUMNS[table])
        for row in values:
            for key in row:
                if key not in headers:
                    raise ValueError(f"undocumented reporting column {table}.{key}")
        for header in headers:
            semantic = SEMANTIC_DEFINITION_REGISTRY[(table, header)]
            rows.append(
                {
                    "schema": REPORTING_SEMANTIC_REGISTRY_SCHEMA,
                    "table": table,
                    "column": header,
                    **semantic,
                }
            )
    return rows


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    headers: Sequence[str] | None = None,
) -> bytes:
    ordered_headers = list(headers or ())
    if not ordered_headers:
        for row in rows:
            for key in row:
                if key not in ordered_headers:
                    ordered_headers.append(key)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=ordered_headers,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )
    return output.getvalue().encode("utf-8")


def _forest(
    rows: Sequence[Mapping[str, object]],
    *,
    estimate: str,
    lower: str,
    upper: str,
    output: Path,
    title: str,
    equivalence_band: float | None = None,
    axis_unit: str = "percentage_points",
    concise_condition_labels: bool = False,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if all(row.get(key) is not None for key in (estimate, lower, upper))
    ]
    def display_label(
        row: Mapping[str, object], index: int
    ) -> str:
        explicit = row.get("label")
        if explicit is not None:
            return str(explicit)
        if concise_condition_labels:
            model = {
                "erdos_renyi": "ER",
                "barabasi_albert": "BA",
            }.get(str(row.get("model")), str(row.get("model", "")))
            design = ", ".join(
                f"{field}={row[field]}"
                for field in ("n", "m")
                if row.get(field) is not None
            )
            scope = {
                "model_condition_n_m_cell": "cell",
                "model_condition_n_m_marginal": "model marginal",
            }.get(str(row.get("scope")), str(row.get("scope", "")))
            condition = str(row.get("coordinate_condition_id", ""))
            pieces = [
                piece
                for piece in (model, scope, design, condition)
                if piece
            ]
            if pieces:
                return " | ".join(pieces)
        generated = " | ".join(
            str(row.get(field))
            for field in (
                "model",
                "n",
                "m",
                "coordinate_condition_id",
                "matched_radius",
                "metric",
            )
            if row.get(field) is not None
        )
        return generated or str(index)

    labels = [display_label(row, index) for index, row in enumerate(selected)]
    maximum_label_length = max((len(label) for label in labels), default=0)
    figure_width = min(18.0, max(9.0, 8.0 + 0.07 * maximum_label_length))
    figure_height = max(3.5, 1.8 + 0.34 * len(selected))
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    title_artist = figure.suptitle(title, fontsize=14)
    if equivalence_band is not None:
        axis.axvspan(
            -equivalence_band,
            equivalence_band,
            color="#d9ead3",
            alpha=0.7,
        )
    axis.axvline(0.0, color="black", linewidth=0.8)
    if selected:
        def converted(row: Mapping[str, object], field: str) -> float:
            value = float(row[field])
            unit = str(row.get("unit", ""))
            if axis_unit == "percentage_points" and unit == "proportion":
                return value * 100.0
            if axis_unit == "percentage_points" and unit not in (
                "percentage_points",
                "proportion",
            ):
                raise ValueError("percentage-point forest received incompatible units")
            if axis_unit == "correlation" and unit not in ("correlation", ""):
                raise ValueError("correlation forest received incompatible units")
            return value

        points = np.asarray([converted(row, estimate) for row in selected])
        lows = np.asarray([converted(row, lower) for row in selected])
        highs = np.asarray([converted(row, upper) for row in selected])
        y = np.arange(len(selected))
        axis.errorbar(
            points,
            y,
            xerr=np.vstack((points - lows, highs - points)),
            fmt="o",
            capsize=2,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=10)
    axis.set_xlabel(axis_unit.replace("_", " "))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas_box = figure.bbox
    title_box = title_artist.get_window_extent(renderer)
    label_boxes = [
        label.get_window_extent(renderer) for label in axis.get_yticklabels()
    ]

    def inside_canvas(box: object) -> bool:
        return bool(
            box.x0 >= canvas_box.x0
            and box.y0 >= canvas_box.y0
            and box.x1 <= canvas_box.x1
            and box.y1 <= canvas_box.y1
        )

    layout = {
        "rendered_row_count": len(selected),
        "display_labels": labels,
        "title_inside_canvas": inside_canvas(title_box),
        "ytick_labels_inside_canvas": all(
            inside_canvas(box) for box in label_boxes
        ),
        "axes_do_not_overlap_title": not axis.get_window_extent(
            renderer
        ).overlaps(title_box),
        "figure_width_inches": figure_width,
        "figure_height_inches": figure_height,
    }
    figure.savefig(output, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    return layout


def _relative_luminance(rgba: Sequence[float]) -> float:
    """Return WCAG relative luminance for an sRGB colour."""

    channels = []
    for component in rgba[:3]:
        value = float(component)
        channels.append(
            value / 12.92
            if value <= 0.04045
            else ((value + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(
    foreground: Sequence[float], background: Sequence[float]
) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _annotation_text_colour(
    background: Sequence[float],
) -> tuple[str, float]:
    """Choose the higher-contrast accessible annotation colour."""

    candidates = {
        "#111111": matplotlib.colors.to_rgba("#111111"),
        "white": matplotlib.colors.to_rgba("white"),
    }
    colour, rgba = max(
        candidates.items(),
        key=lambda item: _contrast_ratio(item[1], background),
    )
    return colour, _contrast_ratio(rgba, background)


def _save_empty_or_faceted_points(
    rows: Sequence[Mapping[str, object]],
    *,
    output: Path,
    title: str,
    group_fields: Sequence[str],
    x_field: str,
    y_field: str,
    x_label: str,
    y_label: str,
) -> dict[str, object]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get(x_field) is None or row.get(y_field) is None:
            continue
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    count = max(1, len(groups))
    columns = min(3, count)
    rows_count = int(np.ceil(count / columns))
    longest_component = max(
        (
            len(str(value))
            for group in groups
            for value in group
        ),
        default=20,
    )
    panel_width = min(8.0, max(5.0, 3.5 + 0.08 * longest_component))
    figure_width = max(7.0, panel_width * columns, 0.1 * len(title))
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(figure_width, 3.8 * rows_count),
        squeeze=False,
        constrained_layout=True,
    )
    figure.get_layout_engine().set(w_pad=0.18, h_pad=0.18)
    flat_axes = list(axes.flat)
    if not groups:
        flat_axes[0].text(0.5, 0.5, "No applicable rows", ha="center", va="center")
    for axis, (group, selected) in zip(flat_axes, sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])), strict=False):
        axis.scatter(
            [float(row[x_field]) for row in selected],
            [float(row[y_field]) for row in selected],
            alpha=0.55,
        )
        axis.margins(x=0.12, y=0.12)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        axis.set_title("\n".join(str(value) for value in group))
        axis.set(xlabel=x_label, ylabel=y_label)
    for axis in flat_axes[len(groups):]:
        axis.set_visible(False)
    title_artist = figure.suptitle(title)
    tolerance = 0.5

    def inside(box: object, canvas: object) -> bool:
        return bool(
            box.x0 >= canvas.x0 - tolerance
            and box.y0 >= canvas.y0 - tolerance
            and box.x1 <= canvas.x1 + tolerance
            and box.y1 <= canvas.y1 + tolerance
        )

    visible_axes = [axis for axis in flat_axes if axis.get_visible()]
    layout_expansion_attempts = 0
    for layout_expansion_attempts in range(5):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        canvas = figure.bbox
        panel_titles = [
            axis.title.get_window_extent(renderer) for axis in visible_axes
        ]
        label_boxes = [
            artist.get_window_extent(renderer)
            for axis in visible_axes
            for artist in (
                axis.xaxis.label,
                axis.yaxis.label,
                *axis.get_xticklabels(),
                *axis.get_yticklabels(),
            )
            if artist.get_visible() and artist.get_text()
        ]
        suptitle_box = title_artist.get_window_extent(renderer)
        title_intersections = sum(
            suptitle_box.overlaps(box) for box in panel_titles
        )
        adjacent_title_intersections = sum(
            panel_titles[left].overlaps(panel_titles[right])
            for left in range(len(panel_titles))
            for right in range(left + 1, len(panel_titles))
        )
        layout_is_valid = bool(
            inside(suptitle_box, canvas)
            and all(inside(box, canvas) for box in panel_titles)
            and all(inside(box, canvas) for box in label_boxes)
            and not title_intersections
            and not adjacent_title_intersections
        )
        if layout_is_valid:
            break
        width, height = figure.get_size_inches()
        figure.set_size_inches(width * 1.12, height * 1.15, forward=True)
    if not layout_is_valid:
        raise RuntimeError(
            "faceted figure layout is clipped or overlapping; "
            f"suptitle_inside={inside(suptitle_box, canvas)}, "
            f"panel_titles_inside={[inside(box, canvas) for box in panel_titles]}, "
            f"labels_inside={[inside(box, canvas) for box in label_boxes]}, "
            f"outside_label_boxes={[list(box.bounds) for box in label_boxes if not inside(box, canvas)]}, "
            f"suptitle_intersections={title_intersections}, "
            f"adjacent_title_intersections={adjacent_title_intersections}"
        )
    layout = {
        "canvas_bbox_pixels": [float(value) for value in canvas.bounds],
        "suptitle_bbox_pixels": [float(value) for value in suptitle_box.bounds],
        "panel_title_bboxes_pixels": [
            [float(value) for value in box.bounds] for box in panel_titles
        ],
        "label_bboxes_pixels": [
            [float(value) for value in box.bounds] for box in label_boxes
        ],
        "suptitle_inside_canvas": inside(suptitle_box, canvas),
        "panel_titles_inside_canvas": all(
            inside(box, canvas) for box in panel_titles
        ),
        "labels_inside_canvas": all(inside(box, canvas) for box in label_boxes),
        "suptitle_panel_title_intersection_count": int(title_intersections),
        "adjacent_panel_title_intersection_count": int(
            adjacent_title_intersections
        ),
        "panel_count": len(groups),
        "row_count": len(rows),
        "layout_expansion_attempts": layout_expansion_attempts,
    }
    figure.savefig(output, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    return layout


_ABSOLUTE_SUCCESS_SPEC = _FIGURE_SPEC_BY_FILE["absolute_success.png"]
_FAILURE_COMPOSITION_SPEC = _FIGURE_SPEC_BY_FILE["failure_composition.png"]
_CONDITION_ORDER = tuple(
    _ABSOLUTE_SUCCESS_SPEC["deterministic_condition_order"]
)
_CONDITION_LABELS = {
    "hydra_native": "Native Hydra",
    "mds_native": "Native MDS",
    "hydra_scaled_r050": "Hydra 0.50",
    "mds_scaled_r050": "MDS 0.50",
    "hydra_scaled_r070": "Hydra 0.70",
    "mds_scaled_r070": "MDS 0.70",
    "hydra_scaled_r085": "Hydra 0.85",
    "mds_scaled_r085": "MDS 0.85",
    "hydra_scaled_r095": "Hydra 0.95",
    "mds_scaled_r095": "MDS 0.95",
}
_MODEL_ORDER = ("erdos_renyi", "barabasi_albert")
_MODEL_LABELS = {
    "erdos_renyi": "ER (Erdos-Renyi)",
    "barabasi_albert": "BA (Barabasi-Albert)",
}
_SUCCESS_METHOD_LABELS = {
    "euclidean_success": "Euclidean",
    "poincare_success": "Ordinary Poincare",
    "repaired_poincare_success": "Repaired Poincare",
}
_METHOD_ORDER = (
    "euclidean_greedy",
    "poincare_greedy",
    "repaired_poincare_greedy",
)


def _plot_absolute_success(
    rows: Sequence[Mapping[str, object]], output: Path
) -> dict[str, int]:
    """Plot every native/matched condition with explicit method encodings."""

    method_encodings = _ABSOLUTE_SUCCESS_SPEC["method_encodings"]
    method_order = tuple(_SUCCESS_METHOD_LABELS)
    indexed = {
        (
            str(row.get("model")),
            str(row.get("coordinate_condition_id")),
            str(row.get("metric")),
        ): row
        for row in rows
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(18, 6.5),
        sharey=True,
        squeeze=False,
    )
    offsets = dict(zip(method_order, (-0.20, 0.0, 0.20), strict=True))
    applicable_count = 0
    na_count = 0
    for axis, model in zip(axes.flat, _MODEL_ORDER, strict=True):
        axis.axvspan(-0.5, 1.5, color="#ececec", alpha=0.9, zorder=0)
        axis.axvline(1.5, color="#777777", linewidth=0.8, linestyle="--")
        for method in method_order:
            encoding = method_encodings[method]
            label_used = False
            for condition_index, condition in enumerate(_CONDITION_ORDER):
                row = indexed.get((model, condition, method))
                position = condition_index + offsets[method]
                if row is None or row.get("estimate") is None:
                    na_count += 1
                    axis.text(
                        position,
                        0.018,
                        "N/A",
                        color=str(encoding["colour"]),
                        fontsize=6,
                        rotation=90,
                        ha="center",
                        va="bottom",
                    )
                    continue
                estimate = float(row["estimate"])
                lower = row.get("ci_lower")
                upper = row.get("ci_upper")
                yerr = None
                if lower is not None and upper is not None:
                    yerr = np.array(
                        [[estimate - float(lower)], [float(upper) - estimate]]
                    )
                axis.errorbar(
                    [position],
                    [estimate],
                    yerr=yerr,
                    fmt=str(encoding["marker"]),
                    color=str(encoding["colour"]),
                    markerfacecolor=(
                        "white"
                        if condition.endswith("_native")
                        else str(encoding["colour"])
                    ),
                    markeredgewidth=1.2,
                    capsize=2.5,
                    linewidth=1.0,
                    label=(
                        _SUCCESS_METHOD_LABELS[method]
                        if not label_used
                        else None
                    ),
                    zorder=3,
                )
                label_used = True
                applicable_count += 1
        axis.set_title(_MODEL_LABELS[model])
        axis.set_xticks(
            range(len(_CONDITION_ORDER)),
            [_CONDITION_LABELS[item] for item in _CONDITION_ORDER],
            rotation=40,
            ha="right",
        )
        axis.set_xlim(-0.55, len(_CONDITION_ORDER) - 0.45)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Complete coordinate condition (grey = native reference)")
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].set_ylabel("Routing success proportion (0 to 1)")
    handles, labels = axes.flat[1].get_legend_handles_labels()
    if handles:
        axes.flat[1].legend(
            handles, labels, loc="lower right", title="Routing method"
        )
    figure.suptitle(
        "Absolute routing success by graph model, coordinate condition, and method"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return {
        "applicable_point_count": applicable_count,
        "na_annotation_count": na_count,
    }


def _plot_failure_composition(
    rows: Sequence[Mapping[str, object]], output: Path
) -> dict[str, object]:
    """Plot stage/category-separated exact conditional failure proportions."""

    failure_order = tuple(
        str(item)
        for item in _FAILURE_COMPOSITION_SPEC["deterministic_failure_order"]
    )
    pooled: dict[tuple[str, str, str, str], list[int]] = {}
    for row in rows:
        key = (
            str(row.get("model")),
            str(row.get("method_id")),
            str(row.get("coordinate_condition_id")),
            f"{row.get('failure_stage')}:{row.get('failure_type')}",
        )
        numerator = row.get("numerator")
        denominator = row.get("denominator")
        if numerator is None or denominator is None or int(denominator) <= 0:
            continue
        sums = pooled.setdefault(key, [0, 0])
        sums[0] += int(numerator)
        sums[1] += int(denominator)

    figure, axes = plt.subplots(
        len(_MODEL_ORDER),
        len(_METHOD_ORDER),
        figsize=(23, 17),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    colour_map = matplotlib.colormaps["viridis"].with_extremes(bad="#bdbdbd")
    defined_cell_count = 0
    zero_cell_count = 0
    na_cell_count = 0
    minimum_annotation_contrast = float("inf")
    minimum_zero_annotation_contrast = float("inf")
    minimum_na_annotation_contrast = float("inf")
    image = None
    for model_index, model in enumerate(_MODEL_ORDER):
        for method_index, method in enumerate(_METHOD_ORDER):
            axis = axes[model_index, method_index]
            matrix = np.full(
                (len(_CONDITION_ORDER), len(failure_order)),
                np.nan,
                dtype=float,
            )
            for condition_index, condition in enumerate(_CONDITION_ORDER):
                for failure_index, failure in enumerate(failure_order):
                    counts = pooled.get((model, method, condition, failure))
                    if counts is None:
                        na_cell_count += 1
                        continue
                    matrix[condition_index, failure_index] = (
                        counts[0] / counts[1]
                    )
                    defined_cell_count += 1
                    if counts[0] == 0:
                        zero_cell_count += 1
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                vmin=0.0,
                vmax=1.0,
                cmap=colour_map,
                aspect="auto",
                interpolation="nearest",
            )
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    background = image.cmap(
                        np.ma.masked
                        if np.isnan(value)
                        else image.norm(float(value))
                    )
                    annotation_colour, contrast_ratio = (
                        _annotation_text_colour(background)
                    )
                    if contrast_ratio < 4.5:
                        raise RuntimeError(
                            "failure-composition annotation contrast is below 4.5:1"
                        )
                    minimum_annotation_contrast = min(
                        minimum_annotation_contrast, contrast_ratio
                    )
                    if np.isnan(value):
                        minimum_na_annotation_contrast = min(
                            minimum_na_annotation_contrast, contrast_ratio
                        )
                    elif value == 0.0:
                        minimum_zero_annotation_contrast = min(
                            minimum_zero_annotation_contrast, contrast_ratio
                        )
                    axis.text(
                        column_index,
                        row_index,
                        "N/A" if np.isnan(value) else f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=annotation_colour,
                    )
            axis.set_title(
                f"{_MODEL_LABELS[model]} | {method.replace('_', ' ')}"
            )
            axis.set_xticks(
                range(len(failure_order)),
                [item.replace(":", "\n", 1).replace("_", " ") for item in failure_order],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            axis.set_yticks(
                range(len(_CONDITION_ORDER)),
                [_CONDITION_LABELS[item] for item in _CONDITION_ORDER],
                fontsize=7,
            )
            axis.axhline(1.5, color="white", linewidth=1.5)
    figure.suptitle(
        "Failure composition by model, routing method, coordinate condition, stage, and category\n"
        "Grey N/A cells are not zero; initial and final stages are separate"
    )
    figure.subplots_adjust(
        left=0.10, right=0.90, bottom=0.16, top=0.90, wspace=0.16, hspace=0.24
    )
    if image is not None:
        colour_axis = figure.add_axes((0.925, 0.20, 0.012, 0.62))
        figure.colorbar(
            image,
            cax=colour_axis,
            label="Exact pooled numerator / failures at stated stage",
        )
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return {
        "matrix_shape_per_panel": [len(_CONDITION_ORDER), len(failure_order)],
        "panel_count": len(_MODEL_ORDER) * len(_METHOD_ORDER),
        "defined_cell_count": defined_cell_count,
        "zero_cell_count": zero_cell_count,
        "na_cell_count": na_cell_count,
        "annotation_contrast_algorithm": (
            "wcag_relative_luminance_choose_stronger_of_dark_or_white"
        ),
        "minimum_annotation_contrast_ratio": minimum_annotation_contrast,
        "minimum_zero_annotation_contrast_ratio": (
            None
            if minimum_zero_annotation_contrast == float("inf")
            else minimum_zero_annotation_contrast
        ),
        "minimum_na_annotation_contrast_ratio": (
            None
            if minimum_na_annotation_contrast == float("inf")
            else minimum_na_annotation_contrast
        ),
    }


def generate_required_figures(
    figure_directory: Path | str,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[Path, ...]:
    """Generate complete prespecified composites and record row coverage."""

    root = Path(figure_directory)
    root.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, dict[str, object]] = {}

    contrasts = list(tables.get("Matched Success Contrasts", ()))
    primary = [row for row in contrasts if row.get("metric") == "poincare_minus_euclidean"]
    _forest(primary, estimate="estimate", lower="ci_lower", upper="ci_upper", output=root / FIGURE_FILES[0], title="Poincare minus Euclidean success", axis_unit="percentage_points")
    coverage[FIGURE_FILES[0]] = {"source_table": "Matched Success Contrasts", "row_count": len(primary), "selection": "prespecified_poincare_minus_euclidean_rows"}

    repair = [row for row in contrasts if row.get("metric") in ("repaired_minus_unrepaired_poincare", "conditional_repair_recovery")]
    _forest(repair, estimate="estimate", lower="ci_lower", upper="ci_upper", output=root / FIGURE_FILES[1], title="Repair improvement", axis_unit="percentage_points")
    coverage[FIGURE_FILES[1]] = {"source_table": "Matched Success Contrasts", "row_count": len(repair), "selection": "prespecified_repair_improvement_and_recovery_rows"}

    interactions_all = list(tables.get("Matched Embedding Interactions", ()))
    interactions = [row for row in interactions_all if row.get("ci_type") == "simultaneous_95_percent_familywise" and row.get("stratum_count") == 9]
    _forest(interactions, estimate="estimate", lower="simultaneous_ci_lower", upper="simultaneous_ci_upper", output=root / FIGURE_FILES[2], title="Matched-radius embedding interaction", equivalence_band=EQUIVALENCE_MARGIN_PERCENTAGE_POINTS, axis_unit="percentage_points")
    coverage[FIGURE_FILES[2]] = {"source_table": "Matched Embedding Interactions", "row_count": len(interactions), "selection": "model_equal_stratum_simultaneous_rows"}

    marginals = [
        row
        for row in tables.get("Model Marginals", ())
        if row.get("metric") in _SUCCESS_METHOD_LABELS
        and row.get("coordinate_condition_id") in _CONDITION_ORDER
    ]
    absolute_counts = _plot_absolute_success(
        marginals, root / FIGURE_FILES[3]
    )
    coverage[FIGURE_FILES[3]] = {
        "source_table": "Model Marginals",
        "row_count": len(marginals),
        "selection": "all_native_and_matched_absolute_success_rows",
        **absolute_counts,
    }

    heat = [row for row in tables.get("Cell Estimates", ()) if row.get("metric") == "poincare_success" and row.get("estimate") is not None]
    heat_groups: dict[tuple[object, object], list[Mapping[str, object]]] = {}
    for row in heat:
        heat_groups.setdefault((row.get("model"), row.get("coordinate_condition_id")), []).append(row)
    panel_count = max(1, len(heat_groups))
    columns = min(4, panel_count)
    panel_rows = int(np.ceil(panel_count / columns))
    figure, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(max(6.5, 4.2 * columns), 3.7 * panel_rows),
        squeeze=False,
        constrained_layout=True,
    )
    if not heat_groups:
        axes.flat[0].text(0.5, 0.5, "No applicable rows", ha="center", va="center")
    for axis, (group, selected) in zip(axes.flat, sorted(heat_groups.items(), key=lambda item: tuple(str(v) for v in item[0])), strict=False):
        ns = sorted({int(row["n"]) for row in selected})
        ms = sorted({int(row["m"]) for row in selected})
        matrix = np.full((len(ns), len(ms)), np.nan)
        for row in selected:
            matrix[ns.index(int(row["n"])), ms.index(int(row["m"]))] = float(row["estimate"])
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto")
        figure.colorbar(image, ax=axis, label="success proportion")
        axis.set_xticks(range(len(ms)), ms)
        axis.set_yticks(range(len(ns)), ns)
        axis.set(xlabel="m", ylabel="n", title=" | ".join(str(v) for v in group))
    for axis in list(axes.flat)[len(heat_groups):]:
        axis.set_visible(False)
    title_artist = figure.suptitle(
        "Poincare success: every model-condition n-m cell",
        fontsize=14,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas_bbox = figure.bbox
    title_bbox = title_artist.get_window_extent(renderer=renderer)
    panel_title_bboxes = [
        axis.title.get_window_extent(renderer=renderer)
        for axis in axes.flat
        if axis.get_visible() and axis.title.get_text()
    ]
    tolerance = 0.5
    def inside_canvas(box: object) -> bool:
        return bool(
            box.x0 >= canvas_bbox.x0 - tolerance
            and box.y0 >= canvas_bbox.y0 - tolerance
            and box.x1 <= canvas_bbox.x1 + tolerance
            and box.y1 <= canvas_bbox.y1 + tolerance
        )

    title_inside_canvas = inside_canvas(title_bbox)
    panel_titles_inside_canvas = all(inside_canvas(box) for box in panel_title_bboxes)
    title_panel_intersections = sum(
        bool(title_bbox.overlaps(box)) for box in panel_title_bboxes
    )
    if (
        not title_inside_canvas
        or not panel_titles_inside_canvas
        or title_panel_intersections
    ):
        raise RuntimeError("heatmap title layout is outside or overlaps the canvas")
    figure.savefig(root / FIGURE_FILES[4], dpi=160)
    plt.close(figure)
    coverage[FIGURE_FILES[4]] = {
        "source_table": "Cell Estimates",
        "row_count": len(heat),
        "panel_count": len(heat_groups),
        "selection": "all_poincare_success_cells",
        "canvas_bbox_pixels": list(canvas_bbox.bounds),
        "suptitle_bbox_pixels": list(title_bbox.bounds),
        "panel_title_bboxes_pixels": [list(box.bounds) for box in panel_title_bboxes],
        "suptitle_inside_canvas": title_inside_canvas,
        "panel_titles_inside_canvas": panel_titles_inside_canvas,
        "suptitle_panel_title_intersection_count": title_panel_intersections,
    }

    failures = [
        row
        for row in tables.get("Failure Composition", ())
        if row.get("estimand") == "composition_among_failures_at_stage"
    ]
    failure_counts = _plot_failure_composition(
        failures, root / FIGURE_FILES[5]
    )
    coverage[FIGURE_FILES[5]] = {
        "source_table": "Failure Composition",
        "row_count": len(failures),
        "selection": "all_stage_separated_conditional_composition_rows",
        **failure_counts,
    }

    common = [row for row in tables.get("Common-Success Stretch", ()) if row.get("estimate") is not None]
    common_layout = _forest(
        common,
        estimate="estimate",
        lower="ci_lower",
        upper="ci_upper",
        output=root / FIGURE_FILES[6],
        title="Paired Poincare-minus-Euclidean common-success stretch",
        axis_unit="hop_stretch_ratio",
        concise_condition_labels=True,
    )
    full_identity_fields = (
        "scope",
        "model",
        "n",
        "m",
        "coordinate_condition_id",
        "embedding_family",
        "condition_kind",
        "matched_radius",
        "metric",
    )
    coverage[FIGURE_FILES[6]] = {
        "source_table": "Common-Success Stretch",
        "row_count": len(common),
        "selection": "all_common_success_paired_difference_rows",
        "display_label_rule": "model_abbreviation_n_m_exact_coordinate_condition",
        "full_identity_fields": list(full_identity_fields),
        "full_identities": [
            {field: row.get(field) for field in full_identity_fields}
            for row in common
        ],
        **common_layout,
    }

    distortion = [
        {**row, "diagnostic_index": index}
        for index, row in enumerate(
            row
            for row in tables.get("Distortion Diagnostics", ())
            if row.get("estimate") is not None
        )
    ]
    distortion_layout = _save_empty_or_faceted_points(distortion, output=root / FIGURE_FILES[7], title="Geometry-specific distortion diagnostics (exploratory, noncausal)", group_fields=("model", "geometry", "metric"), x_field="diagnostic_index", y_field="estimate", x_label="condition-stratum index (see table)", y_label="diagnostic value")
    coverage[FIGURE_FILES[7]] = {"source_table": "Distortion Diagnostics", "row_count": len(distortion), "selection": "all_geometry_specific_rows", **distortion_layout}

    runtimes = [row for row in tables.get("Operational Runtime", ()) if row.get("total_seconds") is not None]
    runtime_layout = _save_empty_or_faceted_points(runtimes, output=root / FIGURE_FILES[8], title="Operational total runtime by graph model", group_fields=("model",), x_field="n", y_field="total_seconds", x_label="n", y_label="seconds")
    coverage[FIGURE_FILES[8]] = {"source_table": "Operational Runtime", "row_count": len(runtimes), "selection": "all_graph_total_runtime_rows_no_method_claim", **runtime_layout}

    for filename, specification in _FIGURE_SPEC_BY_FILE.items():
        observed = coverage.get(filename)
        if not isinstance(observed, dict) or any(
            observed.get(field) != specification[field]
            for field in ("source_table", "selection")
        ):
            raise RuntimeError(
                f"figure coverage differs from frozen specification: {filename}"
            )
        for field, value in specification.items():
            if field != "file":
                observed[field] = value

    (root / "figure_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tuple(root / name for name in FIGURE_FILES)


def _validate_finite(value: object, path: str = "root") -> None:
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"non-finite reporting value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def _csv_filename(sheet: str) -> str:
    return sheet.lower().replace(" ", "_").replace("-", "_") + ".csv"


def _validate_workbook(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        forbidden = [name for name in names if "externalLink" in name or name.endswith("vbaProject.bin")]
        if forbidden:
            raise ValueError("workbook contains external links or macros")
        workbook = archive.read("xl/workbook.xml")
        xml = b"".join(archive.read(name) for name in names if name.endswith(".xml") or name.endswith(".rels"))
    if workbook.count(b"<sheet ") != len(WORKBOOK_SHEETS):
        raise ValueError("workbook sheet count is incorrect")
    if b"<f>" in xml or b"<f " in xml or b'TargetMode="External"' in xml or b'state="hidden"' in xml:
        raise ValueError("workbook contains formulas, external relationships, or hidden sheets")


def validate_reporting_bundle(output_directory: Path | str) -> dict[str, object]:
    """Validate hashes, row counts, workbook safety, figures, and completion."""

    output = Path(output_directory)
    manifest_path = output / "analysis_manifest.json"
    completion_path = output / "analysis_complete.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise ValueError("analysis manifest or completion marker is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    validation_evidence = manifest.get("analysis_validation_evidence")
    if not isinstance(validation_evidence, Mapping):
        raise ValueError("measured analysis validation evidence is missing")
    ledger_snapshot = validation_evidence.get("scientific_operation_ledger")
    if not isinstance(ledger_snapshot, Mapping):
        raise ValueError("measured scientific-operation ledger is missing")
    require_zero_scientific_operations(
        ledger_snapshot,
        context="Iteration 2 reporting validation",
    )
    if completion.get("schema") != "greedy_routing_iteration2_analysis_completion_v2":
        raise ValueError("analysis completion schema is invalid")
    if completion.get("analysis_manifest_sha256") != sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("analysis completion manifest hash mismatch")
    if manifest.get("protocol_hash") != COMBINED_PROTOCOL_HASH or manifest.get("output_schema_hash") != OUTPUT_SCHEMA_HASH:
        raise ValueError("analysis manifest identity mismatch")
    payload = manifest.get("excluded_fixture_payload")
    raw_generation = manifest.get("raw_generation_identity")
    if not isinstance(raw_generation, Mapping):
        raise ValueError("reporting raw-generation provenance is invalid")
    if payload is None:
        expected_analysis_identity = ITERATION2_ANALYSIS_IDENTITY
        if (
            manifest.get("scientific_status")
            != "iteration2_prespecified_scientific_analysis"
            or manifest.get("production_compatible") is not True
            or raw_generation.get("run_identity") != ITERATION2_RUN_IDENTITY
            or raw_generation.get("scientific_status")
            != "iteration2_prespecified_scientific_run"
            or raw_generation.get("production_compatible") is not True
        ):
            raise ValueError("production reporting provenance is invalid")
    else:
        if not isinstance(payload, Mapping):
            raise ValueError("excluded reporting payload is invalid")
        normalized = validate_excluded_fixture_payload(payload)
        expected_analysis_identity = excluded_analysis_identity(normalized)
        if (
            manifest.get("excluded_fixture_payload_sha256")
            != excluded_fixture_payload_hash(normalized)
            or manifest.get("scientific_status") != EXCLUDED_SCIENTIFIC_STATUS
            or manifest.get("production_compatible") is not False
            or raw_generation.get("run_identity")
            != excluded_raw_identity(normalized)
            or raw_generation.get("scientific_status")
            != EXCLUDED_SCIENTIFIC_STATUS
            or raw_generation.get("production_compatible") is not False
        ):
            raise ValueError("excluded reporting provenance is invalid")
    if (
        manifest.get("analysis_identity") != expected_analysis_identity
        or completion.get("analysis_identity") != expected_analysis_identity
        or (
            output.name != expected_analysis_identity
            and not output.name.startswith(f".{expected_analysis_identity}.tmp-")
        )
    ):
        raise ValueError("analysis publication identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("analysis file manifest is missing")
    for relative, identity in files.items():
        path = output / str(relative)
        if not path.is_file():
            raise ValueError(f"analysis artifact is missing: {relative}")
        payload = path.read_bytes()
        if len(payload) != int(identity["size_bytes"]) or sha256(payload).hexdigest() != identity["sha256"]:
            raise ValueError(f"analysis artifact hash mismatch: {relative}")
    expected_files = set(files) | {"analysis_manifest.json", "analysis_complete.json"}
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("analysis bundle has missing or unexpected files")
    row_counts = manifest.get("table_row_counts", {})
    for sheet in WORKBOOK_SHEETS:
        csv_path = output / _csv_filename(sheet)
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            observed = sum(1 for _ in csv.DictReader(stream))
        if observed != int(row_counts[sheet]):
            raise ValueError(f"CSV row count mismatch for {sheet}")
    _validate_workbook(output / "iteration2_results.xlsx")
    for name in FIGURE_FILES:
        path = output / "figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required figure is missing or empty: {name}")
    coverage = json.loads((output / "figures" / "figure_coverage.json").read_text(encoding="utf-8"))
    if set(coverage) != set(FIGURE_FILES):
        raise ValueError("figure coverage does not match the prespecified figure set")
    return manifest


def build_reporting_bundle(
    output_directory: Path | str,
    *,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    raw_location: str,
    raw_file_hashes: Mapping[str, str],
    limitations: Sequence[str],
    raw_generation_identity: Mapping[str, object] | None = None,
    analysis_source_identity: Mapping[str, object] | None = None,
    raw_tree_fingerprint: Mapping[str, object] | str | None = None,
    raw_manifest_sha256: str | None = None,
    raw_completion_sha256: str | None = None,
    analysis_validation_evidence: Mapping[str, object] | None = None,
    source_commit: str | None = None,
    excluded_fixture_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically publish a complete, validated Iteration 2 analysis bundle."""

    output = Path(output_directory)
    if not isinstance(analysis_validation_evidence, Mapping):
        raise ValueError("measured analysis validation evidence is required")
    ledger_snapshot = analysis_validation_evidence.get(
        "scientific_operation_ledger"
    )
    if not isinstance(ledger_snapshot, Mapping):
        raise ValueError("measured scientific-operation ledger is required")
    require_zero_scientific_operations(
        ledger_snapshot,
        context="Iteration 2 reporting publication",
    )
    if excluded_fixture_payload is None:
        publication_identity = ITERATION2_ANALYSIS_IDENTITY
        scientific_status = "iteration2_prespecified_scientific_analysis"
        production_compatible = True
        normalized_excluded_payload = None
        excluded_payload_hash = None
        raw_generation = dict(raw_generation_identity or {})
        if (
            raw_generation.get("run_identity") != ITERATION2_RUN_IDENTITY
            or raw_generation.get("scientific_status")
            != "iteration2_prespecified_scientific_run"
            or raw_generation.get("production_compatible") is not True
        ):
            raise ValueError(
                "production reporting requires production raw provenance"
            )
    else:
        normalized_excluded_payload = validate_excluded_fixture_payload(
            excluded_fixture_payload
        )
        publication_identity = excluded_analysis_identity(
            normalized_excluded_payload
        )
        scientific_status = EXCLUDED_SCIENTIFIC_STATUS
        production_compatible = False
        excluded_payload_hash = excluded_fixture_payload_hash(
            normalized_excluded_payload
        )
        raw_generation = dict(raw_generation_identity or {})
        if (
            raw_generation.get("run_identity")
            != excluded_raw_identity(normalized_excluded_payload)
            or raw_generation.get("scientific_status")
            != EXCLUDED_SCIENTIFIC_STATUS
            or raw_generation.get("production_compatible") is not False
        ):
            raise ValueError("excluded reporting raw identity mismatch")
    if output.name != publication_identity:
        raise ValueError("reporting target does not match its publication identity")
    if output.exists():
        raise FileExistsError("Iteration 2 analysis directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_identity = dict(analysis_source_identity or {})
    resolved_source_commit = str(source_identity.get("source_commit") or source_commit or "")
    if len(resolved_source_commit) != 40:
        raise ValueError("analysis source commit provenance is required")
    machine_tables = {name: [dict(row) for row in tables.get(name, ())] for name in WORKBOOK_SHEETS}
    _validate_finite(machine_tables)
    machine_tables["README"] = [
        {
            "analysis_schema": ITERATION2_ANALYSIS_SCHEMA,
            "protocol_hash": COMBINED_PROTOCOL_HASH,
            "data_generation_hash": DATA_GENERATION_HASH,
            "analysis_plan_hash": ANALYSIS_PLAN_HASH,
            "output_schema_hash": OUTPUT_SCHEMA_HASH,
            "source_commit": resolved_source_commit,
            "independent_experimental_unit": "graph",
            "primary_estimand": "matched_radius_embedding_interaction_within_graph_model",
            "primary_conditions": "Hydra and MDS compared at the same radius",
            "native_references_are_secondary": True,
            "scaled_hydra_is_non_isometric_sensitivity": True,
            "raw_data_location": raw_location,
            "raw_data_schema": (
                ITERATION2_RUN_IDENTITY
                if normalized_excluded_payload is None
                else excluded_raw_identity(normalized_excluded_payload)
            ),
            "limitations": list(limitations),
        }
    ]
    machine_tables["Provenance"] = [
        {
            "protocol_hash": COMBINED_PROTOCOL_HASH,
            "data_generation_hash": DATA_GENERATION_HASH,
            "analysis_plan_hash": ANALYSIS_PLAN_HASH,
            "output_schema_hash": OUTPUT_SCHEMA_HASH,
            "source_commit": resolved_source_commit,
            "raw_location": raw_location,
            "raw_generation_identity": dict(raw_generation_identity or {}),
            "analysis_source_identity": source_identity,
            "raw_tree_fingerprint": raw_tree_fingerprint,
            "raw_manifest_sha256": raw_manifest_sha256,
            "raw_completion_sha256": raw_completion_sha256,
            "raw_file_hashes": dict(raw_file_hashes),
            "analysis_validation_evidence": dict(
                analysis_validation_evidence or {}
            ),
            "formula_cells": 0,
            "hidden_outcome_dependent_filters": False,
        }
    ]
    machine_tables["Data Dictionary"] = data_dictionary(machine_tables)
    _validate_finite(machine_tables)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for sheet, rows in machine_tables.items():
            (staging / _csv_filename(sheet)).write_bytes(
                _csv_bytes(rows, TABLE_SCHEMA_COLUMNS[sheet])
            )
        write_xlsx(staging / "iteration2_results.xlsx", machine_tables)
        figures = generate_required_figures(staging / "figures", machine_tables)
        _validate_workbook(staging / "iteration2_results.xlsx")
        files: dict[str, dict[str, object]] = {}
        for path in sorted((path for path in staging.rglob("*") if path.is_file()), key=lambda item: item.relative_to(staging).as_posix()):
            payload = path.read_bytes()
            files[path.relative_to(staging).as_posix()] = {
                "size_bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
        manifest = {
            "schema": "greedy_routing_iteration2_analysis_manifest_v2",
            "analysis_schema": ITERATION2_ANALYSIS_SCHEMA,
            "protocol_hash": COMBINED_PROTOCOL_HASH,
            "data_generation_hash": DATA_GENERATION_HASH,
            "analysis_plan_hash": ANALYSIS_PLAN_HASH,
            "output_schema_hash": OUTPUT_SCHEMA_HASH,
            "analysis_identity": publication_identity,
            "scientific_status": scientific_status,
            "production_compatible": production_compatible,
            "excluded_fixture_payload": normalized_excluded_payload,
            "excluded_fixture_payload_sha256": excluded_payload_hash,
            "source_commit": resolved_source_commit,
            "raw_location": raw_location,
            "raw_generation_identity": dict(raw_generation_identity or {}),
            "analysis_source_identity": source_identity,
            "raw_tree_fingerprint": raw_tree_fingerprint,
            "raw_manifest_sha256": raw_manifest_sha256,
            "raw_completion_sha256": raw_completion_sha256,
            "raw_file_hashes": dict(raw_file_hashes),
            "analysis_validation_evidence": dict(
                analysis_validation_evidence or {}
            ),
            "workbook_sheets": list(WORKBOOK_SHEETS),
            "table_row_counts": {name: len(rows) for name, rows in machine_tables.items()},
            "figures": [path.name for path in figures],
            "figure_coverage": json.loads((staging / "figures" / "figure_coverage.json").read_text(encoding="utf-8")),
            "files": files,
        }
        manifest_path = staging / "analysis_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        completion = {
            "schema": "greedy_routing_iteration2_analysis_completion_v2",
            "protocol_hash": COMBINED_PROTOCOL_HASH,
            "analysis_identity": publication_identity,
            "analysis_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            "artifact_file_count": len(files),
        }
        (staging / "analysis_complete.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_reporting_bundle(staging)
        staging.replace(output)
        return validate_reporting_bundle(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
