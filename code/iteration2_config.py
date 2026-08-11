"""Frozen, outcome-independent protocol for the balanced Iteration 2 design."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2s, sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Iterable

from iteration2_runtime_guard import scientific_operation_boundary


ITERATION2_PROTOCOL_SCHEMA = "greedy_routing_iteration2_protocol_v3"
ITERATION2_RESULT_SCHEMA = "greedy_routing_iteration2_graph_result_v3"
ITERATION2_ANALYSIS_SCHEMA = "greedy_routing_iteration2_analysis_v3"
ITERATION2_OUTPUT_SCHEMA = "greedy_routing_iteration2_output_schema_v2"
ITERATION2_CAPACITY_PROFILE_SCHEMA = (
    "greedy_routing_iteration2_capacity_profile_v2"
)
ITERATION2_SEED_IDENTITY_VERSION = 2
CANONICAL_SERIALIZATION_SCHEMA = "greedy_routing_canonical_json_v1"
ROUTING_PRIORITY_VERSION = "iteration2_keyed_blake2s_priority_v1"
ITERATION2_FLOAT_TYPE = "float64"

OUTPUT_WORKBOOK_SHEETS = (
    "README",
    "Results Dashboard",
    "Data Dictionary",
    "Graph Metrics",
    "Graph-Level Estimates",
    "Cell Estimates",
    "Model Marginals",
    "Matched Success Contrasts",
    "Matched Embedding Interactions",
    "Complete-System Comparisons",
    "Native References",
    "Equivalence Sensitivity",
    "Repair Denominators",
    "Failure Composition",
    "Distance Bands",
    "Method Stretch",
    "Common-Success Stretch",
    "Recovered Stretch",
    "Distortion Diagnostics",
    "Property Associations",
    "Model Contrasts",
    "Routing Resources",
    "Operational Runtime",
    "Evidence Table",
    "Validation Summary",
    "Provenance",
)
OUTPUT_FIGURE_FILES = (
    "poincare_minus_euclidean_forest.png",
    "repair_improvement_forest.png",
    "matched_radius_interactions.png",
    "absolute_success.png",
    "n_m_success_heatmaps.png",
    "failure_composition.png",
    "success_stretch_paired.png",
    "distortion_success_exploratory.png",
    "runtime_scaling.png",
)
OUTPUT_FIGURE_SPECIFICATIONS = (
    {
        "file": OUTPUT_FIGURE_FILES[0],
        "source_table": "Matched Success Contrasts",
        "selection": "prespecified_poincare_minus_euclidean_rows",
        "display_unit": "percentage_points",
        "faceting": "one_labelled_forest_row_per_reported_table_row",
    },
    {
        "file": OUTPUT_FIGURE_FILES[1],
        "source_table": "Matched Success Contrasts",
        "selection": "prespecified_repair_improvement_and_recovery_rows",
        "display_unit": "percentage_points",
        "faceting": "one_labelled_forest_row_per_reported_table_row",
    },
    {
        "file": OUTPUT_FIGURE_FILES[2],
        "source_table": "Matched Embedding Interactions",
        "selection": "model_equal_stratum_simultaneous_rows",
        "display_unit": "percentage_points",
        "faceting": "model_by_radius_labelled_forest_rows",
    },
    {
        "file": OUTPUT_FIGURE_FILES[3],
        "source_table": "Model Marginals",
        "selection": "all_native_and_matched_absolute_success_rows",
        "display_unit": "proportion",
        "faceting": "separate_graph_model_panels",
        "model_facet_field": "model",
        "coordinate_condition_field": "coordinate_condition_id",
        "embedding_family_field": "embedding_family",
        "radius_field": "matched_radius",
        "condition_role_field": "condition_kind",
        "method_field": "metric",
        "method_encodings": {
            "euclidean_success": {"colour": "#1f77b4", "marker": "o"},
            "poincare_success": {"colour": "#d62728", "marker": "s"},
            "repaired_poincare_success": {
                "colour": "#2ca02c",
                "marker": "^",
            },
        },
        "estimate_field": "estimate",
        "interval_fields": ("ci_lower", "ci_upper"),
        "na_policy": "explicit_NA_annotation_never_zero",
        "deterministic_condition_order": (
            "hydra_native",
            "mds_native",
            "hydra_scaled_r050",
            "mds_scaled_r050",
            "hydra_scaled_r070",
            "mds_scaled_r070",
            "hydra_scaled_r085",
            "mds_scaled_r085",
            "hydra_scaled_r095",
            "mds_scaled_r095",
        ),
    },
    {
        "file": OUTPUT_FIGURE_FILES[4],
        "source_table": "Cell Estimates",
        "selection": "all_poincare_success_cells",
        "display_unit": "proportion",
        "faceting": "graph_model_by_coordinate_condition",
        "layout": "constrained_layout_with_reserved_figure_suptitle_space",
        "title_acceptance": (
            "suptitle_and_all_panel_titles_inside_canvas_without_intersection"
        ),
    },
    {
        "file": OUTPUT_FIGURE_FILES[5],
        "source_table": "Failure Composition",
        "selection": "all_stage_separated_conditional_composition_rows",
        "display_unit": "proportion",
        "faceting": "graph_model_by_routing_method_heatmaps",
        "model_field": "model",
        "method_field": "method_id",
        "coordinate_condition_field": "coordinate_condition_id",
        "embedding_family_field": "embedding_family",
        "radius_field": "matched_radius",
        "condition_role_field": "condition_kind",
        "failure_stage_field": "failure_stage",
        "failure_category_field": "failure_type",
        "estimate_field": "estimate",
        "estimate_unit": "proportion_among_failures_at_stage",
        "numerator_field": "numerator",
        "denominator_field": "denominator",
        "aggregation": "pooled_exact_numerator_over_denominator",
        "na_policy": "masked_grey_and_annotated_NA_never_zero",
        "annotation_contrast_algorithm": (
            "wcag_relative_luminance_choose_stronger_of_dark_or_white"
        ),
        "minimum_annotation_contrast_ratio_required": 4.5,
        "deterministic_condition_order": (
            "hydra_native",
            "mds_native",
            "hydra_scaled_r050",
            "mds_scaled_r050",
            "hydra_scaled_r070",
            "mds_scaled_r070",
            "hydra_scaled_r085",
            "mds_scaled_r085",
            "hydra_scaled_r095",
            "mds_scaled_r095",
        ),
        "deterministic_failure_order": (
            "initial:local_minimum",
            "final:local_minimum",
            "final:repair_unavailable_at_source",
            "final:no_alternative_after_backtracking",
            "final:post_repair_local_minimum",
            "final:post_repair_attempted_revisit",
        ),
    },
    {
        "file": OUTPUT_FIGURE_FILES[6],
        "source_table": "Common-Success Stretch",
        "selection": "all_common_success_paired_difference_rows",
        "display_unit": "hop_stretch_ratio_difference",
        "faceting": "one_labelled_forest_row_per_reported_table_row",
        "display_label_rule": (
            "model_abbreviation_n_m_exact_coordinate_condition"
        ),
        "full_identity_fields": (
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
        "layout": (
            "dynamic_dimensions_constrained_layout_figure_title_tight_bbox"
        ),
    },
    {
        "file": OUTPUT_FIGURE_FILES[7],
        "source_table": "Distortion Diagnostics",
        "selection": "all_geometry_specific_rows",
        "display_unit": "metric_specific_natural_scale",
        "faceting": "graph_model_by_geometry_by_diagnostic",
    },
    {
        "file": OUTPUT_FIGURE_FILES[8],
        "source_table": "Operational Runtime",
        "selection": "all_graph_total_runtime_rows_no_method_claim",
        "display_unit": "seconds",
        "faceting": "graph_model",
    },
)
OUTPUT_EXCEL_MAXIMUM_ROWS = 1_048_576
REPORTING_TABLE_SCHEMA_HASH = (
    "b76a40e5a950599f15dfaf0ba6318f4dab73372da3af500d01cdecfae5e5fa40"
)
REPORTING_SEMANTIC_REGISTRY_SCHEMA = (
    "greedy_routing_iteration2_semantic_definition_registry_v1"
)
OUTPUT_DISPLAY_UNIT_RULES = {
    "stored_success_and_recovery": "proportion_0_to_1",
    "success_contrast_axes": "percentage_points_multiply_proportion_once_by_100",
    "already_percentage_point_values": "no_additional_scaling",
    "correlations": "natural_scale_minus_1_to_1_never_percentage",
    "stretch": "dimensionless_hop_ratio_or_paired_ratio_difference",
    "routing_resources": "integer_or_mean_count_as_labelled",
    "operational_runtime": "seconds_no_method_speed_claim",
}

MATCHED_RADII = (0.50, 0.70, 0.85, 0.95)
MATCHED_RADIUS_LABELS = ("r050", "r070", "r085", "r095")
EMBEDDING_FAMILIES = ("hydra", "mds")
ROUTING_METHODS = (
    "euclidean_greedy",
    "poincare_greedy",
    "repaired_poincare_greedy",
)
HYDRA_NATIVE_CONDITION = "hydra_native"
MDS_NATIVE_ARTIFACT = "mds_native"
MATCHED_CONDITIONS = tuple(
    f"{family}_scaled_{label}"
    for family in EMBEDDING_FAMILIES
    for label in MATCHED_RADIUS_LABELS
)
ROUTABLE_CONDITIONS = (HYDRA_NATIVE_CONDITION, *MATCHED_CONDITIONS)

GRAPH_MODELS = ("erdos_renyi", "barabasi_albert")
N_VALUES = (100, 300, 1_000)
M_VALUES = (4, 8, 16)
GRAPH_REPETITIONS = 20
GRAPH_REPETITIONS_APPROVED = True
PAIRS_PER_GRAPH = 1_000
MAX_ER_ATTEMPTS = 50

GRAPH_MASTER_SEED = 11_000_003
EMBEDDING_PROVENANCE_MASTER_SEED = 12_000_003
PAIR_MASTER_SEED = 13_000_003
BOOTSTRAP_MASTER_SEED = 14_000_003
ROUTING_PRIORITY_MASTER_SEED = 15_000_017
VALIDATION_SENTINEL_MASTER_SEED = 16_000_019
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
EQUIVALENCE_MARGIN_PERCENTAGE_POINTS = 1.0
EQUIVALENCE_MARGIN_APPROVED = True
MULTIPLE_RADIUS_PROCEDURE = (
    "paired_graph_bootstrap_maximum_absolute_deviation_band"
)
PROPERTY_MULTIPLICITY_PROCEDURE = (
    "stratified_shared_permutation_maximum_absolute_statistic_fwer"
)
PROPERTY_ASSOCIATION_FAMILY_ID = "iteration2_property_associations_v2"
PROPERTY_ASSOCIATION_REPLICATES = 10_000
PROPERTY_ASSOCIATION_PRIMARY = "within_stratum_residual_pearson"
PROPERTY_ASSOCIATION_SENSITIVITY = "within_stratum_residual_midrank_spearman"
PROPERTY_ASSOCIATION_P_VALUE = (
    "stratified_graph_permutation_two_sided_plus_one"
)
PROPERTY_ASSOCIATION_CI = (
    "stratified_paired_graph_bootstrap_noninterpolated_percentile"
)
PROPERTY_ASSOCIATION_STRATA = "model_by_n_by_m"
PROPERTY_ASSOCIATION_EXPLORATORY = True
PROPERTY_ASSOCIATION_ALPHA = 0.05
NETWORK_PROPERTIES = (
    "average_degree",
    "maximum_degree",
    "population_degree_variance",
    "average_clustering_coefficient",
    "diameter",
    "average_shortest_path_length",
)
PROPERTY_ASSOCIATION_CONDITIONS = (
    "hydra_native",
    *MATCHED_CONDITIONS,
)
PROPERTY_ASSOCIATION_OUTCOMES = (
    "poincare_minus_euclidean_success",
    "common_success_poincare_minus_euclidean_stretch",
)
PROPERTY_MULTIPLICITY_FAMILIES = (
    "er_success_contrast",
    "ba_success_contrast",
    "er_common_success_stretch_contrast",
    "ba_common_success_stretch_contrast",
)
MIN_DEFINED_BOOTSTRAP_FRACTION = 0.95
FAILURE_SCHEMA_VERSION = (
    "iteration2_failure_v3_stage_specific_applicability"
)
EXCLUDED_FIXTURE_SEEDS = (
    41_000_003,
    41_000_019,
    41_000_037,
    41_000_063,
)

HYDRA_DIMENSION = 2
HYDRA_CURVATURE = -1.0
HYDRA_KAPPA = 1.0
HYDRA_CENTERING_TOLERANCE = 1e-10
HYDRA_CENTERING_MAX_ITERATIONS = 256
SPECTRAL_RELATIVE_TOLERANCE = 1e-12
MDS_CENTROID_TOLERANCE = 1e-12
ISOMETRY_ABSOLUTE_TOLERANCE = 1e-10
ISOMETRY_RELATIVE_TOLERANCE = 1e-9
BOUNDARY_ROUNDOFF_TOLERANCE = 1e-12
ROUTING_ROUNDOFF_MULTIPLIER = 64
HIGH_PRECISION_DECIMAL_DIGITS = 80
HIGH_PRECISION_SENTINEL_PAIRS_PER_GRAPH = 5
FULL_ORACLE_REPLICATE_INDICES = (0,)
FULL_ORACLE_M_VALUES = (4, 16)

ITERATION2_GRAPH_COUNT = 360
ITERATION2_RAW_GRAPH_FILE_COUNT = 360
ITERATION2_GRAPH_CHECKPOINT_DIRECTORY_COUNT = 360
ITERATION2_GRAPH_CHECKPOINT_FILES_PER_DIRECTORY = 3
ITERATION2_RAW_RUN_LEVEL_FILE_COUNT = 2
ITERATION2_RAW_TOTAL_FILE_COUNT = 1_082
ITERATION2_WORKBOOK_SHEET_COUNT = len(OUTPUT_WORKBOOK_SHEETS)
ITERATION2_FIGURE_COUNT = len(OUTPUT_FIGURE_FILES)

ITERATION1_RAW_DIRECTORY = "final_8e002ef20f96_a121c33a20ea"
ITERATION1_ANALYSIS_DIRECTORY = "analysis_8e002ef20f96_a121c33a20ea"
ITERATION1_RAW_TREE_SHA256 = (
    "b337f56488f8702d31bac8cee16c121d8e79665dd4e5351dd76eba3ee1e4267f"
)
ITERATION1_RAW_FILE_COUNT = 5_042
ITERATION1_RAW_BYTE_COUNT = 124_852_697
ITERATION1_RAW_MANIFEST_SHA256 = (
    "2f212fc3b8aa86fb576c3860d77ae71ef063959e3ee5e100898befb004fcc6a9"
)
ITERATION1_ANALYSIS_MANIFEST_SHA256 = (
    "1822a8540df877df041bf0e750cbdbc360c5ed1e1bf5053315e1338b26cc25ec"
)

_SEED_PERSON = b"GRP2seed"
_PAIR_PERSON = b"GRP2pair"
_SENTINEL_PERSON = b"GRP2sent"
_BOOTSTRAP_PERSON = b"GRP2boot"
_PROPERTY_PERSON = b"GRP2prop"
_UINT64_SPACE = 1 << 64


def _tagged(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("fingerprints reject non-finite floats")
        return {"__float64__": value.hex()}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("fingerprint keys must be strings")
        return {key: _tagged(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tagged(item) for item in value]
    raise ValueError(f"unsupported fingerprint value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize supported values with one explicit, UTF-8 canonical format."""

    return json.dumps(
        {
            "canonical_serialization_schema": CANONICAL_SERIALIZATION_SCHEMA,
            "value": _tagged(value),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def data_generation_payload() -> dict[str, object]:
    return {
        "schema": ITERATION2_PROTOCOL_SCHEMA,
        "phase": "data_generation",
        "scientific_rule": "no_outcome_dependent_tuning",
        "graph_design": {
            "models": GRAPH_MODELS,
            "n": N_VALUES,
            "m": M_VALUES,
            "replicates_per_model_n_m": GRAPH_REPETITIONS,
            "replicates_human_approved": GRAPH_REPETITIONS_APPROVED,
            "er_probability": "2*m*(n-m)/(n*(n-1))",
            "er_conditioning": "connected",
            "maximum_er_attempts": MAX_ER_ATTEMPTS,
        },
        "seeds": {
            "identity_version": ITERATION2_SEED_IDENTITY_VERSION,
            "graph_master": GRAPH_MASTER_SEED,
            "embedding_provenance_master": EMBEDDING_PROVENANCE_MASTER_SEED,
            "pair_master": PAIR_MASTER_SEED,
            "routing_priority_master": ROUTING_PRIORITY_MASTER_SEED,
            "validation_sentinel_master": VALIDATION_SENTINEL_MASTER_SEED,
            "routing_priority_version": ROUTING_PRIORITY_VERSION,
            "identity_scope": "data_generation_hash_only",
            "excluded_fixture": EXCLUDED_FIXTURE_SEEDS,
        },
        "pairs": {
            "count_per_graph": PAIRS_PER_GRAPH,
            "ordered": True,
            "without_replacement": True,
            "same_pairs_all_conditions": True,
        },
        "embeddings": {
            "float_type": ITERATION2_FLOAT_TYPE,
            "dimension": 2,
            "shared_apsp": True,
            "stable_node_order": "ascending_integer_id",
            "hydra_native": {
                "algorithm": "standard_hydra_spectral",
                "kappa": HYDRA_KAPPA,
                "curvature": HYDRA_CURVATURE,
                "centering": "hyperbolic_frechet_isometry",
                "radial_rescaling": False,
            },
            "mds_native": {
                "algorithm": "torgerson_gower_classical_mds",
                "centering": "euclidean_centroid",
                "radial_rescaling": False,
                "routing_methods": ("euclidean_greedy",),
                "euclidean_reference_is_unconditional": True,
            },
            "matched_radii": MATCHED_RADII,
            "matched_condition_ids": MATCHED_CONDITIONS,
            "matched_transform": "x_prime = target_radius*x/max_node_norm",
            "transform_applied_independently": True,
            "scaled_hydra_label": "sensitivity_transformation_not_native_hydra",
            "adaptive_greedy_embedding": "excluded_from_equivalence_claim",
            "complete_collapse": "error",
            "partial_coincidences": "record_without_jitter",
        },
        "routing": {
            "matched_methods": ROUTING_METHODS,
            "native_hydra_methods": ROUTING_METHODS,
            "native_mds_methods": ("euclidean_greedy",),
            "state_machine": "shared_production_core",
            "independent_oracle": "iteration2_oracle_v1",
            "tie_break": ROUTING_PRIORITY_VERSION,
            "tie_priority_inputs": (
                "data_generation_hash",
                "graph_id",
                "pair_index",
                "source",
                "destination",
                "current_node",
                "candidate_node",
            ),
            "tie_priority_excludes": (
                "metric",
                "embedding_family",
                "coordinate_radius",
                "routing_outcome",
                "results",
            ),
            "direct_destination_adjacency_precedence": True,
            "progress": "true_minimum_strict_before_numerical_tie_resolution",
            "ordinary_revisit": "abort_invariant_error_not_scientific_data",
            "failure_schema": FAILURE_SCHEMA_VERSION,
            "repair_specific_failure_applicability": (
                "not_applicable_for_ordinary_routing"
            ),
            "dijkstra_once_per_pair": True,
            "stretch_denominator": "same_unweighted_dijkstra_hops",
            "tolerance": {
                "roundoff_multiplier": ROUTING_ROUNDOFF_MULTIPLIER,
                "euclidean": "64*eps*coordinate_max_radius",
                "poincare": "64*eps*max(1,4*atanh(coordinate_max_radius))",
            },
            "high_precision_sentinels": {
                "decimal_digits": HIGH_PRECISION_DECIMAL_DIGITS,
                "pairs_per_graph": HIGH_PRECISION_SENTINEL_PAIRS_PER_GRAPH,
                "selection": "outcome_independent_blake2s",
                "full_pair_oracle_graphs": {
                    "models": GRAPH_MODELS,
                    "n": N_VALUES,
                    "m": FULL_ORACLE_M_VALUES,
                    "replicate_indices": FULL_ORACLE_REPLICATE_INDICES,
                    "pair_coverage": "every_frozen_pair",
                },
            },
        },
        "reproducibility": {
            "clean_committed_source_required": True,
            "dependency_fingerprint_required": True,
            "capacity_profile_required": True,
            "capacity_profile_schema": ITERATION2_CAPACITY_PROFILE_SCHEMA,
            "capacity_profile_scope": (
                "iteration2_runner_oracles_embeddings_checkpoints_analysis"
            ),
            "atomic_checkpoint": (
                "same_parent_temporary_directory_with_result_manifest_then_"
                "completion_marker_last_and_atomic_directory_rename"
            ),
            "checkpoint_files": (
                "result.json.gz",
                "checkpoint_manifest.json",
                "complete.json",
            ),
            "resume": (
                "validate_sizes_hashes_manifest_identity_row_counts_and_content"
            ),
            "iteration1_output_protection": (
                ITERATION1_RAW_DIRECTORY,
                ITERATION1_ANALYSIS_DIRECTORY,
            ),
        },
    }


def analysis_payload() -> dict[str, object]:
    return {
        "schema": ITERATION2_ANALYSIS_SCHEMA,
        "independent_unit": "graph",
        "route_rows_independent": False,
        "radii_independent": False,
        "absolute_success": "graph_level_then_aggregate",
        "poincare_minus_euclidean": "S_P-S_E",
        "repaired_minus_unrepaired": "S_R-S_P",
        "interaction": "(S_HP-S_HE)-(S_MP-S_ME)_at_same_radius",
        "report_models_separately": GRAPH_MODELS,
        "report_strata": "n_m_cells_and_equal_n_m_stratum_marginal",
        "marginal_weighting": "equal_n_m_strata_then_equal_graph",
        "bootstrap": {
            "master_seed": BOOTSTRAP_MASTER_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "unit": "whole_graph_within_model_n_m",
            "paired_draw_reuse": True,
            "interval": "noninterpolated_nearest_rank_percentile",
        },
        "equivalence": {
            "margin_percentage_points": EQUIVALENCE_MARGIN_PERCENTAGE_POINTS,
            "human_approved": EQUIVALENCE_MARGIN_APPROVED,
            "equivalent": "entire_95pct_CI_inside_closed_margin",
            "inconclusive": (
                "CI_not_inside_margin_and_not_wholly_outside_margin"
            ),
            "non_significance_is_equality": False,
        },
        "multiple_radius_procedure": MULTIPLE_RADIUS_PROCEDURE,
        "property_multiplicity_procedure": PROPERTY_MULTIPLICITY_PROCEDURE,
        "property_missingness": "preserve_na_with_reason_never_impute",
        "property_associations": {
            "family_id": PROPERTY_ASSOCIATION_FAMILY_ID,
            "properties": NETWORK_PROPERTIES,
            "outcomes": PROPERTY_ASSOCIATION_OUTCOMES,
            "coordinate_conditions": PROPERTY_ASSOCIATION_CONDITIONS,
            "multiplicity_families": PROPERTY_MULTIPLICITY_FAMILIES,
            "hypotheses_per_family": 54,
            "alpha": PROPERTY_ASSOCIATION_ALPHA,
            "family_scope": "six_properties_by_nine_conditions_within_family",
            "graph_is_independent_unit": True,
            "strata": PROPERTY_ASSOCIATION_STRATA,
            "primary_estimate": PROPERTY_ASSOCIATION_PRIMARY,
            "confidence_interval": PROPERTY_ASSOCIATION_CI,
            "inference_replicates": PROPERTY_ASSOCIATION_REPLICATES,
            "unadjusted_p_value": PROPERTY_ASSOCIATION_P_VALUE,
            "multiplicity": PROPERTY_MULTIPLICITY_PROCEDURE,
            "permutation_map_reuse": "one_shared_map_per_family_and_replicate",
            "permutation_statistic": "maximum_absolute_defined_test_statistic",
            "sensitivity": PROPERTY_ASSOCIATION_SENSITIVITY,
            "spline_claim": "none",
            "exploratory": PROPERTY_ASSOCIATION_EXPLORATORY,
            "causal_interpretation": False,
        },
        "conditional_estimands": {
            "applicable_case_rule": (
                "mean_over_graphs_where_estimand_is_defined"
            ),
            "undefined_graphs": "preserve_na_with_reason",
            "bootstrap": (
                "resample_all_graphs_then_drop_undefined_within_replicate"
            ),
            "minimum_defined_replicate_fraction": (
                MIN_DEFINED_BOOTSTRAP_FRACTION
            ),
            "marginal_missing_stratum": "report_na_do_not_drop_stratum",
        },
        "equivalence_endpoint_inclusivity": {
            "inside_closed_margin": True,
            "outside_requires_strictly_beyond_margin": True,
            "zero_exclusion_requires_strict_sign": True,
            "reported_fields": (
                "ci_excludes_zero",
                "ci_wholly_inside_margin",
                "ci_wholly_positive_beyond_margin",
                "ci_wholly_negative_beyond_margin",
                "practical_magnitude_unresolved",
            ),
            "embedding_neutrality_claim": False,
        },
        "stretch": {
            "method_specific": "condition_on_method_success",
            "common_success": "paired_euclidean_and_poincare",
            "recovered": "repair_success_after_ordinary_poincare_failure",
            "failure_penalty": "none",
        },
        "failure_uncertainty": "whole_graph_bootstrap",
        "distortion": (
            "separate_geometry_objectives_no_cross_geometry_pooling"
        ),
        "model_contrast": (
            "descriptive_independent_graph_resampling_within_n_m"
        ),
        "configuration_selection": "forbid_results_as_configuration_inputs",
    }


def output_schema_payload() -> dict[str, object]:
    reporting_source = Path(__file__).with_name("iteration2_reporting.py")
    normalized_reporting_source = reporting_source.read_text(
        encoding="utf-8"
    ).replace("\r\n", "\n").encode("utf-8")
    return {
        "schema": ITERATION2_OUTPUT_SCHEMA,
        "raw": {
            "graph_result_schema": ITERATION2_RESULT_SCHEMA,
            "layout": "manifest_plus_atomic_gzip_json_per_graph",
            "graph_file_count": ITERATION2_RAW_GRAPH_FILE_COUNT,
            "graph_checkpoint_directory_count": (
                ITERATION2_GRAPH_CHECKPOINT_DIRECTORY_COUNT
            ),
            "files_per_graph_checkpoint_directory": (
                ITERATION2_GRAPH_CHECKPOINT_FILES_PER_DIRECTORY
            ),
            "run_level_file_count": ITERATION2_RAW_RUN_LEVEL_FILE_COUNT,
            "total_file_count": ITERATION2_RAW_TOTAL_FILE_COUNT,
            "coordinate_precision": ITERATION2_FLOAT_TYPE,
            "route_record_count_per_pair": 28,
            "walk_includes_physical_repair_backtrack": True,
            "failure_applicability_field": (
                "initial_repair_final_stage_specific"
            ),
            "failure_diagnostics": (
                "stage_vertex_hop_degree_distances_gap_tolerance"
            ),
            "gauge_and_centering_diagnostics": True,
            "high_precision_oracle_diagnostics": True,
            "atomic_checkpoint_completion_marker_written_last": True,
            "atomic_checkpoint_publication": (
                "same_filesystem_directory_rename_from_same_parent"
            ),
            "nonfinite_values": "forbidden",
        },
        "analysis": {
            "analysis_schema": ITERATION2_ANALYSIS_SCHEMA,
            "machine_readable_csv_companions": True,
            "raw_routes_in_workbook": False,
            "equivalence_rows_require": (
                "estimate_pointwise_95pct_ci_bounds_classification"
            ),
            "simultaneous_radius_rows_require": (
                "estimate_simultaneous_95pct_ci_bounds_classification"
            ),
            "missingness": "explicit_status_applicability_and_na_reason",
            "separate_generation_and_analysis_provenance": True,
            "raw_validation_policy": "read_only_analysis_consumer",
            "raw_validation_execution_evidence": (
                "regeneration_requested",
                "scientific_graphs_executed_during_analysis",
                "dijkstra_executions_during_analysis",
                "routing_executions_during_analysis",
                "raw_checkpoints_written_during_analysis",
                "raw_tree_before",
                "raw_tree_after",
                "raw_tree_unchanged",
                "scientific_operation_ledger",
            ),
            "scientific_execution_during_analysis": "publication_fails_closed",
            "atomic_bundle_completion_marker": True,
        },
        "workbook": {
            "sheet_count": ITERATION2_WORKBOOK_SHEET_COUNT,
            "sheets": OUTPUT_WORKBOOK_SHEETS,
            "table_schema_hash": REPORTING_TABLE_SCHEMA_HASH,
            "table_schema_hash_encoding": (
                "canonical_json_v1_mapping_sheet_to_ordered_columns"
            ),
            "semantic_registry_schema": REPORTING_SEMANTIC_REGISTRY_SCHEMA,
            "semantic_registry_contract_sha256": sha256(
                normalized_reporting_source
            ).hexdigest(),
            "semantic_registry_contract_encoding": (
                "normalized_lf_utf8_complete_reporting_contract_source"
            ),
            "frozen_header_and_first_column": True,
            "formula_cells": 0,
            "hidden_outcome_dependent_filters": False,
            "excel_maximum_rows": OUTPUT_EXCEL_MAXIMUM_ROWS,
        },
        "figures": {
            "count": ITERATION2_FIGURE_COUNT,
            "files": OUTPUT_FIGURE_FILES,
            "specifications": OUTPUT_FIGURE_SPECIFICATIONS,
            "prespecified": True,
            "primary_and_repair_effects_separate": True,
            "all_required_rows_included_without_silent_truncation": True,
            "display_unit_rules": OUTPUT_DISPLAY_UNIT_RULES,
        },
    }


DATA_GENERATION_HASH = fingerprint(data_generation_payload())
ANALYSIS_PLAN_HASH = fingerprint(analysis_payload())
OUTPUT_SCHEMA_HASH = fingerprint(output_schema_payload())
COMBINED_PROTOCOL_HASH = fingerprint(
    {
        "schema": ITERATION2_PROTOCOL_SCHEMA,
        "data_generation_hash": DATA_GENERATION_HASH,
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "output_schema_hash": OUTPUT_SCHEMA_HASH,
    }
)
ITERATION2_RUN_IDENTITY = f"iteration2_raw_{DATA_GENERATION_HASH[:16]}"
ITERATION2_ANALYSIS_IDENTITY = (
    f"iteration2_analysis_{COMBINED_PROTOCOL_HASH[:16]}"
)
FULL_RUN_CONFIRMATION_TOKEN = COMBINED_PROTOCOL_HASH


@dataclass(frozen=True)
class GraphSpec:
    schedule_index: int
    model: str
    n: int
    m: int
    replicate_index: int

    @property
    def graph_id(self) -> str:
        prefix = "er" if self.model == "erdos_renyi" else "ba"
        return (
            f"i2_{prefix}_n{self.n:04d}_m{self.m:02d}_"
            f"rep{self.replicate_index:03d}"
        )

    @property
    def er_probability(self) -> float:
        return (
            2 * self.m * (self.n - self.m)
            / (self.n * (self.n - 1))
        )


@dataclass(frozen=True)
class GraphSeeds:
    graph: int
    embedding_provenance: int
    pairs: int
    routing_priority: int
    validation_sentinel: int
    er_attempts: tuple[int, ...]


def full_schedule() -> tuple[GraphSpec, ...]:
    result: list[GraphSpec] = []
    for model in GRAPH_MODELS:
        for n in N_VALUES:
            for m in M_VALUES:
                for replicate in range(GRAPH_REPETITIONS):
                    result.append(
                        GraphSpec(len(result), model, n, m, replicate)
                    )
    if (
        len(result) != ITERATION2_GRAPH_COUNT
        or len({item.graph_id for item in result}) != ITERATION2_GRAPH_COUNT
    ):
        raise RuntimeError("Iteration 2 schedule identity is invalid")
    return tuple(result)


def is_full_oracle_graph(spec: GraphSpec) -> bool:
    """Identify the prespecified sparse/dense ER/BA oracle census graphs."""

    if spec not in full_schedule():
        raise ValueError("graph specification is outside the frozen schedule")
    return (
        spec.replicate_index in FULL_ORACLE_REPLICATE_INDICES
        and spec.m in FULL_ORACLE_M_VALUES
    )


def _derive_seed(
    master: int,
    domain: str,
    identity: object,
    *,
    identity_hash: str,
    digest_size: int = 4,
) -> int:
    payload = {
        "identity_hash": identity_hash,
        "seed_identity_version": ITERATION2_SEED_IDENTITY_VERSION,
        "domain": domain,
        "master": master,
        "identity": identity,
    }
    return int.from_bytes(
        blake2s(
            canonical_json_bytes(payload),
            digest_size=digest_size,
            person=_SEED_PERSON,
        ).digest(),
        "big",
    )


def routing_priority_seed(graph_id: str) -> int:
    """Return the data-identity-only key seed for a graph's routing ties."""

    if not isinstance(graph_id, str) or not graph_id:
        raise ValueError("graph_id must be a non-empty string")
    return _derive_seed(
        ROUTING_PRIORITY_MASTER_SEED,
        "routing_priority",
        graph_id,
        identity_hash=DATA_GENERATION_HASH,
        digest_size=8,
    )


def validation_sentinel_seed(graph_id: str) -> int:
    """Return the data-identity-only validation-sentinel seed for a graph."""

    if not isinstance(graph_id, str) or not graph_id:
        raise ValueError("graph_id must be a non-empty string")
    return _derive_seed(
        VALIDATION_SENTINEL_MASTER_SEED,
        "validation_sentinel",
        graph_id,
        identity_hash=DATA_GENERATION_HASH,
        digest_size=8,
    )


def seeds_for_graph(spec: GraphSpec) -> GraphSeeds:
    if spec not in full_schedule():
        raise ValueError("graph specification is outside the frozen schedule")
    identity = (
        spec.model,
        spec.n,
        spec.m,
        spec.replicate_index,
        spec.graph_id,
    )
    attempts = (
        tuple(
            _derive_seed(
                GRAPH_MASTER_SEED,
                "er_attempt",
                (*identity, index),
                identity_hash=DATA_GENERATION_HASH,
            )
            for index in range(MAX_ER_ATTEMPTS)
        )
        if spec.model == "erdos_renyi"
        else ()
    )
    return GraphSeeds(
        graph=_derive_seed(
            GRAPH_MASTER_SEED,
            "graph",
            identity,
            identity_hash=DATA_GENERATION_HASH,
        ),
        embedding_provenance=_derive_seed(
            EMBEDDING_PROVENANCE_MASTER_SEED,
            "embedding",
            identity,
            identity_hash=DATA_GENERATION_HASH,
        ),
        pairs=_derive_seed(
            PAIR_MASTER_SEED,
            "pairs",
            identity,
            identity_hash=DATA_GENERATION_HASH,
        ),
        routing_priority=routing_priority_seed(spec.graph_id),
        validation_sentinel=validation_sentinel_seed(spec.graph_id),
        er_attempts=attempts,
    )


def _unbiased_indices(
    *,
    count: int,
    population: int,
    person: bytes,
    identity: object,
    identity_hash: str,
) -> tuple[int, ...]:
    if count < 0 or population <= 0 or count > population:
        raise ValueError("invalid deterministic sample size")
    limit = _UINT64_SPACE - (_UINT64_SPACE % population)
    selected: list[int] = []
    used: set[int] = set()
    counter = 0
    while len(selected) < count:
        word = int.from_bytes(
            blake2s(
                canonical_json_bytes(
                    {
                        "identity_hash": identity_hash,
                        "seed_identity_version": ITERATION2_SEED_IDENTITY_VERSION,
                        "identity": identity,
                        "counter": counter,
                    }
                ),
                digest_size=8,
                person=person,
            ).digest(),
            "big",
        )
        counter += 1
        if word >= limit:
            continue
        candidate = word % population
        if candidate not in used:
            selected.append(candidate)
            used.add(candidate)
    return tuple(selected)


@scientific_operation_boundary("pair_sampling")
def sample_ordered_pairs(
    nodes: Iterable[int],
    count: int,
    *,
    graph_id: str,
    pair_seed: int,
) -> tuple[tuple[int, int], ...]:
    ordered_nodes = tuple(nodes)
    n = len(ordered_nodes)
    if set(ordered_nodes) != set(range(n)) or n < 2:
        raise ValueError("nodes must equal integer IDs 0 through n-1")
    population = n * (n - 1)
    indices = _unbiased_indices(
        count=count,
        population=population,
        person=_PAIR_PERSON,
        identity=(graph_id, pair_seed, "ordered_pairs"),
        identity_hash=DATA_GENERATION_HASH,
    )
    pairs = []
    for index in indices:
        source = index // (n - 1)
        remainder = index % (n - 1)
        destination = remainder if remainder < source else remainder + 1
        pairs.append((source, destination))
    return tuple(pairs)


def sentinel_pair_indices(
    graph_id: str,
    pair_count: int,
) -> tuple[int, ...]:
    return _unbiased_indices(
        count=min(HIGH_PRECISION_SENTINEL_PAIRS_PER_GRAPH, pair_count),
        population=pair_count,
        person=_SENTINEL_PERSON,
        identity=(
            graph_id,
            validation_sentinel_seed(graph_id),
            "high_precision_routing_sentinel",
        ),
        identity_hash=DATA_GENERATION_HASH,
    )


def bootstrap_indices(
    *,
    model: str,
    n: int,
    m: int,
    replicate: int,
    graph_count: int = 20,
) -> tuple[int, ...]:
    if model not in GRAPH_MODELS or n not in N_VALUES or m not in M_VALUES:
        raise ValueError("bootstrap stratum is outside the frozen design")
    if not 0 <= replicate < BOOTSTRAP_REPLICATES or graph_count <= 0:
        raise ValueError("invalid bootstrap request")
    result = []
    for position in range(graph_count):
        result.append(
            _unbiased_indices(
                count=1,
                population=graph_count,
                person=_BOOTSTRAP_PERSON,
                identity=(
                    BOOTSTRAP_MASTER_SEED,
                    model,
                    n,
                    m,
                    replicate,
                    position,
                ),
                identity_hash=ANALYSIS_PLAN_HASH,
            )[0]
        )
    return tuple(result)


def property_random_seed(*identity: object) -> int:
    """Return a frozen domain-separated seed for property inference."""

    payload = {
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "seed_identity_version": ITERATION2_SEED_IDENTITY_VERSION,
        "master": BOOTSTRAP_MASTER_SEED,
        "family": PROPERTY_ASSOCIATION_FAMILY_ID,
        "identity": identity,
    }
    return int.from_bytes(
        blake2s(
            canonical_json_bytes(payload),
            digest_size=8,
            person=_PROPERTY_PERSON,
        ).digest(),
        "big",
    )


def resolve_iteration2_output(
    repository_root: Path | str,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    if name not in (
        ITERATION2_RUN_IDENTITY,
        ITERATION2_ANALYSIS_IDENTITY,
    ) and re.fullmatch(
        r"iteration2_excluded_(?:raw|analysis)_[0-9a-f]{16}", name
    ) is None:
        raise ValueError("output name is not a frozen Iteration 2 identity")
    target = (root / "results" / name).resolve()
    protected = {
        (root / "results" / ITERATION1_RAW_DIRECTORY).resolve(),
        (root / "results" / ITERATION1_ANALYSIS_DIRECTORY).resolve(),
    }
    if target in protected:
        raise RuntimeError("Iteration 1 output is immutable")
    return target


def audit_new_seed_uniqueness() -> tuple[int, ...]:
    values: list[int] = []
    for spec in full_schedule():
        seeds = seeds_for_graph(spec)
        values.extend(
            (
                seeds.graph,
                seeds.embedding_provenance,
                seeds.pairs,
                seeds.routing_priority,
                seeds.validation_sentinel,
            )
        )
        values.extend(seeds.er_attempts)
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return tuple(
        sorted(value for value, count in counts.items() if count > 1)
    )


if audit_new_seed_uniqueness():
    raise RuntimeError("Iteration 2 derived seed collision detected")
