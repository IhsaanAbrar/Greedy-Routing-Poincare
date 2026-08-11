"""Graph-level estimands, clustered intervals, and equivalence for Iteration 2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from hashlib import blake2s, sha256
import json
from math import ceil, isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np

from iteration2_config import (
    ANALYSIS_PLAN_HASH,
    BOOTSTRAP_REPLICATES,
    EQUIVALENCE_MARGIN_PERCENTAGE_POINTS,
    GRAPH_MODELS,
    MATCHED_RADII,
    MATCHED_RADIUS_LABELS,
    MIN_DEFINED_BOOTSTRAP_FRACTION,
    M_VALUES,
    N_VALUES,
    ROUTING_METHODS,
    bootstrap_indices,
)
from iteration2_routing import ALL_FAILURE_TYPES, REPAIR_FAILURE_TYPES


GraphRow = Mapping[str, object]
BootstrapProvider = Callable[..., tuple[int, ...]]


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


ROUTE_IDENTITY_FIELDS = (
    "graph_id",
    "pair_index",
    "source",
    "destination",
    "coordinate_condition_id",
    "method_id",
)
SCIENTIFIC_ORDINARY_FAILURES = ("local_minimum",)
SCIENTIFIC_REPAIR_FAILURES = (
    "repair_unavailable_at_source",
    "no_alternative_after_backtracking",
    "post_repair_local_minimum",
    "post_repair_attempted_revisit",
)
FORBIDDEN_INVARIANT_TERMINALS = (
    "attempted_revisit",
    "numerical_invariant_failure",
)
DISTANCE_BANDS = ("1", "2", "3", "4", ">=5")


def _condition_specifications() -> dict[str, tuple[str, str, float | None, tuple[str, ...]]]:
    specifications: dict[
        str, tuple[str, str, float | None, tuple[str, ...]]
    ] = {
        "hydra_native": ("hydra", "native_reference", None, tuple(ROUTING_METHODS)),
        "mds_native": ("mds", "native_reference", None, ("euclidean_greedy",)),
    }
    for family in ("hydra", "mds"):
        for radius, label in zip(MATCHED_RADII, MATCHED_RADIUS_LABELS, strict=True):
            specifications[f"{family}_scaled_{label}"] = (
                family,
                "matched_radius_sensitivity",
                float(radius),
                tuple(ROUTING_METHODS),
            )
    return specifications


def _required_integral(record: Mapping[str, object], field: str) -> int:
    if field not in record:
        raise ValueError(f"route record is missing identity field {field}")
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"route identity field {field} must be an integer")
    return int(value)


def _required_text(record: Mapping[str, object], field: str) -> str:
    if field not in record or not isinstance(record[field], str) or not record[field]:
        raise ValueError(f"route record is missing text identity field {field}")
    return str(record[field])


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise ValueError(f"{field} must be finite numeric data")
    return float(value)


def _distance_band(distance: int) -> str:
    if distance < 1:
        raise ValueError("Dijkstra distance must be positive")
    return str(distance) if distance <= 4 else ">=5"


def _ratio_row(
    numerator: int | float | None,
    denominator: int | None,
    *,
    applicability: str = "applicable",
    na_reason: str | None = None,
) -> dict[str, object]:
    if applicability == "not_applicable":
        return {
            "estimate": None,
            "numerator": numerator,
            "denominator": denominator,
            "graph_count": 1,
            "pair_count": 0,
            "status": "not_applicable",
            "applicability": "not_applicable",
            "na_reason": na_reason or "estimand_not_defined_for_condition",
        }
    if denominator is None or denominator == 0:
        return {
            "estimate": None,
            "numerator": numerator,
            "denominator": denominator,
            "graph_count": 1,
            "pair_count": 0 if denominator is None else denominator,
            "status": "undefined_zero_denominator",
            "applicability": "applicable",
            "na_reason": na_reason or "zero_denominator",
        }
    return {
        "estimate": float(numerator) / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "graph_count": 1,
        "pair_count": denominator,
        "status": "defined",
        "applicability": "applicable",
        "na_reason": None,
    }


def _index_route_records(
    route_records: Sequence[Mapping[str, object]],
    *,
    graph_id: str,
    pair_count: int,
) -> tuple[
    dict[tuple[int, str, str], Mapping[str, object]],
    dict[int, tuple[int, int]],
]:
    specifications = _condition_specifications()
    by_design: dict[tuple[int, str, str], Mapping[str, object]] = {}
    pair_identities: dict[int, tuple[int, int]] = {}
    dijkstra_by_pair: dict[int, int] = {}
    for record in route_records:
        record_graph = _required_text(record, "graph_id")
        pair_index = _required_integral(record, "pair_index")
        source = _required_integral(record, "source")
        destination = _required_integral(record, "destination")
        condition = _required_text(record, "coordinate_condition_id")
        method = _required_text(record, "method_id")
        if record_graph != graph_id:
            raise ValueError("route record graph_id disagrees with graph identity")
        if not 0 <= pair_index < pair_count or source == destination:
            raise ValueError("route pair identity is outside the frozen design")
        if condition not in specifications or method not in specifications[condition][3]:
            raise ValueError("route condition/method identity is outside the design")
        previous_pair = pair_identities.setdefault(pair_index, (source, destination))
        if previous_pair != (source, destination):
            raise ValueError("source/destination are inconsistent for one pair index")
        identity = (pair_index, condition, method)
        if identity in by_design:
            raise ValueError("duplicate route identity")
        by_design[identity] = record
        success = _require_bool(record.get("success"), "success")
        dijkstra = _required_integral(record, "dijkstra_length")
        if dijkstra < 1:
            raise ValueError("Dijkstra distance must be positive")
        previous_dijkstra = dijkstra_by_pair.setdefault(pair_index, dijkstra)
        if previous_dijkstra != dijkstra:
            raise ValueError("Dijkstra distance is inconsistent across route conditions")
        for failure_field in ("initial_failure_type", "final_failure_type"):
            if record.get(failure_field) in FORBIDDEN_INVARIANT_TERMINALS:
                raise ValueError("routing invariant failures cannot enter scientific analysis")
        walk = record.get("walk")
        if not isinstance(walk, (list, tuple)) or not walk:
            raise ValueError("route walk must be a non-empty sequence")
        if int(walk[0]) != source or (success and int(walk[-1]) != destination):
            raise ValueError("route walk disagrees with source/destination outcome")
        physical = record.get("physical_hops", record.get("route_length"))
        if physical is None or _required_integral({"physical": physical}, "physical") != len(walk) - 1:
            raise ValueError("physical hop count must equal complete walk length minus one")
        stretch = record.get("stretch")
        if success:
            observed = _finite_number(stretch, "stretch")
            expected = int(physical) / dijkstra
            if not np.isclose(observed, expected, rtol=1e-12, atol=1e-12):
                raise ValueError("successful-route stretch disagrees with physical hops")
        elif stretch is not None:
            raise ValueError("failed routes must not receive numerical stretch")
    expected_pairs = set(range(pair_count))
    if set(pair_identities) != expected_pairs:
        raise ValueError("route records do not cover the exact sampled pair indices")
    expected = {
        (pair_index, condition, method)
        for pair_index in range(pair_count)
        for condition, (_, _, _, methods) in specifications.items()
        for method in methods
    }
    if set(by_design) != expected:
        raise ValueError("route records do not contain the exact 28-record crossed design")
    return by_design, pair_identities


def graph_level_rows(
    route_records: Sequence[Mapping[str, object]],
    *,
    graph_id: str,
    model: str,
    n: int,
    m: int,
    replicate_index: int,
    pair_count: int,
) -> list[dict[str, object]]:
    """Calculate order-invariant graph estimands from explicit route identities."""

    if (
        not isinstance(graph_id, str)
        or not graph_id
        or model not in GRAPH_MODELS
        or n <= 0
        or m <= 0
        or pair_count <= 0
    ):
        raise ValueError("invalid graph identity")
    condition_specs = _condition_specifications()
    indexed, pair_identities = _index_route_records(
        route_records, graph_id=graph_id, pair_count=pair_count
    )

    rows: list[dict[str, object]] = []
    for condition, (family, kind, radius, methods) in condition_specs.items():
        records_by_method = {
            method: [indexed[(pair, condition, method)] for pair in range(pair_count)]
            for method in methods
        }
        rates = {
            method: sum(_require_bool(row["success"], "success") for row in records)
            / pair_count
            for method, records in records_by_method.items()
        }
        success_counts = {
            method: sum(_require_bool(row["success"], "success") for row in records)
            for method in methods
            for records in (records_by_method[method],)
        }
        euclidean = rates["euclidean_greedy"]
        poincare = rates.get("poincare_greedy")
        repaired = rates.get("repaired_poincare_greedy")
        repair_estimands = _graph_repair_estimands(
            records_by_method, pair_count=pair_count
        )
        if poincare is not None and repaired is not None:
            for pair_index in range(pair_count):
                p_ok = records_by_method["poincare_greedy"][pair_index]["success"]
                r_ok = records_by_method["repaired_poincare_greedy"][pair_index]["success"]
                if p_ok and not r_ok:
                    raise ValueError("repaired Poincare failed after ordinary success")
                repaired_row = records_by_method["repaired_poincare_greedy"][pair_index]
                if p_ok and (
                    repaired_row.get("repair_attempted") is True
                    or repaired_row.get("repair_attempt_count", 0) != 0
                ):
                    raise ValueError("repair was invented after ordinary success")
        stretch_summaries = _graph_stretch_summaries(records_by_method)
        common = stretch_summaries.get("common_success", {})
        recovered = stretch_summaries.get("newly_recovered", {})
        rows.append(
            {
                "graph_id": graph_id,
                "model": model,
                "n": n,
                "m": m,
                "replicate_index": replicate_index,
                "coordinate_condition_id": condition,
                "embedding_family": family,
                "condition_kind": kind,
                "matched_radius": radius,
                "pair_count": pair_count,
                "pair_identity_count": len(pair_identities),
                "independent_unit": "graph",
                "radius_is_independent_replicate": False,
                "euclidean_success_count": success_counts["euclidean_greedy"],
                "euclidean_success": euclidean,
                "poincare_success_count": success_counts.get("poincare_greedy"),
                "poincare_success": poincare,
                "repaired_poincare_success_count": success_counts.get(
                    "repaired_poincare_greedy"
                ),
                "repaired_poincare_success": repaired,
                "poincare_minus_euclidean": (
                    None if poincare is None else poincare - euclidean
                ),
                "repaired_minus_unrepaired_poincare": (
                    None
                    if repaired is None or poincare is None
                    else repaired - poincare
                ),
                "conditional_repair_recovery": repair_estimands[
                    "recovered_over_all_ordinary_failures"
                ]["estimate"],
                "conditional_repair_recovery_status": repair_estimands[
                    "recovered_over_all_ordinary_failures"
                ]["status"],
                "conditional_repair_recovery_applicability": repair_estimands[
                    "recovered_over_all_ordinary_failures"
                ]["applicability"],
                "conditional_repair_recovery_na_reason": repair_estimands[
                    "recovered_over_all_ordinary_failures"
                ]["na_reason"],
                "repair_recovery_numerator": repair_estimands[
                    "recovered_over_all_ordinary_failures"
                ]["numerator"],
                "repair_recovery_denominator": repair_estimands[
                    "recovered_over_all_ordinary_failures"
                ]["denominator"],
                "repair_estimands": repair_estimands,
                "failure_summaries": _graph_failure_summaries(
                    records_by_method,
                    pair_count=pair_count,
                ),
                "stretch_summaries": stretch_summaries,
                "common_success_pair_count": common.get("denominator"),
                "common_success_poincare_minus_euclidean_stretch": common.get(
                    "paired_difference_mean"
                ),
                "common_success_stretch_status": common.get("status"),
                "common_success_stretch_na_reason": common.get("na_reason"),
                "recovered_pair_count": recovered.get("denominator"),
                "recovered_route_stretch": recovered.get("repaired_mean"),
                "physical_recovered_route_stretch": recovered.get(
                    "physical_repaired_mean"
                ),
                "distance_band_summaries": _graph_distance_band_summaries(
                    records_by_method
                ),
                "resource_summaries": _graph_resource_summaries(records_by_method),
                "poincare_status": (
                    "not_applicable_method_not_in_design"
                    if condition == "mds_native"
                    else "defined"
                ),
            }
        )
    validate_graph_level_rows(rows, pair_count=pair_count)
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _graph_repair_estimands(
    records: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    pair_count: int,
) -> dict[str, dict[str, object]]:
    if "poincare_greedy" not in records:
        names = (
            "overall_repaired_minus_ordinary_success",
            "recovered_over_all_ordinary_failures",
            "backtrackable_over_all_ordinary_failures",
            "alternative_available_over_backtrackable_failures",
            "recovered_over_repair_eligible_failures",
            "recovered_over_selected_alternatives",
        )
        return {
            name: _ratio_row(
                None,
                None,
                applicability="not_applicable",
                na_reason="coordinate_condition_has_no_poincare_method",
            )
            for name in names
        }
    ordinary = records["poincare_greedy"]
    repaired = records["repaired_poincare_greedy"]
    failures = [index for index, row in enumerate(ordinary) if row["success"] is False]
    backtrackable = [
        index
        for index in failures
        if bool(repaired[index].get("repair_backtrackable", len(ordinary[index]["walk"]) > 1))
    ]
    alternative_available = [
        index
        for index in backtrackable
        if repaired[index].get("repair_alternative_existed") is True
    ]
    eligible = [
        index
        for index in failures
        if bool(
            repaired[index].get(
                "repair_eligible",
                index in backtrackable and index in alternative_available,
            )
        )
    ]
    selected = [
        index
        for index in eligible
        if bool(
            repaired[index].get(
                "repair_alternative_selected",
                repaired[index].get("repair_selected_alternative") is not None,
            )
        )
    ]
    recovered = [
        index for index in failures if repaired[index].get("success") is True
    ]
    if any(index not in selected for index in recovered):
        raise ValueError("a recovered route lacks a selected repair alternative")
    return {
        "overall_repaired_minus_ordinary_success": _ratio_row(
            len(recovered), pair_count
        ),
        "recovered_over_all_ordinary_failures": _ratio_row(
            len(recovered),
            len(failures),
            na_reason="no_ordinary_poincare_failures",
        ),
        "backtrackable_over_all_ordinary_failures": _ratio_row(
            len(backtrackable),
            len(failures),
            na_reason="no_ordinary_poincare_failures",
        ),
        "alternative_available_over_backtrackable_failures": _ratio_row(
            len(alternative_available),
            len(backtrackable),
            na_reason="no_backtrackable_ordinary_failures",
        ),
        "recovered_over_repair_eligible_failures": _ratio_row(
            len(recovered),
            len(eligible),
            na_reason="no_repair_eligible_failures",
        ),
        "recovered_over_selected_alternatives": _ratio_row(
            len(recovered),
            len(selected),
            na_reason="no_selected_repair_alternatives",
        ),
    }


def _graph_failure_summaries(
    records: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    pair_count: int,
) -> dict[str, dict[str, dict[str, object]]]:
    output: dict[str, dict[str, dict[str, object]]] = {}
    for method, rows in records.items():
        final_failure_denominator = sum(
            row.get("success") is False for row in rows
        )
        initial_failure_denominator = sum(
            row.get("initial_failure_type") is not None for row in rows
        )
        categories: dict[str, dict[str, object]] = {}
        for failure in (*SCIENTIFIC_ORDINARY_FAILURES, *SCIENTIFIC_REPAIR_FAILURES):
            is_repair_method = method == "repaired_poincare_greedy"
            is_repair_failure = failure in SCIENTIFIC_REPAIR_FAILURES
            initial_applicable = failure in SCIENTIFIC_ORDINARY_FAILURES
            final_applicable = (
                is_repair_method
                if is_repair_failure
                else not is_repair_method
            )
            count = (
                sum(row.get("final_failure_type") == failure for row in rows)
                if final_applicable
                else None
            )
            initial_count = (
                sum(row.get("initial_failure_type") == failure for row in rows)
                if initial_applicable
                else None
            )
            categories[failure] = {
                "status": (
                    "observed_or_zero"
                    if final_applicable
                    else "not_applicable"
                ),
                "applicability": (
                    "applicable" if final_applicable else "not_applicable"
                ),
                "na_reason": (
                    None
                    if final_applicable
                    else "failure_category_not_defined_at_final_stage"
                ),
                "count": count,
                "numerator": count,
                "rate_all_pairs": (
                    None if count is None else count / pair_count
                ),
                "denominator": pair_count if count is not None else None,
                "graph_count": 1,
                "pair_count": pair_count,
                "initial_status": (
                    "observed_or_zero"
                    if initial_applicable
                    else "not_applicable"
                ),
                "initial_applicability": (
                    "applicable" if initial_applicable else "not_applicable"
                ),
                "initial_na_reason": (
                    None
                    if initial_applicable
                    else "repair_failure_not_defined_at_initial_stage"
                ),
                "initial_count": initial_count,
                "initial_rate_all_pairs": (
                    None
                    if initial_count is None
                    else initial_count / pair_count
                ),
                "initial_conditional_failure_composition": (
                    None
                    if initial_count is None
                    or initial_failure_denominator == 0
                    else initial_count / initial_failure_denominator
                ),
                "initial_conditional_failure_denominator": (
                    None
                    if initial_count is None
                    else initial_failure_denominator
                ),
                "initial_conditional_failure_status": (
                    "not_applicable"
                    if initial_count is None
                    else "undefined_zero_denominator"
                    if initial_failure_denominator == 0
                    else "defined"
                ),
                "initial_conditional_failure_na_reason": (
                    "failure_category_not_defined_at_initial_stage"
                    if initial_count is None
                    else "method_has_no_initial_failures"
                    if initial_failure_denominator == 0
                    else None
                ),
                "conditional_failure_composition": (
                    None
                    if count is None or final_failure_denominator == 0
                    else count / final_failure_denominator
                ),
                "conditional_failure_denominator": (
                    None if count is None else final_failure_denominator
                ),
                "conditional_failure_status": (
                    "not_applicable"
                    if count is None
                    else "undefined_zero_denominator"
                    if final_failure_denominator == 0
                    else "defined"
                ),
                "conditional_failure_na_reason": (
                    "failure_category_not_defined_at_final_stage"
                    if count is None
                    else "method_has_no_final_failures"
                    if final_failure_denominator == 0
                    else None
                ),
                "repair_status": (
                    "applicable"
                    if is_repair_method
                    else "not_applicable"
                ),
                "repair_applicability": (
                    "applicable"
                    if is_repair_method
                    else "not_applicable"
                ),
                "repair_na_reason": (
                    None
                    if is_repair_method
                    else "routing_method_has_no_repair_stage"
                ),
            }
        output[method] = categories
    return output


def _graph_stretch_summaries(
    records: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    by_method = dict(records)
    output: dict[str, object] = {}
    for method, rows in by_method.items():
        values = [
            float(row["stretch"])
            for row in rows
            if row.get("success") is True
        ]
        output[method] = {
            "success_count": len(values),
            "numerator": float(sum(values)) if values else 0.0,
            "denominator": len(values),
            "graph_count": 1,
            "pair_count": len(rows),
            "success_conditioned_mean": _mean(values),
            "status": "defined" if values else "undefined_zero_denominator",
            "applicability": "applicable",
            "na_reason": None if values else "no_successful_routes",
        }
    if "poincare_greedy" not in by_method:
        return output
    euclidean = by_method["euclidean_greedy"]
    poincare = by_method["poincare_greedy"]
    repaired = by_method["repaired_poincare_greedy"]
    common_indices = [index for index in range(len(euclidean)) if euclidean[index]["success"] and poincare[index]["success"]]
    recovered_indices = [
        index
        for index, (p_row, r_row) in enumerate(
            zip(poincare, repaired, strict=True)
        )
        if not p_row["success"] and r_row["success"]
    ]
    common_e = [float(euclidean[index]["stretch"]) for index in common_indices]
    common_p = [float(poincare[index]["stretch"]) for index in common_indices]
    recovered_forwarding = [
        (
            int(repaired[index].get("physical_hops", repaired[index]["route_length"]))
            - 1
        )
        / int(repaired[index]["dijkstra_length"])
        for index in recovered_indices
    ]
    physical_recovered = [
        float(
            repaired[index].get(
                "physical_stretch",
                int(repaired[index].get("physical_hops", repaired[index]["route_length"]))
                / int(repaired[index]["dijkstra_length"]),
            )
        )
        for index in recovered_indices
    ]
    paired_differences = [
        p_value - e_value for e_value, p_value in zip(common_e, common_p, strict=True)
    ]
    output["common_success"] = {
        "pair_count": len(common_indices),
        "graph_count": 1,
        "numerator": float(sum(paired_differences)),
        "denominator": len(common_indices),
        "euclidean_mean": _mean(common_e),
        "poincare_mean": _mean(common_p),
        "paired_difference_mean": _mean(paired_differences),
        "status": "defined" if common_indices else "undefined_zero_denominator",
        "applicability": "applicable",
        "na_reason": None if common_indices else "no_pairs_delivered_by_both_ordinary_methods",
    }
    output["newly_recovered"] = {
        "pair_count": len(recovered_indices),
        "graph_count": 1,
        "numerator": float(sum(recovered_forwarding)),
        "denominator": len(recovered_indices),
        "repaired_mean": _mean(recovered_forwarding),
        "recovered_forwarding_stretch_mean": _mean(recovered_forwarding),
        "physical_repaired_mean": _mean(physical_recovered),
        "physical_repaired_stretch_mean": _mean(physical_recovered),
        "forwarding_stretch_definition": (
            "(physical_hops-one_physical_backtrack)/dijkstra_hops"
        ),
        "physical_stretch_definition": "physical_hops/dijkstra_hops",
        "unit": "hop_stretch_ratio",
        "conditioning": "newly_recovered_after_ordinary_poincare_failure",
        "status": "defined" if recovered_indices else "undefined_zero_denominator",
        "applicability": "applicable",
        "na_reason": None if recovered_indices else "no_newly_recovered_routes",
    }
    return output


def _graph_distance_band_summaries(
    records: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, dict[str, dict[str, object]]]:
    output: dict[str, dict[str, dict[str, object]]] = {}
    for method, rows in records.items():
        method_rows: dict[str, dict[str, object]] = {}
        for band in DISTANCE_BANDS:
            selected = [row for row in rows if _distance_band(int(row["dijkstra_length"])) == band]
            success = [row for row in selected if row["success"] is True]
            denominator = len(selected)
            method_rows[band] = {
                "distance_band": band,
                "numerator": len(success),
                "denominator": denominator,
                "graph_count": 1,
                "pair_count": denominator,
                "success_rate": None if denominator == 0 else len(success) / denominator,
                "successful_stretch_mean": _mean([float(row["stretch"]) for row in success]),
                "status": "defined" if denominator else "undefined_zero_denominator",
                "applicability": "applicable",
                "na_reason": None if denominator else "no_sampled_pairs_in_distance_band",
            }
        output[method] = method_rows
    return output


def _graph_resource_summaries(
    records: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for method, rows in records.items():
        fields = {
            "physical_hops": [
                int(row.get("physical_hops", row["route_length"])) for row in rows
            ],
            "forwarding_decisions": [int(row["forwarding_decisions"]) for row in rows],
            "logical_distance_evaluations": [
                row.get("logical_distance_evaluations") for row in rows
            ],
            "peak_history_vertices": [row.get("peak_history_vertices") for row in rows],
        }
        summary: dict[str, object] = {
            "graph_count": 1,
            "pair_count": len(rows),
            "applicability": "applicable",
        }
        missing = []
        for field, raw in fields.items():
            if any(value is None for value in raw):
                summary[f"{field}_sum"] = None
                summary[f"{field}_mean"] = None
                missing.append(field)
            else:
                values = [int(value) for value in raw]
                if any(value < 0 for value in values):
                    raise ValueError(f"{field} cannot be negative")
                summary[f"{field}_sum"] = sum(values)
                summary[f"{field}_mean"] = float(np.mean(values))
                if field == "peak_history_vertices":
                    summary[f"{field}_maximum"] = max(values)
        summary["status"] = "defined" if not missing else "incomplete_source_fields"
        summary["na_reason"] = None if not missing else "missing_route_resource_fields:" + ",".join(missing)
        output[method] = summary
    return output


def validate_graph_level_rows(
    rows: Sequence[GraphRow],
    *,
    pair_count: int,
) -> None:
    expected_conditions = set(_condition_specifications())
    if len(rows) != 10 or {
        str(row.get("coordinate_condition_id")) for row in rows
    } != expected_conditions:
        raise ValueError("graph-level rows must contain ten coordinate records")
    identity = {
        (
            row.get("graph_id"),
            row.get("model"),
            row.get("n"),
            row.get("m"),
            row.get("replicate_index"),
        )
        for row in rows
    }
    if len(identity) != 1:
        raise ValueError("graph-level rows mix graph identities")
    for row in rows:
        if (
            row.get("independent_unit") != "graph"
            or row.get("radius_is_independent_replicate") is not False
            or row.get("pair_count") != pair_count
        ):
            raise ValueError("graph-level unit metadata is invalid")
        for metric in (
            "euclidean_success",
            "poincare_success",
            "repaired_poincare_success",
            "poincare_minus_euclidean",
            "repaired_minus_unrepaired_poincare",
        ):
            value = row.get(metric)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{metric} is non-finite")
        if row["coordinate_condition_id"] == "mds_native":
            if any(
                row.get(metric) is not None
                for metric in (
                    "poincare_success",
                    "repaired_poincare_success",
                    "poincare_minus_euclidean",
                    "repaired_minus_unrepaired_poincare",
                )
            ):
                raise ValueError("native MDS Poincare values must be undefined")
        for nested_name in (
            "repair_estimands",
            "failure_summaries",
            "stretch_summaries",
            "distance_band_summaries",
            "resource_summaries",
        ):
            if not isinstance(row.get(nested_name), Mapping):
                raise ValueError(f"graph-level row lacks {nested_name}")


def graph_level_interactions(
    rows: Sequence[GraphRow],
) -> list[dict[str, object]]:
    """Return I(r)=[S_HP-S_HE]-[S_MP-S_ME] for each graph and radius."""

    if not rows:
        raise ValueError("interaction requires graph-level rows")
    validate_graph_level_rows(rows, pair_count=int(rows[0]["pair_count"]))
    by_condition = {str(row["coordinate_condition_id"]): row for row in rows}
    if len(by_condition) != 10:
        raise ValueError("interaction requires one complete graph-level design")
    result = []
    for radius, label in zip(
        MATCHED_RADII,
        MATCHED_RADIUS_LABELS,
        strict=True,
    ):
        hydra = by_condition[f"hydra_scaled_{label}"]
        mds = by_condition[f"mds_scaled_{label}"]
        hydra_contrast = float(hydra["poincare_minus_euclidean"])
        mds_contrast = float(mds["poincare_minus_euclidean"])
        result.append(
            {
                "graph_id": hydra["graph_id"],
                "model": hydra["model"],
                "n": hydra["n"],
                "m": hydra["m"],
                "replicate_index": hydra["replicate_index"],
                "matched_radius": radius,
                "hydra_poincare_minus_euclidean": hydra_contrast,
                "mds_poincare_minus_euclidean": mds_contrast,
                "interaction": hydra_contrast - mds_contrast,
                "pair_count": hydra["pair_count"],
                "unit": "proportion",
                "independent_unit": "graph",
            }
        )
    return result


def percentile_interval(
    estimates: Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(estimates, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or confidence_level != 0.95
    ):
        raise ValueError("interval requires finite estimates and frozen 95% level")
    ordered = np.sort(values, kind="stable")
    lower_rank = ceil(0.025 * len(ordered))
    upper_rank = ceil(0.975 * len(ordered))
    return (
        float(ordered[lower_rank - 1]),
        float(ordered[upper_rank - 1]),
    )


def equivalence_status_fields(
    ci_lower_percentage_points: float,
    ci_upper_percentage_points: float,
    *,
    margin_percentage_points: float = (
        EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
    ),
) -> dict[str, bool]:
    """Return the five frozen, non-overclaiming interval-status fields."""

    lower = float(ci_lower_percentage_points)
    upper = float(ci_upper_percentage_points)
    margin = float(margin_percentage_points)
    if (
        not all(isfinite(value) for value in (lower, upper, margin))
        or lower > upper
        or margin <= 0.0
    ):
        raise ValueError("invalid confidence interval or equivalence margin")
    inside = lower >= -margin and upper <= margin
    positive = lower > margin
    negative = upper < -margin
    return {
        "ci_excludes_zero": lower > 0.0 or upper < 0.0,
        "ci_wholly_inside_margin": inside,
        "ci_wholly_positive_beyond_margin": positive,
        "ci_wholly_negative_beyond_margin": negative,
        "practical_magnitude_unresolved": not (inside or positive or negative),
    }


def equivalence_classification(
    ci_lower_percentage_points: float,
    ci_upper_percentage_points: float,
    *,
    margin_percentage_points: float = EQUIVALENCE_MARGIN_PERCENTAGE_POINTS,
) -> str:
    """Compatibility label derived from the frozen separate status fields."""

    fields = equivalence_status_fields(
        ci_lower_percentage_points,
        ci_upper_percentage_points,
        margin_percentage_points=margin_percentage_points,
    )
    for name in (
        "ci_wholly_inside_margin",
        "ci_wholly_positive_beyond_margin",
        "ci_wholly_negative_beyond_margin",
    ):
        if fields[name]:
            return name
    return "practical_magnitude_unresolved"


def _bootstrap_matrix(
    *,
    model: str,
    n: int,
    m: int,
    graph_count: int,
    replicates: int,
    provider: BootstrapProvider,
) -> NDArray[np.int64]:
    matrix = np.empty((replicates, graph_count), dtype=np.int64)
    for replicate in range(replicates):
        draw = provider(
            model=model,
            n=n,
            m=m,
            replicate=replicate,
            graph_count=graph_count,
        )
        if len(draw) != graph_count:
            raise ValueError("bootstrap provider returned the wrong draw size")
        numeric = np.asarray(draw, dtype=np.int64)
        if np.any(numeric < 0) or np.any(numeric >= graph_count):
            raise ValueError("bootstrap provider returned an out-of-range index")
        matrix[replicate] = numeric
    return matrix


def aggregate_interactions(
    interaction_rows: Sequence[GraphRow],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_provider: BootstrapProvider = bootstrap_indices,
) -> list[dict[str, object]]:
    """Report model-specific n/m cells and equal-stratum marginals."""

    if not 2 <= bootstrap_replicates <= BOOTSTRAP_REPLICATES:
        raise ValueError("invalid bootstrap replicate count")
    grouped: dict[
        tuple[str, int, int, float],
        list[GraphRow],
    ] = defaultdict(list)
    for row in interaction_rows:
        grouped[
            (
                str(row["model"]),
                int(row["n"]),
                int(row["m"]),
                float(row["matched_radius"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for model in GRAPH_MODELS:
        cell_points: dict[tuple[int, int, float], float] = {}
        cell_boots: dict[
            tuple[int, int, float], NDArray[np.float64]
        ] = {}
        marginal_points: dict[float, list[float]] = {
            radius: [] for radius in MATCHED_RADII
        }
        marginal_boots: dict[float, list[NDArray[np.float64]]] = {
            radius: [] for radius in MATCHED_RADII
        }
        for n in N_VALUES:
            for m in M_VALUES:
                first_key = (model, n, m, MATCHED_RADII[0])
                graph_count = len(grouped.get(first_key, ()))
                if graph_count == 0:
                    raise ValueError("interaction stratum is missing")
                draw = _bootstrap_matrix(
                    model=model,
                    n=n,
                    m=m,
                    graph_count=graph_count,
                    replicates=bootstrap_replicates,
                    provider=bootstrap_provider,
                )
                expected_graph_ids: tuple[str, ...] | None = None
                for radius in MATCHED_RADII:
                    key = (model, n, m, radius)
                    selected = sorted(
                        grouped.get(key, ()),
                        key=lambda row: int(row["replicate_index"]),
                    )
                    if len(selected) != graph_count:
                        raise ValueError("radius rows are not paired within stratum")
                    current_graph_ids = tuple(str(row["graph_id"]) for row in selected)
                    if len(set(current_graph_ids)) != graph_count:
                        raise ValueError("duplicate graph identity in interaction stratum")
                    if expected_graph_ids is None:
                        expected_graph_ids = current_graph_ids
                    elif current_graph_ids != expected_graph_ids:
                        raise ValueError("interaction radii are not paired by graph identity")
                    values = np.asarray(
                        [float(row["interaction"]) for row in selected],
                        dtype=np.float64,
                    )
                    pair_count = sum(
                        int(row.get("pair_count", 0)) for row in selected
                    )
                    boot = values[draw].mean(axis=1)
                    lower, upper = percentile_interval(boot)
                    point_pp = float(values.mean()) * 100.0
                    lower_pp, upper_pp = lower * 100.0, upper * 100.0
                    output.append(
                        {
                            "scope": "model_n_m_radius_cell",
                            "model": model,
                            "n": n,
                            "m": m,
                            "matched_radius": radius,
                            "interaction_definition": (
                                "(Poincare-Euclidean)_Hydra-"
                                "(Poincare-Euclidean)_MDS"
                            ),
                            "estimate": point_pp,
                            "ci_lower": lower_pp,
                            "ci_upper": upper_pp,
                            "ci_type": "pointwise_95_percent",
                            "unit": "percentage_points",
                            "graph_count": graph_count,
                            "pair_count": pair_count,
                            "numerator": None,
                            "denominator": None,
                            "status": "defined",
                            "applicability": "applicable",
                            "na_reason": None,
                            "stratum_count": 1,
                            "weighting": "equal_graph",
                            "bootstrap_unit": "whole_graph",
                            "bootstrap_replicates": bootstrap_replicates,
                            "equivalence_margin_lower": (
                                -EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
                            ),
                            "equivalence_margin_upper": (
                                EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
                            ),
                            **equivalence_status_fields(lower_pp, upper_pp),
                            "equivalence_classification": equivalence_classification(
                                lower_pp, upper_pp
                            ),
                            "simultaneous_result_applicability": (
                                "reported_in_simultaneous_radius_rows"
                            ),
                        }
                    )
                    marginal_points[radius].append(float(values.mean()))
                    marginal_boots[radius].append(boot)
                    cell_points[(n, m, radius)] = float(values.mean())
                    cell_boots[(n, m, radius)] = boot
        for radius in MATCHED_RADII:
            if len(marginal_points[radius]) != 9:
                raise ValueError("marginal interaction requires nine strata")
            boot = np.stack(marginal_boots[radius]).mean(axis=0)
            lower, upper = percentile_interval(boot)
            point_pp = float(np.mean(marginal_points[radius])) * 100.0
            lower_pp, upper_pp = lower * 100.0, upper * 100.0
            output.append(
                {
                    "scope": "model_radius_n_m_marginal",
                    "model": model,
                    "n": None,
                    "m": None,
                    "matched_radius": radius,
                    "interaction_definition": (
                        "(Poincare-Euclidean)_Hydra-"
                        "(Poincare-Euclidean)_MDS"
                    ),
                    "estimate": point_pp,
                    "ci_lower": lower_pp,
                    "ci_upper": upper_pp,
                    "ci_type": "pointwise_95_percent",
                    "unit": "percentage_points",
                    "graph_count": sum(
                        len(grouped[(model, n, m, radius)])
                        for n in N_VALUES
                        for m in M_VALUES
                    ),
                    "pair_count": sum(
                        int(row.get("pair_count", 0))
                        for n in N_VALUES
                        for m in M_VALUES
                        for row in grouped[(model, n, m, radius)]
                    ),
                    "numerator": None,
                    "denominator": None,
                    "status": "defined",
                    "applicability": "applicable",
                    "na_reason": None,
                    "stratum_count": 9,
                    "weighting": "equal_n_m_strata_then_equal_graph",
                    "bootstrap_unit": "whole_graph_within_n_m",
                    "bootstrap_replicates": bootstrap_replicates,
                    "equivalence_margin_lower": (
                        -EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
                    ),
                    "equivalence_margin_upper": (
                        EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
                    ),
                    **equivalence_status_fields(lower_pp, upper_pp),
                    "equivalence_classification": equivalence_classification(
                        lower_pp, upper_pp
                    ),
                    "simultaneous_result_applicability": (
                        "reported_in_simultaneous_radius_rows"
                    ),
                }
            )
        for radius in MATCHED_RADII:
            for fixed_name, fixed_values, varying_values in (
                ("n", N_VALUES, M_VALUES),
                ("m", M_VALUES, N_VALUES),
            ):
                for fixed in fixed_values:
                    keys = (
                        [(fixed, varying, radius) for varying in varying_values]
                        if fixed_name == "n"
                        else [(varying, fixed, radius) for varying in varying_values]
                    )
                    point = float(np.mean([cell_points[key] for key in keys]))
                    boot = np.stack([cell_boots[key] for key in keys]).mean(axis=0)
                    lower, upper = percentile_interval(boot)
                    lower_pp, upper_pp = lower * 100.0, upper * 100.0
                    output.append(
                        {
                            "scope": f"model_{fixed_name}_radius_marginal",
                            "model": model,
                            "n": fixed if fixed_name == "n" else None,
                            "m": fixed if fixed_name == "m" else None,
                            "matched_radius": radius,
                            "interaction_definition": (
                                "(Poincare-Euclidean)_Hydra-"
                                "(Poincare-Euclidean)_MDS"
                            ),
                            "estimate": point * 100.0,
                            "ci_lower": lower_pp,
                            "ci_upper": upper_pp,
                            "ci_type": "pointwise_95_percent",
                            "unit": "percentage_points",
                            "graph_count": sum(
                                len(grouped[(model, key[0], key[1], radius)])
                                for key in keys
                            ),
                            "pair_count": sum(
                                int(row.get("pair_count", 0))
                                for key in keys
                                for row in grouped[
                                    (model, key[0], key[1], radius)
                                ]
                            ),
                            "numerator": None,
                            "denominator": None,
                            "status": "defined",
                            "applicability": "applicable",
                            "na_reason": None,
                            "stratum_count": 3,
                            "weighting": "equal_strata_then_equal_graph",
                            "bootstrap_unit": "whole_graph_within_n_m",
                            "bootstrap_replicates": bootstrap_replicates,
                            "equivalence_margin_lower": (
                                -EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
                            ),
                            "equivalence_margin_upper": (
                                EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
                            ),
                            **equivalence_status_fields(lower_pp, upper_pp),
                            "equivalence_classification": equivalence_classification(
                                lower_pp, upper_pp
                            ),
                            "simultaneous_result_applicability": (
                                "not_applicable_to_n_or_m_specific_marginal"
                            ),
                        }
                    )
    return output


def graph_level_complete_system_comparisons(
    rows: Sequence[GraphRow],
) -> list[dict[str, object]]:
    """Return matched Hydra-Poincare versus MDS-Euclidean system contrasts."""

    if not rows:
        raise ValueError("complete-system comparison requires graph-level rows")
    validate_graph_level_rows(rows, pair_count=int(rows[0]["pair_count"]))
    by_condition = {str(row["coordinate_condition_id"]): row for row in rows}
    output: list[dict[str, object]] = []
    for radius, label in zip(
        MATCHED_RADII,
        MATCHED_RADIUS_LABELS,
        strict=True,
    ):
        hydra = by_condition[f"hydra_scaled_{label}"]
        mds = by_condition[f"mds_scaled_{label}"]
        hydra_poincare = float(hydra["poincare_success"])
        mds_euclidean = float(mds["euclidean_success"])
        output.append(
            {
                "graph_id": hydra["graph_id"],
                "model": hydra["model"],
                "n": hydra["n"],
                "m": hydra["m"],
                "replicate_index": hydra["replicate_index"],
                "matched_radius": radius,
                "hydra_poincare_success": hydra_poincare,
                "mds_euclidean_success": mds_euclidean,
                "complete_system_difference": hydra_poincare - mds_euclidean,
                "comparison_type": "matched_complete_system_comparison",
                "interpretation": (
                    "coordinate_construction_and_metric_are_jointly_contrasted"
                ),
                "metric_only_effect": False,
                "pair_count": hydra["pair_count"],
                "unit": "proportion",
                "independent_unit": "graph",
            }
        )
    return output


def graph_level_native_interactions(
    rows: Sequence[GraphRow],
) -> list[dict[str, object]]:
    """Compatibility API for the single secondary native system reference."""

    if not rows:
        raise ValueError("native reference requires graph-level rows")
    validate_graph_level_rows(rows, pair_count=int(rows[0]["pair_count"]))
    by_condition = {str(row["coordinate_condition_id"]): row for row in rows}
    hydra = by_condition["hydra_native"]
    mds = by_condition["mds_native"]
    estimate = float(hydra["poincare_success"]) - float(mds["euclidean_success"])
    return [
        {
            "graph_id": hydra["graph_id"],
            "model": hydra["model"],
            "n": hydra["n"],
            "m": hydra["m"],
            "replicate_index": hydra["replicate_index"],
            "matched_radius": None,
            "native_reference_difference": estimate,
            "interaction": estimate,
            "comparison_type": "secondary_native_complete_system_reference",
            "metric_only_effect": False,
            "pair_count": hydra["pair_count"],
            "unit": "proportion",
            "independent_unit": "graph",
        }
    ]


def _graph_metric_counts(row: GraphRow, metric: str) -> tuple[float | int | None, int | None]:
    pair_count = int(row.get("pair_count", 0))
    count_fields = {
        "euclidean_success": "euclidean_success_count",
        "poincare_success": "poincare_success_count",
        "repaired_poincare_success": "repaired_poincare_success_count",
    }
    if metric in count_fields:
        return row.get(count_fields[metric]), pair_count
    if metric == "poincare_minus_euclidean":
        left = row.get("poincare_success_count")
        right = row.get("euclidean_success_count")
        return (None if left is None or right is None else int(left) - int(right), pair_count)
    if metric == "repaired_minus_unrepaired_poincare":
        left = row.get("repaired_poincare_success_count")
        right = row.get("poincare_success_count")
        return (None if left is None or right is None else int(left) - int(right), pair_count)
    if metric == "conditional_repair_recovery":
        return row.get("repair_recovery_numerator"), row.get("repair_recovery_denominator")
    if metric == "common_success_poincare_minus_euclidean_stretch":
        summary = row.get("stretch_summaries", {}).get("common_success", {})
        return summary.get("numerator"), summary.get("denominator")
    return None, None


def aggregate_graph_metrics(
    graph_rows: Sequence[GraphRow],
    *,
    metrics: Sequence[str] = (
        "euclidean_success",
        "poincare_success",
        "repaired_poincare_success",
        "poincare_minus_euclidean",
        "repaired_minus_unrepaired_poincare",
        "conditional_repair_recovery",
    ),
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_provider: BootstrapProvider = bootstrap_indices,
) -> list[dict[str, object]]:
    """Cell and equal-stratum marginal estimates using whole graphs."""

    if not 2 <= bootstrap_replicates <= BOOTSTRAP_REPLICATES:
        raise ValueError("invalid bootstrap replicate count")
    grouped: dict[tuple[str, int, int, str], list[GraphRow]] = defaultdict(list)
    for row in graph_rows:
        grouped[
            (
                str(row["model"]),
                int(row["n"]),
                int(row["m"]),
                str(row["coordinate_condition_id"]),
            )
        ].append(row)
    conditions = sorted(
        {str(row["coordinate_condition_id"]) for row in graph_rows}
    )
    for model in GRAPH_MODELS:
        for n in N_VALUES:
            for m in M_VALUES:
                expected_ids: tuple[str, ...] | None = None
                for condition in conditions:
                    selected = sorted(
                        grouped.get((model, n, m, condition), ()),
                        key=lambda row: (int(row["replicate_index"]), str(row["graph_id"])),
                    )
                    current_ids = tuple(str(row["graph_id"]) for row in selected)
                    if not current_ids:
                        raise ValueError("graph-metric stratum is missing")
                    if len(set(current_ids)) != len(current_ids):
                        raise ValueError("duplicate graph identity in graph-metric stratum")
                    if expected_ids is None:
                        expected_ids = current_ids
                    elif current_ids != expected_ids:
                        raise ValueError("conditions are not paired by graph identity")
    minimum_defined = max(
        2,
        ceil(bootstrap_replicates * MIN_DEFINED_BOOTSTRAP_FRACTION),
    )
    output: list[dict[str, object]] = []
    for model in GRAPH_MODELS:
        for condition in conditions:
            family, condition_kind, matched_radius, _ = (
                _condition_specifications()[condition]
            )
            marginal: dict[str, list[dict[str, object]]] = {
                metric: [] for metric in metrics
            }
            for n in N_VALUES:
                for m in M_VALUES:
                    selected = sorted(
                        grouped.get((model, n, m, condition), ()),
                        key=lambda row: (int(row["replicate_index"]), str(row["graph_id"])),
                    )
                    if not selected:
                        raise ValueError("graph-metric stratum is missing")
                    draw = _bootstrap_matrix(
                        model=model,
                        n=n,
                        m=m,
                        graph_count=len(selected),
                        replicates=bootstrap_replicates,
                        provider=bootstrap_provider,
                    )
                    for metric in metrics:
                        metric_unit = (
                            "hop_stretch_ratio"
                            if "stretch" in metric
                            else "proportion"
                        )
                        raw = [row.get(metric) for row in selected]
                        applicable = [
                            float(value) for value in raw if value is not None
                        ]
                        bootstrap_values = np.full(
                            bootstrap_replicates,
                            np.nan,
                            dtype=np.float64,
                        )
                        for replicate, indices in enumerate(draw):
                            sampled = [
                                float(raw[int(index)])
                                for index in indices
                                if raw[int(index)] is not None
                            ]
                            if sampled:
                                bootstrap_values[replicate] = float(
                                    np.mean(sampled)
                                )
                        defined_boot = bootstrap_values[
                            np.isfinite(bootstrap_values)
                        ]
                        interval_defined = len(defined_boot) >= minimum_defined
                        lower: float | None
                        upper: float | None
                        if interval_defined:
                            lower, upper = percentile_interval(defined_boot)
                        else:
                            lower, upper = None, None
                        total_count = len(selected)
                        applicable_count = len(applicable)
                        na_count = total_count - applicable_count
                        estimate = (
                            None
                            if not applicable
                            else float(np.mean(applicable))
                        )
                        row_reasons = {
                            str(row.get(f"{metric}_na_reason"))
                            for row in selected
                            if row.get(metric) is None
                            and row.get(f"{metric}_na_reason") is not None
                        }
                        if not applicable:
                            reason = (
                                next(iter(row_reasons))
                                if len(row_reasons) == 1
                                else "native_mds_poincare_undefined_by_design"
                                if condition == "mds_native"
                                and metric != "euclidean_success"
                                else "all_graph_values_undefined"
                            )
                            status = "all_graphs_na"
                            applicability = "not_applicable"
                        elif not interval_defined:
                            reason = (
                                "insufficient_defined_bootstrap_replicates"
                            )
                            status = "estimate_defined_interval_unavailable"
                            applicability = "applicable"
                        else:
                            reason = None
                            status = "defined"
                            applicability = "applicable"
                        cell = {
                            "scope": "model_n_m_condition_cell",
                            "model": model,
                            "n": n,
                            "m": m,
                            "coordinate_condition_id": condition,
                            "embedding_family": family,
                            "condition_kind": condition_kind,
                            "matched_radius": matched_radius,
                            "metric": metric,
                            "status": status,
                            "applicability": applicability,
                            "na_reason": reason,
                            "estimate": estimate,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "unit": metric_unit,
                            "independent_unit": "graph",
                            "graph_count": total_count,
                            "pair_count": sum(int(row.get("pair_count", 0)) for row in selected),
                            "total_graph_count": total_count,
                            "applicable_graph_count": applicable_count,
                            "na_graph_count": na_count,
                            "stratum_count": 1,
                            "weighting": "equal_applicable_graph",
                            "bootstrap_unit": (
                                "whole_graph_drop_undefined_within_replicate"
                            ),
                            "bootstrap_replicates": bootstrap_replicates,
                            "defined_bootstrap_replicates": len(defined_boot),
                            "minimum_defined_bootstrap_replicates": (
                                minimum_defined
                            ),
                        }
                        exact_counts = [_graph_metric_counts(row, metric) for row in selected]
                        numerators = [value[0] for value in exact_counts]
                        denominators = [value[1] for value in exact_counts]
                        cell["numerator"] = (
                            None
                            if any(value is None for value in numerators)
                            else float(sum(float(value) for value in numerators))
                        )
                        cell["denominator"] = (
                            None
                            if any(value is None for value in denominators)
                            else int(sum(int(value) for value in denominators))
                        )
                        output.append(cell)
                        marginal[metric].append(
                            {
                                "estimate": estimate,
                                "bootstrap": bootstrap_values,
                                "total_graph_count": total_count,
                                "applicable_graph_count": applicable_count,
                                "na_graph_count": na_count,
                                "pair_count": cell["pair_count"],
                                "numerator": cell["numerator"],
                                "denominator": cell["denominator"],
                            }
                        )
            for metric, strata in marginal.items():
                metric_unit = (
                    "hop_stretch_ratio"
                    if "stretch" in metric
                    else "proportion"
                )
                if len(strata) != 9:
                    raise ValueError("marginal metric requires nine strata")
                point_values = [item["estimate"] for item in strata]
                stacked = np.stack(
                    [item["bootstrap"] for item in strata]
                )
                complete_replicates = np.all(np.isfinite(stacked), axis=0)
                defined_boot = np.mean(
                    stacked[:, complete_replicates],
                    axis=0,
                )
                all_strata_defined = all(
                    value is not None for value in point_values
                )
                interval_defined = (
                    all_strata_defined
                    and len(defined_boot) >= minimum_defined
                )
                if interval_defined:
                    lower, upper = percentile_interval(defined_boot)
                else:
                    lower, upper = None, None
                estimate = (
                    float(np.mean(point_values))
                    if all_strata_defined
                    else None
                )
                if not all_strata_defined:
                    status = "not_estimable_missing_stratum"
                    reason = "one_or_more_strata_all_na"
                elif not interval_defined:
                    status = "estimate_defined_interval_unavailable"
                    reason = "insufficient_defined_bootstrap_replicates"
                else:
                    status = "defined"
                    reason = None
                output.append(
                    {
                        "scope": "model_condition_n_m_marginal",
                        "model": model,
                        "n": None,
                        "m": None,
                        "coordinate_condition_id": condition,
                        "embedding_family": family,
                        "condition_kind": condition_kind,
                        "matched_radius": matched_radius,
                        "metric": metric,
                        "status": status,
                        "applicability": (
                            "applicable"
                            if all_strata_defined
                            else "not_applicable"
                        ),
                        "na_reason": reason,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "unit": metric_unit,
                        "independent_unit": "graph",
                        "graph_count": sum(
                            int(item["total_graph_count"])
                            for item in strata
                        ),
                        "pair_count": sum(int(item["pair_count"]) for item in strata),
                        "numerator": (
                            None
                            if any(item["numerator"] is None for item in strata)
                            else float(sum(float(item["numerator"]) for item in strata))
                        ),
                        "denominator": (
                            None
                            if any(item["denominator"] is None for item in strata)
                            else int(sum(int(item["denominator"]) for item in strata))
                        ),
                        "total_graph_count": sum(
                            int(item["total_graph_count"])
                            for item in strata
                        ),
                        "applicable_graph_count": sum(
                            int(item["applicable_graph_count"])
                            for item in strata
                        ),
                        "na_graph_count": sum(
                            int(item["na_graph_count"])
                            for item in strata
                        ),
                        "stratum_count": 9,
                        "applicable_stratum_count": sum(
                            item["estimate"] is not None for item in strata
                        ),
                        "weighting": "equal_n_m_strata_then_equal_graph",
                        "bootstrap_unit": "whole_graph_within_n_m",
                        "bootstrap_replicates": bootstrap_replicates,
                        "defined_bootstrap_replicates": len(defined_boot),
                        "minimum_defined_bootstrap_replicates": minimum_defined,
                    }
                )
    return output


def simultaneous_radius_bands(
    interaction_rows: Sequence[GraphRow],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_provider: BootstrapProvider = bootstrap_indices,
) -> list[dict[str, object]]:
    """Paired maximum-deviation bands across the four repeated radii."""

    if not 2 <= bootstrap_replicates <= BOOTSTRAP_REPLICATES:
        raise ValueError("invalid bootstrap replicate count")
    grouped: dict[tuple[str, int, int, float], list[GraphRow]] = defaultdict(list)
    for row in interaction_rows:
        grouped[
            (
                str(row["model"]),
                int(row["n"]),
                int(row["m"]),
                float(row["matched_radius"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for model in GRAPH_MODELS:
        stratum_points: list[NDArray[np.float64]] = []
        stratum_boots: list[NDArray[np.float64]] = []
        total_graphs = 0
        for n in N_VALUES:
            for m in M_VALUES:
                columns: list[NDArray[np.float64]] = []
                identities: tuple[str, ...] | None = None
                for radius in MATCHED_RADII:
                    selected = sorted(
                        grouped.get((model, n, m, radius), ()),
                        key=lambda row: int(row["replicate_index"]),
                    )
                    current = tuple(str(row["graph_id"]) for row in selected)
                    if not current or (
                        identities is not None and current != identities
                    ):
                        raise ValueError("radii are not paired by graph")
                    identities = current
                    columns.append(
                        np.asarray(
                            [float(row["interaction"]) for row in selected],
                            dtype=np.float64,
                        )
                    )
                values = np.column_stack(columns)
                draw = _bootstrap_matrix(
                    model=model,
                    n=n,
                    m=m,
                    graph_count=len(values),
                    replicates=bootstrap_replicates,
                    provider=bootstrap_provider,
                )
                point = values.mean(axis=0)
                boot = values[draw].mean(axis=1)
                stratum_points.append(point)
                stratum_boots.append(boot)
                total_graphs += len(values)
                output.extend(
                    _simultaneous_rows(
                        point,
                        boot,
                        scope="model_n_m_radius_cell",
                        model=model,
                        n=n,
                        m=m,
                        graph_count=len(values),
                        pair_count=sum(
                            int(row.get("pair_count", 0))
                            for row in selected
                        ),
                        stratum_count=1,
                        bootstrap_replicates=bootstrap_replicates,
                    )
                )
        marginal_point = np.stack(stratum_points).mean(axis=0)
        marginal_boot = np.stack(stratum_boots).mean(axis=0)
        output.extend(
            _simultaneous_rows(
                marginal_point,
                marginal_boot,
                scope="model_radius_n_m_marginal",
                model=model,
                n=None,
                m=None,
                graph_count=total_graphs,
                pair_count=sum(
                    int(row.get("pair_count", 0))
                    for n in N_VALUES
                    for m in M_VALUES
                    for radius in MATCHED_RADII[:1]
                    for row in grouped[(model, n, m, radius)]
                ),
                stratum_count=9,
                bootstrap_replicates=bootstrap_replicates,
            )
        )
    return output


def _simultaneous_rows(
    point: NDArray[np.float64],
    boot: NDArray[np.float64],
    *,
    scope: str,
    model: str,
    n: int | None,
    m: int | None,
    graph_count: int,
    pair_count: int,
    stratum_count: int,
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    deviations = np.max(np.abs(boot - point[None, :]), axis=1)
    ordered = np.sort(deviations, kind="stable")
    critical = float(
        ordered[ceil(0.95 * len(ordered)) - 1]
    )
    rows: list[dict[str, object]] = []
    all_inside_margin = True
    for index, radius in enumerate(MATCHED_RADII):
        lower = float(point[index] - critical) * 100.0
        upper = float(point[index] + critical) * 100.0
        statuses = equivalence_status_fields(lower, upper)
        classification = equivalence_classification(lower, upper)
        all_inside_margin = (
            all_inside_margin and statuses["ci_wholly_inside_margin"]
        )
        rows.append(
            {
                "scope": scope,
                "model": model,
                "n": n,
                "m": m,
                "matched_radius": radius,
                "estimate": float(point[index]) * 100.0,
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_type": "simultaneous_95_percent_familywise",
                "simultaneous_ci_lower": lower,
                "simultaneous_ci_upper": upper,
                "simultaneous_critical_value": critical * 100.0,
                "unit": "percentage_points",
                "graph_count": graph_count,
                "pair_count": pair_count,
                "numerator": None,
                "denominator": None,
                "status": "defined",
                "applicability": "applicable",
                "na_reason": None,
                "stratum_count": stratum_count,
                "bootstrap_replicates": bootstrap_replicates,
                "procedure": (
                    "paired_graph_bootstrap_maximum_absolute_deviation_band"
                ),
                "equivalence_margin_lower": (
                    -EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
                ),
                "equivalence_margin_upper": (
                    EQUIVALENCE_MARGIN_PERCENTAGE_POINTS
                ),
                **statuses,
                "equivalence_classification": classification,
                "simultaneous_result_applicability": "applicable",
                "all_radius_ci_wholly_inside_margin": None,
            }
        )
    for row in rows:
        row["all_radius_ci_wholly_inside_margin"] = all_inside_margin
    return rows


def _descriptive_model_contrast_cells(
    graph_rows: Sequence[GraphRow],
    *,
    metrics: Sequence[str] = (
        "euclidean_success",
        "poincare_success",
        "repaired_poincare_success",
        "poincare_minus_euclidean",
        "repaired_minus_unrepaired_poincare",
    ),
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_provider: BootstrapProvider = bootstrap_indices,
) -> list[dict[str, object]]:
    """BA-minus-ER with independent graph resampling in matched strata."""

    grouped: dict[
        tuple[str, int, int, str], list[GraphRow]
    ] = defaultdict(list)
    for row in graph_rows:
        grouped[
            (
                str(row["model"]),
                int(row["n"]),
                int(row["m"]),
                str(row["coordinate_condition_id"]),
            )
        ].append(row)
    conditions = sorted(
        {str(row["coordinate_condition_id"]) for row in graph_rows}
    )
    output: list[dict[str, object]] = []
    for n in N_VALUES:
        for m in M_VALUES:
            for condition in conditions:
                er = sorted(
                    grouped[("erdos_renyi", n, m, condition)],
                    key=lambda row: int(row["replicate_index"]),
                )
                ba = sorted(
                    grouped[("barabasi_albert", n, m, condition)],
                    key=lambda row: int(row["replicate_index"]),
                )
                if not er or not ba:
                    raise ValueError("model contrast stratum is incomplete")
                er_draw = _bootstrap_matrix(
                    model="erdos_renyi",
                    n=n,
                    m=m,
                    graph_count=len(er),
                    replicates=bootstrap_replicates,
                    provider=bootstrap_provider,
                )
                ba_draw = _bootstrap_matrix(
                    model="barabasi_albert",
                    n=n,
                    m=m,
                    graph_count=len(ba),
                    replicates=bootstrap_replicates,
                    provider=bootstrap_provider,
                )
                for metric in metrics:
                    er_values_raw = [row.get(metric) for row in er]
                    ba_values_raw = [row.get(metric) for row in ba]
                    if all(value is None for value in er_values_raw + ba_values_raw):
                        continue
                    if any(value is None for value in er_values_raw + ba_values_raw):
                        raise ValueError("model contrast has asymmetric missingness")
                    er_values = np.asarray(er_values_raw, dtype=np.float64)
                    ba_values = np.asarray(ba_values_raw, dtype=np.float64)
                    boot = (
                        ba_values[ba_draw].mean(axis=1)
                        - er_values[er_draw].mean(axis=1)
                    )
                    lower, upper = percentile_interval(boot)
                    output.append(
                        {
                            "scope": "n_m_condition_cell",
                            "n": n,
                            "m": m,
                            "coordinate_condition_id": condition,
                            "metric": metric,
                            "contrast": "barabasi_albert_minus_erdos_renyi",
                            "estimate": float(
                                ba_values.mean() - er_values.mean()
                            ),
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "unit": "proportion",
                            "status": "defined",
                            "applicability": "applicable",
                            "na_reason": None,
                            "graph_count": len(er_values) + len(ba_values),
                            "pair_count": sum(
                                int(row.get("pair_count", 0))
                                for row in (*er, *ba)
                            ),
                            "numerator": None,
                            "denominator": None,
                            "er_graph_count": len(er_values),
                            "ba_graph_count": len(ba_values),
                            "stratum_count": 1,
                            "weighting": "equal_graph_within_model",
                            "bootstrap_replicates": bootstrap_replicates,
                            "resampling": "independent_graphs_within_matched_n_m",
                            "causal_interpretation": False,
                            "models_differ_in_multiple_properties": True,
                        }
                    )
    return output


def descriptive_model_contrasts(
    graph_rows: Sequence[GraphRow],
    *,
    metrics: Sequence[str] = (
        "euclidean_success",
        "poincare_success",
        "repaired_poincare_success",
        "poincare_minus_euclidean",
        "repaired_minus_unrepaired_poincare",
    ),
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_provider: BootstrapProvider = bootstrap_indices,
) -> list[dict[str, object]]:
    """Return descriptive BA-minus-ER cells and equal-stratum marginals."""

    if not 2 <= bootstrap_replicates <= BOOTSTRAP_REPLICATES:
        raise ValueError("invalid bootstrap replicate count")
    output = _descriptive_model_contrast_cells(
        graph_rows,
        metrics=metrics,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_provider=bootstrap_provider,
    )
    grouped: dict[tuple[str, int, int, str], list[GraphRow]] = defaultdict(list)
    for row in graph_rows:
        grouped[
            (
                str(row["model"]),
                int(row["n"]),
                int(row["m"]),
                str(row["coordinate_condition_id"]),
            )
        ].append(row)
    conditions = sorted({str(row["coordinate_condition_id"]) for row in graph_rows})
    for condition in conditions:
        for metric in metrics:
            stratum_points: list[float] = []
            stratum_boots: list[np.ndarray] = []
            er_count = 0
            ba_count = 0
            pair_count = 0
            unavailable_reason: str | None = None
            for n in N_VALUES:
                for m in M_VALUES:
                    model_values: dict[str, np.ndarray] = {}
                    model_boots: dict[str, np.ndarray] = {}
                    for model in GRAPH_MODELS:
                        selected = sorted(
                            grouped.get((model, n, m, condition), ()),
                            key=lambda row: (
                                int(row["replicate_index"]),
                                str(row["graph_id"]),
                            ),
                        )
                        if not selected or len({str(row["graph_id"]) for row in selected}) != len(selected):
                            raise ValueError("model contrast stratum is incomplete or duplicated")
                        raw = [row.get(metric) for row in selected]
                        if any(value is None for value in raw):
                            unavailable_reason = "one_or_more_graph_values_undefined"
                            break
                        values = np.asarray(raw, dtype=np.float64)
                        draw = _bootstrap_matrix(
                            model=model,
                            n=n,
                            m=m,
                            graph_count=len(values),
                            replicates=bootstrap_replicates,
                            provider=bootstrap_provider,
                        )
                        model_values[model] = values
                        model_boots[model] = values[draw].mean(axis=1)
                        pair_count += sum(int(row.get("pair_count", 0)) for row in selected)
                        if model == "erdos_renyi":
                            er_count += len(selected)
                        else:
                            ba_count += len(selected)
                    if unavailable_reason is not None:
                        break
                    er = model_values["erdos_renyi"]
                    ba = model_values["barabasi_albert"]
                    stratum_points.append(float(ba.mean() - er.mean()))
                    stratum_boots.append(
                        model_boots["barabasi_albert"]
                        - model_boots["erdos_renyi"]
                    )
                if unavailable_reason is not None:
                    break
            if unavailable_reason is None and len(stratum_points) == 9:
                boot = np.stack(stratum_boots).mean(axis=0)
                lower, upper = percentile_interval(boot)
                estimate: float | None = float(np.mean(stratum_points))
                status = "defined"
                applicability = "applicable"
                reason = None
            else:
                estimate = lower = upper = None
                status = "not_estimable_missing_stratum"
                applicability = "not_applicable"
                reason = unavailable_reason or "one_or_more_strata_missing"
            output.append(
                {
                    "scope": "equal_n_m_stratum_model_marginal",
                    "n": None,
                    "m": None,
                    "coordinate_condition_id": condition,
                    "metric": metric,
                    "contrast": "barabasi_albert_minus_erdos_renyi",
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "unit": "proportion",
                    "status": status,
                    "applicability": applicability,
                    "na_reason": reason,
                    "graph_count": er_count + ba_count,
                    "er_graph_count": er_count,
                    "ba_graph_count": ba_count,
                    "pair_count": pair_count,
                    "numerator": None,
                    "denominator": None,
                    "stratum_count": 9,
                    "weighting": "equal_n_m_strata_then_equal_graph",
                    "bootstrap_replicates": bootstrap_replicates,
                    "resampling": "independent_whole_graphs_within_model_n_m",
                    "causal_interpretation": False,
                    "models_differ_in_multiple_properties": True,
                }
            )
    return output


NETWORK_PROPERTIES = (
    "average_degree",
    "maximum_degree",
    "population_degree_variance",
    "average_clustering_coefficient",
    "diameter",
    "average_shortest_path_length",
)
PROPERTY_OUTCOMES = {
    "success_contrast": "poincare_minus_euclidean",
    "common_success_stretch_contrast": (
        "common_success_poincare_minus_euclidean_stretch"
    ),
}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _association_statistic(
    x: np.ndarray,
    y: np.ndarray,
    strata: Sequence[tuple[int, int]],
    *,
    ranks: bool,
) -> tuple[float | None, str | None, int]:
    if len(x) != len(y) or len(x) != len(strata):
        raise ValueError("association arrays and strata must align")
    finite = np.isfinite(x) & np.isfinite(y)
    selected_indices = np.flatnonzero(finite)
    if len(selected_indices) < 3:
        return None, "fewer_than_three_defined_graphs", len(selected_indices)
    x_selected = x[selected_indices]
    y_selected = y[selected_indices]
    selected_strata = [strata[int(index)] for index in selected_indices]
    residual_x = np.empty(len(selected_indices), dtype=np.float64)
    residual_y = np.empty(len(selected_indices), dtype=np.float64)
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for local_index, stratum in enumerate(selected_strata):
        grouped[stratum].append(local_index)
    for indices in grouped.values():
        numeric = np.asarray(indices, dtype=np.int64)
        x_values = x_selected[numeric]
        y_values = y_selected[numeric]
        if ranks:
            x_values = _average_ranks(x_values)
            y_values = _average_ranks(y_values)
        residual_x[numeric] = x_values - float(np.mean(x_values))
        residual_y[numeric] = y_values - float(np.mean(y_values))
    if float(np.std(residual_x)) == 0.0:
        return None, "zero_within_stratum_residual_variance_predictor", len(selected_indices)
    if float(np.std(residual_y)) == 0.0:
        return None, "zero_within_stratum_residual_variance_outcome", len(selected_indices)
    value = float(np.corrcoef(residual_x, residual_y)[0, 1])
    if not isfinite(value):
        return None, "nonfinite_residual_correlation", len(selected_indices)
    return value, None, len(selected_indices)


def _family_id(model: str, outcome: str) -> str:
    prefix = "er" if model == "erdos_renyi" else "ba"
    suffix = (
        "success_contrast"
        if outcome == "success_contrast"
        else "common_success_stretch_contrast"
    )
    return f"{prefix}_{suffix}"


def _permutation_seed(
    family_id: str,
    *,
    n: int,
    m: int,
    replicate: int,
) -> int:
    payload = "\0".join(
        (
            "greedy-routing-iteration2-property-permutation-v1",
            ANALYSIS_PLAN_HASH,
            family_id,
            str(n),
            str(m),
            str(replicate),
        )
    ).encode("utf-8")
    return int.from_bytes(
        blake2s(payload, digest_size=16, person=b"GRP2perm").digest(),
        "big",
    )


def _shared_resampling_maps(
    *,
    model: str,
    family_id: str,
    strata: Sequence[tuple[int, int]],
    replicates: int,
    bootstrap_provider: BootstrapProvider,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[stratum].append(index)
    expected = {(n, m) for n in N_VALUES for m in M_VALUES}
    if set(grouped) != expected:
        raise ValueError("property association requires every n-m stratum")
    bootstrap = np.empty((replicates, len(strata)), dtype=np.int64)
    permutations = np.empty((replicates, len(strata)), dtype=np.int64)
    for replicate in range(replicates):
        bootstrap_parts: list[int] = []
        permutation = np.arange(len(strata), dtype=np.int64)
        for (n, m), indices in sorted(grouped.items()):
            local_draw = bootstrap_provider(
                model=model,
                n=n,
                m=m,
                replicate=replicate,
                graph_count=len(indices),
            )
            if len(local_draw) != len(indices):
                raise ValueError("association bootstrap draw has wrong size")
            if any(not 0 <= int(index) < len(indices) for index in local_draw):
                raise ValueError("association bootstrap draw is out of range")
            bootstrap_parts.extend(indices[int(index)] for index in local_draw)
            generator = np.random.Generator(
                np.random.PCG64(
                    _permutation_seed(
                        family_id,
                        n=n,
                        m=m,
                        replicate=replicate,
                    )
                )
            )
            selected = np.asarray(indices, dtype=np.int64)
            permutation[selected] = generator.permutation(selected)
        bootstrap[replicate] = np.asarray(bootstrap_parts, dtype=np.int64)
        permutations[replicate] = permutation
    return bootstrap, permutations


def _family_fingerprint(
    *,
    family_id: str,
    model: str,
    outcome: str,
    conditions: Sequence[str],
    replicates: int,
) -> str:
    payload = {
        "schema": "iteration2_property_family_v2",
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "family_id": family_id,
        "model": model,
        "outcome": outcome,
        "properties": list(NETWORK_PROPERTIES),
        "coordinate_conditions": list(conditions),
        "hypotheses": [
            {"property": property_name, "coordinate_condition_id": condition}
            for property_name in NETWORK_PROPERTIES
            for condition in conditions
        ],
        "permutation": (
            "shared_within_n_m_graph_mapping_maximum_absolute_correlation"
        ),
        "bootstrap": "shared_whole_graph_within_n_m",
        "replicates": replicates,
        "alpha": 0.05,
        "missingness": "preserve_na_no_jitter_no_imputation",
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def property_associations(
    graph_metrics: Sequence[Mapping[str, object]],
    graph_rows: Sequence[GraphRow],
    *,
    inference_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_provider: BootstrapProvider = bootstrap_indices,
) -> list[dict[str, object]]:
    """Compute the exact four 54-hypothesis max-statistic families."""

    if not 2 <= inference_replicates <= BOOTSTRAP_REPLICATES:
        raise ValueError("invalid property-association replicate count")
    metric_by_graph: dict[str, Mapping[str, object]] = {}
    for row in graph_metrics:
        graph_id = str(row.get("graph_id", ""))
        if not graph_id or graph_id in metric_by_graph:
            raise ValueError("graph metrics contain missing or duplicate graph identity")
        for property_name in NETWORK_PROPERTIES:
            _finite_number(row.get(property_name), property_name)
        metric_by_graph[graph_id] = row
    row_by_identity: dict[tuple[str, str], GraphRow] = {}
    for row in graph_rows:
        identity = (str(row.get("graph_id", "")), str(row.get("coordinate_condition_id", "")))
        if not identity[0] or identity in row_by_identity:
            raise ValueError("graph-level association rows contain duplicate identity")
        row_by_identity[identity] = row
    conditions = tuple(
        condition for condition in _condition_specifications() if condition != "mds_native"
    )
    if len(conditions) != 9:
        raise RuntimeError("property family must contain nine P/E-comparable conditions")
    output: list[dict[str, object]] = []
    for model in GRAPH_MODELS:
        model_metrics = sorted(
            (row for row in graph_metrics if str(row.get("model")) == model),
            key=lambda row: (
                int(row["n"]),
                int(row["m"]),
                int(row["replicate_index"]),
                str(row["graph_id"]),
            ),
        )
        graph_ids = tuple(str(row["graph_id"]) for row in model_metrics)
        if not graph_ids or len(set(graph_ids)) != len(graph_ids):
            raise ValueError("model graph metrics are missing or duplicated")
        strata = [(int(row["n"]), int(row["m"])) for row in model_metrics]
        for graph_id, metric_row in zip(graph_ids, model_metrics, strict=True):
            for condition in conditions:
                route_row = row_by_identity.get((graph_id, condition))
                if route_row is None:
                    raise ValueError("property family is missing a graph-condition row")
                if (
                    route_row.get("model") != model
                    or int(route_row["n"]) != int(metric_row["n"])
                    or int(route_row["m"]) != int(metric_row["m"])
                ):
                    raise ValueError("property and routing graph identities disagree")
        x_by_property = {
            property_name: np.asarray(
                [float(row[property_name]) for row in model_metrics],
                dtype=np.float64,
            )
            for property_name in NETWORK_PROPERTIES
        }
        for outcome_name, outcome_field in PROPERTY_OUTCOMES.items():
            family_id = _family_id(model, outcome_name)
            bootstrap_map, permutation_map = _shared_resampling_maps(
                model=model,
                family_id=family_id,
                strata=strata,
                replicates=inference_replicates,
                bootstrap_provider=bootstrap_provider,
            )
            family_fingerprint = _family_fingerprint(
                family_id=family_id,
                model=model,
                outcome=outcome_name,
                conditions=conditions,
                replicates=inference_replicates,
            )
            hypotheses: list[dict[str, object]] = []
            permutation_statistics = np.full(
                (len(NETWORK_PROPERTIES) * len(conditions), inference_replicates),
                np.nan,
                dtype=np.float64,
            )
            hypothesis_index = 0
            for property_name in NETWORK_PROPERTIES:
                x = x_by_property[property_name]
                for condition in conditions:
                    raw_y = [row_by_identity[(graph_id, condition)].get(outcome_field) for graph_id in graph_ids]
                    y = np.asarray(
                        [np.nan if value is None else float(value) for value in raw_y],
                        dtype=np.float64,
                    )
                    estimate, reason, defined_graphs = _association_statistic(
                        x, y, strata, ranks=False
                    )
                    rank_estimate, rank_reason, _ = _association_statistic(
                        x, y, strata, ranks=True
                    )
                    bootstrap_primary: list[float] = []
                    bootstrap_rank: list[float] = []
                    for replicate, sampled in enumerate(bootstrap_map):
                        sampled_strata = [strata[int(index)] for index in sampled]
                        primary, _, _ = _association_statistic(
                            x[sampled], y[sampled], sampled_strata, ranks=False
                        )
                        sensitivity, _, _ = _association_statistic(
                            x[sampled], y[sampled], sampled_strata, ranks=True
                        )
                        if primary is not None:
                            bootstrap_primary.append(primary)
                        if sensitivity is not None:
                            bootstrap_rank.append(sensitivity)
                        permuted, _, _ = _association_statistic(
                            x,
                            y[permutation_map[replicate]],
                            strata,
                            ranks=False,
                        )
                        if permuted is not None:
                            permutation_statistics[hypothesis_index, replicate] = permuted
                    minimum_defined = max(
                        2,
                        ceil(inference_replicates * MIN_DEFINED_BOOTSTRAP_FRACTION),
                    )
                    if len(bootstrap_primary) >= minimum_defined:
                        ci_lower, ci_upper = percentile_interval(bootstrap_primary)
                    else:
                        ci_lower = ci_upper = None
                    if len(bootstrap_rank) >= minimum_defined:
                        rank_ci_lower, rank_ci_upper = percentile_interval(bootstrap_rank)
                    else:
                        rank_ci_lower = rank_ci_upper = None
                    status = (
                        "not_applicable"
                        if estimate is None
                        else "estimate_defined_interval_unavailable"
                        if ci_lower is None
                        else "defined"
                    )
                    na_reason = (
                        reason
                        if estimate is None
                        else "insufficient_defined_bootstrap_replicates"
                        if ci_lower is None
                        else None
                    )
                    hypotheses.append(
                        {
                            "model": model,
                            "outcome": outcome_name,
                            "coordinate_condition_id": condition,
                            "property": property_name,
                            "family_id": family_id,
                            "multiplicity_family_fingerprint": family_fingerprint,
                            "hypothesis_family_size": 54,
                            "association_estimate": estimate,
                            "correlation": estimate,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "rank_sensitivity_estimate": rank_estimate,
                            "rank_sensitivity_ci_lower": rank_ci_lower,
                            "rank_sensitivity_ci_upper": rank_ci_upper,
                            "rank_sensitivity_na_reason": rank_reason,
                            "status": status,
                            "applicability": (
                                "applicable" if estimate is not None else "not_applicable"
                            ),
                            "na_reason": na_reason,
                            "graph_count": len(graph_ids),
                            "defined_graph_count": defined_graphs,
                            "pair_count": None,
                            "numerator": None,
                            "denominator": defined_graphs,
                            "unit": "correlation",
                            "independent_unit": "graph",
                            "stratification": "within_model_n_m",
                            "bootstrap_replicates": inference_replicates,
                            "defined_bootstrap_replicates": len(bootstrap_primary),
                            "permutation_replicates": inference_replicates,
                            "permutation_mapping": "shared_across_all_54_family_hypotheses",
                            "multiplicity_procedure": "maximum_absolute_statistic_familywise",
                            "exploratory": True,
                            "associative": True,
                            "causal_interpretation": False,
                        }
                    )
                    hypothesis_index += 1
            if len(hypotheses) != 54:
                raise RuntimeError("property family does not contain exactly 54 hypotheses")
            maximum_statistics = np.full(inference_replicates, np.nan, dtype=np.float64)
            for replicate in range(inference_replicates):
                values = np.abs(permutation_statistics[:, replicate])
                defined = values[np.isfinite(values)]
                if len(defined):
                    maximum_statistics[replicate] = float(np.max(defined))
            for index, row in enumerate(hypotheses):
                estimate = row["association_estimate"]
                hypothesis_permutations = np.abs(permutation_statistics[index])
                defined_raw = hypothesis_permutations[np.isfinite(hypothesis_permutations)]
                defined_max = maximum_statistics[np.isfinite(maximum_statistics)]
                if estimate is None:
                    raw_p = adjusted_p = None
                else:
                    observed = abs(float(estimate))
                    raw_p = (1 + int(np.sum(defined_raw >= observed))) / (len(defined_raw) + 1)
                    adjusted_p = (1 + int(np.sum(defined_max >= observed))) / (len(defined_max) + 1)
                row["raw_permutation_p_value"] = raw_p
                row["familywise_adjusted_p_value"] = adjusted_p
                row["familywise_alpha"] = 0.05
                row["defined_family_hypothesis_count"] = sum(
                    item["association_estimate"] is not None for item in hypotheses
                )
            output.extend(hypotheses)
    expected_families = {
        "er_success_contrast",
        "ba_success_contrast",
        "er_common_success_stretch_contrast",
        "ba_common_success_stretch_contrast",
    }
    observed_families = {str(row["family_id"]) for row in output}
    if observed_families != expected_families or len(output) != 216:
        raise RuntimeError("property analysis does not contain four exact 54-hypothesis families")
    return output


def reject_result_dependent_configuration(
    proposed_configuration_inputs: Mapping[str, Any],
) -> None:
    """Reject any attempted tuning request that contains routing outcomes."""

    forbidden_fragments = (
        "success",
        "stretch",
        "failure",
        "interaction",
        "equivalence",
        "result",
        "outcome",
    )

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).lower()
                if any(fragment in normalized for fragment in forbidden_fragments):
                    raise ValueError(
                        f"result-dependent configuration is forbidden: {path}{key}"
                    )
                visit(item, f"{path}{key}.")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}{index}.")

    visit(proposed_configuration_inputs, "")
