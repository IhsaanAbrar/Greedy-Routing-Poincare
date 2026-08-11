"""Canonical identities for explicitly non-scientific Iteration 2 fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from iteration2_config import (
    ANALYSIS_PLAN_HASH,
    COMBINED_PROTOCOL_HASH,
    DATA_GENERATION_HASH,
    ITERATION2_ANALYSIS_IDENTITY,
    ITERATION2_RUN_IDENTITY,
    OUTPUT_SCHEMA_HASH,
    canonical_json_bytes,
)


EXCLUDED_FIXTURE_SCHEMA = "greedy_routing_iteration2_excluded_fixture_v1"
EXCLUDED_SCIENTIFIC_STATUS = "excluded_non_scientific"


def _identity_hash(domain: str, payload: Mapping[str, object]) -> str:
    return sha256(
        canonical_json_bytes(
            {
                "domain": domain,
                "excluded_fixture_payload": dict(payload),
            }
        )
    ).hexdigest()


def excluded_raw_identity(payload: Mapping[str, object]) -> str:
    return f"iteration2_excluded_raw_{_identity_hash('raw', payload)[:16]}"


def excluded_analysis_identity(payload: Mapping[str, object]) -> str:
    return (
        "iteration2_excluded_analysis_"
        f"{_identity_hash('analysis', payload)[:16]}"
    )


def excluded_fixture_payload_hash(payload: Mapping[str, object]) -> str:
    return sha256(canonical_json_bytes(dict(payload))).hexdigest()


def validate_excluded_fixture_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "mode",
        "fixture_schema",
        "fixture_tag",
        "excluded_graph_schedule",
        "excluded_seeds",
        "pair_count",
        "bootstrap_replicates",
        "property_resampling_replicates",
        "permutation_replicates",
        "protocol_identities",
    }
    if set(payload) != required:
        raise ValueError("excluded fixture payload fields are incomplete or unexpected")
    if payload["mode"] != "excluded_fixture":
        raise ValueError("excluded fixture payload mode is invalid")
    if payload["fixture_schema"] != EXCLUDED_FIXTURE_SCHEMA:
        raise ValueError("excluded fixture payload schema is invalid")
    tag = payload["fixture_tag"]
    if not isinstance(tag, str) or not tag or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in tag
    ):
        raise ValueError("excluded fixture tag is invalid")
    schedule = payload["excluded_graph_schedule"]
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
        raise ValueError("excluded fixture schedule is invalid")
    if any(not isinstance(value, str) for value in schedule):
        raise ValueError("excluded fixture graph IDs must be strings")
    graph_ids = tuple(schedule)
    if not graph_ids or len(set(graph_ids)) != len(graph_ids) or any(
        not graph_id.startswith("excluded_") for graph_id in graph_ids
    ):
        raise ValueError("excluded fixture graph schedule is invalid")
    seeds = payload["excluded_seeds"]
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise ValueError("excluded fixture seeds are invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in seeds
    ):
        raise ValueError("excluded fixture seeds must be non-negative integers")
    normalized_seeds = tuple(seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("excluded fixture seeds must be nonempty and unique")
    counts: dict[str, int] = {}
    for name in (
        "pair_count",
        "bootstrap_replicates",
        "property_resampling_replicates",
        "permutation_replicates",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"excluded fixture {name} must be a positive integer")
        counts[name] = int(value)
    protocol_identities = payload["protocol_identities"]
    expected_protocol_identities = {
        "data_generation_hash": DATA_GENERATION_HASH,
        "analysis_plan_hash": ANALYSIS_PLAN_HASH,
        "output_schema_hash": OUTPUT_SCHEMA_HASH,
        "combined_protocol_hash": COMBINED_PROTOCOL_HASH,
    }
    if protocol_identities != expected_protocol_identities:
        raise ValueError("excluded fixture protocol identities are invalid")
    normalized: dict[str, object] = {
        "mode": "excluded_fixture",
        "fixture_schema": EXCLUDED_FIXTURE_SCHEMA,
        "fixture_tag": tag,
        "excluded_graph_schedule": list(graph_ids),
        "excluded_seeds": list(normalized_seeds),
        "protocol_identities": expected_protocol_identities,
        **counts,
    }
    raw_identity = excluded_raw_identity(normalized)
    analysis_identity = excluded_analysis_identity(normalized)
    if (
        raw_identity == ITERATION2_RUN_IDENTITY
        or analysis_identity == ITERATION2_ANALYSIS_IDENTITY
        or raw_identity.startswith(f"{ITERATION2_RUN_IDENTITY}_")
        or analysis_identity.startswith(f"{ITERATION2_ANALYSIS_IDENTITY}_")
    ):
        raise RuntimeError("excluded fixture identity collided with production")
    return normalized


@dataclass(frozen=True)
class ExcludedAnalysisFixtureContract:
    """Complete identity-bearing contract for one disposable fixture."""

    fixture_tag: str
    expected_graph_ids: tuple[str, ...]
    excluded_seeds: tuple[int, ...]
    pair_count: int
    bootstrap_replicates: int = 2
    property_resampling_replicates: int = 2
    permutation_replicates: int = 2

    def __post_init__(self) -> None:
        validate_excluded_fixture_payload(self.payload)

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "mode": "excluded_fixture",
                "fixture_schema": EXCLUDED_FIXTURE_SCHEMA,
                "fixture_tag": self.fixture_tag,
                "excluded_graph_schedule": list(self.expected_graph_ids),
                "excluded_seeds": list(self.excluded_seeds),
                "protocol_identities": {
                    "data_generation_hash": DATA_GENERATION_HASH,
                    "analysis_plan_hash": ANALYSIS_PLAN_HASH,
                    "output_schema_hash": OUTPUT_SCHEMA_HASH,
                    "combined_protocol_hash": COMBINED_PROTOCOL_HASH,
                },
                "pair_count": self.pair_count,
                "bootstrap_replicates": self.bootstrap_replicates,
                "property_resampling_replicates": (
                    self.property_resampling_replicates
                ),
                "permutation_replicates": self.permutation_replicates,
            }
        )

    @property
    def payload_hash(self) -> str:
        return excluded_fixture_payload_hash(self.payload)

    @property
    def raw_identity(self) -> str:
        return excluded_raw_identity(self.payload)

    @property
    def analysis_identity(self) -> str:
        return excluded_analysis_identity(self.payload)
