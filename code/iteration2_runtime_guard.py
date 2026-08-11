"""Context-local enforcement and measurement of Iteration 2 science.

The reporting analysis and production preflight paths are intentionally allowed
to read and validate existing artifacts, but they must never execute a
scientific operation.  Every operation in ``SCIENTIFIC_OPERATION_CATALOG`` is
registered by a decorator at its real execution boundary.  A context-local
ledger therefore both blocks prohibited calls before their body runs and
provides measured provenance for successful read-only consumers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from types import MappingProxyType
from threading import RLock
from typing import ParamSpec, TypeVar, cast


P = ParamSpec("P")
R = TypeVar("R")

ANALYSIS_READ_ONLY = "analysis_read_only"
PREFLIGHT_READ_ONLY = "preflight_read_only"
SCIENTIFIC_EXECUTION = "scientific_execution"
SCIENTIFIC_REGENERATION_AUDIT = "scientific_regeneration_audit"
READ_ONLY_MODES = frozenset((ANALYSIS_READ_ONLY, PREFLIGHT_READ_ONLY))

SCIENTIFIC_OPERATION_CATALOG: Mapping[str, str] = MappingProxyType(
    {
        "graph_generation": "Generate an Iteration 2 graph.",
        "scheduled_graph_execution": "Execute one scheduled graph workload.",
        "graph_workload_execution": (
            "Execute an Iteration 2 graph workload from an existing graph."
        ),
        "excluded_fixture_execution": (
            "Construct and execute an excluded non-scientific graph fixture."
        ),
        "pair_sampling": "Sample source-destination pairs.",
        "dijkstra": "Execute the per-pair Dijkstra benchmark.",
        "euclidean_greedy_routing": "Execute Euclidean greedy routing.",
        "poincare_greedy_routing": "Execute Poincare greedy routing.",
        "repaired_poincare_routing": "Execute one-backtrack Poincare routing.",
        "scientific_embedding_coordinates": (
            "Construct scientific embeddings and coordinate conditions."
        ),
        "raw_checkpoint_construction": (
            "Construct serialized raw-checkpoint payload bytes."
        ),
        "raw_checkpoint_publication": "Publish a raw graph checkpoint.",
        "scientific_regeneration_audit": (
            "Regenerate a scheduled graph for an explicit scientific audit."
        ),
    }
)


class ScientificOperationBlocked(RuntimeError):
    """Raised before a scientific operation enters its body in read-only mode."""


@dataclass
class ScientificOperationLedger:
    """Mutable only inside its owning context; snapshots are plain immutable data."""

    mode: str
    attempted: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SCIENTIFIC_OPERATION_CATALOG}
    )
    executed: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SCIENTIFIC_OPERATION_CATALOG}
    )
    blocked: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SCIENTIFIC_OPERATION_CATALOG}
    )
    attempted_sequence: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "iteration2_scientific_operation_ledger_v1",
            "mode": self.mode,
            "attempted_operation_counts": dict(self.attempted),
            "executed_operation_counts": dict(self.executed),
            "blocked_operation_counts": dict(self.blocked),
            "attempted_operation_sequence": list(self.attempted_sequence),
            "total_attempted": sum(self.attempted.values()),
            "total_executed": sum(self.executed.values()),
            "total_blocked": sum(self.blocked.values()),
        }


_ACTIVE_LEDGER: ContextVar[ScientificOperationLedger | None] = ContextVar(
    "iteration2_active_scientific_operation_ledger", default=None
)
_READ_ONLY_LOCK = RLock()
_PROCESS_READ_ONLY_LEDGER: ScientificOperationLedger | None = None
_REGISTERED_BOUNDARIES: dict[str, str] = {}


def registered_scientific_boundaries() -> Mapping[str, str]:
    """Return an immutable snapshot of operation-to-callable registrations."""

    return MappingProxyType(dict(_REGISTERED_BOUNDARIES))


def validate_scientific_boundary_registry() -> None:
    """Fail if any required operation lacks exactly one registered boundary."""

    expected = set(SCIENTIFIC_OPERATION_CATALOG)
    observed = set(_REGISTERED_BOUNDARIES)
    if observed != expected:
        raise RuntimeError(
            "scientific operation boundary registry mismatch; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def current_scientific_ledger() -> ScientificOperationLedger | None:
    local = _ACTIVE_LEDGER.get()
    if local is not None:
        return local
    with _READ_ONLY_LOCK:
        return _PROCESS_READ_ONLY_LEDGER


@contextmanager
def scientific_operation_context(mode: str) -> Iterator[ScientificOperationLedger]:
    """Create an isolated context-local ledger and reliably reset it on exit."""

    if mode not in (
        ANALYSIS_READ_ONLY,
        PREFLIGHT_READ_ONLY,
        SCIENTIFIC_EXECUTION,
        SCIENTIFIC_REGENERATION_AUDIT,
    ):
        raise ValueError("unknown scientific operation context mode")
    global _PROCESS_READ_ONLY_LEDGER
    if _ACTIVE_LEDGER.get() is not None:
        raise RuntimeError("nested scientific operation ledger contexts are forbidden")
    ledger = ScientificOperationLedger(mode=mode)
    with _READ_ONLY_LOCK:
        if _PROCESS_READ_ONLY_LEDGER is not None:
            raise RuntimeError("another process-wide read-only ledger is active")
        if mode in READ_ONLY_MODES:
            _PROCESS_READ_ONLY_LEDGER = ledger
    token = _ACTIVE_LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE_LEDGER.reset(token)
        if mode in READ_ONLY_MODES:
            with _READ_ONLY_LOCK:
                if _PROCESS_READ_ONLY_LEDGER is not ledger:
                    raise RuntimeError("read-only ledger ownership changed unexpectedly")
                _PROCESS_READ_ONLY_LEDGER = None


def scientific_operation_boundary(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Register and guard a real scientific-operation function boundary."""

    if name not in SCIENTIFIC_OPERATION_CATALOG:
        raise ValueError(f"unrecognized scientific operation boundary: {name}")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        qualified_name = f"{function.__module__}.{function.__qualname__}"
        prior = _REGISTERED_BOUNDARIES.get(name)
        if prior is not None and prior != qualified_name:
            raise RuntimeError(
                f"scientific operation {name} has multiple boundaries: "
                f"{prior}, {qualified_name}"
            )
        _REGISTERED_BOUNDARIES[name] = qualified_name

        @wraps(function)
        def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
            ledger = current_scientific_ledger()
            if ledger is not None:
                with _READ_ONLY_LOCK:
                    ledger.attempted[name] += 1
                    ledger.attempted_sequence.append(name)
                    if ledger.mode in READ_ONLY_MODES:
                        ledger.blocked[name] += 1
                        raise ScientificOperationBlocked(
                            f"{name} is prohibited in {ledger.mode} mode"
                        )
            result = function(*args, **kwargs)
            if ledger is not None:
                with _READ_ONLY_LOCK:
                    ledger.executed[name] += 1
            return cast(R, result)

        setattr(guarded, "__iteration2_scientific_operation__", name)
        return guarded

    return decorate


def require_zero_scientific_operations(
    snapshot: Mapping[str, object], *, context: str
) -> None:
    """Fail closed when a read-only phase attempted or executed science."""

    if (
        int(snapshot.get("total_attempted", -1)) != 0
        or int(snapshot.get("total_executed", -1)) != 0
        or int(snapshot.get("total_blocked", -1)) != 0
    ):
        raise RuntimeError(f"{context} recorded a prohibited scientific operation")
