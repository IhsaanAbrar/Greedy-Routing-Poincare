"""Independent float64 and high-precision routing oracle for Iteration 2."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import blake2s
from math import asinh, hypot, isfinite, sqrt
from numbers import Integral
from typing import TypeVar

import networkx as nx

from iteration2_config import (
    DATA_GENERATION_HASH,
    HIGH_PRECISION_DECIMAL_DIGITS,
    ROUTING_PRIORITY_VERSION,
    canonical_json_bytes,
    routing_priority_seed,
)


LOCAL_MINIMUM = "local_minimum"
REPAIR_UNAVAILABLE_AT_SOURCE = "repair_unavailable_at_source"
NO_ALTERNATIVE_AFTER_BACKTRACKING = "no_alternative_after_backtracking"
POST_REPAIR_LOCAL_MINIMUM = "post_repair_local_minimum"
POST_REPAIR_ATTEMPTED_REVISIT = "post_repair_attempted_revisit"

Number = TypeVar("Number", float, Decimal)
Distance = Callable[[Sequence[float], Sequence[float]], Number]


class OracleInvariantError(RuntimeError):
    """Raised when the independent state machine reaches an impossible state."""


@dataclass(frozen=True)
class _PriorityIdentity:
    data_generation_hash: str
    graph_id: str
    pair_index: int
    source: int
    destination: int

    @property
    def identity(self) -> str:
        return blake2s(
            canonical_json_bytes(
                {
                    "version": ROUTING_PRIORITY_VERSION,
                    "data_generation_hash": self.data_generation_hash,
                    "graph_id": self.graph_id,
                    "pair_index": self.pair_index,
                    "source": self.source,
                    "destination": self.destination,
                }
            ),
            digest_size=16,
            person=b"I2ctxV1",
        ).hexdigest()


@dataclass(frozen=True)
class OracleRoute:
    success: bool
    walk: tuple[int, ...]
    final_failure_type: str | None
    initial_failure_type: str | None
    repair_attempted: bool
    repair_succeeded: bool
    repair_backtrackable: bool | None
    repair_eligible: bool
    repair_alternative_existed: bool | None
    repair_alternative_selected: bool
    repair_attempt_count: int
    forwarding_decisions: int
    logical_distance_evaluations: int
    peak_history_vertices: int
    priority_context_id: str
    repair_backtracked_vertex: int | None = None
    repair_excluded_branch: int | None = None
    repair_selected_alternative: int | None = None

    @property
    def route_length(self) -> int:
        return len(self.walk) - 1

    @property
    def physical_hops(self) -> int:
        return len(self.walk) - 1

    @property
    def decision_signature(self) -> tuple[object, ...]:
        return (
            self.success,
            self.walk,
            self.final_failure_type,
            self.initial_failure_type,
            self.repair_attempted,
            self.repair_succeeded,
            self.repair_backtrackable,
            self.repair_eligible,
            self.repair_alternative_existed,
            self.repair_alternative_selected,
            self.repair_attempt_count,
            self.physical_hops,
            self.forwarding_decisions,
            self.logical_distance_evaluations,
            self.peak_history_vertices,
            self.priority_context_id,
            self.repair_backtracked_vertex,
            self.repair_excluded_branch,
            self.repair_selected_alternative,
        )


@dataclass(frozen=True)
class OracleAgreement:
    float64_matches_production: bool
    high_precision_matches_production: bool
    float64_matches_high_precision: bool
    production_signature: tuple[object, ...]
    float64_signature: tuple[object, ...]
    high_precision_signature: tuple[object, ...]


@dataclass
class _Counters:
    forwarding_decisions: int = 0
    logical_distance_evaluations: int = 0
    peak_history_vertices: int = 1


@dataclass(frozen=True)
class _Outcome:
    success: bool
    walk: tuple[int, ...]
    failure_type: str | None


def _float_point(point: Sequence[float], name: str) -> tuple[float, float]:
    try:
        values = tuple(float(value) for value in point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite two-vector") from exc
    if len(values) != 2 or not all(isfinite(value) for value in values):
        raise ValueError(f"{name} must be a finite two-vector")
    return values[0], values[1]


def independent_euclidean_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    left = _float_point(first, "first")
    right = _float_point(second, "second")
    return hypot(left[0] - right[0], left[1] - right[1])


def independent_poincare_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    left = _float_point(first, "first")
    right = _float_point(second, "second")
    left_radius = hypot(*left)
    right_radius = hypot(*right)
    if not left_radius < 1.0 or not right_radius < 1.0:
        raise ValueError("Poincare points must be in the open unit disk")
    difference = hypot(left[0] - right[0], left[1] - right[1])
    if difference == 0.0:
        return 0.0
    denominator = sqrt(
        (1.0 - left_radius)
        * (1.0 + left_radius)
        * (1.0 - right_radius)
        * (1.0 + right_radius)
    )
    result = 2.0 * asinh(difference / denominator)
    if not isfinite(result) or result < 0.0:
        raise ValueError("independent Poincare distance is invalid")
    return result


def _decimal_point(
    point: Sequence[float],
    name: str,
) -> tuple[Decimal, Decimal]:
    values = _float_point(point, name)
    return Decimal.from_float(values[0]), Decimal.from_float(values[1])


def decimal_euclidean_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> Decimal:
    left = _decimal_point(first, "first")
    right = _decimal_point(second, "second")
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2).sqrt()


def _decimal_asinh(value: Decimal) -> Decimal:
    if value == 0:
        return Decimal(0)
    return (value + (value * value + Decimal(1)).sqrt()).ln()


def decimal_poincare_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> Decimal:
    left = _decimal_point(first, "first")
    right = _decimal_point(second, "second")
    left_squared = left[0] * left[0] + left[1] * left[1]
    right_squared = right[0] * right[0] + right[1] * right[1]
    if left_squared >= 1 or right_squared >= 1:
        raise ValueError("Poincare points must be in the open unit disk")
    difference = (
        (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2
    ).sqrt()
    if difference == 0:
        return Decimal(0)
    denominator = (
        (Decimal(1) - left_squared) * (Decimal(1) - right_squared)
    ).sqrt()
    return Decimal(2) * _decimal_asinh(difference / denominator)


def _graph_signature(
    graph: nx.Graph,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    nodes = tuple(sorted(int(node) for node in graph.nodes))
    edges = tuple(
        sorted(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in graph.edges
        )
    )
    return nodes, edges


def _validate_graph_and_coordinates(
    graph: nx.Graph,
    coordinates: Mapping[int, Sequence[float]],
    source: int,
    destination: int,
) -> None:
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or nx.number_of_selfloops(graph)
        or not nx.is_connected(graph)
    ):
        raise ValueError("oracle graph must be simple, connected, and undirected")
    nodes, _ = _graph_signature(graph)
    if nodes != tuple(range(len(nodes))):
        raise ValueError("oracle graph nodes must equal 0 through n-1")
    if any("weight" in data for _, _, data in graph.edges(data=True)):
        raise ValueError("oracle graph must be unweighted")
    if (
        isinstance(source, bool)
        or not isinstance(source, Integral)
        or isinstance(destination, bool)
        or not isinstance(destination, Integral)
        or source == destination
        or source not in graph
        or destination not in graph
    ):
        raise ValueError("oracle source and destination must be distinct graph nodes")
    if set(coordinates) != set(graph.nodes):
        raise ValueError("oracle coordinates must exactly cover graph nodes")
    for node in graph.nodes:
        _float_point(coordinates[node], f"coordinate[{node}]")


def _priority_identity(
    graph: nx.Graph,
    source: int,
    destination: int,
    priority_context: object | None,
) -> _PriorityIdentity:
    if priority_context is None:
        graph_id = "ad_hoc_" + blake2s(
            canonical_json_bytes(_graph_signature(graph)),
            digest_size=8,
            person=b"I2graph",
        ).hexdigest()
        return _PriorityIdentity(
            DATA_GENERATION_HASH, graph_id, 0, int(source), int(destination)
        )
    try:
        identity = _PriorityIdentity(
            str(priority_context.data_generation_hash),
            str(priority_context.graph_id),
            int(priority_context.pair_index),
            int(priority_context.source),
            int(priority_context.destination),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("oracle priority context is invalid") from exc
    if (
        identity.data_generation_hash != DATA_GENERATION_HASH
        or not identity.graph_id
        or identity.pair_index < 0
        or identity.source != source
        or identity.destination != destination
    ):
        raise ValueError("oracle priority context differs from the routed pair")
    return identity


def _priority(
    identity: _PriorityIdentity,
    current: int,
    candidate: int,
) -> bytes:
    key = blake2s(
        canonical_json_bytes(
            {
                "version": ROUTING_PRIORITY_VERSION,
                "data_generation_hash": identity.data_generation_hash,
                "graph_id": identity.graph_id,
                "routing_priority_seed": routing_priority_seed(identity.graph_id),
            }
        ),
        digest_size=32,
        person=b"I2keyV1",
    ).digest()
    return blake2s(
        canonical_json_bytes(
            {
                "version": ROUTING_PRIORITY_VERSION,
                "graph_id": identity.graph_id,
                "pair_index": identity.pair_index,
                "source": identity.source,
                "destination": identity.destination,
                "current": int(current),
                "candidate": int(candidate),
            }
        ),
        key=key,
        digest_size=32,
        person=b"I2tieV1",
    ).digest()


def _choose(
    candidates: Sequence[int],
    *,
    current: int,
    identity: _PriorityIdentity,
) -> int:
    if len(candidates) == 1:
        return int(candidates[0])
    ranked = [(_priority(identity, current, node), int(node)) for node in candidates]
    if len({digest for digest, _ in ranked}) != len(ranked):
        raise OracleInvariantError("oracle routing priority digest collision")
    return min(ranked, key=lambda item: item[0])[1]


def _distance(
    function: Distance[Number],
    coordinates: Mapping[int, Sequence[float]],
    node: int,
    destination: int,
    counters: _Counters,
) -> Number:
    counters.logical_distance_evaluations += 1
    value = function(coordinates[node], coordinates[destination])
    if value < 0:
        raise OracleInvariantError("oracle distance became negative")
    return value


def _ordinary_walk(
    graph: nx.Graph,
    coordinates: Mapping[int, Sequence[float]],
    destination: int,
    distance: Distance[Number],
    tolerance: Number,
    identity: _PriorityIdentity,
    walk: list[int],
    visited: set[int],
    counters: _Counters,
    *,
    stage: str,
) -> _Outcome:
    strict_steps = 0
    while walk[-1] != destination:
        if strict_steps >= graph.number_of_nodes():
            raise OracleInvariantError("oracle strict-progress bound exceeded")
        current = walk[-1]
        neighbours = tuple(sorted(int(node) for node in graph.neighbors(current)))
        if destination in neighbours:
            if destination in visited:
                raise OracleInvariantError("oracle revisited an adjacent destination")
            walk.append(destination)
            visited.add(destination)
            counters.forwarding_decisions += 1
            counters.peak_history_vertices = max(
                counters.peak_history_vertices, len(visited)
            )
            strict_steps += 1
            continue
        current_distance = _distance(
            distance, coordinates, current, destination, counters
        )
        ranked = tuple(
            (node, _distance(distance, coordinates, node, destination, counters))
            for node in neighbours
        )
        if not ranked:
            raise OracleInvariantError("connected oracle route reached no neighbour")
        minimum = min(value for _, value in ranked)
        if not minimum < current_distance - tolerance:
            failure = LOCAL_MINIMUM if stage == "ordinary" else POST_REPAIR_LOCAL_MINIMUM
            return _Outcome(False, tuple(walk), failure)
        tied = tuple(
            node
            for node, value in ranked
            if value < current_distance - tolerance and value <= minimum + tolerance
        )
        if not tied:
            raise OracleInvariantError("oracle progressing minimum left its tie set")
        selected = _choose(tied, current=current, identity=identity)
        if selected in visited:
            if stage == "ordinary":
                raise OracleInvariantError("ordinary oracle route attempted a revisit")
            return _Outcome(False, tuple(walk), POST_REPAIR_ATTEMPTED_REVISIT)
        walk.append(selected)
        visited.add(selected)
        counters.forwarding_decisions += 1
        counters.peak_history_vertices = max(
            counters.peak_history_vertices, len(visited)
        )
        strict_steps += 1
    return _Outcome(True, tuple(walk), None)


def _route(
    graph: nx.Graph,
    coordinates: Mapping[int, Sequence[float]],
    source: int,
    destination: int,
    distance: Distance[Number],
    tolerance: Number,
    identity: _PriorityIdentity,
    *,
    repaired: bool,
) -> OracleRoute:
    counters = _Counters()
    initial = _ordinary_walk(
        graph,
        coordinates,
        destination,
        distance,
        tolerance,
        identity,
        [source],
        {source},
        counters,
        stage="ordinary",
    )
    if not repaired or initial.success:
        return OracleRoute(
            success=initial.success,
            walk=initial.walk,
            final_failure_type=initial.failure_type,
            initial_failure_type=initial.failure_type,
            repair_attempted=False,
            repair_succeeded=False,
            repair_backtrackable=None,
            repair_eligible=False,
            repair_alternative_existed=None,
            repair_alternative_selected=False,
            repair_attempt_count=0,
            forwarding_decisions=counters.forwarding_decisions,
            logical_distance_evaluations=counters.logical_distance_evaluations,
            peak_history_vertices=counters.peak_history_vertices,
            priority_context_id=identity.identity,
        )
    if initial.failure_type != LOCAL_MINIMUM:
        raise OracleInvariantError("oracle repair did not follow a local minimum")
    if len(initial.walk) < 2:
        return OracleRoute(
            success=False,
            walk=initial.walk,
            final_failure_type=REPAIR_UNAVAILABLE_AT_SOURCE,
            initial_failure_type=LOCAL_MINIMUM,
            repair_attempted=False,
            repair_succeeded=False,
            repair_backtrackable=False,
            repair_eligible=False,
            repair_alternative_existed=False,
            repair_alternative_selected=False,
            repair_attempt_count=0,
            forwarding_decisions=counters.forwarding_decisions,
            logical_distance_evaluations=counters.logical_distance_evaluations,
            peak_history_vertices=counters.peak_history_vertices,
            priority_context_id=identity.identity,
        )
    failed_branch = initial.walk[-1]
    predecessor = initial.walk[-2]
    physical_walk = [*initial.walk, predecessor]
    explored = set(initial.walk)
    counters.forwarding_decisions += 1
    alternatives = tuple(
        sorted(
            int(node)
            for node in graph.neighbors(predecessor)
            if int(node) != failed_branch and int(node) not in explored
        )
    )
    if not alternatives:
        return OracleRoute(
            success=False,
            walk=tuple(physical_walk),
            final_failure_type=NO_ALTERNATIVE_AFTER_BACKTRACKING,
            initial_failure_type=LOCAL_MINIMUM,
            repair_attempted=True,
            repair_succeeded=False,
            repair_backtrackable=True,
            repair_eligible=True,
            repair_alternative_existed=False,
            repair_alternative_selected=False,
            repair_attempt_count=1,
            forwarding_decisions=counters.forwarding_decisions,
            logical_distance_evaluations=counters.logical_distance_evaluations,
            peak_history_vertices=counters.peak_history_vertices,
            priority_context_id=identity.identity,
            repair_backtracked_vertex=predecessor,
            repair_excluded_branch=failed_branch,
        )
    if destination in alternatives:
        alternative = destination
    else:
        ranked = tuple(
            (node, _distance(distance, coordinates, node, destination, counters))
            for node in alternatives
        )
        minimum = min(value for _, value in ranked)
        tied = tuple(node for node, value in ranked if value <= minimum + tolerance)
        alternative = _choose(tied, current=predecessor, identity=identity)
    physical_walk.append(alternative)
    explored.add(alternative)
    counters.forwarding_decisions += 1
    counters.peak_history_vertices = max(
        counters.peak_history_vertices, len(explored)
    )
    outcome = (
        _Outcome(True, tuple(physical_walk), None)
        if alternative == destination
        else _ordinary_walk(
            graph,
            coordinates,
            destination,
            distance,
            tolerance,
            identity,
            physical_walk,
            explored,
            counters,
            stage="post_repair",
        )
    )
    return OracleRoute(
        success=outcome.success,
        walk=outcome.walk,
        final_failure_type=outcome.failure_type,
        initial_failure_type=LOCAL_MINIMUM,
        repair_attempted=True,
        repair_succeeded=outcome.success,
        repair_backtrackable=True,
        repair_eligible=True,
        repair_alternative_existed=True,
        repair_alternative_selected=True,
        repair_attempt_count=1,
        forwarding_decisions=counters.forwarding_decisions,
        logical_distance_evaluations=counters.logical_distance_evaluations,
        peak_history_vertices=counters.peak_history_vertices,
        priority_context_id=identity.identity,
        repair_backtracked_vertex=predecessor,
        repair_excluded_branch=failed_branch,
        repair_selected_alternative=alternative,
    )


def oracle_route(
    graph: nx.Graph,
    coordinates: Mapping[int, Sequence[float]],
    source: int,
    destination: int,
    *,
    metric: str,
    tolerance: float,
    repaired: bool = False,
    precision: str = "float64",
    priority_context: object | None = None,
) -> OracleRoute:
    """Execute the independent routing specification."""

    _validate_graph_and_coordinates(graph, coordinates, source, destination)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not isfinite(float(tolerance))
        or tolerance <= 0
    ):
        raise ValueError("tolerance must be positive and finite")
    if metric not in ("euclidean", "poincare"):
        raise ValueError("metric must be euclidean or poincare")
    identity = _priority_identity(
        graph, int(source), int(destination), priority_context
    )
    if precision == "float64":
        function = (
            independent_euclidean_distance
            if metric == "euclidean"
            else independent_poincare_distance
        )
        return _route(
            graph,
            coordinates,
            int(source),
            int(destination),
            function,
            float(tolerance),
            identity,
            repaired=repaired,
        )
    if precision != "decimal":
        raise ValueError("precision must be float64 or decimal")
    with localcontext() as context:
        context.prec = HIGH_PRECISION_DECIMAL_DIGITS
        function = (
            decimal_euclidean_distance
            if metric == "euclidean"
            else decimal_poincare_distance
        )
        return _route(
            graph,
            coordinates,
            int(source),
            int(destination),
            function,
            Decimal.from_float(float(tolerance)),
            identity,
            repaired=repaired,
        )


def production_signature(result: object) -> tuple[object, ...]:
    try:
        return (
            result.success,
            tuple(result.walk),
            result.final_failure_type,
            result.initial_failure_type,
            result.repair_attempted,
            result.repair_succeeded,
            result.repair_backtrackable,
            result.repair_eligible,
            result.repair_alternative_existed,
            result.repair_alternative_selected,
            result.repair_attempt_count,
            result.physical_hops,
            result.forwarding_decisions,
            result.logical_distance_evaluations,
            result.peak_history_vertices,
            result.priority_context_id,
            result.repair_backtracked_vertex,
            result.repair_excluded_branch,
            result.repair_selected_alternative,
        )
    except AttributeError as exc:
        raise ValueError("result is not a production routing result") from exc


def audit_production_result(
    production_result: object,
    *,
    graph: nx.Graph,
    coordinates: Mapping[int, Sequence[float]],
    source: int,
    destination: int,
    metric: str,
    tolerance: float,
    repaired: bool,
    priority_context: object | None = None,
) -> OracleAgreement:
    """Compare production to independent float64 and decimal state machines."""

    float_result = oracle_route(
        graph,
        coordinates,
        source,
        destination,
        metric=metric,
        tolerance=tolerance,
        repaired=repaired,
        precision="float64",
        priority_context=priority_context,
    )
    precise_result = oracle_route(
        graph,
        coordinates,
        source,
        destination,
        metric=metric,
        tolerance=tolerance,
        repaired=repaired,
        precision="decimal",
        priority_context=priority_context,
    )
    production = production_signature(production_result)
    float_signature = float_result.decision_signature
    precise_signature = precise_result.decision_signature
    return OracleAgreement(
        float64_matches_production=production == float_signature,
        high_precision_matches_production=production == precise_signature,
        float64_matches_high_precision=float_signature == precise_signature,
        production_signature=production,
        float64_signature=float_signature,
        high_precision_signature=precise_signature,
    )
