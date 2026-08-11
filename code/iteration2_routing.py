"""Deterministic strict-progress routing for the Iteration 2 protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import blake2s
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType

import networkx as nx

from iteration2_config import (
    DATA_GENERATION_HASH,
    ROUTING_PRIORITY_VERSION,
    canonical_json_bytes,
    routing_priority_seed,
)
from iteration2_runtime_guard import scientific_operation_boundary
from poincare_distance import euclidean_distance, poincare_distance


LOCAL_MINIMUM = "local_minimum"
# Kept as import-compatible sentinels only. Ordinary revisits always raise.
ATTEMPTED_REVISIT = "attempted_revisit"
NUMERICAL_INVARIANT_FAILURE = "numerical_invariant_failure"
REPAIR_UNAVAILABLE_AT_SOURCE = "repair_unavailable_at_source"
NO_ALTERNATIVE_AFTER_BACKTRACKING = "no_alternative_after_backtracking"
POST_REPAIR_LOCAL_MINIMUM = "post_repair_local_minimum"
POST_REPAIR_ATTEMPTED_REVISIT = "post_repair_attempted_revisit"

ORDINARY_FAILURE_TYPES = (LOCAL_MINIMUM,)
REPAIR_FAILURE_TYPES = (
    REPAIR_UNAVAILABLE_AT_SOURCE,
    NO_ALTERNATIVE_AFTER_BACKTRACKING,
    POST_REPAIR_LOCAL_MINIMUM,
    POST_REPAIR_ATTEMPTED_REVISIT,
)
ALL_FAILURE_TYPES = (*ORDINARY_FAILURE_TYPES, *REPAIR_FAILURE_TYPES)

Coordinate = Sequence[float]
DistanceFunction = Callable[[Coordinate, Coordinate], float]


class Iteration2RoutingInvariantError(RuntimeError):
    """Raised for an implementation or mathematical routing invariant breach."""


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class RoutingPriorityContext:
    """Outcome-independent identity used for every numerical routing tie."""

    data_generation_hash: str
    graph_id: str
    pair_index: int
    source: int
    destination: int

    def __post_init__(self) -> None:
        if self.data_generation_hash != DATA_GENERATION_HASH:
            raise ValueError("routing priority must use the frozen data identity")
        if not isinstance(self.graph_id, str) or not self.graph_id:
            raise ValueError("graph_id must be a non-empty string")
        object.__setattr__(self, "pair_index", _integer("pair_index", self.pair_index))
        object.__setattr__(self, "source", _integer("source", self.source))
        object.__setattr__(
            self, "destination", _integer("destination", self.destination)
        )
        if self.source == self.destination:
            raise ValueError("routing pair source and destination must differ")

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

    def priority(self, current: int, candidate: int) -> bytes:
        """Return a keyed digest; raw node-label magnitude never orders ties."""

        current_id = _integer("current", current)
        candidate_id = _integer("candidate", candidate)
        key = blake2s(
            canonical_json_bytes(
                {
                    "version": ROUTING_PRIORITY_VERSION,
                    "data_generation_hash": self.data_generation_hash,
                    "graph_id": self.graph_id,
                    "routing_priority_seed": routing_priority_seed(self.graph_id),
                }
            ),
            digest_size=32,
            person=b"I2keyV1",
        ).digest()
        return blake2s(
            canonical_json_bytes(
                {
                    "version": ROUTING_PRIORITY_VERSION,
                    "graph_id": self.graph_id,
                    "pair_index": self.pair_index,
                    "source": self.source,
                    "destination": self.destination,
                    "current": current_id,
                    "candidate": candidate_id,
                }
            ),
            key=key,
            digest_size=32,
            person=b"I2tieV1",
        ).digest()


@dataclass(frozen=True)
class FailureDiagnostic:
    """Numerical and topological state at one terminal greedy decision."""

    failure_type: str
    stage: str
    stopping_vertex: int
    failure_hop: int
    terminal_degree: int
    current_distance: float | None
    best_neighbor_distance: float | None
    progress_gap: float | None
    distance_tolerance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Iteration2RoutingResult:
    """Complete physical walk, terminal state, and logical resource costs."""

    method: str
    source: int
    destination: int
    success: bool
    walk: tuple[int, ...]
    final_failure_type: str | None
    initial_failure_type: str | None
    repair_attempted: bool
    repair_succeeded: bool
    repair_alternative_existed: bool | None
    repair_attempt_count: int
    forwarding_decisions: int
    logical_distance_evaluations: int
    peak_history_vertices: int
    distance_tolerance: float
    priority_context_id: str
    initial_failure_diagnostic: FailureDiagnostic | None
    final_failure_diagnostic: FailureDiagnostic | None
    repair_backtrackable: bool | None = None
    repair_eligible: bool = False
    repair_alternative_selected: bool = False
    repair_backtracked_vertex: int | None = None
    repair_excluded_branch: int | None = None
    repair_selected_alternative: int | None = None

    def __post_init__(self) -> None:
        walk = tuple(int(node) for node in self.walk)
        object.__setattr__(self, "walk", walk)
        if not walk or walk[0] != self.source:
            raise ValueError("walk must be non-empty and begin at source")
        if self.success != (walk[-1] == self.destination):
            raise ValueError("success must agree with the terminal vertex")
        if self.success and self.final_failure_type is not None:
            raise ValueError("successful routing cannot have a final failure")
        if not self.success and self.final_failure_type not in ALL_FAILURE_TYPES:
            raise ValueError("failed routing must use a scientific terminal category")
        if self.repair_attempted != (self.repair_attempt_count == 1):
            raise ValueError("repair attempt fields disagree")
        if self.repair_succeeded != (self.repair_attempted and self.success):
            raise ValueError("repair success fields disagree")
        if self.repair_alternative_selected != (
            self.repair_selected_alternative is not None
        ):
            raise ValueError("repair alternative selection fields disagree")
        if self.initial_failure_diagnostic is not None and (
            self.initial_failure_diagnostic.failure_type != self.initial_failure_type
        ):
            raise ValueError("initial failure diagnostic disagrees with label")
        if self.final_failure_diagnostic is not None and (
            self.final_failure_diagnostic.failure_type != self.final_failure_type
        ):
            raise ValueError("final failure diagnostic disagrees with label")
        for name in (
            "repair_attempt_count",
            "forwarding_decisions",
            "logical_distance_evaluations",
            "peak_history_vertices",
        ):
            _integer(name, getattr(self, name))
        if self.forwarding_decisions != self.physical_hops:
            raise ValueError("every physical traversal must be a forwarding decision")
        if self.peak_history_vertices < 1:
            raise ValueError("peak history must contain at least the source")
        if (
            isinstance(self.distance_tolerance, bool)
            or not isinstance(self.distance_tolerance, Real)
            or not isfinite(float(self.distance_tolerance))
            or float(self.distance_tolerance) <= 0.0
        ):
            raise ValueError("distance tolerance must be positive and finite")
        if not isinstance(self.priority_context_id, str) or not self.priority_context_id:
            raise ValueError("priority context identity is missing")

    @property
    def failure_type(self) -> str | None:
        return self.final_failure_type

    @property
    def route_length(self) -> int:
        return len(self.walk) - 1

    @property
    def physical_hops(self) -> int:
        return len(self.walk) - 1

    @property
    def stopping_vertex(self) -> int:
        return self.walk[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "source": self.source,
            "destination": self.destination,
            "success": self.success,
            "walk": list(self.walk),
            "route_length": self.route_length,
            "physical_hops": self.physical_hops,
            "failure_type": self.final_failure_type,
            "initial_failure_type": self.initial_failure_type,
            "final_failure_type": self.final_failure_type,
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "repair_backtrackable": self.repair_backtrackable,
            "repair_eligible": self.repair_eligible,
            "repair_alternative_existed": self.repair_alternative_existed,
            "repair_alternative_selected": self.repair_alternative_selected,
            "repair_attempt_count": self.repair_attempt_count,
            "forwarding_decisions": self.forwarding_decisions,
            "logical_distance_evaluations": self.logical_distance_evaluations,
            "peak_history_vertices": self.peak_history_vertices,
            "distance_tolerance": self.distance_tolerance,
            "priority_context_id": self.priority_context_id,
            "initial_failure_diagnostic": (
                None
                if self.initial_failure_diagnostic is None
                else self.initial_failure_diagnostic.to_dict()
            ),
            "final_failure_diagnostic": (
                None
                if self.final_failure_diagnostic is None
                else self.final_failure_diagnostic.to_dict()
            ),
            "repair_backtracked_vertex": self.repair_backtracked_vertex,
            "repair_excluded_branch": self.repair_excluded_branch,
            "repair_selected_alternative": self.repair_selected_alternative,
        }


def _graph_signature(graph: nx.Graph) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    nodes = tuple(sorted(int(node) for node in graph.nodes))
    edges = tuple(
        sorted(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in graph.edges
        )
    )
    return nodes, edges


def _validated_graph(graph: nx.Graph) -> nx.Graph:
    if (
        not isinstance(graph, nx.Graph)
        or graph.is_directed()
        or graph.is_multigraph()
        or graph.number_of_nodes() < 2
        or nx.number_of_selfloops(graph)
        or not nx.is_connected(graph)
    ):
        raise ValueError("routing graph must be simple, connected, and undirected")
    nodes, edges = _graph_signature(graph)
    if nodes != tuple(range(len(nodes))):
        raise ValueError("routing nodes must be integer IDs 0 through n-1")
    if any("weight" in data for _, _, data in graph.edges(data=True)):
        raise ValueError("routing graph must be unweighted")
    frozen = nx.Graph()
    frozen.add_nodes_from(nodes)
    frozen.add_edges_from(edges)
    nx.freeze(frozen)
    return frozen


def _coordinate_snapshot(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate],
) -> Mapping[int, tuple[float, float]]:
    nodes = tuple(graph.nodes)
    if not isinstance(coordinates, Mapping) or set(coordinates) != set(nodes):
        raise ValueError("coordinates must exactly cover graph nodes")
    snapshot: dict[int, tuple[float, float]] = {}
    for node in nodes:
        try:
            point = tuple(float(value) for value in coordinates[node])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coordinates must be finite two-vectors") from exc
        if len(point) != 2 or not all(isfinite(value) for value in point):
            raise ValueError("coordinates must be finite two-vectors")
        snapshot[int(node)] = (point[0], point[1])
    return MappingProxyType(snapshot)


@dataclass
class PreparedIteration2Routing(Mapping[int, Coordinate]):
    """Validated metric context with a destination-scoped distance cache."""

    metric_name: str
    _source_graph: nx.Graph = field(repr=False)
    _graph: nx.Graph = field(repr=False)
    _coordinates: Mapping[int, tuple[float, float]] = field(repr=False)
    _distance_function: DistanceFunction = field(repr=False)
    _active_destination: int | None = field(default=None, init=False, repr=False)
    _distance_cache: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __getitem__(self, node: int) -> Coordinate:
        return self._coordinates[node]

    def __iter__(self) -> Iterator[int]:
        return iter(self._coordinates)

    def __len__(self) -> int:
        return len(self._coordinates)

    @property
    def cache_size(self) -> int:
        return len(self._distance_cache)

    @property
    def active_destination(self) -> int | None:
        return self._active_destination

    def require_compatible(
        self,
        graph: nx.Graph,
        distance_function: DistanceFunction,
    ) -> None:
        if graph is not self._source_graph:
            raise ValueError("prepared routing context belongs to another graph")
        if (
            _graph_signature(graph) != _graph_signature(self._graph)
            or any("weight" in data for _, _, data in graph.edges(data=True))
        ):
            raise ValueError("prepared routing source graph was mutated")
        if distance_function is not self._distance_function:
            raise ValueError("prepared routing context belongs to another metric")

    def distance(self, node: int, destination: int) -> float:
        if self._active_destination != destination:
            self._active_destination = destination
            self._distance_cache.clear()
        if node in self._distance_cache:
            return self._distance_cache[node]
        value = float(
            self._distance_function(
                self._coordinates[node], self._coordinates[destination]
            )
        )
        if not isfinite(value) or value < 0.0:
            raise Iteration2RoutingInvariantError(
                "distance function returned an invalid value"
            )
        self._distance_cache[node] = value
        return value


def prepare_iteration2_routing(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate],
    distance_function: DistanceFunction,
    *,
    metric_name: str,
) -> PreparedIteration2Routing:
    """Validate a graph/coordinate/metric context once for repeated routes."""

    if not callable(distance_function):
        raise ValueError("distance_function must be callable")
    if not isinstance(metric_name, str) or not metric_name:
        raise ValueError("metric_name must be a non-empty string")
    frozen = _validated_graph(graph)
    coordinates_snapshot = _coordinate_snapshot(frozen, coordinates)
    # Boundary and arithmetic validation is performed once for every point.
    for node in frozen:
        value = float(distance_function(coordinates_snapshot[node], coordinates_snapshot[node]))
        if value != 0.0:
            raise ValueError("distance function must return exact self-distance zero")
    return PreparedIteration2Routing(
        metric_name=metric_name,
        _source_graph=graph,
        _graph=frozen,
        _coordinates=coordinates_snapshot,
        _distance_function=distance_function,
    )


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
    diagnostic: FailureDiagnostic | None


def _validated_tolerance(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("tolerance must be positive and finite")
    return float(value)


def _route_context(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedIteration2Routing,
    distance_function: DistanceFunction,
) -> PreparedIteration2Routing:
    if isinstance(coordinates, PreparedIteration2Routing):
        coordinates.require_compatible(graph, distance_function)
        return coordinates
    return prepare_iteration2_routing(
        graph,
        coordinates,
        distance_function,
        metric_name=getattr(distance_function, "__qualname__", "metric"),
    )


def _default_priority_context(
    graph: nx.Graph,
    source: int,
    destination: int,
) -> RoutingPriorityContext:
    graph_id = "ad_hoc_" + blake2s(
        canonical_json_bytes(_graph_signature(_validated_graph(graph))),
        digest_size=8,
        person=b"I2graph",
    ).hexdigest()
    return RoutingPriorityContext(
        data_generation_hash=DATA_GENERATION_HASH,
        graph_id=graph_id,
        pair_index=0,
        source=source,
        destination=destination,
    )


def _validated_priority_context(
    graph: nx.Graph,
    source: int,
    destination: int,
    priority_context: RoutingPriorityContext | None,
) -> RoutingPriorityContext:
    context = (
        _default_priority_context(graph, source, destination)
        if priority_context is None
        else priority_context
    )
    if not isinstance(context, RoutingPriorityContext):
        raise ValueError("priority_context must be a RoutingPriorityContext")
    if context.source != source or context.destination != destination:
        raise ValueError("priority context pair differs from the routed pair")
    return context


def _metric_distance(
    prepared: PreparedIteration2Routing,
    node: int,
    destination: int,
    counters: _Counters,
) -> float:
    counters.logical_distance_evaluations += 1
    return prepared.distance(node, destination)


def _priority_choice(
    candidates: Sequence[int],
    *,
    current: int,
    priority_context: RoutingPriorityContext,
) -> int:
    if len(candidates) == 1:
        return int(candidates[0])
    priorities = [
        (priority_context.priority(current, int(candidate)), int(candidate))
        for candidate in candidates
    ]
    digests = [digest for digest, _ in priorities]
    if len(set(digests)) != len(digests):
        raise Iteration2RoutingInvariantError("routing priority digest collision")
    return min(priorities, key=lambda item: item[0])[1]


def _diagnostic(
    *,
    graph: nx.Graph,
    failure_type: str,
    stage: str,
    walk: Sequence[int],
    current_distance: float | None,
    best_distance: float | None,
    tolerance: float,
) -> FailureDiagnostic:
    return FailureDiagnostic(
        failure_type=failure_type,
        stage=stage,
        stopping_vertex=int(walk[-1]),
        failure_hop=len(walk) - 1,
        terminal_degree=int(graph.degree[walk[-1]]),
        current_distance=current_distance,
        best_neighbor_distance=best_distance,
        progress_gap=(
            None
            if current_distance is None or best_distance is None
            else current_distance - best_distance
        ),
        distance_tolerance=tolerance,
    )


def _continue(
    *,
    prepared: PreparedIteration2Routing,
    destination: int,
    tolerance: float,
    priority_context: RoutingPriorityContext,
    walk: list[int],
    visited: set[int],
    counters: _Counters,
    stage: str,
) -> _Outcome:
    graph = prepared._graph
    strict_steps = 0
    while walk[-1] != destination:
        if strict_steps >= graph.number_of_nodes():
            raise Iteration2RoutingInvariantError(
                "strict-progress routing exceeded its invariant step bound"
            )
        current = walk[-1]
        neighbours = tuple(sorted(int(node) for node in graph.neighbors(current)))
        if destination in neighbours:
            if destination in visited:
                raise Iteration2RoutingInvariantError(
                    "destination adjacency encountered an already delivered vertex"
                )
            walk.append(destination)
            visited.add(destination)
            counters.forwarding_decisions += 1
            counters.peak_history_vertices = max(
                counters.peak_history_vertices, len(visited)
            )
            strict_steps += 1
            continue

        current_distance = _metric_distance(
            prepared, current, destination, counters
        )
        ranked = tuple(
            (
                node,
                _metric_distance(prepared, node, destination, counters),
            )
            for node in neighbours
        )
        if not ranked:
            raise Iteration2RoutingInvariantError(
                "connected routing state reached no neighbour"
            )
        true_minimum = min(distance for _, distance in ranked)
        if not true_minimum < current_distance - tolerance:
            failure = LOCAL_MINIMUM if stage == "ordinary" else POST_REPAIR_LOCAL_MINIMUM
            return _Outcome(
                False,
                tuple(walk),
                failure,
                _diagnostic(
                    graph=graph,
                    failure_type=failure,
                    stage=stage,
                    walk=walk,
                    current_distance=current_distance,
                    best_distance=true_minimum,
                    tolerance=tolerance,
                ),
            )
        tied = tuple(
            node
            for node, distance in ranked
            if distance < current_distance - tolerance
            and distance <= true_minimum + tolerance
        )
        if not tied:
            raise Iteration2RoutingInvariantError(
                "progressing true minimum was absent from its tie set"
            )
        selected = _priority_choice(
            tied, current=current, priority_context=priority_context
        )
        if selected in visited:
            if stage == "ordinary":
                raise Iteration2RoutingInvariantError(
                    "ordinary strict-progress routing attempted a revisit"
                )
            failure = POST_REPAIR_ATTEMPTED_REVISIT
            return _Outcome(
                False,
                tuple(walk),
                failure,
                _diagnostic(
                    graph=graph,
                    failure_type=failure,
                    stage=stage,
                    walk=walk,
                    current_distance=current_distance,
                    best_distance=dict(ranked)[selected],
                    tolerance=tolerance,
                ),
            )
        walk.append(selected)
        visited.add(selected)
        counters.forwarding_decisions += 1
        counters.peak_history_vertices = max(
            counters.peak_history_vertices, len(visited)
        )
        strict_steps += 1
    return _Outcome(True, tuple(walk), None, None)


def _make_result(
    *,
    method: str,
    source: int,
    destination: int,
    outcome: _Outcome,
    counters: _Counters,
    tolerance: float,
    priority_context: RoutingPriorityContext,
    initial_failure_type: str | None = None,
    initial_diagnostic: FailureDiagnostic | None = None,
    repair_attempted: bool = False,
    repair_succeeded: bool = False,
    repair_alternative_existed: bool | None = None,
    repair_attempt_count: int = 0,
    repair_backtrackable: bool | None = None,
    repair_eligible: bool = False,
    repair_alternative_selected: bool = False,
    repair_backtracked_vertex: int | None = None,
    repair_excluded_branch: int | None = None,
    repair_selected_alternative: int | None = None,
) -> Iteration2RoutingResult:
    return Iteration2RoutingResult(
        method=method,
        source=source,
        destination=destination,
        success=outcome.success,
        walk=outcome.walk,
        final_failure_type=outcome.failure_type,
        initial_failure_type=(
            outcome.failure_type if initial_failure_type is None else initial_failure_type
        ),
        repair_attempted=repair_attempted,
        repair_succeeded=repair_succeeded,
        repair_alternative_existed=repair_alternative_existed,
        repair_attempt_count=repair_attempt_count,
        forwarding_decisions=counters.forwarding_decisions,
        logical_distance_evaluations=counters.logical_distance_evaluations,
        peak_history_vertices=counters.peak_history_vertices,
        distance_tolerance=tolerance,
        priority_context_id=priority_context.identity,
        initial_failure_diagnostic=(
            outcome.diagnostic if initial_diagnostic is None else initial_diagnostic
        ),
        final_failure_diagnostic=outcome.diagnostic,
        repair_backtrackable=repair_backtrackable,
        repair_eligible=repair_eligible,
        repair_alternative_selected=repair_alternative_selected,
        repair_backtracked_vertex=repair_backtracked_vertex,
        repair_excluded_branch=repair_excluded_branch,
        repair_selected_alternative=repair_selected_alternative,
    )


def strict_greedy_route(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedIteration2Routing,
    source: int,
    destination: int,
    distance_function: DistanceFunction,
    *,
    method: str,
    tolerance: float,
    priority_context: RoutingPriorityContext | None = None,
) -> Iteration2RoutingResult:
    source_id = _integer("source", source)
    destination_id = _integer("destination", destination)
    if source_id == destination_id:
        raise ValueError("source and destination must differ")
    prepared = _route_context(graph, coordinates, distance_function)
    if source_id not in prepared or destination_id not in prepared:
        raise ValueError("source and destination must be graph nodes")
    numeric_tolerance = _validated_tolerance(tolerance)
    priority = _validated_priority_context(
        graph, source_id, destination_id, priority_context
    )
    counters = _Counters()
    outcome = _continue(
        prepared=prepared,
        destination=destination_id,
        tolerance=numeric_tolerance,
        priority_context=priority,
        walk=[source_id],
        visited={source_id},
        counters=counters,
        stage="ordinary",
    )
    return _make_result(
        method=method,
        source=source_id,
        destination=destination_id,
        outcome=outcome,
        counters=counters,
        tolerance=numeric_tolerance,
        priority_context=priority,
        repair_backtrackable=None,
    )


@scientific_operation_boundary("euclidean_greedy_routing")
def euclidean_greedy_route_v2(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedIteration2Routing,
    source: int,
    destination: int,
    *,
    tolerance: float,
    priority_context: RoutingPriorityContext | None = None,
) -> Iteration2RoutingResult:
    return strict_greedy_route(
        graph,
        coordinates,
        source,
        destination,
        euclidean_distance,
        method="euclidean_greedy",
        tolerance=tolerance,
        priority_context=priority_context,
    )


@scientific_operation_boundary("poincare_greedy_routing")
def poincare_greedy_route_v2(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedIteration2Routing,
    source: int,
    destination: int,
    *,
    tolerance: float,
    priority_context: RoutingPriorityContext | None = None,
) -> Iteration2RoutingResult:
    return strict_greedy_route(
        graph,
        coordinates,
        source,
        destination,
        poincare_distance,
        method="poincare_greedy",
        tolerance=tolerance,
        priority_context=priority_context,
    )


@scientific_operation_boundary("repaired_poincare_routing")
def repaired_poincare_greedy_route_v2(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedIteration2Routing,
    source: int,
    destination: int,
    *,
    tolerance: float,
    priority_context: RoutingPriorityContext | None = None,
    ordinary_result: Iteration2RoutingResult | None = None,
) -> Iteration2RoutingResult:
    """Apply at most one physical backtrack after an ordinary local minimum."""

    source_id = _integer("source", source)
    destination_id = _integer("destination", destination)
    if source_id == destination_id:
        raise ValueError("source and destination must differ")
    prepared = _route_context(graph, coordinates, poincare_distance)
    numeric_tolerance = _validated_tolerance(tolerance)
    priority = _validated_priority_context(
        graph, source_id, destination_id, priority_context
    )
    if ordinary_result is None:
        initial = poincare_greedy_route_v2(
            graph,
            prepared,
            source_id,
            destination_id,
            tolerance=numeric_tolerance,
            priority_context=priority,
        )
    else:
        initial = ordinary_result
        if (
            not isinstance(initial, Iteration2RoutingResult)
            or initial.method != "poincare_greedy"
            or initial.source != source_id
            or initial.destination != destination_id
            or initial.distance_tolerance != numeric_tolerance
            or initial.priority_context_id != priority.identity
        ):
            raise ValueError("ordinary_result is incompatible with repaired routing")
    counters = _Counters(
        forwarding_decisions=initial.forwarding_decisions,
        logical_distance_evaluations=initial.logical_distance_evaluations,
        peak_history_vertices=initial.peak_history_vertices,
    )
    if initial.success:
        outcome = _Outcome(True, initial.walk, None, None)
        return _make_result(
            method="repaired_poincare_greedy",
            source=source_id,
            destination=destination_id,
            outcome=outcome,
            counters=counters,
            tolerance=numeric_tolerance,
            priority_context=priority,
            initial_failure_type=None,
            initial_diagnostic=None,
            repair_backtrackable=None,
        )
    if initial.final_failure_type != LOCAL_MINIMUM:
        raise Iteration2RoutingInvariantError(
            "repair may follow only an ordinary local minimum"
        )
    if len(initial.walk) < 2:
        final = _diagnostic(
            graph=prepared._graph,
            failure_type=REPAIR_UNAVAILABLE_AT_SOURCE,
            stage="repair_unavailable",
            walk=initial.walk,
            current_distance=(
                None
                if initial.final_failure_diagnostic is None
                else initial.final_failure_diagnostic.current_distance
            ),
            best_distance=(
                None
                if initial.final_failure_diagnostic is None
                else initial.final_failure_diagnostic.best_neighbor_distance
            ),
            tolerance=numeric_tolerance,
        )
        outcome = _Outcome(False, initial.walk, REPAIR_UNAVAILABLE_AT_SOURCE, final)
        return _make_result(
            method="repaired_poincare_greedy",
            source=source_id,
            destination=destination_id,
            outcome=outcome,
            counters=counters,
            tolerance=numeric_tolerance,
            priority_context=priority,
            initial_failure_type=initial.final_failure_type,
            initial_diagnostic=initial.final_failure_diagnostic,
            repair_backtrackable=False,
            repair_eligible=False,
            repair_alternative_existed=False,
        )

    failed_branch = initial.walk[-1]
    predecessor = initial.walk[-2]
    physical_walk = [*initial.walk, predecessor]
    explored = set(initial.walk)
    counters.forwarding_decisions += 1
    counters.peak_history_vertices = max(counters.peak_history_vertices, len(explored))
    alternatives = tuple(
        sorted(
            int(node)
            for node in prepared._graph.neighbors(predecessor)
            if int(node) != failed_branch and int(node) not in explored
        )
    )
    if not alternatives:
        final = _diagnostic(
            graph=prepared._graph,
            failure_type=NO_ALTERNATIVE_AFTER_BACKTRACKING,
            stage="repair_selection",
            walk=physical_walk,
            current_distance=None,
            best_distance=None,
            tolerance=numeric_tolerance,
        )
        outcome = _Outcome(
            False,
            tuple(physical_walk),
            NO_ALTERNATIVE_AFTER_BACKTRACKING,
            final,
        )
        return _make_result(
            method="repaired_poincare_greedy",
            source=source_id,
            destination=destination_id,
            outcome=outcome,
            counters=counters,
            tolerance=numeric_tolerance,
            priority_context=priority,
            initial_failure_type=initial.final_failure_type,
            initial_diagnostic=initial.final_failure_diagnostic,
            repair_attempted=True,
            repair_alternative_existed=False,
            repair_attempt_count=1,
            repair_backtrackable=True,
            repair_eligible=True,
            repair_backtracked_vertex=predecessor,
            repair_excluded_branch=failed_branch,
        )

    if destination_id in alternatives:
        alternative = destination_id
    else:
        ranked = tuple(
            (
                node,
                _metric_distance(prepared, node, destination_id, counters),
            )
            for node in alternatives
        )
        minimum = min(distance for _, distance in ranked)
        tied = tuple(
            node for node, distance in ranked if distance <= minimum + numeric_tolerance
        )
        alternative = _priority_choice(
            tied, current=predecessor, priority_context=priority
        )
    physical_walk.append(alternative)
    explored.add(alternative)
    counters.forwarding_decisions += 1
    counters.peak_history_vertices = max(counters.peak_history_vertices, len(explored))
    if alternative == destination_id:
        repaired = _Outcome(True, tuple(physical_walk), None, None)
    else:
        repaired = _continue(
            prepared=prepared,
            destination=destination_id,
            tolerance=numeric_tolerance,
            priority_context=priority,
            walk=physical_walk,
            visited=explored,
            counters=counters,
            stage="post_repair",
        )
    return _make_result(
        method="repaired_poincare_greedy",
        source=source_id,
        destination=destination_id,
        outcome=repaired,
        counters=counters,
        tolerance=numeric_tolerance,
        priority_context=priority,
        initial_failure_type=initial.final_failure_type,
        initial_diagnostic=initial.final_failure_diagnostic,
        repair_attempted=True,
        repair_succeeded=repaired.success,
        repair_alternative_existed=True,
        repair_attempt_count=1,
        repair_backtrackable=True,
        repair_eligible=True,
        repair_alternative_selected=True,
        repair_backtracked_vertex=predecessor,
        repair_excluded_branch=failed_branch,
        repair_selected_alternative=alternative,
    )
