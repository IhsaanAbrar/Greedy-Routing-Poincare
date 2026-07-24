"""Deterministic shortest-path and local greedy-routing methods.

The two ordinary greedy wrappers share exactly one routing core.  Their only
difference is the distance function used to rank a current vertex's neighbours.
The repaired variant adds one history-only backtracking attempt after an
ordinary hyperbolic ``local_minimum`` or ``attempted_revisit`` failure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any
from weakref import ReferenceType, ref

import networkx as nx

from experiment_config import DEVELOPMENT_CONFIG
from poincare_distance import euclidean_distance, poincare_distance


Coordinate = Sequence[float]
DistanceFunction = Callable[[Coordinate, Coordinate], float]

DIJKSTRA_METHOD = "dijkstra"
EUCLIDEAN_GREEDY_METHOD = "euclidean_greedy"
HYPERBOLIC_GREEDY_METHOD = "hyperbolic_greedy"
REPAIRED_HYPERBOLIC_GREEDY_METHOD = "repaired_hyperbolic_greedy"

LOCAL_MINIMUM = "local_minimum"
ATTEMPTED_REVISIT = "attempted_revisit"
CYCLE = ATTEMPTED_REVISIT
REPAIR_UNAVAILABLE_AT_SOURCE = "repair_unavailable_at_source"
REPAIR_UNAVAILABLE = REPAIR_UNAVAILABLE_AT_SOURCE
NO_ALTERNATIVE_AFTER_BACKTRACKING = "no_alternative_after_backtracking"
REPAIR_FAILED = NO_ALTERNATIVE_AFTER_BACKTRACKING
POST_REPAIR_LOCAL_MINIMUM = "post_repair_local_minimum"
POST_REPAIR_ATTEMPTED_REVISIT = "post_repair_attempted_revisit"

DEFAULT_NUMERICAL_TOLERANCE = DEVELOPMENT_CONFIG.numerical_tolerance


class RoutingInvariantError(RuntimeError):
    """Raised when an experimental graph violates a routing invariant."""


@dataclass(frozen=True)
class PreparedRoutingCoordinates(Mapping[int, Coordinate]):
    """Immutable coordinates validated once for one graph and metric.

    A prepared mapping is bound to the exact graph object and an immutable copy
    of its topology at preparation time, plus the exact distance function used
    for validation.  Routing functions reject use with any other graph object
    or metric and route on the captured topology.  Coordinate values are copied
    into tuples behind a read-only mapping, so later mutations of caller-owned
    inputs cannot affect a prepared run.
    """

    metric_name: str
    topology_nodes: tuple[int, ...]
    topology_edges: tuple[tuple[int, int], ...]
    _coordinates: Mapping[int, Coordinate] = field(repr=False, compare=False)
    _graph_reference: ReferenceType[nx.Graph] = field(repr=False, compare=False)
    _topology_graph: nx.Graph = field(repr=False, compare=False)
    _distance_function: DistanceFunction = field(repr=False, compare=False)

    def __getitem__(self, node: int) -> Coordinate:
        return self._coordinates[node]

    def __iter__(self) -> Iterator[int]:
        return iter(self._coordinates)

    def __len__(self) -> int:
        return len(self._coordinates)

    def require_compatible(
        self,
        graph: nx.Graph,
        distance_function: DistanceFunction,
    ) -> None:
        """Reject reuse with a different graph object or distance metric."""

        if self._graph_reference() is not graph:
            raise ValueError(
                "prepared routing coordinates are bound to a different graph topology"
            )
        if self._distance_function is not distance_function:
            raise ValueError(
                "prepared routing coordinates are bound to a different distance metric"
            )

    def routing_graph(self) -> nx.Graph:
        """Return the private frozen topology snapshot used for routing."""

        return self._topology_graph


@dataclass(frozen=True)
class RoutingResult:
    """Serializable outcome of a benchmark or routing method.

    ``walk`` is the complete sequence of physically traversed vertices and
    ``route_length`` therefore always equals ``len(walk) - 1``, including on a
    failure.  A repair backtrack appears explicitly in ``walk``.

    ``forwarding_decisions`` counts neighbour-selection operations.  It does
    not count Dijkstra exploration or the deterministic backtracking edge.
    """

    method: str
    source: int
    destination: int
    success: bool
    walk: tuple[int, ...]
    route_length: int
    failure_type: str | None
    repair_attempted: bool
    repair_succeeded: bool
    forwarding_decisions: int
    initial_failure_type: str | None = None
    final_failure_type: str | None = None
    repair_alternative_existed: bool | None = None
    repair_attempt_count: int = 0

    def __post_init__(self) -> None:
        for name, node in (
            ("source", self.source),
            ("destination", self.destination),
        ):
            if isinstance(node, bool) or not isinstance(node, Integral):
                raise ValueError(f"{name} must be an integer node ID")

        try:
            walk = tuple(self.walk)
        except TypeError as exc:
            raise ValueError("walk must be an iterable of integer node IDs") from exc
        if any(isinstance(node, bool) or not isinstance(node, Integral) for node in walk):
            raise ValueError("walk must contain only integer node IDs")

        source = int(self.source)
        destination = int(self.destination)
        walk = tuple(int(node) for node in walk)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "walk", walk)

        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        if isinstance(self.route_length, bool) or not isinstance(
            self.route_length, Integral
        ):
            raise ValueError("route_length must be an integer")
        route_length = int(self.route_length)
        object.__setattr__(self, "route_length", route_length)
        if route_length < 0:
            raise ValueError("route_length must be non-negative")

        if not walk or walk[0] != source:
            raise ValueError("walk must be non-empty and begin at source")
        if route_length != len(walk) - 1:
            raise ValueError("route_length must equal len(walk) - 1")
        if isinstance(self.forwarding_decisions, bool) or not isinstance(
            self.forwarding_decisions, Integral
        ):
            raise ValueError("forwarding_decisions must be an integer")
        forwarding_decisions = int(self.forwarding_decisions)
        object.__setattr__(self, "forwarding_decisions", forwarding_decisions)
        if forwarding_decisions < 0:
            raise ValueError("forwarding_decisions must be non-negative")

        if not isinstance(self.repair_attempted, bool):
            raise ValueError("repair_attempted must be a boolean")
        if not isinstance(self.repair_succeeded, bool):
            raise ValueError("repair_succeeded must be a boolean")
        if (
            self.repair_alternative_existed is not None
            and not isinstance(self.repair_alternative_existed, bool)
        ):
            raise ValueError("repair_alternative_existed must be a boolean or None")
        if isinstance(self.repair_attempt_count, bool) or not isinstance(
            self.repair_attempt_count, Integral
        ):
            raise ValueError("repair_attempt_count must be an integer")
        repair_attempt_count = int(self.repair_attempt_count)
        object.__setattr__(self, "repair_attempt_count", repair_attempt_count)
        if repair_attempt_count not in (0, 1):
            raise ValueError("repair_attempt_count must be zero or one")
        if self.repair_attempted != (repair_attempt_count == 1):
            raise ValueError(
                "repair_attempted must agree with repair_attempt_count"
            )

        for name, failure_type in (
            ("failure_type", self.failure_type),
            ("initial_failure_type", self.initial_failure_type),
            ("final_failure_type", self.final_failure_type),
        ):
            if failure_type is not None and (
                not isinstance(failure_type, str) or not failure_type.strip()
            ):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.failure_type != self.final_failure_type:
            raise ValueError("failure_type and final_failure_type must agree")

        if self.success:
            if walk[-1] != destination:
                raise ValueError("a successful walk must end at destination")
            if self.failure_type is not None:
                raise ValueError("a successful result cannot have a final failure")
        else:
            if self.failure_type is None:
                raise ValueError("a failed result must record a failure type")
            if walk[-1] == destination:
                raise ValueError("a failed walk cannot end at destination")

        if self.repair_succeeded != (self.repair_attempted and self.success):
            raise ValueError(
                "repair_succeeded must be true exactly when an attempted repair delivers"
            )

        alternative_existed = self.repair_alternative_existed
        if self.repair_attempted:
            if self.initial_failure_type not in (LOCAL_MINIMUM, CYCLE):
                raise ValueError(
                    "an attempted repair must follow a local minimum or "
                    "attempted revisit"
                )
            if alternative_existed is None:
                raise ValueError(
                    "an attempted repair must record whether an alternative existed"
                )
            if alternative_existed:
                if not self.success and self.final_failure_type not in (
                    POST_REPAIR_LOCAL_MINIMUM,
                    POST_REPAIR_ATTEMPTED_REVISIT,
                ):
                    raise ValueError(
                        "a failed repair with an alternative must record its terminal "
                        "greedy failure"
                    )
            elif self.final_failure_type != REPAIR_FAILED:
                raise ValueError(
                    "a repair without an alternative must record no alternative"
                )
        else:
            if alternative_existed is True:
                raise ValueError(
                    "an unattempted repair cannot report that an alternative existed"
                )
            if alternative_existed is False:
                if (
                    self.success
                    or self.final_failure_type != REPAIR_UNAVAILABLE
                    or self.initial_failure_type not in (LOCAL_MINIMUM, CYCLE)
                ):
                    raise ValueError(
                        "an unavailable repair must follow a source-level local "
                        "minimum or attempted revisit"
                    )
            elif self.success:
                if self.initial_failure_type is not None:
                    raise ValueError(
                        "a successful route without repair cannot have an initial failure"
                    )
            elif self.initial_failure_type != self.final_failure_type:
                raise ValueError(
                    "an unrepaired failure must have identical initial and final types"
                )

        if self.final_failure_type == REPAIR_UNAVAILABLE and not (
            not self.repair_attempted and alternative_existed is False
        ):
            raise ValueError(
                "repair_unavailable_at_source requires an unattempted source repair"
            )
        if self.final_failure_type == REPAIR_FAILED and not (
            self.repair_attempted and alternative_existed is False
        ):
            raise ValueError(
                "no_alternative_after_backtracking requires an attempted repair "
                "with no alternative"
            )

    @property
    def route(self) -> tuple[int, ...]:
        """Alias retained for callers that use the paper's route terminology."""

        return self.walk

    @property
    def total_traversed_edges(self) -> int:
        """Return the complete physical-walk length."""

        return self.route_length

    @property
    def alternative_existed(self) -> bool | None:
        """Concise alias for the repair-alternative status."""

        return self.repair_alternative_existed

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary with stable field names."""

        return {
            "method": self.method,
            "source": self.source,
            "destination": self.destination,
            "success": self.success,
            "walk": list(self.walk),
            "route_length": self.route_length,
            "failure_type": self.failure_type,
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "forwarding_decisions": self.forwarding_decisions,
            "initial_failure_type": self.initial_failure_type,
            "final_failure_type": self.final_failure_type,
            "repair_alternative_existed": self.repair_alternative_existed,
            "repair_attempt_count": self.repair_attempt_count,
        }


@dataclass(frozen=True)
class _GreedyOutcome:
    success: bool
    walk: tuple[int, ...]
    failure_type: str | None
    forwarding_decisions: int


def _validate_integer_node_ids(graph: nx.Graph) -> None:
    invalid = [
        node
        for node in graph.nodes
        if isinstance(node, bool) or not isinstance(node, Integral)
    ]
    if invalid:
        raise ValueError(
            "routing requires integer node IDs for deterministic tie-breaking"
        )


def _validate_routing_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise ValueError("graph must be an undirected NetworkX graph")
    _validate_integer_node_ids(graph)


def _validate_graph_and_nodes(
    graph: nx.Graph, source: int, destination: int
) -> None:
    _validate_routing_graph(graph)
    _validate_source_and_destination(graph, source, destination)


def _validate_source_and_destination(
    graph: nx.Graph,
    source: int,
    destination: int,
) -> None:
    for name, node in (("source", source), ("destination", destination)):
        if isinstance(node, bool) or not isinstance(node, Integral):
            raise ValueError(f"{name} must be an integer node ID")
    if source not in graph:
        raise ValueError(f"source node {source!r} is not in the graph")
    if destination not in graph:
        raise ValueError(
            f"destination node {destination!r} is not in the graph"
        )


def _validate_tolerance(tolerance: float | None) -> float:
    value = DEFAULT_NUMERICAL_TOLERANCE if tolerance is None else tolerance
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("tolerance must be numeric")
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    return value


def _validate_step_limit(
    graph: nx.Graph, step_limit: int | None, *, repair_allowed: bool
) -> int:
    if step_limit is None:
        multiplier = 2 if repair_allowed else 1
        return max(1, multiplier * graph.number_of_nodes())
    if isinstance(step_limit, bool) or not isinstance(step_limit, int):
        raise ValueError("step_limit must be an integer")
    if step_limit <= 0:
        raise ValueError("step_limit must be positive")
    return step_limit


def _checked_distance(
    distance_function: DistanceFunction,
    first: Coordinate,
    second: Coordinate,
) -> float:
    try:
        value = distance_function(first, second)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"invalid coordinate or distance value: {exc}") from exc
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("distance function must return a real number")
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError("distance function must return a finite non-negative value")
    return value


def _validate_coordinate_coverage(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate],
    distance_function: DistanceFunction,
) -> None:
    if not isinstance(coordinates, Mapping):
        raise ValueError("coordinates must be a node-to-coordinate mapping")
    missing = [node for node in graph.nodes if node not in coordinates]
    if missing:
        raise ValueError(f"coordinates are missing graph nodes: {missing}")

    # Self-distance calls force the selected metric to validate every supplied
    # graph coordinate before any routing decision is made.
    for node in graph.nodes:
        _checked_distance(
            distance_function, coordinates[node], coordinates[node]
        )


def _topology_signature(
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


def _default_metric_name(distance_function: DistanceFunction) -> str:
    module = getattr(distance_function, "__module__", "")
    qualified_name = getattr(distance_function, "__qualname__", "")
    if qualified_name:
        return f"{module}.{qualified_name}" if module else qualified_name
    return type(distance_function).__qualname__


def prepare_routing_coordinates(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedRoutingCoordinates,
    distance_function: DistanceFunction,
    *,
    metric_name: str | None = None,
) -> PreparedRoutingCoordinates:
    """Validate and snapshot coordinates for repeated routing on one graph.

    Raw coordinate mappings retain the existing per-call validation behaviour.
    This explicit preparation step is intended for experiment runners that route
    many pairs on one immutable graph.  A Poincare context may be shared by the
    ordinary and repaired hyperbolic methods because both use the same metric.
    """

    _validate_routing_graph(graph)
    if not callable(distance_function):
        raise ValueError("distance_function must be callable")
    if metric_name is not None and (
        not isinstance(metric_name, str) or not metric_name.strip()
    ):
        raise ValueError("metric_name must be a non-empty string or None")

    if isinstance(coordinates, PreparedRoutingCoordinates):
        coordinates.require_compatible(graph, distance_function)
        return coordinates
    if not isinstance(coordinates, Mapping):
        raise ValueError("coordinates must be a node-to-coordinate mapping")

    missing = [node for node in graph.nodes if node not in coordinates]
    if missing:
        raise ValueError(f"coordinates are missing graph nodes: {missing}")

    snapshot: dict[int, Coordinate] = {}
    for node in graph.nodes:
        try:
            snapshot[node] = tuple(coordinates[node])
        except TypeError as exc:
            raise ValueError(
                f"coordinate for node {node!r} must be an immutable sequence"
            ) from exc

    _validate_coordinate_coverage(graph, snapshot, distance_function)
    topology_nodes, topology_edges = _topology_signature(graph)
    topology_graph = nx.Graph()
    topology_graph.add_nodes_from(graph.nodes)
    topology_graph.add_edges_from(graph.edges)
    nx.freeze(topology_graph)
    return PreparedRoutingCoordinates(
        metric_name=(
            _default_metric_name(distance_function)
            if metric_name is None
            else metric_name.strip()
        ),
        topology_nodes=topology_nodes,
        topology_edges=topology_edges,
        _coordinates=MappingProxyType(snapshot),
        _graph_reference=ref(graph),
        _topology_graph=topology_graph,
        _distance_function=distance_function,
    )


def _routing_inputs(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedRoutingCoordinates,
    distance_function: DistanceFunction,
    source: int,
    destination: int,
) -> tuple[nx.Graph, Mapping[int, Coordinate]]:
    if isinstance(coordinates, PreparedRoutingCoordinates):
        coordinates.require_compatible(graph, distance_function)
        routing_graph = coordinates.routing_graph()
        _validate_source_and_destination(routing_graph, source, destination)
        return routing_graph, coordinates

    _validate_graph_and_nodes(graph, source, destination)
    _validate_coordinate_coverage(graph, coordinates, distance_function)
    return graph, coordinates


def _validate_walk_edges(graph: nx.Graph, walk: Sequence[int]) -> None:
    for first, second in zip(walk, walk[1:]):
        if not graph.has_edge(first, second):
            raise RoutingInvariantError(
                f"recorded walk contains non-edge ({first!r}, {second!r})"
            )


def _select_best_neighbour(
    graph: nx.Graph,
    current: int,
    destination: int,
    coordinates: Mapping[int, Coordinate],
    distance_function: DistanceFunction,
    tolerance: float,
    *,
    candidates: Sequence[int] | None = None,
) -> tuple[int, float] | None:
    neighbours = (
        list(graph.neighbors(current))
        if candidates is None
        else list(candidates)
    )
    if not neighbours:
        return None

    ranked = [
        (
            neighbour,
            _checked_distance(
                distance_function,
                coordinates[neighbour],
                coordinates[destination],
            ),
        )
        for neighbour in neighbours
    ]
    minimum_distance = min(distance for _, distance in ranked)
    tied = [
        (node, distance)
        for node, distance in ranked
        if distance <= minimum_distance + tolerance
    ]
    return min(tied, key=lambda item: int(item[0]))


def _continue_greedy_walk(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate],
    destination: int,
    distance_function: DistanceFunction,
    tolerance: float,
    step_limit: int,
    walk: list[int],
    visited: set[int],
    forwarding_decisions: int,
) -> _GreedyOutcome:
    while walk[-1] != destination:
        if forwarding_decisions >= step_limit:
            raise RoutingInvariantError(
                "defensive routing step limit reached; this indicates an "
                "implementation invariant failure"
            )

        current = walk[-1]
        selected = _select_best_neighbour(
            graph,
            current,
            destination,
            coordinates,
            distance_function,
            tolerance,
        )
        if selected is None:
            return _GreedyOutcome(
                False, tuple(walk), LOCAL_MINIMUM, forwarding_decisions
            )

        selected_node, selected_distance = selected
        forwarding_decisions += 1

        # The attempted revisit is classified before geometric progress, as
        # specified in the experimental stopping rule.  It is not traversed.
        if selected_node in visited:
            return _GreedyOutcome(
                False, tuple(walk), CYCLE, forwarding_decisions
            )

        current_distance = _checked_distance(
            distance_function,
            coordinates[current],
            coordinates[destination],
        )
        if not selected_distance < current_distance - tolerance:
            return _GreedyOutcome(
                False, tuple(walk), LOCAL_MINIMUM, forwarding_decisions
            )

        if not graph.has_edge(current, selected_node):
            raise RoutingInvariantError(
                "selected neighbour is not joined to the current vertex"
            )
        walk.append(selected_node)
        visited.add(selected_node)

    return _GreedyOutcome(True, tuple(walk), None, forwarding_decisions)


def _result_from_outcome(
    graph: nx.Graph,
    *,
    method: str,
    source: int,
    destination: int,
    outcome: _GreedyOutcome,
) -> RoutingResult:
    _validate_walk_edges(graph, outcome.walk)
    return RoutingResult(
        method=method,
        source=source,
        destination=destination,
        success=outcome.success,
        walk=outcome.walk,
        route_length=len(outcome.walk) - 1,
        failure_type=outcome.failure_type,
        repair_attempted=False,
        repair_succeeded=False,
        forwarding_decisions=outcome.forwarding_decisions,
        initial_failure_type=outcome.failure_type,
        final_failure_type=outcome.failure_type,
        repair_alternative_existed=None,
        repair_attempt_count=0,
    )


def dijkstra_benchmark(
    graph: nx.Graph,
    source: int,
    destination: int,
    *,
    expected_shortest_path_length: int | None = None,
) -> RoutingResult:
    """Return NetworkX's unweighted Dijkstra path as the global benchmark."""

    _validate_graph_and_nodes(graph, source, destination)
    try:
        path = tuple(
            nx.dijkstra_path(
                graph, source=source, target=destination, weight=None
            )
        )
    except nx.NetworkXNoPath as exc:
        raise RoutingInvariantError(
            f"no path exists between {source!r} and {destination!r}"
        ) from exc

    route_length = len(path) - 1
    if expected_shortest_path_length is not None:
        if (
            isinstance(expected_shortest_path_length, bool)
            or not isinstance(expected_shortest_path_length, Integral)
            or int(expected_shortest_path_length) < 0
        ):
            raise ValueError(
                "expected_shortest_path_length must be a non-negative integer"
            )
        if route_length != int(expected_shortest_path_length):
            raise RoutingInvariantError(
                "Dijkstra route length disagrees with the prepared APSP matrix"
            )
    _validate_walk_edges(graph, path)
    return RoutingResult(
        method=DIJKSTRA_METHOD,
        source=source,
        destination=destination,
        success=True,
        walk=path,
        route_length=route_length,
        failure_type=None,
        repair_attempted=False,
        repair_succeeded=False,
        forwarding_decisions=0,
        initial_failure_type=None,
        final_failure_type=None,
        repair_alternative_existed=None,
        repair_attempt_count=0,
    )


def greedy_route(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedRoutingCoordinates,
    source: int,
    destination: int,
    distance_function: DistanceFunction,
    *,
    method_name: str = "greedy",
    tolerance: float | None = None,
    step_limit: int | None = None,
) -> RoutingResult:
    """Route locally using the supplied metric and deterministic stopping rules."""

    if not callable(distance_function):
        raise ValueError("distance_function must be callable")
    routing_graph, routing_coordinates = _routing_inputs(
        graph,
        coordinates,
        distance_function,
        source,
        destination,
    )
    numeric_tolerance = _validate_tolerance(tolerance)
    maximum_steps = _validate_step_limit(
        routing_graph,
        step_limit,
        repair_allowed=False,
    )

    outcome = _continue_greedy_walk(
        routing_graph,
        routing_coordinates,
        destination,
        distance_function,
        numeric_tolerance,
        maximum_steps,
        [source],
        {source},
        0,
    )
    return _result_from_outcome(
        routing_graph,
        method=method_name,
        source=source,
        destination=destination,
        outcome=outcome,
    )


def euclidean_greedy_route(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedRoutingCoordinates,
    source: int,
    destination: int,
    *,
    tolerance: float | None = None,
    step_limit: int | None = None,
) -> RoutingResult:
    """Apply ordinary greedy routing using Euclidean distance."""

    return greedy_route(
        graph,
        coordinates,
        source,
        destination,
        euclidean_distance,
        method_name=EUCLIDEAN_GREEDY_METHOD,
        tolerance=tolerance,
        step_limit=step_limit,
    )


def hyperbolic_greedy_route(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedRoutingCoordinates,
    source: int,
    destination: int,
    *,
    tolerance: float | None = None,
    step_limit: int | None = None,
) -> RoutingResult:
    """Apply the shared greedy core using Poincare disk distance."""

    return greedy_route(
        graph,
        coordinates,
        source,
        destination,
        poincare_distance,
        method_name=HYPERBOLIC_GREEDY_METHOD,
        tolerance=tolerance,
        step_limit=step_limit,
    )


def _repaired_result(
    graph: nx.Graph,
    *,
    source: int,
    destination: int,
    success: bool,
    walk: Sequence[int],
    final_failure_type: str | None,
    forwarding_decisions: int,
    initial_failure_type: str | None,
    repair_attempted: bool,
    repair_succeeded: bool,
    repair_alternative_existed: bool | None,
) -> RoutingResult:
    walk_tuple = tuple(walk)
    _validate_walk_edges(graph, walk_tuple)
    return RoutingResult(
        method=REPAIRED_HYPERBOLIC_GREEDY_METHOD,
        source=source,
        destination=destination,
        success=success,
        walk=walk_tuple,
        route_length=len(walk_tuple) - 1,
        failure_type=final_failure_type,
        repair_attempted=repair_attempted,
        repair_succeeded=repair_succeeded,
        forwarding_decisions=forwarding_decisions,
        initial_failure_type=initial_failure_type,
        final_failure_type=final_failure_type,
        repair_alternative_existed=repair_alternative_existed,
        repair_attempt_count=1 if repair_attempted else 0,
    )


def repaired_hyperbolic_greedy_route(
    graph: nx.Graph,
    coordinates: Mapping[int, Coordinate] | PreparedRoutingCoordinates,
    source: int,
    destination: int,
    *,
    tolerance: float | None = None,
    step_limit: int | None = None,
) -> RoutingResult:
    """Apply hyperbolic greedy routing with at most one local repair.

    Repair is triggered only by the first ordinary ``local_minimum`` or
    ``attempted_revisit``. The walk backtracks one physical edge, excludes the failed
    branch and all explored vertices, chooses the best remaining neighbour by
    Poincare distance (with node-ID tie-breaking), and then resumes the strict
    ordinary greedy rule.  No graph-distance or shortest-path information is
    consulted.
    """

    routing_graph, routing_coordinates = _routing_inputs(
        graph,
        coordinates,
        poincare_distance,
        source,
        destination,
    )
    numeric_tolerance = _validate_tolerance(tolerance)
    maximum_steps = _validate_step_limit(
        routing_graph,
        step_limit,
        repair_allowed=True,
    )

    initial = _continue_greedy_walk(
        routing_graph,
        routing_coordinates,
        destination,
        poincare_distance,
        numeric_tolerance,
        maximum_steps,
        [source],
        {source},
        0,
    )

    if initial.success:
        return _repaired_result(
            routing_graph,
            source=source,
            destination=destination,
            success=True,
            walk=initial.walk,
            final_failure_type=None,
            forwarding_decisions=initial.forwarding_decisions,
            initial_failure_type=None,
            repair_attempted=False,
            repair_succeeded=False,
            repair_alternative_existed=None,
        )

    if initial.failure_type not in (LOCAL_MINIMUM, CYCLE):
        return _repaired_result(
            routing_graph,
            source=source,
            destination=destination,
            success=False,
            walk=initial.walk,
            final_failure_type=initial.failure_type,
            forwarding_decisions=initial.forwarding_decisions,
            initial_failure_type=initial.failure_type,
            repair_attempted=False,
            repair_succeeded=False,
            repair_alternative_existed=None,
        )

    # At the source there is no physical edge over which to backtrack.
    if len(initial.walk) < 2:
        return _repaired_result(
            routing_graph,
            source=source,
            destination=destination,
            success=False,
            walk=initial.walk,
            final_failure_type=REPAIR_UNAVAILABLE,
            forwarding_decisions=initial.forwarding_decisions,
            initial_failure_type=initial.failure_type,
            repair_attempted=False,
            repair_succeeded=False,
            repair_alternative_existed=False,
        )

    failed_branch = initial.walk[-1]
    preceding_vertex = initial.walk[-2]
    physical_walk = list(initial.walk)
    if not routing_graph.has_edge(failed_branch, preceding_vertex):
        raise RoutingInvariantError("repair backtrack is not a graph edge")
    physical_walk.append(preceding_vertex)

    explored = set(initial.walk)
    alternatives = [
        neighbour
        for neighbour in routing_graph.neighbors(preceding_vertex)
        if neighbour != failed_branch and neighbour not in explored
    ]
    if not alternatives:
        return _repaired_result(
            routing_graph,
            source=source,
            destination=destination,
            success=False,
            walk=physical_walk,
            final_failure_type=REPAIR_FAILED,
            forwarding_decisions=initial.forwarding_decisions,
            initial_failure_type=initial.failure_type,
            repair_attempted=True,
            repair_succeeded=False,
            repair_alternative_existed=False,
        )

    if initial.forwarding_decisions >= maximum_steps:
        raise RoutingInvariantError(
            "defensive repaired-routing step limit reached; this indicates "
            "an implementation invariant failure"
        )

    selected = _select_best_neighbour(
        routing_graph,
        preceding_vertex,
        destination,
        routing_coordinates,
        poincare_distance,
        numeric_tolerance,
        candidates=alternatives,
    )
    if selected is None:  # Defensive: alternatives is known to be non-empty.
        raise RoutingInvariantError("repair alternatives disappeared")
    alternative, _ = selected
    forwarding_decisions = initial.forwarding_decisions + 1

    # This one alternative move is deliberately allowed without a strict
    # progress check so that repair can escape the failed branch.
    physical_walk.append(alternative)
    explored.add(alternative)

    repaired = _continue_greedy_walk(
        routing_graph,
        routing_coordinates,
        destination,
        poincare_distance,
        numeric_tolerance,
        maximum_steps,
        physical_walk,
        explored,
        forwarding_decisions,
    )
    final_failure_type = repaired.failure_type
    if final_failure_type == LOCAL_MINIMUM:
        final_failure_type = POST_REPAIR_LOCAL_MINIMUM
    elif final_failure_type == CYCLE:
        final_failure_type = POST_REPAIR_ATTEMPTED_REVISIT
    return _repaired_result(
        routing_graph,
        source=source,
        destination=destination,
        success=repaired.success,
        walk=repaired.walk,
        final_failure_type=final_failure_type,
        forwarding_decisions=repaired.forwarding_decisions,
        initial_failure_type=initial.failure_type,
        repair_attempted=True,
        repair_succeeded=repaired.success,
        repair_alternative_existed=True,
    )


# Readable aliases for callers that use "routing" rather than "route".
euclidean_greedy_routing = euclidean_greedy_route
hyperbolic_greedy_routing = hyperbolic_greedy_route
repaired_hyperbolic_greedy_routing = repaired_hyperbolic_greedy_route
