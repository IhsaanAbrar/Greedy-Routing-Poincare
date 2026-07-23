from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from math import hypot
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from embedding import embed_graph_in_poincare_disk  # noqa: E402
from graph_generation import (  # noqa: E402
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
)
from poincare_distance import euclidean_distance, poincare_distance  # noqa: E402
from routing import (  # noqa: E402
    CYCLE,
    LOCAL_MINIMUM,
    REPAIR_FAILED,
    REPAIR_UNAVAILABLE,
    STEP_LIMIT,
    PreparedRoutingCoordinates,
    RoutingInvariantError,
    RoutingResult,
    dijkstra_benchmark,
    euclidean_greedy_route,
    greedy_route,
    hyperbolic_greedy_route,
    prepare_routing_coordinates,
    repaired_hyperbolic_greedy_route,
)


class RoutingTestCase(unittest.TestCase):
    def assert_walk_uses_edges(
        self, graph: nx.Graph, walk: tuple[int, ...]
    ) -> None:
        for first, second in zip(walk, walk[1:]):
            self.assertTrue(
                graph.has_edge(first, second),
                msg=f"({first}, {second}) is not a graph edge",
            )

    def assert_result_invariants(self, graph: nx.Graph, result) -> None:
        self.assertEqual(result.walk[0], result.source)
        self.assertEqual(result.route_length, len(result.walk) - 1)
        self.assertEqual(result.failure_type, result.final_failure_type)
        self.assertEqual(
            result.repair_attempted,
            result.repair_attempt_count == 1,
        )
        self.assertEqual(
            result.repair_succeeded,
            result.repair_attempted and result.success,
        )
        self.assert_walk_uses_edges(graph, result.walk)
        if result.success:
            self.assertEqual(result.walk[-1], result.destination)
            self.assertIsNone(result.failure_type)
        else:
            self.assertIsNotNone(result.failure_type)


class DijkstraBenchmarkTests(RoutingTestCase):
    def test_unique_shortest_path_and_serializable_result(self):
        graph = nx.path_graph(4)

        result = dijkstra_benchmark(graph, 0, 3)

        self.assertTrue(result.success)
        self.assertEqual(result.walk, (0, 1, 2, 3))
        self.assertEqual(result.route_length, 3)
        self.assertEqual(result.forwarding_decisions, 0)
        self.assertEqual(result.repair_attempt_count, 0)
        json.dumps(result.to_dict())
        self.assert_result_invariants(graph, result)

    def test_equal_shortest_paths_require_only_optimal_length(self):
        graph = nx.cycle_graph(4)

        result = dijkstra_benchmark(graph, 0, 2)

        self.assertEqual(result.route_length, 2)
        self.assert_result_invariants(graph, result)

    def test_source_equal_to_destination_has_zero_length(self):
        graph = nx.path_graph(3)

        result = dijkstra_benchmark(graph, 1, 1)

        self.assertEqual(result.walk, (1,))
        self.assertEqual(result.route_length, 0)
        self.assert_result_invariants(graph, result)

    def test_missing_nodes_are_rejected(self):
        graph = nx.path_graph(3)

        for source, destination in ((9, 1), (1, 9)):
            with self.subTest(source=source, destination=destination):
                with self.assertRaises(ValueError):
                    dijkstra_benchmark(graph, source, destination)

    def test_disconnected_pair_raises_invariant_error(self):
        graph = nx.Graph([(0, 1), (2, 3)])

        with self.assertRaisesRegex(RoutingInvariantError, "no path exists"):
            dijkstra_benchmark(graph, 0, 3)

    def test_path_length_is_derived_without_a_second_dijkstra_search(self):
        graph = nx.path_graph(5)
        original_dijkstra_path = nx.dijkstra_path

        with patch(
            "routing.nx.dijkstra_path",
            wraps=original_dijkstra_path,
        ) as path_search, patch(
            "routing.nx.dijkstra_path_length",
            side_effect=AssertionError("duplicate Dijkstra search"),
        ) as length_search:
            result = dijkstra_benchmark(graph, 0, 4)

        self.assertEqual(result.route_length, 4)
        path_search.assert_called_once_with(
            graph,
            source=0,
            target=4,
            weight=None,
        )
        length_search.assert_not_called()

    def test_boolean_and_non_integral_node_aliases_are_rejected(self):
        graph = nx.path_graph(3)

        for invalid in (False, True, 0.0, 1.0):
            for source, destination in ((invalid, 2), (0, invalid)):
                with self.subTest(
                    source=source,
                    destination=destination,
                ), self.assertRaisesRegex(ValueError, "integer node ID"):
                    dijkstra_benchmark(graph, source, destination)


class RoutingResultValidationTests(unittest.TestCase):
    @staticmethod
    def successful_result(**overrides):
        values = {
            "method": "fixture",
            "source": 0,
            "destination": 1,
            "success": True,
            "walk": (0, 1),
            "route_length": 1,
            "failure_type": None,
            "repair_attempted": False,
            "repair_succeeded": False,
            "forwarding_decisions": 1,
            "initial_failure_type": None,
            "final_failure_type": None,
            "repair_alternative_existed": None,
            "repair_attempt_count": 0,
        }
        values.update(overrides)
        return RoutingResult(**values)

    def test_boolean_and_non_integral_result_fields_are_rejected(self):
        invalid_overrides = (
            {"source": True},
            {"destination": 1.0},
            {"walk": (0, 1.0)},
            {"success": 1},
            {"route_length": True},
            {"repair_attempted": 0},
            {"repair_succeeded": 0},
            {"repair_alternative_existed": 1},
            {"repair_attempt_count": True},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.successful_result(**overrides)

    def test_repair_state_fields_must_form_a_consistent_state(self):
        base = self.successful_result()

        invalid_changes = (
            {"repair_alternative_existed": True},
            {
                "repair_attempted": True,
                "repair_succeeded": True,
                "repair_attempt_count": 1,
                "initial_failure_type": CYCLE,
                "repair_alternative_existed": None,
            },
            {
                "repair_attempted": True,
                "repair_succeeded": False,
                "repair_attempt_count": 1,
                "initial_failure_type": CYCLE,
                "repair_alternative_existed": True,
            },
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(base, **changes)

        failed_without_initial_type = {
            "success": False,
            "walk": (0,),
            "route_length": 0,
            "failure_type": LOCAL_MINIMUM,
            "final_failure_type": LOCAL_MINIMUM,
        }
        with self.assertRaisesRegex(ValueError, "initial and final"):
            replace(base, **failed_without_initial_type)


class PreparedRoutingCoordinatesTests(unittest.TestCase):
    @staticmethod
    def repair_fixture():
        graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
        coordinates = {
            0: [-0.8, 0.0],
            1: [-0.2, 0.0],
            2: [0.65, 0.0],
            3: [-0.5, 0.4],
            4: [0.85, 0.0],
        }
        return graph, coordinates

    def test_prepared_and_raw_routes_have_identical_semantics(self):
        graph, coordinates = self.repair_fixture()
        raw_results = (
            euclidean_greedy_route(graph, coordinates, 0, 4),
            hyperbolic_greedy_route(graph, coordinates, 0, 4),
            repaired_hyperbolic_greedy_route(graph, coordinates, 0, 4),
        )

        euclidean_context = prepare_routing_coordinates(
            graph,
            coordinates,
            euclidean_distance,
            metric_name="euclidean",
        )
        poincare_context = prepare_routing_coordinates(
            graph,
            coordinates,
            poincare_distance,
            metric_name="poincare",
        )
        prepared_results = (
            euclidean_greedy_route(graph, euclidean_context, 0, 4),
            hyperbolic_greedy_route(graph, poincare_context, 0, 4),
            repaired_hyperbolic_greedy_route(
                graph,
                poincare_context,
                0,
                4,
            ),
        )

        self.assertIsInstance(
            euclidean_context,
            PreparedRoutingCoordinates,
        )
        self.assertEqual(prepared_results, raw_results)
        self.assertIs(
            prepare_routing_coordinates(
                graph,
                poincare_context,
                poincare_distance,
            ),
            poincare_context,
        )

    def test_prepared_context_rejects_graph_and_metric_mismatches(self):
        graph, coordinates = self.repair_fixture()
        same_topology_different_graph = graph.copy()
        euclidean_context = prepare_routing_coordinates(
            graph,
            coordinates,
            euclidean_distance,
        )
        poincare_context = prepare_routing_coordinates(
            graph,
            coordinates,
            poincare_distance,
        )

        with self.assertRaisesRegex(ValueError, "different graph topology"):
            euclidean_greedy_route(
                same_topology_different_graph,
                euclidean_context,
                0,
                4,
            )
        with self.assertRaisesRegex(ValueError, "different distance metric"):
            hyperbolic_greedy_route(graph, euclidean_context, 0, 4)
        with self.assertRaisesRegex(ValueError, "different distance metric"):
            euclidean_greedy_route(graph, poincare_context, 0, 4)
        with self.assertRaisesRegex(ValueError, "different distance metric"):
            repaired_hyperbolic_greedy_route(
                graph,
                euclidean_context,
                0,
                4,
            )

    def test_preparation_snapshots_coordinates_without_mutating_inputs(self):
        graph, coordinates = self.repair_fixture()
        graph_before = deepcopy(graph)
        coordinates_before = deepcopy(coordinates)
        context = prepare_routing_coordinates(
            graph,
            coordinates,
            poincare_distance,
        )
        baseline = repaired_hyperbolic_greedy_route(
            graph,
            context,
            0,
            4,
        )

        self.assertTrue(nx.utils.graphs_equal(graph, graph_before))
        self.assertEqual(coordinates, coordinates_before)
        coordinates[0][0] = 0.7
        coordinates[1] = [0.1, 0.1]
        coordinates[99] = [0.0, 0.0]

        self.assertEqual(context[0], (-0.8, 0.0))
        self.assertEqual(context[1], (-0.2, 0.0))
        self.assertNotIn(99, context)
        with self.assertRaises(TypeError):
            context[0] = (0.0, 0.0)
        with self.assertRaises(TypeError):
            context[0][0] = 0.0

        repeated = repaired_hyperbolic_greedy_route(
            graph,
            context,
            0,
            4,
        )
        self.assertEqual(repeated, baseline)
        self.assertTrue(nx.utils.graphs_equal(graph, graph_before))

    def test_prepared_context_routes_on_its_captured_topology(self):
        graph = nx.path_graph(3)
        coordinates = {
            0: (-0.6, 0.0),
            1: (0.0, 0.0),
            2: (0.6, 0.0),
        }
        context = prepare_routing_coordinates(
            graph,
            coordinates,
            euclidean_distance,
        )
        prepared_before_mutation = euclidean_greedy_route(
            graph,
            context,
            0,
            2,
        )

        graph.add_edge(0, 2)
        prepared_after_mutation = euclidean_greedy_route(
            graph,
            context,
            0,
            2,
        )
        raw_after_mutation = euclidean_greedy_route(
            graph,
            coordinates,
            0,
            2,
        )

        self.assertEqual(prepared_before_mutation.walk, (0, 1, 2))
        self.assertEqual(prepared_after_mutation, prepared_before_mutation)
        self.assertEqual(raw_after_mutation.walk, (0, 2))

    def test_prepared_context_avoids_repeated_whole_graph_validation(self):
        graph = nx.path_graph(4)
        coordinates = {
            0: (-0.6, 0.0),
            1: (-0.2, 0.0),
            2: (0.2, 0.0),
            3: (0.6, 0.0),
        }
        calls = []

        def counted_euclidean_distance(first, second):
            calls.append((tuple(first), tuple(second)))
            return euclidean_distance(first, second)

        context = prepare_routing_coordinates(
            graph,
            coordinates,
            counted_euclidean_distance,
        )
        preparation_call_count = len(calls)
        self.assertEqual(preparation_call_count, graph.number_of_nodes())

        for node in graph:
            result = greedy_route(
                graph,
                context,
                node,
                node,
                counted_euclidean_distance,
            )
            self.assertTrue(result.success)
        self.assertEqual(len(calls), preparation_call_count)

        for node in graph:
            greedy_route(
                graph,
                coordinates,
                node,
                node,
                counted_euclidean_distance,
            )
        self.assertEqual(
            len(calls),
            preparation_call_count + graph.number_of_nodes() ** 2,
        )


class EuclideanGreedyTests(RoutingTestCase):
    def test_direct_success(self):
        graph = nx.Graph([(0, 1)])
        coordinates = {0: (0.0, 0.0), 1: (0.5, 0.0)}

        result = euclidean_greedy_route(graph, coordinates, 0, 1)

        self.assertTrue(result.success)
        self.assertEqual(result.walk, (0, 1))
        self.assertEqual(result.route_length, 1)
        self.assertEqual(result.forwarding_decisions, 1)
        self.assert_result_invariants(graph, result)

    def test_multi_step_success(self):
        graph = nx.path_graph(4)
        coordinates = {
            0: (-0.6, 0.0),
            1: (-0.2, 0.0),
            2: (0.2, 0.0),
            3: (0.6, 0.0),
        }

        result = euclidean_greedy_route(graph, coordinates, 0, 3)

        self.assertEqual(result.walk, (0, 1, 2, 3))
        self.assertEqual(result.route_length, 3)
        self.assert_result_invariants(graph, result)

    def test_local_minimum_can_fail_despite_existing_graph_path(self):
        graph = nx.path_graph(3)
        coordinates = {0: (0.7, 0.0), 1: (-0.8, 0.0), 2: (0.8, 0.0)}

        result = euclidean_greedy_route(graph, coordinates, 0, 2)

        self.assertFalse(result.success)
        self.assertEqual(result.walk, (0,))
        self.assertEqual(result.failure_type, LOCAL_MINIMUM)
        self.assertTrue(nx.has_path(graph, 0, 2))
        self.assert_result_invariants(graph, result)

    def test_ties_choose_smallest_integer_node_id(self):
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        coordinates = {
            0: (0.0, 0.8),
            1: (-0.2, 0.0),
            2: (0.2, 0.0),
            3: (0.0, 0.0),
        }

        result = euclidean_greedy_route(graph, coordinates, 0, 3)

        self.assertEqual(result.walk, (0, 1, 3))
        self.assert_result_invariants(graph, result)

    def test_tolerance_aware_tie_evaluates_every_neighbour(self):
        graph = nx.Graph([(0, 2), (0, 1), (1, 3), (2, 3)])
        coordinates = {node: (float(node), 0.0) for node in graph}

        def route_with_distances(distances):
            evaluated_to_destination = []

            def table_distance(first, second):
                first_node = int(first[0])
                second_node = int(second[0])
                if first_node == second_node:
                    return 0.0
                if second_node == 3:
                    evaluated_to_destination.append(first_node)
                    return distances[first_node]
                return abs(first_node - second_node)

            result = greedy_route(
                graph,
                coordinates,
                0,
                3,
                table_distance,
                tolerance=0.1,
            )
            return result, evaluated_to_destination

        tied_result, evaluated = route_with_distances(
            {0: 1.0, 1: 0.55, 2: 0.5, 3: 0.0}
        )
        outside_tolerance, _ = route_with_distances(
            {0: 1.0, 1: 0.61, 2: 0.5, 3: 0.0}
        )

        self.assertEqual(tied_result.walk, (0, 1, 3))
        self.assertTrue({1, 2}.issubset(evaluated))
        self.assertEqual(outside_tolerance.walk, (0, 2, 3))

    def test_strict_progress_requires_more_than_the_tolerance(self):
        graph = nx.path_graph(3)
        coordinates = {node: (float(node), 0.0) for node in graph}

        def route_with_middle_distance(middle_distance):
            distances = {0: 1.0, 1: middle_distance, 2: 0.0}

            def table_distance(first, second):
                first_node = int(first[0])
                second_node = int(second[0])
                if first_node == second_node:
                    return 0.0
                if second_node == 2:
                    return distances[first_node]
                return abs(first_node - second_node)

            return greedy_route(
                graph,
                coordinates,
                0,
                2,
                table_distance,
                tolerance=0.25,
            )

        boundary = route_with_middle_distance(0.75)
        strict_progress = route_with_middle_distance(0.74)

        self.assertFalse(boundary.success)
        self.assertEqual(boundary.failure_type, LOCAL_MINIMUM)
        self.assertEqual(boundary.walk, (0,))
        self.assertTrue(strict_progress.success)
        self.assertEqual(strict_progress.walk, (0, 1, 2))

    def test_attempted_revisit_is_cycle_and_is_not_traversed(self):
        graph = nx.path_graph(4)
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.2, 0.0),
            2: (-0.9, 0.0),
            3: (0.8, 0.0),
        }

        result = euclidean_greedy_route(graph, coordinates, 0, 3)

        self.assertFalse(result.success)
        self.assertEqual(result.failure_type, CYCLE)
        self.assertEqual(result.walk, (0, 1))
        self.assertEqual(len(set(result.walk)), len(result.walk))
        self.assert_result_invariants(graph, result)

    def test_source_equal_to_destination(self):
        graph = nx.path_graph(2)
        coordinates = {0: (0.0, 0.0), 1: (0.2, 0.0)}

        result = euclidean_greedy_route(graph, coordinates, 0, 0)

        self.assertTrue(result.success)
        self.assertEqual(result.walk, (0,))
        self.assertEqual(result.forwarding_decisions, 0)
        self.assert_result_invariants(graph, result)

    def test_missing_coordinates_and_invalid_nodes_are_rejected(self):
        graph = nx.path_graph(3)
        incomplete = {0: (0.0, 0.0), 2: (0.5, 0.0)}

        with self.assertRaisesRegex(ValueError, "missing graph nodes"):
            euclidean_greedy_route(graph, incomplete, 0, 2)
        with self.assertRaises(ValueError):
            euclidean_greedy_route(graph, incomplete, 9, 2)
        with self.assertRaises(ValueError):
            euclidean_greedy_route(graph, incomplete, 0, 9)

    def test_defensive_step_limit_is_reported(self):
        graph = nx.path_graph(3)
        coordinates = {0: (-0.6, 0.0), 1: (0.0, 0.0), 2: (0.6, 0.0)}

        result = euclidean_greedy_route(
            graph, coordinates, 0, 2, step_limit=1
        )

        self.assertFalse(result.success)
        self.assertEqual(result.failure_type, STEP_LIMIT)
        self.assertEqual(result.walk, (0, 1))
        self.assert_result_invariants(graph, result)

    def test_generic_core_is_equivalent_for_equivalent_metrics(self):
        graph = nx.path_graph(3)
        coordinates = {0: (-0.6, 0.0), 1: (0.0, 0.0), 2: (0.6, 0.0)}

        def metric_one(first, second):
            return hypot(first[0] - second[0], first[1] - second[1])

        def metric_two(first, second):
            return euclidean_distance(first, second)

        first = greedy_route(
            graph, coordinates, 0, 2, metric_one, method_name="first"
        )
        second = greedy_route(
            graph, coordinates, 0, 2, metric_two, method_name="second"
        )

        self.assertEqual(first.walk, second.walk)
        self.assertEqual(first.success, second.success)
        self.assertEqual(first.failure_type, second.failure_type)


class HyperbolicGreedyTests(RoutingTestCase):
    def test_hyperbolic_success_and_determinism(self):
        graph = nx.path_graph(3)
        coordinates = {0: (-0.6, 0.0), 1: (0.0, 0.0), 2: (0.6, 0.0)}

        first = hyperbolic_greedy_route(graph, coordinates, 0, 2)
        second = hyperbolic_greedy_route(graph, coordinates, 0, 2)

        self.assertEqual(first, second)
        self.assertTrue(first.success)
        self.assertEqual(first.walk, (0, 1, 2))
        self.assert_result_invariants(graph, first)

    def test_hyperbolic_local_minimum(self):
        graph = nx.path_graph(3)
        coordinates = {0: (0.7, 0.0), 1: (-0.8, 0.0), 2: (0.8, 0.0)}

        result = hyperbolic_greedy_route(graph, coordinates, 0, 2)

        self.assertFalse(result.success)
        self.assertEqual(result.failure_type, LOCAL_MINIMUM)
        self.assert_result_invariants(graph, result)

    def test_invalid_disk_coordinate_is_rejected_before_routing(self):
        graph = nx.path_graph(3)
        coordinates = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 0.0)}

        with self.assertRaises(ValueError):
            hyperbolic_greedy_route(graph, coordinates, 0, 2)

    def test_methods_share_coordinates_without_mutating_them(self):
        graph = nx.path_graph(3)
        coordinates = {0: (-0.6, 0.0), 1: (0.0, 0.0), 2: (0.6, 0.0)}
        before = dict(coordinates)

        euclidean = euclidean_greedy_route(graph, coordinates, 0, 2)
        hyperbolic = hyperbolic_greedy_route(graph, coordinates, 0, 2)

        self.assertEqual(coordinates, before)
        self.assertEqual(euclidean.walk, hyperbolic.walk)

    def test_euclidean_and_poincare_rankings_can_differ(self):
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        coordinates = {
            0: (0.0, -0.2),
            1: (0.8, 0.5),
            2: (0.2, 0.0),
            3: (0.8, 0.0),
        }

        self.assertLess(
            euclidean_distance(coordinates[1], coordinates[3]),
            euclidean_distance(coordinates[2], coordinates[3]),
        )
        self.assertGreater(
            poincare_distance(coordinates[1], coordinates[3]),
            poincare_distance(coordinates[2], coordinates[3]),
        )

        euclidean = euclidean_greedy_route(graph, coordinates, 0, 3)
        hyperbolic = hyperbolic_greedy_route(graph, coordinates, 0, 3)

        self.assertEqual(euclidean.walk[1], 1)
        self.assertEqual(hyperbolic.walk[1], 2)
        self.assert_result_invariants(graph, euclidean)
        self.assert_result_invariants(graph, hyperbolic)


class RepairedHyperbolicGreedyTests(RoutingTestCase):
    def test_repair_backtracks_and_reaches_destination(self):
        graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.2, 0.0),
            2: (0.65, 0.0),
            3: (-0.5, 0.4),
            4: (0.85, 0.0),
        }

        ordinary = hyperbolic_greedy_route(graph, coordinates, 0, 4)
        repaired = repaired_hyperbolic_greedy_route(
            graph, coordinates, 0, 4
        )

        self.assertFalse(ordinary.success)
        self.assertEqual(ordinary.failure_type, CYCLE)
        self.assertGreater(
            poincare_distance(coordinates[3], coordinates[4]),
            poincare_distance(coordinates[1], coordinates[4]),
        )
        self.assertTrue(repaired.success)
        self.assertEqual(repaired.initial_failure_type, CYCLE)
        self.assertIsNone(repaired.final_failure_type)
        self.assertTrue(repaired.repair_attempted)
        self.assertTrue(repaired.repair_alternative_existed)
        self.assertTrue(repaired.repair_succeeded)
        self.assertEqual(repaired.repair_attempt_count, 1)
        self.assertEqual(repaired.walk, (0, 1, 2, 1, 3, 4))
        self.assertEqual(repaired.route_length, 5)
        self.assert_result_invariants(graph, repaired)

    def test_non_source_local_minimum_can_be_repaired(self):
        graph = nx.Graph(
            [(0, 1), (1, 2), (2, 5), (1, 3), (3, 4)]
        )
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.2, 0.0),
            2: (0.65, 0.0),
            3: (-0.5, 0.4),
            4: (0.85, 0.0),
            5: (0.6, 0.0),
        }

        ordinary = hyperbolic_greedy_route(graph, coordinates, 0, 4)
        repaired = repaired_hyperbolic_greedy_route(
            graph,
            coordinates,
            0,
            4,
        )

        self.assertFalse(ordinary.success)
        self.assertEqual(ordinary.failure_type, LOCAL_MINIMUM)
        self.assertEqual(ordinary.walk, (0, 1, 2))
        self.assertTrue(repaired.success)
        self.assertEqual(repaired.initial_failure_type, LOCAL_MINIMUM)
        self.assertEqual(repaired.walk, (0, 1, 2, 1, 3, 4))
        self.assertEqual(repaired.repair_attempt_count, 1)
        self.assert_result_invariants(graph, repaired)

    def test_repair_does_not_mutate_graph_or_coordinates(self):
        graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
        graph.graph["provenance"] = {"fixture": "repair"}
        graph.nodes[1]["marker"] = [1, 2]
        coordinates = {
            0: [-0.8, 0.0],
            1: [-0.2, 0.0],
            2: [0.65, 0.0],
            3: [-0.5, 0.4],
            4: [0.85, 0.0],
        }
        graph_before = deepcopy(graph)
        coordinates_before = deepcopy(coordinates)

        result = repaired_hyperbolic_greedy_route(
            graph,
            coordinates,
            0,
            4,
        )

        self.assertTrue(result.repair_attempted)
        self.assertTrue(nx.utils.graphs_equal(graph, graph_before))
        self.assertEqual(coordinates, coordinates_before)

    def test_greedy_and_repair_use_no_shortest_path_information(self):
        graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.2, 0.0),
            2: (0.65, 0.0),
            3: (-0.5, 0.4),
            4: (0.85, 0.0),
        }
        forbidden_functions = (
            "dijkstra_path",
            "dijkstra_path_length",
            "shortest_path",
            "shortest_path_length",
            "single_source_shortest_path_length",
            "all_pairs_shortest_path_length",
        )

        with ExitStack() as stack:
            for function_name in forbidden_functions:
                stack.enter_context(
                    patch(
                        f"routing.nx.{function_name}",
                        side_effect=AssertionError(
                            f"greedy routing called {function_name}"
                        ),
                    )
                )
            euclidean = euclidean_greedy_route(graph, coordinates, 0, 4)
            hyperbolic = hyperbolic_greedy_route(graph, coordinates, 0, 4)
            repaired = repaired_hyperbolic_greedy_route(
                graph,
                coordinates,
                0,
                4,
            )

        for result in (euclidean, hyperbolic, repaired):
            self.assert_result_invariants(graph, result)

    def test_step_limit_after_escape_records_one_failed_repair(self):
        graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.2, 0.0),
            2: (0.65, 0.0),
            3: (-0.5, 0.4),
            4: (0.85, 0.0),
        }

        result = repaired_hyperbolic_greedy_route(
            graph,
            coordinates,
            0,
            4,
            step_limit=4,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.initial_failure_type, CYCLE)
        self.assertEqual(result.final_failure_type, STEP_LIMIT)
        self.assertEqual(result.failure_type, STEP_LIMIT)
        self.assertTrue(result.repair_attempted)
        self.assertTrue(result.repair_alternative_existed)
        self.assertFalse(result.repair_succeeded)
        self.assertEqual(result.repair_attempt_count, 1)
        self.assertEqual(result.forwarding_decisions, 4)
        self.assertEqual(result.walk, (0, 1, 2, 1, 3))
        self.assert_result_invariants(graph, result)

    def test_no_repair_needed_matches_ordinary_route(self):
        graph = nx.path_graph(3)
        coordinates = {0: (-0.6, 0.0), 1: (0.0, 0.0), 2: (0.6, 0.0)}

        ordinary = hyperbolic_greedy_route(graph, coordinates, 0, 2)
        repaired = repaired_hyperbolic_greedy_route(
            graph, coordinates, 0, 2
        )

        self.assertEqual(repaired.walk, ordinary.walk)
        self.assertEqual(repaired.route_length, ordinary.route_length)
        self.assertEqual(
            repaired.forwarding_decisions, ordinary.forwarding_decisions
        )
        self.assertFalse(repaired.repair_attempted)
        self.assertEqual(repaired.repair_attempt_count, 0)
        self.assert_result_invariants(graph, repaired)

    def test_repair_is_unavailable_when_failure_occurs_at_source(self):
        graph = nx.path_graph(3)
        coordinates = {0: (0.7, 0.0), 1: (-0.8, 0.0), 2: (0.8, 0.0)}

        result = repaired_hyperbolic_greedy_route(
            graph, coordinates, 0, 2
        )

        self.assertFalse(result.success)
        self.assertEqual(result.initial_failure_type, LOCAL_MINIMUM)
        self.assertEqual(result.final_failure_type, REPAIR_UNAVAILABLE)
        self.assertFalse(result.repair_attempted)
        self.assertFalse(result.repair_alternative_existed)
        self.assertEqual(result.repair_attempt_count, 0)
        self.assert_result_invariants(graph, result)

    def test_no_alternative_records_failed_single_repair(self):
        graph = nx.Graph([(0, 1), (1, 2), (0, 3), (3, 4)])
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.1, 0.0),
            2: (0.65, 0.0),
            3: (-0.9, 0.05),
            4: (0.85, 0.0),
        }

        result = repaired_hyperbolic_greedy_route(
            graph, coordinates, 0, 4
        )

        self.assertFalse(result.success)
        self.assertEqual(result.final_failure_type, REPAIR_FAILED)
        self.assertTrue(result.repair_attempted)
        self.assertFalse(result.repair_alternative_existed)
        self.assertEqual(result.repair_attempt_count, 1)
        self.assertEqual(result.walk, (0, 1, 2, 1))
        self.assert_result_invariants(graph, result)

    def test_failure_after_repair_does_not_trigger_second_attempt(self):
        graph = nx.Graph(
            [(0, 1), (1, 2), (1, 3), (0, 5), (5, 4)]
        )
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.1, 0.0),
            2: (0.65, 0.0),
            3: (0.0, 0.35),
            4: (0.85, 0.0),
            5: (-0.9, 0.05),
        }

        result = repaired_hyperbolic_greedy_route(
            graph, coordinates, 0, 4
        )

        self.assertFalse(result.success)
        self.assertEqual(result.initial_failure_type, CYCLE)
        self.assertEqual(result.final_failure_type, CYCLE)
        self.assertEqual(result.repair_attempt_count, 1)
        self.assertTrue(result.repair_alternative_existed)
        self.assertEqual(result.walk, (0, 1, 2, 1, 3))
        self.assert_result_invariants(graph, result)

    def test_repair_alternative_tie_uses_smallest_node_id(self):
        graph = nx.Graph(
            [(0, 1), (1, 2), (1, 3), (1, 4), (3, 6), (4, 6)]
        )
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.2, 0.0),
            2: (0.65, 0.0),
            3: (0.0, 0.3),
            4: (0.0, -0.3),
            6: (0.85, 0.0),
        }

        first = repaired_hyperbolic_greedy_route(
            graph, coordinates, 0, 6
        )
        second = repaired_hyperbolic_greedy_route(
            graph, coordinates, 0, 6
        )

        self.assertEqual(first, second)
        self.assertEqual(first.walk, (0, 1, 2, 1, 3, 6))
        self.assertEqual(first.repair_attempt_count, 1)
        self.assert_result_invariants(graph, first)


class GeneratedGraphRoutingInvariantTests(RoutingTestCase):
    @staticmethod
    def coordinate_snapshot(coordinates):
        return {
            node: tuple(float(component) for component in coordinate)
            for node, coordinate in coordinates.items()
        }

    def test_invariants_hold_across_small_deterministic_er_and_ba_graphs(self):
        generated_graphs = []
        for replicate_index, graph_seed in enumerate((1101, 1102)):
            generated_graphs.append(
                generate_connected_erdos_renyi(
                    n=10,
                    p=0.6,
                    graph_seed=graph_seed,
                    replicate_index=replicate_index,
                    max_attempts=20,
                ).graph
            )
        for replicate_index, graph_seed in enumerate((2101, 2102)):
            generated_graphs.append(
                generate_connected_barabasi_albert(
                    n=10,
                    m=2,
                    graph_seed=graph_seed,
                    replicate_index=replicate_index,
                ).graph
            )

        for graph_index, graph in enumerate(generated_graphs):
            with self.subTest(graph_index=graph_index):
                coordinates = embed_graph_in_poincare_disk(
                    graph,
                    seed=3100 + graph_index,
                    embedding_radius=0.8,
                    iterations=30,
                    disk_epsilon=1e-6,
                    tolerance=1e-12,
                )
                graph_before = deepcopy(graph)
                coordinates_before = self.coordinate_snapshot(coordinates)

                for source, destination in ((0, 9), (9, 0), (1, 8)):
                    benchmark = dijkstra_benchmark(
                        graph,
                        source,
                        destination,
                    )
                    greedy_results = (
                        euclidean_greedy_route(
                            graph,
                            coordinates,
                            source,
                            destination,
                        ),
                        hyperbolic_greedy_route(
                            graph,
                            coordinates,
                            source,
                            destination,
                        ),
                        repaired_hyperbolic_greedy_route(
                            graph,
                            coordinates,
                            source,
                            destination,
                        ),
                    )

                    self.assertEqual(
                        benchmark.route_length,
                        nx.shortest_path_length(graph, source, destination),
                    )
                    self.assert_result_invariants(graph, benchmark)
                    for result in greedy_results:
                        self.assertEqual(result.source, source)
                        self.assertEqual(result.destination, destination)
                        self.assertLessEqual(result.repair_attempt_count, 1)
                        self.assert_result_invariants(graph, result)
                        if result.success:
                            self.assertGreaterEqual(
                                result.route_length,
                                benchmark.route_length,
                            )

                    for ordinary_result in greedy_results[:2]:
                        self.assertEqual(
                            len(set(ordinary_result.walk)),
                            len(ordinary_result.walk),
                        )

                self.assertTrue(nx.utils.graphs_equal(graph, graph_before))
                self.assertEqual(
                    self.coordinate_snapshot(coordinates),
                    coordinates_before,
                )


if __name__ == "__main__":
    unittest.main()
