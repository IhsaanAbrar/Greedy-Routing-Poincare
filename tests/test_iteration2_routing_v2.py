from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from iteration2_config import DATA_GENERATION_HASH  # noqa: E402
from iteration2_oracle import audit_production_result  # noqa: E402
from iteration2_routing import (  # noqa: E402
    LOCAL_MINIMUM,
    NO_ALTERNATIVE_AFTER_BACKTRACKING,
    POST_REPAIR_ATTEMPTED_REVISIT,
    POST_REPAIR_LOCAL_MINIMUM,
    REPAIR_UNAVAILABLE_AT_SOURCE,
    RoutingPriorityContext,
    euclidean_greedy_route_v2,
    poincare_greedy_route_v2,
    prepare_iteration2_routing,
    repaired_poincare_greedy_route_v2,
)
from poincare_distance import euclidean_distance  # noqa: E402


def context(
    *, pair_index: int, source: int, destination: int, graph_id: str = "fixture"
) -> RoutingPriorityContext:
    return RoutingPriorityContext(
        data_generation_hash=DATA_GENERATION_HASH,
        graph_id=graph_id,
        pair_index=pair_index,
        source=source,
        destination=destination,
    )


class Iteration2RoutingStateMachineTests(unittest.TestCase):
    def test_adjacent_destination_precedes_coincident_tied_neighbour(self) -> None:
        graph = nx.Graph([(0, 1), (0, 2), (1, 2)])
        coordinates = {0: (-0.5, 0.0), 1: (0.5, 0.0), 2: (0.5, 0.0)}
        result = euclidean_greedy_route_v2(
            graph,
            coordinates,
            0,
            2,
            tolerance=1e-14,
            priority_context=context(pair_index=0, source=0, destination=2),
        )
        self.assertEqual(result.walk, (0, 2))
        self.assertEqual(result.logical_distance_evaluations, 0)
        self.assertEqual(result.physical_hops, result.forwarding_decisions)

    def test_keyed_tie_can_choose_larger_id_and_ignores_insertion_order(self) -> None:
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        coordinates = {
            0: (-0.5, 0.0),
            1: (0.0, 0.2),
            2: (0.0, -0.2),
            3: (0.5, 0.0),
        }
        chosen_context = next(
            item
            for index in range(1_000)
            if (
                item := context(pair_index=index, source=0, destination=3)
            ).priority(0, 2)
            < item.priority(0, 1)
        )
        walks = []
        for edge_order in (edges, tuple(reversed(edges))):
            graph = nx.Graph()
            graph.add_edges_from(edge_order)
            walks.append(
                euclidean_greedy_route_v2(
                    graph,
                    coordinates,
                    0,
                    3,
                    tolerance=1e-14,
                    priority_context=chosen_context,
                ).walk
            )
        self.assertEqual(walks, [(0, 2, 3), (0, 2, 3)])

    def test_ba_style_relabelling_preserves_a_frozen_priority_mapping(self) -> None:
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        coordinates = {
            0: (-0.5, 0.0),
            1: (0.0, 0.2),
            2: (0.0, -0.2),
            3: (0.5, 0.0),
        }
        original_context = context(
            pair_index=31,
            source=0,
            destination=3,
            graph_id="excluded_ba_label_fixture",
        )
        original = euclidean_greedy_route_v2(
            graph,
            coordinates,
            0,
            3,
            tolerance=1e-14,
            priority_context=original_context,
        )
        relabel = {0: 0, 1: 2, 2: 1, 3: 3}
        inverse = {new: old for old, new in relabel.items()}
        relabelled_graph = nx.relabel_nodes(graph, relabel, copy=True)
        relabelled_coordinates = {
            relabel[node]: point for node, point in coordinates.items()
        }
        relabelled_context = context(
            pair_index=31,
            source=0,
            destination=3,
            graph_id="excluded_ba_label_fixture",
        )
        priority_method = RoutingPriorityContext.priority

        def frozen_physical_priority(
            self: RoutingPriorityContext,
            current: int,
            candidate: int,
        ) -> bytes:
            del self
            return priority_method(
                original_context,
                inverse[current],
                inverse[candidate],
            )

        with patch.object(
            RoutingPriorityContext,
            "priority",
            new=frozen_physical_priority,
        ):
            relabelled = euclidean_greedy_route_v2(
                relabelled_graph,
                relabelled_coordinates,
                0,
                3,
                tolerance=1e-14,
                priority_context=relabelled_context,
            )
        self.assertEqual(
            relabelled.walk,
            tuple(relabel[node] for node in original.walk),
        )

    def test_progress_eligibility_is_applied_before_tie_priority(self) -> None:
        graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        coordinates = {
            0: (1.0, 0.0),
            1: (0.89, 0.0),
            2: (0.95, 0.0),
            3: (0.0, 0.0),
        }
        priority = next(
            item
            for index in range(1_000)
            if (
                item := context(pair_index=index, source=0, destination=3)
            ).priority(0, 2)
            < item.priority(0, 1)
        )
        result = euclidean_greedy_route_v2(
            graph,
            coordinates,
            0,
            3,
            tolerance=0.1,
            priority_context=priority,
        )
        self.assertEqual(result.walk, (0, 1, 3))

    def test_prepared_cache_preserves_walks_and_logical_costs(self) -> None:
        graph = nx.path_graph(6)
        coordinates = {node: (node / 10.0, 0.0) for node in graph}
        priority = context(pair_index=7, source=0, destination=5)
        raw = euclidean_greedy_route_v2(
            graph,
            coordinates,
            0,
            5,
            tolerance=1e-14,
            priority_context=priority,
        )
        prepared = prepare_iteration2_routing(
            graph, coordinates, euclidean_distance, metric_name="fixture:euclidean"
        )
        first = euclidean_greedy_route_v2(
            graph,
            prepared,
            0,
            5,
            tolerance=1e-14,
            priority_context=priority,
        )
        cache_size = prepared.cache_size
        second = euclidean_greedy_route_v2(
            graph,
            prepared,
            0,
            5,
            tolerance=1e-14,
            priority_context=priority,
        )
        self.assertEqual(raw.walk, first.walk)
        self.assertEqual(raw.logical_distance_evaluations, first.logical_distance_evaluations)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(prepared.cache_size, cache_size)
        other = context(pair_index=8, source=0, destination=4)
        euclidean_greedy_route_v2(
            graph,
            prepared,
            0,
            4,
            tolerance=1e-14,
            priority_context=other,
        )
        self.assertEqual(prepared.active_destination, 4)
        with self.assertRaises(ValueError):
            euclidean_greedy_route_v2(
                graph.copy(),
                prepared,
                0,
                5,
                tolerance=1e-14,
                priority_context=priority,
            )
        with self.assertRaises(ValueError):
            poincare_greedy_route_v2(
                graph,
                prepared,
                0,
                5,
                tolerance=1e-14,
                priority_context=priority,
            )
        graph.remove_edge(2, 3)
        graph.add_edge(1, 3)
        self.assertEqual(graph.number_of_edges(), 5)
        with self.assertRaisesRegex(ValueError, "mutated"):
            euclidean_greedy_route_v2(
                graph,
                prepared,
                0,
                5,
                tolerance=1e-14,
                priority_context=priority,
            )

    def test_repair_reuses_ordinary_result_and_counts_every_physical_edge(self) -> None:
        graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
        coordinates = {
            0: (-0.8, 0.0),
            1: (-0.2, 0.0),
            2: (0.65, 0.0),
            3: (-0.5, 0.4),
            4: (0.85, 0.0),
        }
        priority = context(pair_index=3, source=0, destination=4)
        ordinary = poincare_greedy_route_v2(
            graph,
            coordinates,
            0,
            4,
            tolerance=1e-14,
            priority_context=priority,
        )
        self.assertEqual(ordinary.final_failure_type, LOCAL_MINIMUM)
        repaired = repaired_poincare_greedy_route_v2(
            graph,
            coordinates,
            0,
            4,
            tolerance=1e-14,
            priority_context=priority,
            ordinary_result=ordinary,
        )
        fresh = repaired_poincare_greedy_route_v2(
            graph,
            coordinates,
            0,
            4,
            tolerance=1e-14,
            priority_context=priority,
        )
        self.assertEqual(repaired.to_dict(), fresh.to_dict())
        self.assertEqual(repaired.walk, (0, 1, 2, 1, 3, 4))
        self.assertEqual(repaired.physical_hops, 5)
        self.assertEqual(repaired.forwarding_decisions, 5)
        self.assertTrue(repaired.repair_backtrackable)
        self.assertTrue(repaired.repair_eligible)
        self.assertTrue(repaired.repair_alternative_selected)
        agreement = audit_production_result(
            repaired,
            graph=graph,
            coordinates=coordinates,
            source=0,
            destination=4,
            metric="poincare",
            tolerance=1e-14,
            repaired=True,
            priority_context=priority,
        )
        self.assertTrue(agreement.float64_matches_production)
        self.assertTrue(agreement.high_precision_matches_production)

    def test_source_failure_is_not_backtrackable(self) -> None:
        graph = nx.path_graph(3)
        coordinates = {0: (0.1, 0.0), 1: (0.5, 0.0), 2: (0.0, 0.0)}
        priority = context(pair_index=11, source=0, destination=2)
        result = repaired_poincare_greedy_route_v2(
            graph,
            coordinates,
            0,
            2,
            tolerance=1e-14,
            priority_context=priority,
        )
        self.assertEqual(result.final_failure_type, REPAIR_UNAVAILABLE_AT_SOURCE)
        self.assertFalse(result.repair_attempted)
        self.assertFalse(result.repair_backtrackable)
        self.assertEqual(result.physical_hops, 0)

    def test_ordinary_success_never_triggers_repair(self) -> None:
        graph = nx.path_graph(4)
        coordinates = {
            0: (0.8, 0.0),
            1: (0.6, 0.0),
            2: (0.3, 0.0),
            3: (0.0, 0.0),
        }
        result = repaired_poincare_greedy_route_v2(
            graph,
            coordinates,
            0,
            3,
            tolerance=1e-14,
            priority_context=context(pair_index=12, source=0, destination=3),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.walk, (0, 1, 2, 3))
        self.assertFalse(result.repair_attempted)
        self.assertEqual(result.repair_attempt_count, 0)
        self.assertIsNone(result.initial_failure_type)

    def test_one_backtrack_with_no_unexplored_alternative_stops(self) -> None:
        graph = nx.Graph([(0, 1), (1, 2), (0, 3), (3, 4)])
        coordinates = {
            0: (0.8, 0.0),
            1: (0.6, 0.0),
            2: (0.5, 0.0),
            3: (0.9, 0.0),
            4: (0.0, 0.0),
        }
        result = repaired_poincare_greedy_route_v2(
            graph,
            coordinates,
            0,
            4,
            tolerance=1e-14,
            priority_context=context(pair_index=14, source=0, destination=4),
        )
        self.assertEqual(result.walk, (0, 1, 2, 1))
        self.assertEqual(
            result.final_failure_type,
            NO_ALTERNATIVE_AFTER_BACKTRACKING,
        )
        self.assertTrue(result.repair_attempted)
        self.assertEqual(result.repair_attempt_count, 1)
        self.assertFalse(result.repair_alternative_existed)
        self.assertEqual(result.physical_hops, 3)

    def test_post_repair_local_minimum_does_not_trigger_second_repair(self) -> None:
        graph = nx.Graph(
            [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5)]
        )
        coordinates = {
            0: (0.8, 0.0),
            1: (0.6, 0.0),
            2: (0.3, 0.0),
            3: (0.4, 0.0),
            4: (0.5, 0.0),
            5: (0.0, 0.0),
        }
        result = repaired_poincare_greedy_route_v2(
            graph,
            coordinates,
            0,
            5,
            tolerance=1e-14,
            priority_context=context(pair_index=15, source=0, destination=5),
        )
        self.assertEqual(result.walk, (0, 1, 2, 1, 3))
        self.assertEqual(result.final_failure_type, POST_REPAIR_LOCAL_MINIMUM)
        self.assertEqual(result.repair_attempt_count, 1)
        self.assertTrue(result.repair_alternative_selected)
        self.assertFalse(result.success)

    def test_post_repair_revisit_is_separate_terminal_failure(self) -> None:
        graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4), (4, 5)])
        coordinates = {
            0: (0.8, 0.0),
            1: (0.6, 0.0),
            2: (0.5, 0.0),
            3: (0.9, 0.0),
            4: (0.95, 0.0),
            5: (0.0, 0.0),
        }
        priority = context(pair_index=13, source=0, destination=5)
        result = repaired_poincare_greedy_route_v2(
            graph,
            coordinates,
            0,
            5,
            tolerance=1e-14,
            priority_context=priority,
        )
        self.assertEqual(result.walk, (0, 1, 2, 1, 3))
        self.assertEqual(result.final_failure_type, POST_REPAIR_ATTEMPTED_REVISIT)
        self.assertTrue(result.repair_attempted)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
