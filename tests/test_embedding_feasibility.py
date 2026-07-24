from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import textwrap
import unittest
from unittest.mock import patch

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from experiment_config import (  # noqa: E402
    APPROVED_EMBEDDING_DESIGN,
    BARABASI_ALBERT,
    DEVELOPMENT_CONFIG,
    ERDOS_RENYI,
    FEASIBILITY_PILOT_SEEDS,
    FULL_EXPERIMENT_CONFIG,
    HYDRA_CONDITION_ID,
    MDS_CONDITION_IDS,
    MDS_MAXIMUM_RADII,
)
import embedding as embedding_module  # noqa: E402
import network_metrics as network_metrics_module  # noqa: E402
import run_embedding_feasibility as feasibility_module  # noqa: E402
from routing import (  # noqa: E402
    DIJKSTRA_METHOD,
    EUCLIDEAN_GREEDY_METHOD,
    HYPERBOLIC_GREEDY_METHOD,
    REPAIRED_HYPERBOLIC_GREEDY_METHOD,
)
from run_embedding_feasibility import (  # noqa: E402
    PILOT_CONFIGURATION_FINGERPRINT,
    PILOT_EVIDENCE_LABEL,
    PILOT_PAIR_COUNT,
    run_approved_embedding_pipeline,
    run_embedding_feasibility_pilot,
    print_feasibility_summary,
)


GREEDY_METHODS = {
    EUCLIDEAN_GREEDY_METHOD,
    HYPERBOLIC_GREEDY_METHOD,
    REPAIRED_HYPERBOLIC_GREEDY_METHOD,
}


def _stable_record(record):
    return (
        record.graph_id,
        record.graph_family,
        record.graph_replicate,
        record.n,
        record.number_of_edges,
        record.graph_seed,
        record.pair_seed,
        record.er_p,
        record.ba_m,
        tuple(record.graph_generation_metadata.items()),
        record.configuration_fingerprint,
        record.embedding_input_fingerprint,
        record.hydra_metadata,
        record.mds_base_metadata,
        record.source_destination_pairs,
        record.pair_ids,
        record.coordinate_condition_ids,
        tuple(
            replace(diagnostic, transformation_runtime_ns=0)
            for diagnostic in record.condition_diagnostics
        ),
        tuple(
            replace(route_record, runtime_ns=0)
            for route_record in record.route_records
        ),
    )


class ApprovedEmbeddingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.barabasi_albert_graph(12, 2, seed=17)
        self.arguments = {
            "graph_id": "integration_cycle_8",
            "graph_family": BARABASI_ALBERT,
            "graph_replicate": 0,
            "graph_seed": 101,
            "pair_seed": 202,
            "pair_count": 2,
        }

    def test_shared_work_is_executed_at_the_required_frequency(self):
        with (
            patch.object(
                feasibility_module,
                "prepare_all_pairs_shortest_paths",
                wraps=feasibility_module.prepare_all_pairs_shortest_paths,
            ) as prepare_shortest_paths,
            patch.object(
                embedding_module,
                "prepare_all_pairs_shortest_paths",
                side_effect=AssertionError("embedding recomputed shortest paths"),
            ),
            patch.object(
                network_metrics_module,
                "prepare_all_pairs_shortest_paths",
                side_effect=AssertionError("metrics recomputed shortest paths"),
            ),
            patch.object(
                feasibility_module,
                "embed_hydra",
                wraps=feasibility_module.embed_hydra,
            ) as hydra,
            patch.object(
                feasibility_module,
                "classical_mds",
                wraps=feasibility_module.classical_mds,
            ) as mds,
            patch.object(
                feasibility_module,
                "_timed_mds_sensitivity_conditions",
                wraps=feasibility_module._timed_mds_sensitivity_conditions,
            ) as mds_conditions,
            patch.object(
                feasibility_module,
                "sample_ordered_pairs",
                wraps=feasibility_module.sample_ordered_pairs,
            ) as sample_pairs,
            patch.object(
                feasibility_module,
                "dijkstra_benchmark",
                wraps=feasibility_module.dijkstra_benchmark,
            ) as dijkstra,
            patch.object(
                feasibility_module,
                "euclidean_greedy_route",
                wraps=feasibility_module.euclidean_greedy_route,
            ) as euclidean_greedy,
            patch.object(
                feasibility_module,
                "hyperbolic_greedy_route",
                wraps=feasibility_module.hyperbolic_greedy_route,
            ) as hyperbolic_greedy,
            patch.object(
                feasibility_module,
                "repaired_hyperbolic_greedy_route",
                wraps=feasibility_module.repaired_hyperbolic_greedy_route,
            ) as repaired_greedy,
        ):
            record = run_approved_embedding_pipeline(
                self.graph,
                **self.arguments,
            )

        self.assertEqual(prepare_shortest_paths.call_count, 1)
        self.assertEqual(hydra.call_count, 1)
        self.assertEqual(mds.call_count, 1)
        self.assertIs(hydra.call_args.args[0], mds.call_args.args[0])
        self.assertEqual(
            hydra.call_args.args[0].input_fingerprint,
            record.embedding_input_fingerprint,
        )
        self.assertEqual(
            hydra.call_args.args[0].configuration_fingerprint,
            record.configuration_fingerprint,
        )
        self.assertEqual(mds_conditions.call_count, 1)
        self.assertEqual(sample_pairs.call_count, 1)
        self.assertEqual(dijkstra.call_count, self.arguments["pair_count"])
        self.assertEqual(
            tuple(call.args[1:3] for call in dijkstra.call_args_list),
            record.source_destination_pairs,
        )
        expected_greedy_calls = (
            self.arguments["pair_count"]
            * APPROVED_EMBEDDING_DESIGN.coordinate_condition_count
        )
        self.assertEqual(euclidean_greedy.call_count, expected_greedy_calls)
        self.assertEqual(hyperbolic_greedy.call_count, expected_greedy_calls)
        self.assertEqual(repaired_greedy.call_count, expected_greedy_calls)
        self.assertEqual(len(record.route_records), 32)

    def test_each_pair_uses_all_conditions_and_methods(self):
        record = run_approved_embedding_pipeline(
            self.graph,
            **self.arguments,
        )

        self.assertEqual(
            record.coordinate_condition_ids,
            (HYDRA_CONDITION_ID, *MDS_CONDITION_IDS),
        )
        self.assertEqual(len(record.condition_diagnostics), 5)
        radius_by_condition = {
            diagnostic.coordinate_condition_id: diagnostic.mds_radius
            for diagnostic in record.condition_diagnostics
        }
        self.assertIsNone(radius_by_condition[HYDRA_CONDITION_ID])
        self.assertEqual(
            tuple(radius_by_condition[item] for item in MDS_CONDITION_IDS),
            MDS_MAXIMUM_RADII,
        )
        for diagnostic in record.condition_diagnostics:
            self.assertEqual(
                diagnostic.embedding_metadata.coordinate_condition_id,
                diagnostic.coordinate_condition_id,
            )
            self.assertEqual(
                diagnostic.poincare_routing_tolerance,
                DEVELOPMENT_CONFIG.numerical_tolerance,
            )
            if diagnostic.mds_radius is None:
                self.assertEqual(
                    diagnostic.euclidean_routing_tolerance,
                    DEVELOPMENT_CONFIG.numerical_tolerance,
                )
            else:
                self.assertAlmostEqual(
                    diagnostic.euclidean_routing_tolerance
                    / diagnostic.mds_radius,
                    DEVELOPMENT_CONFIG.numerical_tolerance
                    / max(MDS_MAXIMUM_RADII),
                    places=18,
                )

        for pair_id, pair in zip(
            record.pair_ids,
            record.source_destination_pairs,
            strict=True,
        ):
            pair_records = [
                item for item in record.route_records if item.pair_id == pair_id
            ]
            self.assertEqual(len(pair_records), 16)
            self.assertTrue(
                all(
                    (item.source, item.destination) == pair
                    for item in pair_records
                )
            )
            dijkstra_records = [
                item for item in pair_records if item.routing_method == DIJKSTRA_METHOD
            ]
            self.assertEqual(len(dijkstra_records), 1)
            self.assertIsNone(dijkstra_records[0].embedding_family)
            self.assertIsNone(dijkstra_records[0].coordinate_condition_id)

            for condition_id in record.coordinate_condition_ids:
                condition_records = [
                    item
                    for item in pair_records
                    if item.coordinate_condition_id == condition_id
                ]
                self.assertEqual(len(condition_records), 3)
                self.assertEqual(
                    {item.routing_method for item in condition_records},
                    GREEDY_METHODS,
                )
                self.assertTrue(
                    all(
                        item.graph_replicate == self.arguments["graph_replicate"]
                        for item in condition_records
                    )
                )

    def test_mds_radius_transformations_are_measured_individually(self):
        shortest_paths = feasibility_module.prepare_all_pairs_shortest_paths(
            self.graph
        )
        embedding_input = embedding_module.prepare_embedding_input(
            self.graph,
            shortest_paths,
            configuration_fingerprint=(
                DEVELOPMENT_CONFIG.configuration_fingerprint
            ),
        )
        base = feasibility_module.classical_mds(embedding_input)
        clock_values = (0, 11, 20, 42, 50, 83, 100, 144)

        with patch.object(
            feasibility_module,
            "perf_counter_ns",
            side_effect=clock_values,
        ):
            conditions, runtimes = (
                feasibility_module._timed_mds_sensitivity_conditions(
                    base,
                    radii=MDS_MAXIMUM_RADII,
                    tolerance=DEVELOPMENT_CONFIG.numerical_tolerance,
                )
            )

        self.assertEqual(
            tuple(
                condition.metadata.coordinate_condition_id
                for condition in conditions
            ),
            MDS_CONDITION_IDS,
        )
        self.assertEqual(runtimes, (11, 22, 33, 44))

        original = feasibility_module._timed_mds_sensitivity_conditions

        def measured_conditions(*args, **kwargs):
            transformed, _ = original(*args, **kwargs)
            return transformed, (101, 202, 303, 404)

        with patch.object(
            feasibility_module,
            "_timed_mds_sensitivity_conditions",
            side_effect=measured_conditions,
        ):
            record = run_approved_embedding_pipeline(
                self.graph,
                **self.arguments,
            )
        self.assertEqual(
            tuple(
                diagnostic.transformation_runtime_ns
                for diagnostic in record.condition_diagnostics
                if diagnostic.mds_radius is not None
            ),
            (101, 202, 303, 404),
        )

    def test_k6_and_twin_graph_coincidences_do_not_crash_pipeline(self):
        graphs = (
            ("k6", nx.complete_graph(6)),
            (
                "twin_leaves",
                nx.Graph(
                    ((0, 5), (1, 5), (2, 5), (3, 5), (4, 5))
                ),
            ),
        )
        for index, (name, graph) in enumerate(graphs):
            with self.subTest(graph=name):
                if name == "twin_leaves":
                    self.assertEqual(
                        set(graph.neighbors(2)),
                        set(graph.neighbors(4)),
                    )
                record = run_approved_embedding_pipeline(
                    graph,
                    graph_id=f"coincident_{name}",
                    graph_family=BARABASI_ALBERT,
                    graph_replicate=0,
                    graph_seed=600 + index,
                    pair_seed=700 + index,
                    pair_count=2,
                )

                self.assertGreaterEqual(
                    record.mds_base_metadata.coincident_coordinate_group_count,
                    1,
                )
                self.assertGreaterEqual(
                    record.mds_base_metadata.coincident_vertex_count,
                    2,
                )
                self.assertGreaterEqual(
                    record.mds_base_metadata.coincident_vertex_pair_count,
                    1,
                )
                self.assertEqual(len(record.route_records), 32)

    def test_approved_pipeline_is_deterministic_across_python_hash_seeds(self):
        script = textwrap.dedent(
            """
            from dataclasses import asdict
            import json
            from pathlib import Path
            import sys

            sys.path.insert(0, str(Path.cwd() / "code"))
            import networkx as nx
            from embedding import prepare_embedding_input
            from experiment_config import DEVELOPMENT_CONFIG, ERDOS_RENYI
            from hydra_embedding import embed_hydra
            from mds_embedding import classical_mds, create_mds_sensitivity_conditions
            from network_metrics import prepare_all_pairs_shortest_paths
            from run_embedding_feasibility import run_approved_embedding_pipeline

            graph = nx.complete_graph(6)
            shortest_paths = prepare_all_pairs_shortest_paths(graph)
            embedding_input = prepare_embedding_input(
                graph,
                shortest_paths,
                configuration_fingerprint=DEVELOPMENT_CONFIG.configuration_fingerprint,
                tolerance=DEVELOPMENT_CONFIG.numerical_tolerance,
            )
            design = DEVELOPMENT_CONFIG.approved_embedding_design
            hydra = embed_hydra(
                embedding_input,
                dimension=design.hydra_dimension,
                kappa=design.hydra_kappa,
                centering_tolerance=design.hydra_centering_tolerance,
                centering_max_iterations=design.hydra_centering_max_iterations,
                eigenvalue_tolerance=design.hydra_eigenvalue_tolerance,
                pairwise_isometry_tolerance=design.hydra_isometry_tolerance,
                boundary_roundoff_tolerance=design.hydra_boundary_roundoff_tolerance,
            )
            mds_base = classical_mds(
                embedding_input,
                dimension=design.mds_dimension,
                eigenvalue_relative_tolerance=design.mds_eigenvalue_relative_tolerance,
                centroid_tolerance=design.mds_centroid_tolerance,
            )
            mds_conditions = create_mds_sensitivity_conditions(
                mds_base,
                radii=design.mds_maximum_radii,
                tolerance=DEVELOPMENT_CONFIG.numerical_tolerance,
            )
            record = run_approved_embedding_pipeline(
                graph,
                graph_id="subprocess_k6",
                graph_family=ERDOS_RENYI,
                graph_replicate=0,
                graph_seed=801,
                pair_seed=802,
                pair_count=3,
            )
            diagnostics = []
            for diagnostic in record.condition_diagnostics:
                item = asdict(diagnostic)
                item.pop("transformation_runtime_ns")
                diagnostics.append(item)
            routes = []
            for route in record.route_records:
                item = asdict(route)
                item.pop("runtime_ns")
                routes.append(item)
            payload = {
                "configuration_fingerprint": record.configuration_fingerprint,
                "embedding_input_fingerprint": record.embedding_input_fingerprint,
                "hydra_coordinates": list(hydra.coordinates.items()),
                "mds_base_coordinates": list(mds_base.coordinates.items()),
                "mds_condition_coordinates": [
                    [
                        condition.metadata.coordinate_condition_id,
                        list(condition.coordinates.items()),
                    ]
                    for condition in mds_conditions
                ],
                "hydra_metadata": asdict(record.hydra_metadata),
                "mds_base_metadata": asdict(record.mds_base_metadata),
                "pairs": record.source_destination_pairs,
                "diagnostics": diagnostics,
                "routes": routes,
            }
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            """
        )
        outputs = []
        for hash_seed in ("1", "987654321"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.stderr, "")
            outputs.append(completed.stdout)

        self.assertEqual(outputs[0], outputs[1])
        payload = json.loads(outputs[0])
        self.assertEqual(len(payload["routes"]), 48)

    def test_repeated_pipeline_runs_have_identical_non_timing_results(self):
        first = run_approved_embedding_pipeline(
            self.graph,
            **self.arguments,
        )
        second = run_approved_embedding_pipeline(
            self.graph,
            **self.arguments,
        )

        self.assertEqual(_stable_record(first), _stable_record(second))

    def test_pipeline_does_not_mutate_the_input_graph(self):
        self.graph.nodes[0]["label"] = "preserve"
        self.graph.nodes[0]["nested"] = {"value": 1}
        nodes_before = deepcopy(tuple(self.graph.nodes(data=True)))
        edges_before = deepcopy(tuple(self.graph.edges(data=True)))

        run_approved_embedding_pipeline(
            self.graph,
            **self.arguments,
        )

        self.assertEqual(
            deepcopy(tuple(self.graph.nodes(data=True))),
            nodes_before,
        )
        self.assertEqual(
            deepcopy(tuple(self.graph.edges(data=True))),
            edges_before,
        )


class EmbeddingFeasibilityPilotTests(unittest.TestCase):
    def test_pilot_is_small_excluded_and_uses_reserved_seeds(self):
        original_directory = Path.cwd()
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            try:
                os.chdir(temporary_path)
                report = run_embedding_feasibility_pilot()
            finally:
                os.chdir(original_directory)
            self.assertEqual(tuple(temporary_path.iterdir()), ())

        self.assertEqual(report.evidence_label, PILOT_EVIDENCE_LABEL)
        self.assertTrue(report.excluded_from_final_experiment)
        self.assertEqual(report.pilot_seeds, FEASIBILITY_PILOT_SEEDS)
        self.assertEqual(
            report.configuration_fingerprint,
            PILOT_CONFIGURATION_FINGERPRINT,
        )
        self.assertNotEqual(
            report.configuration_fingerprint,
            DEVELOPMENT_CONFIG.configuration_fingerprint,
        )
        self.assertEqual(
            tuple((graph.graph_family, graph.n) for graph in report.graphs),
            (
                (ERDOS_RENYI, 30),
                (BARABASI_ALBERT, 30),
                (BARABASI_ALBERT, 100),
            ),
        )
        for graph in report.graphs:
            self.assertEqual(graph.graph_replicate, 0)
            self.assertEqual(
                graph.configuration_fingerprint,
                PILOT_CONFIGURATION_FINGERPRINT,
            )
            self.assertEqual(
                graph.hydra_metadata.configuration_fingerprint,
                PILOT_CONFIGURATION_FINGERPRINT,
            )
            self.assertEqual(
                graph.mds_base_metadata.configuration_fingerprint,
                PILOT_CONFIGURATION_FINGERPRINT,
            )
            generation_metadata = dict(graph.graph_generation_metadata)
            self.assertEqual(
                generation_metadata["graph_model"],
                graph.graph_family,
            )
            self.assertEqual(generation_metadata["n"], graph.n)
            self.assertEqual(
                generation_metadata["graph_seed"],
                graph.graph_seed,
            )
            if graph.graph_family == ERDOS_RENYI:
                self.assertIsNotNone(graph.er_p)
                self.assertIsNone(graph.ba_m)
                self.assertEqual(generation_metadata["p"], graph.er_p)
            else:
                self.assertIsNone(graph.er_p)
                self.assertEqual(graph.ba_m, 4)
                self.assertEqual(generation_metadata["m"], graph.ba_m)
            self.assertEqual(len(graph.source_destination_pairs), PILOT_PAIR_COUNT)
            self.assertEqual(len(graph.route_records), PILOT_PAIR_COUNT * 16)
            self.assertEqual(
                graph.coordinate_condition_ids,
                APPROVED_EMBEDDING_DESIGN.coordinate_condition_ids,
            )

        workload = dict(report.workload_projection)
        full_workload = FULL_EXPERIMENT_CONFIG.workload_estimate
        self.assertEqual(
            workload,
            {key: full_workload[key] for key in workload},
        )
        output = StringIO()
        with redirect_stdout(output):
            print_feasibility_summary(report)
        summary_lines = output.getvalue().splitlines()
        condition_lines = [
            line
            for line in summary_lines
            if any(condition_id in line for condition_id in report.graphs[0].coordinate_condition_ids)
        ]
        self.assertEqual(len(condition_lines), len(report.graphs) * 5)
        self.assertTrue(
            all("routing_failures=" in line for line in condition_lines)
        )
        self.assertTrue(
            all("repair_opportunities=" in line for line in condition_lines)
        )


if __name__ == "__main__":
    unittest.main()
