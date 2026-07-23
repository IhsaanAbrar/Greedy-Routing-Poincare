import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from experiment_config import (  # noqa: E402
    BA_FINITE_DEGREE_MATCH_RULE,
    BA_INITIAL_GRAPH,
    BARABASI_ALBERT,
    CONFIGURATION_SCHEMA_VERSION,
    DEVELOPMENT_CONFIG,
    EMBEDDING_METHOD,
    ERDOS_RENYI,
    MAX_SEED,
)
from graph_generation import (  # noqa: E402
    GraphGenerationError,
    derive_attempt_seed,
    generate_connected_barabasi_albert,
    generate_connected_erdos_renyi,
    generate_graph,
    reproduce_graph_from_metadata,
)


def edge_set(graph):
    return {frozenset(edge) for edge in graph.edges}


class GraphGenerationTests(unittest.TestCase):
    def test_configured_erdos_renyi_graph_is_valid_and_records_metadata(self):
        generated = generate_graph(DEVELOPMENT_CONFIG, 0, ERDOS_RENYI, 0)

        self.assertEqual(set(generated.graph), set(range(30)))
        self.assertTrue(nx.is_connected(generated.graph))
        self.assertFalse(generated.graph.is_directed())
        self.assertFalse(generated.graph.is_multigraph())
        self.assertEqual(nx.number_of_selfloops(generated.graph), 0)
        self.assertEqual(generated.metadata["graph_model"], ERDOS_RENYI)
        self.assertEqual(generated.metadata["n"], 30)
        self.assertEqual(
            generated.metadata["p"],
            DEVELOPMENT_CONFIG.parameter_settings[0].er_p,
        )
        self.assertEqual(generated.metadata["replicate_index"], 0)
        self.assertGreaterEqual(generated.metadata["generation_attempt_count"], 1)
        self.assertEqual(
            generated.metadata["generation_attempt_index"],
            generated.metadata["generation_attempt_count"] - 1,
        )
        self.assertEqual(
            generated.metadata["generation_attempt_seed"],
            DEVELOPMENT_CONFIG.seed_for_graph_attempt(
                0,
                ERDOS_RENYI,
                0,
                generated.metadata["generation_attempt_index"],
            ),
        )
        self.assertEqual(
            generated.metadata["configuration_fingerprint"],
            DEVELOPMENT_CONFIG.configuration_fingerprint,
        )
        self.assertEqual(
            generated.metadata["configuration_schema_version"],
            CONFIGURATION_SCHEMA_VERSION,
        )
        self.assertEqual(generated.metadata["setting_label"], "dev_n30_m2")
        self.assertEqual(generated.metadata["embedding_method"], EMBEDDING_METHOD)
        self.assertEqual(
            generated.metadata["erdos_renyi_sampling"],
            "G(n,p) conditioned on connectedness",
        )
        self.assertIn("python_version", generated.metadata)
        self.assertIn("networkx_version", generated.metadata)
        json.dumps(generated.metadata)

    def test_configured_barabasi_albert_graph_is_valid_and_records_metadata(self):
        generated = generate_graph(DEVELOPMENT_CONFIG, 0, BARABASI_ALBERT, 0)

        self.assertEqual(set(generated.graph), set(range(30)))
        self.assertTrue(nx.is_connected(generated.graph))
        self.assertEqual(nx.number_of_selfloops(generated.graph), 0)
        self.assertEqual(generated.metadata["graph_model"], BARABASI_ALBERT)
        self.assertEqual(generated.metadata["m"], 2)
        self.assertEqual(generated.graph.number_of_edges(), 2 * (30 - 2))
        self.assertEqual(generated.metadata["ba_exact_edge_count"], 56)
        self.assertEqual(generated.metadata["ba_initial_graph"], BA_INITIAL_GRAPH)
        self.assertEqual(
            generated.metadata["degree_match_rule"],
            BA_FINITE_DEGREE_MATCH_RULE,
        )
        self.assertAlmostEqual(
            generated.metadata["er_expected_average_degree"],
            2 * generated.graph.number_of_edges() / 30,
        )
        self.assertEqual(generated.metadata["generation_attempt_count"], 1)
        self.assertEqual(
            generated.metadata["generation_attempt_seed"],
            generated.metadata["graph_seed"],
        )
        json.dumps(generated.metadata)

    def test_generation_is_deterministic_for_the_same_configured_replicate(self):
        for model in (ERDOS_RENYI, BARABASI_ALBERT):
            with self.subTest(model=model):
                first = generate_graph(DEVELOPMENT_CONFIG, 0, model, 0)
                second = generate_graph(DEVELOPMENT_CONFIG, 0, model, 0)

                self.assertEqual(first.metadata, second.metadata)
                self.assertEqual(edge_set(first.graph), edge_set(second.graph))

    def test_graph_can_be_reproduced_from_json_round_tripped_provenance(self):
        for model in (ERDOS_RENYI, BARABASI_ALBERT):
            with self.subTest(model=model):
                original = generate_graph(DEVELOPMENT_CONFIG, 0, model, 1)
                metadata = json.loads(json.dumps(original.metadata))
                reproduced = reproduce_graph_from_metadata(
                    DEVELOPMENT_CONFIG,
                    metadata,
                )

                self.assertEqual(original.metadata, reproduced.metadata)
                self.assertEqual(edge_set(original.graph), edge_set(reproduced.graph))

    def test_reproduction_rejects_missing_or_tampered_provenance(self):
        generated = generate_graph(DEVELOPMENT_CONFIG, 0, ERDOS_RENYI, 0)
        missing = dict(generated.metadata)
        del missing["configuration_fingerprint"]
        with self.assertRaises(ValueError):
            reproduce_graph_from_metadata(DEVELOPMENT_CONFIG, missing)

        tampered = dict(generated.metadata)
        tampered["generation_attempt_seed"] = (
            int(tampered["generation_attempt_seed"]) + 1
        ) % (MAX_SEED + 1)
        with self.assertRaises(GraphGenerationError):
            reproduce_graph_from_metadata(DEVELOPMENT_CONFIG, tampered)

    def test_different_replicates_receive_different_graph_seeds(self):
        for model in (ERDOS_RENYI, BARABASI_ALBERT):
            with self.subTest(model=model):
                first = generate_graph(DEVELOPMENT_CONFIG, 0, model, 0)
                second = generate_graph(DEVELOPMENT_CONFIG, 0, model, 1)

                self.assertNotEqual(
                    first.metadata["graph_seed"], second.metadata["graph_seed"]
                )

    def test_networkx_ba_edge_count_matches_finite_formula(self):
        for n, m in ((6, 1), (10, 3), (30, 4), (100, 16)):
            with self.subTest(n=n, m=m):
                generated = generate_connected_barabasi_albert(
                    n=n,
                    m=m,
                    graph_seed=123,
                    replicate_index=0,
                )
                self.assertEqual(
                    generated.graph.number_of_edges(),
                    m * (n - m),
                )

    def test_ba_generation_passes_the_recorded_initial_star_explicitly(self):
        networkx_generator = nx.barabasi_albert_graph
        with patch(
            "graph_generation.nx.barabasi_albert_graph",
            wraps=networkx_generator,
        ) as generator:
            generated = generate_connected_barabasi_albert(
                n=12,
                m=3,
                graph_seed=9,
                replicate_index=0,
            )

        initial_graph = generator.call_args.kwargs["initial_graph"]
        self.assertEqual(edge_set(initial_graph), edge_set(nx.star_graph(3)))
        self.assertEqual(generated.graph.number_of_edges(), 3 * (12 - 3))
        self.assertEqual(generated.metadata["ba_initial_graph"], BA_INITIAL_GRAPH)

    def test_retry_uses_unique_deterministic_attempt_seeds(self):
        disconnected = nx.empty_graph(6)
        connected = nx.path_graph(6)

        with patch(
            "graph_generation.nx.gnp_random_graph",
            side_effect=[disconnected, connected],
        ) as generator:
            generated = generate_connected_erdos_renyi(
                n=6,
                p=0.25,
                graph_seed=123,
                replicate_index=0,
                max_attempts=2,
            )

        seeds = [call.kwargs["seed"] for call in generator.call_args_list]
        self.assertEqual(seeds, [derive_attempt_seed(123, 0), derive_attempt_seed(123, 1)])
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(generated.metadata["generation_attempt_count"], 2)
        self.assertEqual(generated.metadata["generation_attempt_seed"], seeds[-1])

    def test_retry_rejects_a_seed_derivation_collision(self):
        with (
            patch("graph_generation.derive_attempt_seed", return_value=5),
            patch(
                "graph_generation.nx.gnp_random_graph",
                return_value=nx.empty_graph(6),
            ),
            self.assertRaisesRegex(GraphGenerationError, "seed derivation"),
        ):
            generate_connected_erdos_renyi(
                n=6,
                p=0.25,
                graph_seed=123,
                replicate_index=0,
                max_attempts=2,
            )

    def test_erdos_renyi_retry_limit_raises_without_substituting_a_component(self):
        with patch(
            "graph_generation.nx.gnp_random_graph",
            return_value=nx.empty_graph(5),
        ) as generator:
            with self.assertRaisesRegex(GraphGenerationError, "after 3 attempts"):
                generate_connected_erdos_renyi(
                    n=5,
                    p=0.1,
                    graph_seed=77,
                    replicate_index=0,
                    max_attempts=3,
                )

        seeds = [call.kwargs["seed"] for call in generator.call_args_list]
        self.assertEqual(generator.call_count, 3)
        self.assertEqual(len(set(seeds)), 3)

    def test_invalid_erdos_renyi_parameters_are_rejected(self):
        for values in (
            {"n": 1},
            {"n": 3.0},
            {"p": 0.0},
            {"p": 1.0},
            {"p": float("nan")},
            {"p": True},
            {"graph_seed": -1},
            {"graph_seed": MAX_SEED + 1},
            {"replicate_index": -1},
            {"max_attempts": 0},
        ):
            arguments = {
                "n": 6,
                "p": 0.5,
                "graph_seed": 1,
                "replicate_index": 0,
                "max_attempts": 2,
            }
            arguments.update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                generate_connected_erdos_renyi(**arguments)

    def test_invalid_barabasi_albert_parameters_are_rejected(self):
        for values in (
            {"n": 1},
            {"m": 0},
            {"m": 6},
            {"m": 1.5},
            {"graph_seed": -1},
            {"replicate_index": -1},
        ):
            arguments = {
                "n": 6,
                "m": 2,
                "graph_seed": 1,
                "replicate_index": 0,
            }
            arguments.update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                generate_connected_barabasi_albert(**arguments)

    def test_generate_graph_validates_config_indices_and_model(self):
        invalid_calls = (
            (DEVELOPMENT_CONFIG, -1, ERDOS_RENYI, 0),
            (DEVELOPMENT_CONFIG, 0, "unknown", 0),
            (DEVELOPMENT_CONFIG, 0, ERDOS_RENYI, -1),
            (object(), 0, ERDOS_RENYI, 0),
        )

        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                generate_graph(*arguments)


if __name__ == "__main__":
    unittest.main()
