"""Focused regression tests for the final Step 13 methodology freeze."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from embedding import (  # noqa: E402
    calculate_embedding_distortion,
    embedding_input_from_distance_matrix,
    prepare_embedding_input,
)
from experiment_config import (  # noqa: E402
    ANALYSIS_PLAN_HASH,
    COMBINED_FREEZE_HASH,
    CONFIGURATION_SCHEMA_VERSION,
    DATA_GENERATION_HASH,
    FEASIBILITY_PILOT_SEEDS,
    FULL_EXPERIMENT_CONFIG,
    SEED_IDENTITY_VERSION,
    canonical_fingerprint_json,
)
from experiment_protocol import (  # noqa: E402
    bootstrap_graph_indices,
    build_run_manifest,
    calculate_graph_level_estimands,
    descriptive_unpaired_model_contrast,
    equally_weighted_cell_mean,
    equally_weighted_n_m_marginal,
    graph_identity,
    ordered_pair_from_index,
    partial_spearman_by_n_m_stratum,
    percentile_bootstrap_interval,
    sample_ordered_pairs,
)
from graph_generation import generate_graph  # noqa: E402
import hydra_embedding as hydra_module  # noqa: E402
from hydra_embedding import HydraEmbeddingError, embed_hydra  # noqa: E402
from mds_embedding import ClassicalMDSError, classical_mds  # noqa: E402
from network_metrics import prepare_all_pairs_shortest_paths  # noqa: E402
from routing import (  # noqa: E402
    NO_ALTERNATIVE_AFTER_BACKTRACKING,
    POST_REPAIR_ATTEMPTED_REVISIT,
    POST_REPAIR_LOCAL_MINIMUM,
    REPAIR_UNAVAILABLE_AT_SOURCE,
    RoutingInvariantError,
    dijkstra_benchmark,
)


EXPECTED_DATA_GENERATION_HASH = (
    "d7c37cd573e96a0f7c5178d83721c596e4451a9277fef9591d8a319df89611d7"
)
EXPECTED_ANALYSIS_PLAN_HASH = (
    "a3650d2ad45c935500334fa145df2880702059db480f9eb50f1558e8229045d8"
)
EXPECTED_COMBINED_FREEZE_HASH = (
    "8e002ef20f96a4f66c80440c9734cd28b6c0851a95a7977d5e2b7cf905f7a78a"
)


def embedding_input_for(graph: nx.Graph, identity: str = "step13"):
    return prepare_embedding_input(
        graph,
        prepare_all_pairs_shortest_paths(graph),
        configuration_fingerprint=identity,
    )


class ConfigurationFreezeTests(unittest.TestCase):
    def test_schema_and_seed_identity_versions_are_separate(self):
        self.assertEqual(CONFIGURATION_SCHEMA_VERSION, 4)
        self.assertEqual(SEED_IDENTITY_VERSION, 3)
        snapshot = FULL_EXPERIMENT_CONFIG.as_dict()
        self.assertEqual(snapshot["configuration_schema_version"], 4)
        self.assertEqual(snapshot["seed_identity_version"], 3)
        self.assertIn(
            "seed_identity_version",
            FULL_EXPERIMENT_CONFIG.seed_derivation_metadata["identity_fields"],
        )
        self.assertNotIn(
            "configuration_schema_version",
            FULL_EXPERIMENT_CONFIG.seed_derivation_metadata["identity_fields"],
        )

    def test_pre_freeze_graph_and_attempt_seeds_are_unchanged(self):
        expected_graph_seeds = {
            (0, "erdos_renyi", 0): 2_595_873_943,
            (0, "barabasi_albert", 0): 3_955_446_999,
            (4, "erdos_renyi", 7): 2_997_342_518,
            (8, "barabasi_albert", 19): 1_049_214_616,
        }
        for identity, expected in expected_graph_seeds.items():
            self.assertEqual(
                FULL_EXPERIMENT_CONFIG.seeds_for_replicate(
                    *identity
                ).graph_generation,
                expected,
            )
        self.assertEqual(
            tuple(
                FULL_EXPERIMENT_CONFIG.seed_for_graph_attempt(
                    0, "erdos_renyi", 0, index
                )
                for index in (0, 1, 49)
            ),
            (2_905_629_785, 157_627_160, 4_099_013_193),
        )

    def test_master_and_excluded_seeds_are_frozen(self):
        config = FULL_EXPERIMENT_CONFIG
        self.assertEqual(config.graph_generation_master_seed, 1_000_003)
        self.assertEqual(config.embedding_initialization_master_seed, 2_000_003)
        self.assertEqual(
            config.source_destination_sampling_master_seed, 3_000_003
        )
        self.assertEqual(
            FEASIBILITY_PILOT_SEEDS,
            (4_000_003, 4_000_019, 4_000_037, 4_000_063, 4_000_099, 4_000_121),
        )

    def test_exact_degree_matched_grid_and_frozen_workload(self):
        config = FULL_EXPERIMENT_CONFIG
        self.assertEqual(config.graph_sizes, (100, 300, 1_000))
        self.assertEqual(
            {setting.ba_m for setting in config.parameter_settings},
            {4, 8, 16},
        )
        for setting in config.parameter_settings:
            numerator = 2 * setting.ba_m * (setting.n - setting.ba_m)
            denominator = setting.n * (setting.n - 1)
            self.assertEqual(setting.er_probability_numerator, numerator)
            self.assertEqual(setting.er_probability_denominator, denominator)
            self.assertEqual(setting.er_p, numerator / denominator)
        workload = config.workload_estimate
        self.assertEqual(workload["independent_graph_replicates"], 360)
        self.assertEqual(workload["sampled_ordered_pairs"], 360_000)
        self.assertEqual(
            workload["coordinate_dependent_routing_executions"], 5_400_000
        )
        self.assertEqual(workload["actual_dijkstra_executions"], 360_000)
        self.assertEqual(
            workload["total_routing_and_benchmark_executions"], 5_760_000
        )
        self.assertEqual(
            workload["distortion_metric_pair_evaluations"], 461_412_000
        )

    def test_canonical_float_encoding_and_deliberate_expected_hashes(self):
        encoded = canonical_fingerprint_json({"z": 0.5, "a": 1})
        self.assertEqual(
            encoded,
            '{"a":1,"z":{"__float64__":"0x1.0000000000000p-1"}}',
        )
        with self.assertRaises(ValueError):
            canonical_fingerprint_json({"bad": float("nan")})
        self.assertEqual(DATA_GENERATION_HASH, EXPECTED_DATA_GENERATION_HASH)
        self.assertEqual(ANALYSIS_PLAN_HASH, EXPECTED_ANALYSIS_PLAN_HASH)
        self.assertEqual(COMBINED_FREEZE_HASH, EXPECTED_COMBINED_FREEZE_HASH)

    def test_machine_readable_freeze_contains_runtime_and_analysis_rules(self):
        data = FULL_EXPERIMENT_CONFIG.data_generation_freeze_payload()
        analysis = FULL_EXPERIMENT_CONFIG.analysis_plan_freeze_payload()
        self.assertEqual(data["runtime"]["timer"], "time.perf_counter_ns")
        self.assertEqual(
            data["routing"]["defensive_step_limits"]["limit_reached"],
            "implementation_error",
        )
        self.assertFalse(
            analysis["statistical_method"][
                "null_hypothesis_significance_testing"
            ]
        )
        self.assertFalse(analysis["statistical_method"]["holm_correction"])
        self.assertEqual(
            analysis["statistical_method"]["bootstrap_replicates"], 10_000
        )
        self.assertEqual(
            analysis["network_property_associations"]["status"],
            "exploratory_non_causal",
        )


class GraphAndPairFreezeTests(unittest.TestCase):
    def test_ba_uses_explicit_star_and_records_realised_graph(self):
        with patch(
            "graph_generation.nx.barabasi_albert_graph",
            wraps=nx.barabasi_albert_graph,
        ) as generator:
            generated = generate_graph(
                FULL_EXPERIMENT_CONFIG, 0, "barabasi_albert", 0
            )
        initial = generator.call_args.kwargs["initial_graph"]
        self.assertTrue(nx.utils.graphs_equal(initial, nx.star_graph(4)))
        self.assertTrue(nx.is_connected(generated.graph))
        self.assertEqual(
            generated.metadata["realised_edge_count"],
            generated.graph.number_of_edges(),
        )

    def test_connected_er_records_exact_probability_and_attempt_provenance(self):
        generated = generate_graph(
            FULL_EXPERIMENT_CONFIG, 0, "erdos_renyi", 0
        )
        setting = FULL_EXPERIMENT_CONFIG.parameter_settings[0]
        metadata = generated.metadata
        self.assertEqual(metadata["p_exact_numerator"], 768)
        self.assertEqual(metadata["p_exact_denominator"], 9_900)
        self.assertEqual(metadata["p"], 768 / 9_900)
        self.assertEqual(metadata["p_float64_hex"], float(768 / 9_900).hex())
        self.assertEqual(
            len(metadata["generation_attempt_seeds"]),
            metadata["generation_attempt_count"],
        )
        self.assertEqual(
            metadata["rejected_disconnected_count"],
            metadata["generation_attempt_count"] - 1,
        )
        self.assertEqual(
            metadata["realised_average_degree"],
            2.0 * generated.graph.number_of_edges() / setting.n,
        )
        self.assertEqual(metadata["max_connected_graph_generation_attempts"], 50)

    def test_ordered_pair_mapping_is_exact_and_bijective(self):
        expected = (
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        )
        self.assertEqual(
            tuple(ordered_pair_from_index(3, index) for index in range(6)),
            expected,
        )

    def test_sampler_is_unique_deterministic_and_graph_separated(self):
        identity = graph_identity(
            configuration_name="full_experiment",
            setting_index=0,
            setting_label="full_n100_m4",
            model="erdos_renyi",
            replicate_index=0,
        )
        first = sample_ordered_pairs(
            range(100), 1_000, 3_000_003, graph_identity=identity
        )
        second = sample_ordered_pairs(
            range(100), 1_000, 3_000_003, graph_identity=identity
        )
        other = sample_ordered_pairs(
            range(100), 1_000, 3_000_003, graph_identity=identity + ":other"
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first[:10],
            (
                (94, 23),
                (88, 74),
                (0, 93),
                (56, 95),
                (79, 75),
                (97, 58),
                (30, 21),
                (5, 52),
                (65, 14),
                (56, 65),
            ),
        )
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(source != destination for source, destination in first))

    def test_sampler_matches_across_python_hash_seeds(self):
        script = (
            "import json,sys;"
            f"sys.path.insert(0,{str(CODE_ROOT)!r});"
            "from experiment_protocol import sample_ordered_pairs;"
            "print(json.dumps(sample_ordered_pairs(range(30),25,3000003,"
            "graph_identity='hash-seed-fixture'),separators=(',',':')))"
        )
        outputs = []
        for hash_seed in ("1", "8675309"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            outputs.append(
                subprocess.run(
                    [sys.executable, "-B", "-c", script],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
        self.assertEqual(outputs[0], outputs[1])


class SpectralAndRoutingFreezeTests(unittest.TestCase):
    def test_hydra_rank_one_is_allowed_and_rank_zero_is_rejected(self):
        rank_one = embed_hydra(embedding_input_for(nx.path_graph(5)))
        coordinates = np.asarray(
            [rank_one.coordinates[node] for node in rank_one.metadata.node_order],
            dtype=np.float64,
        )
        self.assertEqual(rank_one.metadata.effective_spatial_rank, 1)
        self.assertTrue(np.any(coordinates[:, 0] != 0.0))
        np.testing.assert_array_equal(
            coordinates[:, 1],
            np.zeros(len(coordinates), dtype=np.float64),
        )
        self.assertTrue(np.isfinite(coordinates).all())
        self.assertTrue(np.all(np.linalg.norm(coordinates, axis=1) < 1.0))
        collapsed = embedding_input_from_distance_matrix(
            np.zeros((3, 3), dtype=np.float64),
            tuple(range(3)),
            configuration_fingerprint="rank-zero",
        )
        with self.assertRaisesRegex(HydraEmbeddingError, "rank is zero"):
            embed_hydra(collapsed)

    def test_hydra_uses_global_eigenvalue_threshold(self):
        fixture = embedding_input_from_distance_matrix(
            np.ones((3, 3), dtype=np.float64) - np.eye(3),
            tuple(range(3)),
            configuration_fingerprint="hydra-global-threshold",
        )
        eigenvalues = np.asarray([-1e15, -1e14, 1.0], dtype=np.float64)
        eigenvectors = np.eye(3, dtype=np.float64)
        with patch.object(
            hydra_module.np.linalg,
            "eigh",
            return_value=(eigenvalues, eigenvectors),
        ), self.assertRaisesRegex(HydraEmbeddingError, "global threshold"):
            embed_hydra(fixture)

    def test_hydra_combined_isometry_rule_is_recorded(self):
        result = embed_hydra(embedding_input_for(nx.cycle_graph(6)))
        metadata = result.metadata
        self.assertEqual(metadata.pairwise_isometry_absolute_tolerance, 1e-10)
        self.assertEqual(metadata.pairwise_isometry_tolerance, 1e-9)
        self.assertLessEqual(metadata.maximum_pairwise_normalized_error, 1.0)
        self.assertFalse(metadata.radial_rescaling_after_centering)

    def test_partial_coincidences_are_recorded_without_jitter(self):
        fixture = embedding_input_for(nx.complete_graph(6), "k6-coincidence")
        hydra = embed_hydra(fixture)
        mds = classical_mds(fixture)
        for result in (hydra, mds):
            metadata = result.metadata
            self.assertEqual(metadata.coincident_coordinate_group_count, 1)
            self.assertEqual(metadata.coincident_vertex_count, 2)
            self.assertEqual(metadata.coincident_vertex_pair_count, 1)
            group = metadata.coincident_coordinate_groups[0]
            self.assertEqual(
                result.coordinates[group[0]],
                result.coordinates[group[1]],
            )

    def test_mds_rank_one_is_zero_padded_and_rank_zero_is_rejected(self):
        line = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        distances = np.linalg.norm(line[:, None, :] - line[None, :, :], axis=2)
        rank_one = classical_mds(
            embedding_input_from_distance_matrix(
                distances,
                tuple(range(3)),
                configuration_fingerprint="mds-rank-one",
            )
        )
        self.assertEqual(rank_one.metadata.effective_rank, 1)
        np.testing.assert_array_equal(rank_one.coordinate_matrix[:, 1], 0.0)
        collapsed = embedding_input_from_distance_matrix(
            np.zeros((3, 3), dtype=np.float64),
            tuple(range(3)),
            configuration_fingerprint="mds-rank-zero",
        )
        with self.assertRaisesRegex(ClassicalMDSError, "rank 0"):
            classical_mds(collapsed)

    def test_defensive_limit_and_dijkstra_apsp_disagreement_are_errors(self):
        graph = nx.path_graph(3)
        coordinates = {0: (-0.5, 0.0), 1: (0.0, 0.0), 2: (0.5, 0.0)}
        from routing import euclidean_greedy_route

        with self.assertRaises(RoutingInvariantError):
            euclidean_greedy_route(
                graph, coordinates, 0, 2, step_limit=1
            )
        with self.assertRaisesRegex(RoutingInvariantError, "APSP"):
            dijkstra_benchmark(
                graph, 0, 2, expected_shortest_path_length=1
            )
        self.assertEqual(
            dijkstra_benchmark(
                graph, 0, 2, expected_shortest_path_length=2
            ).route_length,
            2,
        )

    def test_repair_taxonomy_literals_are_frozen(self):
        self.assertEqual(
            REPAIR_UNAVAILABLE_AT_SOURCE, "repair_unavailable_at_source"
        )
        self.assertEqual(
            NO_ALTERNATIVE_AFTER_BACKTRACKING,
            "no_alternative_after_backtracking",
        )
        self.assertEqual(
            POST_REPAIR_LOCAL_MINIMUM, "post_repair_local_minimum"
        )
        self.assertEqual(
            POST_REPAIR_ATTEMPTED_REVISIT,
            "post_repair_attempted_revisit",
        )


class DistortionAnalysisAndManifestTests(unittest.TestCase):
    def test_both_distortion_metrics_and_uniform_scaling_invariance(self):
        graph = nx.path_graph(4)
        coordinates = {
            0: (-0.4, 0.0),
            1: (-0.1, 0.1),
            2: (0.15, -0.05),
            3: (0.45, 0.0),
        }
        euclidean = calculate_embedding_distortion(
            graph, coordinates, metric="euclidean"
        )
        poincare = calculate_embedding_distortion(
            graph, coordinates, metric="poincare"
        )
        scaled = calculate_embedding_distortion(
            graph,
            {
                node: (point[0] * 0.5, point[1] * 0.5)
                for node, point in coordinates.items()
            },
            metric="euclidean",
        )
        self.assertEqual(euclidean.metric, "euclidean")
        self.assertEqual(poincare.metric, "poincare")
        self.assertAlmostEqual(
            euclidean.mean_relative_distortion,
            scaled.mean_relative_distortion,
            places=14,
        )
        self.assertAlmostEqual(
            euclidean.rmse_relative_distortion,
            scaled.rmse_relative_distortion,
            places=14,
        )

    def test_graph_estimands_and_recovery_zero_denominator(self):
        all_success = [True] * 1_000
        result = calculate_graph_level_estimands(
            all_success, all_success, all_success
        )
        self.assertEqual(result.poincare_advantage, 0.0)
        self.assertEqual(result.repair_improvement, 0.0)
        self.assertIsNone(result.repair_recovery.value)
        self.assertEqual(result.repair_recovery.display, "N/A")
        self.assertEqual(result.repair_recovery.denominator, 0)

        poincare = [False] * 100 + [True] * 900
        repaired = [True] * 40 + [False] * 60 + [True] * 900
        recovered = calculate_graph_level_estimands(
            [True] * 1_000, poincare, repaired
        )
        self.assertEqual(recovered.repair_recovery.numerator, 40)
        self.assertEqual(recovered.repair_recovery.denominator, 100)
        self.assertEqual(recovered.repair_recovery.value, 0.4)

    def test_equal_graph_and_stratum_weighting(self):
        self.assertEqual(
            equally_weighted_cell_mean([0.25] * 20),
            0.25,
        )
        strata = {
            (n, m): float(index)
            for index, (n, m) in enumerate(
                ( (n, m) for n in (100, 300, 1_000) for m in (4, 8, 16) )
            )
        }
        self.assertEqual(equally_weighted_n_m_marginal(strata), 4.0)
        self.assertAlmostEqual(
            descriptive_unpaired_model_contrast(
                [0.3] * 20,
                [0.1] * 20,
            ),
            0.2,
        )

    def test_bootstrap_is_exact_deterministic_and_model_separated(self):
        er_first = bootstrap_graph_indices(
            bootstrap_replicate=17,
            model="erdos_renyi",
            n=100,
            m=4,
        )
        er_second = bootstrap_graph_indices(
            bootstrap_replicate=17,
            model="erdos_renyi",
            n=100,
            m=4,
        )
        ba = bootstrap_graph_indices(
            bootstrap_replicate=17,
            model="barabasi_albert",
            n=100,
            m=4,
        )
        self.assertEqual(er_first, er_second)
        self.assertEqual(
            er_first,
            (10, 17, 14, 17, 4, 1, 18, 6, 12, 1, 18, 6, 9, 19, 13, 12, 9, 5, 11, 13),
        )
        self.assertNotEqual(er_first, ba)
        self.assertEqual(len(er_first), 20)
        self.assertTrue(all(0 <= index < 20 for index in er_first))

    def test_percentile_interval_uses_frozen_nearest_rank_rule(self):
        interval = percentile_bootstrap_interval(range(10_000))
        self.assertEqual(interval.lower, 249.0)
        self.assertEqual(interval.upper, 9_749.0)
        self.assertEqual(interval.confidence_level, 0.95)
        self.assertEqual(
            interval.quantile_rule,
            "noninterpolated_nearest_rank_order_statistics_v1",
        )

    def test_partial_spearman_adjusts_strata(self):
        strata = [(100, 4)] * 3 + [(300, 4)] * 3
        first = [1, 2, 3, 10, 11, 12]
        second = [2, 4, 6, 20, 22, 24]
        self.assertAlmostEqual(
            partial_spearman_by_n_m_stratum(first, second, strata),
            1.0,
        )

    def test_run_manifest_is_complete_and_final_guard_is_enforced(self):
        fixed_time = datetime(2026, 7, 24, tzinfo=timezone.utc)
        manifest = build_run_manifest(
            "fixture-graph",
            timestamp=fixed_time,
        )
        self.assertEqual(manifest["data_generation_hash"], DATA_GENERATION_HASH)
        self.assertEqual(manifest["analysis_plan_hash"], ANALYSIS_PLAN_HASH)
        self.assertEqual(manifest["combined_freeze_hash"], COMBINED_FREEZE_HASH)
        self.assertEqual(manifest["graph_identity"], "fixture-graph")
        self.assertTrue(manifest["dependency_versions"])
        self.assertEqual(
            manifest["timestamp_utc"], "2026-07-24T00:00:00+00:00"
        )
        with patch(
            "experiment_protocol._git_state",
            return_value=("a" * 40, True),
        ), self.assertRaisesRegex(RuntimeError, "clean committed"):
            build_run_manifest(
                "fixture-graph",
                require_final_scientific_source=True,
            )


if __name__ == "__main__":
    unittest.main()
