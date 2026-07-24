from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from experiment_config import (  # noqa: E402
    APPROVED_EMBEDDING_DESIGN,
    APPROVED_EMBEDDING_FAMILIES,
    BA_FINITE_DEGREE_MATCH_RULE,
    CONFIGURATIONS,
    CONFIGURATION_SCHEMA_VERSION,
    EMBEDDING_METHOD,
    BARABASI_ALBERT,
    DEVELOPMENT_CONFIG,
    ERDOS_RENYI,
    FULL_EXPERIMENT_CONFIG,
    HYDRA_CONDITION_ID,
    HYDRA_CURVATURE,
    HYDRA_KAPPA,
    MAX_SEED,
    MDS_BASE_EMBEDDING_ID,
    MDS_CONDITION_IDS,
    MDS_EUCLIDEAN_TOLERANCE_POLICY,
    MDS_MAXIMUM_RADII,
    SEED_SPACE_SIZE,
    DegreeMatchedParameters,
    ExperimentConfig,
    audit_feasibility_pilot_seed_collisions,
    audit_seed_collisions,
    derive_domain_seed,
    get_config,
    iter_seed_uses,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_development_configuration_is_valid_and_small(self):
        config = DEVELOPMENT_CONFIG

        self.assertEqual(config.graph_sizes, (30,))
        self.assertEqual(config.graph_repetitions, 2)
        self.assertEqual(config.source_destination_pairs_per_graph, 25)
        self.assertEqual(config.graph_replicate_count, 8)
        self.assertEqual(config.embedding_method, EMBEDDING_METHOD)
        self.assertEqual(config.embedding_radius, 0.85)
        self.assertEqual(config.embedding_iterations, 100)
        self.assertEqual(config.development_embedding_radius, 0.85)
        self.assertEqual(config.spring_layout_iterations, 100)
        self.assertIs(config.approved_embedding_design, APPROVED_EMBEDDING_DESIGN)
        self.assertFalse(config.is_provisional)

    def test_full_configuration_is_valid_and_marked_provisional(self):
        config = FULL_EXPERIMENT_CONFIG

        self.assertEqual(config.graph_sizes, (100, 300, 1_000))
        self.assertEqual(config.graph_repetitions, 20)
        self.assertEqual(config.source_destination_pairs_per_graph, 1_000)
        self.assertEqual(config.max_connected_graph_generation_attempts, 50)
        self.assertEqual(config.graph_replicate_count, 360)
        self.assertEqual(config.sampled_ordered_pair_count, 360_000)
        self.assertEqual(config.embedding_method, EMBEDDING_METHOD)
        self.assertEqual(
            config.approved_embedding_design.embedding_families,
            APPROVED_EMBEDDING_FAMILIES,
        )
        self.assertEqual(
            config.approved_embedding_design.coordinate_condition_ids,
            (HYDRA_CONDITION_ID, *MDS_CONDITION_IDS),
        )
        self.assertTrue(config.is_provisional)
        self.assertTrue(config.provisional_values)

    def test_named_configuration_lookup(self):
        self.assertIs(get_config("development"), DEVELOPMENT_CONFIG)
        self.assertIs(get_config("full_experiment"), FULL_EXPERIMENT_CONFIG)
        with self.assertRaises(ValueError):
            get_config("unknown")
        with self.assertRaises(TypeError):
            CONFIGURATIONS["development"] = FULL_EXPERIMENT_CONFIG

    def test_finite_networkx_ba_degree_matching_is_within_tolerance(self):
        for config in (DEVELOPMENT_CONFIG, FULL_EXPERIMENT_CONFIG):
            for setting in config.parameter_settings:
                with self.subTest(config=config.name, setting=setting.label):
                    expected_edges = setting.ba_m * (setting.n - setting.ba_m)
                    expected_average_degree = 2 * expected_edges / setting.n
                    self.assertEqual(setting.ba_exact_edge_count, expected_edges)
                    self.assertEqual(
                        setting.ba_exact_average_degree,
                        expected_average_degree,
                    )
                    self.assertLessEqual(
                        setting.expected_degree_gap,
                        config.expected_degree_match_tolerance,
                    )
                    self.assertAlmostEqual(
                        setting.er_p * (setting.n - 1),
                        expected_average_degree,
                    )
                    self.assertEqual(
                        setting.as_dict()["degree_match_rule"],
                        BA_FINITE_DEGREE_MATCH_RULE,
                    )

    def test_seed_derivation_is_deterministic(self):
        config = FULL_EXPERIMENT_CONFIG

        first = config.seeds_for_replicate(3, BARABASI_ALBERT, 7)
        second = config.seeds_for_replicate(3, BARABASI_ALBERT, 7)

        self.assertEqual(first, second)

        self.assertEqual(
            derive_domain_seed(123, "domain_a", "setting", 1),
            derive_domain_seed(123, "domain_a", "setting", 1),
        )
        self.assertNotEqual(
            derive_domain_seed(123, "domain_a", "setting", 1),
            derive_domain_seed(123, "domain_b", "setting", 1),
        )

    def test_replicate_seed_identity_includes_setting_index_and_label(self):
        config = FULL_EXPERIMENT_CONFIG
        setting = config.parameter_settings[3]
        expected = config.seeds_for_replicate(3, BARABASI_ALBERT, 7)
        reordered = replace(
            config,
            parameter_settings=tuple(reversed(config.parameter_settings)),
        )
        reordered_index = next(
            index
            for index, candidate in enumerate(reordered.parameter_settings)
            if candidate.label == setting.label
        )
        self.assertNotEqual(
            expected,
            reordered.seeds_for_replicate(
                reordered_index,
                BARABASI_ALBERT,
                7,
            ),
        )

    def test_seed_domains_and_graph_attempt_identities_are_separate(self):
        config = DEVELOPMENT_CONFIG
        seeds = config.seeds_for_replicate(0, ERDOS_RENYI, 0)
        attempts = tuple(
            config.seed_for_graph_attempt(0, ERDOS_RENYI, 0, index)
            for index in range(config.max_connected_graph_generation_attempts)
        )

        self.assertEqual(len(attempts), len(set(attempts)))
        self.assertEqual(
            len({*attempts, *seeds.as_dict().values()}),
            len(attempts) + 3,
        )
        self.assertNotEqual(
            attempts[0],
            config.seed_for_graph_attempt(0, ERDOS_RENYI, 1, 0),
        )
        with self.assertRaises(ValueError):
            config.seed_for_graph_attempt(0, BARABASI_ALBERT, 0, 0)

    def test_configured_seed_grids_have_no_collisions(self):
        expected_use_counts = {
            "development": 124,
            "full_experiment": 10_080,
        }
        combined_seeds: list[int] = []
        for config in (DEVELOPMENT_CONFIG, FULL_EXPERIMENT_CONFIG):
            uses = tuple(iter_seed_uses(config))
            with self.subTest(config=config.name):
                self.assertEqual(len(uses), expected_use_counts[config.name])
                self.assertEqual(audit_seed_collisions(config), ())
            combined_seeds.extend(use.seed for use in uses)
        self.assertEqual(len(combined_seeds), len(set(combined_seeds)))
        self.assertEqual(
            audit_feasibility_pilot_seed_collisions(FULL_EXPERIMENT_CONFIG),
            (),
        )
        pilot_seeds = set(
            FULL_EXPERIMENT_CONFIG
            .approved_embedding_design
            .feasibility_pilot_seeds
        )
        self.assertTrue(pilot_seeds.isdisjoint(combined_seeds))

    def test_every_graph_replicate_receives_different_seeds_in_each_stream(self):
        for config in (DEVELOPMENT_CONFIG, FULL_EXPERIMENT_CONFIG):
            replicates = tuple(config.iter_graph_replicates())
            with self.subTest(config=config.name):
                for field_name in (
                    "graph_generation",
                    "embedding_initialization",
                    "source_destination_sampling",
                ):
                    seeds = [
                        getattr(replicate.seeds, field_name)
                        for replicate in replicates
                    ]
                    self.assertEqual(len(seeds), config.graph_replicate_count)
                    self.assertEqual(len(set(seeds)), len(seeds))

    def test_models_and_parameters_are_recorded_for_each_replicate(self):
        replicates = list(DEVELOPMENT_CONFIG.iter_graph_replicates())

        self.assertEqual(replicates[0].model, ERDOS_RENYI)
        self.assertIsInstance(replicates[0].parameter, float)
        self.assertEqual(replicates[2].model, BARABASI_ALBERT)
        self.assertIsInstance(replicates[2].parameter, int)

    def test_invalid_parameter_settings_are_rejected(self):
        invalid_cases = (
            {"label": "bad_n", "n": 1, "er_p": 0.5, "ba_m": 1},
            {"label": "bad_p_zero", "n": 10, "er_p": 0.0, "ba_m": 1},
            {"label": "bad_p_one", "n": 10, "er_p": 1.0, "ba_m": 1},
            {"label": "bad_m_zero", "n": 10, "er_p": 0.5, "ba_m": 0},
            {"label": "bad_m_n", "n": 10, "er_p": 0.5, "ba_m": 10},
        )

        for values in invalid_cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                DegreeMatchedParameters(**values)

    def test_non_positive_counts_are_rejected(self):
        for field_name in (
            "graph_repetitions",
            "source_destination_pairs_per_graph",
            "max_connected_graph_generation_attempts",
        ):
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                replace(DEVELOPMENT_CONFIG, **{field_name: 0})

        with self.assertRaises(ValueError):
            replace(
                DEVELOPMENT_CONFIG,
                max_connected_graph_generation_attempts=SEED_SPACE_SIZE + 1,
            )

    def test_configuration_serialization_is_stable_and_complete(self):
        config = FULL_EXPERIMENT_CONFIG
        first = config.to_json()
        second = config.to_json()
        snapshot = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual(
            snapshot["configuration_schema_version"],
            CONFIGURATION_SCHEMA_VERSION,
        )
        force = snapshot["embedding"]["development_force_only"]
        approved = snapshot["embedding"]["approved_design"]
        self.assertEqual(force["method"], EMBEDDING_METHOD)
        self.assertFalse(force["final_experiment_default"])
        self.assertEqual(
            approved["embedding_families"],
            list(APPROVED_EMBEDDING_FAMILIES),
        )
        self.assertEqual(
            approved["hydra"]["condition_id"],
            HYDRA_CONDITION_ID,
        )
        self.assertEqual(approved["hydra"]["kappa"], HYDRA_KAPPA)
        self.assertEqual(
            approved["hydra"]["sectional_curvature"],
            HYDRA_CURVATURE,
        )
        self.assertEqual(
            approved["classical_mds"]["base_embedding_id"],
            MDS_BASE_EMBEDDING_ID,
        )
        self.assertEqual(
            approved["classical_mds"]["maximum_radii"],
            list(MDS_MAXIMUM_RADII),
        )
        self.assertEqual(
            approved["classical_mds"]["condition_ids"],
            list(MDS_CONDITION_IDS),
        )
        self.assertEqual(
            approved["classical_mds"][
                "euclidean_routing_tolerance_policy"
            ],
            MDS_EUCLIDEAN_TOLERANCE_POLICY,
        )
        self.assertTrue(
            approved["classical_mds"][
                "radii_are_nested_sensitivity_transformations"
            ]
        )
        self.assertEqual(
            snapshot["parameter_settings"][0]["degree_match_rule"],
            BA_FINITE_DEGREE_MATCH_RULE,
        )
        self.assertEqual(snapshot["workload"], config.workload_estimate)
        self.assertEqual(len(config.configuration_fingerprint), 64)
        self.assertEqual(
            config.configuration_fingerprint,
            FULL_EXPERIMENT_CONFIG.configuration_fingerprint,
        )

    def test_full_workload_estimate_covers_all_major_counts(self):
        workload = FULL_EXPERIMENT_CONFIG.workload_estimate
        self.assertEqual(workload["graph_replicates"], 360)
        self.assertEqual(workload["erdos_renyi_graph_replicates"], 180)
        self.assertEqual(workload["barabasi_albert_graph_replicates"], 180)
        self.assertEqual(workload["sampled_ordered_pairs"], 360_000)
        self.assertEqual(workload["independent_embedding_families"], 2)
        self.assertEqual(workload["coordinate_conditions_per_graph"], 5)
        self.assertEqual(workload["hydra_embedding_runs"], 360)
        self.assertEqual(workload["mds_base_embedding_runs"], 360)
        self.assertEqual(workload["independent_embedding_family_runs"], 720)
        self.assertEqual(workload["mds_nested_radius_transformations"], 1_440)
        self.assertEqual(workload["dijkstra_routing_runs"], 360_000)
        self.assertEqual(workload["euclidean_greedy_routing_runs"], 1_800_000)
        self.assertEqual(workload["hyperbolic_greedy_routing_runs"], 1_800_000)
        self.assertEqual(
            workload["repaired_hyperbolic_greedy_routing_runs"],
            1_800_000,
        )
        self.assertEqual(workload["routing_method_runs"], 5_760_000)
        self.assertEqual(
            workload["distortion_unordered_pairs_per_condition"],
            65_916_000,
        )
        self.assertEqual(workload["distortion_unordered_pairs"], 329_580_000)
        self.assertEqual(
            workload["maximum_erdos_renyi_generation_attempts"],
            9_000,
        )
        self.assertEqual(workload["maximum_graph_generation_calls"], 9_180)

    def test_invalid_master_seeds_are_rejected(self):
        for seed in (-1, MAX_SEED + 1, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                replace(DEVELOPMENT_CONFIG, graph_generation_master_seed=seed)

        with self.assertRaises(ValueError):
            replace(
                DEVELOPMENT_CONFIG,
                embedding_initialization_master_seed=(
                    DEVELOPMENT_CONFIG.graph_generation_master_seed
                ),
            )

    def test_non_positive_tolerances_are_rejected(self):
        for field_name in (
            "numerical_tolerance",
            "expected_degree_match_tolerance",
        ):
            for value in (0.0, -1.0, float("inf"), float("nan"), 10**10_000):
                with (
                    self.subTest(field=field_name, value=value),
                    self.assertRaises(ValueError),
                ):
                    replace(DEVELOPMENT_CONFIG, **{field_name: value})

    def test_unit_disk_epsilon_must_be_strictly_between_zero_and_one(self):
        for epsilon in (0.0, -1.0, 1.0, 2.0, float("inf"), float("nan")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                replace(DEVELOPMENT_CONFIG, unit_disk_boundary_epsilon=epsilon)

    def test_invalid_embedding_settings_are_rejected(self):
        for radius in (0.0, -1.0, 1.0, float("inf"), float("nan")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                replace(DEVELOPMENT_CONFIG, embedding_radius=radius)

        with self.assertRaises(ValueError):
            replace(
                DEVELOPMENT_CONFIG,
                embedding_radius=(
                    1 - DEVELOPMENT_CONFIG.unit_disk_boundary_epsilon
                ),
            )

        for iterations in (0, -1, 1.5, True):
            with self.subTest(iterations=iterations), self.assertRaises(ValueError):
                replace(DEVELOPMENT_CONFIG, embedding_iterations=iterations)

        for field_name in ("embedding_algorithm", "embedding_method"):
            for value in ("", 1, None):
                with (
                    self.subTest(field=field_name, value=value),
                    self.assertRaises(ValueError),
                ):
                    replace(DEVELOPMENT_CONFIG, **{field_name: value})

        with self.assertRaises(ValueError):
            replace(DEVELOPMENT_CONFIG, approved_embedding_design=None)

    def test_invalid_approved_embedding_design_is_rejected(self):
        design = APPROVED_EMBEDDING_DESIGN
        invalid_changes = (
            {"embedding_families": ("classical_mds", "hydra")},
            {"hydra_dimension": 3},
            {"hydra_kappa": 2.0},
            {"hydra_curvature": -2.0},
            {"hydra_centering_tolerance": 0.0},
            {"hydra_centering_max_iterations": 0},
            {"hydra_boundary_roundoff_tolerance": 0.1},
            {"mds_dimension": 3},
            {"mds_maximum_radii": (0.5, 0.7, 0.85, 0.9)},
            {"mds_condition_ids": ("a", "b", "c", "d")},
            {"mds_euclidean_tolerance_policy": "fixed_absolute"},
            {"feasibility_pilot_seeds": (4_000_003, 4_000_003)},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(design, **changes)

    def test_approved_embedding_settings_affect_fingerprint(self):
        changed_design = replace(
            APPROVED_EMBEDDING_DESIGN,
            hydra_centering_tolerance=2e-10,
        )
        changed_config = replace(
            DEVELOPMENT_CONFIG,
            approved_embedding_design=changed_design,
        )
        self.assertNotEqual(
            changed_config.configuration_fingerprint,
            DEVELOPMENT_CONFIG.configuration_fingerprint,
        )

    def test_ordered_pair_request_cannot_exceed_available_pairs(self):
        # n=30 permits 30 * 29 = 870 ordered pairs.
        with self.assertRaises(ValueError):
            replace(
                DEVELOPMENT_CONFIG,
                source_destination_pairs_per_graph=871,
            )

    def test_duplicate_parameters_are_rejected(self):
        setting = DEVELOPMENT_CONFIG.parameter_settings[0]
        with self.assertRaises(ValueError):
            replace(
                DEVELOPMENT_CONFIG,
                parameter_settings=(setting, setting),
            )

    def test_expected_degree_mismatch_is_rejected(self):
        mismatched = DegreeMatchedParameters(
            label="mismatched",
            n=30,
            er_p=0.5,
            ba_m=2,
        )

        with self.assertRaises(ValueError):
            replace(
                DEVELOPMENT_CONFIG,
                parameter_settings=(mismatched,),
            )

    def test_invalid_tie_break_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(DEVELOPMENT_CONFIG, routing_tie_break_rule="random")
        with self.assertRaises(ValueError):
            replace(DEVELOPMENT_CONFIG, routing_tie_break_description="")

    def test_replicate_seed_request_validates_indices_and_model(self):
        invalid_calls = (
            (-1, ERDOS_RENYI, 0),
            (len(DEVELOPMENT_CONFIG.parameter_settings), ERDOS_RENYI, 0),
            (0, "unknown", 0),
            (0, ERDOS_RENYI, -1),
            (0, ERDOS_RENYI, DEVELOPMENT_CONFIG.graph_repetitions),
        )

        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                DEVELOPMENT_CONFIG.seeds_for_replicate(*arguments)

    def test_provisional_configuration_identifies_provisional_values(self):
        with self.assertRaises(ValueError):
            replace(
                FULL_EXPERIMENT_CONFIG,
                provisional_values=(),
            )


if __name__ == "__main__":
    unittest.main()
