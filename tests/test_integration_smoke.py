from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from experiment_config import (  # noqa: E402
    DEVELOPMENT_CONFIG,
    FULL_EXPERIMENT_CONFIG,
    MAX_SEED,
)
import run_development_smoke as smoke_module  # noqa: E402
from run_development_smoke import (  # noqa: E402
    main,
    run_development_smoke,
    run_methods_for_pair,
    sample_ordered_pairs,
)


class OrderedPairSamplingTests(unittest.TestCase):
    def test_sampling_is_deterministic_unique_and_ordered(self):
        first = sample_ordered_pairs(range(6), 12, 31415)
        second = sample_ordered_pairs(reversed(range(6)), 12, 31415)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(source != destination for source, destination in first))

    def test_different_pair_seeds_can_change_the_sample(self):
        self.assertNotEqual(
            sample_ordered_pairs(range(8), 10, 1),
            sample_ordered_pairs(range(8), 10, 2),
        )

    def test_sampling_rejects_invalid_requests(self):
        invalid_calls = (
            ((0,), 1, 1),
            ((0, 0), 1, 1),
            ((0, "1"), 1, 1),
            ((0, 1), -1, 1),
            ((0, 1), 3, 1),
            ((0, 1), 1, -1),
            ((0, 1), 1, MAX_SEED + 1),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                sample_ordered_pairs(*arguments)


class IntegrationSmokeTests(unittest.TestCase):
    def test_hand_built_fixture_has_known_successful_outcomes(self):
        graph = nx.path_graph(3)
        coordinates = {
            0: (-0.4, 0.0),
            1: (0.0, 0.0),
            2: (0.4, 0.0),
        }

        records = run_methods_for_pair(
            graph,
            coordinates,
            0,
            2,
            tolerance=1e-12,
        )

        self.assertEqual(len(records), 4)
        self.assertTrue(all(record.success for record in records))
        self.assertTrue(all(record.route_length == 2 for record in records))
        self.assertTrue(all(record.walk == (0, 1, 2) for record in records))
        self.assertEqual(records[0].stretch, None)
        self.assertTrue(all(record.stretch == 1.0 for record in records[1:]))
        self.assertFalse(records[-1].repair_attempted)

    def test_end_to_end_development_smoke_uses_er_ba_and_shared_pairs(self):
        with patch.object(
            smoke_module,
            "prepare_routing_coordinates",
            wraps=smoke_module.prepare_routing_coordinates,
        ) as prepare_coordinates:
            result = run_development_smoke()

        self.assertEqual(result.configuration_name, "development")
        self.assertEqual(prepare_coordinates.call_count, 4)
        self.assertEqual(
            [call.kwargs["metric_name"] for call in prepare_coordinates.call_args_list],
            ["euclidean", "poincare", "euclidean", "poincare"],
        )
        self.assertEqual(
            tuple(graph.graph_model for graph in result.graphs),
            ("erdos_renyi", "barabasi_albert"),
        )
        for graph_record in result.graphs:
            self.assertEqual(len(graph_record.source_destination_pairs), 5)
            self.assertEqual(len(graph_record.method_records), 20)
            for pair in graph_record.source_destination_pairs:
                matching = [
                    record
                    for record in graph_record.method_records
                    if (record.source, record.destination) == pair
                ]
                self.assertEqual(len(matching), 4)
                self.assertLessEqual(
                    max(record.repair_attempt_count for record in matching), 1
                )

    def test_smoke_result_is_deterministic(self):
        self.assertEqual(run_development_smoke(), run_development_smoke())

    def test_smoke_result_has_complete_json_safe_provenance(self):
        result = run_development_smoke()
        payload = result.as_dict()
        encoded = result.to_json()

        self.assertEqual(json.loads(encoded), payload)
        json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertEqual(payload["result_schema_version"], 2)
        self.assertEqual(payload["configuration_name"], "development")
        self.assertEqual(payload["configuration"]["name"], "development")
        self.assertEqual(
            result.configuration_fingerprint,
            DEVELOPMENT_CONFIG.configuration_fingerprint,
        )
        self.assertEqual(len(result.configuration_fingerprint), 64)
        self.assertEqual(len(result.implementation_source_fingerprint), 64)
        self.assertEqual(
            result.implementation_source_fingerprint,
            smoke_module.implementation_source_fingerprint(),
        )
        runtime = dict(result.runtime_metadata)
        self.assertEqual(
            set(runtime),
            {
                "networkx_version",
                "numpy_version",
                "python_implementation",
                "python_version",
            },
        )

        for graph_record in result.graphs:
            graph_metadata = dict(graph_record.graph_metadata)
            embedding = dict(graph_record.embedding_metadata)
            self.assertEqual(
                embedding["embedding_id"],
                "dense_fruchterman_reingold_rescaled_v1",
            )
            self.assertEqual(embedding["embedding_radius"], 0.85)
            self.assertEqual(embedding["embedding_iterations"], 100)
            self.assertIsInstance(embedding["embedding_seed"], int)
            self.assertEqual(
                embedding["embedding_seed"],
                graph_metadata["embedding_initialization_seed"],
            )
            self.assertEqual(
                graph_metadata["configuration_fingerprint"],
                result.configuration_fingerprint,
            )

            sampling = dict(graph_record.pair_sampling_metadata)
            self.assertEqual(
                sampling["algorithm"],
                "blake2s_uint64_rejection_without_replacement_v1",
            )
            self.assertTrue(sampling["ordered_pairs_are_unique"])
            self.assertIsInstance(
                sampling["source_destination_sampling_seed"], int
            )
            self.assertEqual(
                sampling["source_destination_sampling_seed"],
                graph_metadata["source_destination_sampling_seed"],
            )

    def test_full_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            run_development_smoke(FULL_EXPERIMENT_CONFIG)

    def test_configuration_name_cannot_bypass_development_safeguard(self):
        renamed_full = replace(
            FULL_EXPERIMENT_CONFIG, name=DEVELOPMENT_CONFIG.name
        )
        changed_development = replace(
            DEVELOPMENT_CONFIG, source_destination_pairs_per_graph=1
        )
        for config in (renamed_full, changed_development):
            with self.subTest(config=config), self.assertRaises(ValueError):
                run_development_smoke(config)

    def test_command_is_reproducible_across_processes_and_hash_seeds(self):
        command = (
            "import sys;"
            "sys.path.insert(0, 'code');"
            "from run_development_smoke import run_development_smoke;"
            "print(run_development_smoke().to_json())"
        )

        def run_in_subprocess(hash_seed: str) -> subprocess.CompletedProcess[str]:
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            return subprocess.run(
                [sys.executable, "-B", "-c", command],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

        first = run_in_subprocess("1")
        second = run_in_subprocess("987654321")
        self.assertEqual(first.returncode, 0, msg=first.stderr)
        self.assertEqual(second.returncode, 0, msg=second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        payload = json.loads(first.stdout)
        self.assertEqual(payload["configuration_name"], "development")

    def test_command_entry_point_succeeds_and_labels_output(self):
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("not final results", output.getvalue())
        self.assertIn("fixed_pairs=", output.getvalue())


if __name__ == "__main__":
    unittest.main()
