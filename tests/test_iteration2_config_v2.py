from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

import iteration2_config as config  # noqa: E402


class Iteration2ProtocolIdentityTests(unittest.TestCase):
    def test_reporting_repairs_preserve_data_and_analysis_identities(self) -> None:
        self.assertEqual(
            config.DATA_GENERATION_HASH,
            "0010973885fe22195397a3e7775edd1af87801be80f1b286ea5492840e53b004",
        )
        self.assertEqual(
            config.ANALYSIS_PLAN_HASH,
            "56f9bb5914b196434ceaa9b600a91b338433ca64eaa1796f3ad0fd336fbe0e95",
        )
        self.assertNotEqual(
            config.OUTPUT_SCHEMA_HASH,
            "9559b30407472fb42476aaac47dd06b7ecc4dae04a3eb6ceb005c69e05aadc80",
        )
        self.assertNotEqual(
            config.COMBINED_PROTOCOL_HASH,
            "3d1604ef17b1394b76ab1abf3b7bd17278469da0d74f62c422aad2b70edb0e5a",
        )
        workbook = config.output_schema_payload()["workbook"]
        self.assertEqual(
            workbook["semantic_registry_schema"],
            config.REPORTING_SEMANTIC_REGISTRY_SCHEMA,
        )
        self.assertEqual(len(workbook["semantic_registry_contract_sha256"]), 64)

    def test_schedule_workload_and_raw_layout_are_exact(self) -> None:
        schedule = config.full_schedule()
        self.assertEqual(len(schedule), 360)
        self.assertEqual(len({item.graph_id for item in schedule}), 360)
        self.assertEqual(config.PAIRS_PER_GRAPH * len(schedule), 360_000)
        self.assertEqual(360_000 * 28, 10_080_000)
        self.assertEqual(config.ITERATION2_RAW_TOTAL_FILE_COUNT, 1_082)
        self.assertEqual(
            config.ITERATION2_RAW_TOTAL_FILE_COUNT,
            config.ITERATION2_GRAPH_CHECKPOINT_DIRECTORY_COUNT
            * config.ITERATION2_GRAPH_CHECKPOINT_FILES_PER_DIRECTORY
            + config.ITERATION2_RAW_RUN_LEVEL_FILE_COUNT,
        )

    def test_all_data_seed_domains_are_collision_free(self) -> None:
        self.assertEqual(config.audit_new_seed_uniqueness(), ())
        seeds = [config.seeds_for_graph(spec) for spec in config.full_schedule()]
        flattened = {
            value
            for item in seeds
            for value in (
                item.graph,
                item.embedding_provenance,
                item.pairs,
                item.routing_priority,
                item.validation_sentinel,
                *item.er_attempts,
            )
        }
        expected = sum(5 + len(item.er_attempts) for item in seeds)
        self.assertEqual(len(flattened), expected)

    def test_reporting_and_analysis_changes_do_not_change_raw_randomness(self) -> None:
        spec = config.full_schedule()[137]
        nodes = range(spec.n)
        before_seeds = config.seeds_for_graph(spec)
        before_pairs = config.sample_ordered_pairs(
            nodes,
            config.PAIRS_PER_GRAPH,
            graph_id=spec.graph_id,
            pair_seed=before_seeds.pairs,
        )
        before_sentinel = config.sentinel_pair_indices(
            spec.graph_id, config.PAIRS_PER_GRAPH
        )
        with (
            patch.object(config, "COMBINED_PROTOCOL_HASH", "f" * 64),
            patch.object(config, "OUTPUT_SCHEMA_HASH", "e" * 64),
            patch.object(config, "ANALYSIS_PLAN_HASH", "d" * 64),
        ):
            after_seeds = config.seeds_for_graph(spec)
            after_pairs = config.sample_ordered_pairs(
                nodes,
                config.PAIRS_PER_GRAPH,
                graph_id=spec.graph_id,
                pair_seed=after_seeds.pairs,
            )
            after_sentinel = config.sentinel_pair_indices(
                spec.graph_id, config.PAIRS_PER_GRAPH
            )
        self.assertEqual(before_seeds, after_seeds)
        self.assertEqual(before_pairs, after_pairs)
        self.assertEqual(before_sentinel, after_sentinel)

    def test_analysis_randomness_uses_only_analysis_identity(self) -> None:
        before = config.bootstrap_indices(
            model="erdos_renyi", n=100, m=4, replicate=3
        )
        with patch.object(config, "COMBINED_PROTOCOL_HASH", "0" * 64):
            self.assertEqual(
                before,
                config.bootstrap_indices(
                    model="erdos_renyi", n=100, m=4, replicate=3
                ),
            )
        with patch.object(config, "ANALYSIS_PLAN_HASH", "1" * 64):
            self.assertNotEqual(
                before,
                config.bootstrap_indices(
                    model="erdos_renyi", n=100, m=4, replicate=3
                ),
            )

    def test_canonical_serialization_is_versioned_utf8_and_finite_only(self) -> None:
        encoded = config.canonical_json_bytes({"label": "Poincaré", "x": 0.5})
        self.assertIn(config.CANONICAL_SERIALIZATION_SCHEMA.encode(), encoded)
        self.assertIn("Poincaré".encode("utf-8"), encoded)
        self.assertIn(b"0x1.0000000000000p-1", encoded)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                config.canonical_json_bytes({"value": value})

    def test_property_families_are_four_complete_54_hypothesis_families(self) -> None:
        payload = config.analysis_payload()["property_associations"]
        self.assertEqual(len(payload["properties"]), 6)
        self.assertEqual(len(payload["coordinate_conditions"]), 9)
        self.assertEqual(len(payload["multiplicity_families"]), 4)
        self.assertEqual(payload["hypotheses_per_family"], 54)
        self.assertIn("maximum_absolute", payload["multiplicity"])

    def test_hash_seed_does_not_change_protocol_or_pairs(self) -> None:
        script = (
            "import sys;"
            f"sys.path.insert(0,{str(CODE_ROOT)!r});"
            "import iteration2_config as c;"
            "s=c.full_schedule()[0];q=c.seeds_for_graph(s);"
            "print(c.DATA_GENERATION_HASH,q,c.sample_ordered_pairs(range(s.n),5,"
            "graph_id=s.graph_id,pair_seed=q.pairs))"
        )
        outputs = []
        for seed in ("1", "987654321"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
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


if __name__ == "__main__":
    unittest.main()
