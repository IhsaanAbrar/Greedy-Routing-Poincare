"""Focused synthetic tests for independent Step 17 raw-result validation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from embedding import calculate_embedding_distortion  # noqa: E402
from experiment_checkpoint import _json_bytes  # noqa: E402
from network_metrics import prepare_all_pairs_shortest_paths  # noqa: E402
from validate_full_experiment import (  # noqa: E402
    COMBINED_FREEZE_HASH,
    FullResultValidationError,
    _vectorized_distortion,
    committed_source_fingerprint,
    compute_raw_tree_fingerprint,
    derive_route_audit_pair_indices,
    validate_raw_inventory,
    validate_route_record,
)


class RouteAuditSamplingTests(unittest.TestCase):
    def test_route_audit_sampling_is_deterministic_distinct_and_outcome_free(self):
        first = derive_route_audit_pair_indices("er_n0100_m04_rep000")
        second = derive_route_audit_pair_indices("er_n0100_m04_rep000")
        other = derive_route_audit_pair_indices("er_n0100_m04_rep001")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(set(first)), 10)
        self.assertTrue(all(0 <= value < 1_000 for value in first))
        self.assertNotEqual(first, other)

    def test_committed_source_fingerprint_matches_step16_manifest(self):
        actual = committed_source_fingerprint(
            PROJECT_ROOT,
            "a121c33a20ea721c2a5fca96bdfd6e2eeb7dd0bc",
        )
        self.assertEqual(
            actual,
            "72708e43249dbe9c331485585911e903e5fc5562abcd6adeec2dd2017c9d0e3d",
        )
        self.assertEqual(
            COMBINED_FREEZE_HASH,
            "8e002ef20f96a4f66c80440c9734cd28b6c0851a95a7977d5e2b7cf905f7a78a",
        )


class RawInventoryAndImmutabilityTests(unittest.TestCase):
    def _inventory(self, root: Path) -> None:
        (root / "graphs" / "g0").mkdir(parents=True)
        (root / "publication_timings").mkdir()
        (root / "publication_timings" / "g0.json").write_text(
            "{}", encoding="utf-8"
        )
        (root / "progress.json").write_text("{}", encoding="utf-8")
        (root / "run_manifest.json").write_text("{}", encoding="utf-8")

    def test_raw_fingerprint_is_read_only_and_detects_byte_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "raw.bin"
            payload.write_bytes(b"first")
            first = compute_raw_tree_fingerprint(root)
            second = compute_raw_tree_fingerprint(root)
            self.assertEqual(first, second)
            self.assertEqual([path.name for path in root.iterdir()], ["raw.bin"])
            payload.write_bytes(b"other")
            changed = compute_raw_tree_fingerprint(root)
            self.assertNotEqual(first.sha256, changed.sha256)

    def test_inventory_accepts_exact_schedule(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._inventory(root)
            counts = validate_raw_inventory(root, ("g0",))
            self.assertEqual(counts["graph_checkpoints"], 1)
            self.assertEqual(counts["temporary_or_orphan_entries"], 0)

    def test_inventory_rejects_missing_and_orphan_entries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._inventory(root)
            with self.assertRaisesRegex(
                FullResultValidationError, "schedule mismatch"
            ):
                validate_raw_inventory(root, ("g0", "g1"))
            (root / "unexpected.txt").write_text("orphan", encoding="utf-8")
            with self.assertRaisesRegex(
                FullResultValidationError, "top-level inventory"
            ):
                validate_raw_inventory(root, ("g0",))

    def test_inventory_rejects_error_and_temporary_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._inventory(root)
            (root / "graphs" / "g0" / "ERROR.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(FullResultValidationError, "ERROR.json"):
                validate_raw_inventory(root, ("g0",))
            (root / "graphs" / "g0" / "ERROR.json").unlink()
            (root / "graphs" / ".g1.tmp-x").mkdir()
            with self.assertRaisesRegex(
                FullResultValidationError, "temporary/orphan"
            ):
                validate_raw_inventory(root, ("g0",))


class StructuralValidationTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(3)
        self.base = {
            "graph_id": "g",
            "pair_index": 0,
            "pair_id": "g:pair:0000",
            "source": 0,
            "destination": 2,
            "method_id": "euclidean_greedy",
            "success": True,
            "walk": [0, 1, 2],
            "route_length": 2,
            "physical_hop_count": 2,
            "dijkstra_length": 2,
            "dijkstra_hop_count": 2,
            "stretch": 1.0,
            "runtime_ns": 10,
            "initial_failure_type": None,
            "final_failure_type": None,
            "repair_attempted": False,
            "repair_succeeded": False,
            "repair_alternative_existed": None,
            "repair_attempt_count": 0,
            "forwarding_decisions": 2,
        }

    def _validate(self, row):
        return validate_route_record(
            row,
            graph=self.graph,
            graph_id="g",
            pair_index=0,
            source=0,
            destination=2,
            dijkstra_length=2,
        )

    def test_valid_route_adjacency_hops_and_stretch(self):
        result = self._validate(dict(self.base))
        self.assertTrue(result.success)
        self.assertEqual(result.route_length, 2)

    def test_nonedge_and_hop_mismatch_are_rejected(self):
        row = dict(self.base, walk=[0, 2], route_length=1, physical_hop_count=1)
        with self.assertRaisesRegex(FullResultValidationError, "non-edge"):
            self._validate(row)
        row = dict(self.base, physical_hop_count=1)
        with self.assertRaisesRegex(FullResultValidationError, "hop count"):
            self._validate(row)

    def test_inconsistent_failure_and_repair_are_rejected(self):
        row = dict(
            self.base,
            success=False,
            walk=[0, 1],
            route_length=1,
            physical_hop_count=1,
            stretch=None,
            final_failure_type="local_minimum",
        )
        with self.assertRaisesRegex(
            FullResultValidationError, "failure/repair state"
        ):
            self._validate(row)

    def test_repaired_walk_requires_exactly_one_counted_backtrack(self):
        self.graph.add_edge(0, 2)
        row = dict(
            self.base,
            method_id="repaired_poincare_greedy",
            walk=[0, 1, 0, 2],
            route_length=3,
            physical_hop_count=3,
            stretch=1.5,
            initial_failure_type="local_minimum",
            repair_attempted=True,
            repair_succeeded=True,
            repair_alternative_existed=True,
            repair_attempt_count=1,
            forwarding_decisions=2,
        )
        result = self._validate(row)
        self.assertTrue(result.repair_succeeded)
        broken = dict(row, walk=[0, 1, 2], route_length=2, physical_hop_count=2)
        with self.assertRaisesRegex(FullResultValidationError, "backtracking"):
            self._validate(broken)


class DistortionRecomputationTests(unittest.TestCase):
    def test_vectorized_metrics_match_frozen_scalar_formulas(self):
        graph = nx.path_graph(4)
        shortest = prepare_all_pairs_shortest_paths(graph)
        coordinates = {
            0: (-0.4, 0.0),
            1: (-0.1, 0.1),
            2: (0.2, -0.1),
            3: (0.45, 0.0),
        }
        for metric in ("euclidean", "poincare"):
            scalar = calculate_embedding_distortion(
                graph,
                coordinates,
                shortest_paths=shortest,
                metric=metric,
            )
            vectorized = _vectorized_distortion(
                shortest, coordinates, metric=metric
            )
            self.assertEqual(
                vectorized.unordered_pair_count, scalar.unordered_pair_count
            )
            self.assertAlmostEqual(
                vectorized.fitted_scale_alpha,
                scalar.fitted_scale_alpha,
                places=14,
            )
            self.assertAlmostEqual(
                vectorized.mean_relative_distortion,
                scalar.mean_relative_distortion,
                places=14,
            )
            self.assertAlmostEqual(
                vectorized.rmse_relative_distortion,
                scalar.rmse_relative_distortion,
                places=14,
            )

    def test_tagged_float_payload_decodes_in_stream_fixture(self):
        payload = _json_bytes({"stretch": 1.25})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl.gz"
            with path.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=0
                ) as stream:
                    stream.write(payload + b"\n")
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                stored = json.loads(stream.readline())
        self.assertEqual(stored["stretch"]["__float64__"], float(1.25).hex())


if __name__ == "__main__":
    unittest.main()
