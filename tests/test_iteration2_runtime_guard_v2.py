from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import subprocess
from threading import Thread
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import run_iteration2  # noqa: F401,E402 - imports every registered boundary
import iteration2_runtime_guard as guard  # noqa: E402
from analyze_iteration2 import create_analysis_tables  # noqa: E402
from iteration2_config import sample_ordered_pairs  # noqa: E402


def _registered_callable(qualified_name: str) -> object:
    module_name, attribute = qualified_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute)


class Iteration2ScientificOperationGuardTests(unittest.TestCase):
    def test_registry_is_complete_and_missing_registration_fails(self):
        guard.validate_scientific_boundary_registry()
        registered = dict(guard.registered_scientific_boundaries())
        self.assertEqual(set(registered), set(guard.SCIENTIFIC_OPERATION_CATALOG))
        missing = next(iter(registered))
        incomplete = dict(registered)
        del incomplete[missing]
        with patch.object(guard, "_REGISTERED_BOUNDARIES", incomplete):
            with self.assertRaisesRegex(RuntimeError, "registry mismatch"):
                guard.validate_scientific_boundary_registry()

    def test_every_registered_boundary_is_blocked_before_body_execution(self):
        registered = dict(guard.registered_scientific_boundaries())
        with guard.scientific_operation_context(guard.ANALYSIS_READ_ONLY) as ledger:
            for operation, qualified_name in registered.items():
                function = _registered_callable(qualified_name)
                with self.subTest(operation=operation):
                    with self.assertRaisesRegex(
                        guard.ScientificOperationBlocked,
                        operation,
                    ):
                        function()
            snapshot = ledger.snapshot()
        self.assertEqual(snapshot["total_attempted"], len(registered))
        self.assertEqual(snapshot["total_blocked"], len(registered))
        self.assertEqual(snapshot["total_executed"], 0)
        self.assertTrue(
            all(value == 1 for value in snapshot["attempted_operation_counts"].values())
        )
        self.assertTrue(
            all(value == 0 for value in snapshot["executed_operation_counts"].values())
        )

    def test_context_is_reset_and_read_only_guard_covers_child_thread(self):
        observed: list[object] = []
        blocked: list[bool] = []

        def worker() -> None:
            active = guard.current_scientific_ledger()
            observed.append(None if active is None else active.mode)
            try:
                sample_ordered_pairs()
            except guard.ScientificOperationBlocked:
                blocked.append(True)

        with guard.scientific_operation_context(guard.ANALYSIS_READ_ONLY):
            self.assertIsNotNone(guard.current_scientific_ledger())
            thread = Thread(target=worker)
            thread.start()
            thread.join()
        self.assertEqual(observed, [guard.ANALYSIS_READ_ONLY])
        self.assertEqual(blocked, [True])
        self.assertIsNone(guard.current_scientific_ledger())
        with guard.scientific_operation_context(guard.ANALYSIS_READ_ONLY) as fresh:
            self.assertEqual(fresh.snapshot()["total_attempted"], 0)

    def test_caught_violation_still_prevents_analysis_publication(self):
        with guard.scientific_operation_context(guard.ANALYSIS_READ_ONLY) as ledger:
            with self.assertRaises(guard.ScientificOperationBlocked):
                sample_ordered_pairs()
            snapshot = ledger.snapshot()
        fingerprint = {
            "schema": "raw_tree_fingerprint_v1",
            "file_count": 1,
            "byte_count": 1,
            "sha256": "0" * 64,
        }
        evidence = {
            "validation_mode": "read_only_analysis_consumer",
            "regeneration_requested": False,
            "scientific_graphs_executed_during_analysis": 0,
            "dijkstra_executions_during_analysis": 0,
            "routing_executions_during_analysis": 0,
            "raw_checkpoints_written_during_analysis": 0,
            "raw_tree_before": fingerprint,
            "raw_tree_after": dict(fingerprint),
            "raw_tree_unchanged": True,
            "scientific_operation_ledger": snapshot,
        }
        with self.assertRaisesRegex(RuntimeError, "prohibited scientific operation"):
            create_analysis_tables(
                [{"excluded": True}],
                analysis_validation_evidence=evidence,
                bootstrap_replicates=2,
                require_complete_design=False,
            )

    def test_processes_start_and_finish_with_isolated_ledgers(self):
        script = """
import json,sys
sys.path.insert(0, 'code')
from iteration2_config import sample_ordered_pairs
from iteration2_runtime_guard import (
    ANALYSIS_READ_ONLY, ScientificOperationBlocked,
    current_scientific_ledger, scientific_operation_context,
)
initial = current_scientific_ledger() is None
with scientific_operation_context(ANALYSIS_READ_ONLY) as ledger:
    try:
        sample_ordered_pairs()
    except ScientificOperationBlocked:
        pass
    snapshot = ledger.snapshot()
final = current_scientific_ledger() is None
print(json.dumps({'initial': initial, 'final': final, 'snapshot': snapshot}, sort_keys=True))
"""
        outputs = [
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
            for _ in range(2)
        ]
        self.assertEqual(outputs[0], outputs[1])
        payload = json.loads(outputs[0])
        self.assertTrue(payload["initial"])
        self.assertTrue(payload["final"])
        self.assertEqual(payload["snapshot"]["total_attempted"], 1)
        self.assertEqual(payload["snapshot"]["total_blocked"], 1)
        self.assertEqual(payload["snapshot"]["total_executed"], 0)


if __name__ == "__main__":
    unittest.main()
