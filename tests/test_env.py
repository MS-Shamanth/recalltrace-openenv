"""Unit tests for RecallTrace."""

from __future__ import annotations

import unittest

from env.env import RecallTraceEnv
from grader.grader import evaluate_action_plan


class RecallTraceEnvTests(unittest.TestCase):
    def test_phase1_plan_scores_high(self) -> None:
        grade = evaluate_action_plan(
            "phase1_direct_recall",
            [
                {"type": "trace_lot", "lot_id": "LotA"},
                {"type": "inspect_node", "node_id": "warehouse"},
                {"type": "inspect_node", "node_id": "store1"},
                {"type": "inspect_node", "node_id": "store2"},
                {"type": "quarantine", "node_id": "warehouse", "lot_id": "LotA", "quantity": 100},
                {"type": "quarantine", "node_id": "store1", "lot_id": "LotA", "quantity": 50},
                {"type": "quarantine", "node_id": "store2", "lot_id": "LotA", "quantity": 20},
                {"type": "notify", "node_id": "all"},
                {"type": "finalize"},
            ],
        )
        self.assertGreaterEqual(grade.score, 0.95)
        self.assertTrue(grade.success)

    def test_phase2_trace_reveals_relabels(self) -> None:
        env = RecallTraceEnv(task_id="phase2_relabel_recall")
        env.reset()
        observation, reward, done, info = env.step({"type": "trace_lot", "lot_id": "LotA"})
        self.assertFalse(done)
        self.assertGreater(reward, 0)
        self.assertEqual(info["matched_lots"], ["LotA", "LotA_R1", "LotA_R2"])
        self.assertIn("store3", observation.trace_results["LotA"]["affected_nodes"])

    def test_phase3_mixed_inventory_requires_exact_quarantine(self) -> None:
        env = RecallTraceEnv(task_id="phase3_mixed_shipments")
        env.reset()
        env.step({"type": "trace_lot", "lot_id": "LotA"})
        env.step({"type": "inspect_node", "node_id": "crossdock"})
        _, reward, _, info = env.step({"type": "quarantine", "node_id": "crossdock", "lot_id": "LotBlend", "quantity": 15})
        self.assertLess(reward, 0)
        self.assertEqual(info["target_contaminated_quantity"], 12)

    def test_phase3_full_plan_scores_high(self) -> None:
        grade = evaluate_action_plan(
            "phase3_mixed_shipments",
            [
                {"type": "trace_lot", "lot_id": "LotA"},
                {"type": "inspect_node", "node_id": "warehouse"},
                {"type": "inspect_node", "node_id": "crossdock"},
                {"type": "inspect_node", "node_id": "store1"},
                {"type": "inspect_node", "node_id": "store2"},
                {"type": "inspect_node", "node_id": "store3"},
                {"type": "quarantine", "node_id": "warehouse", "lot_id": "LotA", "quantity": 30},
                {"type": "quarantine", "node_id": "crossdock", "lot_id": "LotBlend", "quantity": 12},
                {"type": "quarantine", "node_id": "store1", "lot_id": "LotA", "quantity": 10},
                {"type": "quarantine", "node_id": "store2", "lot_id": "LotBlend", "quantity": 8},
                {"type": "quarantine", "node_id": "store3", "lot_id": "LotBlend", "quantity": 4},
                {"type": "notify", "node_id": "all"},
                {"type": "finalize"},
            ],
        )
        self.assertGreaterEqual(grade.score, 0.95)
        self.assertTrue(grade.final_info["all_affected_stock_quarantined"])


if __name__ == "__main__":
    unittest.main()
