"""CPU integration checks of the driver's pure label and table functions.

AST loading deliberately omits the remote model/runtime imports. These tests
exercise the exact function bodies, not copied implementations or fake modules.
"""

import ast
import math
from pathlib import Path
import unittest

import torch

from fit_metrics import block_prefix_metrics, summarize_positions


_source = ast.parse(Path(__file__).with_name("fit_check.py").read_text())
_names = {"matched_labels", "layout_tables"}
_functions = [node for node in _source.body if isinstance(node, ast.FunctionDef) and node.name in _names]
assert {node.name for node in _functions} == _names
_namespace = {"torch": torch, "block_prefix_metrics": block_prefix_metrics}
exec(compile(ast.Module(body=_functions, type_ignores=[]), "fit_check.py", "exec"), _namespace)
matched_labels = _namespace["matched_labels"]
layout_tables = _namespace["layout_tables"]


class MatchedLayoutTests(unittest.TestCase):
    def test_same_position_labels_and_prompt_anchor_keep_tail_masks(self):
        ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        mask = torch.tensor([[0., 0., 1., 1., 1., 1.]])
        anchors = torch.tensor([[0, 3, 4, 1]])
        labels, weights = matched_labels(ids, mask, anchors, torch.tensor([[1, 1, 1, 0]]), 4, 7.)
        self.assertEqual(labels.tolist(), [[[10, 11, 12, 13], [13, 14, 15, 15], [14, 15, 15, 15], [11, 12, 13, 14]]])
        expected = torch.tensor([[[0., 0., math.exp(-1 / 7), math.exp(-2 / 7)],
                                  [0., 1., math.exp(-1 / 7), 0.], [0., 1., 0., 0.], [0., 0., 0., 0.]]])
        torch.testing.assert_close(weights, expected)
        self.assertEqual((weights > 0).sum().item(), 5)

    def fixture(self, *, finish_reason="length", stop_ids=(), all_match=False):
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        anchors = torch.tensor([[0 if all_match else 2, 4]])
        labels, weights = matched_labels(ids, torch.ones_like(ids), anchors, torch.ones_like(anchors), 4, 7.)
        f_ids = torch.tensor([[[2, 3, 0, 5], [4, 0, 5, 5]]])
        p_ids = torch.tensor([[[2, 0, 4, 5], [4, 5, 5, 5]]])
        if all_match:
            p_ids = labels
        reference = torch.zeros(1, 2, 4, 6).scatter_(-1, f_ids.unsqueeze(-1), 2.)
        current = torch.zeros_like(reference).scatter_(-1, p_ids.unsqueeze(-1), 2.)
        targets = torch.full_like(reference, 0.1).scatter_(-1, labels.unsqueeze(-1), 0.5)
        metrics = {"F": summarize_positions(reference, reference, targets, labels, weights),
                   "P": summarize_positions(reference, current, targets, labels, weights)}
        identity = dict(group="fixture", record_id="source_A", pass_index=0)
        record = dict(finish_reason=finish_reason, stop_token_ids=list(stop_ids))
        return layout_tables(identity, metrics, labels, weights, anchors, {"input_ids": ids}, record), metrics, weights

    def test_tables_filter_padding_and_keep_original_partition_denominator(self):
        (positions, prefixes, summaries), metrics, weights = self.fixture()
        self.assertEqual(len(positions), 8)  # Four valid repeated exposures per arm.
        self.assertEqual({row["absolute_position"] for row in positions}, {3, 4, 5})
        self.assertTrue(all(row["slot"] > 0 and row["weight"] > 0 for row in positions))
        p = {row["partition"]: row for row in summaries if row["arm"] == "P"}
        self.assertEqual(p["all"]["exposure"], 4)
        self.assertEqual(p["all"]["correct"], 3)
        self.assertEqual(p["all"]["repairs"], 2)
        self.assertEqual(p["all"]["damage"], 1)
        self.assertEqual(p["reference_correct"]["exposure"], 2)
        self.assertEqual(p["reference_wrong"]["exposure"], 2)
        denominator = 2 + math.exp(-1 / 7) + math.exp(-2 / 7)
        for row in p.values():
            self.assertAlmostEqual(row["original_denominator"], denominator, places=6)
        for field in ("soft_ce", "target_entropy", "kl_to_r"):
            expected = (metrics["P"][field] * weights).sum().item() / weights.sum().item()
            self.assertAlmostEqual(p["all"][field + "_mean"], expected, places=6)
            self.assertAlmostEqual(p["reference_correct"][field + "_contribution"] + p["reference_wrong"][field + "_contribution"], expected, places=6)
        # The fixed F partition, not current P correctness, assigns repaired rows.
        self.assertEqual(p["reference_wrong"]["repairs"], 2)
        self.assertEqual(p["reference_correct"]["damage"], 1)

    def test_cap_suffix_is_censored_but_internal_block_is_not(self):
        (positions, prefixes, summaries), _, _ = self.fixture(all_match=True)
        p = {row["block"]: row for row in prefixes if row["arm"] == "P"}
        self.assertEqual(p[0]["matching_prefix_length"], 3)
        self.assertIsNone(p[0]["first_mismatch_slot"])
        self.assertFalse(p[0]["cap_boundary_censored"])
        self.assertEqual(p[1]["matching_prefix_length"], 1)
        self.assertTrue(p[1]["cap_boundary_censored"])

    def test_matched_saved_eos_is_included_and_not_cap_censored(self):
        (positions, prefixes, summaries), _, _ = self.fixture(finish_reason="eos", stop_ids=[5])
        p = {row["block"]: row for row in prefixes if row["arm"] == "P"}
        self.assertEqual(p[1]["matching_prefix_length"], 1)
        self.assertEqual(p[1]["matching_non_eos_prefix_length"], 0)
        self.assertTrue(p[1]["matched_eos"])
        self.assertFalse(p[1]["cap_boundary_censored"])


if __name__ == "__main__":
    unittest.main()
