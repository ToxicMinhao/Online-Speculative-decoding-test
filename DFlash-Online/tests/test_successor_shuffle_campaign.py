"""CPU manipulation evidence and strict campaign metadata checks."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from experiments.successor_shuffle import campaign as C, qualification as Q
from experiments.successor_shuffle.contract import budgets, training_arms
from online_feedback.kplus import redistribute_response
from online_feedback.successor_shuffle import SuccessorShuffle, assigned_group_terms
from test_successor_shuffle import mixed_blocks


def manipulation(identity=False, response_id="train_00512"):
    p, y, w, a = mixed_blocks()
    model = SimpleNamespace(
        lm_head=torch.nn.Identity(),
        loss_dtype=torch.bfloat16,
        eos_token_ids=(30,),
        costs={},
    )
    logits = torch.full((*y.shape, 31), -2.0, dtype=torch.bfloat16)
    logits.scatter_(-1, p.unsqueeze(-1), 3.0)
    extension = SuccessorShuffle(identity=identity)
    extension.set_response_id(response_id)
    packed = extension.prepare_response(model, logits, y, w, a)
    loss = torch.linspace(0.1, 8, y.numel()).reshape_as(w).to(torch.bfloat16)
    groups = assigned_group_terms(loss, p, y, w, packed[..., 0], packed, (30,))
    nk = (loss * w * packed[..., 0]).sum()
    selected = (w * packed[..., 0]).sum()
    terms = (
        nk,
        selected,
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        *groups,
    )
    _, metrics = redistribute_response(nk, selected, groups[:6])
    metrics.update(extension.finish_response(terms, metrics))
    row = deepcopy(extension.last_receipt)
    snap = extension.qualification_snapshot()
    snap["eos_token_ids"] = [30]
    observation = dict(
        record_id=response_id,
        loss=row["assigned_loss"],
        successor_shuffle=row,
        sum_metrics={k: float(v) for k, v in metrics.items()},
    )
    return row, snap, observation


class SuccessorCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_bounded_reuse_fallback_startup_and_qualification_counts(self):
        self.assertEqual(training_arms(), ("K", "SHUFFLED"))
        for reuse, steps in ((True, 256), (False, 384)):
            value = budgets(256, reuse_true=reuse, missing_references=17)
            self.assertEqual(value["main"]["optimizer_steps"], steps)
            self.assertEqual(value["main"]["endpoint_generations"], 1536)
            self.assertEqual(value["qualification"]["response_backwards"], 12)
            self.assertEqual(value["qualification"]["next_anchor_draws"], 3)
        for n in (0, 7, 257, 264):
            with self.assertRaises(ValueError):
                budgets(n)

    def test_real_and_identity_masks_include_cross_chunk_eos_gap_and_singleton(self):
        for neutral in (False, True):
            row, snap, obs = manipulation(neutral)
            self.assertEqual(
                C.validate_manipulation(row, identity_assignment=neutral)["status"],
                "passed",
            )
            self.assertEqual(Q.snapshot_check(snap, row)["status"], "passed")
            if not neutral:
                self.assertGreater(row["flipped_bits"], 0)
                self.assertEqual(C.validate_observation(obs)["status"], "passed")
            else:
                self.assertEqual(row["flipped_bits"], 0)

    def test_receipt_flags_hash_permutation_and_work_cannot_be_forged(self):
        original, _, _ = manipulation()
        for mutate in (
            lambda r: r.update(status="prepared"),
            lambda r: r.update(schema="foreign"),
            lambda r: r.update(projection_completed=0),
            lambda r: r.update(mass_preserved=False),
            lambda r: r["strata"][0].update(
                donors=list(reversed(r["strata"][0]["donors"]))
            ),
            lambda r: r.update(uniform_expected_flips=0),
            lambda r: r.update(flipped_bits=0),
        ):
            row = deepcopy(original)
            mutate(row)
            with self.assertRaises(ValueError):
                C.validate_manipulation(row)

    def test_same_forward_metrics_and_loss_are_bound_to_receipt(self):
        _, _, original = manipulation()
        for mutate in (
            lambda r: r["sum_metrics"].update(shuffle_flipped_bits=0),
            lambda r: r["sum_metrics"].update(kplus_W1=0),
            lambda r: r.update(loss=0),
            lambda r: r["sum_metrics"].update(shuffle_response_count=2),
        ):
            row = deepcopy(original)
            mutate(row)
            with self.assertRaises(ValueError):
                C.validate_observation(row)

    def test_saved_mask_reconstruction_detects_scope_and_label_changes(self):
        row, original, _ = manipulation()
        for mutate in (
            lambda s: s["selection"][0][0].__setitem__(0, True),
            lambda s: s["eligible"][0][21].__setitem__(15, True),
            lambda s: s["assigned_high"][0][0].__setitem__(
                1, not s["assigned_high"][0][0][1]
            ),
        ):
            snap = deepcopy(original)
            mutate(snap)
            with self.assertRaises(ValueError):
                Q.snapshot_check(snap, row)

    def test_single_class_keeps_hash_map_while_leaving_bits_identical(self):
        row, snap, _ = manipulation()
        # The fixture includes an all-high group; every n>1 group uses the hash map.
        for s in row["strata"]:
            self.assertEqual(sorted(s["donors"]), s["recipients"])
        self.assertEqual(Q.snapshot_check(snap, row)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
