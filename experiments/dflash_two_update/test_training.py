"""CPU-only checks; metadata tests also run with ordinary Python without torch."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    from . import training
except ImportError:
    import training


HAS_TORCH = importlib.util.find_spec("torch") is not None
RELEASED = "a" * 64
UPDATED = "b" * 64
TARGET = "c" * 64


def response(turn, checkpoint=RELEASED):
    boundary = 20 + turn * 7
    rid = f"episode:turn{turn}"
    result = dict(episode_id="episode", turn=turn, record_id=rid, target_id="T0",
                  target_identity_sha256=TARGET, capture_checkpoint_sha256=checkpoint,
                  input_token_count=boundary, input_ids=list(range(boundary)), output_cap=512,
                  output_ids=[1, 2, 3, 4], eligible_block_count=2, eligible_position_count=10, blocks=[])
    for index, (offset, emitted, finished) in enumerate(((1, [2, 3], False), (3, [4], True))):
        anchor = boundary + offset - 1
        mask = [False] + ([True] * 9 if not finished else [True] + [False] * 8)
        slots = [i for i, value in enumerate(mask) if value]
        result["blocks"].append(dict(
            block_index=index, response_record_id=rid, response_turn=turn, input_token_count=boundary,
            capture_checkpoint_sha256=checkpoint, target_id="T0", target_identity_sha256=TARGET,
            anchor=anchor, context_length=anchor, positions=list(range(anchor, anchor + 10)),
            teacher_row_for_slot=[None] + list(range(9)), eligible_mask=mask, eligible_positions=len(slots),
            slot_owner_record_ids=[rid if i in slots else None for i in range(10)], owned_eligible_slots=slots,
            outcome=dict(output_offset=offset, committed_token_ids=emitted, finished=finished)))
    return result


class MaskAndScheduleTests(unittest.TestCase):
    def test_response17_has_loss_only_in_its_own_example(self):
        # In example18 these same response17 tokens are a prefix, even though
        # both examples belong to the U2 support group.
        owners = [None] * 3 + ["response17"] * 3 + [None] * 2 + ["response18"] * 3 + [None]
        positions = list(range(len(owners)))
        all_candidates = [True] * len(owners)
        own17 = training.response_loss_mask(record_id="response17", input_token_count=3, positions=positions,
                                            slot_owner_record_ids=owners, eligible_mask=all_candidates)
        own18 = training.response_loss_mask(record_id="response18", input_token_count=8, positions=positions,
                                            slot_owner_record_ids=owners, eligible_mask=all_candidates)
        self.assertEqual([i for i, value in enumerate(own17) if value], [3, 4, 5])
        self.assertEqual([i for i, value in enumerate(own18) if value], [8, 9, 10])
        self.assertFalse(any(own18[3:6]))

    def test_history_user_seed_padding_terminal_slots_excluded(self):
        mask = training.response_loss_mask(record_id="r", input_token_count=5,
            positions=list(range(9)), slot_owner_record_ids=["r"] * 9,
            eligible_mask=[True] * 6 + [False, True, True], padding_mask=[False] * 8 + [True])
        self.assertEqual(mask, [False] * 5 + [True, False, True, False])
        r = response(17, UPDATED)
        audit = training.audit_response(r)
        self.assertEqual(audit["eligible_positions"], 10)
        self.assertEqual(audit["excluded_seed_positions"], 2)
        self.assertEqual(audit["native_ineligible_positions"], 8)
        self.assertEqual(audit["dropped_native_eligible_positions"], 0)
        self.assertNotIn(0, training.owned_slots(r, r["blocks"][0]))

    def test_proposal_history_beyond_rejection_and_committed_end_kept(self):
        r = response(17, UPDATED)
        b = r["blocks"][0]
        slots = training.owned_slots(r, b)
        self.assertEqual(slots, list(range(1, 10)))
        # output length is4; the first nonterminal block's proposals extend
        # beyond that eventual answer, yet remain real verified positions.
        self.assertGreater(b["positions"][slots[-1]], len(r["input_ids"]) + len(r["output_ids"]))
        self.assertEqual(training.owned_slots(r, r["blocks"][1]), [1])

    def test_foreign_response_block_and_silent_slot_loss_rejected(self):
        r = response(18, UPDATED)
        r["blocks"][0] = response(17, UPDATED)["blocks"][0]
        with self.assertRaisesRegex(ValueError, "provenance"):
            training.audit_response(r)
        for change in ("owner", "padding", "declared", "native", "anchor", "hash"):
            r = response(18, UPDATED)
            b = r["blocks"][0]
            if change == "owner":
                b["slot_owner_record_ids"][1] = "episode:turn17"
                b["owned_eligible_slots"].remove(1)
            elif change == "padding":
                b["padding_mask"] = [False, True] + [False] * 8
                b["owned_eligible_slots"].remove(1)
            elif change == "declared":
                b["owned_eligible_slots"].remove(1)
            elif change == "native":
                b["eligible_mask"][1] = False
            elif change == "anchor":
                b["anchor"] -= 1
                b["context_length"] -= 1
            else:
                b["capture_checkpoint_sha256"] = RELEASED
            with self.subTest(change=change), self.assertRaises(ValueError):
                training.audit_response(r)

    def test_event2_only_new_eight_responses_and_two_passes(self):
        group = [response(turn, UPDATED) for turn in range(17, 25)]
        schedule = training.event_schedule(2, group)
        order = [r["turn"] for _, pair in schedule for r in pair]
        self.assertEqual(order, list(range(17, 25)) * 2)
        self.assertEqual(len(schedule), 8)
        for invalid in (group[::-1], group[:-1], [response(t) for t in range(9, 17)], group[:1] * 8):
            with self.assertRaises(ValueError):
                training.event_schedule(2, invalid)

    def test_exposure_and_complete_context_bounds_are_hard_failures(self):
        r = response(17, UPDATED)
        r["eligible_position_count"] -= 1
        with self.assertRaisesRegex(ValueError, "exposure"):
            training.audit_response(r)
        r = response(17, UPDATED)
        r["output_cap"] = training.CONTEXT_LIMIT
        with self.assertRaisesRegex(ValueError, "truncation forbidden"):
            training.audit_response(r)


@unittest.skipUnless(HAS_TORCH, "torch unavailable; run with a CPU-capable torch environment for numerical checks")
class NumericalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def fixture(self):
        import torch
        from torch import nn
        torch.manual_seed(17)
        draft = nn.Linear(4, 4, bias=True)

        class Backend:
            def __init__(self):
                self.draft = draft
                self.head = nn.Linear(4, 5, bias=False).requires_grad_(False)
                self.original = self.head.weight.detach().clone()
                self.forwarded = []
                self.finished = 0
                self.features = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 19
                self.teacher = torch.cos(torch.arange(50, dtype=torch.float32).reshape(10, 5))
            def activate(self, target_id):
                assert target_id == "T0"
            def check_frozen(self):
                assert not self.head.weight.requires_grad and self.head.weight.grad is None
                assert torch.equal(self.head.weight, self.original)
            def iter_blocks(self, r):
                return iter(copy.deepcopy(r["blocks"]))
            def eligible_slots(self, b):
                return [i for i, v in enumerate(b["eligible_mask"]) if v]
            def forward_block(self, b, slots):
                self.forwarded.append((b["response_record_id"], tuple(b["positions"][s] for s in slots)))
                student = self.head(self.draft(self.features[slots]))
                return student, self.teacher[slots].detach(), torch.zeros(len(slots), dtype=torch.long)
            def finish_response(self):
                self.finished += 1

        backend = Backend()
        trainer = training.PersistentFullKLTrainer(draft, backend, episode_id="episode", target_id="T0",
            target_identity_sha256=TARGET, released_checkpoint_sha256=RELEASED, position_chunk=2)
        return draft, backend, trainer

    def test_forward_kl_direction_teacher_detachment_and_hierarchical_gradient(self):
        import torch
        torch.manual_seed(5)
        x = torch.randn(10, 5, requires_grad=True)
        teacher = torch.randn(10, 5, requires_grad=True)
        p = teacher.detach().softmax(-1)
        independent = (p * (p.log() - x.log_softmax(-1))).sum(-1)
        torch.testing.assert_close(training.forward_kl(x, teacher), independent)
        # Pair: responseA blocks have2and5slots; responseB one block3slots.
        dense = ((independent[:2].mean() + independent[2:7].mean()) / 2 + independent[7:].mean()) / 2
        dense.backward()
        expected_gradient = x.grad.clone()
        x.grad = None
        total = 0.0
        for start, end, blocks in ((0, 2, 2), (2, 7, 2), (7, 10, 1)):
            for low in range(start, end, 2):
                high = min(low + 2, end)
                value = training.forward_kl(x[low:high], teacher[low:high]).mean()
                weighted = value * training.chunk_weight(high - low, end - start, blocks)
                total += float(weighted.detach())
                weighted.backward()
        self.assertAlmostEqual(total, float(dense.detach()), places=6)
        torch.testing.assert_close(x.grad, expected_gradient, atol=1e-7, rtol=2e-6)
        self.assertIsNone(teacher.grad)

    def test_two_events_keep_weights_optimizer_moments_and_fixed_teacher(self):
        import torch
        draft, backend, trainer = self.fixture()
        optimizer_id = id(trainer.optimizer)
        parameter_ids = tuple(id(p) for p in draft.parameters())
        first = trainer.train_event(1, [response(t) for t in range(9, 17)])
        self.assertEqual(first["optimizer_steps_total"], 8)
        self.assertEqual(first["eligible_position_presentations_this_event"], 160)
        moment_ids = {id(p): (id(s["exp_avg"]), id(s["exp_avg_sq"])) for p, s in trainer.optimizer.state.items()}
        after_first = {name: p.detach().clone() for name, p in draft.named_parameters()}
        trainer.bind_checkpoint(1, UPDATED)
        count = len(backend.forwarded)
        second = trainer.train_event(2, [response(t, UPDATED) for t in range(17, 25)])
        self.assertEqual(second["optimizer_steps_total"], 16)
        self.assertEqual(second["response_presentations_total"], 32)
        self.assertEqual(second["eligible_position_presentations_total"], 320)
        self.assertEqual(second["response_order"], [f"episode:turn{t}" for t in range(17, 25)] * 2)
        self.assertEqual(id(trainer.optimizer), optimizer_id)
        self.assertEqual(tuple(id(p) for p in draft.parameters()), parameter_ids)
        self.assertEqual(moment_ids, {id(p): (id(s["exp_avg"]), id(s["exp_avg_sq"]))
                                     for p, s in trainer.optimizer.state.items()})
        self.assertEqual({int(s["step"].item()) for s in trainer.optimizer.state.values()}, {16})
        self.assertTrue(any(not torch.equal(after_first[n], p) for n, p in draft.named_parameters()))
        self.assertEqual(backend.finished, 32)
        for rid, positions in backend.forwarded[count:]:
            turn = int(rid.split("turn")[1])
            self.assertIn(turn, range(17, 25))
            self.assertTrue(all(pos > response(turn)["input_token_count"] for pos in positions))
        backend.check_frozen()
        trainer.bind_checkpoint(2, "d" * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            receipt = trainer.save_checkpoint_state(path)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["optimizer_steps"], 16)
            self.assertEqual(saved["checkpoint_sha256"], "d" * 64)
            self.assertIn("draft", saved)
            self.assertGreater(receipt["bytes"], 0)
            with self.assertRaisesRegex(ValueError, "fresh path"):
                trainer.save_checkpoint_state(path)

    def test_late_bad_record_rejected_before_any_optimizer_update(self):
        _, _, trainer = self.fixture()
        group = [response(t) for t in range(9, 17)]
        group[-1]["blocks"][0]["slot_owner_record_ids"][1] = "episode:turn9"
        with self.assertRaisesRegex(ValueError, "Owned eligible slots"):
            trainer.train_event(1, group)
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertFalse(trainer.optimizer.state)

    def test_checkpoint_teacher_episode_and_old_support_rejected(self):
        for change in ("checkpoint", "teacher", "episode"):
            _, _, trainer = self.fixture()
            group = [response(t) for t in range(9, 17)]
            key, value = {"checkpoint": ("capture_checkpoint_sha256", UPDATED),
                          "teacher": ("target_identity_sha256", "d" * 64),
                          "episode": ("episode_id", "another-episode")}[change]
            group[0][key] = value
            with self.subTest(change=change), self.assertRaises(ValueError):
                trainer.train_event(1, group)
            self.assertEqual(trainer.optimizer_steps, 0)
        _, _, trainer = self.fixture()
        trainer.train_event(1, [response(t) for t in range(9, 17)])
        with self.assertRaisesRegex(ValueError, "Bind"):
            trainer.train_event(2, [response(t, UPDATED) for t in range(17, 25)])
        trainer.bind_checkpoint(1, UPDATED)
        with self.assertRaisesRegex(ValueError, "eight chronological new"):
            trainer.train_event(2, [response(t, UPDATED) for t in range(9, 17)])

    def test_train_failure_is_not_automatically_retryable(self):
        _, backend, trainer = self.fixture()
        original = backend.forward_block
        def bad(block, slots):
            student, teacher, hard = original(block, slots)
            return student, teacher.requires_grad_(True), hard
        backend.forward_block = bad
        with self.assertRaisesRegex(ValueError, "teacher detachment"):
            trainer.train_event(1, [response(t) for t in range(9, 17)])
        self.assertTrue(trainer.failed)
        with self.assertRaisesRegex(ValueError, "Prior failure"):
            trainer.train_event(1, [response(t) for t in range(9, 17)])


if __name__ == "__main__":
    unittest.main()
