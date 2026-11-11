"""Independent permutation/oracle checks and native tiny-model identity parity."""

import copy
import hashlib
import json
import unittest
from unittest import mock

import numpy as np
import torch

from online_feedback.first_error import first_error_mask
from online_feedback.kplus import install_kplus, redistribute_response
from online_feedback.objective import TargetObjectiveModel
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.successor_shuffle import (
    assigned_group_terms,
    donor_key,
    install_successor_shuffle,
    successor_assignment,
    successor_shuffle_contract,
)
from specforge.core.chunking import checkpointed_chunk_reduce
from test_objective_integration import response_batch, tiny_author


def mixed_blocks():
    labels = torch.arange(16).reshape(1, 1, 16).expand(1, 27, 16).clone()
    proposals = labels.clone()
    proposals[0, :18, 1] = 29
    proposals[0, :8, 2] = 28
    proposals[0, 18:20, 2] = 29
    proposals[0, 19, 3] = 28
    proposals[0, 20, 3] = 29  # Singleton position stratum.
    proposals[0, 21, 15] = 29  # Final slot is ineligible.
    labels[0, 22, 2] = 30  # Wrong canonical EOS is ineligible.
    proposals[0, 23, 1] = 29
    labels[0, 23, 2] = proposals[0, 23, 2] = 30  # Valid EOS successor.
    proposals[0, 24, 1] = 29
    labels[0, 25, 0] = proposals[0, 25, 0] = 30  # EOS seed.
    weights = torch.exp(-(torch.arange(16).float() - 1).clamp_min(0) / 7)
    weights = weights.reshape(1, 1, 16).expand_as(labels).clone()
    weights[..., 0] = 0
    weights[0, 24, 2] = 0  # Later positive weights cannot reopen a gap.
    anchors = (torch.arange(27) * 7 + 20).reshape(1, -1)
    return proposals, labels, weights, anchors


def scalar_oracle(proposals, labels, weights, anchors, response_id):
    """Separate scalar walk and independent hash ordering; no production helper."""
    selected = torch.zeros_like(labels, dtype=torch.bool)
    errors = torch.zeros_like(selected)
    true = torch.zeros_like(selected)
    groups = {}
    for block in range(labels.shape[1]):
        if int(labels[0, block, 0]) == 30:
            continue
        for position in range(1, labels.shape[-1]):
            if float(weights[0, block, position]) <= 0:
                break
            selected[0, block, position] = True
            eos = int(labels[0, block, position]) == 30
            if int(proposals[0, block, position]) != int(labels[0, block, position]):
                errors[0, block, position] = True
                successor = position + 1
                if (
                    not eos
                    and successor < labels.shape[-1]
                    and float(weights[0, block, successor]) > 0
                ):
                    true[0, block, position] = bool(
                        proposals[0, block, successor] == labels[0, block, successor]
                    )
                    groups.setdefault(position, []).append(block)
                break
            if eos:
                break
    assigned = true.clone()
    for position, group in groups.items():
        recipients = sorted(group, key=lambda i: int(anchors[0, i]))
        if not 0 < sum(bool(true[0, i, position]) for i in group) < len(group):
            continue

        def order(i):
            offset = int(anchors[0, i])
            raw = json.dumps(
                [20261111, response_id, position, offset],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(raw).digest(), offset

        donors = sorted(group, key=order)
        for receiver, donor in zip(recipients, donors):
            assigned[0, receiver, position] = true[0, donor, position]
    return torch.stack((selected, errors, true, assigned), dim=-1)


class SuccessorShuffleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def assert_rng_equal(self, left, right):
        self.assertEqual(left["python"], right["python"])
        self.assertEqual(left["numpy"][0], right["numpy"][0])
        np.testing.assert_array_equal(left["numpy"][1], right["numpy"][1])
        self.assertEqual(left["numpy"][2:], right["numpy"][2:])
        self.assertTrue(torch.equal(left["torch"], right["torch"]))
        self.assertEqual(len(left["cuda"]), len(right["cuda"]))
        for a, b in zip(left["cuda"], right["cuda"]):
            self.assertTrue(torch.equal(a, b))

    def test_whole_response_matches_scalar_oracle_and_preserves_inputs_rng(self):
        args = mixed_blocks()
        before = [value.clone() for value in args]
        rng = capture_rng()
        packed, receipt = successor_assignment(
            *args, "train_00512", eos_token_ids=(30,)
        )
        self.assertTrue(torch.equal(packed, scalar_oracle(*args, "train_00512")))
        self.assert_rng_equal(rng, capture_rng())
        for expected, actual in zip(before, args):
            self.assertTrue(torch.equal(expected, actual))
        self.assertGreater(receipt["flipped_bits"], 0)
        self.assertEqual(receipt["mixed_strata"], 2)
        self.assertTrue(receipt["class_counts_exact"])
        group = receipt["strata"][0]
        self.assertGreater(group["n"], 16)
        self.assertTrue(
            any(
                receiver // 56 != donor // 56
                for receiver, donor in zip(group["recipients"], group["donors"])
            )
        )
        self.assertAlmostEqual(
            receipt["uniform_expected_flips"],
            sum(2 * s["k"] * (s["n"] - s["k"]) / s["n"] for s in receipt["strata"]),
        )

    def test_eligible_eos_gap_seed_last_slot_and_singleton(self):
        packed, receipt = successor_assignment(
            *mixed_blocks(), "case", eos_token_ids=(30,)
        )
        blocks = receipt["blocks"]
        self.assertFalse(blocks[21]["eligible_successor"])
        self.assertFalse(blocks[22]["eligible_successor"])
        self.assertTrue(blocks[23]["eligible_successor"])
        self.assertEqual(blocks[23]["true_priority"], 2)
        self.assertFalse(blocks[24]["eligible_successor"])
        self.assertIsNone(blocks[25]["first_error_position"])
        self.assertIsNone(blocks[26]["first_error_position"])
        for block in (21, 22, 24):
            self.assertEqual(blocks[block]["assigned_priority"], 1)
        self.assertEqual(blocks[20]["true_priority"], blocks[20]["assigned_priority"])
        self.assertFalse(bool(packed[0, 25, :, 0].any()))

    def test_block_reordering_cannot_change_anchor_keyed_assignment(self):
        args = mixed_blocks()
        order = torch.arange(26, -1, -1)
        original, _ = successor_assignment(*args, "same", eos_token_ids=(30,))
        reordered, _ = successor_assignment(
            *(value[:, order] for value in args), "same", eos_token_ids=(30,)
        )
        self.assertTrue(torch.equal(original[:, order], reordered))

    def test_hash_serialization_pass_repetition_and_fixed_points(self):
        args = mixed_blocks()
        one, receipt = successor_assignment(*args, "réponse|1", eos_token_ids=(30,))
        two, repeated = successor_assignment(*args, "réponse|1", eos_token_ids=(30,))
        self.assertTrue(torch.equal(one, two))
        self.assertEqual(receipt, repeated)
        raw = '[20261111,"réponse|1",2,20]'.encode("utf-8")
        self.assertEqual(
            donor_key("réponse|1", 2, 20), (hashlib.sha256(raw).hexdigest(), 20)
        )
        # Exact donor ordering admits fixed points; no forced derangement.
        self.assertTrue(
            any(
                a == b
                for s in receipt["strata"]
                for a, b in zip(s["recipients"], s["donors"])
            )
        )

    def test_identity_and_single_class_are_unchanged(self):
        proposals, labels, weights, anchors = mixed_blocks()
        packed, receipt = successor_assignment(
            proposals,
            labels,
            weights,
            anchors,
            "same",
            eos_token_ids=(30,),
            identity=True,
        )
        self.assertTrue(torch.equal(packed[..., 2], packed[..., 3]))
        self.assertEqual(receipt["flipped_bits"], 0)
        labels = proposals.clone()
        labels[0, :, 1] = (proposals[0, :, 1] + 1) % 31
        packed, receipt = successor_assignment(
            proposals, labels, weights, anchors, "same"
        )
        self.assertTrue(torch.equal(packed[..., 2], packed[..., 3]))
        self.assertEqual(receipt["mixed_strata"], 0)

    def test_identical_group_membership_keeps_map_when_classes_change(self):
        labels = torch.tensor([[[0, 1, 2, 3]] * 18])
        proposals = labels.clone()
        proposals[..., 1] = 9
        proposals[:, :8, 2] = 8
        weights = torch.tensor([[[0.0, 1.0, 0.5, 0.25]] * 18])
        anchors = torch.arange(18).reshape(1, -1)
        _, mixed = successor_assignment(proposals, labels, weights, anchors, "stable")
        proposals[..., 2] = labels[..., 2]
        _, single_class = successor_assignment(
            proposals, labels, weights, anchors, "stable"
        )
        self.assertEqual(
            mixed["strata"][0]["donors"], single_class["strata"][0]["donors"]
        )
        self.assertEqual(single_class["flipped_bits"], 0)
        self.assertNotEqual(
            single_class["strata"][0]["donors"], single_class["strata"][0]["recipients"]
        )

    def test_rejects_unequal_stratum_mass_and_duplicate_active_anchors(self):
        proposals, labels, weights, anchors = mixed_blocks()
        weights[0, 0, 1] *= 0.5
        with self.assertRaisesRegex(ValueError, "equal native"):
            successor_assignment(proposals, labels, weights, anchors, "x")
        proposals, labels, weights, anchors = mixed_blocks()
        anchors[0, 1] = anchors[0, 0]
        with self.assertRaisesRegex(ValueError, "unique"):
            successor_assignment(proposals, labels, weights, anchors, "x")
        weights[0, 1] = 0
        successor_assignment(proposals, labels, weights, anchors, "x")

    def test_real_permutation_changes_learning_but_preserves_selected_weights(self):
        proposals, labels, weights, anchors = mixed_blocks()
        packed, receipt = successor_assignment(
            proposals, labels, weights, anchors, "train_00512", eos_token_ids=(30,)
        )
        rows = (
            torch.linspace(0.1, 8, labels.numel()).reshape_as(weights).requires_grad_()
        )

        def chunk(losses, p, y, w, assignment):
            mask = first_error_mask(p, y, w, (30,))
            groups = assigned_group_terms(losses, p, y, w, mask, assignment, (30,))
            return ((losses * w * mask).sum(), (w * mask).sum(), *groups)

        terms = checkpointed_chunk_reduce(
            chunk,
            rows,
            proposals,
            labels,
            weights,
            packed,
            chunk_size=8,
            dim=1,
            checkpoint=True,
        )
        changed, actual = redistribute_response(terms[0], terms[1], terms[2:8])
        native, original = redistribute_response(terms[0], terms[1], terms[8:14])
        self.assertNotEqual(float(changed.detach()), float(native.detach()))
        for name in (
            "W1",
            "W2",
            "coefficient1",
            "coefficient2",
            "first_error_weight_after",
        ):
            torch.testing.assert_close(
                actual["kplus_" + name], original["kplus_" + name], rtol=2e-6, atol=2e-7
            )
        (changed / (terms[1] + 1e-6)).backward()
        expected_weights = weights * packed[..., 0]
        error_coeff = torch.where(
            packed[..., 3], actual["kplus_coefficient2"], actual["kplus_coefficient1"]
        )
        expected_weights = torch.where(
            packed[..., 1], expected_weights * error_coeff, expected_weights
        )
        torch.testing.assert_close(
            rows.grad, expected_weights / (terms[1] + 1e-6), rtol=2e-6, atol=2e-7
        )
        self.assertGreater(receipt["flipped_bits"], 0)

    def models(self, identity=True):
        author = tiny_author(chunk_blocks=8)
        native = TargetObjectiveModel(
            copy.deepcopy(author), "osd_first_error", loss_precision="bfloat16"
        )
        changed = TargetObjectiveModel(
            copy.deepcopy(author), "osd_first_error", loss_precision="bfloat16"
        )
        install_kplus(native)
        install_kplus(changed)
        self.assertEqual(
            install_successor_shuffle(changed, identity=identity),
            successor_shuffle_contract(identity=identity),
        )
        for model in (native, changed):
            model.begin_event()
        return native, changed

    def test_native_bf16_two_response_identity_loss_gradient_adam_rng_exact(self):
        native, changed = self.models()
        batch = response_batch(native)
        batch["target_last_hidden_states"] = torch.randn(
            1, 12, 16, dtype=torch.bfloat16
        )
        optimizers = [
            DirectBF16AdamW(model.draft_model, learning_rate=1e-4)
            for model in (native, changed)
        ]
        initial_rng = capture_rng()
        outputs = []
        for model, optimizer in zip((native, changed), optimizers):
            restore_rng(initial_rng)
            losses = []
            for i in range(2):
                if model is changed:
                    model.successor_shuffle.set_response_id("response-" + str(i))
                loss, _, metrics = model(**batch)
                losses.append(loss.detach().clone())
                (loss / 2).backward()
            gradients = {
                n: p.grad.clone() for n, p in model.draft_model.named_parameters()
            }
            grad_norm = optimizer.step()
            anchors = model._sample_anchor_positions(
                12, batch["loss_mask"], torch.device("cpu")
            )
            outputs.append(
                (
                    losses,
                    gradients,
                    copy.deepcopy(optimizer.state_dict()),
                    anchors,
                    capture_rng(),
                    grad_norm,
                )
            )
        for a, b in zip(outputs[0][0], outputs[1][0]):
            self.assertTrue(torch.equal(a, b))
        for name, grad in outputs[0][1].items():
            self.assertTrue(torch.equal(grad, outputs[1][1][name]), name)
        for (name, a), (_, b) in zip(
            native.draft_model.named_parameters(),
            changed.draft_model.named_parameters(),
        ):
            self.assertTrue(torch.equal(a, b), name)
        self.assertEqual(outputs[0][2]["recipe"], outputs[1][2]["recipe"])
        self.assertEqual(
            outputs[0][2]["optimizer_state_dict"]["param_groups"],
            outputs[1][2]["optimizer_state_dict"]["param_groups"],
        )
        for key, state in outputs[0][2]["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                self.assertTrue(
                    torch.equal(
                        value, outputs[1][2]["optimizer_state_dict"]["state"][key][name]
                    )
                )
        for a, b in zip(outputs[0][3], outputs[1][3]):
            self.assertTrue(torch.equal(a, b))
        self.assert_rng_equal(outputs[0][4], outputs[1][4])
        self.assertTrue(torch.equal(outputs[0][5], outputs[1][5]))
        self.assertEqual(changed.costs["shuffle_presentations"], 2)
        self.assertEqual(changed.costs["shuffle_student_projection_calls"], 2)
        self.assertEqual(changed.successor_shuffle.last_receipt["flipped_bits"], 0)
        self.assertTrue(changed.successor_shuffle.last_receipt["mass_preserved"])

    def test_prepass_extra_projection_count_and_explicit_masks_survive_state_replacement(
        self,
    ):
        native, changed = self.models(identity=False)
        proposals, labels, weights, anchors = mixed_blocks()
        changed.block_size = 16
        changed.eos_token_ids = (30,)
        changed.lm_head = torch.nn.Identity()
        logits = torch.full((*labels.shape, 31), -2.0, dtype=torch.bfloat16)
        logits.scatter_(-1, proposals.unsqueeze(-1), 3.0)
        logits.requires_grad_()
        teacher = torch.zeros_like(logits)
        extension = changed.successor_shuffle
        rng = capture_rng()
        extension.set_response_id("train_00512")
        with mock.patch.object(
            changed.lm_head, "forward", wraps=changed.lm_head.forward
        ) as head:
            packed = extension.prepare_response(
                changed, logits, labels, weights, anchors
            )
            self.assertEqual(head.call_count, 4)
        self.assert_rng_equal(rng, capture_rng())
        terms = checkpointed_chunk_reduce(
            changed._chunk_loss,
            logits,
            None,
            labels,
            weights,
            teacher,
            None,
            None,
            None,
            packed,
            chunk_size=8,
            dim=1,
            checkpoint=True,
        )
        numerator, metrics = changed.first_error_redistribution.reduce_response(
            terms[:11]
        )
        sums = extension.finish_response(terms, metrics)
        self.assertEqual(extension.qualification_snapshot()["eos_token_ids"], [30])
        self.assertGreater(float(sums["shuffle_flipped_bits"]), 0)
        self.assertGreater(float(sums["shuffle_reassigned_weight"]), 0)
        reference_logits = logits.detach().clone().requires_grad_()
        reference_terms = checkpointed_chunk_reduce(
            changed._chunk_loss,
            reference_logits,
            None,
            labels,
            weights,
            teacher,
            None,
            None,
            None,
            packed,
            chunk_size=8,
            dim=1,
            checkpoint=True,
        )
        reference_numerator, _ = changed.first_error_redistribution.reduce_response(
            reference_terms[:11]
        )
        (reference_numerator / (reference_terms[1] + 1e-6)).backward()
        extension.set_response_id("unrelated-later-response")
        extension.prepare_response(changed, logits, labels, weights, anchors)
        (numerator / (terms[1] + 1e-6)).backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(torch.equal(logits.grad, reference_logits.grad))
        self.assertEqual(changed.costs["shuffle_presentations"], 2)
        # Backward replay cannot prepare or count a third response.
        self.assertEqual(changed.costs["shuffle_student_projection_calls"], 8)

    def test_mixed_bf16_identity_preserves_native_graph_exactly(self):
        native, changed = self.models(identity=True)
        proposals, labels, weights, anchors = mixed_blocks()
        original = torch.full((*labels.shape, 31), -2.0, dtype=torch.bfloat16)
        original.scatter_(-1, proposals.unsqueeze(-1), 3.0)
        teacher = torch.zeros_like(original)
        outputs = []
        for model in (native, changed):
            model.block_size = 16
            model.lm_head = torch.nn.Identity()
            logits = original.clone().requires_grad_()
            extra = ()
            if model is changed:
                model.successor_shuffle.set_response_id("mixed-identity")
                packed = model.successor_shuffle.prepare_response(
                    model, logits, labels, weights, anchors
                )
                extra = (None, None, None, packed)
            terms = checkpointed_chunk_reduce(
                model._chunk_loss,
                logits,
                None,
                labels,
                weights,
                teacher,
                *extra,
                chunk_size=8,
                dim=1,
                checkpoint=True,
            )
            numerator, metrics = model.first_error_redistribution.reduce_response(
                terms[:11]
            )
            self.assertGreater(float(metrics["kplus_W1"]), 0)
            self.assertGreater(float(metrics["kplus_W2"]), 0)
            loss = numerator / (terms[1] + 1e-6)
            loss.backward()
            outputs.append((loss.detach(), logits.grad.clone()))
        for a, b in zip(outputs[0], outputs[1]):
            self.assertTrue(torch.equal(a, b))

    def test_failed_prepass_records_attempted_projection_work(self):
        _, changed = self.models()
        proposals, labels, weights, anchors = mixed_blocks()
        hidden = torch.zeros((*labels.shape, 16), dtype=torch.bfloat16)
        changed.successor_shuffle.set_response_id("failed-response")
        with mock.patch.object(
            changed.lm_head,
            "forward",
            side_effect=[
                torch.zeros((*labels[:, :8].shape, 31)),
                RuntimeError("projection failed"),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "projection failed"):
                changed.successor_shuffle.prepare_response(
                    changed, hidden, labels, weights, anchors
                )
        receipt = changed.successor_shuffle.last_receipt
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["projection_calls"], 2)
        self.assertEqual(receipt["projection_completed"], 1)
        self.assertEqual(changed.costs["shuffle_student_projection_calls"], 2)
        self.assertGreaterEqual(receipt["projection_s"], 0)
        changed.successor_shuffle.set_response_id("next-response")
        self.assertEqual(
            changed.successor_shuffle.last_receipt["response_id"], "next-response"
        )
        self.assertEqual(changed.successor_shuffle.last_receipt["status"], "pending")
        with self.assertRaisesRegex(RuntimeError, "No response"):
            changed.successor_shuffle.qualification_snapshot()
        changed.lm_head = torch.nn.Identity()
        logits = torch.full((*labels.shape, 31), -2.0, dtype=torch.bfloat16)
        logits.scatter_(-1, proposals.unsqueeze(-1), 3.0)
        weights[0, 0, 1] *= 0.5
        with self.assertRaisesRegex(ValueError, "equal native"):
            changed.successor_shuffle.prepare_response(
                changed, logits, labels, weights, anchors
            )
        self.assertEqual(changed.successor_shuffle.last_receipt["status"], "failed")
        self.assertEqual(
            changed.successor_shuffle.last_receipt["projection_completed"], 4
        )
        self.assertGreaterEqual(
            changed.successor_shuffle.last_receipt["assignment_s"], 0
        )

    def test_native_bf16_neutral_branches_keep_exact_numerator_and_gradients(self):
        for case in ("no_errors", "only_low", "only_high", "empty"):
            with self.subTest(case=case):
                native, changed = self.models(identity=False)
                labels = torch.arange(16).reshape(1, 1, 16).expand(1, 9, 16).clone()
                proposals = labels.clone()
                weights = torch.ones_like(labels, dtype=torch.float32)
                weights[..., 0] = 0
                if case.startswith("only_"):
                    proposals[..., 1] = 29
                    if case == "only_low":
                        proposals[..., 2] = 28
                if case == "empty":
                    weights.zero_()
                anchors = torch.arange(9).reshape(1, -1)
                original = torch.full((*labels.shape, 31), -2.0, dtype=torch.bfloat16)
                original.scatter_(-1, proposals.unsqueeze(-1), 3.0)
                teacher = torch.zeros_like(original)
                outputs = []
                for model in (native, changed):
                    model.block_size = 16
                    model.lm_head = torch.nn.Identity()
                    logits = original.clone().requires_grad_()
                    extra = ()
                    if model is changed:
                        model.successor_shuffle.set_response_id(case)
                        packed = model.successor_shuffle.prepare_response(
                            model, logits, labels, weights, anchors
                        )
                        extra = (None, None, None, packed)
                    terms = checkpointed_chunk_reduce(
                        model._chunk_loss,
                        logits,
                        None,
                        labels,
                        weights,
                        teacher,
                        *extra,
                        chunk_size=8,
                        dim=1,
                        checkpoint=True,
                    )
                    numerator, _ = model.first_error_redistribution.reduce_response(
                        terms[:11]
                    )
                    self.assertIs(numerator, terms[0])
                    loss = numerator / (terms[1] + 1e-6)
                    loss.backward()
                    outputs.append((loss.detach(), logits.grad.clone()))
                for a, b in zip(outputs[0], outputs[1]):
                    self.assertTrue(torch.equal(a, b))

    def test_native_masks_must_agree_and_installer_does_not_combine_extensions(self):
        proposals, labels, weights, anchors = mixed_blocks()
        packed, _ = successor_assignment(proposals, labels, weights, anchors, "x")
        mask = first_error_mask(proposals, labels, weights)
        packed[..., 0] = False
        with self.assertRaisesRegex(RuntimeError, "masks differ"):
            assigned_group_terms(
                torch.ones_like(weights), proposals, labels, weights, mask, packed
            )
        native, changed = self.models()
        with self.assertRaisesRegex(ValueError, "another objective"):
            install_successor_shuffle(changed)
        with self.assertRaisesRegex(ValueError, "identity"):
            changed.successor_shuffle.set_response_id("")
        batch = response_batch(native)
        batch["target_last_hidden_states"] = torch.zeros(
            1, 12, 16, dtype=torch.bfloat16
        )
        with self.assertRaisesRegex(ValueError, "before every"):
            changed(**batch)


if __name__ == "__main__":
    unittest.main()
