"""Exact Track 2 CPU allocation, descriptor and reservoir contracts."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import random
import unittest

import numpy as np
import torch

from experiments.k_workload_replay_v1.selection import (
    ARMS,
    ReplayRecord,
    ReservoirMemory,
    build_event_plan,
    prompt_descriptor,
)


def event_records(event_index, descriptor=(1.0, 0.0)):
    return [
        ReplayRecord(f"original_{ordinal:04d}", ordinal, event_index, descriptor)
        for ordinal in range((event_index - 1) * 128, event_index * 128)
    ]


def strip_cost(plan):
    result = deepcopy(plan)
    result.pop("cost")
    return result


class PromptDescriptorTests(unittest.TestCase):
    def test_exact_last_prompt_row_full_width_float64_and_no_mutation(self):
        packet = np.zeros((1, 5, 4), dtype=np.float32)
        packet[0, 1] = [1, 2, 2, 4]
        packet[0, 2:] = np.nan
        before = packet.copy()
        descriptor, receipt = prompt_descriptor(packet, 2)
        expected = np.array([1, 2, 2, 4], dtype=np.float64) / 5
        np.testing.assert_array_equal(descriptor, expected)
        np.testing.assert_array_equal(packet, before)
        self.assertEqual(receipt["hidden_dimensions"], 4)
        self.assertEqual(receipt["original_norm"], 5.0)
        self.assertEqual(receipt["normalized_norm"], 1.0)
        self.assertEqual(receipt["row_index"], 1)

    def test_bfloat16_tensor_is_converted_on_cpu_without_autograd(self):
        packet = torch.tensor([[[0, 2, 0, 0], [1, 0, 0, 0]]], dtype=torch.bfloat16,
                              requires_grad=True)
        descriptor, receipt = prompt_descriptor(packet, 1)
        self.assertEqual(descriptor, (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(receipt["dtype"], "float64")
        self.assertIsNone(packet.grad)

    def test_zero_nonfinite_and_bad_prompt_boundaries_fail(self):
        for row in ([0, 0], [float("nan"), 1], [float("inf"), 0]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                prompt_descriptor(np.array([[row]]), 1)
        for length in (0, 3, -1, 1.5):
            with self.subTest(length=length), self.assertRaises(ValueError):
                prompt_descriptor(np.ones((1, 2, 3)), length)
        with self.assertRaises(ValueError):
            prompt_descriptor(np.ones((2, 2, 3)), 1)

    def test_record_copies_descriptor_and_is_immutable(self):
        descriptor = np.array([1.0, 0.0])
        record = ReplayRecord("original", 0, 1, descriptor)
        descriptor[0] = 0
        self.assertEqual(record.descriptor, (1.0, 0.0))
        with self.assertRaises(FrozenInstanceError):
            record.record_id = "renamed"
        for bad in ((0.0, 0.0), (2.0, 0.0), (float("nan"), 0.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ReplayRecord("invalid", 0, 1, bad)
        with self.assertRaisesRegex(ValueError, "chronology"):
            ReplayRecord("invalid", 128, 1, (1.0, 0.0))


class EventSelectionTests(unittest.TestCase):
    def test_event_one_is_exact_neutral_two_passes_for_all_arms(self):
        fresh = event_records(1)
        plan = build_event_plan(1, list(reversed(fresh)), [])
        for arm in ARMS:
            self.assertEqual(plan["pass_record_ids"][arm], [[r.record_id for r in fresh]] * 2)
            self.assertEqual(plan["pass_global_ordinals"][arm], [list(range(128))] * 2)
        self.assertIsNone(plan["uniform"])
        self.assertIsNone(plan["coverage"])

    def test_hand_computed_coverage_and_ties_continue_to_32_unique(self):
        old = event_records(1)
        old[0] = ReplayRecord(old[0].record_id, 0, 1, (0.6, 0.8))
        old[1] = ReplayRecord(old[1].record_id, 1, 1, (1.0, 0.0))
        old[2] = ReplayRecord(old[2].record_id, 2, 1, (0.0, 1.0))
        fresh = event_records(2)
        for index in range(64, 128):
            ordinal = 128 + index
            fresh[index] = ReplayRecord(fresh[index].record_id, ordinal, 2, (0.0, 1.0))
        selection = build_event_plan(2, fresh, list(reversed(old)))["coverage"]
        self.assertEqual(selection["global_ordinals"], list(range(32)))
        np.testing.assert_allclose([s["coverage"] for s in selection["steps"][:3]],
                                   [0.7, 0.9, 1.0], rtol=0, atol=5e-16)
        np.testing.assert_allclose([s["marginal_gain"] for s in selection["steps"][:3]],
                                   [0.7, 0.2, 0.1], rtol=0, atol=5e-16)
        self.assertTrue(all(s["marginal_gain"] == 0 for s in selection["steps"][3:]))
        self.assertEqual(len(set(selection["record_ids"])), 32)
        self.assertEqual(selection["concentration"]["herfindahl"], 0.5)
        self.assertEqual(selection["concentration"]["effective_records"], 2.0)

    def test_negative_cosines_are_zero_and_exact_ties_choose_ordinals(self):
        plan = build_event_plan(2, event_records(2), list(reversed(event_records(1, (-1.0, 0.0)))))
        selection = plan["coverage"]
        self.assertEqual(selection["global_ordinals"], list(range(32)))
        self.assertEqual(selection["coverage"], 0)
        self.assertTrue(all(s["marginal_gain"] == 0 for s in selection["steps"]))
        self.assertEqual(selection["concentration"]["zero_coverage_queries"], 128)
        self.assertEqual(selection["concentration"]["herfindahl"], 0)

    def test_upper_cosine_clip_prevents_scores_above_one(self):
        slightly_long = (1.0 + 2e-13, 0.0)
        plan = build_event_plan(2, event_records(2, slightly_long), event_records(1, slightly_long))
        self.assertEqual(plan["coverage"]["coverage"], 1.0)
        self.assertEqual(plan["memory_similarity_distribution"]["max"], 1.0)

    def test_seed_sequences_exposure_order_and_original_identity(self):
        old, fresh = event_records(1), event_records(2)
        plan = build_event_plan(2, fresh, old)
        expected_fresh = sorted(np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([20261102, 2, 0]))).choice(128, 96, replace=False).tolist())
        expected_old = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([20261102, 2, 1]))).choice(128, 32, replace=False).tolist()
        self.assertEqual(plan["common_fresh_record_ids"], [fresh[i].record_id for i in expected_fresh])
        self.assertEqual(plan["uniform"]["record_ids"], [old[i].record_id for i in expected_old])
        for arm in ARMS:
            for ids, ordinals in zip(plan["pass_record_ids"][arm], plan["pass_global_ordinals"][arm]):
                self.assertEqual(len(ids), 128)
                self.assertEqual(len(set(ids)), 128)
                self.assertEqual(ordinals, sorted(ordinals))
                self.assertEqual(ids, [f"original_{ordinal:04d}" for ordinal in ordinals])
            self.assertEqual(len(plan["pass_record_ids"][arm][0]) + len(plan["pass_record_ids"][arm][1]), 256)
        for arm in ("U", "R"):
            second = plan["pass_global_ordinals"][arm][1]
            self.assertEqual(sum(i < 128 for i in second), 32)
            self.assertEqual([i - 128 for i in second if i >= 128], expected_fresh)

    def test_past_only_exact_event_unique_identity_and_dimension_boundaries(self):
        old, fresh = event_records(1), event_records(2)
        cases = [(2, fresh, fresh), (2, fresh, old[:-1]), (1, old, old),
                 (2, fresh[:-1], old), (3, fresh, old), (2, fresh, old[:-1] + old[:1])]
        for args in cases:
            with self.subTest(event=args[0], old_size=len(args[2])), self.assertRaises(ValueError):
                build_event_plan(*args)
        renamed = list(fresh)
        renamed[0] = ReplayRecord(old[0].record_id, 128, 2, (1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_event_plan(2, renamed, old)
        with self.assertRaisesRegex(ValueError, "dimension"):
            build_event_plan(2, event_records(2, (1.0, 0.0, 0.0)), old)

    def test_input_order_does_not_change_selection_and_receipts_are_json_safe(self):
        old, fresh = event_records(1), event_records(2)
        first = build_event_plan(2, fresh, old)
        second = build_event_plan(2, list(reversed(fresh)), list(reversed(old)))
        self.assertEqual(strip_cost(first), strip_cost(second))
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)
        self.assertEqual(first["coverage"]["steps"][0]["age_events"], 1)
        self.assertEqual(first["coverage"]["steps"][0]["age_fresh_records"], 128)


class ReservoirTests(unittest.TestCase):
    def test_algorithm_r_exact_membership_all_events_and_state_roundtrip(self):
        reservoir = ReservoirMemory()
        reference_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261102, 0, 2])))
        reference_slots = []
        for event in range(1, 9):
            fresh = event_records(event)
            if event > 1:
                self.assertTrue(all(record.event_index < event for record in reservoir.records))
            receipt = reservoir.insert_after_event(event, list(reversed(fresh)), verified_arms=ARMS)
            for record in fresh:
                n = record.global_ordinal + 1
                if n <= 128:
                    reference_slots.append(record.global_ordinal)
                else:
                    slot = int(reference_rng.integers(n))
                    if slot < 128:
                        reference_slots[slot] = record.global_ordinal
            self.assertEqual([r.global_ordinal for r in reservoir.records], reference_slots)
            self.assertEqual(reservoir.seen_count, event * 128)
            self.assertEqual(receipt["rng_state"], reference_rng.bit_generator.state)
            saved = json.loads(json.dumps(reservoir.state_dict()))
            restored = ReservoirMemory.from_state_dict(saved)
            self.assertEqual(restored.records, reservoir.records)
            self.assertEqual(restored.state_dict(), reservoir.state_dict())
            reservoir = restored

    def test_insert_once_and_only_after_all_verified_arms_without_partial_mutation(self):
        reservoir = ReservoirMemory()
        initial = reservoir.state_dict()
        for arms in ((), ("K", "U"), ("K", "U", "R", "O")):
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                reservoir.insert_after_event(1, event_records(1), verified_arms=arms)
            self.assertEqual(reservoir.state_dict(), initial)
        reservoir.insert_after_event(1, event_records(1), verified_arms=ARMS)
        committed = reservoir.state_dict()
        with self.assertRaisesRegex(ValueError, "once"):
            reservoir.insert_after_event(1, event_records(1), verified_arms=ARMS)
        self.assertEqual(reservoir.state_dict(), committed)
        repeated = event_records(2)
        repeated[-1] = ReplayRecord("original_0000", 255, 2, (1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "once"):
            reservoir.insert_after_event(2, repeated, verified_arms=ARMS)
        self.assertEqual(reservoir.state_dict(), committed)

    def test_corrupt_state_counter_slots_identity_and_capacity_fail(self):
        reservoir = ReservoirMemory()
        reservoir.insert_after_event(1, event_records(1), verified_arms=ARMS)
        saved = reservoir.state_dict()
        mutations = [lambda s: s.update(capacity=64), lambda s: s.update(seen_count=127),
                     lambda s: s.update(last_global_ordinal=128), lambda s: s["slots"].pop(),
                     lambda s: s["slots"][0].update(record_id="foreign"),
                     lambda s: s["seen_record_ids"].__setitem__(0, s["seen_record_ids"][1])]
        for mutate in mutations:
            bad = deepcopy(saved)
            mutate(bad)
            with self.assertRaises(ValueError):
                ReservoirMemory.from_state_dict(bad)

    def test_state_dict_and_input_sequences_cannot_mutate_reservoir(self):
        reservoir = ReservoirMemory()
        fresh = event_records(1)
        reservoir.insert_after_event(1, fresh, verified_arms=ARMS)
        fresh.clear()
        saved = reservoir.state_dict()
        saved["slots"][0]["record_id"] = "foreign"
        saved["rng_state"]["state"]["state"] = 0
        self.assertEqual(reservoir.records[0].record_id, "original_0000")
        self.assertNotEqual(reservoir.state_dict()["rng_state"]["state"]["state"], 0)

    def test_all_selection_memory_and_descriptor_work_leave_training_rng_unchanged(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        reservoir = ReservoirMemory()
        reservoir.insert_after_event(1, event_records(1), verified_arms=ARMS)
        memory_before = reservoir.state_dict()
        build_event_plan(2, event_records(2), reservoir.records)
        self.assertEqual(reservoir.state_dict(), memory_before)
        prompt_descriptor(torch.ones((1, 5, 8), dtype=torch.bfloat16), 3)
        reservoir.insert_after_event(2, event_records(2), verified_arms=ARMS)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))


if __name__ == "__main__":
    unittest.main()
