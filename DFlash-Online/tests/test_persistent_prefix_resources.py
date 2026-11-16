"""CPU checks for complete storage admission and shared two-card budgets."""
import math
import unittest

from experiments.persistent_prefix_withholding import resources


class StorageAdmissionTests(unittest.TestCase):
    def plan(self, **overrides):
        values = dict(weight_bytes=1_000_000, recovery_bytes=3_000_000,
                      hidden_size=128, captured_layers=3)
        values.update(overrides)
        return resources.storage_plan([100] * 128, **values)

    def test_all_three_feedback_pools_and_full_output_cap(self):
        result = self.plan()
        row_bytes = 128 * 4 * 2 + 12
        self.assertEqual(result['row_bytes'], row_bytes)
        self.assertEqual(result['prepared_feature_bytes'], 3 * 128 * 2148 * row_bytes)
        self.assertEqual(result['temporary_feature_bytes'], 2 * 2149 * row_bytes)
        self.assertTrue(result['unique_features_retained'])

    def test_checkpoints_serialized_publication_and_all_allowances(self):
        result = self.plan(prior_allocation_bytes=123_456)
        self.assertEqual(result['checkpoint_bytes'], 4 * 3_000_000 + 5 * 1_000_000)
        self.assertEqual(result['atomic_publication_bytes'], 4_000_000)
        self.assertEqual(result['publication_concurrency'], 1)
        payload = sum(result[key] for key in ('prepared_feature_bytes',
            'temporary_feature_bytes', 'checkpoint_bytes', 'atomic_publication_bytes'))
        self.assertEqual(result['serialization_margin_bytes'], math.ceil(payload * .05))
        extras = sum(result[key] for key in ('serialization_margin_bytes',
            'compact_bytes', 'compiler_bytes', 'source_and_workspace_bytes',
            'prior_allocation_bytes'))
        self.assertEqual(result['peak_new_bytes'], payload + extras)
        self.assertEqual(result['required_initial_free_bytes'],
                         result['peak_new_bytes'] + resources.FREE_RESERVE_BYTES)

    def test_fixed_baseline_allocation_is_not_reset(self):
        initial = self.plan()
        previous_writes = resources.ALLOCATION_BYTES - initial['peak_new_bytes']
        self.assertEqual(self.plan(prior_allocation_bytes=previous_writes)['status'], 'passed')
        self.assertEqual(self.plan(prior_allocation_bytes=previous_writes + 1)['status'], 'blocked')

    def test_prior_allocation_cannot_reduce_or_corrupt_peak(self):
        for value in (-1, float('nan'), float('inf'), 0.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan(prior_allocation_bytes=value)

    def test_full_panel_and_context_admission_required(self):
        for lengths in ([100] * 127, [100] * 129, [0] * 128, [2049] * 128,
                        [100.] * 128, [True] * 128):
            with self.subTest(lengths=lengths[:1]), self.assertRaises(ValueError):
                resources.storage_plan(lengths, weight_bytes=1000, recovery_bytes=3000,
                                       hidden_size=128, captured_layers=3)
        boundary = resources.storage_plan([2048] * 128, weight_bytes=1000,
            recovery_bytes=3000, hidden_size=128, captured_layers=3)
        self.assertEqual(boundary['prepared_feature_bytes'], 3 * 128 * 4096 * boundary['row_bytes'])

    def test_file_and_tensor_sizes_must_be_positive(self):
        for key in ('weight_bytes', 'recovery_bytes', 'hidden_size', 'captured_layers'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.plan(**{key: 0})


class TimeAdmissionTests(unittest.TestCase):
    def test_occupied_time_sums_both_cards_while_elapsed_uses_schedule(self):
        result = resources.time_admission(occupied_spent_s=100,
            remaining_components={'card_a': 1000, 'card_b': 1000},
            elapsed_spent_s=50, remaining_schedule_s=1000)
        self.assertEqual(result['raw_remaining_card_s'], 2000)
        self.assertEqual(result['conservative_total_card_s'], 4900)
        self.assertEqual(result['conservative_total_elapsed_s'], 3350)
        self.assertEqual(result['status'], 'passed')

    def test_card_cap_boundary_is_inclusive_and_reserve_is_not_omitted(self):
        for remaining, expected in ((8400, 'passed'), (8401, 'blocked')):
            with self.subTest(remaining=remaining):
                result = resources.time_admission(occupied_spent_s=0,
                    remaining_components={'work': remaining}, elapsed_spent_s=0,
                    remaining_schedule_s=remaining / 2)
                self.assertEqual(result['status'], expected)

    def test_elapsed_cap_independently_blocks(self):
        result = resources.time_admission(occupied_spent_s=0,
            remaining_components={'work': 1}, elapsed_spent_s=14_399,
            remaining_schedule_s=0)
        self.assertEqual(result['status'], 'blocked')
        self.assertLess(result['conservative_total_card_s'], resources.CARD_SECONDS)

    def test_negative_and_nonfinite_time_cannot_be_admitted(self):
        normal = dict(occupied_spent_s=0, remaining_components={'work': 1},
                      elapsed_spent_s=0, remaining_schedule_s=1)
        for field in ('occupied_spent_s', 'elapsed_spent_s', 'remaining_schedule_s', 'work'):
            for invalid in (-1, float('nan'), float('inf')):
                arguments = dict(normal)
                if field == 'work':
                    arguments['remaining_components'] = {'work': invalid}
                else:
                    arguments[field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    resources.time_admission(**arguments)


class WorkingSetTests(unittest.TestCase):
    def test_parallel_admission_retains_host_reserve(self):
        need = 6000 + resources.HOST_HEADROOM_BYTES
        accepted = resources.admit_task_pair(available_host_bytes=need,
            new_working_sets=[1000, 5000], resident_counted=True)
        denied = resources.admit_task_pair(available_host_bytes=need - 1,
            new_working_sets=[1000, 5000], resident_counted=True)
        self.assertEqual(accepted['status'], 'passed')
        self.assertEqual(denied['status'], 'serialize')
        self.assertEqual(accepted['required_bytes'], need)
        self.assertTrue(accepted['resident_workers_already_in_memavailable'])

    def test_negative_working_set_cannot_reduce_reservation(self):
        with self.assertRaises(ValueError):
            resources.admit_task_pair(available_host_bytes=100, new_working_sets=[-1])


if __name__ == '__main__':
    unittest.main()
