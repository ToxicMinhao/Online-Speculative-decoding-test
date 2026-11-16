"""One dependency/card/RAM calendar is used before and during production."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.persistent_prefix_withholding import campaign as C
from experiments.persistent_prefix_withholding.resources import HOST_HEADROOM_BYTES


def tasks():
    def task(name, dependencies, kind, duration):
        return C.Task(name, tuple(dependencies), kind, dict(work=float(duration)))
    rows = [task('E7', (), 'training', 10), task('S7', (), 'training', 10)]
    for arm in ('N', 'E', 'S'):
        rows += [task(arm+'_collect', () if arm == 'N' else (arm+'7',), 'collection', 3),
                 task(arm+'8', (arm+'_collect',), 'training', 10)]
    rows += [task('references', (), 'reference', 2),
             task('final', ('N8', 'E8', 'S8', 'references'), 'final', 5)]
    return rows


class RemainingScheduleTests(unittest.TestCase):
    def setUp(self):
        self.rows = tasks()
        self.devices = dict(A='GPU-A', B='GPU-B')
        self.bounds = dict(training=6, collection=4, reference=4, final=4)

    def schedule(self, *, completed=(), active=None, capacity=12):
        return C.remaining_schedule(self.rows, completed=set(completed), active=active or {},
            devices=self.devices, task_memory_bytes=self.bounds,
            available_host_bytes=HOST_HEADROOM_BYTES+capacity)

    def test_initial_ready_tasks_and_two_card_assignment(self):
        result = self.schedule()
        self.assertEqual(result['status'], 'passed')
        first = [row for row in result['schedule'] if row['start_s'] == 0]
        self.assertEqual([row['task'] for row in first], ['E7', 'S7'])
        self.assertEqual([row['uuid'] for row in first], ['GPU-A', 'GPU-B'])
        self.assertEqual(result['remaining_card_s'], sum(row.estimated_card_s for row in self.rows))
        self.assertLess(result['remaining_elapsed_s'], result['remaining_card_s'])

    def test_completed_dependencies_are_removed_not_charged_again(self):
        result = self.schedule(completed=('E7', 'S7'))
        self.assertNotIn('E7', result['remaining_components'])
        self.assertNotIn('S7', result['remaining_components'])
        self.assertEqual(result['remaining_card_s'], self.schedule()['remaining_card_s']-20)
        first = [row['task'] for row in result['schedule'] if row['start_s'] == 0]
        self.assertEqual(len(first), 2)
        self.assertTrue(set(first) <= {'E_collect', 'S_collect', 'N_collect', 'references'})

    def test_one_active_task_preserves_card_and_explicit_remaining_reservation(self):
        active = dict(E7=dict(uuid='GPU-B', remaining_s=7., resident_bytes=6))
        result = self.schedule(active=active, capacity=6)
        first = result['schedule'][0]
        self.assertEqual((first['task'], first['uuid'], first['finish_s']), ('E7', 'GPU-B', 7.))
        self.assertEqual(result['remaining_components']['E7'], 7.)
        self.assertEqual(result['remaining_card_s'], self.schedule()['remaining_card_s']-3)
        for row in result['schedule'][1:]:
            if row['uuid'] == 'GPU-B':
                self.assertGreaterEqual(row['start_s'], 7.)

    def test_two_active_tasks_use_free_ram_plus_observed_residents_once(self):
        active = dict(E7=dict(uuid='GPU-A', remaining_s=10., resident_bytes=6),
                      S7=dict(uuid='GPU-B', remaining_s=10., resident_bytes=6))
        result = self.schedule(active=active, capacity=0)
        self.assertEqual(result['host_capacity_bytes'], 12)
        self.assertEqual(result['remaining_elapsed_s'], self.schedule()['remaining_elapsed_s'])
        self.assertEqual(result['remaining_components'], self.schedule()['remaining_components'])

    def test_ram_serializes_only_incompatible_pairs(self):
        result = self.schedule(capacity=8)
        self.assertEqual(result['status'], 'passed')
        self.assertGreater(result['remaining_elapsed_s'], self.schedule()['remaining_elapsed_s'])
        self.assertLess(result['remaining_elapsed_s'], result['remaining_card_s'])
        for start in [row['start_s'] for row in result['schedule']]:
            overlapping = [row for row in result['schedule'] if row['start_s'] <= start < row['finish_s']]
            self.assertLessEqual(sum(row['reserved_host_bytes'] for row in overlapping), 8)

    def test_single_task_or_active_growth_overcommit_is_blocked(self):
        self.assertEqual(self.schedule(capacity=5)['status'], 'blocked')
        active = dict(E7=dict(uuid='GPU-A', remaining_s=10., resident_bytes=8),
                      S7=dict(uuid='GPU-B', remaining_s=10., resident_bytes=8))
        self.assertEqual(self.schedule(active=active, capacity=-1)['status'], 'blocked')

    def test_quiet_final_waits_for_all_tasks_and_uses_card_A(self):
        for capacity in (8, 12):
            schedule = self.schedule(capacity=capacity)['schedule']
            final = next(row for row in schedule if row['task'] == 'final')
            self.assertEqual(final['uuid'], 'GPU-A')
            self.assertGreaterEqual(final['start_s'], max(row['finish_s'] for row in schedule if row['task'] != 'final'))

    def test_empty_queue_is_zero_remaining_work(self):
        result = self.schedule(completed=[row.name for row in self.rows], capacity=0)
        self.assertEqual(result['remaining_elapsed_s'], 0.)
        self.assertEqual(result['remaining_card_s'], 0.)
        self.assertEqual(result['schedule'], [])

    def test_duplicate_gpu_or_impossible_dependency_state_rejected(self):
        active = dict(E7=dict(uuid='GPU-A', remaining_s=1., resident_bytes=6),
                      S7=dict(uuid='GPU-A', remaining_s=1., resident_bytes=6))
        with self.assertRaisesRegex(ValueError, 'share a physical card'):
            self.schedule(active=active)
        with self.assertRaisesRegex(ValueError, 'unfinished dependency'):
            self.schedule(completed=['E8'])

    def test_entry_and_first_active_boundary_use_identical_pricing(self):
        plan = dict(devices=self.devices, task_memory_bytes=self.bounds)
        clock = dict(occupied_s=435.09486651420593, elapsed_s=5100.)
        first = C.schedule_admission(plan, self.rows, completed=set(), active={}, clock=clock,
                                     available_host_bytes=HOST_HEADROOM_BYTES+12)
        active = {name: dict(uuid=self.devices[slot], task=next(row for row in self.rows if row.name == name),
                            process=SimpleNamespace(pid=i)) for i,(name,slot) in enumerate((('E7','A'),('S7','B')))}
        with patch.object(C, 'process_tree_rss', return_value=6):
            boundary = C.schedule_admission(plan, self.rows, completed=set(), active=active, clock=clock,
                                            available_host_bytes=HOST_HEADROOM_BYTES)
        for key in ('status', 'raw_remaining_card_s', 'conservative_total_card_s', 'conservative_total_elapsed_s'):
            self.assertEqual(first[key], boundary[key])
        self.assertEqual(first['occupied_spent_s'], clock['occupied_s'])

    def test_elapsed_and_summed_card_time_are_independent_limits(self):
        plan = dict(devices=self.devices, task_memory_bytes=self.bounds)
        for clock in (dict(occupied_s=14400., elapsed_s=0.), dict(occupied_s=0., elapsed_s=14400.)):
            result = C.schedule_admission(plan,self.rows,completed=set(),active={},clock=clock,
                                          available_host_bytes=HOST_HEADROOM_BYTES+12)
            self.assertEqual(result['status'], 'blocked')

    def test_old_commits_require_explicit_authenticated_continuation(self):
        plan = dict(execution_commit='new', devices=self.devices,
            qualification_receipts={key:key for key in ('A','B','cross_card','serving')},
            cpu_gate='cpu', reference_provenance='references', resource_admission='resources')
        receipts = {key:dict(status='passed', execution_commit='old') for key in plan['qualification_receipts']}
        receipts.update(cpu=dict(status='passed',execution_commit='new'),
                        references=dict(status='passed'),resources=dict(status='passed'))
        from experiments.persistent_prefix_withholding import production_continue as continuation
        with patch.object(C,'read',side_effect=receipts.__getitem__), \
             patch.object(continuation,'qualified_receipt_matches',return_value=True) as authenticate:
            with self.assertRaisesRegex(ValueError,'Unpassed qualification'):
                C.mandatory_gates(plan)
            authenticate.assert_not_called()
            plan['production_continuation'] = dict(path='bound-manifest',identity={'sha256':'bound'})
            C.mandatory_gates(plan)
            self.assertEqual(authenticate.call_count,4)
        with patch.object(C,'read',side_effect=receipts.__getitem__), \
             patch.object(continuation,'qualified_receipt_matches',return_value=False):
            with self.assertRaisesRegex(ValueError,'Unpassed qualification'):
                C.mandatory_gates(plan)


if __name__ == '__main__':
    unittest.main()
