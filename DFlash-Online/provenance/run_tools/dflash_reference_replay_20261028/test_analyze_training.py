"""Synthetic accounting fixtures only; no model or runtime imports."""
from copy import deepcopy
import unittest

from analyze_training import ADDITIVE, EXPECTED_WORK, reconcile


def fixture():
    features = [dict(record_id=f'r{index}') for index in range(128)]
    records, updates, schedule_updates = [], [], []
    for pass_index in range(2):
        bindings = []
        for index in range(128):
            count = 254 + int(pass_index*128 + index < 72)
            bindings.append(dict(record_id=f'r{index}',
                                 sampled_anchors=[dict(positions=[list(range(count))], keep_mask=[[True]*count])]))
            bindings[-1]['pass'] = pass_index
        pairs = [bindings[index:index+2] for index in range(0, 128, 2)]
        for batch in range(64):
            for role, pair in (('A', pairs[batch]), ('R', pairs[(batch+1) % 64])):
                step = len(updates)+1
                outer = pass_index*64+batch+1
                schedule_updates.append(dict(optimizer_step=step, outer_window=outer, role=role,
                                             records=deepcopy(pair)))
                own_metrics = []
                for offset, binding in enumerate(pair):
                    anchors = len(binding['sampled_anchors'][0]['keep_mask'][0])
                    metrics = {key: float(anchors) for key in ADDITIVE}
                    own_metrics.append(metrics)
                    reported = ({key: sum(values[key] for values in own_metrics) for key in ADDITIVE}
                                if offset else dict(metrics))
                    reported['loss'] = .25
                    row = dict(record_id=binding['record_id'], role=role, outer_window=outer,
                               scheduled_optimizer_step=step, optimizer_step_after=step-int(offset == 0),
                               optimizer_stepped=bool(offset), learning_rate=3e-4 if offset else None,
                               metrics_scope='accumulation_window' if offset else 'presentation',
                               training_s=1., feature_load_s=.2, metrics=reported,
                               sampled_anchors=deepcopy(binding['sampled_anchors']))
                    row['pass'] = pass_index
                    records.append(row)
                updates.append(dict(optimizer_step=step, native_Adam_step=step, role=role,
                                    outer_window=outer, update_wall_s=2.4, journal_s=.1))
    counts = dict(EXPECTED_WORK)
    replay_costs = {}
    for role in ('A', 'R'):
        counts[role+'_presentations_completed'] = 256
        replay_costs.update({role+'_training_s':256., role+'_feature_load_s':51.2, role+'_update_wall_s':307.2})
    replay = dict(status='completed', counts=counts, updates=updates, costs=replay_costs,
                  correction_steps=0, protected_capture=False, reference_model=False,
                  rng_policy='fixture', pass_draws=[])
    totals = {key: sum(row['metrics'][key] for row in records if row['optimizer_stepped']) for key in ADDITIVE}
    receipt = dict(status='completed', training_variant='KR', objective='osd_first_error',
                   response_presentations=512, optimizer_steps=256, start_step=0, end_step=256,
                   features=features, records=records, reference_replay=replay, selection_totals=totals)
    startup = dict(status='passed', response_evaluations=2, backwards=2, optimizer_steps=0,
                   separate_disposable_runtime=True, exact_initial_state_restored=True)
    costs = dict(startup=startup, event_wall_s=800., event={'training_s':512.}, objective_work={},
                 loading_and_initialization_s=2., peak_allocated_bytes=10, peak_reserved_bytes=12,
                 new_qualification_optimizer_steps=0, new_qualification_presentations=0)
    gates = dict(first_A={'exact_equal':True},
                 first_AR={'status':'passed', 'first_A_record_gate_passed':True,
                           'additional_forwards':0, 'additional_optimizer_steps':0, 'native_Adam_steps':[1,2]},
                 initialization=dict(exact_pinned_initial_state=True, full_rng_restored_after_construction=True,
                                     fresh_optimizer=True, inherited_qualification_learning=False),
                 closed=dict(closed=True, failed=False, accumulation_remainder=0, global_step=256, presentations=512))
    return receipt, costs, dict(updates=schedule_updates), gates


class TrainingAnalysisTests(unittest.TestCase):
    def test_reconciles_full_budget_and_does_not_double_count_boundary_exposure(self):
        receipt, costs, schedule, gates = fixture()
        summary, checks, tables = reconcile(receipt, costs, schedule, gates)
        self.assertEqual(summary['status'], 'passed', summary['issues'])
        self.assertEqual(summary['sampled_anchor_occurrences'], 130192)
        self.assertEqual(summary['roles']['A']['selected_position_occurrences'], 65096)
        self.assertEqual(summary['roles']['R']['selected_position_occurrences'], 65096)
        self.assertEqual(len(tables['PRESENTATIONS']), 512)
        self.assertEqual(len(tables['UPDATES']), 256)
        self.assertEqual(len(tables['RESPONSE_EXPOSURE']), 128)
        self.assertEqual(sum(row['selected_position_occurrences_presentation']
                             for row in tables['PRESENTATIONS']), 130192)
        self.assertEqual(sum(row['selected_position_occurrences_reported']
                             for row in tables['PRESENTATIONS']), 195288)
        self.assertTrue(all(check['passed'] for check in checks))

    def test_work_shortfall_and_failed_gate_are_visible(self):
        receipt, costs, schedule, gates = fixture()
        receipt['reference_replay']['counts']['backward_completed'] = 511
        gates['first_A']['exact_equal'] = False
        summary, checks, tables = reconcile(receipt, costs, schedule, gates)
        self.assertEqual(summary['status'], 'failed')
        self.assertIn('work_backward_completed', summary['issues'])
        self.assertIn('first_A_record_gate', summary['issues'])

    def test_wrong_anchor_binding_and_reset_Adam_counter_are_visible(self):
        receipt, costs, schedule, gates = fixture()
        receipt['records'][2]['sampled_anchors'][0]['positions'][0][0] = 999
        receipt['reference_replay']['updates'][1]['native_Adam_step'] = 1
        summary, _, _ = reconcile(receipt, costs, schedule, gates)
        self.assertIn('binding_002', summary['issues'])
        self.assertIn('continuous_native_Adam_counters', summary['issues'])


if __name__ == '__main__':
    unittest.main()
