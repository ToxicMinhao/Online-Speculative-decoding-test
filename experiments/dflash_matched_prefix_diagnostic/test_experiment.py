"""Controller failure preservation, qualification boundary and exact work budget."""
import copy
import unittest
from unittest.mock import patch

from experiments.dflash_matched_prefix_diagnostic.experiment import run_campaign


def plan_fixture(locations=1):
    return dict(groups=[dict(group_id=f'g{i}', domain='fixture', checkpoints={}, requests=[dict(
        request_id=f'r{i}', population='heldout', anchors=[dict(anchor_id=f'g{i}:{j}', reconstruction={'target_verification_count': j})
        for j in range(locations)])]) for i in range(6)])


class Backend:
    def __init__(self, group, log, mode=None):
        self.group, self.log, self.mode = group, log, mode

    def start(self, group):
        self.log.append(('start', group['group_id']))
        assert 'anchors' in group
        return {'status': 'passed'}

    def probe(self, anchor, arm):
        self.log.append(('probe', anchor['anchor_id'], arm))
        row = dict(status='passed', arm=arm, anchor_id=anchor['anchor_id'], draft_block_evaluations=1,
                   target_verifications=anchor['reconstruction']['target_verification_count'],
                   reconstruction_target_verifications=anchor['reconstruction']['target_verification_count'],
                   reconstruction_target_verify_attempts=anchor['reconstruction']['target_verification_count'],
                   scored_target_verifications=0, regenerated_response_tokens=0, training_steps=0,
                   valid=True, prefix_gain=-1)
        if self.mode == ('reconstruction_partial', arm):
            row.update(draft_block_evaluations=0, reconstruction_target_verifications=1,
                       reconstruction_target_verify_attempts=1, target_verifications=1)
            error = ValueError('saved verification reconstruction failed')
            error.diagnostic_partial = dict(row, status='failed', valid=False)
            raise error
        if self.mode in (('activation_failure', arm), ('unknown_reconstruction', arm),
                         ('missing_reconstruction_counter', arm)):
            count = 0 if self.mode[0] == 'activation_failure' else None
            row.update(draft_block_evaluations=count, reconstruction_target_verifications=count,
                       reconstruction_target_verify_attempts=count, target_verifications=count)
            if self.mode[0] == 'missing_reconstruction_counter':
                del row['reconstruction_target_verifications']
            error = RuntimeError(self.mode[0])
            error.diagnostic_partial = dict(row, status='failed', valid=False)
            raise error
        if self.mode == ('invalid', arm):
            row['valid'] = False
        if self.mode == ('partial', arm):
            error = ValueError('reconstruction failed')
            error.diagnostic_partial = dict(row, valid=False, margin=float('nan'))
            raise error
        if self.mode == ('unreceipted', arm):
            raise RuntimeError('transport failed')
        return row

    def close(self):
        self.log.append(('close', self.group['group_id']))
        if self.mode == 'close_failure':
            raise RuntimeError('worker still alive')
        return {'status': 'passed', 'costs': {'loading_s': 1.0}}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch('experiments.dflash_matched_prefix_diagnostic.experiment.validate_plan', return_value={}),
            patch('experiments.dflash_matched_prefix_diagnostic.experiment.validate_row',
                  side_effect=lambda a, arm, r: dict(status='passed' if r['valid'] else 'failed',
                                                    errors=[] if r['valid'] else ['invalid native row'])),
            patch('experiments.dflash_matched_prefix_diagnostic.experiment.compare_anchor',
                  side_effect=lambda a, f, r, arm: (dict(valid=True, correct_prefix_gain=r['prefix_gain']), []))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def execute(self, modes=None, plan=None):
        log, snapshots, events = [], [], []
        result = run_campaign(plan or plan_fixture(),
            lambda group: Backend(group, log, (modes or {}).get(group['group_id'])),
            snapshot=lambda value: snapshots.append(copy.deepcopy(value)),
            emit=lambda kind, value: events.append((kind, copy.deepcopy(value))))
        return result, log, snapshots, events

    def test_harmful_results_continue_all_groups_once(self):
        result, log, _, _ = self.execute()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['actual_draft_block_evaluations'], 18)
        self.assertEqual(result['attempted_arm_evaluations'], 18)
        self.assertEqual(result['valid_pair_comparisons'], 12)
        self.assertEqual(result['first_group_qualification']['status'], 'passed')
        self.assertEqual(result['unexecuted_groups'], [])
        self.assertEqual(result['explicit_warmups'], 0)
        for i in range(6):
            self.assertEqual([x[2] for x in log if x[:2] == ('probe', f'g{i}:0')], ['F', 'S8', 'L8'])

    def test_first_invalid_stops_before_remaining_arms_and_groups(self):
        result, log, snapshots, events = self.execute({'g0': ('invalid', 'F')})
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['first_group_qualification']['status'], 'failed')
        self.assertEqual(result['attempted_arm_evaluations'], 1)
        self.assertEqual(result['unexecuted_groups'], ['g1', 'g2', 'g3', 'g4', 'g5'])
        self.assertIn(('close', 'g0'), log)
        self.assertTrue(any(s['groups'][0]['locations'][0]['rows'] and not
                            s['groups'][0]['locations'][0]['validations'] for s in snapshots))
        self.assertEqual([kind for kind, _ in events].count('probe_raw'), 1)

    def test_later_failure_stops_only_affected_group(self):
        result, log, _, _ = self.execute({'g2': ('invalid', 'S8')})
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['first_group_qualification']['status'], 'passed')
        self.assertEqual(result['attempted_arm_evaluations'], 17)
        self.assertEqual(result['groups'][2]['unattempted_arms'], [{'anchor_id': 'g2:0', 'arms': ['L8']}])
        self.assertIn(('start', 'g5'), log)
        self.assertNotIn(('probe', 'g2:0', 'L8'), log)

    def test_transport_partial_is_retained_including_nonfinite(self):
        result, _, snapshots, events = self.execute({'g0': ('partial', 'F')})
        row = result['groups'][0]['locations'][0]['rows']['F']
        self.assertEqual(row['margin']['nonfinite_float'], 'nan')
        self.assertIn('ieee754_binary64_hex', row['margin'])
        self.assertEqual(result['actual_draft_block_evaluations'], 1)
        self.assertEqual(result['unknown_work_receipts'], 0)
        self.assertTrue(any(kind == 'probe_raw' for kind, _ in events))

    def test_missing_work_receipt_is_unknown_not_zero_proven(self):
        result, _, _, _ = self.execute({'g0': ('unreceipted', 'F')})
        self.assertEqual(result['unknown_work_receipts'], 1)
        self.assertEqual(result['attempted_arm_evaluations'], 1)
        self.assertEqual(result['actual_draft_block_evaluations'], 0)

    def test_close_failure_prevents_independent_next_worker(self):
        result, log, _, _ = self.execute({'g2': 'close_failure'})
        self.assertEqual(result['unexecuted_groups'], ['g3', 'g4', 'g5'])
        self.assertNotIn(('start', 'g3'), log)

    def test_zero_round_requests_retained_without_replacement(self):
        plan = plan_fixture()
        plan['groups'][3]['requests'][0]['anchors'] = []
        result, log, _, _ = self.execute(plan=plan)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['planned_arm_evaluations'], 15)
        self.assertEqual(result['groups'][3]['requests'][0]['anchors'], [])
        self.assertNotIn(('start', 'g3'), log)

    def test_empty_first_group_cannot_claim_native_qualification(self):
        plan = plan_fixture()
        plan['groups'][0]['requests'][0]['anchors'] = []
        result, log, _, _ = self.execute(plan=plan)
        self.assertEqual(result['first_group_qualification']['status'], 'failed')
        self.assertEqual(result['attempted_arm_evaluations'], 0)
        self.assertEqual(log, [])

    def test_counter_ceiling_includes_first_group(self):
        log = []
        result = run_campaign(plan_fixture(40), lambda group: Backend(group, log))
        self.assertEqual(result['planned_arm_evaluations'], 720)
        self.assertEqual(result['actual_draft_block_evaluations'], 720)
        self.assertEqual(len([row for row in log if row[0] == 'probe']), 720)

    def test_extra_reconstruction_work_is_not_draft_work(self):
        result, _, _, _ = self.execute(plan=plan_fixture(3))
        self.assertEqual(result['actual_draft_block_evaluations'], 54)
        self.assertEqual(result['planned_reconstruction_target_verifications'], 54)
        self.assertEqual(result['reconstruction_target_verifications'], 54)
        self.assertEqual(result['reconstruction_target_verify_attempts'], 54)
        self.assertEqual(result['target_verifications'], 54)
        self.assertEqual(result['scored_target_verifications'], 0)
        self.assertEqual(result['unknown_reconstruction_work_receipts'], 0)
        self.assertEqual(result['unattempted_reconstruction_target_verifications'], 0)

    def test_failed_reconstruction_retains_actual_forward_before_measured_block(self):
        plan = plan_fixture()
        plan['groups'][0]['requests'][0]['anchors'][0]['reconstruction']['target_verification_count'] = 2
        result, _, _, events = self.execute({'g0': ('reconstruction_partial', 'F')}, plan=plan)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['actual_draft_block_evaluations'], 0)
        self.assertEqual(result['reconstruction_target_verifications'], 1)
        self.assertEqual(result['target_verifications'], 1)
        self.assertEqual(result['planned_reconstruction_target_verifications'], 6)
        self.assertEqual(result['unknown_reconstruction_work_receipts'], 0)
        self.assertEqual(result['unattempted_reconstruction_target_verifications'], 5)
        self.assertEqual(result['valid_pair_comparisons'], 0)
        self.assertTrue(any(kind == 'probe_raw' for kind, _ in events))

    def test_activation_failure_has_known_zero_reconstruction_work(self):
        plan = plan_fixture()
        plan['groups'][0]['requests'][0]['anchors'][0]['reconstruction']['target_verification_count'] = 2
        result, _, _, _ = self.execute({'g0': ('activation_failure', 'F')}, plan=plan)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['planned_reconstruction_target_verifications'], 6)
        self.assertEqual(result['reconstruction_target_verifications'], 0)
        self.assertEqual(result['unknown_reconstruction_work_receipts'], 0)
        self.assertEqual(result['unattempted_reconstruction_target_verifications'], 6)

    def test_unknown_reconstruction_does_not_become_known_unattempted_work(self):
        for mode in ('unreceipted', 'unknown_reconstruction', 'missing_reconstruction_counter'):
            with self.subTest(mode=mode):
                plan = plan_fixture()
                plan['groups'][0]['requests'][0]['anchors'][0]['reconstruction']['target_verification_count'] = 2
                result, _, _, _ = self.execute({'g0': (mode, 'F')}, plan=plan)
                self.assertEqual(result['status'], 'incomplete')
                self.assertEqual(result['planned_reconstruction_target_verifications'], 6)
                self.assertEqual(result['reconstruction_target_verifications'], 0)
                self.assertEqual(result['unknown_reconstruction_work_receipts'], 1)
                # Only the two wholly unattempted arms have known remaining work.
                self.assertEqual(result['unattempted_reconstruction_target_verifications'], 4)

    def test_over_budget_plan_fails_before_worker(self):
        log = []
        with self.assertRaisesRegex(ValueError, 'budget'):
            run_campaign(plan_fixture(41), lambda group: Backend(group, log))
        self.assertEqual(log, [])


if __name__ == '__main__':
    unittest.main()
