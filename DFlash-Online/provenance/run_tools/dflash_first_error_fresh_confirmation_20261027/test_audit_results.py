"""CPU tests for audit arithmetic, native evidence corruption and path handling."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import audit_results as audit


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace_record, cls.validate_generation = audit.load_helpers()
        old = audit.HERE.parent / 'dflash_first_error_execution_20261026/raw/campaign_001'
        cls.source_record = audit.read(old / 'evaluation/0000_r0_O.json')
        cls.source_request = audit.read(old / 'MANIFEST.json')['eval_requests'][0]
        cls.source_row = next(r for r in audit.lines(old / 'evaluation/MEASUREMENTS.jsonl')
                              if r['panel_ordinal'] == r['repeat'] == 0 and r['method'] == 'O')

    def fixture(self):
        record, request, row = map(copy.deepcopy, (self.source_record, self.source_request, self.source_row))
        record['session_id'] = 'fresh_confirmation_evaluation'
        row['label'] = '0000_r0_O'
        canonical = {key: copy.deepcopy(record[key]) for key in ('response_ids', 'finish_reason')}
        expected = {**request, **canonical}
        return record, request, row, canonical, expected

    def test_retained_real_native_trace_reconstructs_on_cpu(self):
        record, request, row, canonical, expected = self.fixture()
        issues = []
        result = type(self).trace_record(record, expected, record['checkpoint_id'], issues, row['label'])
        audit.trace_extras(record, request, row, canonical, type(self).validate_generation, issues)
        self.assertEqual(issues, [])
        self.assertEqual(result['verification_calls'], len(record['metadata']['trace']))
        self.assertEqual(result['committed_progress'], len(record['response_ids']) - 1)

    def test_corrupt_accepted_count_is_detected(self):
        record, _, row, _, expected = self.fixture()
        record['metadata']['trace'][0]['accepted_draft_tokens'] += 1
        issues = []
        type(self).trace_record(record, expected, record['checkpoint_id'], issues, row['label'])
        self.assertTrue(any('accepted' in issue for issue in issues))

    def test_corrupt_canonical_output_is_detected(self):
        record, _, row, _, expected = self.fixture()
        expected['response_ids'][1] += 1
        issues = []
        type(self).trace_record(record, expected, record['checkpoint_id'], issues, row['label'])
        self.assertTrue(any('canonical response_ids' in issue for issue in issues))

    def test_wrong_session_identity_is_detected(self):
        record, request, row, canonical, _ = self.fixture()
        record['session_id'] = 'old_experiment'
        issues = []
        audit.trace_extras(record, request, row, canonical, type(self).validate_generation, issues)
        self.assertTrue(any('native identity' in issue for issue in issues))

    def test_native_validator_rejects_continuation_after_eos(self):
        with self.assertRaisesRegex(RuntimeError, 'after first EOS'):
            type(self).validate_generation([1], [1, 99, 2], [], max_new_tokens=2048, stop_tokens=[99])

    def test_alternation_is_exact_and_balanced(self):
        requests = [{'request_id': str(i), 'panel_ordinal': i} for i in range(64)]
        slots = audit.expected_slots(requests)
        self.assertEqual(len(slots), 384)
        self.assertEqual(slots[:6], [('0', 0, 'O'), ('0', 0, 'K'), ('0', 1, 'K'),
                                    ('0', 1, 'O'), ('0', 2, 'O'), ('0', 2, 'K')])
        self.assertEqual(slots[6:8], [('1', 0, 'K'), ('1', 0, 'O')])
        self.assertEqual(sum(arm == 'O' for _, _, arm in slots[::2]), 96)

    def test_pooling_uses_one_count_trajectory_and_three_time_means(self):
        requests = [{'request_id': str(i), 'panel_ordinal': i} for i in range(2)]
        rows = []
        for i in range(2):
            for repeat, factor in enumerate((.5, 1., 1.5)):
                for arm in audit.ARMS:
                    calls = {'O': (2, 8), 'K': (1, 9)}[arm][i]
                    time = {'O': (1, 9), 'K': (1, 3)}[arm][i]
                    rows.append(dict(request_id=str(i), method=arm, repeat=repeat,
                        verification_calls=calls, accepted_draft_tokens=20-calls,
                        committed_progress=20, output_tokens=21, request_latency_s=time*factor))
        totals, comparison, _ = audit.totals_and_comparison(rows, requests)
        self.assertEqual(totals['O']['verification_calls'], 10)
        self.assertEqual(totals['K']['verification_calls'], 10)
        self.assertEqual(totals['O']['summed_mean_request_latency_s'], 10)
        self.assertEqual(totals['K']['summed_mean_request_latency_s'], 4)
        self.assertEqual(totals['K']['all_repeats_request_latency_s'], 12)
        self.assertEqual(comparison['call_request_directions'], {'improved': 1, 'regressed': 1, 'tied': 0})
        self.assertEqual(comparison['call_reduction_pct'], 0)
        self.assertEqual(comparison['request_time_reduction_pct'], 60)

    def test_missing_repeated_slot_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            audit.totals_and_comparison([], [{'request_id': 'one'}])

    def test_output_lengths_and_caps_count_canonical_requests_once(self):
        result = audit.output_summary({
            'one': dict(response_ids=[3, 99], finish_reason='eos'),
            'two': dict(response_ids=[3] * 2048, finish_reason='length')})
        self.assertEqual(result, dict(requests=2, minimum_tokens=2, maximum_tokens=2048,
            mean_tokens=1025, total_tokens=2050, cap_hit_requests=1,
            finish_reason_counts={'eos': 1, 'length': 1}))

    def test_remote_path_maps_by_basename(self):
        self.assertEqual(audit.local_record('/local/raw/evaluation', '/remote/elsewhere/0001_r0_K.json'),
                         Path('/local/raw/evaluation/0001_r0_K.json'))

    def test_missing_evidence_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit.run_audit(directory)
        self.assertEqual(result['status'], 'issues_found')
        self.assertTrue(result['issues'])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
