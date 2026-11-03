"""Fixed two-method endpoint order and retained K58 count gate."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kplus_full_gsm8k import evaluation as E
from test_full_k_gsm8k_evaluation import requests, FullBackend


class EndpointTests(unittest.TestCase):
    def test_full_order_and_budget(self):
        slots = E.final_schedule(requests(1024))
        balance = E.schedule_balance(slots)
        self.assertEqual(len(slots), 6144)
        self.assertEqual(balance['method_positions'], {'K': [24, 24], 'KPLUS': [24, 24]})
        self.assertEqual(set(balance['pairwise_precedence'].values()), {24})
        self.assertEqual([s['panel_ordinal'] for s in slots[:64]], list(range(64)))
        self.assertEqual(E.method_order(0, 0), ('K', 'KPLUS'))
        self.assertEqual(E.method_order(0, 1), ('KPLUS', 'K'))
        self.assertEqual(E.method_order(1, 0), ('KPLUS', 'K'))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = requests(8)
        self.canonicals = {r['request_id']: dict(response_ids=[10, 11, 12, 13, 151645],
            finish_reason='eos', source_record_path='/original/' + r['request_id']) for r in self.inputs}
        self.baseline = {r['request_id']: dict(verification_calls=2, accepted_draft_tokens=2,
            committed_progress=4, output_tokens=5) for r in self.inputs}
        self.checkpoints = dict(K='/K58', KPLUS='/KP58')

    def tiny(self):
        return patch.multiple(E, PANEL_SIZE=8, REQUEST_BLOCK=2,
                              HISTORICAL_K_TOTALS=dict(verification_calls=16, output_tokens=40))

    def test_raw_endpoint_all_repeats_and_exact_old_count_gate(self):
        backend = FullBackend(self.inputs)
        checks = []
        with self.tiny():
            result = E.evaluate_final(backend, self.inputs, self.checkpoints, self.root / 'final',
                canonicals=self.canonicals, baseline_counts=self.baseline,
                boundary_check=lambda: checks.append('outside request') or {'status': 'passed'})
            audit = E.audit_stage(self.root / 'final', self.inputs, self.checkpoints,
                                 canonicals=self.canonicals, baseline_counts=self.baseline)
        self.assertEqual(result['completed_valid_generations'], 48)
        self.assertEqual(audit['errors'], [])
        self.assertEqual(len(checks), 24)

    def test_baseline_discrepancy_stops_without_replacing_request(self):
        bad = deepcopy(self.baseline)
        bad['eval0']['verification_calls'] = 3
        bad['eval1']['verification_calls'] = 1
        with self.tiny(), self.assertRaisesRegex(RuntimeError, 'historical_K58_verification_calls'):
            E.evaluate_final(FullBackend(self.inputs), self.inputs, self.checkpoints,
                             self.root / 'failed', canonicals=self.canonicals, baseline_counts=bad)
        status = json.loads((self.root / 'failed/STATUS.json').read_text())
        self.assertEqual(status['attempted_generations'], 1)
        self.assertEqual(status['invalid_generations'], 1)
        self.assertEqual(status['unattempted_generations'], 47)


if __name__ == '__main__':
    unittest.main()
