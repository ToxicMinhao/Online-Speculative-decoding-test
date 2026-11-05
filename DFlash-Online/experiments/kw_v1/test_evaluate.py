"""CPU fake-native tests: fixed serving budget, exact evidence and failure retention."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.fresh_confirmation.test_evaluate import Backend
from test_osd_one_event import request as original_request
from test_prefix_ranking_evaluation import harmful
from .evaluate import evaluate
from .evaluation_protocol import ARMS, CANONICAL_SOURCES, method_order
from .report import summarize


def requests(panel='development', count=3):
    values = [original_request(index) for index in range(count)]
    for row in values:
        if panel == 'final':
            row.pop('response_ids')
            row.pop('finish_reason')
        else:
            row['canonical_source_method'] = CANONICAL_SOURCES[panel]
    return values


def panels():
    values = []
    for offset, name in enumerate(('development', 'additional', 'fresh', 'final')):
        rows = requests(name, 64)
        for row in rows:
            row['request_id'] = name + '_' + row['request_id']
            row['prompt_ids'] = [1, offset * 100 + row['panel_ordinal']]
        value = dict(name=name, requests=rows,
                     canonical_mode='fresh' if name == 'final' else 'retained')
        if name == 'final':
            value['selection_seed'] = 20261030
        values.append(value)
    return values


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoints = {arm: '/' + arm for arm in ARMS}

    def run_panel(self, *, panel='development', event=10, count=3, mutation=None,
                  name='evaluation', gate=0, rows=None):
        rows = requests(panel, count) if rows is None else rows
        backend = Backend(rows, mutation)
        for native, provided in zip(backend.requests, rows):
            native['prompt_ids'] = list(provided['prompt_ids'])
        status = evaluate(backend, rows, self.checkpoints, self.root / name,
            event_number=event, panel_name=panel,
            canonical_mode='fresh' if panel == 'final' else 'retained', qualification_requests=gate)
        return backend, status

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def test_three_arms_nine_slots_each_and_pinned_schedule(self):
        backend, status = self.run_panel()
        self.assertEqual(status['returned_generations'], 27)
        self.assertEqual(status['completed_valid_generations'], 27)
        self.assertEqual(backend.weight_checks, ['/O', '/K', '/KW'])
        actual = [row[0][1:] for row in backend.executions]
        expected = [arm for ordinal in range(3) for repeat in range(3) for arm in method_order(ordinal, repeat)]
        self.assertEqual(actual, expected)
        self.assertEqual([row['method'] for row in self.read('evaluation/PROTOCOL.json')['schedule']], expected)
        self.assertEqual(self.read('evaluation/report/SUMMARY.json')['primary_comparison'], 'KW_vs_K')
        self.assertFalse((self.root / 'evaluation/QUALIFICATION.json').exists())

    def test_included_eight_request_gate_is_72_with_no_added_generation(self):
        _, status = self.run_panel(count=8, gate=8)
        self.assertEqual(status['returned_generations'], 72)
        self.assertEqual(self.read('evaluation/QUALIFICATION.json')['generations'], 72)
        self.assertEqual(self.read('evaluation/QUALIFICATION.json')['arms_with_native_equality'], ['K', 'KW', 'O'])

    def test_final_both_K_and_KW_first_defer_to_first_O12(self):
        _, status = self.run_panel(panel='final', event=12)
        self.assertEqual(status['returned_generations'], 27)
        events = [json.loads(line) for line in (self.root / 'evaluation/VALIDATION_EVENTS.jsonl').read_text().splitlines()]
        for method in ('K', 'KW'):
            self.assertEqual([row['status'] for row in events if row['label'] == '0001_r0_' + method],
                             ['canonical_pending', 'valid'])
        canonical = self.read('evaluation/CANONICAL.json')['q1']
        self.assertEqual((canonical['source_method'], canonical['source_repeat']), ('O', 0))
        self.assertTrue(canonical['source_record_path'].endswith('0001_r0_O.json'))

    def test_all_three_retained_canonical_sources(self):
        for panel, source in CANONICAL_SOURCES.items():
            self.run_panel(panel=panel, event=12, name=panel)
            self.assertEqual(self.read(panel + '/CANONICAL.json')['q0']['source_method'], source)

    def test_historical_counts_do_not_gate_new_checkpoints_and_harmful_KW_continues(self):
        rows = requests()
        for row in rows:
            row['reference_calls'] = {'O': 999, 'K': 999, 'KW': 999, 'F': 999}
        def changed(record, options):
            if record['checkpoint_id'] == '/KW':
                harmful(record)
        _, status = self.run_panel(rows=rows, mutation=changed)
        self.assertEqual(status['status'], 'completed')
        summary = self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(summary['comparisons']['KW_vs_K']['call_reduction_pct'], -50)
        self.assertEqual(summary['KW']['verification_calls'], 9)

    def test_retained_token_failure_is_raw_before_rejection(self):
        def changed(record, options):
            record['response_ids'][1] = 777
            record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.run_panel(mutation=changed)
        self.assertEqual(self.read('evaluation/0000_r0_O.json')['response_ids'][1], 777)
        self.assertEqual(self.read('evaluation/STATUS.json')['invalid_generations'], 1)
        self.assertEqual(self.read('evaluation/report/SUMMARY.json')['valid_requests'], 0)

    def test_deferred_invalidity_and_unknown_attempt_accounting(self):
        def changed(record, options):
            if options['request_id'] == '0001_r0_KW':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.run_panel(panel='final', event=12, mutation=changed)
        status = self.read('evaluation/STATUS.json')
        self.assertEqual((status['returned_generations'], status['invalid_generations']), (12, 1))
        def lost(record, options):
            if options['request_id'] == '0001_r0_O':
                raise RuntimeError('reply lost')
        with self.assertRaisesRegex(RuntimeError, 'reply lost'):
            self.run_panel(panel='final', event=12, mutation=lost, name='unknown')
        status = self.read('unknown/STATUS.json')
        self.assertEqual((status['returned_generations'], status['pending_canonical_generations'],
                          status['unknown_generation_attempts'], status['unattempted_generations']), (11, 2, 1, 15))

    def test_repeat_count_disagreement_invalidates_both_same_arm_rows(self):
        def changed(record, options):
            if options['request_id'] == '0000_r1_KW':
                harmful(record)
        with self.assertRaisesRegex(RuntimeError, 'repeat_count_mismatch'):
            self.run_panel(mutation=changed)
        status = self.read('evaluation/STATUS.json')
        self.assertEqual((status['returned_generations'], status['invalid_generations']), (5, 2))

    def test_nonfinite_and_malformed_records_preserved_without_report_crash(self):
        for index, field in enumerate(('nan', 'missing', 'boolean_count')):
            def changed(record, options):
                if field == 'nan':
                    record['metadata']['request_latency_s'] = float('nan')
                elif field == 'missing':
                    del record['metadata']['trace']
                else:
                    record['metadata']['verification_calls'] = True
            name = str(index)
            with self.assertRaises(RuntimeError):
                self.run_panel(mutation=changed, name=name)
            self.assertTrue((self.root / name / '0000_r0_O.json').is_file())
            self.assertTrue((self.root / name / 'report/SUMMARY.json').is_file())
            self.assertFalse((self.root / name / 'REPORT_FAILURE.json').exists())
            self.assertEqual(self.read(name + '/STATUS.json')['invalid_generations'], 1)

    def test_report_reaudits_raw_counts_scalar_and_canonical_not_just_valid_flag(self):
        self.run_panel()
        rows = [json.loads(line) for line in (self.root / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
        canonical = self.read('evaluation/CANONICAL.json')
        rows[0]['cache_maintenance_s'] = .5
        canonical['q1']['response_ids'][1] = 777
        summary, per_request, _, _ = summarize(rows, requests(), checkpoints=self.checkpoints,
            canonicals=canonical, canonical_mode='retained')
        self.assertEqual(summary['valid_requests'], 1)
        self.assertIn('retained_canonical_binding', per_request[1]['errors'])

    def test_import_does_not_initialize_model_or_GPU(self):
        code = "import sys; import experiments.kw_v1.evaluate; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules; assert 'online_feedback.training' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
