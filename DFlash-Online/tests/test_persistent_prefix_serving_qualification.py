"""Bounded serving qualification fixtures; no GPU or model execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.persistent_prefix_withholding import serving_qualification as Q
from online_feedback.storage import write_json
from test_full_k_gsm8k_evaluation import FullBackend, requests, harmful
from test_persistent_prefix_evaluation import references


class QualifiedBackend(FullBackend):
    def generate(self, prompt_ids, **options):
        record = super().generate(prompt_ids, **options)
        capture = record['metadata']['capture']
        covered = prompt_ids + record['response_ids'][:-1]
        capture.update(status='passed', label_binding_scope='native_prefill_and_committed_verification',
            target_argmax_ids=record['response_ids'], covered_token_ids=covered,
            covered_rows=len(covered))
        write_json(Path(record['metadata']['native_trace_path']), capture)
        return record


class ServingQualificationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.requests = requests(2)
        for row in self.requests:
            row['source_id'] = 'gsm8k:main:test:'+str(row['panel_ordinal'])
        self.refs = references(self.requests)

    def evaluate(self, backend, name='calls', slot='A', **kwargs):
        return Q.evaluate(backend, self.requests, self.refs, '/F', self.root/name,
                          slot=slot, uuid='GPU-'+slot, **kwargs)

    def test_exact_four_calls_per_card_and_cross_card_identity(self):
        a, b = QualifiedBackend(self.requests), QualifiedBackend(self.requests)
        first = self.evaluate(a)
        second = self.evaluate(b, 'B', slot='B')
        self.assertEqual(len(a.executions), 4)
        self.assertEqual(len(b.executions), 4)
        self.assertEqual(first['valid_generations'], 4)
        self.assertEqual(first['unattempted_generations'], 0)
        self.assertEqual(first['training_steps'], 0)
        self.assertEqual(first['extra_generation_warmups'], 0)
        self.assertEqual([r['repeat'] for r in first['rows']], [0, 0, 1, 1])
        self.assertEqual(Q.compare_serving_cards(first, second)['generations'], 8)
        changed = deepcopy(second)
        changed['signatures']['eval0']['verification_calls'] += 1
        with self.assertRaisesRegex(ValueError, 'Cross-card'):
            Q.compare_serving_cards(first, changed)

    def test_canonical_failure_is_saved_before_stop_and_accounted(self):
        def mutate(record, options):
            record['response_ids'][-1] = 151643
        with self.assertRaisesRegex(ValueError, 'qualification failed'):
            self.evaluate(QualifiedBackend(self.requests, mutate))
        result = json.loads((self.root/'calls/RESULT.json').read_text())
        self.assertEqual(result['returned_generations'], 1)
        self.assertEqual(result['invalid_generations'], 1)
        self.assertEqual(result['unknown_generations'], 0)
        self.assertEqual(result['unattempted_generations'], 3)
        raw = json.loads((self.root/'calls/qA_r0_w0.json').read_text())
        self.assertEqual(raw['response_ids'][-1], 151643)

    def test_within_card_native_count_difference_stops(self):
        def mutate(record, options):
            if options['request_id'] == 'qA_r1_w0':
                harmful(record)
        with self.assertRaisesRegex(ValueError, 'within_card_repeat'):
            self.evaluate(QualifiedBackend(self.requests, mutate))
        result = json.loads((self.root/'calls/RESULT.json').read_text())
        self.assertEqual(result['valid_generations'], 2)
        self.assertEqual(result['invalid_generations'], 1)
        self.assertEqual(result['unattempted_generations'], 1)

    def test_malformed_returned_evidence_is_invalid_not_unknown(self):
        class Malformed(QualifiedBackend):
            def generate(self, *args, **kwargs):
                result = super().generate(*args, **kwargs)
                del result['finish_reason']
                return result
        with self.assertRaisesRegex(ValueError, 'malformed_returned_record'):
            self.evaluate(Malformed(self.requests))
        result = json.loads((self.root/'calls/RESULT.json').read_text())
        self.assertEqual(result['invalid_generations'], 1)
        self.assertEqual(result['unknown_generations'], 0)
        self.assertTrue((self.root/'calls/qA_r0_w0.json').exists())

    def test_transport_failure_remains_unknown_no_retry(self):
        class Failed(QualifiedBackend):
            def generate(self, *args, **kwargs):
                raise RuntimeError('worker disappeared')
        with self.assertRaisesRegex(RuntimeError, 'disappeared'):
            self.evaluate(Failed(self.requests))
        result = json.loads((self.root/'calls/RESULT.json').read_text())
        self.assertEqual(result['attempted_generations'], 1)
        self.assertEqual(result['returned_generations'], 0)
        self.assertEqual(result['unknown_generations'], 1)
        self.assertEqual(result['unattempted_generations'], 3)

    def test_deadline_checked_before_next_attempt(self):
        backend = QualifiedBackend(self.requests)
        with self.assertRaises(TimeoutError):
            self.evaluate(backend, deadline=0)
        self.assertEqual(backend.executions, [])
        result = json.loads((self.root/'calls/RESULT.json').read_text())
        self.assertEqual(result['unattempted_generations'], 4)
        self.assertEqual(result['unknown_generations'], 0)

    def test_interrupt_saves_failure_and_closes_owned_backend(self):
        warmups = self.root/'warmups.json'
        warmups.write_text(json.dumps(dict(requests=self.requests)))
        refs = self.root/'references.json'
        refs.write_text(json.dumps(self.refs))
        backend = Mock()
        backend._call.return_value = dict(initialization=[], physical={})
        backend.check_weights.side_effect = QualifiedBackend(self.requests).check_weights
        backend.generate.side_effect = InterruptedError('qualification deadline')
        plan = dict(root=str(self.root), devices=dict(A='GPU-A'), warmups=str(warmups),
                    warmup_references=str(refs), execution_commit='c'*40,
                    target='/target', origin_export='/F', worker_python='/qualified/python')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-A'), \
             patch.object(Q, 'SGLangDFlashBackend', return_value=backend), \
             patch.object(Q, 'gpu_memory', return_value=dict(uuid='GPU-A', used_bytes=0)), \
             self.assertRaisesRegex(InterruptedError, 'deadline'):
            Q.run(plan, 'A')
        backend.close.assert_called_once()
        status = json.loads((self.root/'serving_qualification/A/STATUS.json').read_text())
        work = json.loads((self.root/'serving_qualification/A/calls/RESULT.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertFalse(status['automatic_retry'])
        self.assertEqual(work['unknown_generations'], 1)
        self.assertEqual(work['unattempted_generations'], 3)

    def test_physical_gpu_binding_fails_before_loading(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-wrong'), \
                self.assertRaisesRegex(ValueError, 'sealed CUDA'):
            Q.run(dict(devices={'A':'GPU-correct'}), 'A')


if __name__ == '__main__':
    unittest.main()
