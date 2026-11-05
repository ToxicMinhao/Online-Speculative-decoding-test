"""CPU fixtures for immutable references, exact budgets and native failures."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.ok_coding.prepare import canonical_digest
from experiments.o_half_matched_lr import archive as A
from experiments.o_half_matched_lr import evaluation as E
from test_k_prefix_protection_evaluation import Backend
from test_k_prefix_protection_report import fixture as archived_fixture
from test_ok_coding_campaign import requests
from test_prefix_ranking_evaluation import harmful


def full_native(record, path):
    """Complete the native fixture's abbreviated proposals without model work."""
    for block in record['metadata']['trace']:
        accepted = block['accepted_draft_tokens']
        target = block['committed_tokens'] + [0] * (16 - len(block['committed_tokens']))
        proposed = list(target[:15])
        if accepted < 15:
            proposed[accepted] = target[accepted] + 100
        start = len(record['prompt_ids']) + block['output_start']
        block.update(draft_tokens=proposed, target_argmax=target, verify_size=16,
                     positions=list(range(start, start + 16)),
                     native_accepted_draft_tokens=block['committed_progress'] - 1,
                     raw_accepted_draft_tokens_before_stop=accepted,
                     stop_clipped=block['committed_progress'] < accepted + 1)
    capture = record['metadata']['capture']
    capture.update(prompt_ids=record['prompt_ids'], response_ids=record['response_ids'],
                   blocks=deepcopy(record['metadata']['trace']))
    record['metadata']['native_trace_path'] = str(path)
    write_json(path, capture)
    return record


class FullBackend(Backend):
    def __init__(self, inputs, mutation=None):
        super().__init__(inputs)
        self.final_mutation = mutation

    def generate(self, prompt_ids, **options):
        record = super().generate(prompt_ids, **options)
        if self.final_mutation:
            self.final_mutation(record, options)
        record['metadata']['trace'][-1]['committed_tokens'][-1] = 151645
        path = Path(options['save_to'])
        full_native(record, path.with_name(path.stem + '_NATIVE.json'))
        write_json(path, record)
        return record


def fixture_archive(inputs):
    references = {}
    for request in inputs:
        output = [10, 11, 12, 13, 151645]
        references[request['request_id']] = dict(response_ids=output, finish_reason='eos',
            source_record_path='/original/' + request['request_id'] + '.json')
    values = dict(verification_calls=2, accepted_draft_tokens=2, committed_progress=4, output_tokens=5)
    return dict(status='passed', requests=inputs, canonicals=references,
                canonical_path='/original/EVALUATION_CANONICAL.json',
                canonical_identity={'bytes': 123, 'sha256': 'fixed-fixture-identity'},
                counts={str(seed): {method: {r['request_id']: dict(values) for r in inputs}
                                   for method in A.ARCHIVED_METHODS} for seed in E.SEEDS},
                checkpoints={str(seed): {'F': '/F', 'K': f'/K{seed}', 'K_HALF': f'/KH{seed}'}
                             for seed in E.SEEDS})


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = requests()
        self.archive = fixture_archive(self.requests)

    def run_final(self, mutation=None):
        backend = FullBackend(self.requests, mutation)
        checkpoints = {'O': '/O17', 'O_HALF': '/OH17'}
        status = E.evaluate_final(backend, self.requests, checkpoints, self.root / 'final',
                                  seed=17, archive=self.archive)
        return backend, checkpoints, status

    def test_exact1072_schedule_and_request_major_two_repeats(self):
        bridge = [slot for seed in E.SEEDS for slot in E.bridge_schedule(self.requests, seed)]
        final = [slot for seed in E.SEEDS for slot in E.final_schedule(self.requests, seed)]
        self.assertEqual((len(bridge), len(final)), (48, 1024))
        self.assertEqual([(row['seed'], row['method']) for row in bridge[::8]],
                         [(seed, method) for seed in E.SEEDS for method in A.ARCHIVED_METHODS])
        self.assertEqual([(row['panel_ordinal'], row['repeat']) for row in final[:4]],
                         [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual([row['method'] for row in final[::256]], ['O', 'O_HALF', 'O', 'O_HALF'])

    def test_bridge_all48_count_bound_and_no_canonical_mutation(self):
        before = deepcopy(self.archive)
        for seed in E.SEEDS:
            backend = FullBackend(self.requests)
            folder = self.root / f'bridge{seed}'
            checkpoints = self.archive['checkpoints'][str(seed)]
            status = E.evaluate_bridge(backend, self.requests, checkpoints, folder,
                                       seed=seed, archive=self.archive)
            self.assertEqual(status['completed_valid_generations'], 24)
            self.assertEqual(backend.switches, list(checkpoints.values()))
            audited = E.audit_stage(folder, self.requests, checkpoints,
                                    seed=seed, phase='bridge', archive=self.archive)
            self.assertEqual(audited['errors'], [])
        self.assertEqual(self.archive, before)

    def test_final_512_generations_included_gates_and_two_publications(self):
        backend, checkpoints, status = self.run_final()
        self.assertEqual(status['completed_valid_generations'], 512)
        self.assertEqual(status['unattempted_generations'], 0)
        self.assertEqual(backend.switches, ['/O17', '/OH17'])
        for method in E.NEW_METHODS:
            gate = json.loads((self.root / 'final' / f'QUALIFICATION_{method}.json').read_text())
            self.assertEqual(gate['generations'], 16)
        audited = E.audit_stage(self.root / 'final', self.requests, checkpoints,
                                seed=17, phase='final', archive=self.archive)
        self.assertEqual(audited['errors'], [])
        self.assertEqual(len(audited['rows']), 512)

    def test_valid_harmful_final_counts_complete_without_historical_count_gate(self):
        def regression(record, options):
            if record['checkpoint_id'] == '/OH17':
                harmful(record)
        _, checkpoints, status = self.run_final(regression)
        self.assertEqual(status['completed_valid_generations'], 512)
        audited = E.audit_stage(self.root / 'final', self.requests, checkpoints,
                                seed=17, phase='final', archive=self.archive)
        self.assertEqual(audited['errors'], [])
        self.assertEqual({row['verification_calls'] for row in audited['rows']
                          if row['method'] == 'O_HALF'}, {3})

    def test_full_token_and_termination_failures_preserve_raw(self):
        for field, value in [('response_ids', [20, 11, 12, 13, 151645]), ('finish_reason', 'length')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                backend = FullBackend(self.requests, lambda record, _: record.__setitem__(field, value))
                folder = Path(directory) / 'failure'
                with self.assertRaisesRegex(RuntimeError, 'correctness'):
                    E.evaluate_final(backend, self.requests, {'O': '/O', 'O_HALF': '/OH'}, folder,
                                     seed=17, archive=self.archive)
                self.assertTrue((folder / '0000_r0_O.json').exists())
                status = json.loads((folder / 'STATUS.json').read_text())
                self.assertEqual((status['returned_generations'], status['invalid_generations'],
                                  status['unattempted_generations']), (1, 1, 511))

    def test_bridge_count_mismatch_does_not_replace_archive(self):
        backend = FullBackend(self.requests, lambda record, _: harmful(record))
        with self.assertRaisesRegex(RuntimeError, 'archived_count'):
            E.evaluate_bridge(backend, self.requests, self.archive['checkpoints']['17'],
                               self.root / 'bridge', seed=17, archive=self.archive)
        row = json.loads((self.root / 'bridge/MEASUREMENTS.jsonl').read_text())
        self.assertFalse(row['valid_measurement'])
        self.assertIn('archived_count_verification_calls', row['errors'])

    def test_repeat_count_mismatch_and_unknown_attempt_accounting(self):
        def changed(record, options):
            if '_r1_' in options['request_id']:
                harmful(record)
        with self.assertRaisesRegex(RuntimeError, 'repeat_count_mismatch'):
            self.run_final(changed)
        status = json.loads((self.root / 'final/STATUS.json').read_text())
        self.assertEqual((status['attempted_generations'], status['invalid_generations'],
                          status['unknown_generation_attempts'], status['unattempted_generations']), (2, 1, 0, 510))
        backend = FullBackend(self.requests)
        backend.generate = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('native transport stopped'))
        with self.assertRaisesRegex(RuntimeError, 'transport'):
            E.evaluate_final(backend, self.requests, {'O': '/O', 'O_HALF': '/OH'}, self.root / 'unknown',
                             seed=17, archive=self.archive)
        status = json.loads((self.root / 'unknown/STATUS.json').read_text())
        self.assertEqual((status['returned_generations'], status['unknown_generation_attempts'],
                          status['unattempted_generations']), (0, 1, 511))

    def test_report_revalidates_raw_and_does_not_trust_saved_valid_flag(self):
        _, checkpoints, _ = self.run_final()
        path = self.root / 'final/0000_r0_O.json'
        raw = json.loads(path.read_text())
        raw['metadata']['trace'][0]['draft_tokens'][0] += 7
        write_json(path, raw)
        result = E.audit_stage(self.root / 'final', self.requests, checkpoints,
                               seed=17, phase='final', archive=self.archive)
        self.assertEqual(result['status'], 'invalid_or_incomplete')
        self.assertTrue(any('native_capture_binding' in value for value in result['errors']))


class ArchiveTests(unittest.TestCase):
    def test_complete_original_repeats_and_separate_native_source_then_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = archived_fixture(root)
            binding = canonical_digest(plan['evaluation_requests'])
            plan['execution_source']['commit'] = A.PR21_COMMIT
            plan['training_binding']['evaluation_binding'] = binding
            write_json(root / 'sealed/PLAN.json', plan)
            for path in root.glob('campaign_001/seed_*/event_004/evaluation/*_r?_*.json'):
                if path.name.endswith('_VALIDATION.json'):
                    continue
                raw = json.loads(path.read_text())
                full_native(raw, path.with_name(path.stem + '_NATIVE.json'))
                write_json(path, raw)
            totals = {str(seed): {method: 256 for method in A.ARCHIVED_METHODS} for seed in E.SEEDS}
            with patch.object(A, 'PANEL_BINDING', binding), patch.object(A, 'EXPECTED_CALL_TOTALS', totals):
                result = A.audit_archive(root)
                self.assertEqual(result['raw_generations'], 3072)
                self.assertEqual(result['native_blocks'], 6144)
                self.assertEqual(result['call_totals'], totals)
                path = root / 'campaign_001/seed_29/event_004/evaluation/0127_r2_K_HALF_NATIVE.json'
                native = json.loads(path.read_text())
                native['response_ids'][0] = 999
                write_json(path, native)
                receipt = root / 'FAILED_REUSE_AUDIT.json'
                with self.assertRaisesRegex(ValueError, 'native capture differs'):
                    A.audit_archive(root, receipt)
                failed = json.loads(receipt.read_text())
                self.assertEqual(failed['status'], 'failed')
                self.assertTrue(failed['files'])


if __name__ == '__main__':
    unittest.main()
