"""CPU review of paired reductions, native validity and fixed-start outcomes."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.recent_repair_retention import analysis as A
from experiments.recent_repair_retention import evaluation as E
from experiments.recent_repair_retention.fixed_starts import outcome_row
from test_full_k_gsm8k_evaluation import FullBackend, requests


def inputs(count=8):
    panel = requests(count)
    checkpoints = {method: '/' + method for method in E.METHODS}
    canonicals = {r['request_id']: dict(prompt_ids=r['prompt_ids'],
        response_ids=[10, 11, 12, 13, 151645], finish_reason='eos',
        verified=True, binding_digest='a' * 64) for r in panel}
    return panel, checkpoints, canonicals


def boundary():
    return dict(physical_gpu='physical0',
        storage=dict(status='passed', free_bytes=1000, required_bytes=500),
        quiescence=dict(status='passed', training=[], transfers=[], foreign_gpu_processes=[]))


class ServingReviewTests(unittest.TestCase):
    def test_full_budget_and_counterbalanced_order_with_partial_final_block(self):
        for count in (1, 7, 8, 65, 256):
            slots = E.evaluation_schedule(requests(count))
            self.assertEqual(len(slots), 6 * count)
            self.assertEqual(len({(s['request_id'], s['method'], s['repeat']) for s in slots}), 6 * count)
            precedence = E.schedule_balance(slots)['pair_precedence']
            self.assertEqual(len(set(precedence.values())), 1)
            for block in range((count + 7) // 8):
                orders = [[s['method'] for s in slots if s['repeat'] == repeat
                           and s['panel_ordinal'] == block * 8] for repeat in (0, 1)]
                self.assertEqual(orders[0], orders[1][::-1])

    def test_complete_tiny_run_and_raw_audit(self):
        panel, checkpoints, canonicals = inputs()
        schedule = E.evaluation_schedule(panel)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'evaluation'
            status = E.evaluate_stage(FullBackend(panel), panel, checkpoints, root,
                canonicals=canonicals, dataset='gsm8k', event=7, schedule=schedule,
                boundary_check=boundary)
            self.assertEqual(status['completed_valid_generations'], 48)
            audit = E.audit_stage(root, panel, checkpoints, canonicals=canonicals,
                                  dataset='gsm8k', event=7, schedule=schedule)
            self.assertEqual(audit['status'], 'passed', audit['errors'])
            collapsed = A.paired_requests(audit['rows'])
            self.assertEqual(len(collapsed), 24)
            self.assertEqual(A.compare(collapsed, 'NATIVE', draws=20)['recent_calls'], 16)

    def test_failure_preserves_raw_invalid_and_unattempted_work(self):
        panel, checkpoints, canonicals = inputs()
        def mutate(record, _options):
            record['response_ids'][0] = 99
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'evaluation'
            with self.assertRaisesRegex(RuntimeError, 'invalid'):
                E.evaluate_stage(FullBackend(panel, mutate), panel, checkpoints, root,
                    canonicals=canonicals, dataset='gsm8k', event=7,
                    schedule=E.evaluation_schedule(panel), boundary_check=boundary)
            status = json.loads((root / 'STATUS.json').read_text())
            self.assertEqual(status['invalid_generations'], 1)
            self.assertEqual(status['unattempted_generations'], 47)
            self.assertTrue((root / '0000_r0_NATIVE.json').exists())
            rows = [json.loads(r) for r in (root / 'MEASUREMENTS.jsonl').read_text().splitlines()]
            with self.assertRaises(ValueError):
                A.paired_requests(rows)

    def test_no_concurrent_training_or_transfers_before_timing(self):
        panel, checkpoints, canonicals = inputs()
        for kind in ('training', 'transfers', 'foreign_gpu_processes'):
            observed = boundary()
            observed['quiescence'][kind] = ['busy']
            backend = FullBackend(panel)
            with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, 'boundary'):
                E.evaluate_stage(backend, panel, checkpoints, Path(directory) / 'evaluation',
                    canonicals=canonicals, dataset='gsm8k', event=7,
                    schedule=E.evaluation_schedule(panel), boundary_check=lambda: observed)
            self.assertEqual(backend.executions, [])

    def test_raw_clipped_acceptance_is_not_native_progress_counter(self):
        panel, checkpoints, canonicals = inputs()
        backend = FullBackend(panel)
        backend.publish(checkpoints['NATIVE'])
        with tempfile.TemporaryDirectory() as directory:
            record = backend.generate(panel[0]['prompt_ids'], request_id='0000_r0_NATIVE',
                session_id=E.SESSION, request_index=0, purpose='evaluation',
                max_new_tokens=2048, save_to=Path(directory) / 'record.json')
            block = record['metadata']['trace'][-1]
            target = [13, 151645, 99] + [0] * 13
            proposed = target[:15]
            proposed[3] = 100
            block.update(target_argmax=target, draft_tokens=proposed,
                accepted_draft_tokens=2, raw_accepted_draft_tokens_before_stop=3,
                native_accepted_draft_tokens=1, stop_clipped=True)
            record['metadata']['accepted_draft_tokens'] = 3
            record['metadata']['capture']['blocks'] = deepcopy(record['metadata']['trace'])
            row, errors = E.measure(record, panel[0], checkpoints['NATIVE'],
                                     canonicals['eval0'], E.evaluation_schedule(panel)[0])
        self.assertEqual(errors, [])
        self.assertEqual(row['accepted_draft_tokens'], 3)
        self.assertEqual(row['native_accepted_draft_tokens'], 2)


class ReductionReviewTests(unittest.TestCase):
    def paired_rows(self):
        records = []
        for rid, calls, accepted, duration in (
                ('a', (10, 20, 20), (15, 16, 16), (2., 3., 3.)),
                ('b', (100, 50, 50), (1000, 120, 120), (9., 8., 8.))):
            for method, c, a, t in zip(('RECENT', 'NATIVE', 'CONTROL'), calls, accepted, duration):
                records.append(dict(request_id=rid, method=method, verification_calls=c,
                    accepted_draft_tokens=a, request_latency_s=t))
        return records

    def test_ratio_of_sums_and_bootstrap_match_independent_request_loops(self):
        rows = self.paired_rows()
        result = A.compare(rows, 'NATIVE', draws=100)
        expected = 100 * (((15 + 1000) / (10 + 100)) / ((16 + 120) / (20 + 50)) - 1)
        self.assertAlmostEqual(result['accepted_length_relative_gain_pct'], expected, places=12)
        self.assertEqual(result['recent_calls'], 110)
        self.assertEqual(result['reference_calls'], 70)
        self.assertEqual((result['gross_calls_saved'], result['gross_calls_added']), (10, 50))
        self.assertEqual(result['time_reduction_pct'], 0)
        rng = np.random.Generator(np.random.PCG64(1707))
        samples = []
        for sample in rng.integers(0, 2, size=(100, 2)):
            a = sum((15, 1000)[i] for i in sample) / sum((10, 100)[i] for i in sample)
            b = sum((16, 120)[i] for i in sample) / sum((20, 50)[i] for i in sample)
            samples.append(100 * (a / b - 1))
        interval = result['accepted_length_interval']
        self.assertAlmostEqual(interval['lower'], float(np.quantile(samples, .025, method='linear')), places=12)
        self.assertAlmostEqual(interval['upper'], float(np.quantile(samples, .975, method='linear')), places=12)

    def test_undefined_ratio_draws_remain_explicit(self):
        rows = self.paired_rows()
        for row in rows:
            row.update(verification_calls=0, accepted_draft_tokens=0)
        result = A.compare(rows, 'NATIVE', draws=50)
        self.assertIsNone(result['accepted_length_relative_gain_pct'])
        self.assertEqual(result['accepted_length_interval']['undefined_draws'], 50)

    def test_repeats_cannot_become_extra_count_samples(self):
        raw = []
        for row in self.paired_rows():
            for repeat in (0, 1):
                raw.append(dict(row, repeat=repeat, valid_measurement=True, errors=[],
                    output_tokens=2000, committed_progress=1999, finish_reason='eos',
                    native_generate_s=row['request_latency_s'] - .1, cache_maintenance_s=.1))
        pooled = A.paired_requests(raw)
        self.assertEqual(len(pooled), 6)
        self.assertEqual(A.compare(pooled, 'NATIVE', draws=20)['recent_calls'], 110)
        changed = deepcopy(raw)
        changed[1]['verification_calls'] += 1
        with self.assertRaisesRegex(ValueError, 'differ'):
            A.paired_requests(changed)
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            A.paired_requests(raw[:-1])

    def test_earlier_break_is_not_direct_repair_and_remains_in_denominator(self):
        start = dict(request_id='a', canonical_start=2, repair_slot=3, a6=4,
                     genuine_k6_rejection=True)
        block = dict(draft_tokens=[10, 99, 12, 13, 14])
        observation = dict(accepted=1, valid_proposals=5, genuine_rejection=True)
        early = outcome_row(start, observation, block, [10, 11, 12, 13, 14], 'RECENT')
        self.assertTrue(early['earlier_break'])
        self.assertTrue(early['direct_canonical_agreement'])
        self.assertFalse(early['accepts_through'])
        self.assertFalse(early['exact_slot_relapse'])
        outcomes = [early, dict(early, method='NATIVE'), dict(early, method='CONTROL')]
        result = A.fixed_start_summary([start], outcomes, draws=20)
        self.assertEqual(result['totals']['RECENT']['repairs'], 1)
        self.assertEqual(result['totals']['RECENT']['earlier_break'], 1)
        self.assertEqual(result['contrasts']['NATIVE']['exact_slot_relapse']['denominator'], 1)
        self.assertEqual(result['totals']['RECENT']['lost_prefix_occurrences'], 3)

    def test_no_start_requests_remain_in_cluster_population(self):
        result = A.fixed_start_summary([], [], draws=20, request_ids=['seed_eos_a', 'seed_eos_b'])
        self.assertEqual(len(result['per_request']), 6)
        self.assertEqual(result['totals']['RECENT']['starts'], 0)
        gap = result['contrasts']['NATIVE']['exact_slot_relapse']
        self.assertEqual(gap['denominator'], 0)
        self.assertIsNone(gap['difference_per_eligible_start'])
        self.assertEqual(gap['interval']['undefined_draws'], 20)


if __name__ == '__main__':
    unittest.main()
