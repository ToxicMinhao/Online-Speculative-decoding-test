"""CPU-only schedule, retained-reference, reduction and decision checks."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from experiments.continuation_repair_pilot import analysis, serving
from experiments.ok_coding.storage import identity
from experiments.repair_conditional_kl_pilot.test_evaluation import inputs, Backend, Ledger


def run_fixture(root, backend):
    panel, warmups, references, warmup_references = inputs()
    ledger = Ledger()
    def activate(method, checkpoint, entry):
        backend.checkpoint = checkpoint
        return backend, {}
    result = serving.execute(dict(execution_commit='fixture', devices={'A': 'GPUa'},
        canonical_provenance={'source': 'PR56'}), root/'evaluation', panel, warmups,
        references, warmup_references, dict(BASE='base', TAIL='tail'), activate=activate,
        boundary=lambda _: dict(quiet=True), ledger=ledger)
    rows = [json.loads(line) for line in (root/'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
    return result, rows, ledger


def tiny_measurements():
    rows = []
    for i in range(2):
        for method in analysis.METHODS:
            calls = (10 + i) if method == 'BASE' else (9 + 2*i)
            for repeat in range(3):
                rows.append(dict(phase='scored', method=method, request_id=f'q{i}', source_id=f's{i}',
                    repeat=repeat, valid_measurement=True, errors=[], checkpoint_id=method, physical_gpu_uuid='GPUa',
                    output_tokens=101, verification_calls=calls, accepted_draft_tokens=100-calls,
                    native_accepted_draft_tokens=100-calls, committed_progress=100, finish_reason='eos',
                    request_latency_s=(1+i) * (1 if method == 'BASE' else .9) + repeat*.1,
                    native_generate_s=(1+i) * (1 if method == 'BASE' else .9), cache_maintenance_s=.01))
    return rows


def decision_report(point=1., lower=.1, upper=2., calls=.2):
    return {'contrasts': {'TAIL_vs_BASE': {
        'request_latency_s_reduction_pct': dict(value=point, ci95=[lower, upper]),
        'verification_call_reduction_pct': dict(value=calls, ci95=[-.1, .3])}}}


class ServingTests(unittest.TestCase):
    def test_explicit_two_arm_budget_and_balance(self):
        panel, *_ = inputs()
        slots = serving.schedule(panel)
        self.assertEqual(len(slots), 12)
        for index, slot in enumerate(slots):
            repeat, group = divmod(index//2, 2)
            order = analysis.METHODS if (repeat+group)%2 == 0 else analysis.METHODS[::-1]
            self.assertEqual(slot['method'], order[index%2])
            self.assertEqual(len(slot['request_ids']), 64)
        self.assertEqual({s['method'] for s in slots}, {'BASE', 'TAIL'})

    def test_full_fake_native_path_keeps792_and_zero_training(self):
        with TemporaryDirectory() as folder:
            result, rows, ledger = run_fixture(Path(folder), Backend())
            self.assertEqual(result['completed_valid'], 792)
            self.assertEqual(ledger.counts, dict(scored_generation=768, warmup_generation=24))
            reduced = analysis.summarize(rows)
            self.assertEqual(reduced['generation_accounting']['warmups'], 24)
            self.assertEqual(reduced['bootstrap']['seed'], 20261126)
            self.assertEqual(reduced['contrasts']['TAIL_vs_BASE']['request_latency_s_reduction_pct']['value'], 0.)
            self.assertEqual(reduced['contrasts']['TAIL_vs_BASE']['verification_call_reduction_pct']['status'], 'undefined')

    def test_invalid_and_unknown_failures_remain_visible(self):
        for backend, expected in [(Backend(malformed=True), (1,0)), (Backend(transport_failure=True),(0,1))]:
            with self.subTest(expected=expected), TemporaryDirectory() as folder:
                root = Path(folder)
                with self.assertRaises((ValueError, RuntimeError)):
                    run_fixture(root, backend)
                result = json.loads((root/'evaluation/STATUS.json').read_text())
                self.assertEqual((result['invalid'],result['unknown']), expected)
                self.assertEqual(result['unattempted'], 791)

    def test_offline_reconciliation_rejects_warmup_duplicate_and_group_swap(self):
        with TemporaryDirectory() as folder:
            _, rows, _ = run_fixture(Path(folder), Backend())
            changed = deepcopy(rows)
            warmups = [r for r in changed if r['activation'] == 0 and r['phase'] == 'warmup']
            warmups[1]['request_id'] = warmups[0]['request_id']
            with self.assertRaisesRegex(ValueError, 'lost or duplicated'):
                analysis.complete_generation_accounting(changed)
            changed = deepcopy(rows)
            next(r for r in changed if r['activation'] == 4 and r['phase'] == 'scored')['request_id'] = 'foreign'
            with self.assertRaisesRegex(ValueError, 'group changed'):
                analysis.complete_generation_accounting(changed)

    def test_pooled_means_and_fixed_paired_bootstrap(self):
        report = analysis.summarize(tiny_measurements(), expected_requests=2)
        expected_base = np.asarray([1.1,2.1]); expected_tail = np.asarray([1.,1.9])
        draws = np.random.Generator(np.random.PCG64(20261126)).integers(0,2,size=(10000,2))
        samples = 100*(1-expected_tail[draws].sum(1)/expected_base[draws].sum(1))
        result = report['contrasts']['TAIL_vs_BASE']['request_latency_s_reduction_pct']
        self.assertAlmostEqual(result['value'],100*(1-2.9/3.2))
        np.testing.assert_allclose(result['ci95'],np.percentile(samples,[2.5,97.5],method='linear'))
        self.assertEqual(report['totals']['BASE']['verification_calls'],21)
        self.assertEqual(report['totals']['TAIL']['verification_calls'],20)
        self.assertEqual(report['contrasts']['TAIL_vs_BASE']['wins'],1)
        self.assertEqual(report['contrasts']['TAIL_vs_BASE']['ties'],1)

    def test_measurement_rejection_not_hidden_by_report(self):
        rows = tiny_measurements()
        for row in rows[:3]:
            row['accepted_draft_tokens']=102
        with self.assertRaisesRegex(ValueError,'count identities'):
            analysis.summarize(rows,expected_requests=2)

    def test_completed_endpoint_receipt_uses_authenticated_checkpoint_identity(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            statuses = {}
            for method in ('BASE', 'TAIL'):
                export = root/method/'export'; export.mkdir(parents=True)
                (export/'model.safetensors').write_bytes(method.encode())
                receipt = dict(status='completed', global_step=256, presentations=512,
                    completed_event=2, checkpoint=str(export),
                    checkpoint_identity=identity(export/'model.safetensors'))
                path = root/('BASE_STATUS.json' if method == 'BASE' else 'arms/TAIL/STATUS.json')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(receipt)); statuses[method]=path
            plan=dict(root=str(root),base_status=str(statuses['BASE']),
                      base_status_identity=identity(statuses['BASE']))
            self.assertEqual(set(serving.endpoint_receipts(plan)), {'BASE','TAIL'})
            tail=json.loads(statuses['TAIL'].read_text());tail['global_step']=255
            statuses['TAIL'].write_text(json.dumps(tail))
            with self.assertRaisesRegex(ValueError,'complete event-two'):
                serving.endpoint_receipts(plan)

    def test_pool_activity_counts_original_responses_once(self):
        rows=[dict(record_id='private',pass_index=0,optimizer_step_after=128,metrics={'ignored':100},
                   response_observation=dict(record_id='private',continuation_repair={
                       'continuation_changed_high_positions':2,'continuation_total_correction':0.}))]
        rows[0]['pass']=0
        state=dict(training={'records':rows},activity={'continuation_changed_high_positions':2})
        table=analysis.pool_activity_table(state)
        self.assertEqual(table[0]['continuation_changed_high_positions'],2)
        self.assertEqual(table[0]['continuation_total_correction'],0.)
        self.assertNotIn('private',json.dumps(table))
        state['activity']['continuation_changed_high_positions']=4
        with self.assertRaisesRegex(ValueError,'double-counts'):
            analysis.pool_activity_table(state)

    def test_phase_gate_cannot_reuse_foreign_or_partial_qualification(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'qualification.json'
            plan=dict(qualification_result=str(path),binding='expected')
            good=dict(status='passed',binding='expected',base_disabled_exact=True,retained_BASE_reusable=True)
            path.write_text(json.dumps(good))
            self.assertEqual(serving.require_qualification(plan),good)
            for key,value in [('binding','foreign'),('base_disabled_exact',False),('retained_BASE_reusable',False)]:
                bad=dict(good,**{key:value});path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'exact disabled neutrality'):
                    serving.require_qualification(plan)

    def test_decision_thresholds_and_identical_endpoint(self):
        choose=lambda r, **kw: analysis.freeze_decision(r,distinct_weights=kw.get('distinct',True),
            active=kw.get('active',True),correctness_complete=kw.get('complete',True))['label']
        self.assertEqual(choose(decision_report()),'SUPPORTED_FOR_CONFIRMATION')
        self.assertEqual(choose(decision_report(),distinct=False),'NO_DEMONSTRATED_BENEFIT')
        self.assertEqual(choose(decision_report(lower=-.49)),'TENTATIVE')
        self.assertEqual(choose(decision_report(lower=-.5)),'NO_DEMONSTRATED_BENEFIT')
        self.assertEqual(choose(decision_report(lower=0)),'TENTATIVE')
        self.assertEqual(choose(decision_report(lower=0),active=False),'NO_DEMONSTRATED_BENEFIT')
        self.assertEqual(choose(decision_report(calls=-.01)),'NO_DEMONSTRATED_BENEFIT')
        self.assertEqual(choose(decision_report(),complete=False),'BLOCKED')


if __name__ == '__main__': unittest.main()
