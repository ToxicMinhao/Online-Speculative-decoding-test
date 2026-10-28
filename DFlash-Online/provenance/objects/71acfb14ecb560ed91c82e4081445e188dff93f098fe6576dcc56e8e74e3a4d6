"""CPU-native stubs for genuine KR/KRC panels, retained controls and accounting."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_osd_one_event import Backend as OldBackend, request as old_request
from test_prefix_ranking_evaluation import harmful
from .evaluate import evaluate
from .evaluation_protocol import method_order, validate_inputs, prepare_requests, validate_plan
from .report import summarize


def panel(name='development', count=64):
    source = 'F' if name == 'development' else 'K1'
    offset = 0 if source == 'F' else 1000
    requests = [dict(old_request(i), request_id=f'{name}_{i}',
                     prompt_ids=[1, 2, offset+i], canonical_source_method=source)
                for i in range(count)]
    return dict(name=name, requests=requests,
                retained_krc_counts={row['request_id']: 2 for row in requests})


class Backend(OldBackend):
    def generate(self, prompt_ids, **kwargs):
        mutation = self.mutation
        self.mutation = None
        try:
            record = super().generate(prompt_ids, **kwargs)
        finally:
            self.mutation = mutation
        record.update(record_id=kwargs['request_id'], session_id=kwargs['session_id'],
                      request_index=kwargs['request_index'])
        if mutation is not None:
            mutation(record, kwargs)
        return record


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.panel = panel(count=3)
        self.requests = self.panel['requests']; self.counts = self.panel['retained_krc_counts']
        self.checkpoints = dict(KR='/KR', KRC='/KRC')

    def evaluate(self, mutate=None, name='eval', gate=1):
        return evaluate(Backend(self.requests, mutate), self.requests, self.checkpoints,
                        self.root/name, retained_krc_counts=self.counts, qualification_requests=gate)

    def read(self, path):
        return json.loads((self.root/path).read_text())

    def rows(self):
        return [json.loads(line) for line in (self.root/'eval/MEASUREMENTS.jsonl').read_text().splitlines()]

    def test_exact_384_rotation_and_included48_gate(self):
        selected = panel(); backend = Backend(selected['requests'])
        result = evaluate(backend, selected['requests'], self.checkpoints, self.root/'eval',
                          retained_krc_counts=selected['retained_krc_counts'])
        self.assertEqual(result['returned_generations'], 384)
        self.assertEqual(result['completed_valid_generations'], 384)
        self.assertEqual(result['unattempted_generations'], 0)
        self.assertEqual(self.read('eval/QUALIFICATION.json')['generations'], 48)
        expected = [arm for i in range(64) for r in range(3) for arm in method_order(i,r)]
        self.assertEqual([row[0][1:] for row in backend.executions], expected)
        self.assertEqual(backend.weight_checks, ['/KR','/KRC'])
        self.assertEqual(len(backend.switches), 193)
        self.assertTrue(all(row[2]['purpose']=='evaluation' and row[2]['max_new_tokens']==2048
                            for row in backend.executions))
        self.assertEqual(set(row['method'] for row in self.rows()), {'KR','KRC'})

    def test_additional_retained_K1_canonical_needs_no_second_gate(self):
        self.panel = panel('additional',3); self.requests=self.panel['requests']; self.counts=self.panel['retained_krc_counts']
        result=self.evaluate(gate=0)
        self.assertEqual(result['qualification'],'not_required_prior_panel_gate')
        self.assertFalse((self.root/'eval/QUALIFICATION.json').exists())
        self.assertTrue(all(row['source_method']=='K1' for row in self.read('eval/CANONICAL.json').values()))

    def test_retained_KRC_counts_gate_and_KR_harmful_results_continue(self):
        def harmful_kr(record, options):
            if record['checkpoint_id']=='/KR': harmful(record)
        self.assertEqual(self.evaluate(harmful_kr)['status'],'completed')
        summary=self.read('eval/report/SUMMARY.json')
        self.assertEqual(summary['comparisons']['KR_vs_KRC']['call_reduction_pct'],-50)
        self.assertEqual(summary['comparisons']['KR_vs_KRC']['call_request_directions'],dict(improved=0,regressed=3,tied=0))
        def harmful_krc(record, options):
            if record['checkpoint_id']=='/KRC': harmful(record)
        with self.assertRaisesRegex(RuntimeError,'retained_KRC_verification_calls'):
            self.evaluate(harmful_krc,name='bad')
        self.assertEqual(self.read('bad/STATUS.json')['returned_generations'],2)
        self.assertEqual(self.read('bad/STATUS.json')['invalid_generations'],1)
        self.assertEqual(self.read('bad/STATUS.json')['unattempted_generations'],16)

    def test_retained_tokens_cannot_be_replaced_by_first_measured_output(self):
        def mutate(record,options):
            record['response_ids'][1]=777
            record['metadata']['trace'][0]['committed_tokens'][0]=777
        with self.assertRaisesRegex(RuntimeError,'reference_response_ids'): self.evaluate(mutate)
        self.assertEqual(self.read('eval/0000_r0_KR.json')['response_ids'][1],777)
        self.assertEqual(self.read('eval/CANONICAL.json')['development_0']['response_ids'][1],11)
        self.assertEqual(self.read('eval/STATUS.json')['unattempted_generations'],17)

    def test_repeated_KR_counts_must_remain_consistent(self):
        def mutate(record,options):
            if options['request_id']=='0000_r1_KR': harmful(record)
        with self.assertRaisesRegex(RuntimeError,'repeat_count_mismatch'): self.evaluate(mutate)
        self.assertEqual(self.read('eval/STATUS.json')['invalid_generations'],2)

    def test_unknown_attempt_and_nonfinite_failure_preserve_raw_evidence(self):
        def interrupted(record, options):
            if options['request_id']=='0000_r0_KRC': raise RuntimeError('lost reply')
        with self.assertRaisesRegex(RuntimeError,'lost reply'): self.evaluate(interrupted)
        result=self.read('eval/STATUS.json')
        self.assertEqual((result['returned_generations'],result['unknown_generation_attempts'],result['unattempted_generations']),(1,1,16))
        def nonfinite(record,options): record['metadata']['request_latency_s']=float('nan')
        with self.assertRaises(RuntimeError): self.evaluate(nonfinite,name='nan')
        self.assertEqual(self.read('nan/STATUS.json')['invalid_generations'],1)
        self.assertTrue((self.root/'nan/0000_r0_KR.json').exists())
        self.assertFalse((self.root/'nan/REPORT_FAILURE.json').exists())

    def test_pooled_timing_nested_components_and_each_repeat_are_reported(self):
        def mutate(record,options):
            i=options['request_index']; r=int(options['request_id'].split('_')[1][1:])
            method=record['checkpoint_id'][1:]
            latency={'KR':(1.,3.,10.),'KRC':(1.,9.,10.)}[method][i]*(.5,1.,1.5)[r]
            record['metadata']['request_latency_s']=latency
            record['metadata']['native_generate_s']=.9*latency
            record['metadata']['capture']['cache_maintenance_s']=.1*latency
        self.evaluate(mutate)
        result=self.read('eval/report/SUMMARY.json')
        self.assertEqual(result['KR']['summed_mean_request_latency_s'],14)
        self.assertAlmostEqual(result['KR']['summed_mean_native_generate_s'],12.6)
        self.assertAlmostEqual(result['KR']['summed_mean_cache_maintenance_s'],1.4)
        self.assertAlmostEqual(result['comparisons']['KR_vs_KRC']['request_time_reduction_pct'],30)
        rows=list(csv.DictReader((self.root/'eval/report/REPEAT_TOTALS.csv').read_text().splitlines()))
        self.assertEqual(len(rows),6)
        self.assertEqual({row['method'] for row in rows},{'KR','KRC'})
        self.assertIn('cache_maintenance_s',rows[0])
        self.assertIn('do not add',result['timer_nesting'])

    def test_report_independently_rejects_execution_and_native_tampering(self):
        self.evaluate(); rows=self.rows(); rows[0]['valid_measurement']=False
        result,requests,*_=summarize(rows,self.requests,checkpoints=self.checkpoints,retained_krc_counts=self.counts)
        self.assertEqual(result['valid_requests'],2)
        self.assertIn('execution_rejected',requests[0]['errors'])
        rows=self.rows(); path=Path(rows[1]['record_path']); raw=json.loads(path.read_text()); harmful(raw);path.write_text(json.dumps(raw))
        result,requests,*_=summarize(rows,self.requests,checkpoints=self.checkpoints,retained_krc_counts=self.counts)
        self.assertEqual(result['valid_requests'],2)
        self.assertIn('row_native_disagreement',requests[0]['errors'])

    def test_input_schema_requires_two_exact_panels_and_count_bindings(self):
        plan={key:'path-'+key for key in ('config','ordinary_receipt','k_receipt','pinned_initial',
              'reference_schedule','replay_schedule','target','draft','worker_python','retained_krc_checkpoint')}
        plan['panels']=[panel(),panel('additional')]
        self.assertEqual(validate_plan(plan)['planned_generations'],768)
        plan['panels'][1]['requests'][0]['canonical_source_method']='F'
        with self.assertRaisesRegex(ValueError,'panel-specific'): validate_plan(plan)
        for counts in (None,{},dict(self.counts,development_0=True)):
            with self.assertRaises(ValueError):
                validate_inputs(self.requests,self.checkpoints,counts,1)
        with self.assertRaisesRegex(ValueError,'two distinct'):
            validate_inputs(self.requests,dict(KR='/KR',KRC='/KRC',K1='/K1'),self.counts,1)

    def test_import_does_not_construct_training_or_CUDA(self):
        code="import sys; import experiments.reference_replay_event.evaluate; assert 'torch' not in sys.modules; assert 'online_feedback.training' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__': unittest.main()
