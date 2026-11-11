from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.survival_confirmation import report as R
from experiments.survival_head.contract import ANALYSIS


def rows(n=136):
    values = []
    for i in range(n):
        for method, calls, latency, prefix in [('KPLUS', 10, 2., 1), ('LOCAL', 9, 1.9, 2), ('SURVIVAL', 8, 1.8, 3)]:
            for repeat in range(2):
                values.append(dict(phase='final', event=5, dataset='gsm8k', seed=17,
                    method=method, validation_status='valid', valid_measurement=True, errors=[],
                    panel_ordinal=i, request_id=f'r{i}', repeat=repeat, verification_calls=calls,
                    accepted_draft_tokens=64-calls, native_accepted_draft_tokens=64-calls,
                    committed_progress=64, output_tokens=65, checkpoint_id='/'+method,
                    finish_reason='eos', cap_hit=False, request_latency_s=latency+repeat*.2,
                    native_generate_s=latency-.1+repeat*.2, cache_maintenance_s=.01,
                    first_round_present=True, first_round_start_digest='a'*64,
                    first_round_accepted_prefix_tokens=prefix, first_round_raw_prefix_tokens=prefix,
                    first_round_output_start=0))
    return values


def endpoint(sample=None):
    return R.endpoint_from_audit(dict(status='passed', errors=[], rows=rows() if sample is None else sample), 136)[0]


class ConfirmationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = endpoint()['summary']

    def test_nonpilot_size_and_four_strict_intervals(self):
        self.assertEqual(self.summary['request_count'], 136)
        self.assertEqual(R.decision(self.summary)['outcome'], 'confirmed_on_study_heldout_gsm8k')
        for control in ('KPLUS', 'LOCAL'):
            for metric in ('call_reduction', 'request_latency_s_reduction'):
                changed = deepcopy(self.summary)
                changed['comparisons']['SURVIVAL_vs_'+control][metric]['ci95_pct'][0] = 0.
                decision = R.decision(changed)
                self.assertEqual(decision['outcome'], 'directionally_replicated_unresolved')
                self.assertEqual(len(decision['unresolved_intervals']), 1)

    def test_every_primary_point_and_sweep_is_required(self):
        for control in ('KPLUS', 'LOCAL'):
            for metric in ('call_reduction', 'request_latency_s_reduction'):
                changed = deepcopy(self.summary)
                changed['comparisons']['SURVIVAL_vs_'+control][metric]['point_pct'] = 0.
                d = R.decision(changed)
                self.assertEqual(d['outcome'], 'mixed_or_unfavorable')
                self.assertIn(control, d['failures'][0])
            for repeat in ('0', '1'):
                changed = deepcopy(self.summary)
                changed['repeat_totals'][repeat][control]['request_latency_s'] = changed['repeat_totals'][repeat]['SURVIVAL']['request_latency_s']
                d = R.decision(changed)
                self.assertEqual(d['outcome'], 'mixed_or_unfavorable')
                self.assertTrue(any('sweep_'+repeat in x for x in d['failures']))

    def test_failed_raw_audit_missing_arm_or_duplicate_repeat_suppresses_inference(self):
        for audit in [dict(status='failed', errors=['native_output'], rows=rows()),
                      dict(status='passed', errors=['native_output'], rows=rows()),
                      dict(status='passed', errors=[], rows=rows()[:-1]),
                      dict(status='passed', errors=[], rows=rows()[:-1]+[rows()[0]])]:
            result, pairs = R.endpoint_from_audit(audit, 136)
            self.assertFalse(result['inference_permitted'])
            self.assertEqual(pairs, [])

    def test_counts_are_once_and_times_are_within_request_means(self):
        totals = self.summary['method_totals']['KPLUS']
        self.assertEqual(totals['verification_calls'], 1360)
        self.assertAlmostEqual(totals['request_latency_s'], 136*2.1)

    def test_confirmation_preserves_exact_seed_eos_but_pilot_default_stays_strict(self):
        sample = rows()
        for row in sample:
            if row['panel_ordinal'] == 0:
                row.update(verification_calls=0, accepted_draft_tokens=0, native_accepted_draft_tokens=0,
                    committed_progress=0, output_tokens=1, first_round_present=False,
                    first_round_start_digest=None, first_round_accepted_prefix_tokens=0,
                    first_round_raw_prefix_tokens=0, first_round_output_start=None)
        result = endpoint(sample)
        self.assertTrue(result['inference_permitted'])
        self.assertEqual(result['summary']['request_count'],136)
        self.assertEqual(result['summary']['method_totals']['KPLUS']['verification_calls'],1350)
        self.assertEqual(result['summary']['comparisons']['SURVIVAL_vs_KPLUS']['first_native_round']['seed_eos_requests'],1)
        with self.assertRaises(ValueError):
            R.reduce_stage(sample,dataset='gsm8k',event=5,panel_size=136)
        with self.assertRaises(ValueError):
            R.reduce_stage(sample[:128*6],dataset='gsm8k',event=5)
        for key, wrong in [('output_tokens',2),('committed_progress',1),('finish_reason','length'),('first_round_present',True)]:
            changed=deepcopy(sample)
            for row in changed:
                if row['panel_ordinal']==0: row[key]=wrong
            self.assertFalse(endpoint(changed)['inference_permitted'],key)

    def test_only_confirmation_sample_sizes_are_allowed(self):
        for n in [128, 135, 137, 769, 0, True, 136.]:
            with self.assertRaises(ValueError): R.validate_panel_size(n)
        self.assertEqual(R.validate_panel_size(768), 768)

    def test_resources_keep_missing_values_and_nested_phases_separate(self):
        values = R.resource_rows({'worker': {'wall_s': 12., 'loading_s': 2.}, 'timing': {'evaluation_wall_s': 9.}})
        keyed = {(v['phase'], v['metric']): v['value'] for v in values}
        self.assertEqual(keyed['worker', 'wall_s'], 12.)
        self.assertEqual(keyed['references', 'wall_s'], 'unavailable')
        self.assertNotIn('total', keyed)

    def test_partial_build_retains_raw_rows_without_comparisons(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';(root/'sealed').mkdir(parents=True);(root/'evaluation').mkdir()
            plan=dict(root=str(root), panel_size=136, analysis=ANALYSIS,
                execution_source={'commit': 'a'*40}, checkpoints={})
            (root/'sealed/PLAN.json').write_text(json.dumps(plan))
            (root/'evaluation/MEASUREMENTS.jsonl').write_text(json.dumps({'errors': ['partial']})+'\n')
            result=R.build(root, Path(temp)/'report')
            self.assertEqual(result['status'], 'partial_or_incomplete')
            self.assertEqual(result['decision']['outcome'], 'invalid_or_incomplete')
            self.assertFalse(result['endpoint']['inference_permitted'])
            self.assertIn('partial', (Path(temp)/'report/TIMING_REPEATS.csv').read_text())
            self.assertIn('No complete valid', (Path(temp)/'report/REPORT.md').read_text())

    def test_complete_build_calls_strict_adapter_and_reports_separate_phase_costs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'
            plan=dict(root=str(root), panel_size=136, analysis=ANALYSIS, evaluation_schedule=[],
                execution_source={'commit': 'a'*40}, serving_gpu_identity='GPU-fixture',
                checkpoints={m: {'checkpoint_path': '/'+m} for m in ('KPLUS','LOCAL','SURVIVAL')})
            values={'sealed/PLAN.json': plan, 'sealed/PANEL.json': {'requests': []},
                'sealed/EXPOSURE_INVENTORY.json': {'held_out_from_survival': True},
                'references/CANONICAL.json': {}, 'sealed/SEAL.json': {'fixture': True},
                'STATUS.json': {'status':'completed','wall_s': 100.,'optimizer_steps':0},
                'POSTFLIGHT.json': {'status':'passed'},
                'references/STATUS.json': {'status':'completed','wall_s':10.,'loading_s':2.,
                    'generation_wall_s':7.,'valid_generations':4,'reused':132,'optimizer_steps':0},
                'evaluation/STATUS.json': {'status':'completed','evaluation_wall_s':80.,
                    'activation_s':5.,'weight_validation_s':2.},
                'evaluation/WORKER.json': {'status':'completed','wall_s':90.,'loading_s':8.}}
            for name,value in values.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            (root/'evaluation/MEASUREMENTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows()))
            with patch('experiments.survival_confirmation.evaluation.audit_stage',
                       return_value=dict(status='passed',errors=[],rows=rows())) as auditor:
                result=R.build(root,Path(temp)/'report')
            self.assertEqual(result['status'],'complete')
            self.assertEqual(auditor.call_args.kwargs['sealed_n'],136)
            self.assertEqual(auditor.call_args.kwargs['physical_gpu'],'GPU-fixture')
            self.assertNotIn('panel_size',auditor.call_args.kwargs)
            text=(Path(temp)/'report/REPORT.md').read_text()
            self.assertIn('references/generation_wall_s | 7.0',text)
            self.assertIn('worker/loading_s | 8.0',text)
            self.assertIn('timing/evaluation_wall_s | 80.0',text)

    def test_fallacy_coverage_and_no_automatic_followup(self):
        self.assertEqual(len(R.fallacy_checks()['checks']), 11)
        self.assertFalse(R.decision(self.summary)['automatic_followup_authorized'])


if __name__ == '__main__':
    unittest.main()
