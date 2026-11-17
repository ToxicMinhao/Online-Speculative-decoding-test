"""Resource forecasts never change hard scientific caps or retain fewer states."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.c_paired_confirmation import resources as P, runtime as R


def geometry():
    return {dataset:dict(hidden_size=4096,captured_layers=5,next_prompt_lengths=[100]*128,
        recovery_bytes=6_298_000_000,export_bytes=2_097_259_104) for dataset in ('gsm8k','mathqa')}


class StorageTests(unittest.TestCase):
    def test_alternate_policy_preserves_eight_recoveries_four_banks_all_final_exports(self):
        full=P.storage_plan(geometry(),retain_final_exports=True)
        alternate=P.storage_plan(geometry(),retain_final_exports=False)
        self.assertEqual((alternate['retained_complete_recoveries'],alternate['retained_unique_feature_banks']),(8,4))
        self.assertEqual(alternate['geometry_row_bytes'],{'gsm8k':49164,'mathqa':49164})
        self.assertTrue(alternate['all_four_final_exports_retained_after_timing'])
        self.assertTrue(alternate['gsm_final_exports_temporarily_retired'])
        phases={x['name']:x for x in alternate['phases']}
        self.assertEqual(phases['gsm8k_complete']['export_bytes'],0)
        self.assertEqual(phases['mathqa_complete']['export_bytes'],2*alternate['export_bound_bytes']['mathqa'])
        for name in ('gsm8k_timing','mathqa_timing'):
            self.assertEqual(phases[name]['recovery_bytes'],4*sum(alternate['recovery_bound_bytes'].values()))
            self.assertEqual(phases[name]['feature_bytes'],2*sum(alternate['bank_bound_bytes'].values()))
            self.assertEqual(phases[name]['export_bytes'],2*sum(alternate['export_bound_bytes'].values()))
        self.assertLess(alternate['peak_new_bytes'],full['peak_new_bytes'])
        self.assertEqual(alternate['required_initial_free_bytes'],alternate['peak_new_bytes']+400_000_000_000)
        self.assertFalse(alternate['historical_retirement'])

    def test_worst_output_cap_prompt_rows_and_unpriced_panel_are_not_averages(self):
        values=geometry();values['mathqa']['next_prompt_lengths'][-1]=2048
        plan=P.storage_plan(values,retain_final_exports=False)
        self.assertEqual(plan['bank_bound_bytes']['mathqa'],(127*100+2048+128*2048)*49164)
        self.assertEqual(plan['temporary_feature_bytes'],2*(2048+2049)*49164)
        for phase in plan['phases']:
            subtotal=sum(phase[k] for k in ('recovery_bytes','feature_bytes','export_bytes','transient_bytes'))+1_000_000_000+1_000_000_000+4*2**30
            self.assertGreaterEqual(phase['serialization_allowance_bytes'],subtotal*.05)
            self.assertEqual(phase['peak_new_bytes'],subtotal+phase['serialization_allowance_bytes'])
        values['mathqa']['next_prompt_lengths'].pop()
        with self.assertRaisesRegex(ValueError,'128'):P.storage_plan(values)

    def test_only_expected_overlap_and_final_order(self):
        groups=P.production_groups()
        self.assertEqual(len(groups),10)
        jobs=[job for group in groups for job in group]
        self.assertEqual(len(jobs),16)
        self.assertEqual(groups[-2:],[[('gsm8k','final','A')],[('mathqa','final','A')]])
        self.assertTrue(all(slot=='A' for d,t,slot in jobs if t.endswith(('_first','_second'))))
        costs={d:{task:(10 if task.endswith(('_first','_second')) else 3)
                  for name,task,slot in jobs if name==d} for d in ('gsm8k','mathqa')}
        estimate=P.price_groups(groups,costs)
        self.assertEqual(estimate,dict(elapsed_seconds=86,occupied_card_seconds=104))


class LedgerTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        R.initialize_work(self.root)

    def test_exact_hard_caps_and_no_clock_reset(self):
        self.assertEqual((R.LIMITS['adam'],R.LIMITS['response_forward'],R.LIMITS['response_backward'],R.LIMITS['training_initialization']),(1056,2144,2144,16))
        self.assertEqual(sum(R.LIMITS[k] for k in ('feedback_generation','canonical_generation','qualification_generation','warmup_generation','scored_generation')),4212)
        self.assertEqual((R.TIME_CAP_S,R.CARD_CAP_S,R.ALLOCATION_BYTES,R.RESERVE_BYTES),(14400,18000,125_000_000_000,400_000_000_000))
        before=(self.root/'CLOCK.json').read_bytes()
        with self.assertRaises(FileExistsError):R.initialize_work(self.root)
        self.assertEqual((self.root/'CLOCK.json').read_bytes(),before)

    def test_failed_call_stays_attempted_and_does_not_complete_or_retry(self):
        ledger=R.WorkLedger(self.root,'gsm8k_N_first');calls=[]
        def failed():calls.append(1);raise RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError,'native failure'):
            ledger.call({'adam':1,'response_forward':2},'window',failed)
        receipt=R.read(self.root/'WORK.json')
        self.assertEqual(len(calls),1);self.assertEqual(receipt['attempted']['adam'],1)
        self.assertEqual(receipt['completed']['adam'],0);self.assertEqual(len(receipt['pending']),1)
        receipt['attempted']['adam']=R.LIMITS['adam'];write_json(self.root/'WORK.json',receipt)
        with self.assertRaisesRegex(RuntimeError,'cap'):
            ledger.call({'adam':1},'overrun',lambda:calls.append(2))
        self.assertEqual(calls,[1])
        for bad in ({'new_arm':1},{'adam':-1},{'adam':True}):
            with self.assertRaises(ValueError):ledger.call(bad,'bad',lambda:None)

    def test_reserve_or_expired_clock_blocks_before_future_writes(self):
        plan=dict(root=str(self.root),allocation_roots=[str(self.root)],compiler_growth_reserved_bytes=4*2**30,
                  storage={'peak_new_bytes':124_000_000_000})
        with patch.object(R,'clock_status',return_value={'elapsed_s':0,'occupied_s':0}),patch.object(R,'allocated_bytes',return_value=10_000_000_000):
            with patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=513_999_999_999)):
                with self.assertRaisesRegex(RuntimeError,'reserve'):R.boundary(plan)
            with patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=514_000_000_000)):
                self.assertEqual(R.boundary(plan)['remaining_peak_bytes'],114_000_000_000)
        with patch.object(R,'clock_status',return_value={'elapsed_s':14400,'occupied_s':0}):
            with self.assertRaises(TimeoutError):R.boundary(plan)

    def test_prospectively_declared_forecast_does_not_change_hard_clock(self):
        plan={'root':str(self.root),'forecast':{'factor':1.3,'fixed_margin_s':600}}
        with patch.object(R,'clock_status',return_value={'elapsed_s':100,'occupied_s':200}):
            result=R.require_time_for(plan,1000,1200)
            self.assertEqual(result['conservative_elapsed_s'],1900)
            self.assertEqual(result['conservative_card_s'],2160)
            with self.assertRaisesRegex(RuntimeError,'clock'):R.require_time_for(plan,11000,1200)
        self.assertEqual((R.TIME_CAP_S,R.CARD_CAP_S),(14400,18000))


if __name__=='__main__':unittest.main()
