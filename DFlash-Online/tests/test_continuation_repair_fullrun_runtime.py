"""Accounting failures and atomic overlap are scientific execution boundaries."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.continuation_repair_fullrun import runtime, launch
from experiments.continuation_repair_fullrun.storage import capacity_plan


class RuntimeTest(unittest.TestCase):
    def test_failed_operation_remains_pending_and_cannot_reset_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime.initialize(folder)
            ledger=runtime.WorkLedger(folder,'gate','qualification')
            def fails():raise RuntimeError('fixture')
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                ledger.call({'adam':1},'failed_adam',fails)
            state=runtime.read(Path(folder)/'WORK.json')
            self.assertEqual(state['attempted'],{'qualification.adam':1})
            self.assertEqual(state['completed'],{})
            self.assertEqual(len(state['pending']),1)
            with self.assertRaises(FileExistsError):runtime.initialize(folder)

    def test_failed_qualification_consumes_existing_eight_step_ceiling(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime.initialize(folder);ledger=runtime.WorkLedger(folder,'gate','qualification')
            first=ledger.reserve('BASE')
            for _ in range(7):ledger.complete(ledger.reserve('TAIL'))
            with self.assertRaisesRegex(RuntimeError,'ceiling'):ledger.reserve('BASE')
            state=runtime.read(Path(folder)/'WORK.json')
            self.assertIn(first,state['pending'])
            self.assertEqual(state['completed']['qualification.adam'],7)

    def test_two_arm_peak_includes_completed_arm_and_atomic_successor(self):
        requests=[dict(prompt_ids=[1]*10) for _ in range(7424)]
        result=capacity_plan(requests,2_097_259_104,6_300_000_000,
            compact_bytes=9_000_000_000,compiler_bytes=2_000_000_000,
            source_bytes=500_000_000,workspace_bytes=1_000_000_000,
            state_metadata_growth_bytes=500_000_000,qualification_peak_bytes=20_000_000_000)
        self.assertEqual(result['complete_state_copies_at_peak'],5)
        self.assertEqual(result['export_copies_at_peak'],3)
        self.assertEqual(result['phases']['evaluation']['complete_state_copies'],4)
        self.assertEqual(result['status'],'admitted')

    def test_unique_receipt_temporaries_allow_independent_writers(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'status.json'
            runtime.write_json(path,{'status':'one'})
            runtime.write_json(path,{'status':'two'})
            self.assertEqual(runtime.read(path),{'status':'two'})
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_compact_growth_cannot_consume_future_tensor_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            components=dict(tensor_peak_with_five_percent_margin_bytes=10000,
                compact_bytes=2000,source_bytes=0,workspace_bytes=0,compiler_bytes=0)
            plan=dict(root=folder,allocation_roots=[folder],compiler_baseline=dict(directories=[],allocated_bytes=0),
                storage=dict(external_charge_bytes=0,projected_peak_bytes=12000,
                    phases=dict(training=components,evaluation=dict(components,tensor_peak_with_five_percent_margin_bytes=1000))))
            compiler=dict(directories=[],allocated_bytes=0)
            with patch.object(runtime,'allocated_bytes',return_value=3000),patch.object(runtime,'compiler_snapshot',return_value=compiler),patch.object(runtime.shutil,'disk_usage',return_value=SimpleNamespace(free=400_000_009_000)):
                with self.assertRaisesRegex(RuntimeError,'storage admission'):runtime.storage(plan)
            with patch.object(runtime,'allocated_bytes',return_value=3000),patch.object(runtime,'compiler_snapshot',return_value=compiler),patch.object(runtime.shutil,'disk_usage',return_value=SimpleNamespace(free=400_000_010_000)):
                result=runtime.storage(plan)
                self.assertEqual(result['remaining_components']['tensors'],10000)
                self.assertEqual(result['charged_bytes']+result['remaining_peak_bytes'],13000)
                for arm in ('BASE','TAIL'):
                    runtime.write_json(Path(folder)/'campaign_001/arms'/arm/'STATUS.json',dict(status='completed',events_completed=58))
                    self.assertEqual(runtime.storage(plan)['storage_phase'],'evaluation' if arm=='TAIL' else 'training')

    def test_qualification_has_one_shared_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertEqual(launch.qualification_attempt(root,'qualification_training'),1)
            runtime.write_json(root/'processes/qualification_training_001.json',dict(exit_code=1))
            self.assertEqual(launch.qualification_attempt(root,'qualification_training'),2)
            runtime.write_json(root/'processes/qualification_training_002.json',dict(exit_code=0))
            self.assertEqual(launch.qualification_attempt(root,'qualification_serving'),1)
            runtime.write_json(root/'processes/qualification_serving_001.json',dict(exit_code=1))
            with self.assertRaisesRegex(RuntimeError,'overall qualification retry'):
                launch.qualification_attempt(root,'qualification_serving')

    def test_remaining_forecast_uses_completed_slots_and_common_slowest_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            runtime.initialize(root)
            plan=dict(root=folder,forecast=dict(full_arm_wall_s=580.,final_evaluation_wall_s=13408.,
                qualification_wall_s=100.,audit_wall_s=20.))
            runtime.write_json(root/'campaign_001/arms/BASE/STATUS.json',dict(events_completed=10,worker_wall_s=100.))
            runtime.write_json(root/'campaign_001/arms/TAIL/STATUS.json',dict(events_completed=20,worker_wall_s=400.))
            runtime.write_json(root/'qualification/QUALIFICATION.json',dict(status='passed'))
            work=runtime.read(root/'WORK.json');work['completed']={'production.canonical_generation':1024}
            runtime.write_json(root/'WORK.json',work)
            result=launch.remaining_forecast(plan)
            self.assertEqual(result,dict(wall_s=(48+38)*20+12384+20,card_s=2*(48+38)*20+12384))
            runtime.write_json(root/'evaluation/SUMMARY.json',dict(status='completed'))
            self.assertEqual(launch.remaining_forecast(plan)['wall_s'],(48+38)*20+20)


class ChildFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        runtime.initialize(self.root);(self.root/'logs').mkdir()
        self.plan=dict(root=str(self.root),devices=dict(A='GPU-A',B='GPU-B'),training_python='/python',
            source_root=str(self.root),zstd='/bin/zstd')
    def tearDown(self):self.temp.cleanup()
    def test_preprocess_start_failure_closes_only_begun_card_leases(self):
        with patch.object(launch,'gpu_inventory',return_value={'processes':[]}),patch.object(launch,'child_environment',return_value={'PATH':'/bin'}),patch.object(launch.subprocess,'Popen',side_effect=OSError('startup failed')):
            with self.assertRaisesRegex(OSError,'startup failed'):
                launch.child('/plan',self.plan,'train',arm='BASE')
        record=runtime.read(self.root/'processes/train_BASE_001.json')
        self.assertFalse(record['process_created']);self.assertTrue(record['termination_confirmed'])
        clock=runtime.read(self.root/'CLOCK.json')
        self.assertFalse(clock['active']);self.assertEqual(len(clock['finished']),2)
    def test_failed_termination_preserves_receipt_and_open_card_interval(self):
        process=SimpleNamespace(pid=123,returncode=None,poll=lambda:None)
        with patch.object(launch,'gpu_inventory',return_value={'processes':[]}),patch.object(launch,'child_environment',return_value={'PATH':'/bin'}),patch.object(launch.subprocess,'Popen',return_value=process),patch.object(launch,'check_clock',side_effect=TimeoutError('cap')),patch.object(launch,'stop_owned',side_effect=OSError('still alive')):
            with self.assertRaisesRegex(OSError,'still alive'):
                launch.child('/plan',self.plan,'qualification_training')
        record=runtime.read(self.root/'processes/qualification_training_001.json')
        self.assertTrue(record['process_created']);self.assertFalse(record['termination_confirmed'])
        self.assertIsNone(record['finished_epoch']);self.assertIn('still alive',record['cleanup_error'])
        self.assertEqual(len(runtime.read(self.root/'CLOCK.json')['active']),1)

if __name__=='__main__':unittest.main()
