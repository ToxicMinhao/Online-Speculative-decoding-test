"""CPU-only admission, durable reservation, clock and witness binding tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .point1_prepare import authenticate_parent_export, authenticate_witness, resource_admission
from .point1_launch import Point1StepLedger, child_memory, completed_cases, phase_clock, remaining_time


class AdmissionTests(unittest.TestCase):
    def test_export_directory_resolves_to_authenticated_weight_file(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.safetensors'
            path.write_bytes(b'protected weights')
            with patch('experiments.frontier_preservation_pilot.point1_prepare.EXPORT_IDENTITY',identity(path)):
                self.assertEqual(authenticate_parent_export(d),path)
                path.write_bytes(b'changed weights')
                with self.assertRaisesRegex(ValueError,'export changed'):
                    authenticate_parent_export(d)

    def admission(self, **changes):
        values = dict(free_bytes=408_000_000_000, charged_bytes=600_000_000,
            remaining_write_bytes=1_000_000_000, host_available_bytes=103_000_000_000,
            gpu_free_bytes=80_000_000_000, gpu_busy=False)
        values.update(changes)
        return resource_admission(**values)

    def test_each_resource_gate(self):
        self.assertTrue(self.admission()['admitted'])
        for change, failed in (
            (dict(free_bytes=400_500_000_000),'disk_reserve'),
            (dict(charged_bytes=1_500_000_000),'all_locations'),
            (dict(host_available_bytes=86_000_000_000),'host_memory'),
            (dict(gpu_free_bytes=59_000_000_000),'gpu_memory'),
            (dict(gpu_busy=True),'gpu_idle')):
            with self.subTest(change=change):
                result=self.admission(**change)
                self.assertFalse(result['admitted'])
                self.assertFalse(result['checks'][failed])

    def test_exact_boundaries_and_negative_accounting(self):
        self.assertTrue(self.admission(free_bytes=401_000_000_000,
            charged_bytes=1_000_000_000,host_available_bytes=87_000_000_000,
            gpu_free_bytes=60_000_000_000)['admitted'])
        with self.assertRaises(ValueError):
            self.admission(charged_bytes=-1)


class StepLedgerTests(unittest.TestCase):
    def test_completion_requires_all_four_completed_cases(self):
        rows = [dict(attempt=1,case=case,status='completed')
                for case in ('BETA','C','F','C_REPEAT')]
        self.assertTrue(completed_cases(dict(calls=rows),1))
        self.assertFalse(completed_cases(dict(calls=rows[:-1]),1))
        rows[-1]['status']='reserved'
        self.assertFalse(completed_cases(dict(calls=rows),1))

    def test_durable_pending_calls_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'STEP_LEDGER.json'
            ledger=Point1StepLedger(path,dict(sha256='contract',bytes=1))
            ledger.start_attempt(1)
            first=ledger.reserve('BETA')
            self.assertEqual(json.loads(path.read_text())['calls'][0]['status'],'reserved')
            with self.assertRaises(RuntimeError):
                ledger.reserve('BETA')
            ledger.complete(first)
            with self.assertRaises(ValueError):
                ledger.complete(first)
            ledger.finish_attempt(1,False)
            reopened=Point1StepLedger(path,dict(sha256='contract',bytes=1))
            self.assertEqual(len(reopened.value['calls']),1)
            self.assertEqual(reopened.value['calls'][0]['status'],'completed')

    def test_one_identified_retry_and_six_call_cap(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Point1StepLedger(Path(d)/'steps.json',{'binding':'original'})
            ledger.start_attempt(1)
            for case in ('BETA','C','F','C_REPEAT'):
                ledger.complete(ledger.reserve(case))
            ledger.finish_attempt(1,False)
            with self.assertRaises(ValueError):
                ledger.start_attempt(2)
            ledger.start_attempt(2,'Documented implementation correction')
            for case in ('BETA','C'):
                ledger.complete(ledger.reserve(case))
            with self.assertRaises(RuntimeError):
                ledger.reserve('F')
            ledger.finish_attempt(2,False)
            with self.assertRaises(ValueError):
                ledger.start_attempt(3,'Another change')
            with self.assertRaises(ValueError):
                Point1StepLedger(ledger.path,{'binding':'replacement'})

    def test_completed_attempt_cannot_retry(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Point1StepLedger(Path(d)/'steps.json',{'original':True})
            with self.assertRaises(ValueError):
                ledger.reserve('C')
            ledger.start_attempt(1)
            with self.assertRaises(ValueError):
                ledger.reserve('new_arm')
            ledger.finish_attempt(1,True)
            with self.assertRaises(ValueError):
                ledger.start_attempt(2,'No failed run')


class ClockTests(unittest.TestCase):
    def test_retry_preserves_original_clock_and_new_consumption(self):
        with tempfile.TemporaryDirectory() as d:
            clock=phase_clock(d,{'original':True},1,1000.0)
            clock['completed_occupied_card_s']=300.0
            write_json(Path(d)/'CLOCK.json',clock)
            with self.assertRaises(FileExistsError):
                phase_clock(d,{'original':True},1,1600.0)
            resumed=phase_clock(d,{'original':True},2,1600.0)
            self.assertEqual(resumed['first_gpu_action_epoch'],1000.0)
            self.assertEqual(remaining_time(resumed,1600.0,100),3000.0)
            self.assertAlmostEqual(resumed['historical_occupied_card_s'],5175.047027264)
            self.assertEqual(remaining_time(resumed,1050.0,3400.0),-100.0)

    def test_active_attempt_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            clock=phase_clock(d,{'original':True},1,1000.0)
            clock['active_attempt']={'number':1}
            write_json(Path(d)/'CLOCK.json',clock)
            with self.assertRaises(ValueError):
                phase_clock(d,{'original':True},2,1100.0)


class MemoryTests(unittest.TestCase):
    def test_child_rss_and_peak_are_reported_and_exited_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'123'
            path.mkdir()
            (path/'status').write_text('Name:\tpython\nVmRSS:\t1234 kB\nVmHWM:\t2345 kB\n')
            result=child_memory(123,d)
            self.assertEqual(result['resident_bytes'],1234*1024)
            self.assertEqual(result['peak_resident_bytes'],2345*1024)
            self.assertEqual(child_memory(456,d)['status'],'exited')


class WitnessTests(unittest.TestCase):
    def fixture(self, root):
        root=Path(root)
        witnesses, features, parents=[],[],[]
        for i in range(8):
            feature=root/f'features/{i}.pt'
            feature.parent.mkdir(exist_ok=True)
            feature.write_bytes(f'fixture{i}'.encode())
            expected=identity(feature)
            witnesses.append(dict(record_id=f'new{i}',feature_path=str(feature),feature_identity=expected,anchors=[1,3]))
            features.append(dict(path=str(feature),identity=expected,record_id=f'new{i}'))
            parents.append(dict(record_id=f'new{i}',anchors=[1,3],blocks=[]))
        binding_path=root/'witness/BINDINGS.json'
        write_json(binding_path,dict(status='passed',sets=dict(new_buffer=witnesses)))
        write_json(root/'witness/parent.json',dict(status='passed',endpoint='parent',counters=[128,256,1],
            binding_identity=identity(binding_path),sets=dict(new_buffer=parents)))
        write_json(root/'shared/FEATURE_IDENTITIES.json',features)
        entries=[dict(path=name,**identity(root/name)) for name in (
            'witness/BINDINGS.json','witness/parent.json','shared/FEATURE_IDENTITIES.json')]
        manifest=root/'MANIFEST.json'
        write_json(manifest,dict(execution_commit='225802657d630c0985abd68d6e7453f3e47c5b74',files=entries))
        return manifest,[dict(path=str(root/'original.pt'))],dict(responses=[dict(record_id='old')])

    def test_authentication_and_live_feature_change(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,selected,window=self.fixture(d)
            with patch('experiments.frontier_preservation_pilot.point1_prepare.ARCHIVE_MANIFEST_IDENTITY',identity(manifest)):
                result=authenticate_witness(d,manifest,selected,window)
                self.assertEqual(result['status'],'authenticated')
                self.assertEqual(result['responses'],8)
                self.assertEqual(result['set_name'],'new_buffer')
                Path(result['items'][0]['feature_path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'feature bytes'):
                    authenticate_witness(d,manifest,selected,window)

    def test_record_and_path_disjointness(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,selected,window=self.fixture(d)
            with patch('experiments.frontier_preservation_pilot.point1_prepare.ARCHIVE_MANIFEST_IDENTITY',identity(manifest)):
                with self.assertRaisesRegex(ValueError,'overlaps'):
                    authenticate_witness(d,manifest,selected,dict(responses=[dict(record_id='new0')]))
                with self.assertRaisesRegex(ValueError,'overlaps'):
                    authenticate_witness(d,manifest,[dict(path=str(Path(d)/'features/0.pt'))],window)


if __name__=='__main__':
    unittest.main()
