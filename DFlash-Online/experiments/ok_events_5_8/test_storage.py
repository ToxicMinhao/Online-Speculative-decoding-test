import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .storage import StorageLifecycle, GIB, required_bytes, validate_ack, identity


class StorageTests(unittest.TestCase):
    def setup_run(self, root):
        output = root/'campaign_001'; output.mkdir()
        plan = {'storage': {'reserve_bytes': 8*GIB, 'offload_timeout_s': 1,
                            'offload_destination': '/fenrir/offload'}}
        return output, StorageLifecycle(plan, output)

    def artifacts(self, output, event):
        event_root = output/f'event_{event:03d}'
        (event_root/'responses').mkdir(parents=True)
        (event_root/'features').mkdir()
        feature = event_root/'features/p.pt'; feature.write_bytes(b'features')
        capture = event_root/'capture.pt'; capture.write_bytes(b'capture')
        (event_root/'responses/p.json').write_text(json.dumps(
            {'metadata': {'capture': {'feature_path': str(capture)}}}))
        parents = {}
        for arm in ('O','K'):
            folder = output/arm/'events'/f'event_{event:03d}'
            export = folder/'export'; export.mkdir(parents=True)
            (export/'model.safetensors').write_bytes(arm.encode())
            latest = output/arm/'training_state.pt'; latest.write_bytes(b'state'+arm.encode())
            state = latest
            if event in (6,8):
                state = folder/'training_state.pt'; os.link(latest,state)
            parents[arm] = {'state': str(state), 'export': str(export)}
        return parents, {'features':[str(feature)]}, feature, capture

    def ack(self, lifecycle, event):
        request = json.loads((lifecycle.root/f'event_{event:03d}_REQUEST.json').read_text())
        receipt = dict(request, status='verified')
        (lifecycle.root/f'event_{event:03d}_ACK.json').write_text(json.dumps(receipt))

    def test_idle_offload_retains_live_state_and_required_endpoint_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            output, lifecycle = self.setup_run(Path(directory))
            parents,pool,feature,capture = self.artifacts(output,6)
            with patch('time.sleep', side_effect=lambda _: self.ack(lifecycle,6)):
                row = lifecycle.after_event(6,parents,pool,evaluated=True)
            self.assertEqual(row['status'],'completed')
            self.assertFalse(feature.exists()); self.assertFalse(capture.exists())
            for arm,parent in parents.items():
                self.assertEqual(parent['state'],str(output/arm/'training_state.pt'))
                self.assertTrue(Path(parent['state']).exists())
                self.assertTrue((Path(parent['export'])/'model.safetensors').exists())
                self.assertIn(f'{arm}/events/event_006/training_state.pt',lifecycle.archived)

    def test_missing_or_invalid_ack_never_deletes(self):
        with tempfile.TemporaryDirectory() as directory:
            output,lifecycle = self.setup_run(Path(directory))
            parents,pool,feature,capture = self.artifacts(output,5)
            lifecycle.settings['offload_timeout_s']=0
            with self.assertRaises(TimeoutError):
                lifecycle.after_event(5,parents,pool,evaluated=False)
            self.assertTrue(feature.exists()); self.assertTrue(capture.exists())
            request = json.loads((lifecycle.root/'event_005_REQUEST.json').read_text())
            with self.assertRaises(ValueError):
                validate_ack(request,dict(request,status='verified',files={}))

    def test_retention_not_allowed_before_required_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output,lifecycle = self.setup_run(Path(directory))
            with self.assertRaises(ValueError):
                lifecycle.after_event(6,{}, {}, evaluated=False)

    def test_phase_budget_preserves_reserve_and_atomic_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            output,lifecycle = self.setup_run(Path(directory))
            with patch('shutil.disk_usage') as usage:
                usage.return_value.free=12*GIB
                with self.assertRaises(RuntimeError):
                    lifecycle.admit_training('O',5,{}, {})
            self.assertGreater(required_bytes(10*GIB),22*GIB-1)
            row=json.loads((lifecycle.root/'event_005_O_training_STORAGE.json').read_text())
            self.assertEqual(row['reserve_bytes'],8*GIB)
            self.assertFalse(row['admitted'])

    def test_retire_previous_O_export_only_after_O_advanced_and_copy_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            output,lifecycle=self.setup_run(Path(directory))
            parents,_,_,_=self.artifacts(output,5)
            old_O=Path(parents['O']['export'])/'model.safetensors'
            old_K=Path(parents['K']['export'])/'model.safetensors'
            for path in (old_O,old_K):
                lifecycle.archived[str(path.relative_to(output))]=identity(path)
            lifecycle.retire_obsolete_exports(parents)
            self.assertTrue(old_O.exists()); self.assertTrue(old_K.exists())
            new_export=output/'O/events/event_006/export';new_export.mkdir(parents=True)
            parents['O']['export']=str(new_export)
            lifecycle.retire_obsolete_exports(parents)
            self.assertFalse(old_O.exists()); self.assertTrue(old_K.exists())


if __name__ == '__main__':
    unittest.main()
