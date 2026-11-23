import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from .storage import capacity_plan, _retire, retire_window, retire_penultimate_export
from experiments.ok_coding.storage import identity
from .runtime import read, storage
from .test_protected import seed_protection


class StorageTests(unittest.TestCase):
    def test_final_tensor_reservation_requires_all_three_complete_arms(self):
        from .contract import ARMS
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            components=dict(tensor_peak_with_five_percent_margin_bytes=70_000_000,
                compact_bytes=3_000_000,source_bytes=1_000_000,workspace_bytes=1_000_000,compiler_bytes=1_000_000)
            final=dict(components,tensor_peak_with_five_percent_margin_bytes=40_000_000,compact_bytes=20_000_000)
            plan=dict(root=str(root),allocation_roots=[str(root)],
                compiler_baseline=dict(directories=[],allocated_bytes=0),
                storage=dict(projected_peak_bytes=76_000_000,phases=dict(training=components,evaluation=final)))
            write_json(root/'STORAGE_PHASE.json',dict(phase='evaluation',training_complete=True))
            for arm in ARMS:
                write_json(root/'campaign_001/arms'/arm/'STATUS.json',dict(status='completed',events_completed=58))
            with patch('experiments.opencode_native_comparison.runtime.compiler_snapshot',return_value=plan['compiler_baseline']), \
                    patch('experiments.opencode_native_comparison.runtime.RESOURCE_LIMITS',
                          dict(new_allocated_bytes=80_000_000,free_reserve_bytes=0)):
                self.assertEqual(storage(plan)['storage_phase'],'evaluation')
                write_json(root/'campaign_001/arms'/ARMS[-1]/'STATUS.json',dict(status='running',events_completed=57))
                with self.assertRaisesRegex(ValueError,'all58-event'):
                    storage(plan)

    def test_phase_forecasts_keep_tensor_and_compact_allowances_distinct(self):
        requests=[dict(prompt_ids=[1]*10) for _ in range(7424)]
        result=capacity_plan(requests,2_097_259_104,6_291_777_312,
            compact_bytes=9_000_000_000,final_compact_bytes=25_000_000_000,
            compiler_bytes=2_000_000_000,source_bytes=500_000_000,
            workspace_bytes=1_000_000_000,state_metadata_growth_bytes=500_000_000,
            qualification_peak_bytes=20_000_000_000)
        train,final=result['phases']['training'],result['phases']['evaluation']
        self.assertEqual((train['complete_state_copies'],train['export_copies']),(7,4))
        self.assertEqual((final['complete_state_copies'],final['export_copies']),(6,3))
        self.assertGreater(train['feature_peak_bytes'],0)
        self.assertEqual(final['feature_peak_bytes'],0)
        self.assertEqual(final['compact_bytes'],25_000_000_000)
        self.assertEqual(train['compact_bytes'],9_000_000_000)
        self.assertEqual(result['projected_peak_bytes'],max(
            train['projected_peak_bytes'],final['projected_peak_bytes']))
        self.assertEqual(result['bound_bytes'],90_000_000_000)
        self.assertLessEqual(result['projected_peak_bytes'],90_000_000_000)
        self.assertEqual(result['status'],'admitted')
        with patch('experiments.opencode_native_comparison.storage.RESOURCE_LIMITS',
                   dict(new_allocated_bytes=80_000_000_000)):
            blocked=capacity_plan(requests,2_097_259_104,6_291_777_312,
                compact_bytes=9_000_000_000,final_compact_bytes=25_000_000_000,
                compiler_bytes=2_000_000_000,source_bytes=500_000_000,
                workspace_bytes=1_000_000_000,state_metadata_growth_bytes=500_000_000,
                qualification_peak_bytes=20_000_000_000)
        self.assertEqual(blocked['status'],'blocked')

    def test_uses_actual_window_prompt_sum_and_atomic_states(self):
        requests=[dict(prompt_ids=[1]*10) for _ in range(7424)]
        result=capacity_plan(requests,2_097_259_104,6_532_631_103,
            compact_bytes=8_000_000_000,compiler_bytes=4_000_000_000,
            source_bytes=1_000_000_000,workspace_bytes=1_000_000_000,
            state_metadata_growth_bytes=100_000_000,qualification_peak_bytes=20_000_000_000)
        self.assertEqual(result['max_window_full_cap_rows'],128*2058)
        self.assertEqual(result['complete_state_copies_at_peak'],7)
        self.assertEqual(result['export_copies_at_peak'],4)
        self.assertEqual(result['feature_replacement_overlap_rows'],2059)
        self.assertFalse(result['offload_credit'])

    def test_rolling_boundaries_keep_two_states_and_only_current_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            seed_protection(root)
            arm = 'O_CODE58'
            base = root/'campaign_001/arms'/arm
            write_json(base/'trainer/INITIAL_CHECK.json',dict(status='passed'))

            def fingerprint(path, *, expected_identity, export_path, receipt_path):
                result = dict(status='passed',path=str(path),identity=expected_identity,
                    export_identity=identity(export_path/'model.safetensors'), exact_export_tensors=True)
                write_json(receipt_path,result)
                return result

            def finish(number):
                event_id=f'event_{number:03d}'
                event_root=base/'events'/event_id
                exported=base/'trainer/events'/event_id/'export'
                exported.mkdir(parents=True)
                weights=exported/'model.safetensors'
                weights.write_bytes(f'weights{number}'.encode())
                state=base/'trainer'/f'state_{number}.pt'
                state.write_bytes(f'state{number}'.encode())
                marker=dict(status='passed',event=number,state=str(state),weights=str(weights),
                    state_identity=identity(state),weights_identity=identity(weights))
                write_json(base/'boundaries'/f'{event_id}.json',marker)
                checkpoint_path=str(exported)
                receipt=dict(preparation=[],training=dict(event_id=event_id,status='completed',
                    checkpoint_path=checkpoint_path))
                write_json(exported.parent/'RESULT.json',dict(status='completed'))
                write_json(event_root/'loop/events'/f'{event_id}.json',receipt)
                write_json(event_root/'loop/feedback_buffer/state.json',dict(records=[],
                    events=[dict(status='committed',receipt=receipt)]))
                write_json(event_root/'PUBLICATION_CHECK.json',dict(status='passed',all_equal=True,
                    event_id=event_id,checkpoint_path=checkpoint_path))
                with patch('experiments.opencode_native_comparison.storage.checkpoint_fingerprints',fingerprint):
                    result=retire_window(root,arm,event_root,[],receipt,marker)
                return marker,result

            first,_=finish(1)
            second,_=finish(2)
            self.assertTrue(Path(first['state']).exists())
            self.assertFalse(Path(first['weights']).exists())
            third,result=finish(3)
            self.assertFalse(Path(first['state']).exists())
            self.assertEqual([row['kind'] for row in result['old_boundary']['removed']],['older_state'])
            self.assertTrue(Path(second['state']).exists())
            self.assertTrue(Path(third['state']).exists())
            self.assertFalse(Path(second['weights']).exists())
            self.assertTrue(Path(third['weights']).exists())
            # The final helper validates an already completed retirement without re-deleting.
            from .storage import _export_retirement_evidence
            receipt,_ = _export_retirement_evidence(base,second)
            saved=read(receipt)
            fingerprint_path=base/'boundaries/event_002_TENSOR_FINGERPRINTS.json'
            fingerprint_path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'exact tensor evidence'):
                _export_retirement_evidence(base,second)
            self.assertEqual(read(receipt),saved)

    def test_retirement_preserves_plan_and_allocated_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            seed_protection(root)
            tensor=root/'old.pt'
            tensor.write_bytes(b'x'*8192)
            evidence=root/'evidence.json'
            write_json(evidence,dict(status='passed'))
            receipt=root/'retire.json'
            _retire(root,receipt,[(tensor,'fixture')],[evidence],reason='test',retained={})
            self.assertFalse(tensor.exists())
            from experiments.ok_coding.storage import identity
            saved=read(receipt)
            allocation_plan=root/'retire.allocated.plan.json'
            self.assertEqual(saved['compact_evidence'][str(allocation_plan)],identity(allocation_plan))
            allocated=read(root/'retire.allocated.json')
            self.assertGreaterEqual(allocated['retired_allocated_bytes'],8192)
            self.assertTrue(allocated['all_exclusive_links'])

    def test_compact_growth_cannot_consume_future_tensor_allowance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'compact.txt').write_bytes(b'x'*16384)
            plan=dict(root=str(root),allocation_roots=[str(root)],
                compiler_baseline=dict(directories=[],allocated_bytes=0),
                storage=dict(projected_peak_bytes=79000,tensor_peak_with_five_percent_margin_bytes=70000,
                    compact_bytes=1000,source_bytes=1000,workspace_bytes=1000,compiler_bytes=6000))
            with patch('experiments.opencode_native_comparison.runtime.compiler_snapshot',return_value=plan['compiler_baseline']), \
                    patch('experiments.opencode_native_comparison.runtime.RESOURCE_LIMITS',
                          dict(new_allocated_bytes=80000,free_reserve_bytes=0)):
                with self.assertRaisesRegex(RuntimeError,'storage admission'):
                    storage(plan)


if __name__=='__main__':
    unittest.main()
