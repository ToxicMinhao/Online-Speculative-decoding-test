"""Bounded campaign orchestration, qualification admission and selected retention."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.role_loss_pilot import campaign as C,contract as P,storage as S,qualify as Q
from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def test_six_independent_work_budgets_and_no_fifth_event(self):
        limits=[P.arm_limits(arm) for _ in P.DATASETS for arm in P.ARMS]
        for key,total in [('optimizer_steps',3072),('response_forwards',6144),
                          ('response_backwards',6144),('online_generations',3072),('correctness_generations',8)]:
            self.assertEqual(sum(x[key] for x in limits),total)
        for arm in P.ARMS:self.assertEqual(P.expected_work(4,arm),P.arm_limits(arm))
        with self.assertRaises(ValueError):P.expected_work(5,'ROLE')
        with self.assertRaises(ValueError):P.trajectory_root(self.root,'gsm8k','A')

    def test_storage_uses_phase_maximum_and_preserves_reserve(self):
        plan=S.capacity_plan(2097259104)
        self.assertEqual(plan['raw_budget_generations'],8204)
        self.assertEqual(plan['reserve_bytes'],8*2**30)
        self.assertEqual(plan['phases']['training']['state_weight_equivalents'],40)
        self.assertEqual(plan['phases']['evaluation']['state_weight_equivalents'],36)
        self.assertEqual(plan['required_initial_free_bytes'],149072926082)
        self.assertEqual(plan['required_initial_free_bytes'],max(p['required_initial_free_bytes'] for p in plan['phases'].values()))
        self.assertFalse(plan['future_retirement_credit']);self.assertFalse(plan['offload_assumed'])

    def qualification(self):
        files=[]
        for n in range(8):
            path=self.root/f'input{n}';path.write_text(str(n));files.append(str(path))
        prepared=self.root/'prepared.json';write_json(prepared,dict(feature_paths=files[2:]))
        plan=dict(execution_source={'commit':'a'*40},loss_contracts={'fixed':True},
            qualification=dict(config=files[0],initial_state=files[1],prepared=str(prepared)))
        work={}
        for kind,n in [('adam_step',6),('training_response_forward',12),('core_backward',12)]:
            for suffix in ('attempted','completed'):work[kind+'_'+suffix]=n
        receipt=dict(status='passed',execution_commit='a'*40,loss_contracts=plan['loss_contracts'],
            budget=Q.BUDGET,input_states_unchanged=True,cases=[{}]*6,work=work,
            input_identities={p:identity(p) for p in (*files,str(prepared))})
        write_json(self.root/'qualification/QUALIFICATION.json',receipt)
        return plan,receipt,files

    def test_qualification_requires_exact_source_states_and_six_steps(self):
        plan,good,files=self.qualification()
        self.assertEqual(C.qualification_gate(self.root,plan)['status'],'passed')
        for edit in (dict(status='failed'),dict(execution_commit='b'*40),dict(input_states_unchanged=False),
                     dict(cases=[{}]*5),dict(work=dict(good['work'],adam_step_attempted=7)),
                     dict(budget=dict(Q.BUDGET,startup_response_forwards=2))):
            write_json(self.root/'qualification/QUALIFICATION.json',dict(good,**edit))
            with self.assertRaises(ValueError):C.qualification_gate(self.root,plan)
        write_json(self.root/'qualification/QUALIFICATION.json',good)
        Path(files[-1]).write_text('changed')
        with self.assertRaises(ValueError):C.qualification_gate(self.root,plan)

    def test_failed_qualification_cannot_construct_a_backend(self):
        plan=dict(binding='a'*64,datasets={'gsm8k':{'training_manifest':'unused'}},storage={})
        with patch.object(C,'read',return_value=plan),patch.object(C,'read_requests',return_value=[]), \
                patch.object(C,'validate_plan'),patch.object(C,'admit'), \
                patch.object(C,'qualification_gate',side_effect=ValueError('failed qualification')), \
                patch('online_feedback.sglang_backend.SGLangDFlashBackend') as backend:
            with self.assertRaisesRegex(ValueError,'qualification'):C.run_trajectory(self.root,'gsm8k','ROLE')
        backend.assert_not_called()
        result=json.loads((P.trajectory_root(self.root,'gsm8k','ROLE')/'STATUS.json').read_text())
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['actual_work']['attempts'],dict.fromkeys(P.arm_limits('ROLE'),0))

    def test_failed_storage_admission_preserves_exact_receipt(self):
        plan=S.capacity_plan(1000)
        with patch.object(S.native,'compiler_growth',return_value={'compiler_growth_bytes':0}), \
                patch.object(S.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaises(ValueError):S.admit(self.root,plan,receipt_path=self.root/'FAIL.json')
        receipt=json.loads((self.root/'FAIL.json').read_text())
        self.assertEqual(receipt['status'],'failed');self.assertTrue(receipt['no_future_deletion_credit'])
        self.assertEqual(receipt['validated_existing_tensor_bytes'],0)

    def retirement_fixture(self,event):
        base=P.trajectory_root(self.root,'mathqa','ROLE');event_root=base/'events'/f'event_{event:03d}'
        previous=base/'trainer/events'/f'event_{event-1:03d}'/'export';previous.mkdir(parents=True)
        current=base/'trainer/events'/f'event_{event:03d}'/'export';current.mkdir(parents=True)
        boundaries=[]
        for number,folder in [(event-1,previous),(event,current)]:
            state=folder.parent/'training_state.pt';state.write_bytes(b'state'+str(number).encode())
            weights=folder/'model.safetensors';weights.write_bytes(b'weights'+str(number).encode())
            boundaries.append(dict(status='passed',event=number,state=str(state),weights=str(weights),
                state_identity=identity(state),weights_identity=identity(weights)))
        feature=event_root/'loop/features/r.pt';feature.parent.mkdir(parents=True);feature.write_bytes(b'feature')
        records=[dict(record_id='r',checkpoint_id=str(previous))]
        receipt=dict(training=dict(status='completed',event_id=f'event_{event:03d}',checkpoint_path=str(current)),
            preparation=[dict(record_id='r',feature_path=str(feature))])
        write_json(event_root/'loop/events'/f'event_{event:03d}.json',receipt)
        write_json(event_root/'loop/feedback_buffer/state.json',dict(records=records,events=[dict(status='committed',receipt=receipt)]))
        write_json(event_root/'PUBLICATION_CHECK.json',dict(status='passed',all_equal=True,event_id=f'event_{event:03d}',checkpoint_path=str(current)))
        write_json(event_root/'preparation_retirement/r.json',dict(status='completed',retained={'prepared_packet':str(feature)}))
        write_json(current.parent/'RESULT.json',receipt['training'])
        return dict(root=self.root,event_root=event_root,records=records,receipt=receipt,
            previous_export=previous,previous_boundary=boundaries[0],current_boundary=boundaries[1],dataset='mathqa',arm='ROLE')

    def test_event2_export_survives_and_only_own_obsolete_state_retires(self):
        kwargs=self.retirement_fixture(3);previous=kwargs['previous_boundary'];current=kwargs['current_boundary']
        result=S.retire_event(**kwargs)
        self.assertEqual(result['status'],'completed')
        self.assertTrue(Path(previous['weights']).exists());self.assertFalse(Path(previous['state']).exists())
        self.assertTrue(Path(current['weights']).exists());self.assertTrue(Path(current['state']).exists())
        self.assertEqual({r['kind'] for r in result['removed']},{'prepared_feature','previous_state'})

    def test_event1_export_is_retired_after_successor(self):
        kwargs=self.retirement_fixture(2);previous=kwargs['previous_boundary']
        result=S.retire_event(**kwargs)
        self.assertFalse(Path(previous['weights']).exists());self.assertFalse(Path(previous['state']).exists())
        self.assertEqual({r['kind'] for r in result['removed']},{'prepared_feature','previous_state','previous_export'})

    def test_failed_publication_prevents_all_retirement(self):
        kwargs=self.retirement_fixture(2)
        marker=kwargs['event_root']/'PUBLICATION_CHECK.json';row=json.loads(marker.read_text());row['all_equal']=False;write_json(marker,row)
        with self.assertRaises(ValueError):S.retire_event(**kwargs)
        self.assertTrue(Path(kwargs['previous_boundary']['state']).exists())
        self.assertTrue(Path(kwargs['previous_boundary']['weights']).exists())

    def test_evaluation_boundary_matches_raw_audit_schema(self):
        from experiments.role_loss_pilot.evaluation import boundary_errors
        storage=dict(status='passed',free_bytes=100,required_additional_and_reserve_bytes=90)
        quiet=dict(status='passed',training=[],transfers=[],foreign_gpu_processes=[])
        with patch.object(C,'resource_check',return_value=storage), \
                patch('experiments.kplus_full_gsm8k.preflight.timing_barrier',return_value=quiet):
            receipt=C.evaluation_boundary(self.root,{'storage':{}})
        self.assertEqual(boundary_errors(receipt),[])
        self.assertEqual(receipt['storage']['required_bytes'],90)

    def test_selected_checkpoint_identity_must_match_training_boundary(self):
        for arm in P.ARMS:
            base=P.trajectory_root(self.root,'gsm8k',arm)
            export=base/'trainer/events/event_002/export';export.mkdir(parents=True)
            weights=export/'model.safetensors';weights.write_bytes(arm.encode())
            write_json(base/'STATUS.json',dict(status='completed',events_completed=4,event_exports={'2':str(export)}))
            write_json(base/'boundaries/event_002.json',dict(status='passed',dataset='gsm8k',arm=arm,event=2,
                weights=str(weights),weights_identity=identity(weights)))
        checked=C.selected_checkpoints(self.root,'gsm8k',2)
        self.assertEqual(set(checked),set(P.ARMS))
        (Path(checked['ROLE'])/'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'authenticated own boundary'):
            C.selected_checkpoints(self.root,'gsm8k',2)

    def test_timeout_signal_raises_into_cleanup_instead_of_silent_exit(self):
        import signal
        registered={}
        with patch.object(C.signal,'signal',side_effect=lambda number,handler:registered.update({number:handler})):
            C.install_signal_handlers()
        self.assertEqual(set(registered),{signal.SIGTERM,signal.SIGINT})
        for number,handler in registered.items():
            with self.assertRaisesRegex(InterruptedError,'Operational stop signal'):
                handler(number,None)

    def test_launcher_preserves_one_qualification_then_full_admission(self):
        text=Path(C.__file__).with_name('run_campaign.sh').read_text()
        self.assertLess(text.index('preflight "$run_root" --qualification-only'),text.index('qualify "$run_root"'))
        self.assertLess(text.index('qualify "$run_root"'),text.index('preflight "$run_root"\n'))
        self.assertLess(text.index('preflight "$run_root"\n'),text.index('campaign "$run_root"'))
        self.assertIn('30m',text);self.assertIn('48h',text)
        for module in ('test_role_loss_objective','test_kplus_objective','test_adaflash_repo'):
            self.assertIn(module,text)
            self.assertTrue((Path(__file__).parent/(module+'.py')).is_file())


if __name__=='__main__':unittest.main()
