"""CPU task orchestration: own feedback, durable work and complete retention."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.bounded_prefix_allocation import task_runner as T, runtime
from online_feedback.storage import write_json


class TaskTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        runtime.initialize_work(self.root)
        self.requests = [dict(request_id=f'own_{i:03d}', prompt_ids=[1,i+2], request_index=i) for i in range(128)]
        write_json(self.root/'inputs.json',dict(requests=self.requests))
        (self.root/'original.yaml').write_text('native configuration')
        write_json(self.root/'schedule.json',dict(retained=True))
        self.plan = dict(root=str(self.root), binding='d'*64, execution_commit='c'*40,
            devices={'A':'GPU-A','B':'GPU-B'}, origin_state='/retained/K6.pt', origin_export='/retained/K6',
            origin_config=str(self.root/'original.yaml'), configs={'C':str(self.root/'C.yaml')},
            event7_schedule=str(self.root/'schedule.json'), event7_feature_paths=[f'/retained/{i}.pt' for i in range(128)],
            event8_inputs=str(self.root/'inputs.json'), candidate=dict(cap=.25,smoothing=.5),
            storage=dict(atomic_publication_bytes=8,temporary_feature_bytes=1))
        self.events = []

    def save_endpoint(self,event):
        base=self.root/'tasks'/f'C{event}'
        state=base/'trainer/training_state.pt'
        state.parent.mkdir(parents=True,exist_ok=True)
        state.write_bytes(f'own complete C{event}'.encode())
        export=base/'trainer/events'/f'event_{event:03d}'/'export'
        export.mkdir(parents=True,exist_ok=True)
        (export/'model.safetensors').write_bytes(f'weights C{event}'.encode())
        saved=dict(state=str(state),export=str(export),global_step=event*128,presentations=event*256,
                   state_identity={'fixture':event},weights_identity={'fixture':event})
        write_json(base/'RESULT.json',dict(status='completed',arm='C',event=event,
            binding=self.plan['binding'],execution_commit=self.plan['execution_commit'],endpoint=saved))
        return saved

    def backend(self):
        events=self.events
        class Backend:
            failed=False
            loading_s=.25
            worker=SimpleNamespace(pid=123)
            def generate(self,prompt,**kwargs):
                events.append(('generate',kwargs['request_id']))
                assert kwargs['max_new_tokens']==2048 and kwargs['purpose']=='serving'
                record=dict(record_id=kwargs['request_id'],prompt_ids=prompt,response_ids=[5],
                    checkpoint_id=T.endpoint(self_plan,7)['export'])
                write_json(kwargs['save_to'],record)
                return record
            def prepare_feedback(self,record,destination):
                events.append(('prepare',record['record_id']))
                Path(destination).write_bytes(b'durable own packet')
                return dict(status='prepared',feature_path=str(destination))
            def _call(self,command):
                events.append((command,None))
                return dict(generations=128)
            def close(self):
                events.append(('close',None))
        self_plan=self.plan
        return Backend()

    def collection(self):
        base=self.root/'tasks/C_collect'
        base.mkdir(parents=True)
        backend=self.backend()
        def retire(record,prepared,root,event):
            self.assertTrue(Path(prepared['feature_path']).exists())
            self.events.append(('retire_duplicate',record['record_id']))
        with patch.object(T,'create_backend',return_value=backend), patch.object(T,'boundary'), \
                patch('experiments.full_k_gsm8k.campaign.online_record_check'), \
                patch('experiments.kplus_full_gsm8k.storage.retire_redundant_features',side_effect=retire):
            result=T.collect_task(self.plan,'C_collect',base,runtime.WorkLedger(self.root,'C_collect'),'GPU-A')
        return result

    def test_collection_uses_own_C7_and_prepares_every_response_once_before_C8(self):
        own=self.save_endpoint(7)
        result=self.collection()
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['own_checkpoint'],own['export'])
        self.assertEqual(len(T.own_features(self.plan,own)),128)
        work=T.read(self.root/'WORK.json')
        self.assertEqual(work['completed']['feedback_generation'],128)
        self.assertEqual(work['completed']['teacher_preparation'],128)
        self.assertEqual(work['completed']['adam'],0)
        self.assertEqual(work['pending'],{})
        self.assertEqual(self.events[:3],[('generate','own_000'),('prepare','own_000'),('retire_duplicate','own_000')])
        self.assertEqual(self.events[-2:],[('work_receipt',None),('close',None)])
        self.assertTrue(Path(own['state']).exists())

    def test_foreign_or_partial_feedback_fails_before_trainer_initialization(self):
        own=self.save_endpoint(7)
        result=self.collection()
        destination=self.root/'tasks/C_collect/FEATURES.json'
        for field,value in [('status','preparing'),('own_checkpoint','/other/weights'),('generated_responses',127),
                            ('feature_paths',list(reversed(result['feature_paths'])))]:
            wrong=dict(result,**{field:value})
            write_json(destination,wrong)
            with patch('experiments.bounded_prefix_allocation.training.CandidateTrainer') as trainer:
                with self.assertRaises(ValueError):
                    T.train_task(self.plan,'C8',self.root/'unused',runtime.WorkLedger(self.root,'C8'))
                trainer.assert_not_called()
        write_json(destination,result)
        self.assertEqual(T.own_features(self.plan,own),result['feature_paths'])

    def test_invalid_generation_is_durable_and_prevents_teacher_work(self):
        self.save_endpoint(7)
        base=self.root/'tasks/C_collect'
        base.mkdir(parents=True)
        with patch.object(T,'create_backend',return_value=self.backend()), patch.object(T,'boundary'), \
                patch('experiments.full_k_gsm8k.campaign.online_record_check',side_effect=ValueError('invalid trace')):
            with self.assertRaisesRegex(ValueError,'invalid trace'):
                T.collect_task(self.plan,'C_collect',base,runtime.WorkLedger(self.root,'C_collect'),'GPU-A')
        self.assertTrue((base/'records/own_000.json').exists())
        result=T.read(base/'FEATURES.json')
        self.assertEqual((result['status'],result['generated_responses'],result['teacher_preparations']),('failed',1,0))
        self.assertEqual(T.read(self.root/'WORK.json')['completed']['feedback_generation'],1)
        self.assertEqual(self.events[-1],('close',None))

    def fake_trainer(self):
        outer=self
        class Trainer:
            failed=False
            def __init__(self,config,output,**kwargs):
                outer.events.append(('init',kwargs))
                self.output=Path(output)
                self.output.mkdir(parents=True)
                self.latest=self.output/'training_state.pt'
                self.global_step=kwargs['completed_event']*128
                self.presentations=self.global_step*2
                self.core=SimpleNamespace(accumulation_remainder=0)
                self.work={}
                self.strategy=SimpleNamespace(forward_loss=lambda *a,**k:None)
                self.backend=SimpleNamespace(backward=lambda *a,**k:None,
                    optimizer=SimpleNamespace(optimizer=SimpleNamespace(step=lambda *a,**k:None)))
            def _train_passes(self,*args):
                for ordinal in range(256):
                    self.strategy.forward_loss()
                    self.backend.backward()
                    self.presentations+=1
                    if ordinal%2:
                        self.backend.optimizer.optimizer.step()
                        self.global_step+=1
                return []
            def train_event(self,event,features,passes):
                outer.events.append(('train_inputs',list(features)))
                self._train_passes(features,passes)
                self.latest.write_bytes(b'own recovery')
                export=self.output/'events'/event/'export'
                export.mkdir(parents=True)
                (export/'model.safetensors').write_bytes(b'own weights')
                return dict(checkpoint_path=str(export),costs={'training_s':1.})
            def verify_boundary(self,export,receipt):
                write_json(receipt,dict(status='passed'))
                return dict(status='passed')
            def close(self):
                outer.events.append(('trainer_close',None))
        return Trainer

    def test_C8_uses_own_state_and_counts_startup_without_deleting_C7(self):
        self.save_endpoint(7)
        collected=self.collection()
        own=T.endpoint(self.plan,7)
        base=self.root/'tasks/C8'
        base.mkdir()
        def startup(trainer,*args):
            for _ in range(2):
                trainer.strategy.forward_loss()
                trainer.backend.backward()
            return dict(status='completed',optimizer_steps=0,response_forwards=2,response_backwards=2)
        with patch('experiments.bounded_prefix_allocation.training.CandidateTrainer',self.fake_trainer()), \
                patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime',return_value={}), \
                patch('experiments.functional_step_geometry.native_pilot.discarded_startup',side_effect=startup), \
                patch.object(T,'boundary'):
            result=T.train_task(self.plan,'C8',base,runtime.WorkLedger(self.root,'C8'))
        kwargs=next(value for kind,value in self.events if kind=='init')
        self.assertEqual(kwargs['origin_state'],own['state'])
        self.assertEqual(kwargs['origin_export'],own['export'])
        self.assertEqual(kwargs['completed_event'],7)
        self.assertIsNone(kwargs['schedule'])
        self.assertEqual(next(value for kind,value in self.events if kind=='train_inputs'),collected['feature_paths'])
        self.assertEqual(result['endpoint']['global_step'],1024)
        self.assertEqual(result['retained_complete_candidate_boundaries'],[7,8])
        self.assertTrue(Path(own['state']).exists())
        self.assertFalse((base/'RETIREMENT.json').exists())
        counts=T.read(self.root/'WORK.json')['completed']
        self.assertEqual((counts['training_initialization'],counts['adam'],counts['response_forward'],counts['response_backward']),(1,128,258,258))

    def test_partial_startup_failure_keeps_actual_work_and_closes_trainer(self):
        base=self.root/'tasks/C7';base.mkdir(parents=True)
        def startup(trainer,*args):
            trainer.strategy.forward_loss()
            raise RuntimeError('startup fixture failure after one forward')
        with patch('experiments.bounded_prefix_allocation.training.CandidateTrainer',self.fake_trainer()), \
                patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime',return_value={}), \
                patch('experiments.functional_step_geometry.native_pilot.discarded_startup',side_effect=startup):
            with self.assertRaisesRegex(RuntimeError,'startup fixture'):
                T.train_task(self.plan,'C7',base,runtime.WorkLedger(self.root,'C7'))
        work=T.read(self.root/'WORK.json')
        self.assertEqual(work['completed']['response_forward'],1)
        self.assertEqual(work['completed']['response_backward'],0)
        self.assertEqual(work['completed']['adam'],0)
        self.assertEqual(T.read(base/'FAILED.json')['status'],'failed')
        self.assertEqual(self.events[-1],('trainer_close',None))

    def test_wrong_gpu_or_existing_task_never_restarts_work(self):
        write_json(self.root/'plan.json',self.plan)
        with patch.dict('os.environ',CUDA_VISIBLE_DEVICES='GPU-B'):
            with self.assertRaises(ValueError):
                T.run_task(self.root/'plan.json','C7','GPU-A')
        with patch.dict('os.environ',CUDA_VISIBLE_DEVICES='GPU-B'):
            with self.assertRaisesRegex(ValueError,'fixed physical card'):
                T.run_task(self.root/'plan.json','final','GPU-B')
        (self.root/'tasks/C7').mkdir(parents=True)
        with patch.dict('os.environ',CUDA_VISIBLE_DEVICES='GPU-A'),patch.object(T,'train_task') as train:
            with self.assertRaises(FileExistsError):
                T.run_task(self.root/'plan.json','C7','GPU-A')
            train.assert_not_called()

    def test_final_path_uses_fresh_three_arm_work_and_paired_analysis(self):
        from test_persistent_prefix_evaluation import inputs,permutations,references
        from test_full_k_gsm8k_evaluation import FullBackend
        own=self.save_endpoint(8)
        requests=inputs()
        warmups=deepcopy(requests[:2])
        for i,row in enumerate(warmups):
            row.update(request_id=f'warm{i}',source_id=f'old:{i}',prompt_ids=[1,9000+i])
        for key,value in [('data_panel',dict(requests=requests,within_block_permutations=permutations())),
                          ('references',references(requests)),('warmups',dict(requests=warmups)),
                          ('warmup_references',references(warmups))]:
            path=self.root/(key+'.json');write_json(path,value);self.plan[key]=str(path)
        self.plan['controls']={'N':{'export':'/N'},'S':{'export':'/S'}}
        backend=FullBackend(requests)
        backend.failed=False;backend.loading_s=.5;backend.worker=SimpleNamespace(pid=123)
        backend._call=lambda command: {'command':command}
        backend.close=lambda:None
        base=self.root/'tasks/final';base.mkdir()
        with patch.object(T,'create_backend',return_value=backend),patch.object(T,'boundary'), \
                patch('experiments.kplus_full_gsm8k.preflight.timing_barrier') as quiet:
            result=T.final_task(self.plan,base,runtime.WorkLedger(self.root,'final'),'GPU-A')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['checkpoints'],{'N':'/N','C':own['export'],'S':'/S'})
        counts=T.read(self.root/'WORK.json')['completed']
        self.assertEqual((counts['scored_generation'],counts['warmup_generation']),(1152,72))
        self.assertEqual(counts['feedback_generation'],0)
        self.assertEqual(quiet.call_count,37)  #36 activations plus the final quiet boundary.
        summary=T.read(base/'evaluation/SUMMARY.json')
        self.assertEqual(summary['request_count'],128)
        self.assertEqual(set(summary['method_totals']),{'N','C','S'})
        self.assertEqual(len((base/'evaluation/ALL_REPEATS.csv').read_text().splitlines()),1153)


if __name__=='__main__':
    unittest.main()
