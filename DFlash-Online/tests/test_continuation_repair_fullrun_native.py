"""CPU checks for the full-run wrapper; no native model or CUDA construction."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.continuation_repair_fullrun import training,qualification,witness
from experiments.continuation_repair_fullrun.contract import experiment_contract
from experiments.ok_continuation.resume import differences
from test_continuation_repair import reduce_fixture


class FullRunNativeTests(unittest.TestCase):
    def tearDown(self):self.assertFalse(torch.cuda.is_initialized())

    def observation(self):
        return dict(loss=3.,ratio_metrics={'forward_kl':[8.,4.],
            'repair_pair_auxiliary':[4.,4.],'objective_loss':[12.,4.]},sum_metrics={})

    def test_effective_components_count_both_corrections(self):
        row=self.observation();row['sum_metrics'].update(continuation_fk_correction=.5,continuation_pair_correction=-.25)
        row['ratio_metrics']['objective_loss'][0]=12.25
        result=training.component_logging(row)
        self.assertEqual(result['effective_FKL_numerator'],8.5)
        self.assertEqual(result['effective_PAIR_numerator'],3.75)
        self.assertEqual(result['effective_total'],12.25/4)
        self.assertEqual(result['reconciliation_error'],0.)
        self.assertEqual(row['ratio_metrics']['forward_kl'],[8.,4.])

    def test_wrong_component_and_denominator_are_rejected(self):
        row=self.observation();row['ratio_metrics']['objective_loss'][0]=13.
        with self.assertRaisesRegex(ValueError,'reconcile'):training.component_logging(row)
        row=self.observation();row['ratio_metrics']['repair_pair_auxiliary'][1]=8.
        with self.assertRaisesRegex(ValueError,'denominator'):training.component_logging(row)

    def test_scalar_cancellation_does_not_hide_component_movement(self):
        row=self.observation();row['sum_metrics'].update(continuation_fk_correction=.5,continuation_pair_correction=-.5)
        result=training.component_logging(row)
        self.assertEqual(result['effective_total_numerator'],12.)
        self.assertEqual(result['effective_FKL_numerator'],8.5)
        self.assertEqual(result['effective_PAIR_numerator'],3.5)

    def test_original_tensor_summary_serializes_in_wrapper(self):
        _,_,summary=reduce_fixture(torch.ones(6),torch.tensor([1,1,0,0,0,0],dtype=torch.bool),
            torch.tensor([0.,1.,0.,0.,0.,0.]),torch.arange(6.).requires_grad_(),torch.ones(6))
        row=self.observation()
        trainer=SimpleNamespace(bundle=SimpleNamespace(model=SimpleNamespace(
            prefix_allocation=SimpleNamespace(last_summary={'checked':True}),
            continuation_repair=SimpleNamespace(last_summary=summary))))
        with patch.object(training.native.FullKTrainer,'_append_observation'):
            training.FullContinuationTrainer._append_observation(trainer,'RESPONSE_OBSERVATIONS.jsonl',row)
        json.dumps(row,allow_nan=False)
        self.assertEqual(row['continuation_repair']['continuation_changed_high_positions'],2.)

    def test_initial_witness_comparison_allows_only_declared_arm(self):
        def init(arm):
            return dict(schema=training.WITNESS_SCHEMA,global_step=0,presentations=0,events=[],
                backend={'optimizer':{'optimizer_state_dict':{'state':{}}}},
                algorithm_contract={training.CONTRACT_KEY:experiment_contract(arm,'a'*64),'native':True},
                config_text=f'run_id: {arm}\noutput_dir: /{arm}\ntraining: {{seed: 17}}\n',
                weights={'w':'same'},rng={'seed':17},reduction='response_mean',accumulation_steps=2,horizon=7424)
        first,second=init('BASE'),init('TAIL')
        self.assertEqual(training.compare_initial_witnesses(first,second)['status'],'passed')
        second['rng']['seed']=18
        with self.assertRaisesRegex(ValueError,'scientific state'):training.compare_initial_witnesses(first,second)
        second=init('TAIL');second['backend']['optimizer']['optimizer_state_dict']['state']={'w':1}
        with self.assertRaisesRegex(ValueError,'fresh'):training.compare_initial_witnesses(first,second)

    def test_production_refuses_disposable_parent_api(self):
        trainer=SimpleNamespace(qualification_only=False)
        with self.assertRaisesRegex(ValueError,'qualification-only'):
            training.FullContinuationTrainer.import_qualification_parent(trainer,{}, {'status':'passed'})
        with self.assertRaisesRegex(ValueError,'cannot resume production'):
            training.FullContinuationTrainer.restore_qualification(trainer,'nonexistent')
        with self.assertRaisesRegex(ValueError,'cannot train production'):
            training.FullContinuationTrainer.run_pair(trainer,[])

    def test_scientific_comparison_excludes_only_added_logging(self):
        case=dict(rows=[dict(response_observation=dict(event_id='disposable',loss=2.,effective_components={'new':3}))],
                  gradient={'weight':'a'},endpoint={'rng':'x'},gradient_norm=4.)
        old=deepcopy(case);old['rows'][0]['response_observation'].pop('effective_components')
        self.assertEqual(qualification.scientific_case(case),qualification.scientific_case(old))
        old['endpoint']['rng']='changed'
        self.assertNotEqual(qualification.scientific_case(case),qualification.scientific_case(old))

    def test_witness_state_stamp_catches_inplace_gradient_and_moment_changes(self):
        model=torch.nn.Linear(2,2,bias=False);p=next(model.parameters());p.grad=torch.ones_like(p)
        optimizer=torch.optim.AdamW(model.parameters());optimizer.state[p]={'exp_avg':torch.zeros_like(p)}
        trainer=SimpleNamespace(bundle=SimpleNamespace(model=model),backend=SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer)),global_step=0,presentations=0,events=[],
            core=SimpleNamespace(_micro=0))
        before=witness.state_stamp(trainer);p.grad.add_(1.)
        self.assertTrue(differences(before,witness.state_stamp(trainer)))
        before=witness.state_stamp(trainer);optimizer.state[p]['exp_avg'].add_(1.)
        self.assertTrue(differences(before,witness.state_stamp(trainer)))

    def test_old_warm_runner_keeps_its_event_two_guard(self):
        from experiments.continuation_repair_pilot.native import WarmContinuationTrainer
        trainer=SimpleNamespace(qualification_only=False,global_step=0,presentations=0,events=[])
        with self.assertRaisesRegex(ValueError,'warm event'):
            WarmContinuationTrainer._prepare_event(trainer,'event_001',['x']*128,2)

    def test_sealed_witness_uses_input_manifest_locator(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'WITNESS_SELECTION.json'
            path.write_text(json.dumps({'responses':[{'request_id':f'r{i}'} for i in range(8)]}))
            self.assertEqual(witness.sealed_selection_ids({'witness_selection':str(path)}),[f'r{i}' for i in range(8)])

    def test_disposable_work_counts_real_startup_and_core_calls(self):
        from experiments.continuation_repair_fullrun.runtime import initialize,WorkLedger,read
        with tempfile.TemporaryDirectory() as directory:
            initialize(directory);ledger=WorkLedger(directory,'fixture','qualification')
            trainer=SimpleNamespace(_active_event='disposable_startup',author_strategy=SimpleNamespace(
                forward_loss=lambda value:value+1),backend=SimpleNamespace(backward=lambda value:value*2))
            qualification.install_work_accounting(trainer,ledger)
            self.assertEqual(trainer.author_strategy.forward_loss(2),3)
            self.assertEqual(trainer.backend.backward(2),4)
            trainer._active_event='qualification_window'
            trainer.author_strategy.forward_loss(2);trainer.backend.backward(2)
            counts=read(Path(directory)/'WORK.json')['completed']
            self.assertEqual(counts,{'startup.response_forward':1,'startup.response_backward':1,
                'qualification.response_forward':1,'qualification.response_backward':1})

    def test_witness_close_failure_preserves_cleanup_and_receipt(self):
        from experiments.continuation_repair_fullrun.runtime import initialize
        model=torch.nn.Linear(2,2,bias=False);optimizer=torch.optim.AdamW(model.parameters())
        model.prefix_allocation=SimpleNamespace(last_summary={'original':1})
        trainer=SimpleNamespace(arm='BASE',bundle=SimpleNamespace(model=model),backend=SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer)),global_step=0,presentations=0,events=[],
            core=SimpleNamespace(_micro=0,accumulation_remainder=0),work={'draft_forwards':2},
            _anchor_observations=[{'original':1}],cfg=SimpleNamespace(data=SimpleNamespace(max_length=4096)))
        class BrokenClose:
            def __init__(self,*args,**kwargs):self.work={'response_reads_completed':0}
            def close(self):
                trainer.work['draft_forwards']+=1
                trainer._anchor_observations.append({'diagnostic':1})
                model.prefix_allocation.last_summary={'diagnostic':1}
                raise RuntimeError('close failure')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);initialize(root);(root/'witness').mkdir()
            (root/'witness/BINDINGS.json').write_text(json.dumps({'features':[]}))
            with patch.object(witness,'ReadOnlyEndpoint',BrokenClose),self.assertRaisesRegex(RuntimeError,'close failure'):
                witness.read_witness(trainer,{'root':str(root)})
            receipt=json.loads((root/'witness/released/STATUS.json').read_text())
            self.assertEqual(receipt['status'],'failed')
            self.assertEqual(trainer.work,{'draft_forwards':2})
            self.assertEqual(trainer._anchor_observations,[{'original':1}])
            self.assertEqual(model.prefix_allocation.last_summary,{'original':1})


class FullRunTinyIntegrationTests(unittest.TestCase):
    """Native five-layer tiny BF16 computation; CPU eager attention fixture only."""
    @classmethod
    def setUpClass(cls):
        from test_full_k_training import FullKTinyIntegrationTests
        FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        from test_full_k_training import FullKTinyIntegrationTests
        FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        from test_full_k_training import FullKTinyIntegrationTests
        import yaml
        FullKTinyIntegrationTests.setUp(self)
        raw=yaml.safe_load(self.config.read_text());raw['data']['max_length']=4096
        self.config.write_text(yaml.safe_dump(raw));self.original_config_text=self.config.read_text()

    def build(self,name,arm,**kwargs):
        trainer=training.FullContinuationTrainer(self.config,self.root/name,arm=arm,binding='a'*64,
            original_config_text=self.original_config_text,expected_cuda_rng_states=0,**kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_fresh_flags_complete_native_event_and_exact_production_restore(self):
        from safetensors.torch import save_file
        first=None
        for arm in ('BASE','TAIL'):
            trainer=self.build(arm,arm)
            released=self.root/'released';released.mkdir(exist_ok=True)
            if first is None:save_file(dict(trainer.bundle.draft_model.state_dict()),released/'model.safetensors')
            initial=trainer.save_initial(first)
            if first is None:first=initial['witness_path']
            event=trainer.train_event('event_001',self.features)
            self.assertEqual((trainer.global_step,trainer.presentations,len(trainer.events)),(128,256,1))
            self.assertEqual(event['objective_work']['work']['response_forwards'],256)
            self.assertEqual(event['objective_work']['work']['backwards_completed'],256)
            trainer.verify_boundary(event['export_path'])
            state=torch.load(trainer.latest,weights_only=False)
            trainer.close()
            restored=self.build(arm+'_restored',arm,resume_state=trainer.latest,completed_event=1)
            self.assertTrue(training.native.verify_live(restored,state)['exact_resume'])
            self.assertEqual(bool(restored.bundle.model.continuation_repair),arm=='TAIL')
            self.assertTrue(all('effective_components' in row['response_observation'] for row in event['records']))
        self.assertFalse(torch.cuda.is_initialized())

    def test_typed_disposable_state_restores_without_fabricated_event(self):
        from experiments.ok_continuation.test_resume import fixture
        trainer=self.build('disposable','TAIL',qualification_only=True)
        state=fixture('osd_first_error',1)
        state['draft_state_dict']=dict(deepcopy(trainer.bundle.draft_model.state_dict()))
        # This test creates a tiny authentic-sized Adam mapping; no optimizer call.
        optimizer=trainer.backend.optimizer.optimizer
        for parameter in trainer.bundle.draft_model.parameters():
            optimizer.state[parameter]=dict(step=torch.tensor(128.),exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
        state['backend']={'optimizer':deepcopy(trainer.backend.optimizer.state_dict())}
        trainer.import_qualification_parent(state,{'status':'passed','global_step':128,'fixture':'CPU serializer'})
        trainer._active_event='qualification_window'
        trainer.bundle.model.begin_event()
        before=deepcopy(training.native.live_state(trainer));path=self.root/'disposable.pt'
        trainer.save_qualification(path)
        with torch.no_grad():next(trainer.bundle.draft_model.parameters()).add_(1.)
        result=trainer.restore_qualification(path)
        self.assertTrue(result['exact_resume'])
        self.assertFalse(differences(before,training.native.live_state(trainer)))
        saved=torch.load(path,weights_only=False)
        self.assertEqual(len(saved['events']),1)
        self.assertFalse(saved['qualification_state']['production_recovery'])
        bad=deepcopy(saved);bad['qualification_state']['arm']='BASE';torch.save(bad,self.root/'foreign.pt')
        with self.assertRaisesRegex(ValueError,'Foreign'):
            trainer.restore_qualification(self.root/'foreign.pt')

    def test_released_witness_native_geometry_and_training_state_restoration(self):
        from experiments.continuation_repair_fullrun.runtime import initialize,read
        from online_feedback.rng import capture_rng
        trainer=self.build('witness','BASE')
        root=self.root/'campaign';root.mkdir();initialize(root)
        requests=[{'request_id':torch.load(path,weights_only=True)['record_id']} for path in self.features[:8]]
        selection=root/'WITNESS_SELECTION.json';selection.write_text(json.dumps({'responses':requests}))
        plan={'root':str(root),'witness_selection':str(selection)}
        before=witness.state_stamp(trainer)
        bindings=witness.seal_witness(trainer,plan,self.features)
        self.assertFalse(differences(before,witness.state_stamp(trainer)))
        self.assertTrue(all(len(row['anchors'])==min(8,row['available_anchors']) for row in bindings['features']))
        result=witness.read_witness(trainer,plan)
        self.assertEqual(result['status'],'completed')
        self.assertTrue(result['training_state_unchanged'])
        self.assertFalse(differences(before,witness.state_stamp(trainer)))
        self.assertEqual(result['work']['draft_forwards_completed'],8)
        self.assertEqual(read(root/'WORK.json')['completed']['production.witness_response_forward'],8)
        self.assertEqual(trainer.global_step,0)
        self.assertEqual(trainer.backend.optimizer.optimizer.state,{})


if __name__=='__main__':unittest.main()
