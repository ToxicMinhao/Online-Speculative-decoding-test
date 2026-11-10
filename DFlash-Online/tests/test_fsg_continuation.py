"""Input-only selection and exact additive continuation contracts; CPU only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.functional_step_geometry import contract as C, data as D, training as T
from experiments.role_loss_pilot import training as old
from online_feedback.rng import capture_rng
from test_role_loss_campaign_training import pilot_state
from test_full_k_training import BINDING
import test_full_k_training as tiny_fixture


NEW_BINDING = 'b'*64


def continuation(event=4, arm='FSG', dataset='gsm8k'):
    origin, original_args = pilot_state(4, 'KPLUS', dataset)
    state, _ = pilot_state(event, 'KPLUS', dataset)
    state['events'][:4] = deepcopy(origin['events'])
    state['algorithm_contract'] = T.continuation_contract(origin, binding=NEW_BINDING, arm=arm, dataset=dataset)
    for receipt in state['events'][4:]:
        receipt.pop('pilot_loss')
        receipt.update(method=arm, dataset=dataset, kplus_loss=T.kplus_contract())
    args = dict(origin=origin, binding=NEW_BINDING, arm=arm, dataset=dataset,
        config_text=state['config_text'], original_config_text=original_args['original_config_text'],
        completed_event=event, expected_recipe=origin['backend']['optimizer']['recipe'],
        expected_cuda_rng_states=0)
    return origin, state, args


def input_row(index, dataset='gsm8k'):
    return dict(dataset=dataset, source_id=f'{dataset}:train:{index}',
        request_id=f'train_{index:05d}', prompt_ids=[1,index+10,2],
        normalized_stem=f'question {index}', rendered_prompt_sha256=f'rendered{index}',
        prompt_ids_sha256=f'tokens{index}', collect_feedback=True, max_new_tokens=2048,
        event_id=f'event_{index//128+1:03d}')


class ContinuationTests(unittest.TestCase):
    def test_all_boundaries_and_arms_preserve_original_history(self):
        for arm in C.ARMS:
            for dataset in C.DATASETS:
                for event in (4,5,6):
                    origin,state,args = continuation(event,arm,dataset)
                    original, current, rng = deepcopy(origin), deepcopy(state), capture_rng()
                    checked = T.validate_snapshot(state, **args)
                    self.assertEqual(checked['global_step'],128*event)
                    self.assertFalse(T.differences(origin,original))
                    self.assertFalse(T.differences(state,current))
                    self.assertFalse(T.differences(rng,capture_rng()))

    def test_original_pilot_guards_stay_closed(self):
        origin,state,args = continuation(5)
        old_args = dict(arm='KPLUS',dataset='gsm8k',binding=BINDING,
            config_text=origin['config_text'],original_config_text=args['original_config_text'],
            expected_contract=origin['algorithm_contract'],expected_recipe=args['expected_recipe'],
            expected_cuda_rng_states=0)
        old.validate_snapshot(origin,completed_event=4,**old_args)
        with self.assertRaises(ValueError): old.validate_snapshot(state,completed_event=5,**old_args)
        with self.assertRaises(ValueError): T.validate_snapshot(state,**dict(args,completed_event=7))

    def test_source_migration_changes_no_tensor_rng_optimizer_or_receipt(self):
        origin,state,args = continuation()
        candidate = T.migrate_origin(origin,binding=NEW_BINDING,arm='FSG',dataset='gsm8k',
            config_text=origin['config_text'],run_id='new_output_identity',
            original_config_text=args['original_config_text'],expected_recipe=args['expected_recipe'],
            expected_cuda_rng_states=0)
        for name in ('draft_state_dict','backend','rng','events'):
            self.assertIs(candidate[name],origin[name])
        self.assertEqual(candidate['global_step'],512)
        self.assertEqual(candidate['presentations'],1024)
        self.assertEqual(candidate['algorithm_contract'][old.CONTRACT_KEY],origin['algorithm_contract'][old.CONTRACT_KEY])

    def test_foreign_origin_missing_moments_and_changed_history_reject(self):
        origin,state,args = continuation(5)
        for mutation in (
            lambda s:s['events'][0].update(method='ROLE'),
            lambda s:s['algorithm_contract'][T.CONTRACT_KEY].update(arm='SCALAR'),
            lambda s:s['backend']['optimizer']['optimizer_state_dict']['state'].clear(),
            lambda s:s.update(global_step=512),
        ):
            bad=deepcopy(state);mutation(bad)
            with self.assertRaises(ValueError):T.validate_snapshot(bad,**args)
        with self.assertRaisesRegex(ValueError,'dataset'):
            T.validate_origin(origin,dataset='mathqa',original_config_text=args['original_config_text'],expected_cuda_rng_states=0)
        with self.assertRaises(ValueError):
            T.validate_origin(state,dataset='gsm8k',original_config_text=args['original_config_text'],expected_cuda_rng_states=0)
        _,migrated,migrated_args=continuation(4)
        with self.assertRaisesRegex(ValueError,'PR32 KPLUS4'):
            T.validate_origin(migrated,dataset='gsm8k',
                original_config_text=migrated_args['original_config_text'],expected_cuda_rng_states=0)

    def test_budget_accounts_only_new_events_and_separate_diagnostics(self):
        self.assertEqual(C.expected_work('KPLUS',4)['optimizer_steps'],0)
        self.assertEqual(C.expected_work('FSG',6)['ordinary_presentations'],512)
        self.assertEqual(C.expected_work('SCALAR',6)['auxiliary_forward_maximum'],1536)
        self.assertEqual(C.MAIN_BUDGET['total_forwards'],3072+6144)
        self.assertEqual(C.MAIN_BUDGET['total_backwards'],3072+2048)
        self.assertEqual(C.MAIN_BUDGET['endpoint_generations'],2*256*3*2)
        self.assertEqual(C.DIAGNOSTIC_BUDGET['response_forwards'],16*40)
        self.assertEqual(C.QUALIFICATION_BUDGET['optimizer_steps'],8)

    def test_production_rejects_neutral_wrong_arm_or_closed_hook(self):
        trainer=object.__new__(T.FunctionalGeometryTrainer)
        trainer.events=[{}]*4
        trainer.arm='FSG'
        for arm,strength,closed in [('FSG',0.,False),('SCALAR',1.,False),('FSG',1.,True)]:
            trainer.functional_geometry=SimpleNamespace(arm=arm,strength=strength,closed=closed)
            with self.assertRaisesRegex(ValueError,'fixed strength 1'):
                trainer._prepare_event('event_005',[],2)
        trainer.arm='KPLUS'
        with self.assertRaisesRegex(ValueError,'no geometry hook'):
            trainer._prepare_event('event_005',[],2)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.full=[input_row(i) for i in range(768)]
        self.prefix=deepcopy(self.full[:512])
        self.panel=[input_row(1000)]

    def test_exact_next_windows_without_reordering(self):
        selected=D.validate_continuation(self.full,self.prefix,self.panel,dataset='gsm8k')
        self.assertEqual(selected,self.full[512:])
        self.assertEqual([r['event_id'] for r in selected],['event_005']*128+['event_006']*128)
        self.assertIsNot(selected[0],self.full[512])

    def test_original_prefix_token_order_and_normalization_must_match(self):
        for key,value in [('source_id','different'),('prompt_ids',[3]),('normalized_stem','changed')]:
            full=deepcopy(self.full);full[1][key]=value
            with self.assertRaisesRegex(ValueError,'exact PR32 prefix'):
                D.validate_continuation(full,self.prefix,self.panel,dataset='gsm8k')

    def test_overlap_is_failure_not_replacement(self):
        for key in ('source_id','normalized_stem','rendered_prompt_sha256','prompt_ids_sha256'):
            full=deepcopy(self.full);full[513][key]=self.prefix[0][key]
            with self.assertRaisesRegex(ValueError,'overlaps'):
                D.validate_continuation(full,self.prefix,self.panel,dataset='gsm8k')
        with self.assertRaisesRegex(ValueError,'overlaps'):
            D.validate_continuation(self.full,self.prefix,[self.full[600]],dataset='gsm8k')
        full=deepcopy(self.full);full[600]['normalized_stem']=full[599]['normalized_stem']
        with self.assertRaisesRegex(ValueError,'overlaps'):
            D.validate_continuation(full,self.prefix,self.panel,dataset='gsm8k')

    def test_shortfall_and_wrong_event_cap_reject(self):
        with self.assertRaises(ValueError):D.validate_continuation(self.full[:-1],self.prefix,self.panel,dataset='gsm8k')
        for key,value in [('event_id','event_001'),('max_new_tokens',1024),('collect_feedback',False)]:
            full=deepcopy(self.full);full[512][key]=value
            with self.assertRaises(ValueError):D.validate_continuation(full,self.prefix,self.panel,dataset='gsm8k')

    def test_shared_feature_bytes_not_only_answer_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);features=[];canonicals=[]
            for index in range(128):
                feature=root/f'{index}.pt';feature.write_bytes(bytes([index]));features.append(feature)
                canonical=root/f'{index}.json';canonical.write_text(json.dumps(dict(
                    record_id=f'r{index}',session_id='feedback',request_index=index,
                    checkpoint_id='retained_KPLUS4',purpose='serving',prompt_ids=[1,2],
                    response_ids=[7],finish_reason='stop',stop_token_ids=[7],max_new_tokens=2048,
                    metadata=dict(verification_calls=1,trace=[]))))
                canonicals.append(canonical)
            sealed=D.seal_shared_packets(features,canonicals,dataset='gsm8k',event=5,output=root/'SEALED.json')
            self.assertTrue(D.verify_shared_packets(sealed)['byte_identical'])
            features[0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):D.verify_shared_packets(sealed)
            with self.assertRaisesRegex(ValueError,'immutable'):
                D.seal_shared_packets(features,canonicals,dataset='gsm8k',event=5,output=root/'SEALED.json')


class TinyContinuationTests(unittest.TestCase):
    """Synthetic event-4 history around a real tiny CPU K+ runtime, not GPU qualification."""
    @classmethod
    def setUpClass(cls):
        if os.environ.get('FSG_TINY_CPU_CHILD')=='1':
            tiny_fixture.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('FSG_TINY_CPU_CHILD')=='1':
            tiny_fixture.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('FSG_TINY_CPU_CHILD')=='1':
            tiny_fixture.FullKTinyIntegrationTests.setUp(self)

    def run_in_cpu_child(self):
        if os.environ.get('FSG_TINY_CPU_CHILD')!='1':
            root=Path(__file__).resolve().parents[1]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',FSG_TINY_CPU_CHILD='1')
            env['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('.','tests','training','inference'))
            result=subprocess.run([sys.executable,'-m','unittest','-v',
                __name__+'.TinyContinuationTests.'+self._testMethodName],cwd=root,env=env,
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return True
        return False

    def build_origin(self):
        from safetensors.torch import save_file
        from experiments.full_k_gsm8k.training import live_state
        source=old.RoleLossTrainer(self.config,self.root/'source',binding=BINDING,
            arm='KPLUS',dataset='gsm8k',original_config_text=self.original_config_text,
            expected_cuda_rng_states=0)
        self.addCleanup(source.close)
        state=deepcopy(live_state(source))
        history,_=pilot_state(4,'KPLUS','gsm8k')
        state.update(global_step=512,presentations=1024,events=history['events'])
        saved=state['backend']['optimizer']['optimizer_state_dict']
        names=state['backend']['optimizer']['recipe']['parameters']
        for parameter_id,(name,_,_) in zip(saved['param_groups'][0]['params'],names):
            weight=state['draft_state_dict'][name]
            saved['state'][parameter_id]=dict(step=torch.tensor(512.),
                exp_avg=torch.zeros_like(weight),exp_avg_sq=torch.zeros_like(weight))
        origin=self.root/'origin.pt';torch.save(state,origin)
        export=self.root/'origin_export';export.mkdir()
        save_file(state['draft_state_dict'],export/'model.safetensors')
        source.close()
        kwargs=dict(arm='KPLUS',dataset='gsm8k',binding=NEW_BINDING,
            original_config_text=self.original_config_text,origin_state=origin,
            origin_export=export,expected_cuda_rng_states=0)
        return state,kwargs

    def test_complete_event5_and_own_recovery_preserve_old_history(self):
        if self.run_in_cpu_child(): return
        state,kwargs=self.build_origin()
        trainer=T.FunctionalGeometryTrainer(self.config,self.root/'new',**kwargs)
        self.addCleanup(trainer.close)
        self.assertEqual(trainer.global_step,512)
        self.assertFalse(T.differences(trainer.events,state['events']))
        receipt=trainer.train_event('event_005',self.features)
        self.assertEqual(receipt['end_step'],640)
        self.assertEqual(receipt['objective_work']['reference_operations'],0)
        trainer.verify_boundary(receipt['export_path'])
        recovery=trainer.latest
        trainer.close()
        resumed=T.FunctionalGeometryTrainer(self.config,self.root/'resumed',
            resume_state=recovery,completed_event=5,**kwargs)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.global_step,640)
        self.assertFalse(T.differences(resumed.events[:4],state['events']))
        with self.assertRaisesRegex(ValueError,'events 5 and 6'):
            resumed.events.extend([{},{}])
            resumed._prepare_event('event_007',self.features,2)
        resumed.close()
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_step_keeps_origin_immutable_across_repeated_restore(self):
        if self.run_in_cpu_child(): return
        from specforge.training.strategies.base import StepContext
        from experiments.full_k_gsm8k.training import live_state
        from experiments.ok_coding.storage import identity
        state,kwargs=self.build_origin()
        trainer=T.FunctionalGeometryTrainer(self.config,self.root/'new',**kwargs)
        self.addCleanup(trainer.close)
        retained=deepcopy(trainer.origin)
        disk_identity=identity(trainer.origin_path)
        endpoints=[]
        for attempt in range(2):
            trainer._restore(trainer.origin_path)
            self.assertFalse(T.differences(trainer.origin,retained))
            origin_adam=trainer.origin['backend']['optimizer']['optimizer_state_dict']['state']
            live_adam=trainer.backend.optimizer.state_dict()['optimizer_state_dict']['state']
            for parameter_id,moments in origin_adam.items():
                for name in ('step','exp_avg','exp_avg_sq'):
                    self.assertNotEqual(live_adam[parameter_id][name].data_ptr(),moments[name].data_ptr())
            for name,parameter in trainer.bundle.draft_model.named_parameters():
                self.assertNotEqual(parameter.data_ptr(),trainer.origin['draft_state_dict'][name].data_ptr())
            trainer._active_event=f'disposable_repeat_{attempt}'
            (trainer.events_root/trainer._active_event).mkdir()
            trainer.bundle.model.begin_event()
            for path in self.features[:2]:
                raw=torch.load(path,map_location='cpu',weights_only=True)
                result=trainer.core.train_step(trainer._batch(raw),StepContext(
                    global_step=trainer.global_step,total_steps=trainer.horizon,
                    collect_detailed_metrics=True))
                trainer.presentations+=1
                trainer.global_step+=int(result.optimizer_stepped)
            self.assertEqual(trainer.global_step,513)
            self.assertEqual(trainer.presentations,1026)
            for moments in trainer.backend.optimizer.state_dict()['optimizer_state_dict']['state'].values():
                self.assertEqual(moments['step'].item(),513)
            # Compare the whole original payload: weights, moments, counters,
            # recipe, RNG, contract, event history and metadata remain frozen.
            self.assertFalse(T.differences(trainer.origin,retained))
            self.assertEqual(identity(trainer.origin_path),disk_identity)
            endpoints.append(deepcopy(live_state(trainer)))
        self.assertFalse(T.differences(endpoints[0],endpoints[1]))
        trainer._restore(trainer.origin_path)
        self.assertEqual(trainer.global_step,512)
        self.assertEqual(trainer.presentations,1024)
        self.assertFalse(T.differences(trainer.origin,retained))
        self.assertTrue(T.native.verify_live(trainer,T.migrate_origin(
            retained,binding=NEW_BINDING,arm='KPLUS',dataset='gsm8k',
            config_text=trainer.config_text,run_id=trainer.cfg.run_id,
            original_config_text=self.original_config_text,
            expected_recipe=trainer.backend.optimizer.recipe,
            expected_cuda_rng_states=0))['exact_resume'])
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
