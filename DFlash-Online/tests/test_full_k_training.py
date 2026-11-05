"""Full-horizon contracts and tiny native K continuity; CPU only."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from experiments.full_k_gsm8k import training as T
from experiments.ok_continuation.test_resume import fixture
from online_feedback.rng import capture_rng, restore_rng
from test_ok_coding_training import snapshot, configuration

BINDING='a'*64


def own_state(arm='K',event=0):
    value=snapshot(17,'K') if event==0 else fixture('osd_first_error',event)
    original=configuration(17)
    config=yaml.safe_load(original)
    config['training']['learning_rate']=T.LEARNING_RATES[arm]
    value['config_text']=yaml.safe_dump(config)
    value['algorithm_contract']={
        'online_target_objective':{'name':'osd_first_error'},
        'dflash_target_layer_ids':(1,15,29),
        T.CONTRACT_KEY:T.experiment_contract(arm,BINDING)}
    optimizer=value['backend']['optimizer']
    optimizer['recipe']['optimizer_options']['lr']=T.LEARNING_RATES[arm]
    optimizer['optimizer_state_dict']['param_groups'][0]['lr']=T.LEARNING_RATES[arm]
    for number,entry in enumerate(value['events'],1):
        for row in entry['features']:
            row['record_id']=f'e{number}_'+row['record_id']
        for row in entry['records']:
            row['record_id']=f'e{number}_'+row['record_id']
            row['learning_rate']=T.LEARNING_RATES[arm] if row['optimizer_stepped'] else None
            row['metrics']={'loss':1.0}
    kwargs=dict(arm=arm,binding=BINDING,config_text=value['config_text'],
        original_config_text=original,completed_event=event,
        expected_contract=json.loads(json.dumps(value['algorithm_contract'])),
        expected_recipe=deepcopy(optimizer['recipe']),expected_cuda_rng_states=0)
    return value,kwargs


class FullKStateTests(unittest.TestCase):
    def setUp(self):
        rng=capture_rng()
        self.addCleanup(lambda:restore_rng(rng))

    def test_real_zero_and_full_horizon_boundaries_for_both_arms(self):
        for arm in T.LEARNING_RATES:
            for event in (0,1,14,29,44,58):
                value,kwargs=own_state(arm,event)
                before=capture_rng()
                result=T.validate_snapshot(value,**kwargs)
                self.assertEqual(result['global_step'],128*event)
                self.assertEqual(result['presentations'],256*event)
                self.assertEqual(result['adam_state_entries'],58 if event else 0)
                self.assertFalse(T.differences(before,capture_rng()))

    def test_strict_configuration_allows_only_lr_and_output_metadata(self):
        value,kwargs=own_state('K_HALF')
        T.validate_config_text(value['config_text'],'K_HALF',kwargs['original_config_text'])
        for section,key,changed in [('training','seed',29),('training','num_anchors',256),
                ('online_feedback','weight_decay',0.),('online_feedback','max_events',4),
                ('data','max_length',2048),('model','draft_checkpoint_path','/adapted')]:
            config=yaml.safe_load(value['config_text']);config[section][key]=changed
            with self.subTest(key=key),self.assertRaises(ValueError):
                T.validate_config_text(yaml.safe_dump(config),'K_HALF',kwargs['original_config_text'])

    def test_foreign_objective_arm_binding_and_extra_state_fail(self):
        value,kwargs=own_state('K',1)
        mutations=[lambda x:x['algorithm_contract']['online_target_objective'].update(name='osd'),
            lambda x:x['algorithm_contract'][T.CONTRACT_KEY].update(arm='K_HALF'),
            lambda x:x['algorithm_contract'].update(prefix_guard={}),
            lambda x:x.update(controller={}),lambda x:x.update(presentations=255),
            lambda x:x['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=1.5e-4),
            lambda x:x['rng'].pop('numpy')]
        for mutation in mutations:
            bad=deepcopy(value);mutation(bad)
            with self.assertRaises(ValueError):T.validate_snapshot(bad,**kwargs)

    def test_incomplete_moments_nonfinite_weights_and_parameter_order_fail(self):
        value,kwargs=own_state('K_HALF',1)
        bad=deepcopy(value);bad['backend']['optimizer']['recipe']['parameters']=tuple(reversed(
            bad['backend']['optimizer']['recipe']['parameters']))
        with self.assertRaises(ValueError):T.validate_snapshot(bad,**kwargs)
        bad=deepcopy(value);next(iter(bad['draft_state_dict'].values())).fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):T.validate_snapshot(bad,**kwargs)
        bad=deepcopy(value);next(iter(bad['backend']['optimizer']['optimizer_state_dict']['state'].values())).pop('exp_avg')
        with self.assertRaises(ValueError):T.validate_snapshot(bad,**kwargs)

    def test_events_are_truthful_no_historical_range_or_lr_normalization(self):
        value,_=own_state('K_HALF',58)
        T.validate_event_receipt(value['events'][-1],arm='K_HALF',event_number=58)
        for event in (0,59):
            with self.assertRaises(ValueError):
                T.validate_event_receipt(value['events'][-1],arm='K_HALF',event_number=event)
        with self.assertRaises(ValueError):
            T.validate_event_receipt(value['events'][-1],arm='K',event_number=58)

    def test_initial_pair_requires_identical_rng_weights_and_empty_adam(self):
        first,_=own_state('K');second,_=own_state('K_HALF')
        self.assertTrue(T.compare_initial_states(first,second)['exact_released_weights_empty_adam_rng'])
        second['rng']['torch']=second['rng']['torch'].clone()
        second['rng']['torch'][0]^=1
        with self.assertRaisesRegex(ValueError,'initialization differs'):
            T.compare_initial_states(first,second)

    def test_reused_feedback_and_invalid_anchor_shape_fail(self):
        value,kwargs=own_state('K',2)
        value['events'][1]['features']=deepcopy(value['events'][0]['features'])
        for row,prior in zip(value['events'][1]['records'],value['events'][0]['records']):
            row['record_id']=prior['record_id']
        with self.assertRaisesRegex(ValueError,'reuses'):
            T.validate_snapshot(value,**kwargs)
        value,_=own_state('K',1)
        value['events'][0]['records'][0]['sampled_anchors'][0]['keep_mask']=[[True]]
        with self.assertRaisesRegex(ValueError,'Malformed'):
            T.validate_event_receipt(value['events'][0],arm='K',event_number=1)


class FullKTinyIntegrationTests(unittest.TestCase):
    """Five-layer tiny DFlash, real K loss/sampling/BF16 Adam and full events."""
    @classmethod
    def setUpClass(cls):
        from test_o_half_matched_lr_training import OHalfTinyIntegrationTests
        OHalfTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        from test_o_half_matched_lr_training import OHalfTinyIntegrationTests
        OHalfTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        from test_o_half_matched_lr_training import OHalfTinyIntegrationTests
        import specforge.export.to_hf as exporter
        OHalfTinyIntegrationTests.setUp(self)
        raw=yaml.safe_load(self.config.read_text())
        raw['model']['draft_checkpoint_path']=str(self.root/'released')
        self.config.write_text(yaml.safe_dump(raw))
        self.original_config_text=self.config.read_text()
        def export(state,config,output,**kwargs):
            target=Path(output);target.mkdir()
            decoded=torch.load(state,map_location='cpu',weights_only=False)
            save_file(decoded['draft_state_dict'],target/'model.safetensors')
        exporter.export_to_hf.side_effect=export

    def build(self,name,arm='K',**kwargs):
        config=self.root/(name+'.yaml')
        raw=yaml.safe_load(self.original_config_text)
        raw['training']['learning_rate']=T.LEARNING_RATES[arm]
        config.write_text(yaml.safe_dump(raw))
        trainer=T.FullKTrainer(config,self.root/name,arm=arm,binding=BINDING,
            original_config_text=self.original_config_text,expected_cuda_rng_states=0,**kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_full_native_path_initialization_observation_continuity_and_restore(self):
        trainer=self.build('full')
        released=self.root/'released';released.mkdir()
        save_file(dict(trainer.bundle.draft_model.state_dict()),released/'model.safetensors')
        before=capture_rng();initial=trainer.save_initial()
        self.assertFalse(T.differences(before,capture_rng()))
        initial_path=initial['state_path']
        first=trainer.train_event('event_001',self.features)
        trainer.verify_boundary(first['export_path'])
        state1=torch.load(trainer.latest,weights_only=False)
        checkpoint=self.root/'boundary.pt';checkpoint.write_bytes(trainer.latest.read_bytes())
        self.assertEqual(len(trainer.response_records),256)
        self.assertEqual(len(trainer.window_records),128)
        self.assertEqual(first['objective_work']['work']['response_forwards'],256)
        self.assertEqual(first['objective_work']['work']['draft_forwards'],256)
        for row in first['records']:
            observation=row['response_observation']
            self.assertIn('selected_position_occurrences',observation['sum_metrics'])
            self.assertIn('loss_denominator',observation['sum_metrics'])
            self.assertEqual(observation['scope'],'one_original_response_forward')
        second=[]
        for index,path in enumerate(self.features):
            raw=torch.load(path,weights_only=True)
            raw['record_id']=f'event2_{index}'
            raw['label_binding']['record_id']=raw['record_id']
            destination=self.root/f'new_{index}.pt';torch.save(raw,destination);second.append(destination)
        before=capture_rng()
        with T.preserve_training_rng(trainer):
            random.random();np.random.rand();torch.rand(7)
            trainer.bundle.model.eval()
        self.assertFalse(T.differences(before,capture_rng()))
        self.assertTrue(trainer.bundle.model.training)
        expected=trainer.train_event('event_002',second)
        trainer.verify_boundary(expected['export_path'])
        end=torch.load(trainer.latest,weights_only=False)
        trainer.close()
        restored=self.build('restored',resume_state=checkpoint,completed_event=1)
        self.assertTrue(T.verify_live(restored,state1)['exact_resume'])
        actual=restored.train_event('event_002',second)
        restored.verify_boundary(actual['export_path'])
        restored_state=torch.load(restored.latest,weights_only=False)
        for key in ('draft_state_dict','backend','rng','global_step','presentations'):
            self.assertFalse(T.differences(end[key],restored_state[key]),key)
        for key in ('record_id','pass','sampled_anchors','metrics'):
            self.assertEqual([row[key] for row in expected['records']],
                             [row[key] for row in actual['records']],key)
        restored.close()
        half=self.build('half','K_HALF')
        compared=half.save_initial(comparator_path=initial_path)
        self.assertTrue(compared['paired_initialization']['exact_released_weights_empty_adam_rng'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_observation_hooks_preserve_exact_plain_K_update(self):
        trainer=self.build('observed')
        released=self.root/'released';released.mkdir()
        save_file(dict(trainer.bundle.draft_model.state_dict()),released/'model.safetensors')
        trainer.save_initial()
        observed=trainer.train_event('event_001',self.features)
        state=torch.load(trainer.latest,weights_only=False);trainer.close()
        plain=T.SpecForgeEventTrainer(self.config,self.root/'plain',objective='osd_first_error',ranking_lambda=0.)
        self.addCleanup(plain.close)
        native=plain.train_event('event_001',self.features)
        native_state=torch.load(plain.latest,weights_only=False)
        for key in ('draft_state_dict','backend','rng','global_step','presentations'):
            self.assertFalse(T.differences(state[key],native_state[key]),key)
        for key in ('metrics','sampled_anchors'):
            self.assertEqual([row[key] for row in observed['records']],[row[key] for row in native['records']])


if __name__=='__main__':
    unittest.main()
