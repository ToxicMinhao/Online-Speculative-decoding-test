"""ABT resumes complete PR32 state without weakening historical event guards."""
from copy import deepcopy
import unittest

from experiments.kplus_abt import training as T
from experiments.kplus_abt.contract import ARMS, DATASETS
from experiments.role_loss_pilot import training as historical
from online_feedback.rng import capture_rng
from test_role_loss_campaign_training import pilot_state

BINDING='b'*64


def fixture(event=4,arm='ADV',dataset='gsm8k'):
    origin,args=pilot_state(4,'KPLUS',dataset)
    state,_=pilot_state(event,'KPLUS',dataset)
    state['events'][:4]=deepcopy(origin['events'])
    state['algorithm_contract']=T.continuation_contract(origin,binding=BINDING,arm=arm,dataset=dataset)
    for receipt in state['events'][4:]:
        receipt.pop('pilot_loss')
        receipt.update(method=arm,dataset=dataset,kplus_loss=T.kplus_contract())
    kwargs=dict(origin=origin,binding=BINDING,arm=arm,dataset=dataset,
        config_text=state['config_text'],original_config_text=args['original_config_text'],
        completed_event=event,expected_recipe=origin['backend']['optimizer']['recipe'],expected_cuda_rng_states=0)
    return origin,state,kwargs


class ResumeTests(unittest.TestCase):
    def test_every_arm_boundary_preserves_history_optimizer_rng(self):
        for dataset in DATASETS:
            for arm in ARMS:
                for event in (4,5,6):
                    origin,state,args=fixture(event,arm,dataset)
                    before=deepcopy(state);rng=capture_rng()
                    result=T.validate_snapshot(state,**args)
                    self.assertEqual(result['global_step'],128*event)
                    self.assertFalse(T.differences(before,state))
                    self.assertFalse(T.differences(rng,capture_rng()))

    def test_metadata_migration_keeps_original_scientific_payload(self):
        origin,state,args=fixture()
        migrated=T.migrate_origin(origin,binding=BINDING,arm='ADV',dataset='gsm8k',
            config_text=state['config_text'],run_id='new_arm',original_config_text=args['original_config_text'],
            expected_recipe=args['expected_recipe'],expected_cuda_rng_states=0)
        for field in ('draft_state_dict','backend','rng','events'):
            self.assertIs(migrated[field],origin[field])
        self.assertEqual(migrated['global_step'],512)

    def test_missing_moments_foreign_history_or_later_event_fail(self):
        origin,state,args=fixture(5)
        for change in (lambda s:s['backend']['optimizer']['optimizer_state_dict']['state'].clear(),
                       lambda s:s['events'][0].update(method='ROLE'),
                       lambda s:s['algorithm_contract'][T.CONTRACT_KEY].update(arm='RANDOM')):
            bad=deepcopy(state);change(bad)
            with self.assertRaises(ValueError):T.validate_snapshot(bad,**args)
        with self.assertRaises(ValueError):T.validate_snapshot(state,**dict(args,completed_event=7))
        with self.assertRaises(ValueError):T.validate_origin(state,dataset='gsm8k',
            original_config_text=args['original_config_text'],expected_cuda_rng_states=0)


if __name__=='__main__':unittest.main()
