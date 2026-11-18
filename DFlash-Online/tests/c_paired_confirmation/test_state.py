from copy import deepcopy
import io
import unittest

import torch

from experiments.c_paired_confirmation import state as S
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng, restore_rng
from test_mathqa_fixed_lr_transfer_training import origin
import test_recent_repair_retention_training as retention_fixture

BINDING = 'a'*64


def fixture(dataset='mathqa', arm='C'):
    if dataset == 'mathqa':
        source, args = origin()
        text = args['original_config_text']
    else:
        source, text = retention_fixture.CompleteStateTests().origin()
    patched = deepcopy(source['rng'])
    patched['torch'] = torch.Generator().manual_seed(29).get_state()
    lineage = S.fork_record(source,patched,dataset=dataset,origin_identity={'sha256':'b'*64,'bytes':123},physical_gpu_uuid='GPU-test')
    own = S.fork_state(source,dataset=dataset,arm=arm,binding=BINDING,lineage=lineage,
        config_text=source['config_text'],run_id=source['run_id'],original_config_text=text,expected_cuda_rng_states=0)
    args = dict(origin=source,dataset=dataset,arm=arm,binding=BINDING,lineage=lineage,
        config_text=source['config_text'],original_config_text=text,completed_event=S.ORIGIN_EVENT[dataset],
        expected_recipe=source['backend']['optimizer']['recipe'],expected_cuda_rng_states=0)
    return source,own,args


class RNGForkTests(unittest.TestCase):
    def test_private_generators_change_only_assigned_fields_and_no_live_rng(self):
        live=capture_rng();self.addCleanup(lambda:restore_rng(live))
        saved=deepcopy(live)
        saved['cuda']=[torch.Generator().manual_seed(1).get_state(),torch.Generator().manual_seed(2).get_state()]
        before=deepcopy(saved);seen=[]
        def generator(*,device):
            seen.append(str(device));return torch.Generator(device='cpu')
        patched=S.fork_rng_state(saved,device='cuda:1',logical_cuda_index=1,generator_factory=generator)
        self.assertEqual(seen,['cpu','cuda:1'])
        self.assertEqual(differences(saved,before),[])
        self.assertEqual(differences(live,capture_rng()),[])
        self.assertTrue(torch.equal(patched['cuda'][0],saved['cuda'][0]))
        self.assertFalse(torch.equal(patched['cuda'][1],saved['cuda'][1]))
        self.assertTrue(torch.equal(patched['torch'],torch.Generator().manual_seed(29).get_state()))
        self.assertEqual(differences(saved['python'],patched['python']),[])
        self.assertEqual(differences(saved['numpy'],patched['numpy']),[])
        with self.assertRaises(ValueError):S.fork_rng_state(saved,device='cuda:2',logical_cuda_index=2,generator_factory=generator)

    def test_both_dataset_origins_remain_immutable_and_native_contract_strict(self):
        for dataset in S.ORIGIN_EVENT:
            for arm in ('N','C'):
                source,own,args=fixture(dataset,arm)
                before=deepcopy(source)
                result=S.validate_snapshot(own,**args)
                self.assertEqual(result['global_step'],128*S.ORIGIN_EVENT[dataset])
                self.assertEqual(differences(source,before),[])
                self.assertIs(own['backend'],source['backend'])
                self.assertIs(own['rng'],source['rng'])
                self.assertEqual(own['algorithm_contract'][S.CONTRACT_KEY]['continuation_rng_fork']['seed'],29)
                stream=io.BytesIO();torch.save(own,stream);stream.seek(0)
                self.assertEqual(differences(own,torch.load(stream,weights_only=False)),[])
                with self.assertRaises(ValueError):S.validate_origin(own,args['original_config_text'],dataset=dataset,expected_cuda_rng_states=0)

    def test_lineage_history_optimizer_and_horizon_cannot_change(self):
        _,own,args=fixture()
        for mutate in [lambda x:x['algorithm_contract'][S.CONTRACT_KEY]['continuation_rng_fork'].update(seed=17),
                       lambda x:x.update(horizon=256),
                       lambda x:x['backend']['optimizer']['optimizer_state_dict']['state'].clear(),
                       lambda x:x['events'][0]['records'][1].update(learning_rate=1e-4)]:
            changed=deepcopy(own);mutate(changed)
            with self.assertRaises(ValueError):S.validate_snapshot(changed,**args)
        changed=deepcopy(own);changed['config_text']=changed['config_text'].replace('seed: 17','seed: 29')
        with self.assertRaises(ValueError):S.validate_snapshot(changed,**dict(args,config_text=changed['config_text']))

if __name__=='__main__':unittest.main()
