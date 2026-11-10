"""Native receipt controls cannot silently accept conditioning/state drift."""
from types import SimpleNamespace
import unittest

import torch

from experiments.kplus_abt.qualification import NativeInputAudit


class NativeAuditTests(unittest.TestCase):
    def fixture(self):
        module=torch.nn.Linear(3,3,bias=False)
        embeddings=torch.zeros(1,1,4,3,dtype=torch.bfloat16)
        runtime=SimpleNamespace(view='clean',embeddings=embeddings,
            eligible=torch.tensor([[[False,True,True,False]]]))
        state={'step':torch.tensor(512.),'exp_avg':torch.zeros(3),'exp_avg_sq':torch.ones(3)}
        audit=object.__new__(NativeInputAudit)
        audit.trainer=SimpleNamespace(bundle=SimpleNamespace(model=SimpleNamespace(adversarial_block=runtime)),
            backend=SimpleNamespace(optimizer=SimpleNamespace(optimizer=SimpleNamespace(state={0:state}))))
        audit.pending=None;audit.pairs=[]
        kwargs=dict(target_hidden=torch.ones(1,5,3),position_ids=torch.arange(9).unsqueeze(0),
            attention_mask=torch.ones(4,9,dtype=torch.bool),noise_embedding=embeddings.reshape(1,4,3))
        return audit,module,runtime,state,kwargs

    def test_native_pair_keeps_context_rng_and_adam(self):
        audit,module,runtime,state,kwargs=self.fixture()
        audit.observe(module,(),kwargs);runtime.view='second'
        second=dict(kwargs,noise_embedding=kwargs['noise_embedding'].clone())
        second['noise_embedding'][:,1,0]=.01
        audit.observe(module,(),second)
        self.assertEqual(audit.pairs[0]['changed_slots'],1)
        self.assertIsNone(audit.pending)

    def test_seed_change_and_adam_moment_mutation_fail(self):
        for mutation in ('seed','moment'):
            audit,module,runtime,state,kwargs=self.fixture()
            audit.observe(module,(),kwargs);runtime.view='second'
            second=dict(kwargs,noise_embedding=kwargs['noise_embedding'].clone())
            if mutation=='seed':second['noise_embedding'][:,0,0]=.01
            else:state['exp_avg'].add_(1)
            with self.assertRaises(ValueError):audit.observe(module,(),second)

    def test_unknown_mask_is_not_certified_empty(self):
        with self.assertRaises(TypeError):NativeInputAudit.mask_parts(object())


if __name__=='__main__':unittest.main()
