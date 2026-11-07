"""Retained canonical authentication never trusts a copied token list alone."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.role_loss_pilot.seal import authenticate_references
from test_full_k_gsm8k_evaluation import FullBackend, requests


class CanonicalSealTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.root=Path(directory.name);self.inputs=requests(1)
        self.raw=FullBackend(self.inputs).generate(self.inputs[0]['prompt_ids'],
            request_id='old-record',session_id='old-panel',request_index=0,
            purpose='evaluation',save_to=self.root/'old.json')
        self.refs={'eval0':dict(response_ids=self.raw['response_ids'],finish_reason='eos',
                                source_record_path='/remote/old.json')}

    def audit(self):
        return authenticate_references(self.inputs,self.refs,self.root,'/remote','a'*64)

    def test_reordering_keeps_original_raw_identity_and_exact_tokens(self):
        self.inputs[0]['panel_ordinal']=17
        enriched,files=self.audit()
        self.assertTrue(enriched['eval0']['verified'])
        self.assertEqual(enriched['eval0']['prompt_ids'],self.raw['prompt_ids'])
        self.assertEqual(enriched['eval0']['source_record']['sha256'],files['/remote/old.json']['sha256'])
        self.assertNotIn('verified',self.refs['eval0'])

    def test_corrupt_native_decision_is_not_sealed(self):
        self.raw['metadata']['trace'][0]['target_argmax'][0]=999
        (self.root/'old.json').write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(ValueError,'Invalid retained canonical'):self.audit()

    def test_new_input_cannot_inherit_old_canonical(self):
        self.inputs[0]['prompt_ids']=[7,8]
        with self.assertRaisesRegex(ValueError,'Invalid retained canonical'):self.audit()


if __name__=='__main__':unittest.main()
