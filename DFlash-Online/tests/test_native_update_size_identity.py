import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from experiments.native_update_size.identity_gate import norm_agrees,validate_copy

class IdentityGateTests(unittest.TestCase):
    def test_fixed_norm_tolerance_and_nonfinite(self):
        self.assertTrue(norm_agrees(1.+1e-11,1.))
        self.assertFalse(norm_agrees(1.+2e-10,1.))
        self.assertFalse(norm_agrees(float('nan'),1.))
        self.assertFalse(norm_agrees(1.,float('inf')))

    def exercise_copy(self,expected_result):
        from experiments.threatened_repair_replay.prepare import identity
        import json
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);original=root/'original';original.mkdir();(original/'config.json').write_text('{}')
            old=original/'model.safetensors';old.write_bytes(b'original')
            row=dict(window=8,state='P',checkpoint=str(original),weights_identity=identity(old))
            destination=root/'new';receipt=root/'receipt.json'
            def exporter(weights,template,where):
                where.mkdir();(where/'model.safetensors').write_bytes(b'original' if expected_result else b'different')
                return dict(weights_identity=identity(where/'model.safetensors'),tensors=1)
            with patch('experiments.native_update_size.identity_gate.export_weights',exporter),patch('experiments.native_update_size.identity_gate.check_write'):
                if expected_result:
                    result=validate_copy(dict(origin_export=str(original)),row,{},destination,receipt)
                    self.assertEqual(result['status'],'passed');self.assertFalse(destination.exists())
                else:
                    with self.assertRaisesRegex(ValueError,'Unqualified file-identity'):
                        validate_copy(dict(origin_export=str(original)),row,{},destination,receipt)
                    self.assertTrue((destination/'model.safetensors').is_file())
                    self.assertEqual(json.loads(receipt.read_text())['copy_class'],'REVIEW')
            self.assertEqual(old.read_bytes(),b'original');self.assertTrue(receipt.is_file())

    def test_success_deletes_only_new_copy_after_durable_receipt(self): self.exercise_copy(True)
    def test_failure_preserves_mismatch_and_original(self): self.exercise_copy(False)
