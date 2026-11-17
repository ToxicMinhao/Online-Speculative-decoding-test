from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from experiments.c_paired_confirmation import export as E
from online_feedback.rng import capture_rng
from experiments.ok_continuation.resume import differences


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.weights={'layer.weight':torch.tensor([[.5,-.5]],dtype=torch.bfloat16)}
        self.state=self.root/'recovery.pt';torch.save(dict(draft_state_dict=self.weights,global_step=640,presentations=1280),self.state)
        self.original=self.root/'original';self.original.mkdir()
        save_file(self.weights,self.original/'model.safetensors')
        (self.original/'config.json').write_text(json.dumps({'model_type':'dflash','block_size':16}))
        self.config=self.root/'config.json';self.config.write_text('{}')

    def writer(self,state,config,output,**kwargs):
        torch.rand(3)
        output=Path(output);output.mkdir()
        save_file(self.weights,output/'model.safetensors')
        (output/'config.json').write_text((self.original/'config.json').read_text())

    def test_exact_cpu_reconstruction_preserves_rng_and_records_tensor_metadata(self):
        before=capture_rng()
        with patch('specforge.export.to_hf.export_to_hf',self.writer):
            report=E.qualify_cpu_export(self.state,self.config,self.original,self.root/'recreated',receipt_path=self.root/'receipt.json')
        self.assertEqual(report['status'],'passed');self.assertTrue(report['exact_metadata_and_tensors'])
        self.assertTrue(report['exact_weights_file_identity'])
        self.assertEqual(report['weights_files'],report['expected_weights_files'])
        self.assertEqual(differences(before,capture_rng()),[])
        self.assertEqual(report['manifest']['tensors']['layer.weight']['dtype'],'torch.bfloat16')
        self.assertEqual(report['optimizer_steps'],0)
        self.assertIn('--embedding-key',report['reconstruction']['argv'])
        with self.assertRaises(FileExistsError):E.reconstruct_export(self.state,self.config,self.root/'recreated')

    def test_reconstruction_command_preserves_optional_embedding_configuration(self):
        command=E.reconstruction_command('state.pt','draft.json','export','receipt.json',
            embedding_source='/models/target',embedding_key='model.embedding.weight')
        argv=command['argv']
        self.assertEqual(argv[argv.index('--embedding-source')+1],'/models/target')
        self.assertEqual(argv[argv.index('--embedding-key')+1],'model.embedding.weight')

    def test_wrong_config_or_tensor_is_not_a_retirable_derived_copy(self):
        (self.original/'config.json').write_text('{"model_type":"dflash","block_size":10}')
        def bad_config(*args,**kwargs):
            self.writer(*args,**kwargs)
            (Path(args[2])/'config.json').write_text('{"model_type":"dflash","block_size":16}')
        with patch('specforge.export.to_hf.export_to_hf',bad_config):
            with self.assertRaisesRegex(ValueError,'metadata'):
                E.qualify_cpu_export(self.state,self.config,self.original,self.root/'bad',receipt_path=self.root/'failed.json')
        self.assertEqual(json.loads((self.root/'failed.json').read_text())['status'],'failed')
        save_file({'layer.weight':self.weights['layer.weight']+1},self.original/'model.safetensors')
        with self.assertRaises(ValueError):E.verify_derived_export(self.state,self.original,reconstruction={})

    def test_equal_tensors_with_changed_container_bytes_do_not_pass_bound_consumer(self):
        def changed_container(*args,**kwargs):
            self.writer(*args,**kwargs)
            save_file(self.weights,Path(args[2])/'model.safetensors',metadata={'different':'header'})
        with patch('specforge.export.to_hf.export_to_hf',changed_container):
            with self.assertRaisesRegex(ValueError,'weights_file_identity'):
                E.qualify_cpu_export(self.state,self.config,self.original,self.root/'different_bytes',
                    receipt_path=self.root/'bytes_failed.json')
        result=json.loads((self.root/'bytes_failed.json').read_text())
        self.assertTrue(result['exact_metadata_and_tensors'])
        self.assertFalse(result['exact_weights_file_identity'])
        self.assertEqual(result['status'],'failed')

if __name__=='__main__':unittest.main()
