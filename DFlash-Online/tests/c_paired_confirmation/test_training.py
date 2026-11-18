from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from experiments.c_paired_confirmation.training import PairedContinuationTrainer
from experiments.c_paired_confirmation import state as S
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.kplus import KPlusRedistribution
from online_feedback.prefix_allocation import PrefixAllocation


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.rng=capture_rng();self.addCleanup(lambda:restore_rng(self.rng))
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def bare(self):
        t=object.__new__(PairedContinuationTrainer)
        t.output_root=Path(self.temp.name)
        t._fork_applied=False;t.completed_event=4;t.dataset='mathqa'
        t.core=SimpleNamespace(accumulation_remainder=0)
        t.origin={'rng':deepcopy(self.rng)}
        t._fork_rng=deepcopy(self.rng);t._fork_rng['torch']=torch.Generator().manual_seed(29).get_state()
        t.lineage={'seed':29,'historical_initialization_seed':17}
        return t

    def test_fork_is_live_once_and_reload_never_reseeds(self):
        trainer=self.bare()
        result=trainer.apply_continuation_fork()
        self.assertTrue(result['reseeded'])
        expected=torch.Generator().manual_seed(29)
        self.assertEqual(torch.randperm(128).tolist(),torch.randperm(128,generator=expected).tolist())
        current=capture_rng()
        self.assertFalse(trainer.apply_continuation_fork()['reseeded'])
        self.assertEqual(differences(current,capture_rng()),[])
        trainer.completed_event=5
        self.assertFalse(trainer.apply_continuation_fork()['reseeded'])

    def test_unrestored_startup_rng_is_not_silently_hidden(self):
        trainer=self.bare();torch.rand(1)
        with self.assertRaisesRegex(ValueError,'Startup'):trainer.apply_continuation_fork()
        self.assertFalse(trainer._fork_applied)

    def test_second_event_cannot_apply_a_fresh_fork(self):
        trainer=self.bare();trainer.completed_event=5
        with self.assertRaises(ValueError):trainer.apply_continuation_fork()

    def test_controller_and_sampler_paths_are_reused(self):
        trainer=self.bare();trainer.qualification_only=False
        trainer.relative_cap=trainer._qualification_cap=.25
        trainer.bundle=SimpleNamespace(model=SimpleNamespace(objective='osd_first_error',loss_dtype=torch.bfloat16,
            first_error_redistribution=KPlusRedistribution(),prefix_allocation=PrefixAllocation()))
        trainer._check_native_path()
        trainer.bundle.model.prefix_guard=object()
        with self.assertRaises(ValueError):trainer._check_native_path()
        with self.assertRaises(ValueError):trainer.reset_qualification()
        with self.assertRaises(ValueError):trainer.train_event('event_005',[],presentation_schedule={})

    def test_natural_pass_receipt_does_not_force_historical_schedule(self):
        trainer=self.bare();trainer.events_root=Path(self.temp.name);trainer._active_event='event_005'
        (trainer.events_root/trainer._active_event).mkdir()
        rows=[{'record_id':f'p{p}r{i}','pass':p} for p in range(2) for i in range(2)]
        observed=[]
        def train(current,features,passes):
            observed.append(capture_rng());torch.randperm(4);return rows
        with patch('experiments.role_loss_pilot.training.RoleLossTrainer._train_passes',train):
            actual=trainer._train_passes([1,2],2)
        self.assertEqual(actual,rows)
        self.assertEqual(differences(observed[0],trainer._fork_rng),[])
        self.assertTrue((trainer.events_root/trainer._active_event/'NATURAL_SAMPLING.json').is_file())

class NativeTinyIntegrationTests(unittest.TestCase):
    """Real tiny BF16 draft/Adam; only the missing CUDA RNG generator is a CPU fixture."""
    @classmethod
    def setUpClass(cls):
        import os
        if os.environ.get('C_PAIRED_TINY_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        import os
        if os.environ.get('C_PAIRED_TINY_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        import os
        if os.environ.get('C_PAIRED_TINY_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    def test_real_two_event_continuation_uses_own_rng_without_refork(self):
        import os, subprocess, sys
        if os.environ.get('C_PAIRED_TINY_CHILD') != '1':
            root=Path(__file__).resolve().parents[2]
            env=dict(os.environ,C_PAIRED_TINY_CHILD='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            env['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('.','tests','training','inference'))
            result=subprocess.run([sys.executable,'-m','unittest','-v','c_paired_confirmation.test_training.NativeTinyIntegrationTests.'+self._testMethodName],cwd=root,env=env,text=True,capture_output=True,timeout=240)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return
        import test_mathqa_fixed_lr_transfer_training as baseline
        path,export,source=baseline.MathQATinyIntegrationTests.make_parent(self)
        def cpu_fork(saved,**unused):
            copied=deepcopy(saved);copied['torch']=torch.Generator().manual_seed(29).get_state();return copied
        def build(name,**kwargs):
            with patch.object(S,'fork_rng_state',side_effect=cpu_fork), patch.object(torch.cuda,'current_device',return_value=0):
                trainer=PairedContinuationTrainer(self.config,self.root/name,dataset='mathqa',arm='C',binding='a'*64,
                    origin_state=path,origin_export=export,original_config_text=self.original_config_text,
                    origin_identity={'sha256':'b'*64,'bytes':123},physical_gpu_uuid='CPU-fixture',
                    expected_cuda_rng_states=0,**kwargs)
            self.addCleanup(trainer.close);return trainer
        trainer=build('C5',completed_event=4)
        fifth=trainer.train_event('event_005',self.features)
        trainer.verify_boundary(fifth['export_path'])
        checkpoint=self.root/'C5-recovery.pt';checkpoint.write_bytes(trainer.latest.read_bytes())
        self.assertEqual(differences(source,trainer.origin),[])
        own=[]
        for i,path0 in enumerate(self.features):
            payload=torch.load(path0,weights_only=True);payload['record_id']=f'own_C5_{i}'
            payload['label_binding']['record_id']=payload['record_id']
            path1=self.root/f'own_{i}.pt';torch.save(payload,path1);own.append(path1)
        sixth=trainer.train_event('event_006',own)
        expected=torch.load(trainer.latest,weights_only=False);trainer.close()
        restored=build('C6-reload',completed_event=5,resume_state=checkpoint,resume_export=fifth['export_path'])
        self.assertTrue(restored._fork_applied)
        repeated=restored.train_event('event_006',own)
        restored.verify_boundary(repeated['export_path'])
        actual=torch.load(restored.latest,weights_only=False)
        for key in ('draft_state_dict','backend','rng','global_step','presentations'):
            self.assertEqual(differences(expected[key],actual[key]),[],key)
        for key in ('record_id','pass','sampled_anchors','metrics'):
            self.assertEqual([r[key] for r in sixth['records']],[r[key] for r in repeated['records']])
        self.assertEqual(actual['algorithm_contract'][S.CONTRACT_KEY]['continuation_rng_fork']['seed'],29)
        self.assertFalse((restored.output_root/'CONTINUATION_RNG_FORK.json').exists())
        self.assertEqual((restored.global_step,restored.presentations),(768,1536))
        with self.assertRaises(ValueError):restored.train_event('event_007',own)


if __name__=='__main__':unittest.main()
