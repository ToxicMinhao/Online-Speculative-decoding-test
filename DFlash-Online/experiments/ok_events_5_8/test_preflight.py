"""CPU preflight gates, including complete own states and no CUDA use."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from experiments.ok_continuation.test_resume import fixture
from . import preflight as P


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)

    def states(self):
        plan = dict(initial={},windows=[dict(requests=[dict(request_id='new')])])
        identities = {}
        for arm,objective in (('O','osd'),('K','osd_first_error')):
            state = fixture(objective,4)
            for number,event in enumerate(state['events'],1):
                for feature in event['features']:
                    feature['record_id'] = f"event{number}_{feature['record_id']}"
                for record in event['records']:
                    record['record_id'] = f"event{number}_{record['record_id']}"
            directory = self.root/arm
            export = directory/'export'
            export.mkdir(parents=True)
            path = directory/'state.pt'
            torch.save(state,path)
            save_file(state['draft_state_dict'],export/'model.safetensors')
            receipt = directory/'RESULT.json'
            write(receipt,dict(state['events'][-1],full_state_path=str(path),export_path=str(export)))
            for file in (path,receipt,export/'model.safetensors'):
                identities[str(file)] = P.identity(file)
            plan['initial'][arm] = dict(state=str(path),export=str(export),receipt=str(receipt),state_identity=P.identity(path))
        config = self.root/'config.yaml'
        config.write_text(state['config_text'])
        plan['config'] = str(config)
        return plan,identities

    def test_complete_own_states_and_normal_history_are_compared_without_cuda(self):
        plan,identities = self.states()
        before = torch.random.get_rng_state().clone()
        result = P.verify_initial_states(plan,identities,expected_cuda_rng_states=0)
        self.assertTrue(result['final_O_K_rng_exact'] and result['normal_history_exact'])
        self.assertEqual(result['arms']['O']['prior_distinct_training_records'],512)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_rng_history_and_new_overlap_are_rejected(self):
        for mutation in ('rng','history','overlap'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                plan,identities = self.states()
                source = plan['initial']['K']
                if mutation == 'overlap':
                    plan['windows'][0]['requests'][0]['request_id'] = 'event1_r000'
                else:
                    state = torch.load(source['state'],weights_only=False)
                    if mutation == 'rng':
                        generator = torch.Generator().manual_seed(123456)
                        state['rng']['torch'] = generator.get_state()
                    else:
                        state['events'][0]['records'][0]['sampled_anchors'][0]['positions'][0][0] += 1
                    torch.save(state,source['state'])
                    identities[source['state']] = source['state_identity'] = P.identity(source['state'])
                with self.assertRaises(ValueError):
                    P.verify_initial_states(plan,identities,expected_cuda_rng_states=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_seal_requires_all_pins_and_rejects_source_or_file_set_change(self):
        sealed = self.root/'sealed'
        names = ('PLAN.json','config.yaml','MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json','REMOTE_INPUT_IDENTITIES.json')
        for name in names:
            write(sealed/name,{})
        seal = dict(files={str(sealed/name):P.identity(sealed/name) for name in names},exact_file_sets={})
        write(sealed/'SEAL.json',seal)
        P.verify_seal(self.root)
        (sealed/'config.yaml').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Sealed input changed'):
            P.verify_seal(self.root)
        write(sealed/'config.yaml',{})
        seal['exact_file_sets'] = {str(sealed):sorted(path.name for path in sealed.iterdir())}
        write(sealed/'SEAL.json',seal)
        (sealed/'unexpected').write_text('extra')
        with self.assertRaisesRegex(ValueError,'file set changed'):
            P.verify_seal(self.root)

    def test_idle_two_A800_capacity_and_no_active_compute(self):
        good = '0, NVIDIA A800 80GB PCIe, 81920, 1\n1, NVIDIA A800 80GB PCIe, 81920, 1\n'
        self.assertEqual(len(P.gpu_admission(good,'')['gpus']),2)
        for csv,processes in ((good,'123, GPU-one, 120'),(good.replace('81920','40960'),''),
                              (good.replace('A800','H100'),''),(good.splitlines()[0],'')):
            with self.assertRaises(ValueError):
                P.gpu_admission(csv,processes)

    def test_first_phase_admission_preserves_reserve_and_future_guards(self):
        plan = dict(storage=dict(reserve_bytes=8*P.GIB,offload_timeout_s=600,offload_destination='/verified'),
            windows=[dict(requests=[dict(prompt_ids=[1,2],max_new_tokens=2048) for _ in range(128)])])
        good = P.storage_admission(plan,free_bytes=100*P.GIB,available_host_bytes=100*P.GIB)
        self.assertTrue(good['admitted'] and good['no_assumption_all_future_phases_fit'])
        self.assertEqual(good['maximum_capture_rows'],128*2050)
        self.assertFalse(P.storage_admission(plan,free_bytes=good['required_bytes']-1,available_host_bytes=100*P.GIB)['admitted'])
        self.assertFalse(P.storage_admission(plan,free_bytes=100*P.GIB,available_host_bytes=79*P.GIB)['admitted'])

    def test_postflight_preserves_preflight_and_skips_initial_capacity(self):
        import transformers
        from .test_campaign import Harness
        harness = Harness(self.root)
        (self.root/'campaign_001').mkdir()
        initial = self.root/'PREFLIGHT.json'
        initial.write_text('original preflight')
        sealed = self.root/'sealed'
        write(sealed/'MODEL_PIN.json',dict(target=dict(files={}),draft=dict(files={})))
        write(sealed/'RUNTIME_PIN.json',{})
        write(sealed/'TRAINING_PIN.json',dict(environment=dict(python=P.platform.python_version(),
            torch=torch.__version__,transformers=transformers.__version__)))
        with patch.object(P,'verify_seal',return_value=(dict(files={}),{})), \
                patch.object(P,'hardware_admission',side_effect=AssertionError('hardware must be skipped')), \
                patch.object(P,'storage_admission',side_effect=AssertionError('admission must be skipped')), \
                patch.object(P,'verify_initial_states',return_value=dict(status='passed')) as states, \
                patch('online_feedback.qualify_sglang.check_recipe',return_value={}), \
                patch('experiments.osd_components.prepare.verify_files'), \
                patch('experiments.osd_components.prepare.verify_native_runtime',return_value={}):
            result = P.verify(self.root,postflight=True)
        self.assertEqual(result['status'],'passed')
        states.assert_called_once()
        self.assertEqual(initial.read_text(),'original preflight')
        self.assertTrue((self.root/'POSTFLIGHT.json').is_file())


if __name__ == '__main__':
    unittest.main()
