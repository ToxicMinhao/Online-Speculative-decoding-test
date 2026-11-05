"""Postflight resource failures remain launch failures; no model operations."""
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

import torch
import transformers
from experiments.kplus_full_gsm8k import preflight as P


class PostflightTests(unittest.TestCase):
    def test_postflight_resource_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            root = Path(directory)
            sealed = root / 'sealed'
            sealed.mkdir()
            draft = root / 'draft'
            draft.mkdir()
            (draft / 'model.safetensors').write_bytes(b'x')
            config = root / 'config.yaml'
            config.write_text('fixture: true\n')
            marker = dict(bytes=1, sha256='a' * 64)
            values = {
                'PLAN.json': dict(baseline_counts=str(root / 'counts.json'), references={'K': str(root / 'K')},
                    original_config=str(config), configs={'KPLUS': str(config)}, target='/target',
                    draft=str(draft), worker_python='/worker', storage={}),
                'CONTROL_AUDIT.json': dict(status='passed', model_work=0, original_root='/original',
                    files={f'/original/sealed/{name}': marker for name in (
                        'ORIGINAL_MANIFEST.json', 'ORIGINAL_CONFIG.yaml', 'CANONICALS.json',
                        'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json')},
                    baseline_counts_identity=marker, control_identity=marker),
                'MODEL_PIN.json': dict(target={'files': {}}, draft={'files': {}}),
                'RUNTIME_PIN.json': {},
                'TRAINING_PIN.json': dict(environment=dict(python=platform.python_version(),
                    torch=torch.__version__, transformers=transformers.__version__)),
            }
            for name, value in values.items():
                (sealed / name).write_text(json.dumps(value))
            for name, value in [('verify_seal', {}), ('validate_plan', {}), ('identity', marker),
                                ('validate_config_text', None), ('verify_files', None),
                                ('verify_native_runtime', {}), ('capacity_plan', {})]:
                patches.enter_context(patch.object(P, name, return_value=value))
            patches.enter_context(patch.object(torch, 'load', side_effect=AssertionError('Unexpected tensor load')))
            gate = patches.enter_context(patch.object(P, 'check_growth', side_effect=ValueError('fixture reserve exceeded')))
            with self.assertRaisesRegex(ValueError, 'reserve exceeded'):
                P.verify(root, postflight=True)
            gate.assert_called_once_with(root, {})
            receipt = json.loads((root / 'POSTFLIGHT.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['model_work'], 0)
            self.assertIn('reserve exceeded', receipt['error'])
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
