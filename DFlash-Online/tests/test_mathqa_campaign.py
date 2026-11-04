"""No training paths, exact work accounting and bounded storage admission."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mathqa_transfer.campaign import actual_work
from experiments.mathqa_transfer.preflight import capacity_plan, check_space, verify_exports


class CampaignTests(unittest.TestCase):
    def test_unstarted_and_partial_work_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(actual_work(root)['unattempted_generations'], 9219)
            path = root / 'evaluation/controls/STATUS.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(attempted_generations=2, returned_generations=1,
                completed_valid_generations=1, invalid_generations=0, pending_canonical_generations=0,
                unknown_generation_attempts=1, unattempted_generations=1)))
            work = actual_work(root)
            self.assertEqual(work['returned_generations'], 1)
            self.assertEqual(work['unknown_generation_attempts'], 1)
            self.assertEqual(work['unattempted_generations'], 9217)
            self.assertEqual(work['optimizer_steps'], 0)

    def test_full_capacity_and_reserve_are_not_average_estimates(self):
        from types import SimpleNamespace
        required = capacity_plan()['peak_required_bytes']
        self.assertEqual(required, 9219 * 5327030 + 15 * 2**30)
        with patch('experiments.mathqa_transfer.preflight.shutil.disk_usage', return_value=SimpleNamespace(free=required)):
            self.assertEqual(check_space('.', initial=True)['status'], 'passed')
        with patch('experiments.mathqa_transfer.preflight.shutil.disk_usage', return_value=SimpleNamespace(free=required-1)):
            with self.assertRaises(RuntimeError):
                check_space('.', initial=True)
        with patch('experiments.mathqa_transfer.preflight.shutil.disk_usage', return_value=SimpleNamespace(free=8*2**30-1)):
            with self.assertRaises(RuntimeError):
                check_space('.')

    def test_export_identity_is_checked_again_after_serving(self):
        from experiments.ok_coding.storage import identity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints, exports = {}, {}
            for method in ('O', 'K', 'KPLUS'):
                folder = root / method
                folder.mkdir()
                (folder / 'model.safetensors').write_bytes(method.encode())
                checkpoints[method] = str(folder)
                exports[method] = dict(path=str(folder), files={'model.safetensors': identity(folder/'model.safetensors')})
            audit = dict(status='passed', model_work=0, checkpoints=checkpoints, exports=exports, verified_files={})
            self.assertEqual(len(verify_exports(dict(checkpoints=checkpoints), audit)), 3)
            (root / 'K/model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_exports(dict(checkpoints=checkpoints), audit)


if __name__ == '__main__':
    unittest.main()
