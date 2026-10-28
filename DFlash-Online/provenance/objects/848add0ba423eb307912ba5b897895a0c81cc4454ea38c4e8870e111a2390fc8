"""Driver shutdown must retain the original failure and final receipts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import campaign


class Backend:
    loading_s = 0.25

    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        raise RuntimeError('cleanup failed')


class DriverTests(unittest.TestCase):
    def test_cleanup_failure_keeps_receipts_and_prior_error(self):
        template = json.loads((Path(__file__).parent / 'inventory_audit/NEW_PANEL.json').read_text())
        for evaluation_error in (None, ValueError('original native failure')):
            with self.subTest(evaluation_error=evaluation_error), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                (root / 'sealed').mkdir()
                plan = dict(requests=template, checkpoints={'O': 'O', 'K': 'K'},
                            selection_seed=20261027, planned_generations=384,
                            freshness_scope='current_one_event_O_K_study', target='target',
                            released_draft='draft', worker_python='unused')
                (root / 'sealed/PLAN.json').write_text(json.dumps(plan))
                module = SimpleNamespace(SGLangDFlashBackend=Backend)
                with patch.dict('sys.modules', {'online_feedback.sglang_backend': module}), \
                     patch.object(campaign, 'evaluate', side_effect=evaluation_error, return_value={'status': 'completed'}):
                    expected = 'original native failure' if evaluation_error else 'cleanup failed'
                    with self.assertRaisesRegex(Exception, expected):
                        campaign.run(root)
                status = json.loads((root / 'campaign_001/STATUS.json').read_text())
                self.assertEqual(status['status'], 'failed')
                self.assertIn(expected, status['error'])
                self.assertIn('cleanup failed', status['cleanup_error'])
                self.assertTrue((root / 'campaign_001/COSTS.json').is_file())


if __name__ == '__main__':
    unittest.main()
