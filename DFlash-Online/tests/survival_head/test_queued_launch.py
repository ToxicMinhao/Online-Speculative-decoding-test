import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.survival_head import queued_launch as Q


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'pilot'; (self.root/'sealed').mkdir(parents=True)
        self.prior = Path(self.temp.name)/'priority'; (self.prior/'campaign_001').mkdir(parents=True)
        (self.root/'sealed/PLAN.json').write_text(json.dumps(dict(priority_root=str(self.prior),
            execution_source=dict(source_path='/declared/code'))))

    def finish_prior(self, code='0'):
        (self.prior/'EXIT_CODE').write_text(code)
        (self.prior/'campaign_001/STATUS.json').write_text(json.dumps(dict(status='completed')))

    def test_wait_does_no_gpu_work_then_launches_once(self):
        def tick(_): self.finish_prior()
        with patch.object(Q.time, 'sleep', side_effect=tick) as wait, patch.object(Q.subprocess, 'call', return_value=0) as launch:
            Q.run(self.root, check_interval_s=1)
        wait.assert_called_once(); launch.assert_called_once()
        saved = json.loads((self.root/'QUEUE_STATUS.json').read_text())
        self.assertEqual((saved['status'], saved['execution_attempts']), ('completed',1))
        with self.assertRaises(ValueError): Q.run(self.root)

    def test_priority_failure_never_launches_pilot(self):
        self.finish_prior('1')
        with patch.object(Q.subprocess, 'call') as launch, self.assertRaises(ValueError): Q.run(self.root)
        launch.assert_not_called()
        saved = json.loads((self.root/'QUEUE_STATUS.json').read_text())
        self.assertEqual(saved['execution_attempts'],0)

    def test_wait_timeout_preserves_unattempted_status(self):
        with patch.object(Q.subprocess, 'call') as launch, self.assertRaises(TimeoutError):
            Q.run(self.root, maximum_wait_s=0)
        launch.assert_not_called()

    def test_pilot_failure_is_not_retried(self):
        self.finish_prior()
        with patch.object(Q.subprocess, 'call', return_value=1) as launch, self.assertRaises(RuntimeError):
            Q.run(self.root)
        launch.assert_called_once()
        self.assertEqual(json.loads((self.root/'QUEUE_STATUS.json').read_text())['status'],'failed')
