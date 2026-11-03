"""Treatment diagnostics are counted once at the original-response boundary."""
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.kplus_full_gsm8k import report as R
from experiments.kplus_full_gsm8k import training as T
from experiments.kplus_full_gsm8k.campaign import ARM_LIMITS


class ReportTests(unittest.TestCase):
    def test_response_metrics_once_not_optimizer_accumulations(self):
        sums = dict(kplus_W1=1., kplus_W2=2., kplus_response_count=1.,
            kplus_mixed_response=1., kplus_error_weight_moved=.3,
            kplus_error_weight_moved_share=.1, kplus_conservation_error=0.)
        records = [dict(**{'pass': i // 128}, prompt_tokens=2, response_tokens=3, active_anchors=4,
            metrics={'kplus_W1': 999.}, response_observation={'sum_metrics': sums}) for i in range(256)]
        status = dict(status='completed', events_completed=58, tail_completed=49,
            actual_work=dict(attempts=ARM_LIMITS, completed=ARM_LIMITS, unknown=dict.fromkeys(ARM_LIMITS, 0)),
            event_costs=[], worker_wall_s=1., backend_loading_s=1., trainer_loading_s=1., initial_validation_s=1.)
        def read(path):
            path = Path(path)
            if path.name == 'STATUS.json':
                return status
            if path.name == 'RESULT.json':
                return dict(records=records, objective_work={})
            if path.parent.name == 'boundaries':
                return dict(status='passed', event=int(path.stem.split('_')[1]))
            return {}
        with patch.object(R, 'read', side_effect=read), patch.object(T, 'validate_event_receipt'):
            result = R.training_summary(Path('/fixture'))
        totals = result['exposure']
        self.assertEqual(totals['presentations'], 14848)
        self.assertEqual(totals['kplus_W1'], 14848)
        self.assertEqual(totals['kplus_W2'], 29696)
        self.assertEqual(totals['mixed_response_frequency'], 1.)
        self.assertAlmostEqual(totals['pooled_error_weight_moved_share'], .1)
        self.assertAlmostEqual(totals['mean_response_moved_share'], .1)
        self.assertEqual(len(result['pass_rows']), 116)
        self.assertEqual(len(result['objective_work']), 58)


if __name__ == '__main__':
    unittest.main()
