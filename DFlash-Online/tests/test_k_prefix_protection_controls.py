"""Independent CPU checks for experiment admission and recovery chronology."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from experiments.k_prefix_protection_v1 import campaign, preflight, storage
from test_k_prefix_protection_training import config


class AdmissionControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def configs(self):
        plan = dict(configs={}, original_configs={}, draft='/released', target='/target')
        for seed in (17, 29):
            source = yaml.safe_load(config(seed))
            source['model']['target_model_path'] = '/target'
            original = self.root / f'original_{seed}.yaml'
            original.write_text(yaml.safe_dump(source))
            plan['original_configs'][str(seed)] = str(original)
            plan['configs'][str(seed)] = {}
            for arm, lr in (('KP', 3e-4), ('K_HALF', 1.5e-4)):
                raw = deepcopy(source)
                raw['run_id'] = f'new-{seed}-{arm}'
                raw['output_dir'] = f'/output/{seed}/{arm}'
                raw['training']['learning_rate'] = lr
                path = self.root / f'{seed}_{arm}.yaml'
                path.write_text(yaml.safe_dump(raw))
                plan['configs'][str(seed)][arm] = str(path)
        return plan

    def test_exact_original_configs_allow_only_identity_and_half_lr(self):
        plan = self.configs()
        self.assertEqual(len(preflight.check_configs(plan)), 4)
        path = Path(plan['configs']['17']['KP'])
        raw = yaml.safe_load(path.read_text())
        raw['model']['unexpected_attention_override'] = 'other'
        path.write_text(yaml.safe_dump(raw))
        with self.assertRaisesRegex(ValueError, 'Unexpected configuration'):
            preflight.check_configs(plan)

    def test_even_matching_original_and_new_paths_must_bind_pinned_models(self):
        plan = self.configs()
        for path in [plan['original_configs']['17'], *plan['configs']['17'].values()]:
            raw = yaml.safe_load(Path(path).read_text())
            raw['model']['draft_checkpoint_path'] = '/wrong-release'
            Path(path).write_text(yaml.safe_dump(raw))
        with self.assertRaisesRegex(ValueError, 'Model paths'):
            preflight.check_configs(plan)

    def materialize_jobs(self, seeds=(17,), end_event=4):
        for seed in seeds:
            for event in range(1, end_event + 1):
                for arm in campaign.ARMS:
                    audit, _ = campaign.job_paths(self.root, seed, arm, event)
                    audit.mkdir(parents=True)

    def test_recovery_rejects_seed29_work_without_seed17_evaluation(self):
        self.materialize_jobs()
        audit, _ = campaign.job_paths(self.root, 29, 'K_HALF', 1)
        audit.mkdir(parents=True)
        with mock.patch.object(campaign, 'completed_job', return_value=({}, {})):
            with self.assertRaisesRegex(ValueError, 'earlier evaluation'):
                campaign.inspect_boundary(self.root, {})

    def test_recovery_accepts_complete_seed17_and_then_next_seed(self):
        self.materialize_jobs()
        evaluation = self.root/'campaign_001/seed_17/event_004/evaluation'
        evaluation.mkdir(parents=True)
        audit, _ = campaign.job_paths(self.root, 29, 'K_HALF', 1)
        audit.mkdir(parents=True)
        with mock.patch.object(campaign, 'completed_job', return_value=({}, {})), \
             mock.patch('experiments.k_prefix_protection_v1.report.audit_cell',
                        return_value=([{}] * 128, [])):
            result = campaign.inspect_boundary(self.root, {})
        self.assertEqual(len(result['completed_jobs']), 9)
        self.assertEqual(result['evaluation_complete'], [17])

    def test_recovery_stops_on_gap_or_partial_job_before_capacity_credit(self):
        audit, _ = campaign.job_paths(self.root, 17, 'KP', 1)
        audit.mkdir(parents=True)
        with mock.patch.object(campaign, 'completed_job', return_value=({}, {})):
            with self.assertRaisesRegex(ValueError, 'chronology gap'):
                campaign.inspect_boundary(self.root, {})
        first, _ = campaign.job_paths(self.root, 17, 'K_HALF', 1)
        first.mkdir(parents=True)
        with mock.patch.object(campaign, 'completed_job', side_effect=ValueError('partial work')):
            with self.assertRaisesRegex(ValueError, 'partial work'):
                campaign.inspect_boundary(self.root, {})

    def test_capacity_uses_no_existing_deletion_credit_and_keeps_reserve(self):
        fresh = storage.future_capacity()
        partial = storage.future_capacity(completed_jobs=1, completed_initials=1,
                                          qualification_complete=True)
        finished = storage.future_capacity(completed_jobs=16, completed_initials=4,
                                           qualification_complete=True)
        self.assertGreater(fresh['peak_required_bytes'], partial['peak_required_bytes'])
        self.assertGreater(partial['peak_required_bytes'], finished['peak_required_bytes'])
        for row in (fresh, partial, finished):
            self.assertEqual(row['preexisting_deletion_credit_bytes'], 0)
            self.assertEqual(row['reserve_bytes'], 8 * storage.GIB)
            self.assertGreaterEqual(row['peak_required_bytes'], row['reserve_bytes'])


if __name__ == '__main__':
    unittest.main()
