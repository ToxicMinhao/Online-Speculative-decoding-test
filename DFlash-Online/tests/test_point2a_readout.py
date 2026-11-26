"""CPU checks for the narrow Point2A endpoint boundary and inherited arithmetic."""

from copy import deepcopy
import unittest

import torch

from experiments.frontier_preservation_pilot import point1_readout, point2a_readout
from experiments.frontier_preservation_pilot.native import rng_digest
from online_feedback.rng import capture_rng
from test_point1_readout import fixture


class Point2AReadoutTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_new_labels_validate_without_disguising_old_labels(self):
        trainer, response = fixture(1)
        for endpoint in point2a_readout.ENDPOINTS:
            result = point2a_readout.evaluate_responses(trainer, [response], endpoint=endpoint)
            self.assertEqual(result['endpoint'], endpoint)
            self.assertEqual(result['schema'], 'point2a_native_endpoint_readout_v1')
        for endpoint in ('N_F_BETA', 'N_F_C'):
            with self.assertRaisesRegex(ValueError, 'Unknown Point1 endpoint'):
                point1_readout.evaluate_responses(trainer, [response], endpoint=endpoint)
        for endpoint in ('F_C', 'F_BETA', 'N_F', 'N_BETA', 'Q_C', 'Q_F_C', 'unknown'):
            with self.assertRaisesRegex(ValueError, 'Unknown Point2A endpoint'):
                point2a_readout.evaluate_responses(trainer, [response], endpoint=endpoint)

    def test_all_legacy_endpoints_and_arithmetic_remain_unchanged(self):
        trainer, response = fixture(9)
        parent = point2a_readout.evaluate_responses(trainer, [response], endpoint='parent')
        for endpoint in point1_readout.ENDPOINTS:
            legacy = point1_readout.evaluate_responses(trainer, [response], endpoint=endpoint)
            self.assertEqual(legacy['schema'], 'point1_native_endpoint_readout_v1')
            for field in ('responses', 'mean_response_assessment_fp32', 'mean_response_assessment_fp64'):
                self.assertEqual(legacy[field], parent[field])

    def test_functional_native_readouts_keep_frozen_origins_and_state(self):
        trainer, response = fixture(1)
        model = trainer.bundle.model
        model.train()
        model.lm_head.eval()
        parameter = next(model.draft_model.parameters())
        parameter.grad = torch.ones_like(parameter)
        old_gradient = parameter.grad
        before = parameter.detach().clone()
        old_rng = rng_digest(capture_rng())
        parent = point2a_readout.evaluate_responses(trainer, [response], endpoint='parent')
        changed = before.float().clone()
        changed[4, 31] = 3
        values = {'model.draft_model.weight': changed}
        candidate = point2a_readout.evaluate_responses(
            trainer, [response], endpoint='N_F_C', origins=parent, parameters=values,
        )
        self.assertEqual(parent['responses'][0]['blocks'][0]['origin'],
                         candidate['responses'][0]['blocks'][0]['origin'])
        self.assertGreater(point2a_readout.compare_readouts(parent, candidate)['assessment_benefit_fp64'], 0)
        self.assertTrue(torch.equal(parameter, before))
        self.assertIs(parameter.grad, old_gradient)
        self.assertEqual((trainer.global_step, trainer.presentations), (128, 256))
        self.assertEqual(trainer.work, {'old': 7})
        self.assertEqual(model.costs, {'old': 3})
        self.assertEqual(model.scratch.item(), 7)
        self.assertTrue(model.training)
        self.assertFalse(model.lm_head.training)
        self.assertEqual(rng_digest(capture_rng()), old_rng)

    def test_failure_restores_existing_runtime(self):
        trainer, response = fixture(1)
        model = trainer.bundle.model
        model.fail = True
        old_rng = rng_digest(capture_rng())
        with self.assertRaisesRegex(RuntimeError, 'native fixture failure'):
            point2a_readout.evaluate_responses(trainer, [response], endpoint='N_F_C')
        self.assertEqual(rng_digest(capture_rng()), old_rng)
        self.assertEqual(model.scratch.item(), 7)
        self.assertEqual(model._cached_seq_len, 11)
        self.assertEqual(trainer._anchor_observations, [{'old': 4}])
        self.assertEqual(trainer.work, {'old': 7})

    def test_all_three_paired_contrasts_have_correct_direction(self):
        trainer, response = fixture(1)
        parent = point2a_readout.evaluate_responses(trainer, [response], endpoint='parent')
        results = {}
        for name, reduction in (('C', 0.0), ('F_BETA', 0.25), ('F_C', 0.75)):
            row = deepcopy(parent)
            row['endpoint'] = 'N_' + name
            row['mean_response_assessment_fp32'] -= reduction
            row['responses'][0]['blocks'][0]['positions'][0]['assessment_contribution_fp32'] -= reduction
            results[row['endpoint']] = row
        contrasts = point2a_readout.paired_contrasts(results)
        self.assertEqual(set(contrasts), {'F_C-F_BETA', 'F_C-C', 'F_BETA-C'})
        for name, benefit in (('F_C-F_BETA', 0.5), ('F_C-C', 0.75), ('F_BETA-C', 0.25)):
            self.assertAlmostEqual(contrasts[name]['assessment_benefit_fp64'], benefit)
            self.assertAlmostEqual(contrasts[name]['assessment_benefit_original_fp32'], benefit)


if __name__ == '__main__':
    unittest.main()
