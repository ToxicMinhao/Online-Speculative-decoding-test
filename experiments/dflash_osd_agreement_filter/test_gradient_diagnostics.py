"""Numerical evidence survives failure without weakening the existing gate."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

_spec = importlib.util.spec_from_file_location('agreement_gradient_diagnostic_fixture',
                                               Path(__file__).with_name('gradient_diagnostics.py'))
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def inputs(a, b, reference, *, dtype=torch.bfloat16):
    parameters = {name: torch.nn.Parameter(torch.zeros_like(torch.as_tensor(values), dtype=dtype))
                  for name, values in a.items()}
    return ({name: torch.as_tensor(values, dtype=torch.float32) for name, values in a.items()},
            {name: torch.as_tensor(values, dtype=torch.float32) for name, values in b.items()},
            {name: torch.as_tensor(values, dtype=torch.float32) for name, values in reference.items()}, parameters)


class GradientDiagnosticTests(unittest.TestCase):
    def test_failing_summary_is_durable_before_original_failure_and_counts_all_parameters(self):
        args = inputs({'first': [1., 2.], 'second': [[0., 3.]]},
                      {'first': [-1., -2.], 'second': [[0., -3.]]},
                      {'first': [2., 0.], 'second': [[1., 0.]]})
        context = dict(event=1, pass_index=0, trial_type='disabled_filter',
                       anchors_sha256='anchor-identity', state_sha256='state-identity',
                       denominator=4., full_loss=2., loss_a=.75, loss_b=1.25)
        summary = diag.summarize_gradient_sum(*args, context=context)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'failure.json'
            diag.durable_write(path, summary)
            with self.assertRaisesRegex(ValueError, '^Grouped raw gradients differ beyond declared sum tolerance$'):
                diag.enforce_gradient_sum(summary)
            recorded = json.loads(path.read_text())
        self.assertEqual(recorded['context'], context)
        self.assertEqual(recorded['status'], 'failed')
        self.assertEqual(recorded['coordinates'], 4)
        self.assertEqual(recorded['tolerance_violation_coordinates'], 2)
        self.assertEqual(recorded['tolerance_violation_fraction'], .5)
        self.assertEqual(recorded['first_failure']['parameter'], 'first')
        self.assertEqual(len(recorded['per_parameter']), 2)
        self.assertEqual([row['tolerance_violation_coordinates'] for row in recorded['per_parameter']], [1, 1])
        self.assertEqual(recorded['samples'][1]['index'], [0, 0])
        self.assertEqual(recorded['per_parameter'][1]['shape'], [1, 2])
        self.assertEqual(recorded['per_parameter'][1]['parameter_dtype'], 'torch.bfloat16')
        self.assertEqual(recorded['per_parameter'][1]['raw_a_dtype'], 'torch.float32')

    def test_acceptance_is_identical_to_original_fp32_rule(self):
        for dtype in (torch.bfloat16, torch.float32):
            for scale in (0., 1e-6, 1., 1e6):
                rng = torch.Generator().manual_seed(17)
                a = torch.randn(73, generator=rng) * scale
                b = torch.randn(73, generator=rng) * scale
                reference = (a + b).to(dtype).float()
                reference[0] += 1.
                args = inputs({'p': a}, {'p': b}, {'p': reference}, dtype=dtype)
                summary = diag.summarize_gradient_sum(*args)
                error = (a+b-reference).abs()
                tolerance = 8*torch.finfo(dtype).eps*(a.abs()+b.abs()+reference.abs())+1e-8
                failed = ~(error <= tolerance)
                self.assertEqual(summary['status'] == 'passed', not bool(failed.any()))
                self.assertEqual(summary['tolerance_violation_coordinates'], int(failed.sum()))
                self.assertEqual(summary['max_abs_error'], float(error.max()))
                self.assertEqual(summary['max_error_over_tolerance'], float((error/tolerance).max()))
                if bool(failed.any()):
                    with self.assertRaisesRegex(ValueError, 'declared sum tolerance'):
                        diag.enforce_gradient_sum(summary)
                else:
                    self.assertIs(diag.enforce_gradient_sum(summary), summary)

    def test_global_and_parameter_norms_are_chunked_fp64(self):
        args = inputs({'a': [3., 0., 0.], 'b': [[0., 4.]]},
                      {'a': [0., 4., 0.], 'b': [[3., 0.]]},
                      {'a': [3., 4., 0.], 'b': [[3., 4.]]})
        with patch.object(diag, 'CHUNK_ELEMENTS', 2):
            summary = diag.summarize_gradient_sum(*args)
        self.assertEqual(summary['status'], 'passed')
        self.assertEqual(summary['l2_norms']['a'], 5.)
        self.assertEqual(summary['l2_norms']['b'], 5.)
        self.assertAlmostEqual(summary['reference_l2_norm'], math.sqrt(50))
        self.assertEqual(summary['l2_norms']['discrepancy'], 0.)
        self.assertEqual(summary['per_parameter'][0]['l2_norms']['sum'], 5.)
        self.assertEqual(summary['per_parameter'][1]['l2_norms']['sum'], 5.)
        self.assertEqual(summary['reduction_chunk_elements'], 2)
        self.assertEqual(summary['samples'], [])

    def test_nonfinite_values_are_explicit_strict_json_markers_and_fail(self):
        args = inputs({'p': [float('nan'), float('inf'), -float('inf')]},
                      {'p': [0., 0., 0.]}, {'p': [0., 0., 0.]})
        summary = diag.summarize_gradient_sum(*args, context={'full_loss': float('nan')})
        self.assertEqual(summary['nonfinite_input_coordinates'], 3)
        self.assertEqual(summary['rejected_coordinates'], 3)
        self.assertEqual(summary['nonfinite_counts']['a'], 3)
        self.assertEqual(summary['samples'][0]['g_a']['nonfinite_float'], 'nan')
        self.assertEqual(summary['samples'][1]['g_a']['nonfinite_float'], 'inf')
        self.assertEqual(summary['samples'][2]['g_a']['nonfinite_float'], '-inf')
        self.assertEqual(len(summary['samples'][0]['g_a']['ieee754_binary64_hex']), 16)
        self.assertIn('nonfinite_float', summary['max_abs_error'])
        json.dumps(summary, allow_nan=False)
        with self.assertRaisesRegex(ValueError, '^Nonfinite gradient qualification$'):
            diag.enforce_gradient_sum(summary)

    def test_samples_are_globally_bounded_and_do_not_dump_gradients(self):
        args = inputs({'a': [0.] * 12, 'b': [0.] * 19},
                      {'a': [0.] * 12, 'b': [0.] * 19},
                      {'a': [1.] * 12, 'b': [2.] * 19})
        with patch.object(diag, 'CHUNK_ELEMENTS', 4):
            summary = diag.summarize_gradient_sum(*args, sample_limit=3)
        self.assertEqual(summary['rejected_coordinates'], 31)
        self.assertEqual(len(summary['samples']), 3)
        self.assertEqual([row['flat_index'] for row in summary['samples']], [0, 1, 2])
        for row in summary['per_parameter']:
            self.assertNotIn('gradients', row)
            self.assertNotIn('values', row)
        self.assertEqual(diag.summarize_gradient_sum(*args, sample_limit=0)['samples'], [])

    def test_durable_writer_fsyncs_file_and_parent_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            with patch.object(diag.os, 'fsync', wraps=diag.os.fsync) as sync:
                diag.durable_write(path, {'value': float('inf')})
            self.assertEqual(sync.call_count, 2)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            self.assertEqual(json.loads(path.read_text())['value']['nonfinite_float'], 'inf')

    def test_mismatched_inventory_shape_and_invalid_sample_bound_are_not_accepted(self):
        args = inputs({'p': [1.]}, {'p': [0.]}, {'p': [1.]})
        with self.assertRaisesRegex(ValueError, 'inventories'):
            diag.summarize_gradient_sum({}, *args[1:])
        with self.assertRaisesRegex(ValueError, 'shape/device'):
            diag.summarize_gradient_sum({'p': torch.zeros(2)}, *args[1:])
        for limit in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'sample_limit'):
                diag.summarize_gradient_sum(*args, sample_limit=limit)


if __name__ == '__main__':
    unittest.main()
