"""CPU checks for signed localization, shared context and fixed reproduction."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from .localize_native import (
    localize_attention,
    localize_layers,
    require_reproduced_model,
    scalar_residual,
    tensor_dot,
)
from .native import WorkReceipt


class ContextLayer(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, *, hidden_states, target_hidden):
        return self.weight @ hidden_states + target_hidden.tanh()


class LocalizationTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_attention_uses_actual_input_directions_and_keeps_signed_defect(self):
        inputs = [torch.tensor([1.0, 2.0], dtype=torch.float64) for _ in range(3)]
        output = torch.tensor([2.0, 4.0], dtype=torch.float64)
        native = [dict(inputs=inputs, output=output)]
        directions = [
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            torch.tensor([2.0, 1.0], dtype=torch.float64),
            torch.tensor([-3.0, 0.5], dtype=torch.float64),
        ]
        differentiated = [
            dict(
                inputs=[
                    dict(primal=value, tangent=direction)
                    for value, direction in zip(inputs, directions)
                ],
                output=dict(primal=output, tangent=torch.tensor([1.0, 0.0], dtype=torch.float64)),
            )
        ]
        adjoints = {
            f'attention_0_{name}': torch.ones(2, dtype=torch.float64)
            for name in ('query', 'key', 'value', 'output')
        }
        rows = localize_attention(native, differentiated, adjoints)
        self.assertEqual(rows[0]['input_contributions'], dict(query=0.0, key=3.0, value=-2.5))
        self.assertEqual(rows[0]['signed_residual'], 0.5)
        changed = deepcopy(differentiated)
        changed[0]['inputs'][0]['primal'][0] += 1
        with self.assertRaisesRegex(ValueError, 'primal changed'):
            localize_attention(native, changed, adjoints)

    def test_failed_global_contractions_must_reproduce_exactly(self):
        previous = dict(
            duality=dict(forward_dot=21.2, native_backward_dot=21.7, absolute_error=0.5)
        )
        current = dict(native_primal_bit_equal=True, **deepcopy(previous))
        require_reproduced_model(current, previous)
        current['duality']['native_backward_dot'] += 1e-10
        with self.assertRaisesRegex(ValueError, 'did not reproduce'):
            require_reproduced_model(current, previous)

    def test_five_local_replays_isolate_each_shared_context_use(self):
        layers = nn.ModuleList(
            [
                ContextLayer(torch.tensor([[0.8, 0.1], [-0.2, 0.7]], dtype=torch.float64))
                for _ in range(5)
            ]
        )
        model = SimpleNamespace(layers=layers)
        parameters = dict(layers.named_parameters())
        context = torch.tensor([0.3, -0.5], dtype=torch.float64, requires_grad=True)
        hidden = torch.tensor([0.2, 0.8], dtype=torch.float64)
        hidden_tangent = torch.zeros_like(hidden)
        context_tangent = torch.tensor([0.4, -0.3], dtype=torch.float64)
        directions = [torch.full_like(parameter, 0.1) for parameter in parameters.values()]
        ordinary, differentiated, outputs = {}, {}, []
        for index, (layer, direction) in enumerate(zip(layers, directions)):
            name = f'layer_{index}'
            value = layer(hidden_states=hidden, target_hidden=context)
            tangent = layer.weight.detach() @ hidden_tangent + direction @ hidden.detach()
            tangent += (1 - context.detach().tanh().square()) * context_tangent
            ordinary[name] = dict(
                output=value.detach(),
                args=(),
                kwargs=dict(hidden_states=hidden.detach(), target_hidden=context.detach()),
            )
            differentiated[name] = dict(
                output=dict(primal=value.detach(), tangent=tangent),
                inputs=dict(
                    hidden_states=dict(primal=hidden.detach(), tangent=hidden_tangent),
                    target_hidden=dict(primal=context.detach(), tangent=context_tangent),
                ),
            )
            hidden, hidden_tangent = value, tangent
            outputs.append(value)
        cotangent = torch.tensor([0.6, -0.8], dtype=torch.float64)
        gradients = torch.autograd.grad(
            hidden, (*parameters.values(), *outputs, context), grad_outputs=cotangent
        )
        parameter_gradients = gradients[:5]
        adjoints = {f'layer_{index}': gradient for index, gradient in enumerate(gradients[5:10])}
        with tempfile.TemporaryDirectory() as directory:
            work = WorkReceipt(Path(directory) / 'WORK.json')
            rows = localize_layers(
                model,
                ordinary,
                differentiated,
                adjoints,
                parameters,
                parameter_gradients,
                directions,
                work,
                Path(directory) / 'LAYERS.json',
            )
            self.assertEqual(len(rows), 5)
            self.assertEqual(sum(work.value['completed'].values()), 10)
            self.assertEqual(work.value['native_adam_steps'], 0)
            self.assertEqual(work.value['pending'], [])
        self.assertTrue(all(row['local_parameter_gradients_bit_equal'] for row in rows))
        self.assertLess(max(abs(row['signed_residual']) for row in rows), 1e-14)
        shared_sum = sum(row['shared_context_input_contraction'] for row in rows)
        self.assertAlmostEqual(shared_sum, tensor_dot(gradients[-1], context_tangent), places=14)
        self.assertTrue(all(parameter.grad is None for parameter in parameters.values()))

    def test_shared_context_merge_has_the_correct_telescoping_sign(self):
        # The context producer sees the rounded sum of branch adjoints. Region
        # residuals therefore contain +merge; subtract it to recover the global defect.
        global_defect = -0.4
        global_context, separate_context = 1.2, 1.1
        merge = global_context - separate_context
        producer = scalar_residual(global_context, 0.0, 0.5)
        consumers = scalar_residual(0.6, separate_context, 0.5)
        reconstructed = producer['signed_residual'] + consumers['signed_residual'] - merge
        self.assertAlmostEqual(reconstructed, global_defect)


if __name__ == '__main__':
    unittest.main()
