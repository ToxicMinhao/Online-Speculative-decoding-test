"""Localize a retained whole-model adjoint discrepancy without an update.

Every diagnostic uses the failed probe's single global cotangent and parameter
direction. Local residuals are signed accounting quantities, not new pass gates.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

from online_feedback.rng import restore_rng
from online_feedback.storage import write_json
from .attention_jvp import native_attention_jvp
from .repair_native import (
    aligned_direction,
    duality_receipt,
    fixed_cotangent,
    tensor_dot,
    vector_dot,
)


def scalar_residual(output_contraction, input_contraction, parameter_contraction=0.0):
    residual = output_contraction - input_contraction - parameter_contraction
    return dict(
        output_contraction=output_contraction,
        input_contraction=input_contraction,
        parameter_contraction=parameter_contraction,
        signed_residual=residual,
        relative_to_output=residual / abs(output_contraction) if output_contraction else None,
    )


def split_dual(value):
    pair = torch.autograd.forward_ad.unpack_dual(value)
    return dict(
        primal=pair.primal.detach(), tangent=None if pair.tangent is None else pair.tangent.detach()
    )


def tangent_dot(adjoint, dual):
    tangent = dual['tangent']
    return 0.0 if tangent is None else tensor_dot(adjoint, tangent)


@contextmanager
def capture_boundaries(model, *, dual):
    """Observe existing tensors; no backward hooks or new autograd operators."""
    import specforge.modeling.draft.dflash as draft

    original = draft.compile_friendly_flex_attention
    captured = dict(attention=[], regions={})

    def retain(value):
        return split_dual(value) if dual else value

    def observe_attention(query, key, value, **kwargs):
        output = original(query, key, value, **kwargs)
        captured['attention'].append(
            dict(
                inputs=[retain(item) for item in (query, key, value)],
                output=retain(output),
            )
        )
        return output

    def hook(name):
        def observe(module, args, kwargs, output):
            inputs = list(args)
            tensors = {}
            for key in ('hidden_states', 'target_hidden'):
                if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                    tensors[key] = retain(kwargs[key])
            captured['regions'][name] = dict(
                output=retain(output),
                inputs=tensors,
                args=tuple(
                    item.detach() if isinstance(item, torch.Tensor) else item for item in inputs
                ),
                kwargs={
                    key: item.detach() if isinstance(item, torch.Tensor) else item
                    for key, item in kwargs.items()
                },
            )

        return observe

    handles = []
    draft.compile_friendly_flex_attention = observe_attention
    selected = [('fc', model.fc), ('hidden_norm', model.hidden_norm), ('norm', model.norm)]
    selected.extend((f'layer_{index}', layer) for index, layer in enumerate(model.layers))
    for name, module in selected:
        handles.append(module.register_forward_hook(hook(name), with_kwargs=True))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()
        draft.compile_friendly_flex_attention = original


def boundary_targets(captured):
    targets = {}
    for index, attention in enumerate(captured['attention']):
        for name, value in zip(('query', 'key', 'value'), attention['inputs']):
            targets[f'attention_{index}_{name}'] = value
        targets[f'attention_{index}_output'] = attention['output']
    for name, region in captured['regions'].items():
        targets[name] = region['output']
    return targets


def require_reproduced_model(current, previous):
    if not current['native_primal_bit_equal']:
        raise ValueError('Localization repaired JVP changed the native model primal')
    keys = ('forward_dot', 'native_backward_dot', 'absolute_error')
    if any(current['duality'][key] != previous['duality'][key] for key in keys):
        raise ValueError('Localization did not reproduce the failed global derivative contractions')


def localize_attention(native, dual, adjoints):
    rows = []
    for index, (ordinary, differentiated) in enumerate(zip(native, dual)):
        parity = all(
            torch.equal(left, right['primal'])
            for left, right in zip(ordinary['inputs'], differentiated['inputs'])
        )
        parity = parity and torch.equal(ordinary['output'], differentiated['output']['primal'])
        if not parity:
            raise ValueError('Localization attention boundary primal changed')
        contributions = {
            name: tangent_dot(adjoints[f'attention_{index}_{name}'], direction)
            for name, direction in zip(('query', 'key', 'value'), differentiated['inputs'])
        }
        left = tangent_dot(adjoints[f'attention_{index}_output'], differentiated['output'])
        row = scalar_residual(left, math.fsum(contributions.values()))
        row.update(layer=index, native_primal_bit_equal=True, input_contributions=contributions)
        rows.append(row)
    return rows


def parameter_contraction(module, named_parameters, gradients, directions):
    identities = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    indices = [
        index for index, value in enumerate(named_parameters.values()) if id(value) in identities
    ]
    return vector_dot(
        [gradients[index] for index in indices], [directions[index] for index in indices]
    )


def localize_layers(
    model,
    ordinary,
    differentiated,
    adjoints,
    named_parameters,
    gradients,
    directions,
    work,
    destination,
):
    """Five fixed local replays isolate each use of the shared target context."""
    rows = []
    for index, layer in enumerate(model.layers):
        name = f'layer_{index}'
        record, dual = ordinary[name], differentiated[name]
        kwargs = dict(record['kwargs'])
        for key in ('hidden_states', 'target_hidden'):
            kwargs[key] = kwargs[key].detach().requires_grad_(True)
        local_parameters = tuple(
            parameter for parameter in layer.parameters() if parameter.requires_grad
        )
        work.start(name + '_local_native_forward')
        output = layer(*record['args'], **kwargs)
        work.finish(name + '_local_native_forward')
        if not torch.equal(output, record['output']):
            raise ValueError('Local decoder replay changed the saved native primal')
        work.start(name + '_local_native_vjp')
        values = torch.autograd.grad(
            output,
            (kwargs['hidden_states'], kwargs['target_hidden'], *local_parameters),
            grad_outputs=adjoints[name],
        )
        work.finish(name + '_local_native_vjp')
        hidden_contraction = tangent_dot(values[0], dual['inputs']['hidden_states'])
        context_contraction = tangent_dot(values[1], dual['inputs']['target_hidden'])
        local_parameter_ids = {
            id(parameter): position for position, parameter in enumerate(local_parameters)
        }
        local_directions = [None] * len(local_parameters)
        global_gradients = [None] * len(local_parameters)
        for parameter, gradient, direction in zip(named_parameters.values(), gradients, directions):
            position = local_parameter_ids.get(id(parameter))
            if position is not None:
                local_directions[position] = direction
                global_gradients[position] = gradient
        parameter_dot = vector_dot(values[2:], local_directions)
        output_dot = tangent_dot(adjoints[name], dual['output'])
        row = scalar_residual(output_dot, hidden_contraction + context_contraction, parameter_dot)
        row.update(
            layer=index,
            native_primal_bit_equal=True,
            hidden_input_contraction=hidden_contraction,
            shared_context_input_contraction=context_contraction,
            local_parameter_gradients_bit_equal=all(
                torch.equal(a, b) for a, b in zip(values[2:], global_gradients)
            ),
            global_parameter_contraction=vector_dot(global_gradients, local_directions),
        )
        rows.append(row)
        write_json(destination, dict(status='running', layers=rows))
        del output, values, kwargs
    return rows


def run_localization(module, response, parameters, work, output, failed_repair):
    """One global native VJP and repaired JVP, then at most five local replays."""
    output = Path(output)
    previous = json.loads((Path(failed_repair) / 'MODEL.json').read_text())
    if previous['duality']['passed'] or not previous['native_primal_bit_equal']:
        raise ValueError(
            'Localization requires the retained failed duality with exact native primal'
        )
    model = module.model.draft_model
    restore_rng(response.rng_after_sampling)
    work.start('localization_native_model_forward')
    with capture_boundaries(model, dual=False) as ordinary:
        baseline = module(response)
    work.finish('localization_native_model_forward')
    if len(ordinary['attention']) != 5 or len(ordinary['regions']) != 8:
        raise ValueError('Expected five native attention and eight top-level region boundaries')
    targets = boundary_targets(ordinary)
    cotangent = fixed_cotangent(baseline)
    work.start('localization_native_model_vjp')
    derivatives = torch.autograd.grad(
        baseline,
        (*parameters.values(), *targets.values()),
        grad_outputs=cotangent,
    )
    work.finish('localization_native_model_vjp')
    gradients = derivatives[: len(parameters)]
    adjoints = dict(zip(targets, derivatives[len(parameters) :]))
    directions = aligned_direction(gradients)
    baseline = baseline.detach()
    for record in ordinary['attention']:
        record['inputs'] = [value.detach() for value in record['inputs']]
        record['output'] = record['output'].detach()
    for record in ordinary['regions'].values():
        record['output'] = record['output'].detach()
        record['inputs'] = {key: value.detach() for key, value in record['inputs'].items()}
    del derivatives, targets
    restore_rng(response.rng_after_sampling)
    work.start('localization_repaired_model_jvp')
    with native_attention_jvp() as accounting:
        with capture_boundaries(model, dual=True) as differentiated:
            with torch.no_grad(), torch.autograd.forward_ad.dual_level():
                dual_parameters = {
                    name: torch.autograd.forward_ad.make_dual(parameter, direction)
                    for (name, parameter), direction in zip(parameters.items(), directions)
                }
                result = torch.func.functional_call(module, dual_parameters, (response,))
                primal, tangent = torch.autograd.forward_ad.unpack_dual(result)
                current = dict(
                    native_primal_bit_equal=torch.equal(primal, baseline),
                    duality=duality_receipt(tangent, cotangent, gradients, directions),
                )
    work.finish('localization_repaired_model_jvp')
    write_json(output / 'REPRODUCTION.json', current)
    require_reproduced_model(current, previous)
    if len(differentiated['attention']) != 5 or len(differentiated['regions']) != 8:
        raise ValueError('Repaired JVP lost required native diagnostic boundaries')
    if any(
        not torch.equal(record['output'], differentiated['regions'][name]['output']['primal'])
        for name, record in ordinary['regions'].items()
    ):
        raise ValueError('Repaired JVP changed a top-level native boundary primal')
    attention = localize_attention(ordinary['attention'], differentiated['attention'], adjoints)
    attention_sum = math.fsum(row['signed_residual'] for row in attention)
    global_residual = current['duality']['forward_dot'] - current['duality']['native_backward_dot']
    receipt = dict(
        status='attention_localized',
        failed_repair=str(failed_repair),
        global_reproduction_exact=True,
        global_signed_residual=global_residual,
        attention=attention,
        attention_signed_residual_sum=attention_sum,
        residual_outside_attention=global_residual - attention_sum,
        derivative_scope_accounting=accounting,
        local_acceptance_gate=False,
        native_adam_steps=0,
    )
    write_json(output / 'LOCALIZATION.json', receipt)
    if global_residual != attention_sum:
        layers = localize_layers(
            model,
            ordinary['regions'],
            differentiated['regions'],
            adjoints,
            parameters,
            gradients,
            directions,
            work,
            output / 'LAYER_LOCALIZATION.json',
        )
        dual_regions = differentiated['regions']
        shared_context_merge = tangent_dot(
            adjoints['hidden_norm'], dual_regions['hidden_norm']['output']
        )
        shared_context_merge -= math.fsum(row['shared_context_input_contraction'] for row in layers)
        chain = {}
        for name, previous_name, submodule in (
            ('fc', None, model.fc),
            ('hidden_norm', 'fc', model.hidden_norm),
            ('norm', 'layer_4', model.norm),
        ):
            left = tangent_dot(adjoints[name], dual_regions[name]['output'])
            right = (
                0.0
                if previous_name is None
                else tangent_dot(adjoints[previous_name], dual_regions[previous_name]['output'])
            )
            parameter_dot = parameter_contraction(submodule, parameters, gradients, directions)
            chain[name] = scalar_residual(left, right, parameter_dot)
        region_sum = math.fsum(row['signed_residual'] for row in layers)
        region_sum += math.fsum(row['signed_residual'] for row in chain.values())
        # Layer replays use each context edge independently. The native global
        # backward rounds their shared-input accumulation; account for that merge.
        reconstructed = region_sum - shared_context_merge
        receipt.update(
            layers=layers,
            chain_regions=chain,
            shared_context_merge_residual=shared_context_merge,
            reconstructed_global_residual=reconstructed,
            reconstruction_remainder=global_residual - reconstructed,
        )
        write_json(output / 'LAYER_LOCALIZATION.json', dict(status='completed', layers=layers))
    receipt['status'] = 'completed'
    write_json(output / 'LOCALIZATION.json', receipt)
    return receipt
