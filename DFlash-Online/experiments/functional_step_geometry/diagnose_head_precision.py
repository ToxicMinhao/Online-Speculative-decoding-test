"""Retained-fixture test of IEEE FP32 auxiliary head backward, with no Adam.

Operator controls freeze the actual BF16 incoming gradient. Complete-gradient
controls instead recompute each layout's own native logits and sensitivity.
Neither control changes the ordinary trainer or publishes a checkpoint.
"""
import argparse
from contextlib import contextmanager
import os
import time

import torch

from online_feedback import functional_geometry_math as geometry
from online_feedback.fsg_gradient_diagnostic import tensor_comparison
from online_feedback.fsg_head_precision import auxiliary_head, fp32_head_vjp
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json
from specforge.core.chunking import checkpointed_chunk_reduce
from .diagnose_gradients import parameter_vjp, run
from .native_oracles import gradient_comparison


BUDGET = dict(response_forward=2, head_backward=2, draft_backward=5,
              operator_gemm=7, optimizer_step=0, feedback_generation=0,
              teacher_preparation=0, serving_generation=0)


def precision_settings():
    """Observe the relevant settings without mixing old and new setters."""
    return dict(global_fp32=torch.backends.fp32_precision,
        cuda_matmul_fp32=torch.backends.cuda.matmul.fp32_precision,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        cuda_autocast=torch.is_autocast_enabled('cuda'),
        cpu_autocast=torch.is_autocast_enabled('cpu'),
        environment={key: os.environ.get(key) for key in
            ('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE')})


@contextmanager
def measured(costs, name, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        costs[name] = time.perf_counter() - started


def branch(hook, response, hidden, native_hidden, work, *, chunk_blocks):
    """One branch's complete native forward sensitivity and revised head VJP."""
    live = hidden.detach().clone().requires_grad_(True)
    weights, proposals, mask, margins = hook._scan_reference(live.detach(), response, chunk_blocks)
    indices = torch.arange(live.shape[1], device=live.device).reshape(1, -1)
    captured, incoming = {}, {}

    def terms(current, native, local_weights, block_indices):
        logits = auxiliary_head(hook.model.lm_head, current)
        start = int(block_indices[0, 0])

        def capture_incoming(value):
            if value.dtype != torch.bfloat16:
                raise ValueError('The actual head incoming gradient must be BF16')
            incoming[start] = value.detach().clone()
            return value

        logits.register_hook(capture_incoming)
        with torch.no_grad():
            native_logits = hook.model.lm_head(native)
            sensitivity, energy = geometry.geometry_terms(logits, native_logits, local_weights)
            risk, absolute = geometry.risk_terms(logits, native_logits, local_weights)
        proxy = (logits.float() * sensitivity).sum(dtype=torch.float32)
        if start not in captured:
            captured[start] = dict(pre_logits=logits.detach(), native_logits=native_logits.detach(),
                                   sensitivity=sensitivity.detach())
        return proxy, energy, risk, absolute

    with preserve_rng(), torch.enable_grad():
        proxy, energy, risk, absolute = checkpointed_chunk_reduce(terms,
            live, native_hidden.detach(), weights, indices, chunk_size=chunk_blocks,
            dim=1, checkpoint=True)
        with work.operation('head_backward'):
            adjoint, = torch.autograd.grad(proxy, live)
    geometry.finite(adjoint, 'revised head hidden gradient')
    combined = {name: torch.cat([captured[i][name] for i in sorted(captured)], dim=1)
                for name in ('pre_logits', 'native_logits', 'sensitivity')}
    actual_incoming = torch.cat([incoming[i] for i in sorted(incoming)], dim=1)
    expected_incoming = combined['sensitivity'].to(torch.bfloat16)
    if not torch.equal(actual_incoming, expected_incoming):
        raise ValueError('Native BF16 rounding at the head boundary was not preserved')
    return dict(combined, incoming=actual_incoming, adjoint=adjoint.detach(),
        weights=weights.detach(), proposals=proposals, mask=mask, margins=margins,
        energy=energy.detach().reshape(1), proxy=proxy.detach().reshape(1),
        risk=risk.detach().reshape(1), absolute=absolute.detach().reshape(1))


@torch.no_grad()
def operator_controls(incoming, weight, work, costs):
    """Seven products on exactly the same BF16 inputs; FP64 is independent."""
    if incoming.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError('Operator controls require represented BF16 inputs')
    if incoming.shape[:3] != (1, 2, 16):
        raise ValueError('Operator controls are sealed to the retained 32-row fixture')
    products = {}
    for arithmetic in ('native_bf16', 'ieee_fp32'):
        for layout, size in (('chunked', 1), ('full', 2)):
            label = arithmetic + '/' + layout
            rounded_parts, precast_parts = [], []
            with measured(costs, label, incoming.device):
                for start in range(0, 2, size):
                    chunk = incoming[:, start:start+size]
                    with work.operation('operator_gemm'):
                        if arithmetic == 'native_bf16':
                            rounded = torch.matmul(chunk, weight)
                            precast = rounded  # No claim to observe native GEMM's internal accumulator.
                        else:
                            rounded, precast = fp32_head_vjp(chunk, weight, return_precast=True)
                    geometry.finite(rounded, label)
                    rounded_parts.append(rounded)
                    precast_parts.append(precast)
            products[label] = dict(rounded=torch.cat(rounded_parts, 1),
                                   precast=torch.cat(precast_parts, 1))
    # This reference promotes the original represented operands, not a computed
    # FP32 product. Only the temporary frozen head copy is FP64.
    with measured(costs, 'fp64_reference', incoming.device):
        with torch.autocast(device_type=incoming.device.type, enabled=False):
            with work.operation('operator_gemm'):
                reference = torch.matmul(incoming.double(), weight.double())
    geometry.finite(reference, 'independent FP64 head product')
    rounded_reference = reference.to(torch.bfloat16)
    comparisons = {name: dict(raw_vs_fp64=tensor_comparison(values['precast'], reference),
        bf16_vs_rounded_fp64=tensor_comparison(values['rounded'], rounded_reference))
        for name, values in products.items()}
    comparisons['revised_chunked_vs_full'] = tensor_comparison(
        products['ieee_fp32/chunked']['rounded'], products['ieee_fp32/full']['rounded'])
    accuracy = all(comparisons['ieee_fp32/'+layout]['bf16_vs_rounded_fp64']['matches_pinned_tolerance']
        and comparisons['ieee_fp32/'+layout]['raw_vs_fp64']['discrepancy_norm']
            <= comparisons['native_bf16/'+layout]['raw_vs_fp64']['discrepancy_norm']
        for layout in ('chunked', 'full'))
    summary = dict(comparisons=comparisons, accuracy_check_passed=accuracy,
        policy='BF16 incoming and frozen weight -> IEEE FP32 product -> BF16 hidden gradient',
        native_control='direct BF16 product; not a rerun of the retained full-model baseline',
        input_shape=list(incoming.shape), weight_shape=list(weight.shape),
        transient_fp32_weight_bytes=weight.numel()*4,
        transient_fp64_weight_bytes=weight.numel()*8,
        precast_error_interpretation='descriptive accuracy; no new tolerance or qualified gate',
        fp64_reference='independent full product of BF16-valued operands promoted to FP64')
    adjoints = dict(chunked=products['ieee_fp32/chunked']['rounded'],
                    full=products['ieee_fp32/full']['rounded'], reference=rounded_reference)
    return summary, adjoints


def precision_case(case):
    hook, root, work = case.hook, case.root, case.work
    settings = precision_settings()
    if settings['environment']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] == '1':
        raise ValueError('A forced TF32 environment override prevents this IEEE FP32 test')
    costs, result = {}, dict(policy='auxiliary_head_ieee_fp32_from_bf16_v1')
    # Branches independently reconstruct logits, protected weights and sensitivity.
    with measured(costs, 'end_to_end_chunked_head', case.hidden.device):
        chunked = branch(hook, case.response, case.hidden, case.native_hidden, work, chunk_blocks=1)
    with measured(costs, 'end_to_end_full_head', case.hidden.device):
        full = branch(hook, case.response, case.hidden, case.native_hidden, work, chunk_blocks=0)
    result['branch_comparisons'] = {key: tensor_comparison(chunked[key], full[key])
        for key in ('pre_logits', 'native_logits', 'sensitivity', 'incoming', 'adjoint',
                    'weights', 'margins', 'energy', 'proxy', 'risk', 'absolute')}
    result['prefix_mask_exact'] = bool(torch.equal(chunked['mask'], full['mask']))
    result['proposals_exact'] = bool(torch.equal(chunked['proposals'], full['proposals']))
    result['protected_occurrences'] = [int(row['mask'].sum()) for row in (chunked, full)]
    write_json(root / 'BRANCHES.json', result)
    if not result['prefix_mask_exact'] or not torch.equal(chunked['weights'], full['weights']):
        raise ValueError('The retained exact prefix and weight agreement did not reproduce')
    if precision_settings() != settings:
        raise ValueError('Auxiliary head precision leaked into the draft backward')

    # The unchanged gate receives each branch's own incoming hidden gradient.
    with measured(costs, 'end_to_end_parameter_vjps', case.hidden.device):
        chunk_gradient = parameter_vjp(case.hidden, chunked['adjoint'], case.parameters, work)
        full_gradient = parameter_vjp(case.hidden, full['adjoint'], case.parameters, work)
        result['end_to_end_gradient'] = gradient_comparison(chunk_gradient, full_gradient, case.names)
    write_json(root / 'END_TO_END.json', result['end_to_end_gradient'])
    del chunk_gradient, full_gradient

    frozen_incoming = full['incoming'].clone()
    incoming_before = frozen_incoming.clone()
    head_weight = hook.model.lm_head.weight
    weight_version = head_weight._version
    operator, adjoints = operator_controls(frozen_incoming, head_weight, work, costs)
    operator['incoming_unchanged'] = torch.equal(frozen_incoming, incoming_before)
    operator['weight_version_unchanged'] = head_weight._version == weight_version
    if not operator['incoming_unchanged'] or not operator['weight_version_unchanged']:
        raise ValueError('Operator controls mutated their fixed operands')
    write_json(root / 'OPERATOR.json', operator)
    if precision_settings() != settings:
        raise ValueError('Operator precision settings were not restored')

    # Propagate all three fixed-input controls, with BF16 output at the head
    # boundary even for FP64. These are not independent-branch qualification.
    fixed_parameters = {}
    with measured(costs, 'fixed_input_parameter_vjps', case.hidden.device):
        reference_gradient = parameter_vjp(case.hidden, adjoints['reference'], case.parameters, work)
        for label in ('chunked', 'full'):
            gradient = parameter_vjp(case.hidden, adjoints[label], case.parameters, work)
            fixed_parameters[label+'_vs_fp64_rounded'] = gradient_comparison(
                gradient, reference_gradient, case.names)
            del gradient
        del reference_gradient
    write_json(root / 'FIXED_INPUT_PARAMETERS.json', fixed_parameters)
    result.update(operator_accuracy_passed=operator['accuracy_check_passed'],
        fixed_input_parameter_comparisons=fixed_parameters,
        small_fixture_complete_gradient_passed=result['end_to_end_gradient']['status'] == 'passed',
        precision_before=settings, precision_after=precision_settings(), costs=costs,
        ieee_precision_readback='asserted inside every revised FP32 product before matmul',
        original_gate_overridden=False, full_qualification_completed=False,
        untested='SCALAR, MathQA, production tile and partial shapes, resource and serving checks; pilot',
        fixed_control_does_not_substitute_for_complete_gradient=True)
    if result['precision_after'] != settings:
        raise ValueError('Auxiliary precision settings changed after complete diagnostic')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    run(args.plan, case_runner=precision_case, work_limits=BUDGET)
