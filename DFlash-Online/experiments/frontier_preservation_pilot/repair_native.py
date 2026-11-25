"""Zero-update qualification of the scoped native-attention JVP repair.

The original failed probe is immutable. This attempt reuses its exact parent,
responses and anchors, and measures the repaired derivative against ordinary
native backward before any coefficient or optimizer composition is admitted.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from experiments.full_k_gsm8k.training import verify_live
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot.training import PilotTrainer
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import write_json

from .native import (
    NativeHidden,
    PARENT_COMMIT,
    PARENT_RECOVERY,
    WorkReceipt,
    freeze_two_responses,
    rng_digest,
    select_parent,
    trainable_parameters,
)

DUALITY_ATOL = 1e-5
DUALITY_RTOL = 0.016
DIAGNOSTIC_SEED = 20261125


def tensor_dot(left, right):
    """FP64 reduction of the products of retained finite native values."""
    return float((left.detach().double() * right.detach().double()).sum())


def vector_dot(left, right):
    return math.fsum(tensor_dot(a, b) for a, b in zip(left, right))


def duality_receipt(tangent, cotangent, gradients, direction):
    forward_dot = tensor_dot(tangent, cotangent)
    backward_dot = vector_dot(gradients, direction)
    error = abs(forward_dot - backward_dot)
    tolerance = DUALITY_ATOL + DUALITY_RTOL * abs(backward_dot)
    tangent_norm = math.sqrt(tensor_dot(tangent, tangent))
    cotangent_norm = math.sqrt(tensor_dot(cotangent, cotangent))
    gradient_norm = math.sqrt(vector_dot(gradients, gradients))
    direction_norm = math.sqrt(vector_dot(direction, direction))
    scale = max(tangent_norm * cotangent_norm, gradient_norm * direction_norm)
    return dict(
        passed=math.isfinite(error) and backward_dot > 0 and error <= tolerance,
        forward_dot=forward_dot,
        native_backward_dot=backward_dot,
        absolute_error=error,
        relative_error=error / abs(backward_dot) if backward_dot else None,
        cauchy_normalized_error=error / scale if scale else None,
        tolerance=tolerance,
        atol=DUALITY_ATOL,
        rtol=DUALITY_RTOL,
        tangent_norm=tangent_norm,
        cotangent_norm=cotangent_norm,
        native_gradient_norm=gradient_norm,
        direction_norm=direction_norm,
        direction='normalized native J^T u, rounded to native dtype',
    )


def aligned_direction(gradients):
    if not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise ValueError('Native diagnostic gradients are nonfinite')
    norm = math.sqrt(vector_dot(gradients, gradients))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('Gradient-aligned derivative check needs a finite nonzero gradient')
    return [(value.detach().float() / norm).to(value.dtype) for value in gradients]


def fixed_cotangent(output):
    """A private generator leaves the parent's global RNG streams untouched."""
    generator = torch.Generator(device=output.device).manual_seed(DIAGNOSTIC_SEED)
    signs = torch.randint(0, 2, output.shape, generator=generator, device=output.device)
    return ((signs.float() * 2 - 1) / math.sqrt(output.numel())).to(output.dtype)


def require_original_window(responses, original_window):
    current = dict(
        responses=[response.receipt() for response in responses],
        accumulation_responses=2,
        native_factor=0.5,
        next_rng_sha256=rng_digest(capture_rng()),
    )
    if current != original_window:
        raise ValueError('Repair attempt changed the original selected response/anchor/RNG window')
    return current


@contextmanager
def capture_first_attention():
    """Capture actual native inputs without changing the primal call or output."""
    import specforge.modeling.draft.dflash as draft

    original = draft.compile_friendly_flex_attention
    captured = {}

    def observe(query, key, value, **kwargs):
        output = original(query, key, value, **kwargs)
        if not captured:
            captured.update(
                qkv=tuple(
                    item.detach().clone().requires_grad_(True) for item in (query, key, value)
                ),
                kwargs=dict(kwargs),
                output=output.detach().clone(),
                original=original,
            )
        return output

    draft.compile_friendly_flex_attention = observe
    try:
        yield captured
    finally:
        draft.compile_friendly_flex_attention = original


def native_model_vjp(module, response, parameters, work, label):
    restore_rng(response.rng_after_sampling)
    work.start(label + '_forward')
    output = module(response)
    work.finish(label + '_forward')
    cotangent = fixed_cotangent(output)
    work.start(label + '_parameter_vjp')
    gradients = torch.autograd.grad(output, tuple(parameters.values()), grad_outputs=cotangent)
    work.finish(label + '_parameter_vjp')
    return output.detach(), cotangent, tuple(value.detach() for value in gradients)


def qualify_operator(captured, work, destination):
    from torch.nn.attention.flex_attention import AuxRequest
    from .attention_jvp import attention_tangent

    query, key, value = captured['qkv']
    options = captured['kwargs']
    work.start('native_operator_aux_forward')
    output, auxiliary = captured['original'](
        query, key, value, **options, return_aux=AuxRequest(lse=True)
    )
    work.finish('native_operator_aux_forward')
    parity = torch.equal(output, captured['output'])
    receipt = dict(
        native_primal_bit_equal=parity,
        query_shape=list(query.shape),
        key_shape=list(key.shape),
        value_shape=list(value.shape),
        dtype=str(query.dtype),
        full_native_block_mask=True,
        enable_gqa=options.get('enable_gqa', False),
        scale=options.get('scale'),
        kernel_options=options.get('kernel_options'),
        lse_dtype=str(auxiliary.lse.dtype),
        lse_units='natural logarithm from native AuxOutput',
    )
    write_json(destination, receipt)
    if not parity:
        raise ValueError('Requesting native auxiliary LSE changed the native attention primal')
    cotangent = fixed_cotangent(output)
    work.start('native_operator_qkv_vjp')
    gradients = torch.autograd.grad(output, (query, key, value), grad_outputs=cotangent)
    work.finish('native_operator_qkv_vjp')
    receipt['duality'] = {}
    for name, active in (('query', (0,)), ('key', (1,)), ('value', (2,)), ('joint', (0, 1, 2))):
        masked_gradients = [
            gradient if index in active else torch.zeros_like(gradient)
            for index, gradient in enumerate(gradients)
        ]
        direction = aligned_direction(masked_gradients)
        active_direction = [
            vector if index in active else None for index, vector in enumerate(direction)
        ]
        accounting = {}
        work.start('analytic_operator_tangent_' + name)
        tangent = attention_tangent(
            query,
            key,
            value,
            *active_direction,
            output=output.detach(),
            lse=auxiliary.lse.detach(),
            block_mask=options['block_mask'],
            scale=options['scale'],
            enable_gqa=options.get('enable_gqa', False),
            accounting=accounting,
        )
        work.finish('analytic_operator_tangent_' + name)
        receipt['duality'][name] = duality_receipt(tangent, cotangent, gradients, direction)
        receipt['duality'][name]['tangent_accounting'] = accounting
        write_json(destination, receipt)
        if not receipt['duality'][name]['passed']:
            raise ValueError(
                'Native attention ' + name + ' duality failed the predeclared tolerance'
            )
    return receipt


def qualify_model(module, response, parameters, baseline, cotangent, gradients, work, destination):
    from .attention_jvp import native_attention_jvp

    receipt = dict(native_primal_bit_equal=False, ordinary_backward_bit_equal=False)
    write_json(destination, receipt)
    with native_attention_jvp() as ordinary_accounting:
        output, same_cotangent, replay_gradients = native_model_vjp(
            module, response, parameters, work, 'scoped_ordinary_native'
        )
    receipt['ordinary_primal_bit_equal'] = torch.equal(output, baseline)
    receipt['ordinary_cotangent_bit_equal'] = torch.equal(cotangent, same_cotangent)
    receipt['ordinary_backward_bit_equal'] = all(
        torch.equal(left, right) for left, right in zip(gradients, replay_gradients)
    )
    receipt['ordinary_scope_accounting'] = ordinary_accounting
    write_json(destination, receipt)
    if not all(
        receipt[key]
        for key in (
            'ordinary_primal_bit_equal',
            'ordinary_cotangent_bit_equal',
            'ordinary_backward_bit_equal',
        )
    ):
        raise ValueError('Scoped adapter changed ordinary native forward or backward')
    del output, same_cotangent, replay_gradients
    direction = aligned_direction(gradients)
    restore_rng(response.rng_after_sampling)
    work.start('repaired_model_jvp_forward')
    with native_attention_jvp() as derivative_accounting:
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            dual = {
                name: torch.autograd.forward_ad.make_dual(parameter, vector)
                for (name, parameter), vector in zip(parameters.items(), direction)
            }
            result = torch.func.functional_call(module, dual, (response,))
            primal, tangent = torch.autograd.forward_ad.unpack_dual(result)
            if tangent is None or not bool(torch.isfinite(tangent).all()):
                raise ValueError('Repaired full-model JVP is missing or nonfinite')
            receipt['native_primal_bit_equal'] = torch.equal(primal, baseline)
            receipt['duality'] = duality_receipt(tangent, cotangent, gradients, direction)
    work.finish('repaired_model_jvp_forward')
    receipt['derivative_scope_accounting'] = derivative_accounting
    write_json(destination, receipt)
    if not receipt['native_primal_bit_equal'] or not receipt['duality']['passed']:
        raise ValueError('Repaired full-model primal or derivative failed its declared check')
    return receipt


def first_vjp_comparison(names, actual, expected):
    """Retain compact equality evidence; materialize differences one tensor at a time."""
    rows = []
    for name, left, right in zip(names, actual, expected):
        equal = torch.equal(left.detach(), right)
        row = dict(name=name, equal=equal, max_absolute_difference=0.0, l2_difference=0.0)
        if not equal:
            difference = left.detach().float() - right.float()
            finite = bool(torch.isfinite(difference).all())
            row.update(
                finite_difference=finite,
                max_absolute_difference=float(difference.abs().max()) if finite else None,
                l2_difference=float(difference.norm()) if finite else None,
            )
            del difference
        rows.append(row)
    return rows


def qualify_transposed_model(
    module, response, parameters, baseline, cotangent, gradients, work, destination, failed_repair
):
    """Transpose the native first VJP only in its incoming output cotangent."""
    from .attention_reverse_jvp import native_attention_reverse_jvp, native_silu_reverse_jvp

    previous = json.loads((Path(failed_repair) / 'MODEL.json').read_text())
    if previous['duality']['passed'] or not previous['native_primal_bit_equal']:
        raise ValueError('Require the retained failed whole-model repair with exact primal')
    direction = aligned_direction(gradients)
    reference_dot = vector_dot(gradients, direction)
    if reference_dot != previous['duality']['native_backward_dot']:
        raise ValueError('Transposed check changed the retained native reference direction')
    receipt = dict(
        route='d/du <native J^T u, fixed direction>',
        fixed_cotangent='same nonzero retained diagnostic cotangent',
        native_reference_contraction_exact=True,
        native_backward_reference=reference_dot,
        native_primal_bit_equal=None,
        ordinary_backward_bit_equal=None,
        parameter_hessian_requested=False,
    )
    write_json(destination, receipt)
    restore_rng(response.rng_after_sampling)
    with native_attention_reverse_jvp() as accounting, native_silu_reverse_jvp() as silu_accounting:
        receipt['derivative_scope_accounting'] = accounting
        receipt['silu_scope_accounting'] = silu_accounting
        receipt['expected_silu_calls'] = 5
        try:
            work.start('transposed_native_model_forward')
            output = module(response)
            work.finish('transposed_native_model_forward')
            receipt['native_primal_bit_equal'] = torch.equal(output, baseline)
            if not receipt['native_primal_bit_equal']:
                raise ValueError('Cotangent-transpose wrapper changed the native model primal')
            if silu_accounting['native_silu_forwards'] != 5:
                raise ValueError('Expected exactly five native SiLU calls in the derivative scope')
            incoming = cotangent.detach().clone().requires_grad_(True)
            work.start('transposed_native_parameter_vjp')
            native_gradients = torch.autograd.grad(
                output,
                tuple(parameters.values()),
                grad_outputs=incoming,
                create_graph=True,
                allow_unused=False,
            )
            work.finish('transposed_native_parameter_vjp')
            if silu_accounting['native_silu_first_vjps'] != 5:
                raise ValueError('Expected exactly five native SiLU first VJPs')
            receipt['first_vjp_parameters'] = first_vjp_comparison(
                parameters, native_gradients, gradients
            )
            receipt['ordinary_backward_bit_equal'] = all(
                row['equal'] for row in receipt['first_vjp_parameters']
            )
            write_json(destination, receipt)
            if not receipt['ordinary_backward_bit_equal']:
                raise ValueError('Cotangent-transpose wrapper changed the native first VJP')
            contraction = sum(
                (gradient.float() * vector.float()).sum()
                for gradient, vector in zip(native_gradients, direction)
            )
            work.start('transposed_native_grad_output_vjp')
            tangent = torch.autograd.grad(contraction, incoming, allow_unused=False)[0]
            work.finish('transposed_native_grad_output_vjp')
            if silu_accounting['silu_cotangent_transposes'] != 5:
                raise ValueError('Expected exactly five SiLU cotangent transposes')
            if not bool(torch.isfinite(tangent).all()):
                raise ValueError('Cotangent-transposed full-model tangent is nonfinite')
            receipt['duality'] = duality_receipt(tangent, cotangent, gradients, direction)
            if not receipt['duality']['passed']:
                raise ValueError('Cotangent-transposed derivative failed the unchanged tolerance')
        finally:
            write_json(destination, receipt)
    return receipt


def run_repair_probe(
    parent_plan,
    original_probe,
    output,
    execution_commit,
    *,
    localize=False,
    failed_repair=None,
    transposed=False,
):
    if localize and transposed:
        raise ValueError('Choose only one bounded diagnostic mode')
    if (localize or transposed) and failed_repair is None:
        raise ValueError('This diagnostic requires the retained failed repair path')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original_probe = Path(original_probe)
    receipt = dict(
        schema='frontier_native_derivative_repair_v1',
        status='preparing',
        execution_commit=execution_commit,
        parent_execution_commit=PARENT_COMMIT,
        original_probe=str(original_probe),
        native_adam_steps=0,
        full_coefficient_sensitivity_qualified=False,
        production_admitted=False,
        derivative_tolerance=dict(atol=DUALITY_ATOL, rtol=DUALITY_RTOL),
        diagnostic_seed=DIAGNOSTIC_SEED,
        derivative_model_reverse_graph=transposed,
        precision_policy=dict(
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            global_precision_settings_changed=False,
        ),
        model_responses_attempted=0,
        model_responses_unattempted=2,
        head_calls=0,
        localization_only=localize,
        transposed_only=transposed,
    )
    result_path = output / 'RESULT.json'
    work = WorkReceipt(output / 'WORK.json')
    write_json(result_path, receipt)
    trainer = None
    started = time.perf_counter()
    try:
        original_result = json.loads((original_probe / 'RESULT.json').read_text())
        if (
            original_result['status'] != 'blocked_native_derivative'
            or original_result['native_adam_steps']
        ):
            raise ValueError('Require the retained zero-update original native failure')
        original_window = json.loads((original_probe / 'WINDOW.json').read_text())
        plan, recovery, selected = select_parent(parent_plan)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['devices']['B']:
            raise ValueError('Repair qualification must see only sealed physical training GPU B')
        write_json(
            output / 'SELECTION.json',
            dict(
                parent_plan=str(parent_plan),
                parent_binding=plan['binding'],
                parent_recovery=str(recovery),
                parent_recovery_identity=PARENT_RECOVERY,
                selected_ordered_training_features=selected,
                original_probe=str(original_probe),
            ),
        )
        runtime = pin_retained_graph_runtime(output)
        runtime.update(
            scope='frontier derivative-only native attention repair',
            applied_before='parent restoration',
        )
        write_json(output / 'RUNTIME_COMPATIBILITY.json', runtime)
        work.start('parent_runtime_initialization')
        trainer = PilotTrainer(
            plan['configs']['PAIR'],
            output / 'trainer',
            arm='PAIR',
            binding=plan['binding'],
            original_config_text=Path(plan['original_config']).read_text(),
            maximum_events=1,
            resume_state=recovery,
            completed_event=1,
            qualification_only=True,
        )
        work.finish('parent_runtime_initialization')
        model = trainer.bundle.model
        parent_rng = capture_rng()
        initial_work, initial_costs = deepcopy(trainer.work), deepcopy(model.costs)
        initial_anchors = deepcopy(trainer._anchor_observations)
        modes = [(item, item.training) for item in model.modules()]
        torch.cuda.reset_peak_memory_stats()
        try:
            work.start('native_anchor_window_sampling')
            responses = freeze_two_responses(trainer, [row['path'] for row in selected])
            work.finish('native_anchor_window_sampling')
            write_json(output / 'WINDOW.json', require_original_window(responses, original_window))
            receipt['original_window_exact'] = True
            module = NativeHidden(model)
            parameters = trainable_parameters(module)
            if len(parameters) != 58 or any(
                value.dtype != torch.bfloat16 for value in parameters.values()
            ):
                raise ValueError('Restored native trainable parameter set differs')
            receipt['status'] = 'running'
            receipt['model_responses_attempted'] = 1
            receipt['model_responses_unattempted'] = 1
            write_json(result_path, receipt)
            if localize:
                from .localize_native import run_localization

                if failed_repair is None:
                    raise ValueError('Localization requires the retained failed repair path')
                receipt['localization'] = run_localization(
                    module,
                    responses[0],
                    parameters,
                    work,
                    output,
                    failed_repair,
                )
                receipt['status'] = 'native_derivative_localization_completed'
            elif transposed:
                baseline, cotangent, gradients = native_model_vjp(
                    module, responses[0], parameters, work, 'ordinary_native'
                )
                receipt['model'] = qualify_transposed_model(
                    module,
                    responses[0],
                    parameters,
                    baseline,
                    cotangent,
                    gradients,
                    work,
                    output / 'MODEL.json',
                    failed_repair,
                )
                receipt['status'] = 'transposed_native_derivative_qualified'
            else:
                with capture_first_attention() as captured:
                    baseline, cotangent, gradients = native_model_vjp(
                        module, responses[0], parameters, work, 'ordinary_native'
                    )
                receipt['operator'] = qualify_operator(captured, work, output / 'OPERATOR.json')
                del captured
                gc.collect()
                torch.cuda.empty_cache()
                receipt['model'] = qualify_model(
                    module,
                    responses[0],
                    parameters,
                    baseline,
                    cotangent,
                    gradients,
                    work,
                    output / 'MODEL.json',
                )
                receipt['status'] = 'repaired_native_derivative_qualified'
        finally:
            receipt['physical_work_delta'] = {
                key: value - initial_work.get(key, 0) for key, value in trainer.work.items()
            }
            receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            receipt['work'] = deepcopy(work.value)
            for item, mode in modes:
                item.training = mode
            restore_rng(parent_rng)
            trainer.work, model.costs = initial_work, initial_costs
            trainer._anchor_observations[:] = initial_anchors
            state = torch.load(recovery, map_location='cpu', weights_only=False, mmap=True)
            receipt['restoration'] = verify_live(trainer, state, output / 'FINAL_RESTORATION.json')
            del state
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['wall_s'] = time.perf_counter() - started
        receipt['work'] = deepcopy(work.value)
        write_json(result_path, receipt)
        if trainer is not None:
            trainer.close()
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', required=True)
    parser.add_argument('--original-probe', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execution-commit', required=True)
    parser.add_argument('--localize', action='store_true')
    parser.add_argument('--transposed', action='store_true')
    parser.add_argument('--failed-repair')
    args = parser.parse_args()
    result = run_repair_probe(
        args.parent_plan,
        args.original_probe,
        args.output,
        args.execution_commit,
        localize=args.localize,
        failed_repair=args.failed_repair,
        transposed=args.transposed,
    )
    print(json.dumps({key: result[key] for key in ('status', 'native_adam_steps', 'wall_s')}))
