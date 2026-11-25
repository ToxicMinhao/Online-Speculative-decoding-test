"""Native BF16 derivative prerequisite for one frozen two-response window.

This first vertical slice changes no optimizer or attention implementation. It
checks the parameter-to-hidden JVP needed by matrix-free token sensitivities.
A failed primitive blocks the candidate; it is not replaced by a local score.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
from torch import nn

from online_feedback.ranking import fixed_anchors
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot.training import PilotTrainer
from experiments.full_k_gsm8k.training import verify_live

PARENT_COMMIT = '416cb7931028621c09b8fa983e53fc8a7dcebaab'
PARENT_RECOVERY = dict(
    bytes=6296473051,
    sha256='728770a7abf0ba3ef96805ef760997976ad7bedbbee1d8debe165e785a4d98a2',
)


def rng_digest(state):
    numpy_state = state['numpy']
    values = dict(
        python=state['python'],
        numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        torch=state['torch'].tolist(),
        cuda=[value.cpu().tolist() for value in state['cuda']],
    )
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


@dataclass
class FrozenResponse:
    """Inputs and anchors fixed once; no forward graph or per-token gradient bank."""

    record_id: str
    tensors: dict
    anchors: torch.Tensor
    keep: torch.Tensor
    rng_before_sampling: dict
    rng_after_sampling: dict

    def receipt(self):
        return dict(
            record_id=self.record_id,
            anchors=self.anchors.detach().cpu().tolist(),
            keep=self.keep.detach().cpu().tolist(),
            sampling_rng_before_sha256=rng_digest(self.rng_before_sampling),
            sampling_rng_after_sha256=rng_digest(self.rng_after_sampling),
            input_shape=list(self.tensors['input_ids'].shape),
            hidden_dtype=str(self.tensors['hidden_states'].dtype),
        )


def freeze_two_responses(trainer, feature_paths):
    """Sample exactly two native anchor sets from the currently restored RNG."""
    from specforge.training.strategies.base import _cpu_max_valid_anchors

    if len(feature_paths) != 2 or trainer.core.accumulation_remainder:
        raise ValueError('Derivative qualification requires one complete two-response window')
    model = trainer.bundle.model
    active_dropout = any(
        isinstance(module, nn.modules.dropout._DropoutNd) and module.training and module.p > 0
        for module in model.modules()
    )
    if active_dropout or getattr(model.draft_model.config, 'attention_dropout', 0) != 0:
        raise ValueError('This prerequisite slice requires the qualified dropout-free native draft')
    device = next(model.draft_model.parameters()).device
    frozen = []
    for path in feature_paths:
        raw = torch.load(path, map_location='cpu', weights_only=True)
        batch = trainer._batch(raw)
        tensors = {
            name: batch.tensors[name].to(device)
            for name in ('input_ids', 'hidden_states', 'loss_mask', 'target_last_hidden_states')
        }
        before = capture_rng()
        anchors, keep = model._sample_anchor_positions(
            tensors['input_ids'].shape[1],
            tensors['loss_mask'],
            device,
            max_valid_anchors=_cpu_max_valid_anchors(batch.tensors['loss_mask']),
        )
        after = capture_rng()
        frozen.append(FrozenResponse(str(raw['record_id']), tensors, anchors, keep, before, after))
    return frozen


class NativeHidden(nn.Module):
    """Expose the unchanged author draft and masks to functional parameters."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, response):
        with fixed_anchors(self.model, response.anchors, response.keep):
            anchors, keep, hidden = self.model._forward_draft_blocks(
                response.tensors['input_ids'],
                response.tensors['hidden_states'],
                response.tensors['loss_mask'],
            )
        if not torch.equal(anchors, response.anchors) or not torch.equal(keep, response.keep):
            raise ValueError('Native derivative forward changed frozen anchors')
        return hidden


def trainable_parameters(module):
    named = {
        name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    if not named:
        raise ValueError('No native trainable parameters')
    return named


def unit_direction(parameters):
    """One deterministic parameter direction, not one vector per token."""
    scale = 1 / math.sqrt(sum(parameter.numel() for parameter in parameters.values()))
    return {name: torch.full_like(parameter, scale) for name, parameter in parameters.items()}


def first_order_control(module, response, parameters, direction, work):
    """Establish ordinary native BF16 forward/backward before testing higher AD."""
    work.start('control_native_forward')
    output = module(response)
    work.finish('control_native_forward')
    work.start('control_native_parameter_vjp')
    gradients = torch.autograd.grad(
        output,
        tuple(parameters.values()),
        grad_outputs=torch.ones_like(output),
        create_graph=False,
        allow_unused=False,
    )
    work.finish('control_native_parameter_vjp')
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise ValueError('Ordinary native first-order backward is nonfinite')
    return dict(
        output_shape=list(output.shape),
        output_dtype=str(output.dtype),
        gradient_tensors=len(gradients),
        gradient_l2=float(
            torch.stack([gradient.detach().float().norm() for gradient in gradients]).norm()
        ),
    )


def forward_ad_jvp(module, response, parameters, direction, work):
    """Standard forward AD through functional BF16 parameters and native Flex."""
    work.start('native_forward_ad_forward')
    with torch.autograd.forward_ad.dual_level():
        dual_parameters = {
            name: torch.autograd.forward_ad.make_dual(parameter, direction[name])
            for name, parameter in parameters.items()
        }
        output = torch.func.functional_call(module, dual_parameters, (response,))
        primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
        work.finish('native_forward_ad_forward')
        if tangent is None:
            raise RuntimeError('Native forward AD returned no tangent')
        if not bool(torch.isfinite(primal).all()) or not bool(torch.isfinite(tangent).all()):
            raise ValueError('Nonfinite native forward-AD output or tangent')
        return dict(
            output_shape=list(primal.shape),
            output_dtype=str(primal.dtype),
            tangent_dtype=str(tangent.dtype),
            tangent_norm=float(tangent.detach().float().norm()),
        )


def transposed_reverse_jvp(module, response, parameters, direction, work):
    """Jv = d/du <v, J^T u>; no parameter Hessian or token-gradient bank.

    The second reverse differentiates only the dummy output cotangent u. It
    nevertheless needs the native backward's derivative with respect to its
    grad-output, precisely the missing operation suspected in FlexAttention.
    """
    work.start('native_reverse_forward')
    output = module(response)
    work.finish('native_reverse_forward')
    cotangent = torch.zeros_like(output, requires_grad=True)
    work.start('native_parameter_vjp')
    gradients = torch.autograd.grad(
        output,
        tuple(parameters.values()),
        grad_outputs=cotangent,
        create_graph=True,
        allow_unused=False,
    )
    work.finish('native_parameter_vjp')
    contraction = sum(
        (gradient.float() * direction[name].float()).sum()
        for (name, _), gradient in zip(parameters.items(), gradients)
    )
    work.start('native_grad_output_vjp')
    tangent = torch.autograd.grad(contraction, cotangent, allow_unused=False)[0]
    work.finish('native_grad_output_vjp')
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(tangent).all()):
        raise ValueError('Nonfinite reverse-over-reverse native tangent')
    return dict(
        output_shape=list(output.shape),
        output_dtype=str(output.dtype),
        tangent_dtype=str(tangent.dtype),
        tangent_norm=float(tangent.detach().float().norm()),
    )


class WorkReceipt:
    """Write before each primitive so interruption cannot look like no work."""

    def __init__(self, path):
        self.path = Path(path)
        self.value = dict(attempted={}, completed={}, pending=[], native_adam_steps=0)
        self.save()

    def save(self):
        write_json(self.path, self.value)

    def start(self, operation):
        counts = self.value['attempted']
        counts[operation] = counts.get(operation, 0) + 1
        self.value['pending'].append(operation)
        self.save()

    def finish(self, operation):
        self.value['pending'].remove(operation)
        counts = self.value['completed']
        counts[operation] = counts.get(operation, 0) + 1
        self.save()


def support_attempt(method, module, response, parameters, direction, work, destination):
    result = dict(status='running', method=method.__name__, started_epoch=time.time())
    write_json(destination, result)
    started = time.perf_counter()
    try:
        restore_rng(response.rng_after_sampling)
        result['result'] = method(module, response, parameters, direction, work)
        result['status'] = 'passed'
    except Exception as error:
        result.update(
            status='unsupported_or_failed', error=repr(error), traceback=traceback.format_exc()
        )
    finally:
        result['wall_s'] = time.perf_counter() - started
        result['work'] = deepcopy(work.value)
        write_json(destination, result)
    return result


def select_parent(parent_plan):
    """Bind the original contract and the first two ordered retained PAIR packets."""
    plan = json.loads(Path(parent_plan).read_text())
    root = Path(plan['root'])
    if plan['execution_commit'] != PARENT_COMMIT or plan['maximum_events'] != 1:
        raise ValueError('Expected the completed original PR59 parent contract')
    status = json.loads((root / 'arms/PAIR/STATUS.json').read_text())
    if status['status'] != 'completed' or status['recovery_identity'] != PARENT_RECOVERY:
        raise ValueError('Expected the authenticated PR59 PAIR1 full recovery')
    recovery = Path(status['recovery'])
    if identity(recovery) != PARENT_RECOVERY:
        raise ValueError('The retained parent recovery identity changed')
    rows = json.loads((root / 'arms/PAIR/FEATURE_IDENTITIES.json').read_text())
    if len(rows) != 128:
        raise ValueError('Parent event feature pool is incomplete')
    selected = rows[:2]
    for row in selected:
        if identity(row['path']) != row['identity']:
            raise ValueError('Selected parent training feature changed')
    return plan, recovery, selected


def run_support_probe(parent_plan, output, execution_commit):
    """Restore the real warm parent; attempt necessary derivatives; never step."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        schema='frontier_native_derivative_support_v1',
        status='preparing',
        execution_commit=execution_commit,
        parent_execution_commit=PARENT_COMMIT,
        native_adam_steps=0,
        full_coefficient_sensitivity_qualified=False,
        automatic_retry=False,
        discarded_optimizer_startup_executed=False,
        startup_scope='First-order native control compiles the same path; all control/derivative work is separate from committed training.',
        primitive_scope='parameter-to-native-hidden JVP prerequisite; no alternative attention or loss',
        methods=[],
    )
    result_path = output / 'RESULT.json'
    write_json(result_path, receipt)
    trainer = None
    started = time.perf_counter()
    try:
        plan, recovery, selected = select_parent(parent_plan)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['devices']['B']:
            raise ValueError('Support qualification must see only sealed physical training GPU B')
        write_json(
            output / 'SELECTION.json',
            dict(
                parent_plan=str(parent_plan),
                parent_binding=plan['binding'],
                parent_recovery=str(recovery),
                parent_recovery_identity=PARENT_RECOVERY,
                selected_ordered_training_features=selected,
                selection='first two entries of retained PAIR FEATURE_IDENTITIES; one fixed window',
            ),
        )
        runtime = pin_retained_graph_runtime(output)
        runtime.update(
            scope='frontier native derivative support process', applied_before='parent restoration'
        )
        write_json(output / 'RUNTIME_COMPATIBILITY.json', runtime)
        flex_module = importlib.import_module('torch._higher_order_ops.flex_attention')
        receipt['runtime'] = dict(
            torch_version=str(torch.__version__),
            flex_source=str(flex_module.__file__),
            flex_source_identity=identity(flex_module.__file__),
        )
        work = WorkReceipt(output / 'WORK.json')
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
        modes = [(module, module.training) for module in model.modules()]
        torch.cuda.reset_peak_memory_stats()
        receipt['restored_parent'] = dict(
            global_step=trainer.global_step,
            presentations=trainer.presentations,
            completed_events=len(trainer.events),
            adam_state_entries=len(trainer.backend.optimizer.optimizer.state),
        )
        if receipt['restored_parent'] != dict(
            global_step=128, presentations=256, completed_events=1, adam_state_entries=58
        ):
            raise ValueError('Restored parent counters or warm Adam state differ')
        receipt['optimizer_recipe'] = deepcopy(trainer.backend.optimizer.recipe)
        receipt['status'] = 'running'
        write_json(result_path, receipt)
        try:
            work.start('native_anchor_window_sampling')
            responses = freeze_two_responses(trainer, [row['path'] for row in selected])
            work.finish('native_anchor_window_sampling')
            write_json(
                output / 'WINDOW.json',
                dict(
                    responses=[row.receipt() for row in responses],
                    accumulation_responses=2,
                    native_factor=0.5,
                    next_rng_sha256=rng_digest(capture_rng()),
                ),
            )
            module = NativeHidden(model)
            parameters = trainable_parameters(module)
            if len(parameters) != 58 or any(
                value.dtype != torch.bfloat16 for value in parameters.values()
            ):
                raise ValueError('Restored native trainable parameter set differs')
            direction = unit_direction(parameters)
            control = support_attempt(
                first_order_control,
                module,
                responses[0],
                parameters,
                direction,
                work,
                output / 'FIRST_ORDER_CONTROL.json',
            )
            receipt['first_order_control'] = control
            if control['status'] != 'passed':
                receipt['status'] = 'blocked_native_first_order_runtime'
                receipt['native_jvp_supported'] = False
                return receipt
            for method in (forward_ad_jvp, transposed_reverse_jvp):
                attempts = []
                for index, response in enumerate(responses):
                    result = support_attempt(
                        method,
                        module,
                        response,
                        parameters,
                        direction,
                        work,
                        output / f'{method.__name__}_response_{index}.json',
                    )
                    attempts.append(result)
                    if result['status'] != 'passed' and any(
                        message in result.get('error', '').lower()
                        for message in ('out of memory', 'device-side assert', 'illegal memory')
                    ):
                        raise RuntimeError(
                            'Fatal native runtime failure; stop additional derivative attempts'
                        )
                    gc.collect()
                    torch.cuda.empty_cache()
                    if result['status'] != 'passed':
                        break
                receipt['methods'].append(
                    dict(
                        method=method.__name__,
                        responses=attempts,
                        unattempted_responses=2 - len(attempts),
                        passed=len(attempts) == 2
                        and all(item['status'] == 'passed' for item in attempts),
                    )
                )
                write_json(result_path, receipt)
            receipt['native_jvp_supported'] = any(row['passed'] for row in receipt['methods'])
            receipt['status'] = (
                'primitive_supported_pending_full_composition'
                if receipt['native_jvp_supported']
                else 'blocked_native_derivative'
            )
        finally:
            receipt['physical_work_delta'] = {
                key: value - initial_work.get(key, 0) for key, value in trainer.work.items()
            }
            receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            receipt['work'] = deepcopy(work.value)
            for module, mode in modes:
                module.training = mode
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
        write_json(result_path, receipt)
        if trainer is not None:
            trainer.close()
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execution-commit', required=True)
    args = parser.parse_args()
    result = run_support_probe(args.parent_plan, args.output, args.execution_commit)
    print(json.dumps({key: result[key] for key in ('status', 'native_adam_steps', 'wall_s')}))
