"""Native replay and four disposable optimizer windows for the admitted fork.

The normal training helper adds only frozen anchor/coefficient lookup. Native
TrainerCore still owns response averaging, backward, accumulation and clipping.
All expensive identities and parent restoration belong to qualification only.
"""

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from experiments.full_k_gsm8k.training import verify_live
from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
from experiments.mathqa_fixed_lr_transfer.training import FORBIDDEN_CONTROLLERS
from experiments.ok_coding.storage import identity
from experiments.persistent_prefix_withholding.qualification import isolated_backend_for_load
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot.training import PilotTrainer, discarded_startup
from online_feedback.kplus import KPlusRedistribution
from online_feedback.ranking import fixed_anchors
from online_feedback.repair_conditional_kl import validate_repair_conditional_kl
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import write_json
from specforge.training.strategies.base import StepContext
from .allocation_qualify import json_diagnostics, vector_identity
from .allocation_replay import CachedPrefixAllocation, replay_allocation
from .allocation_scoring import NativeAllocationScorer
from .math import require
from .native import (
    PARENT_RECOVERY,
    WorkReceipt,
    freeze_two_responses,
    rng_digest,
    select_parent,
)
from .repair_native import require_original_window


def _read_feature(feature):
    path = feature['path'] if isinstance(feature, dict) else feature
    return torch.load(path, map_location='cpu', weights_only=True)


def _context(trainer):
    return StepContext(
        global_step=trainer.global_step,
        total_steps=trainer.horizon,
        collect_detailed_metrics=True,
    )


def _finish_response(trainer, raw, result, metrics, learning_rate, elapsed, pass_index):
    require(
        all(math.isfinite(float(value)) for value in metrics.values()),
        'Nonfinite native training metric',
    )
    trainer.presentations += 1
    trainer.global_step += int(result.optimizer_stepped)
    return dict(
        record_id=str(raw['record_id']),
        **{'pass': pass_index},
        optimizer_step_after=trainer.global_step,
        optimizer_stepped=bool(result.optimizer_stepped),
        learning_rate=learning_rate if result.optimizer_stepped else None,
        metrics=metrics,
        training_s=elapsed,
        sampled_anchors=deepcopy(trainer._anchor_observations),
        response_observation=deepcopy(trainer.response_records[-1]),
        metrics_scope='accumulation_window' if result.optimizer_stepped else 'presentation',
    )


def train_window(
    trainer,
    features,
    *,
    enabled=False,
    responses=None,
    snapshots=None,
    coefficients=None,
    work=None,
    pass_index=0,
):
    """Commit exactly one native two-response update, with optional cached weights.

    With enabled=False this never constructs a scorer, samples ahead, replaces
    an extension or changes RNG. With enabled=True the caller has sampled both
    responses exactly once. Each final forward uses its saved after-sampling
    state and cached anchors, including the original anchor-observer schema.
    """
    require(
        len(features) == 2 and not trainer.core.accumulation_remainder,
        'Require a complete two-response native window',
    )
    if enabled:
        require(
            responses is not None
            and snapshots is not None
            and coefficients is not None
            and len(responses) == len(snapshots) == len(coefficients) == 2,
            'Cached replay requires both complete response assignments',
        )
        expected_rng = rng_digest(responses[-1].rng_after_sampling)
        require(
            rng_digest(capture_rng()) == expected_rng,
            'Cached replay must start at the frozen post-window RNG',
        )
    else:
        require(
            responses is None and snapshots is None and coefficients is None,
            'Disabled training must not accept scoring or cached metadata',
        )
    start_step, start_presentations = trainer.global_step, trainer.presentations
    rows = []
    for index, feature in enumerate(features):
        raw = _read_feature(feature)
        trainer._anchor_observations.clear()
        scope = nullcontext()
        anchor_scope = nullcontext()
        if enabled:
            response = responses[index]
            require(
                str(raw['record_id']) == response.record_id,
                'Cached replay response identity changed',
            )
            restore_rng(response.rng_after_sampling)
            trainer._anchor_observations.append(
                dict(
                    positions=response.anchors.detach().cpu().tolist(),
                    keep_mask=response.keep.detach().cpu().tolist(),
                )
            )
            anchor_scope = fixed_anchors(trainer.bundle.model, response.anchors, response.keep)
            scope = replay_allocation(
                trainer.bundle.model, snapshots[index], coefficients[index]
            )
        learning_rate = trainer.backend.optimizer.get_learning_rate()
        started = time.perf_counter()
        if work is not None:
            work.start('native_committed_response_forward_backward')
        with anchor_scope, scope:
            result = trainer.core.train_step(trainer._batch(raw), _context(trainer))
            # Synchronize inside the measured native response boundary.
            metrics = result.materialize_metrics()
        if work is not None:
            work.finish('native_committed_response_forward_backward')
        rows.append(
            _finish_response(
                trainer,
                raw,
                result,
                metrics,
                learning_rate,
                time.perf_counter() - started,
                pass_index,
            )
        )
        require(bool(result.optimizer_stepped) == (index == 1), 'Native Adam boundary changed')
        if enabled:
            require(
                rng_digest(capture_rng()) == rng_digest(response.rng_after_sampling),
                'Cached dropout-free training unexpectedly consumed RNG',
            )
    require(
        trainer.global_step == start_step + 1
        and trainer.presentations == start_presentations + 2
        and not trainer.core.accumulation_remainder,
        'Incomplete native window',
    )
    return dict(
        rows=rows,
        window=dict(
            start_step=start_step,
            end_step=trainer.global_step,
            presentations=2,
            native_adam_steps=1,
            enabled=enabled,
            cached_anchors=enabled,
            sampling_replays=0,
            end_rng_sha256=rng_digest(capture_rng()),
            numerical_contract='native_token_contraction_v2' if enabled else None,
        ),
    )


def ordinary_control_window(trainer, features, work=None):
    """Independent reference: the original native inner presentation loop."""
    rows = []
    for feature in features:
        raw = _read_feature(feature)
        trainer._anchor_observations.clear()
        learning_rate = trainer.backend.optimizer.get_learning_rate()
        started = time.perf_counter()
        if work is not None:
            work.start('native_committed_response_forward_backward')
        result = trainer.core.train_step(trainer._batch(raw), _context(trainer))
        metrics = result.materialize_metrics()
        if work is not None:
            work.finish('native_committed_response_forward_backward')
        rows.append(
            _finish_response(
                trainer, raw, result, metrics, learning_rate, time.perf_counter() - started, 0
            )
        )
    return dict(rows=rows)


def validate_cached_teaching(trainer):
    """Validate the explicitly named new replay contract and all native fields."""
    model = trainer.bundle.model
    extension = model.prefix_allocation
    validate_repair_conditional_kl(model)
    require(
        trainer.arm == 'PAIR'
        and model.repair_conditional_kl
        and isinstance(model.first_error_redistribution, KPlusRedistribution)
        and isinstance(extension, CachedPrefixAllocation)
        and extension.contract
        == dict(
            name='frontier_preservation_v1_cached_native_replay',
            numerical_contract='native_token_contraction_v2',
        )
        and not any(
            getattr(model, name, None) is not None
            for name in set(FORBIDDEN_CONTROLLERS) - {'prefix_allocation'}
        ),
        'Foreign objective or invalid explicit cached replay contract',
    )


@contextmanager
def qualification_cached_contract(trainer):
    """Accept the explicit new cache only in a disposable qualification scope.

    Never relabel this cache as historical C or weaken its existing validator.
    The original check remains active whenever the cache is not installed.
    """
    require(trainer.qualification_only, 'Cached validation override is qualification-only')
    original = trainer.check_installed_arm

    def checked():
        if isinstance(trainer.bundle.model.prefix_allocation, CachedPrefixAllocation):
            return validate_cached_teaching(trainer)
        return original()

    trainer.check_installed_arm = checked
    try:
        yield
    finally:
        trainer.check_installed_arm = original


class DisposableStepLedger:
    """Reserve before native Adam; an interrupted attempt still spends allowance."""

    def __init__(self, path, window_identity, output):
        self.path = Path(path)
        self.window_identity = window_identity
        self.output = str(Path(output).resolve())
        if self.path.exists():
            self.value = json.loads(self.path.read_text())
            require(
                self.value['schema'] == 'frontier_disposable_adam_ledger_v2',
                'Disposable update ledger schema changed',
            )
        else:
            self.value = dict(
                schema='frontier_disposable_adam_ledger_v2',
                maximum_attempted_steps=8,
                maximum_fixed_windows=2,
                fixed_windows=[],
                steps=[],
            )
        require(
            self.value['maximum_attempted_steps'] == 8
            and self.value['maximum_fixed_windows'] == 2,
            'Disposable update limits changed',
        )
        require(
            len(self.value['steps']) <= 4,
            'Four-case integration requires four unused disposable steps',
        )
        if window_identity not in self.value['fixed_windows']:
            require(len(self.value['fixed_windows']) < 2, 'Two-window allowance exhausted')
            self.value['fixed_windows'].append(window_identity)
        write_json(self.path, self.value)

    def reserve(self, case):
        require(len(self.value['steps']) < 8, 'Eight-step disposable allowance exhausted')
        require(
            not any(
                item['output'] == self.output and item['case'] == case
                for item in self.value['steps']
            ),
            'Disposable case already attempted',
        )
        row = dict(
            ordinal=len(self.value['steps']) + 1,
            case=case,
            output=self.output,
            window_identity=self.window_identity,
            status='attempted',
        )
        self.value['steps'].append(row)
        write_json(self.path, self.value)
        return row

    def complete(self, row):
        row['status'] = 'completed'
        write_json(self.path, self.value)


def _restore_parent(trainer, state, runtime, ancillary):
    trainer.backend.optimizer.zero_grad()
    trainer.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
    device = next(trainer.bundle.draft_model.parameters()).device
    trainer.backend.load_state_dict(isolated_backend_for_load(state['backend'], device))
    trainer.global_step, trainer.presentations = state['global_step'], state['presentations']
    trainer.events = deepcopy(state['events'])
    trainer.core._micro = trainer.presentations
    trainer.core._ratio_totals.clear()
    trainer.core._sum_totals.clear()
    restore_runtime(trainer.bundle.model, runtime)
    restore_rng(state['rng'])
    trainer.response_records[:] = ancillary['responses']
    trainer.window_records[:] = ancillary['windows']
    trainer._anchor_observations[:] = ancillary['anchors']
    trainer.work = deepcopy(ancillary['work'])
    trainer.bundle.model.costs = deepcopy(ancillary['costs'])
    trainer.bundle.model.prefix_allocation.last_summary = deepcopy(ancillary['prefix_summary'])
    trainer._active_event = ancillary['active_event']


def _endpoint_identity(trainer):
    named = list(trainer.bundle.draft_model.named_parameters())
    optimizer = trainer.backend.optimizer.optimizer
    return dict(
        parameters=vector_identity([name for name, _ in named], [p for _, p in named]),
        exp_avg=vector_identity(
            [name for name, _ in named], [optimizer.state[p]['exp_avg'] for _, p in named]
        ),
        exp_avg_sq=vector_identity(
            [name for name, _ in named], [optimizer.state[p]['exp_avg_sq'] for _, p in named]
        ),
        steps=[float(optimizer.state[p]['step']) for _, p in named],
        options={
            key: value for key, value in optimizer.param_groups[0].items() if key != 'params'
        },
        global_step=trainer.global_step,
        presentations=trainer.presentations,
        micro=trainer.core._micro,
        rng_sha256=rng_digest(capture_rng()),
        gradients_cleared=all(p.grad is None for _, p in named),
    )


@torch.no_grad()
def shadow_native_displacement(trainer, parent_state, gradients):
    """Compare BETA's native step with its identical-input continuous shadow.

    Recompute only the displacement diagnostic after the native step. A single
    parameter's promoted state is live at a time; no full preview is retained.
    This measures mixed-precision update differences and supplies no new gate.
    """
    named = list(trainer.bundle.draft_model.named_parameters())
    saved = parent_state['backend']['optimizer']['optimizer_state_dict']
    options = saved['param_groups'][0]
    identifiers = options['params']
    require(
        len(named) == len(gradients) == len(identifiers),
        'Shadow diagnostic parameter count changed',
    )
    norm = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients]
        )
    )
    clip = (trainer.backend.optimizer.max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
    beta1, beta2 = options['betas']
    shadow_squared, native_squared, difference_squared = [], [], []
    maximum_difference, cast_differences = 0.0, 0
    for (name, parameter), gradient, identifier in zip(named, gradients, identifiers):
        old = saved['state'][identifier]
        theta = parent_state['draft_state_dict'][name].to(parameter.device, torch.float32)
        mean = old['exp_avg'].to(parameter.device, torch.float32)
        square = old['exp_avg_sq'].to(parameter.device, torch.float32)
        clipped = gradient.detach().float() * clip
        next_mean = torch.lerp(mean, clipped, 1 - beta1)
        next_square = square * beta2 + clipped.square() * (1 - beta2)
        step = float(old['step']) + 1
        denominator = next_square.sqrt() / math.sqrt(1 - beta2**step) + options['eps']
        predicted = (
            theta * (1 - options['lr'] * options['weight_decay'])
            - (options['lr'] / (1 - beta1**step)) * next_mean / denominator
        )
        require(
            torch.isfinite(predicted).all() and torch.isfinite(parameter).all(),
            'Nonfinite same-coefficient shadow or native endpoint',
        )
        shadow_delta = predicted - theta
        native_delta = parameter.float() - theta
        difference = parameter.float() - predicted
        shadow_squared.append(float(shadow_delta.square().sum(dtype=torch.float64)))
        native_squared.append(float(native_delta.square().sum(dtype=torch.float64)))
        difference_squared.append(float(difference.square().sum(dtype=torch.float64)))
        maximum_difference = max(maximum_difference, float(difference.abs().max()))
        cast_differences += int((predicted.to(parameter.dtype) != parameter).sum())
        del theta, mean, square, clipped, next_mean, next_square, denominator, predicted
        del shadow_delta, native_delta, difference
    shadow_norm = math.sqrt(math.fsum(shadow_squared))
    difference_norm = math.sqrt(math.fsum(difference_squared))
    return dict(
        same_restored_state=True,
        same_preclip_gradient=True,
        coefficients='beta prefix plus unchanged native K+PAIR repair',
        continuous_gradient_norm=float(norm),
        continuous_clip=float(clip),
        continuous_shadow_displacement_l2=shadow_norm,
        native_bf16_displacement_l2=math.sqrt(math.fsum(native_squared)),
        native_minus_shadow_l2=difference_norm,
        native_minus_shadow_max_abs=maximum_difference,
        relative_to_shadow_displacement=difference_norm / shadow_norm if shadow_norm else None,
        native_vs_cast_shadow_different_elements=cast_differences,
        full_preview_vectors_retained=0,
        promoted_parameter_states_live_at_once=1,
        reduction='FP32 differences and squares, FP64 sums',
        admission_gate=False,
        native_adam_steps=0,
    )


def _science_rows(rows):
    return [
        {
            key: value
            for key, value in row.items()
            if key not in ('training_s', 'response_observation')
        }
        for row in rows
    ]


def qualify_integration(
    trainer,
    parent_state,
    features,
    responses,
    snapshots,
    coefficients,
    reference_gradient,
    work,
    output,
    *,
    verify=verify_live,
    step_ledger=None,
):
    """Four independently restored one-step cases, with no tensor checkpoints."""
    require(trainer.qualification_only, 'Disposable updates require qualification-only mode')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_state(trainer.bundle.model)
    ancillary = dict(
        responses=deepcopy(trainer.response_records),
        windows=deepcopy(trainer.window_records),
        anchors=deepcopy(trainer._anchor_observations),
        work=deepcopy(trainer.work),
        costs=deepcopy(trainer.bundle.model.costs),
        prefix_summary=deepcopy(trainer.bundle.model.prefix_allocation.last_summary),
        active_event=getattr(trainer, '_active_event', None),
    )
    named = list(trainer.bundle.draft_model.named_parameters())
    expected_names = tuple(name for name, _ in named)
    cases = {}
    beta = [snapshot['roles']['beta'] for snapshot in snapshots]
    for name in ('CONTROL', 'DISABLED', 'BETA', 'CANDIDATE'):
        _restore_parent(trainer, parent_state, runtime, ancillary)
        verify(trainer, parent_state, output / f'{name}_INITIAL_STATE.json')
        trainer._active_event = 'integration_' + name.lower()
        (trainer.events_root / trainer._active_event).mkdir(exist_ok=False)
        original_step = trainer.backend.optimizer.step
        case = dict(status='running', native_adam_attempts=0, native_adam_steps=0)
        write_json(output / f'{name}.json', case)

        def observed_step():
            device = named[0][1].device
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            observer_started = time.perf_counter()
            case['preclip_observations'] = case.get('preclip_observations', 0) + 1
            require(
                case['preclip_observations'] == 1, 'More than one disposable step in a case'
            )
            gradients = tuple(parameter.grad for _, parameter in named)
            require(
                all(g is not None and torch.isfinite(g).all() for g in gradients),
                'Missing or nonfinite normalized preclip gradient',
            )
            case['preclip_gradient'] = vector_identity(expected_names, gradients)
            if name == 'BETA':
                parity = [
                    torch.equal(actual, expected)
                    for actual, expected in zip(gradients, reference_gradient)
                ]
                case['reference_gradient_exact'] = all(parity) and len(parity) == len(
                    reference_gradient
                )
                require(
                    case['reference_gradient_exact'],
                    'Native beta preclip gradient differs from scoring g0',
                )
            reservation = step_ledger.reserve(name) if step_ledger is not None else None
            case['native_adam_attempts'] += 1
            work.start('disposable_native_adam_step')
            write_json(output / f'{name}.json', case)
            native_step_started = time.perf_counter()
            norm = original_step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            native_step_s = time.perf_counter() - native_step_started
            if step_ledger is not None:
                step_ledger.complete(reservation)
            work.finish('disposable_native_adam_step')
            work.value['native_adam_steps'] += 1
            work.save()
            case['native_adam_steps'] += 1
            case['preclip_norm'] = float(norm)
            case['qualification_step_observer_s'] = (
                time.perf_counter() - observer_started - native_step_s
            )
            return norm

        trainer.backend.optimizer.step = observed_step
        try:
            if name == 'CONTROL':
                result = ordinary_control_window(trainer, features, work)
            elif name == 'DISABLED':
                result = train_window(trainer, features, enabled=False, work=work)
            else:
                restore_rng(responses[-1].rng_after_sampling)
                with qualification_cached_contract(trainer):
                    result = train_window(
                        trainer,
                        features,
                        enabled=True,
                        responses=responses,
                        snapshots=snapshots,
                        coefficients=beta if name == 'BETA' else coefficients,
                        work=work,
                    )
            case['native_training_s'] = (
                sum(row['training_s'] for row in result['rows'])
                - case['qualification_step_observer_s']
            )
            require(case['native_training_s'] >= 0, 'Native timing accounting became negative')
            if name == 'BETA':
                diagnostic_started = time.perf_counter()
                work.start('beta_same_coefficient_shadow_diagnostic')
                case['shadow_native'] = shadow_native_displacement(
                    trainer, parent_state, reference_gradient
                )
                work.finish('beta_same_coefficient_shadow_diagnostic')
                case['shadow_native']['wall_s'] = time.perf_counter() - diagnostic_started
            case.update(
                status='completed',
                rows=result['rows'],
                endpoint=_endpoint_identity(trainer),
                physical_work_delta={
                    key: value - ancillary['work'].get(key, 0)
                    for key, value in trainer.work.items()
                },
                native_response_observations=deepcopy(trainer.response_records),
                native_window_observations=deepcopy(trainer.window_records),
            )
            require(
                case['native_adam_steps'] == 1 and case['endpoint']['gradients_cleared'],
                'Disposable case did not complete one clean Adam step',
            )
            cases[name] = case
        except BaseException as error:
            case.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            trainer.backend.optimizer.step = original_step
            write_json(output / f'{name}.json', case)
            _restore_parent(trainer, parent_state, runtime, ancillary)
            case['restoration'] = verify(
                trainer, parent_state, output / f'{name}_RESTORATION.json'
            )
            write_json(output / f'{name}.json', case)
    disabled_exact = (
        cases['CONTROL']['endpoint'] == cases['DISABLED']['endpoint']
        and cases['CONTROL']['preclip_gradient'] == cases['DISABLED']['preclip_gradient']
        and _science_rows(cases['CONTROL']['rows']) == _science_rows(cases['DISABLED']['rows'])
    )
    same_native_support = [row['sampled_anchors'] for row in cases['CONTROL']['rows']] == [
        row['sampled_anchors'] for row in cases['CANDIDATE']['rows']
    ]
    activity = dict(
        coefficient_positions_changed=sum(
            int((a != s['roles']['beta']).sum()) for a, s in zip(coefficients, snapshots)
        ),
        preclip_gradient_changed=cases['BETA']['preclip_gradient']
        != cases['CANDIDATE']['preclip_gradient'],
        bf16_parameters_changed=cases['BETA']['endpoint']['parameters']
        != cases['CANDIDATE']['endpoint']['parameters'],
        first_moment_changed=cases['BETA']['endpoint']['exp_avg']
        != cases['CANDIDATE']['endpoint']['exp_avg'],
        second_moment_changed=cases['BETA']['endpoint']['exp_avg_sq']
        != cases['CANDIDATE']['endpoint']['exp_avg_sq'],
    )
    active = (
        activity['coefficient_positions_changed'] > 0
        and activity['preclip_gradient_changed']
        and activity['bf16_parameters_changed']
    )
    result = dict(
        status=(
            'passed'
            if disabled_exact and same_native_support and active
            else 'inactive_candidate' if not active else 'integration_failed'
        ),
        passed=disabled_exact and same_native_support and active,
        control_disabled_exact=disabled_exact,
        cached_native_anchor_identity=same_native_support,
        beta_g0_exact=cases['BETA']['reference_gradient_exact'],
        activity=activity,
        candidate_active=active,
        candidate_native_training_s=cases['CANDIDATE']['native_training_s'],
        candidate_timing_scope='Two native response boundaries minus qualification preclip hashing/inspection; endpoint hashes, restoration and shadow diagnostic excluded',
        disposable_adam_steps=4,
        fixed_windows=1,
        parent_restored_after_every_case=True,
    )
    write_json(output / 'INTEGRATION_GATES.json', result)
    return result


def run_integration(
    parent_plan, original_probe, numerical_audit, output, execution_commit, step_ledger
):
    """Separate admitted process: startup, one cache calculation, four updates."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    audit_path = Path(numerical_audit) / 'RESULT.json'
    audit = json.loads(audit_path.read_text())
    require(
        audit.get('passed') is True
        and audit.get('status') == 'numerically_qualified'
        and audit.get('restoration', {}).get('status') == 'passed'
        and audit.get('restoration', {}).get('exact_resume') is True,
        'Numerical allocation admission and exact restoration must pass first',
    )
    ledger = DisposableStepLedger(
        step_ledger, identity(Path(original_probe) / 'WINDOW.json'), output
    )
    receipt = dict(
        status='preparing',
        schema='frontier_native_integration_v2',
        execution_commit=execution_commit,
        numerical_audit_identity=identity(audit_path),
        native_adam_steps=0,
        production_qualified=False,
    )
    write_json(output / 'RESULT.json', receipt)
    work = WorkReceipt(output / 'WORK.json')
    trainer = None
    state = baseline_runtime = baseline_ancillary = None
    started = time.perf_counter()
    try:
        plan, recovery, selected = select_parent(parent_plan)
        require(
            os.environ.get('CUDA_VISIBLE_DEVICES') == plan['devices']['B'],
            'Integration must use the sealed physical training GPU B',
        )
        write_json(
            output / 'SOURCE_INPUTS.json',
            dict(
                parent_recovery=PARENT_RECOVERY,
                parent_plan_identity=identity(parent_plan),
                selected_features=selected,
                original_window_identity=identity(Path(original_probe) / 'WINDOW.json'),
            ),
        )
        write_json(output / 'RUNTIME_COMPATIBILITY.json', pin_retained_graph_runtime(output))
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
        state = torch.load(recovery, map_location='cpu', weights_only=False, mmap=True)
        baseline_runtime = runtime_state(trainer.bundle.model)
        baseline_ancillary = dict(
            responses=deepcopy(trainer.response_records),
            windows=deepcopy(trainer.window_records),
            anchors=deepcopy(trainer._anchor_observations),
            work=deepcopy(trainer.work),
            costs=deepcopy(trainer.bundle.model.costs),
            prefix_summary=deepcopy(trainer.bundle.model.prefix_allocation.last_summary),
            active_event=getattr(trainer, '_active_event', None),
        )
        paths = [item['path'] for item in selected]
        receipt['startup'] = discarded_startup(trainer, paths, output / 'startup')
        verify_live(trainer, state, output / 'AFTER_STARTUP_STATE.json')
        torch.cuda.reset_peak_memory_stats()
        original_window = json.loads((Path(original_probe) / 'WINDOW.json').read_text())
        torch.cuda.synchronize()
        cache_started = time.perf_counter()
        responses = freeze_two_responses(trainer, paths)
        window_receipt = require_original_window(responses, original_window)
        scorer = NativeAllocationScorer(trainer, responses, work)
        snapshots = scorer.freeze_current()
        gradient = scorer.reference_gradient()
        preview = scorer.preview_and_direction(gradient)
        fast = scorer.fast_scores(preview['w'])
        torch.cuda.synchronize()
        receipt['cache_scoring_wall_s'] = time.perf_counter() - cache_started
        write_json(output / 'WINDOW.json', window_receipt)
        write_json(
            output / 'CACHE.json',
            dict(
                preview=preview['diagnostics'],
                allocations=json_diagnostics(fast['allocation_diagnostics']),
                memory_inventory=scorer.memory_inventory(),
            ),
        )
        del preview
        del fast['effective_direction']
        scorer.close()
        receipt['cache_physical_work'] = deepcopy(trainer.work)
        receipt.update(
            qualify_integration(
                trainer,
                state,
                selected,
                responses,
                snapshots,
                fast['allocations'],
                gradient,
                work,
                output / 'cases',
                step_ledger=ledger,
            )
        )
        receipt['complete_candidate_window_s'] = (
            receipt['cache_scoring_wall_s'] + receipt['candidate_native_training_s']
        )
        receipt['complete_candidate_timing_scope'] = (
            'Synchronized frozen-window sampling/scoring plus candidate native teaching; excludes startup, parent/state hashes, restores, other branches and BETA shadow diagnostic; bounded work-ledger bookkeeping included'
        )
        receipt['step_ledger_identity'] = identity(step_ledger)
        receipt['production_qualified'] = receipt['passed']
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        restoration_error = None
        if trainer is not None and baseline_runtime is not None:
            try:
                _restore_parent(trainer, state, baseline_runtime, baseline_ancillary)
                receipt['restoration'] = verify_live(
                    trainer, state, output / 'FINAL_RESTORATION.json'
                )
            except BaseException as error:
                restoration_error = error
                receipt['restoration_error'] = traceback.format_exc()
                receipt['status'] = 'failed'
                receipt['passed'] = receipt['production_qualified'] = False
        receipt.update(
            wall_s=time.perf_counter() - started,
            work=deepcopy(work.value),
            native_adam_steps=work.value['native_adam_steps'],
        )
        if torch.cuda.is_initialized():
            receipt.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        write_json(output / 'RESULT.json', receipt)
        if trainer is not None:
            trainer.close()
        if restoration_error is not None and 'error' not in receipt:
            raise restoration_error
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        'parent-plan',
        'original-probe',
        'numerical-audit',
        'output',
        'execution-commit',
        'step-ledger',
    ):
        parser.add_argument('--' + option, required=True)
    run_integration(
        **{key.replace('-', '_'): value for key, value in vars(parser.parse_args()).items()}
    )
