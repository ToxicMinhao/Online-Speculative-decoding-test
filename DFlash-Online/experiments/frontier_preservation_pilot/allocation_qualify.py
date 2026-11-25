"""Zero-Adam full-window audit of the declared native token contractions.

Serial token gradients are a bounded qualification oracle only. This entry
never trains, changes a threshold, reselects pools, or repeats a failed attempt.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import os
import sys
import time
import traceback

import torch

from experiments.full_k_gsm8k.training import verify_live
from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot.training import PilotTrainer
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import write_json

from .allocation_contract import (
    direction_rounding_diagnostics,
    evaluate_pool,
    numerical_contract,
    qualify_selected_pools,
    select_pools,
    validate_native_invariants,
)
from .native import (
    PARENT_RECOVERY,
    WorkReceipt,
    freeze_two_responses,
    rng_digest,
    select_parent,
)
from .repair_native import require_original_window

DOT_CHUNK_ELEMENTS = 1_048_576


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def json_diagnostics(value):
    """Decode compact per-block diagnostics, never model parameters or gradients."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_diagnostics(item) for item in value]
    return value


class ReferenceLedger:
    """Reserve every token before its graph begins; interruptions spend allowance."""

    def __init__(
        self, path, attempt, output, binding, *, retry_of=None, correction_note=None
    ):
        self.path = Path(path)
        self.attempt = str(attempt)
        if attempt not in (1, 2):
            raise ValueError(
                'Only the initial audit and one identified correction are allowed'
            )
        if self.path.exists():
            self.value = json.loads(self.path.read_text())
            if self.value['binding'] != binding:
                raise ValueError(
                    'Reference allowance belongs to a different contract or parent window'
                )
        else:
            if attempt != 1:
                raise ValueError('A correction cannot start a new reference allowance')
            self.value = dict(
                schema='allocation_reference_ledger_v1', binding=binding, attempts={}
            )
        if self.attempt in self.value['attempts']:
            raise ValueError('An allocation attempt cannot be restarted or overwritten')
        if attempt == 2:
            if (
                set(self.value['attempts']) != {'1'}
                or retry_of is None
                or correction_note is None
            ):
                raise ValueError(
                    'Correction requires the original attempt and an identified fix'
                )
            first = self.value['attempts']['1']
            if first['output'] != str(Path(retry_of).resolve()):
                raise ValueError('Correction does not bind the original attempt output')
            note = Path(correction_note).read_text().strip()
            if not note or first.get('status') == 'numerically_qualified':
                raise ValueError(
                    'Correction needs a failed or interrupted audit and a nonempty explanation'
                )
            correction = dict(
                retry_of=str(retry_of), note=note, identity=identity(correction_note)
            )
        else:
            if (
                self.value['attempts']
                or retry_of is not None
                or correction_note is not None
            ):
                raise ValueError('Initial audit cannot discard an earlier allowance')
            correction = None
        self.value['attempts'][self.attempt] = dict(
            output=str(Path(output).resolve()),
            status='running',
            correction=correction,
            records=[],
        )
        self.save()

    @property
    def current(self):
        return self.value['attempts'][self.attempt]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, self.value)

    def bind_selection(self, selection):
        digest = json_digest(selection)
        if self.attempt == '2':
            old = self.value['attempts']['1'].get('selection_sha256')
            if old is not None and old != digest:
                raise ValueError(
                    'Correction changed the prospectively selected complete pools'
                )
        self.current['selection_sha256'] = digest
        self.save()

    def reserve(self, occurrence):
        total = sum(
            len(attempt['records']) for attempt in self.value['attempts'].values()
        )
        if len(self.current['records']) >= 60 or total >= 120:
            raise ValueError('Serial reference allowance is exhausted')
        self.current['records'].append(dict(**occurrence, status='graph_started'))
        self.save()

    def mark(self, status):
        self.current['records'][-1]['status'] = status
        self.save()

    def finish(self, status):
        self.current['status'] = status
        self.save()


def fp64_contractions(
    gradients, primary, effective, *, chunk_elements=DOT_CHUNK_ELEMENTS
):
    """Two dots in bounded FP64 chunks, without a token-gradient bank."""
    if chunk_elements < 1:
        raise ValueError('Contraction chunks must be positive')
    if not len(gradients) == len(primary) == len(effective):
        raise ValueError('Reference gradient and direction membership differ')
    first, second = [], []
    for gradient, direction, fast_direction in zip(gradients, primary, effective):
        if gradient is None:
            raise ValueError('Unexpected unconnected native trainable parameter')
        if gradient.shape != direction.shape or gradient.shape != fast_direction.shape:
            raise ValueError('Reference gradient and direction shapes differ')
        flat = gradient.detach().reshape(-1)
        full = direction.detach().reshape(-1)
        fast = fast_direction.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            g = flat[start : start + chunk_elements].double()
            w = full[start : start + chunk_elements].double()
            v = fast[start : start + chunk_elements].double()
            if not all(bool(torch.isfinite(value).all()) for value in (g, w, v)):
                raise ValueError(
                    'Nonfinite native reference gradient or scoring direction'
                )
            first.append(float((g * w).sum()))
            second.append(float((g * v).sum()))
    return math.fsum(first), math.fsum(second)


def vector_identity(names, values):
    if len(names) != len(values):
        raise ValueError('Direction names and tensors differ')
    digest = hashlib.sha256()
    tensors = []
    for name, value in zip(names, values):
        metadata = dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), DOT_CHUNK_ELEMENTS):
            data = (
                flat[start : start + DOT_CHUNK_ELEMENTS]
                .contiguous()
                .cpu()
                .view(torch.uint8)
            )
            digest.update(data.numpy().tobytes())
        tensors.append(metadata)
    return dict(sha256=digest.hexdigest(), tensors=tensors)


def direction_metadata(names, primary, effective, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Fast direction scale must be finite and positive')
    if len(primary) != len(effective):
        raise ValueError('Primary and effective direction membership differ')
    equal = all(
        torch.equal(w, v.to(w.dtype) / scale) for w, v in zip(primary, effective)
    )
    return dict(
        primary=vector_identity(names, primary),
        effective=vector_identity(names, effective),
        direction_scale=scale,
        normalization='secondary contraction divides by direction_scale once',
        effective_direction_equals_primary=equal,
    )


def selected_values(tensor, pool):
    return tensor.detach().reshape(-1, 16)[pool['block_index'], pool['positions']].cpu()


def serial_references(
    scorer, snapshots, selection, primary, effective, scale, ledger, work, output
):
    rows = []
    for pool in selection['pools']:
        response_index, block = pool['response_index'], pool['block_index']
        denominator = float(snapshots[response_index]['roles']['denominator'])
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError('Invalid frozen native response denominator')
        for position in pool['positions']:
            occurrence = dict(
                response_index=response_index, block_index=block, position=position
            )
            ledger.reserve(occurrence)
            work.start('serial_reference_token_graph')
            token_loss = scorer.token_fkl(response_index, block, position)
            work.finish('serial_reference_token_graph')
            if token_loss.numel() != 1 or not bool(torch.isfinite(token_loss)):
                raise ValueError('Native unweighted token FKL is not a finite scalar')
            ledger.mark('backward_started')
            work.start('serial_reference_token_vjp')
            gradients = torch.autograd.grad(
                token_loss, scorer.parameters, create_graph=False, allow_unused=False
            )
            work.finish('serial_reference_token_vjp')
            ledger.mark('backward_completed')
            work.start('serial_reference_fp64_contractions')
            dot, effective_dot = fp64_contractions(gradients, primary, effective)
            work.finish('serial_reference_fp64_contractions')
            score = -dot / (2 * denominator)
            effective_score = -effective_dot / (2 * denominator * scale)
            rows.append(
                dict(
                    **occurrence,
                    unweighted_native_fkl=float(token_loss.detach()),
                    native_fkl_dtype=str(token_loss.dtype),
                    primary_dot_fp64=dot,
                    effective_dot_fp64=effective_dot,
                    denominator=denominator,
                    accumulation_factor=0.5,
                    reference_score_fp64=score,
                    reference_score_fp32=float(
                        torch.tensor(score, dtype=torch.float32)
                    ),
                    effective_reference_score_fp64=effective_score,
                )
            )
            del gradients, token_loss
            ledger.mark('completed')
            write_json(
                output / 'REFERENCE_PROGRESS.json',
                dict(
                    status='running',
                    completed=len(rows),
                    expected=selection['reference_backward_count'],
                    rows=rows,
                ),
            )
    write_json(
        output / 'REFERENCE_PROGRESS.json',
        dict(status='completed', completed=len(rows), rows=rows),
    )
    return rows


def aggregate_effect(primary, reference_gradient, allocated_gradient, snapshots, fast):
    if not len(primary) == len(reference_gradient) == len(allocated_gradient):
        raise ValueError('Aggregate gradient parameter membership differs')
    native_parts, reference_norm, allocated_norm, delta_norm = [], [], [], []
    for w, before, after in zip(primary, reference_gradient, allocated_gradient):
        wf, bf, af = (value.detach().reshape(-1) for value in (w, before, after))
        for start in range(0, wf.numel(), DOT_CHUNK_ELEMENTS):
            left, right = (
                bf[start : start + DOT_CHUNK_ELEMENTS].double(),
                af[start : start + DOT_CHUNK_ELEMENTS].double(),
            )
            delta = right - left
            native_parts.append(
                float((wf[start : start + DOT_CHUNK_ELEMENTS].double() * delta).sum())
            )
            reference_norm.append(float(left.square().sum()))
            allocated_norm.append(float(right.square().sum()))
            delta_norm.append(float(delta.square().sum()))
    raw, centered, mass_residuals = [], [], []
    for snapshot, scores, coefficients in zip(
        snapshots, fast['scores'], fast['allocations']
    ):
        roles = snapshot['roles']
        beta, prefix = roles['beta'].double(), roles['prefix']
        delta = torch.where(prefix, coefficients.double() - beta, 0.0)
        values = scores.double()
        lowest = torch.where(prefix, values, torch.inf).amin(-1, keepdim=True)
        origin = torch.where(prefix.any(-1, keepdim=True), lowest, 0.0)
        offsets = torch.where(prefix, values - origin, 0.0)
        mass = torch.where(prefix, beta, 0.0).sum(-1, keepdim=True)
        mean = (beta * offsets).sum(-1, keepdim=True) / torch.where(mass > 0, mass, 1.0)
        centered.append(float((delta * (offsets - mean)).sum()))
        raw.append(float((delta * values).sum()))
        mass_residuals.extend(delta.sum(-1).reshape(-1).cpu().tolist())
    native = -math.fsum(native_parts)
    prediction = math.fsum(centered)
    return dict(
        native_effect_fp64=native,
        centered_score_prediction_fp64=prediction,
        raw_score_prediction_fp64=math.fsum(raw),
        native_minus_centered_prediction=native - prediction,
        reference_gradient_l2=math.sqrt(math.fsum(reference_norm)),
        allocated_gradient_l2=math.sqrt(math.fsum(allocated_norm)),
        native_gradient_difference_l2=math.sqrt(math.fsum(delta_norm)),
        block_mass_residuals_fp64=mass_residuals,
        admission_gate=False,
        interpretation='Native rounded aggregate backward need not equal the token-gradient extension.',
    )


def audit_window(scorer, responses, ledger, work, output, verify_callback):
    """Complete both responses before any numerical admission decision."""
    output = Path(output)
    snapshots = scorer.freeze_current()
    selection = select_pools(
        [snapshot['roles']['prefix'] for snapshot in snapshots],
        [response.record_id for response in responses],
    )
    ledger.bind_selection(selection)
    write_json(output / 'POOL_SELECTION.json', selection)
    if not selection['passed']:
        return dict(status='coverage_blocked', selection=selection, passed=False)
    frozen_repair = [
        snapshot['repair_coefficients'].detach().clone() for snapshot in snapshots
    ]
    work.start('reference_window_gradient')
    g0 = scorer.reference_gradient()
    work.finish('reference_window_gradient')
    work.start('optimizer_shadow_and_assessment_direction')
    preview = scorer.preview_and_direction(g0)
    work.finish('optimizer_shadow_and_assessment_direction')
    primary = preview['w']
    if len(primary) != len(scorer.parameters) or any(
        w.dtype != torch.float32 for w in primary
    ):
        raise ValueError(
            'Declared primary direction must contain all represented FP32 parameters'
        )
    write_json(
        output / 'PREVIEW.json',
        dict(
            diagnostics=json_diagnostics(preview['diagnostics']),
            memory_inventory=preview['memory_inventory'],
        ),
    )
    primary_before = vector_identity(scorer.parameter_names, primary)
    repeats, invariant_receipts = [], []
    fast_rng = capture_rng()
    for repeat in range(2):
        restore_rng(fast_rng)
        work.start('fast_scoring_window')
        fast = scorer.fast_scores(primary)
        work.finish('fast_scoring_window')
        if rng_digest(capture_rng()) != rng_digest(fast_rng):
            raise ValueError('Fast scoring changed the frozen post-window RNG')
        if repeat:
            if fast['direction_scale'] != repeats[0]['direction_scale'] or not all(
                torch.equal(a, b)
                for a, b in zip(
                    fast['effective_direction'], repeats[0]['effective_direction']
                )
            ):
                raise ValueError(
                    'Repeated fast calculation changed its represented direction'
                )
        invariants = [
            validate_native_invariants(
                fast['scores'][index],
                fast['allocations'][index],
                snapshot['roles'],
                fast['repair_coefficients'][index],
                frozen_repair[index],
                direct_loss_mask=fast['direct_loss_masks'][index],
                response_denominator=fast['response_denominators'][index],
                accumulation_factor=fast['accumulation_factor'],
            )
            for index, snapshot in enumerate(snapshots)
        ]
        invariant_receipts.append(invariants)
        write_json(
            output / f'FAST_{repeat + 1}.json',
            dict(
                invariants=invariants,
                allocation_diagnostics=json_diagnostics(fast['allocation_diagnostics']),
                attention_work=fast['attention_work'],
            ),
        )
        verify_callback(f'FAST_{repeat + 1}_STATE.json')
        if repeat:
            del fast['effective_direction']
        repeats.append(fast)
    effective = repeats[0]['effective_direction']
    scale = float(repeats[0]['direction_scale'])
    metadata = direction_metadata(scorer.parameter_names, primary, effective, scale)
    if metadata['primary'] != primary_before:
        raise ValueError('Fast calculations mutated the fixed FP32 scoring direction')
    metadata['fixed_primary_unchanged'] = True
    write_json(output / 'DIRECTIONS.json', metadata)
    write_json(
        output / 'REFERENCE_BINDING.json',
        dict(
            parent_and_original_window=ledger.value['binding'],
            selection_sha256=ledger.current['selection_sha256'],
            directions=metadata,
            response_denominators=[
                float(snapshot['roles']['denominator']) for snapshot in snapshots
            ],
            current_state_verification=['FAST_1_STATE.json', 'FAST_2_STATE.json'],
            head_layout='Native full eight-block chunk, including its unchanged final remainder',
            loss='Native BF16 unweighted token FKL; create_graph=False',
            source_binding='RESULT.json execution_commit and SELECTION.json authenticated parent inputs',
        ),
    )
    rows = serial_references(
        scorer, snapshots, selection, primary, effective, scale, ledger, work, output
    )
    pools = []
    for pool in selection['pools']:
        matching = [
            row
            for row in rows
            if row['response_index'] == pool['response_index']
            and row['block_index'] == pool['block_index']
        ]
        if [row['position'] for row in matching] != pool['positions']:
            raise ValueError('Serial reference records lost selected pool order')
        reference = torch.tensor(
            [row['reference_score_fp64'] for row in matching], dtype=torch.float64
        )
        secondary = torch.tensor(
            [row['effective_reference_score_fp64'] for row in matching],
            dtype=torch.float64,
        )
        response_index = pool['response_index']
        first, second = [
            selected_values(fast['scores'][response_index], pool) for fast in repeats
        ]
        pool_receipt = evaluate_pool(
            selected_values(snapshots[response_index]['roles']['beta'], pool),
            reference,
            first,
            second,
        )
        pool_receipt.update(pool)
        pool_receipt['direction_diagnostic'] = direction_rounding_diagnostics(
            reference,
            secondary,
            first,
            metadata=metadata,
        )
        pools.append(pool_receipt)
    gates = qualify_selected_pools(selection, pools, invariant_receipts)
    write_json(
        output / 'NUMERICAL_GATES.json',
        dict(summary=gates, pools=pools, invariants=invariant_receipts),
    )
    work.start('allocated_window_gradient')
    allocated = scorer.reference_gradient(repeats[0]['allocations'])
    work.finish('allocated_window_gradient')
    effect = aggregate_effect(primary, g0, allocated, snapshots, repeats[0])
    write_json(output / 'AGGREGATE_EFFECT.json', effect)
    verify_callback('COMPLETE_AUDIT_STATE.json')
    del allocated, g0, primary, effective
    return dict(**gates, aggregate_effect=effect, selected_pool_receipts=pools)


def run_allocation_audit(
    parent_plan,
    original_probe,
    output,
    execution_commit,
    contract,
    reference_ledger,
    attempt,
    *,
    retry_of=None,
    correction_note=None,
):
    contract_path = Path(contract)
    if json.loads(contract_path.read_text()) != numerical_contract():
        raise ValueError(
            'Prospectively sealed numerical contract differs from this implementation'
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        schema='native_token_contraction_audit_v2',
        status='preparing',
        execution_commit=execution_commit,
        numerical_contract_path=str(contract_path),
        numerical_contract_identity=identity(contract_path),
        attempt=attempt,
        native_adam_steps=0,
        production_qualified=False,
        historical_hidden_state_gate='failed_retained_diagnostic_not_production_veto',
    )
    work = WorkReceipt(output / 'WORK.json')
    write_json(output / 'RESULT.json', receipt)
    trainer, ledger = None, None
    started = time.perf_counter()
    try:
        plan, recovery, selected = select_parent(parent_plan)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['devices']['B']:
            raise ValueError(
                'Allocation audit must see only the sealed physical training GPU B'
            )
        old_window_path = Path(original_probe) / 'WINDOW.json'
        original_window = json.loads(old_window_path.read_text())
        binding = dict(
            parent=PARENT_RECOVERY,
            contract=receipt['numerical_contract_identity'],
            original_window=identity(old_window_path),
        )
        ledger = ReferenceLedger(
            reference_ledger,
            attempt,
            output,
            binding,
            retry_of=retry_of,
            correction_note=correction_note,
        )
        write_json(
            output / 'SELECTION.json',
            dict(
                parent_plan=str(parent_plan),
                parent_binding=plan['binding'],
                parent_recovery=str(recovery),
                parent_recovery_identity=PARENT_RECOVERY,
                selected_ordered_training_features=selected,
                original_window_identity=binding['original_window'],
            ),
        )
        runtime = pin_retained_graph_runtime(output)
        runtime.update(
            scope='full-window native token contraction qualification',
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
        initial_gradients = [
            (parameter, parameter.grad) for parameter in model.parameters()
        ]
        modes = [(module, module.training) for module in model.modules()]
        parent_state = torch.load(
            recovery, map_location='cpu', weights_only=False, mmap=True
        )
        torch.cuda.reset_peak_memory_stats()
        receipt['state_verification_wall_s'] = 0.0
        expected_window_rng = None

        def verify_unchanged(filename):
            current_rng = capture_rng()
            before = time.perf_counter()
            work.start('scientific_state_verification')
            try:
                if rng_digest(current_rng) != expected_window_rng:
                    raise ValueError(
                        'Diagnostic scoring changed the expected post-window RNG'
                    )
                restore_rng(parent_rng)
                verify_live(trainer, parent_state, output / filename)
                work.finish('scientific_state_verification')
            finally:
                restore_rng(current_rng)
                receipt['state_verification_wall_s'] += time.perf_counter() - before

        scorer = None
        try:
            work.start('original_native_anchor_window_sampling')
            responses = freeze_two_responses(trainer, [row['path'] for row in selected])
            work.finish('original_native_anchor_window_sampling')
            expected_window_rng = rng_digest(capture_rng())
            receipt['expected_post_window_rng_sha256'] = expected_window_rng
            write_json(
                output / 'WINDOW.json',
                require_original_window(responses, original_window),
            )
            from .allocation_scoring import NativeAllocationScorer

            scorer = NativeAllocationScorer(trainer, responses, work=work)
            receipt['status'] = 'running'
            write_json(output / 'RESULT.json', receipt)
            outcome = audit_window(
                scorer, responses, ledger, work, output, verify_unchanged
            )
            receipt.update(outcome)
            if any(
                parameter.grad is not initial
                for parameter, initial in initial_gradients
            ):
                raise ValueError('Numerical audit left native gradient buffers changed')
        finally:
            if sys.exc_info()[0] is not None:
                receipt['error_before_restoration'] = traceback.format_exc()
            cleanup_error = None
            try:
                if scorer is not None:
                    scorer.close()
            except BaseException as error:
                cleanup_error = error
                receipt['scorer_cleanup_error'] = traceback.format_exc()
            receipt['physical_work_delta'] = {
                key: value - initial_work.get(key, 0)
                for key, value in trainer.work.items()
            }
            receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            receipt['gradient_buffers_present_before_restoration'] = sum(
                parameter.grad is not None for parameter, _ in initial_gradients
            )
            write_json(output / 'RESULT.json', receipt)
            for parameter, gradient in initial_gradients:
                parameter.grad = gradient
            for module, mode in modes:
                module.training = mode
            restore_rng(parent_rng)
            trainer.work, model.costs = initial_work, initial_costs
            trainer._anchor_observations[:] = initial_anchors
            receipt['restoration'] = verify_live(
                trainer, parent_state, output / 'FINAL_RESTORATION.json'
            )
            del parent_state
            if cleanup_error is not None:
                raise cleanup_error
    except BaseException as error:
        receipt.update(
            status='failed', error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        receipt['wall_s'] = time.perf_counter() - started
        receipt['work'] = deepcopy(work.value)
        if ledger is not None:
            ledger.finish(receipt['status'])
            receipt['reference_ledger_identity'] = identity(reference_ledger)
        write_json(output / 'RESULT.json', receipt)
        if trainer is not None:
            trainer.close()
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        'parent-plan',
        'original-probe',
        'output',
        'execution-commit',
        'contract',
        'reference-ledger',
    ):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2), required=True)
    parser.add_argument('--retry-of')
    parser.add_argument('--correction-note')
    args = parser.parse_args()
    result = run_allocation_audit(
        args.parent_plan,
        args.original_probe,
        args.output,
        args.execution_commit,
        args.contract,
        args.reference_ledger,
        args.attempt,
        retry_of=args.retry_of,
        correction_note=args.correction_note,
    )
    print(
        json.dumps(
            {key: result[key] for key in ('status', 'native_adam_steps', 'wall_s')}
        )
    )
