"""Bounded synthetic allocation check: 2 responses, 4 views, zero Adam steps.

Right padding and repeated retained anchors exercise allocation at the declared
2272-token/512-block shape. Losses are validity telemetry, never learning or
response-quality evidence. This function does not create new feedback.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time

import torch

from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_continuation.resume import differences
from experiments.prefix_ranking.qualify import cpu_copy
from online_feedback.adversarial_block_runtime import AdversarialBlockRuntime
from online_feedback.ranking import fixed_anchors
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import save_torch, write_json


SEQUENCE_LENGTH = 2272
ANCHOR_BLOCKS = 512
RESOURCE_BUDGET = dict(response_forwards=4, response_backwards=4, optimizer_steps=0)


def allocation_fixture(batch, draw, *, sequence_length=SEQUENCE_LENGTH, anchor_blocks=ANCHOR_BLOCKS):
    """Extend normalized sequence tensors, preserving their existing contents."""
    length = batch.tensors['input_ids'].shape[1]
    if length > sequence_length or sequence_length <= 0 or anchor_blocks <= 0:
        raise ValueError('Resource shape must extend the retained response without truncation')
    tensors = dict(batch.tensors)
    for name in ('input_ids', 'loss_mask', 'hidden_states', 'target_last_hidden_states'):
        original = tensors[name]
        if original.ndim < 2 or original.shape[:2] != (1, length):
            raise ValueError('Resource tensors require one aligned normalized response')
        padding = original.new_zeros((1, sequence_length-length, *original.shape[2:]))
        tensors[name] = torch.cat((original, padding), dim=1)
    anchors = torch.as_tensor(draw['positions'], dtype=torch.long)
    keep = torch.as_tensor(draw['keep_mask'], dtype=torch.bool)
    if anchors.ndim != 2 or anchors.shape[0] != 1 or anchors.numel() == 0 or keep.shape != anchors.shape:
        raise ValueError('Resource anchors must come from one nonempty retained native draw')
    if bool((anchors < 0).any() | (anchors >= length).any()) or not bool(keep.any()):
        raise ValueError('Retained resource anchors are out of bounds or all inactive')
    index = torch.arange(anchor_blocks) % anchors.shape[1]
    anchors, keep = anchors[:, index], keep[:, index]
    metadata = dict(batch.metadata, resource_fixture=dict(
        original_length=length, sequence_length=sequence_length,
        right_padded_tokens=sequence_length-length, anchor_blocks=anchor_blocks,
        anchor_construction='cyclic repetition of retained native positions and keep mask',
        input_padding='zero tokens, loss mask and both teacher feature tensors',
        scientific_feedback=False, loss_interpretation=False))
    return replace(batch, tensors=tensors, metadata=metadata), anchors, keep


def _state(trainer):
    return dict(weights=dict(trainer.bundle.draft_model.state_dict()),
                optimizer=trainer.backend.optimizer.state_dict(),
                counters=(trainer.global_step, trainer.presentations, trainer.core._micro),
                ratios=trainer.core._ratio_totals, sums=trainer.core._sum_totals,
                anchor_observations=trainer._anchor_observations,
                response_records=trainer.response_records, window_records=trainer.window_records)


def run(trainer, features, output):
    """Use a warm qualified ADV case; preserve its endpoint and checkpoint.

    The native core is never entered. Each clean/ADV pair uses manual native
    backend backward, with the same .5-view / 2-response scaling as production.
    All synthetic gradients are discarded without an optimizer application.
    """
    from specforge.training.strategies.base import StepContext
    features = tuple(Path(path) for path in features)
    if len(features) != 2 or len(trainer.response_records) != 2:
        raise ValueError('Resource fixture requires the two completed native qualification responses')
    if trainer.core.accumulation_remainder or any(p.grad is not None for p in trainer.bundle.draft_model.parameters()):
        raise ValueError('Resource fixture must start at a clean native Adam boundary')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    ledger = WorkLedger(root/'WORK.jsonl', RESOURCE_BUDGET)
    model, backend = trainer.bundle.model, trainer.backend
    device = next(model.draft_model.parameters()).device
    started = time.perf_counter()
    before = cpu_copy(_state(trainer))
    saved_runtime, saved_rng = runtime_state(model), capture_rng()
    saved_work, saved_costs = deepcopy(trainer.work), deepcopy(model.costs)
    frozen_versions = [(parameter, parameter._version) for parameter in model.parameters()
                       if not parameter.requires_grad]
    receipt = dict(status='running', fixture='synthetic_allocation_only',
        sequence_length=SEQUENCE_LENGTH, anchor_blocks=ANCHOR_BLOCKS,
        budget=RESOURCE_BUDGET, rows=[], additional_feedback=0,
        scientific_loss_interpretation=False, state_capture_s=time.perf_counter()-started)
    failure = None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        receipt['baseline_allocated_bytes'] = torch.cuda.memory_allocated(device)
        receipt['baseline_reserved_bytes'] = torch.cuda.memory_reserved(device)
    execution_started = time.perf_counter()
    try:
        runtime = AdversarialBlockRuntime(model, arm='ADV', rho=.01)
        for ordinal, path in enumerate(features):
            raw = torch.load(path, map_location='cpu', weights_only=True)
            if str(raw['record_id']) != str(trainer.response_records[ordinal]['record_id']):
                raise ValueError('Resource packet order differs from retained native response records')
            draws = trainer.response_records[ordinal]['sampled_anchors']
            if len(draws) != 1:
                raise ValueError('Resource response requires exactly one retained anchor draw')
            batch, anchors, keep = allocation_fixture(trainer._batch(raw), draws[0],
                sequence_length=SEQUENCE_LENGTH, anchor_blocks=ANCHOR_BLOCKS)
            context = StepContext(global_step=trainer.global_step,
                                  total_steps=trainer.horizon, collect_detailed_metrics=True)
            with fixed_anchors(model, anchors.to(device), keep.to(device)):
                with runtime.response(batch, ('resource', trainer.dataset, ordinal)):
                    clean = ledger.call('response_forwards', trainer._native_forward, batch, context)
                    ledger.call('response_backwards', backend.backward, .5*clean.loss/2, is_boundary=False)
                    clean_loss = clean.loss.detach()
                    del clean
                    runtime.finish_clean()
                    with runtime.second_view():
                        second = ledger.call('response_forwards', trainer._native_forward, batch, context)
                        ledger.call('response_backwards', backend.backward, .5*second.loss/2,
                                    is_boundary=ordinal == len(features)-1)
                        second_loss = second.loss.detach()
                        del second
                    diagnostic = runtime.diagnostics(clean_loss, second_loss)
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                       for p in model.draft_model.parameters() if p.requires_grad):
                raise ValueError('Resource fixture has missing or nonfinite draft gradients')
            save_torch(root/f'RESPONSE_{ordinal}.pt', diagnostic)
            receipt['rows'].append(dict(record_id=str(batch.sample_ids[0]),
                **batch.metadata['resource_fixture'],
                teacher_feature_shapes={name:list(batch.tensors[name].shape) for name in
                    ('hidden_states', 'target_last_hidden_states')},
                clean_loss_for_validity=diagnostic['clean_loss'],
                adversarial_loss_for_validity=diagnostic['second_loss']))
            del raw, batch, diagnostic
        require_complete_work(read_work(ledger.path, ledger.limits), RESOURCE_BUDGET)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        receipt['execution_s'] = time.perf_counter()-execution_started
        receipt['status'] = 'passed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['work'] = read_work(ledger.path, ledger.limits)
        receipt['physical_work'] = {key:trainer.work.get(key, 0)-saved_work.get(key, 0)
                                    for key in set(trainer.work) | set(saved_work)}
        if device.type == 'cuda':
            receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        else:
            receipt['peak_allocated_bytes'] = receipt['peak_reserved_bytes'] = 0
        verification_started = time.perf_counter()
        try:
            backend.optimizer.zero_grad()
            restore_rng(saved_rng)
            restore_runtime(model, saved_runtime)
            trainer.work.clear()
            trainer.work.update(saved_work)
            model.costs = saved_costs
            mismatch = differences(before, _state(trainer))
            frozen_unchanged = all(parameter._version == version for parameter, version in frozen_versions)
            receipt.update(endpoint_exact=not mismatch, frozen_parameter_versions_exact=frozen_unchanged,
                           rng_exact=not differences(saved_rng, capture_rng()),
                           gradients_cleared=all(p.grad is None for p in model.parameters()),
                           state_mismatches=mismatch[:16])
            if mismatch or not frozen_unchanged or not receipt['rng_exact'] or not receipt['gradients_cleared']:
                raise ValueError('Resource fixture did not preserve the disposable native endpoint')
        except BaseException as cleanup_error:
            receipt.update(status='failed', state_verification_error=repr(cleanup_error))
            if failure is not None:
                failure.add_note('Resource state verification: '+repr(cleanup_error))
            else:
                failure = cleanup_error
        receipt.update(state_verification_s=time.perf_counter()-verification_started,
                       wall_s=time.perf_counter()-started)
        try:
            write_json(root/'RESULT.json', receipt)
        except BaseException as report_error:
            if failure is not None:
                failure.add_note('Resource receipt write: '+repr(report_error))
            else:
                raise
        if failure is not None and 'state_verification_error' in receipt and 'error' not in receipt:
            raise failure
    return receipt
