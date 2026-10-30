"""One bounded K8/KW native qualification; never publishes a trained model.

The two paths each process the four presealed responses exactly once. Native
K snapshots stay in host memory for exact comparisons. Optimizer-only probes
reuse the first KW window's gradients and pre-step state; no extra model call.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch

from online_feedback.storage import write_json
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.kw_training import KWContinuationTrainer
from online_feedback.response_weighting import NativeAdamWPreview, apply_weighted_gradients_
from experiments.ok_continuation.resume import differences
from .resume import OKContinuationTrainer


LIMITS = dict(response_forwards=8, response_backwards=8,
              disposable_optimizer_applications=16, serving_generations=0,
              persistent_optimizer_updates=0)


def host_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: host_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [host_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(host_copy(item) for item in value)
    return deepcopy(value)


class WorkLedger:
    """Write an attempt before dispatch so a failure cannot hide consumed work."""
    def __init__(self, path):
        self.path = Path(path)
        self.work = dict.fromkeys(LIMITS, 0)
        self.records = []
        self.save()

    def save(self):
        write_json(self.path, dict(work=self.work, limits=LIMITS, attempts=self.records))

    def start(self, kind, **identity):
        if self.work[kind] >= LIMITS[kind]:
            raise RuntimeError('Qualification budget exhausted: ' + kind)
        self.work[kind] += 1
        record = dict(kind=kind, ordinal=self.work[kind], status='started', **identity)
        self.records.append(record)
        self.save()
        return record

    def complete(self, record):
        record['status'] = 'completed'
        self.save()


def exact(left, right, path, checks):
    mismatch = differences(left, right)
    row = dict(check=path, status='failed' if mismatch else 'passed', differences=mismatch[:100])
    checks.append(row)
    if mismatch:
        raise ValueError(path + ': ' + ', '.join(mismatch[:8]))


def scientific_state(trainer):
    return host_copy(dict(weights=trainer.bundle.draft_model.state_dict(),
        optimizer=trainer.backend.optimizer.state_dict(), rng=capture_rng(),
        global_step=trainer.global_step, presentations=trainer.presentations,
        micro=trainer.core._micro, accumulation=trainer.core.accumulation_remainder,
        modes={name: module.training for name, module in trainer.bundle.model.named_modules()}))


def optimizer_state(trainer):
    native = getattr(trainer.backend.optimizer, 'native', trainer.backend.optimizer)
    return host_copy(dict(weights=trainer.bundle.draft_model.state_dict(),
                         optimizer=native.state_dict(), rng=capture_rng()))


def restore_optimizer_state(trainer, saved):
    native = getattr(trainer.backend.optimizer, 'native', trainer.backend.optimizer)
    trainer.bundle.draft_model.load_state_dict(saved['weights'], strict=True)
    native.load_state_dict(saved['optimizer'])
    native.zero_grad()
    restore_rng(saved['rng'])


def count_training_calls(trainer, ledger, arm):
    strategy = trainer.core.strategy
    original_forward = strategy.forward_loss
    original_backward = trainer.backend.backward

    def forward(*args, **kwargs):
        record = ledger.start('response_forwards', arm=arm)
        result = original_forward(*args, **kwargs)
        ledger.complete(record)
        return result

    def backward(*args, **kwargs):
        record = ledger.start('response_backwards', arm=arm)
        result = original_backward(*args, **kwargs)
        ledger.complete(record)
        return result

    strategy.forward_loss = forward
    trainer.backend.backward = backward


def run_two_windows(trainer, rows, ledger, arm, boundary):
    from specforge.training.strategies.base import StepContext
    count_training_calls(trainer, ledger, arm)
    observations = []
    trainer.bundle.model.begin_event()
    try:
        for index, row in enumerate(rows):
            raw = torch.load(row['path'], map_location='cpu', weights_only=True)
            if raw['record_id'] != row['record_id'] or raw['input_ids'].shape[-1] != row['length']:
                raise ValueError('Sealed qualification feature identity differs')
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(trainer._batch(raw), StepContext(
                global_step=trainer.global_step, total_steps=trainer.horizon,
                collect_detailed_metrics=True))
            metrics = result.materialize_metrics()
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError('Nonfinite qualification training metrics')
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
            observations.append(dict(record_id=row['record_id'], metrics=metrics,
                anchors=deepcopy(trainer._anchor_observations),
                stepped=result.optimizer_stepped, step=trainer.global_step))
            write_json(ledger.path.parent / (arm + '_RESPONSES.json'), observations)
            if result.optimizer_stepped:
                boundary(index // 2, trainer)
    finally:
        objective_work = trainer.bundle.model.end_event()
    return dict(responses=observations, objective_work=objective_work)


def optimizer_only_checks(trainer, saved, gradients, ledger, checks):
    """Six optimizer applications, using actual native clipping/Adam dispatch."""
    native = trainer.backend.optimizer.native
    names = trainer.named_parameters
    captured = [[None if g is None else g.to(p.device) for g, (_, p) in zip(group, names)]
                for group in gradients]
    for allocation in (-0.25, 0.137, 0.25):
        restore_optimizer_state(trainer, saved)
        for (_, parameter), gradient in zip(names, captured[0]):
            parameter.grad = None if gradient is None else gradient.clone()
        before = optimizer_state(trainer)
        before_grad = host_copy([p.grad for _, p in names])
        preview = NativeAdamWPreview(native.optimizer, names, max_grad_norm=native.max_grad_norm)
        attempt = None

        def counted(_):
            nonlocal attempt
            attempt = ledger.start('disposable_optimizer_applications',
                                   variant='nonzero_preview', allocation=allocation)

        preview.apply(*captured, allocation, on_optimizer_step=counted)
        ledger.complete(attempt)
        exact(optimizer_state(trainer), before, f'preview_isolation_{allocation}', checks)
        exact(host_copy([p.grad for _, p in names]), before_grad,
              f'preview_gradient_isolation_{allocation}', checks)
        proposed_weights = host_copy([p for p in preview.parameters])
        proposed_optimizer = host_copy(preview.optimizer.state_dict())
        del preview, before, before_grad
        apply_weighted_gradients_(names, *captured, allocation)
        attempt = ledger.start('disposable_optimizer_applications',
                               variant='nonzero_native', allocation=allocation)
        native.step()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        ledger.complete(attempt)
        exact(host_copy([p for _, p in names]), proposed_weights,
              f'nonzero_native_parameters_{allocation}', checks)
        exact(host_copy(native.optimizer.state_dict()), proposed_optimizer,
              f'nonzero_native_Adam_{allocation}', checks)
        exact(capture_rng(), saved['rng'], f'nonzero_native_RNG_{allocation}', checks)
        del proposed_weights, proposed_optimizer
    restore_optimizer_state(trainer, saved)


def qualify(plan, root):
    root = Path(root)
    output = root / 'implementation_qualification'
    output.mkdir()  # Existing evidence is never replaced by a retry.
    ledger = WorkLedger(output / 'WORK.json')
    checks, states = [], []
    started = time.perf_counter()
    trainer = None
    primary_error = None
    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = dict(status='running', bindings={}, checks=checks,
                   work=ledger.work, persistent_updates=0, serving_generations=0)
    write_json(root / 'IMPLEMENTATION_QUALIFICATION.json', receipt)
    try:
        selection = json.loads(Path(plan['qualification']).read_text())
        source = selection['source_state']
        bindings = dict(source_k8_sha256=digest(source),
            qualification_inputs_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(),
            execution_commit=plan['execution_source']['commit'])
        receipt['bindings'] = bindings
        if bindings['source_k8_sha256'] != selection['source_state_identity']['sha256']:
            raise ValueError('Original K8 identity differs before qualification')
        trainer = OKContinuationTrainer(plan['config'], output/'K', resume_state=source,
                                        completed_event=8, arm='K')
        torch.cuda.reset_peak_memory_stats()
        original_step = trainer.backend.optimizer.step
        def ordinary_step():
            record = ledger.start('disposable_optimizer_applications', variant='ordinary_K')
            result = original_step()
            ledger.complete(record)
            return result
        trainer.backend.optimizer.step = ordinary_step
        ordinary = run_two_windows(trainer, selection['rows'], ledger, 'K',
                                   lambda index, live: states.append(scientific_state(live)))
        receipt['training_records'] = dict(K=ordinary)
        write_json(root/'IMPLEMENTATION_QUALIFICATION.json', receipt)
        receipt['ordinary_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        receipt['ordinary_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        trainer.close(); del trainer; trainer = None
        gc.collect(); torch.cuda.empty_cache()
        trainer = KWContinuationTrainer(plan['config'], output/'KW', resume_state=source,
                                        completed_event=8)
        trainer._event_start_presentations = trainer.presentations
        trainer._window_log = output/'KW_WINDOWS.jsonl'
        capture = {}
        pending_applications = []
        def observe(stage, live, payload):
            if stage == 'pre_step' and not capture:
                capture['state'] = optimizer_state(live)
                capture['gradients'] = host_copy([payload['native'], payload['first'], payload['second']])
            if stage == 'post_step':
                for record in pending_applications:
                    ledger.complete(record)
                pending_applications.clear()
        trainer.qualification_observer = observe
        trainer.optimizer_application_observer = lambda a: pending_applications.append(
            ledger.start('disposable_optimizer_applications', variant='KW_window', allocation=a))
        torch.cuda.reset_peak_memory_stats()
        weighted = run_two_windows(trainer, selection['rows'], ledger, 'KW',
            lambda index, live: exact(scientific_state(live), states[index],
                                      f'neutral_window_{index+1}', checks))
        exact(weighted['responses'], ordinary['responses'], 'ordinary_metrics_and_anchors', checks)
        if [row['a'] for row in trainer.window_rows] != [0.0, 0.0]:
            raise ValueError('First two zero-initialized KW windows must be neutral')
        if (trainer.controller.n, trainer.controller.windows, trainer.controller.previews) != (1, 2, 4):
            raise ValueError('Neutral qualification controller accounting differs')
        receipt['controller_after_two_windows'] = dict(phi=trainer.controller.phi.tolist(),
            v=trainer.controller.v, n=trainer.controller.n, windows=trainer.controller.windows,
            previews=trainer.controller.previews)
        receipt['training_records'] = dict(K=ordinary, KW=weighted)
        del states[:]
        trainer.qualification_observer = None
        trainer.optimizer_application_observer = None
        trainer.controller.clear_pending()
        optimizer_only_checks(trainer, capture['state'], capture['gradients'], ledger, checks)
        receipt.update(full_model_memory_checked=True, native_neutral_exact=True,
                       preview_isolation=True, status='passed')
        if digest(source) != bindings['source_k8_sha256']:
            raise ValueError('Retained K8 state changed')
    except BaseException as error:
        primary_error = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                       automatic_retry=False)
        raise
    finally:
        if torch.cuda.is_initialized():
            receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
        receipt.update(wall_s=time.perf_counter()-started, work=ledger.work, checks=checks)
        if trainer is not None:
            trainer.failed = receipt['status'] != 'passed'
            try:
                trainer.close()
            except BaseException as cleanup_error:
                receipt.setdefault('cleanup_errors', []).append(repr(cleanup_error))
                receipt['status'] = 'failed'
                if primary_error is not None:
                    primary_error.add_note('Qualification cleanup also failed: '+repr(cleanup_error))
                else:
                    primary_error = cleanup_error
        try:
            write_json(root/'IMPLEMENTATION_QUALIFICATION.json', receipt)
        except BaseException as receipt_error:
            if primary_error is not None:
                primary_error.add_note('Qualification receipt write also failed: '+repr(receipt_error))
            else:
                primary_error = receipt_error
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    if primary_error is not None:
        raise primary_error
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    qualify(json.loads(Path(args.plan).read_text()), args.root)


if __name__ == '__main__':
    main()
