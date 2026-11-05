"""Bounded KW qualification with identical fresh-process startup for K and KW.

A CPU coordinator executes two fresh Python processes, sequentially, with one
shared durable work ledger. Each child executes the same four retained responses
once. Temporary K states/gradients support exact checks without another model
pass; they remain available for explicit verified retirement after review.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from online_feedback.storage import write_json, save_torch
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.kw_training import KWContinuationTrainer
from online_feedback.response_weighting import NativeAdamWPreview, apply_weighted_gradients_
from .diagnostics import compare_exact
from .resume import OKContinuationTrainer

LIMITS = dict(response_forwards=8, response_backwards=8,
              disposable_optimizer_applications=16, serving_generations=0,
              persistent_optimizer_updates=0)
EXPECTED_WORK = {**LIMITS, 'disposable_optimizer_applications': 14}
GIB = 1024**3
TRANSIENT_CAP_BYTES = 30 * GIB
RESERVE_BYTES = 8 * GIB


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


def file_identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


class WorkLedger:
    """One global ledger, reopened sequentially; charge before every dispatch."""

    def __init__(self, path, *, resume=False):
        self.path = Path(path)
        if resume:
            saved = json.loads(self.path.read_text())
            if saved['limits'] != LIMITS:
                raise ValueError('Qualification work limits changed')
            self.work, self.records = saved['work'], saved['attempts']
            if (set(self.work) != set(LIMITS) or any(type(v) is not int or not 0 <= v <= LIMITS[k]
                                                     for k, v in self.work.items())):
                raise ValueError('Invalid saved qualification counters')
            observed = {kind: sum(row['kind'] == kind for row in self.records) for kind in LIMITS}
            if observed != self.work or any(row['status'] not in ('started', 'completed') for row in self.records):
                raise ValueError('Qualification work receipt and attempts differ')
        else:
            if self.path.exists():
                raise FileExistsError(self.path)
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


def scientific_state(trainer):
    return host_copy(dict(weights=trainer.bundle.draft_model.state_dict(),
                          optimizer=trainer.backend.optimizer.state_dict(), rng=capture_rng(),
                          global_step=trainer.global_step, presentations=trainer.presentations,
                          micro=trainer.core._micro, accumulation=trainer.core.accumulation_remainder,
                          modes={name: module.training for name, module in trainer.bundle.model.named_modules()}))


def gradient_state(trainer):
    return {name: None if p.grad is None else host_copy(p.grad)
            for name, p in trainer.bundle.draft_model.named_parameters() if p.requires_grad}


def optimizer_state(trainer):
    native = getattr(trainer.backend.optimizer, 'native', trainer.backend.optimizer)
    return host_copy(dict(weights=trainer.bundle.draft_model.state_dict(),
                          optimizer=native.state_dict(), rng=capture_rng()))


def restore_optimizer_state(trainer, saved):
    native = getattr(trainer.backend.optimizer, 'native', trainer.backend.optimizer)
    trainer.bundle.draft_model.load_state_dict(saved['weights'], strict=True)
    native.load_state_dict(deepcopy(saved['optimizer']))
    native.zero_grad()
    restore_rng(saved['rng'])


def count_training_calls(trainer, ledger, arm, after_backward=None):
    strategy = trainer.core.strategy
    original_forward, original_backward = strategy.forward_loss, trainer.backend.backward
    backward_index = 0

    def forward(*args, **kwargs):
        record = ledger.start('response_forwards', arm=arm)
        result = original_forward(*args, **kwargs)
        ledger.complete(record)
        return result

    def backward(*args, **kwargs):
        nonlocal backward_index
        record = ledger.start('response_backwards', arm=arm)
        result = original_backward(*args, **kwargs)
        ledger.complete(record)
        if after_backward is not None:
            after_backward(backward_index, trainer)
        backward_index += 1
        return result
    strategy.forward_loss = forward
    trainer.backend.backward = backward


def run_two_windows(trainer, rows, ledger, arm, boundary, *, after_backward=None):
    from specforge.training.strategies.base import StepContext
    count_training_calls(trainer, ledger, arm, after_backward)
    observations = []
    trainer.bundle.model.begin_event()
    try:
        for index, row in enumerate(rows):
            raw = torch.load(row['path'], map_location='cpu', weights_only=True)
            if raw['record_id'] != row['record_id'] or raw['input_ids'].shape[-1] != row['length']:
                raise ValueError('Sealed qualification feature identity differs')
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(trainer._batch(raw), StepContext(
                global_step=trainer.global_step, total_steps=trainer.horizon, collect_detailed_metrics=True))
            metrics = result.materialize_metrics()
            if not all(math.isfinite(float(v)) for v in metrics.values()):
                raise ValueError('Nonfinite qualification training metrics')
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
            observations.append(dict(record_id=row['record_id'], metrics=metrics,
                                     anchors=deepcopy(trainer._anchor_observations), stepped=result.optimizer_stepped, step=trainer.global_step))
            write_json(ledger.path.parent / (arm + '_RESPONSES.json'), observations)
            if result.optimizer_stepped:
                boundary(index // 2, trainer)
    finally:
        objective_work = trainer.bundle.model.end_event()
    return dict(responses=observations, objective_work=objective_work)


def optimizer_only_checks(trainer, saved, gradients, ledger, checks, *, receipt_path=None):
    """Six applications reuse captured gradients; no forward/backward occurs."""
    native = trainer.backend.optimizer.native
    names = trainer.named_parameters
    captured = [[None if g is None else g.to(p.device)
                 for g, (_, p) in zip(group, names)] for group in gradients]

    def exact(left, right, label):
        compare_exact(left, right, label, checks, receipt_path=receipt_path)

    for allocation in (-.25, .137, .25):
        restore_optimizer_state(trainer, saved)
        for (_, parameter), gradient in zip(names, captured[0]):
            parameter.grad = None if gradient is None else gradient.clone()
        before = optimizer_state(trainer)
        before_grad = host_copy([p.grad for _, p in names])
        preview = NativeAdamWPreview(native.optimizer, names, max_grad_norm=native.max_grad_norm)
        attempt = None

        def counted(_):
            nonlocal attempt
            attempt = ledger.start('disposable_optimizer_applications', arm='KW',
                                   variant='nonzero_preview', allocation=allocation)
        preview.apply(*captured, allocation, on_optimizer_step=counted)
        ledger.complete(attempt)
        exact(optimizer_state(trainer), before, f'preview_isolation_{allocation}')
        exact(host_copy([p.grad for _, p in names]), before_grad,
              f'preview_gradient_isolation_{allocation}')
        proposed_weights = host_copy([p for p in preview.parameters])
        proposed_optimizer = host_copy(preview.optimizer.state_dict())
        del preview, before, before_grad
        apply_weighted_gradients_(names, *captured, allocation)
        attempt = ledger.start('disposable_optimizer_applications', arm='KW',
                               variant='nonzero_native', allocation=allocation)
        native.step()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        ledger.complete(attempt)
        exact(host_copy([p for _, p in names]), proposed_weights,
              f'nonzero_native_parameters_{allocation}')
        exact(host_copy(native.optimizer.state_dict()),
              proposed_optimizer, f'nonzero_native_Adam_{allocation}')
        exact(capture_rng(), saved['rng'], f'nonzero_native_RNG_{allocation}')
        del proposed_weights, proposed_optimizer
    restore_optimizer_state(trainer, saved)


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def save_transient(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(exist_ok=True)
    occupied = sum(p.stat().st_size for p in path.parent.glob('*.pt'))
    allowance = _tensor_bytes(value) + 16 * 1024**2
    if occupied + allowance > TRANSIENT_CAP_BYTES:
        raise RuntimeError('Qualification transient tensor budget exceeds30GiB')
    if shutil.disk_usage(path.parent).free < allowance + RESERVE_BYTES:
        raise RuntimeError('Qualification tensor write would cross8GiB reserve')
    save_torch(path, value)
    result = dict(path=str(path), **file_identity(path))
    identity_path = path.parent / 'IDENTITIES.json'
    identities = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    identities[str(path)] = {key: result[key] for key in ('bytes', 'sha256')}
    write_json(identity_path, identities)
    return result


def load_transient(entry):
    if file_identity(entry['path']) != {k: entry[k] for k in ('bytes', 'sha256')}:
        raise ValueError('Ordinary qualification tensor identity changed: ' + entry['path'])
    return torch.load(entry['path'], map_location='cpu', weights_only=False)


def startup_metadata(arm, *, allow_cpu_fixture=False):
    from specforge.modeling.draft.flex_attention import WrappedFlexAttention, WrappedCreateBlockMask
    compiled_flex = WrappedFlexAttention._instance is not None
    compiled_mask = WrappedCreateBlockMask._instance is not None
    if not allow_cpu_fixture and (torch.cuda.is_initialized() or compiled_flex or compiled_mask):
        raise ValueError(
            'Qualification worker must begin with fresh CUDA and compiled-attention state')
    keys = ('CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_COMPILE_DEBUG', 'CUBLAS_WORKSPACE_CONFIG', 'TORCH_LOGS')
    return dict(arm=arm, pid=os.getpid(), parent_pid=os.getppid(), python=sys.executable,
                started_unix_s=time.time(), cuda_initialized=torch.cuda.is_initialized(),
                flex_singleton_exists=compiled_flex, mask_singleton_exists=compiled_mask,
                cpu_fixture=allow_cpu_fixture, environment={k: os.environ.get(k) for k in keys},
                torch_version=torch.__version__, torch_git_version=torch.version.git_version,
                dynamo_assume_static_by_default=torch._dynamo.config.assume_static_by_default,
                dynamo_automatic_dynamic_shapes=torch._dynamo.config.automatic_dynamic_shapes)


def run_worker(plan, root, arm, *, allow_cpu_fixture=False):
    """One child only; production entrypoints never call this inline."""
    if arm not in ('K', 'KW'):
        raise ValueError('Unknown qualification worker')
    root = Path(root)
    output = root / 'implementation_qualification'
    result_path = output / (arm + '_WORKER_RESULT.json')
    if result_path.exists():
        raise FileExistsError(result_path)
    ledger = WorkLedger(output / 'WORK.json', resume=True)
    checks = json.loads((output / 'CHECKS.json').read_text()
                        ) if (output / 'CHECKS.json').exists() else []
    receipt = dict(status='running', arm=arm, checks=checks, work=dict(ledger.work), startup={})
    write_json(result_path, receipt)
    trainer = None
    primary_error = None
    cleanup_error = None
    started = time.perf_counter()
    source = None
    source_before = None

    def exact(left, right, label):
        compare_exact(left, right, label, checks, receipt_path=output / 'CHECKS.json')

    try:
        receipt['startup'] = startup_metadata(arm, allow_cpu_fixture=allow_cpu_fixture)
        write_json(result_path, receipt)
        selection = json.loads(Path(plan['qualification']).read_text())
        source = selection['source_state']
        source_before = file_identity(source)
        if source_before['sha256'] != selection['source_state_identity']['sha256']:
            raise ValueError('Original K8 identity differs before qualification worker')
        if len(selection['rows']) != 4:
            raise ValueError('Require exactly four sealed qualification responses')
        if arm == 'K':
            if any(ledger.work.values()):
                raise ValueError('Ordinary K worker requires unused global work budget')
            trainer = OKContinuationTrainer(
                plan['config'], output / 'K', resume_state=source, completed_event=8, arm='K')
        else:
            expected = {**dict.fromkeys(LIMITS, 0), 'response_forwards': 4,
                        'response_backwards': 4, 'disposable_optimizer_applications': 2}
            if ledger.work != expected or any(r['status'] != 'completed' for r in ledger.records):
                raise ValueError('KW requires the exact completed K child budget')
            trainer = KWContinuationTrainer(
                plan['config'], output / 'KW', resume_state=source, completed_event=8)
            trainer._event_start_presentations = trainer.presentations
            trainer._window_log = output / 'KW_WINDOWS.jsonl'
        if not allow_cpu_fixture and not torch.cuda.is_initialized():
            raise ValueError('Production qualification worker did not initialize its pinned GPU')
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
        receipt['startup']['parameter_names'] = [
            n for n, p in trainer.bundle.draft_model.named_parameters() if p.requires_grad]
        receipt['startup']['optimizer_parameter_names'] = [r[0]
                                                           for r in trainer.backend.optimizer.recipe['parameters']]
        receipt['startup']['backend_wrapper'] = getattr(trainer.backend, '_wrapper_kind', None)
        write_json(result_path, receipt)
        if arm == 'K':
            artifacts = dict(initial=save_transient(output / 'transient/K_initial.pt', scientific_state(trainer)),
                             backward=[], boundaries=[], preclip=[])
            receipt['artifacts'] = artifacts
            write_json(result_path, receipt)

            def after_backward(index, live):
                artifacts['backward'].append(save_transient(
                    output / f'transient/K_backward_{index}.pt', gradient_state(live)))
                write_json(result_path, receipt)
            original_step = trainer.backend.optimizer.step

            def ordinary_step():
                index = len(artifacts['preclip'])
                saved = artifacts['backward'][2 * index + 1]
                exact(gradient_state(trainer), load_transient(
                    saved), f'K_preclip_equals_backward_{index}')
                artifacts['preclip'].append(saved)
                record = ledger.start('disposable_optimizer_applications',
                                      arm='K', variant='ordinary_K')
                result = original_step()
                ledger.complete(record)
                return result
            trainer.backend.optimizer.step = ordinary_step

            def boundary(index, live):
                artifacts['boundaries'].append(save_transient(
                    output / f'transient/K_boundary_{index}.pt', scientific_state(live)))
                write_json(result_path, receipt)
            receipt['training_records'] = run_two_windows(
                trainer, selection['rows'], ledger, 'K', boundary, after_backward=after_backward)
        else:
            ordinary = json.loads((output / 'K_WORKER_RESULT.json').read_text())
            if ordinary['status'] != 'passed':
                raise ValueError('Ordinary K child did not pass')
            artifacts = ordinary['artifacts']
            capture = {}
            isolation = {}
            pending = []
            exact(scientific_state(trainer), load_transient(
                artifacts['initial']), 'initial_K8_scientific_state')

            def after_backward(index, live):
                exact(gradient_state(live), load_transient(
                    artifacts['backward'][index]), f'after_backward_{index}')

            def observe(stage, live, payload):
                index = payload['identity']['global_step'] - 1025
                if stage == 'pre_step':
                    exact(gradient_state(live), load_transient(
                        artifacts['preclip'][index]), f'preclip_gradient_{index}')
                    isolation['state'] = optimizer_state(live)
                    isolation['gradients'] = gradient_state(live)
                    if not capture:
                        capture['state'] = isolation['state']
                        capture['gradients'] = host_copy(
                            [payload['native'], payload['first'], payload['second']])
                elif stage == 'post_previews':
                    # Both previews have returned. Complete their ledger entries
                    # before an isolation assertion can stop the live update.
                    for row in pending:
                        ledger.complete(row)
                    pending.clear()
                    exact(optimizer_state(live), isolation['state'],
                          f'neutral_preview_scientific_isolation_{index}')
                    exact(gradient_state(live), isolation['gradients'],
                          f'neutral_preview_gradient_isolation_{index}')
                    isolation.clear()
                elif stage == 'post_step':
                    for row in pending:
                        ledger.complete(row)
                    pending.clear()
            trainer.qualification_observer = observe
            trainer.optimizer_application_observer = lambda a: pending.append(ledger.start(
                'disposable_optimizer_applications', arm='KW', variant='KW_window', allocation=a))

            def boundary(index, live):
                exact(scientific_state(live), load_transient(artifacts['boundaries'][index]),
                      f'neutral_window_{index+1}')

            weighted = run_two_windows(
                trainer, selection['rows'], ledger, 'KW', boundary, after_backward=after_backward)
            exact(weighted['responses'], ordinary['training_records']
                  ['responses'], 'ordinary_metrics_and_anchors')
            if [row['a'] for row in trainer.window_rows] != [0., 0.]:
                raise ValueError('First two KW windows must be neutral')
            if (trainer.controller.n, trainer.controller.windows, trainer.controller.previews) != (1, 2, 4):
                raise ValueError('Neutral qualification controller accounting differs')
            receipt['controller_after_two_windows'] = dict(phi=trainer.controller.phi.tolist(), v=trainer.controller.v,
                                                           n=trainer.controller.n, windows=trainer.controller.windows, previews=trainer.controller.previews)
            receipt['training_records'] = weighted
            trainer.qualification_observer = None
            trainer.optimizer_application_observer = None
            trainer.controller.clear_pending()
            optimizer_only_checks(
                trainer, capture['state'], capture['gradients'], ledger, checks, receipt_path=output / 'CHECKS.json')
            receipt.update(native_neutral_exact=True, preview_isolation=True)
        if file_identity(source) != source_before:
            raise ValueError('Retained K8 state changed')
        receipt.update(status='passed', cuda_initialized=torch.cuda.is_initialized())
        return receipt
    except BaseException as error:
        primary_error = error
        receipt.update(status='failed', error=repr(error),
                       traceback=traceback.format_exc(), automatic_retry=False)
        raise
    finally:
        if torch.cuda.is_initialized():
            receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
        else:
            receipt.update(peak_allocated_bytes=0, peak_reserved_bytes=0)
        receipt.update(wall_s=time.perf_counter() - started, work=dict(ledger.work), checks=checks)
        if trainer is not None:
            trainer.failed = receipt['status'] != 'passed'
            try:
                trainer.close()
            except BaseException as error:
                receipt.setdefault('cleanup_errors', []).append(repr(error))
                if primary_error is None:
                    cleanup_error = error
                    primary_error = error
                    receipt.update(status='failed', error=repr(error),
                                   traceback=traceback.format_exc(), automatic_retry=False)
        try:
            write_json(result_path, receipt)
        except BaseException as error:
            if primary_error is not None:
                primary_error.add_note('Could not write worker receipt: ' + repr(error))
            else:
                raise
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if cleanup_error is not None:
            raise cleanup_error


def dispatch_worker(arm, plan_path, root):
    """Fresh EXEC, never multiprocessing fork or an inherited model runtime."""
    output = Path(root) / 'implementation_qualification'
    command = [sys.executable, '-u', '-m', 'experiments.kw_v1.qualify', '--worker', arm,
               '--plan', str(plan_path), '--root', str(root)]
    write_json(output / (arm + '_COMMAND.json'), dict(command=command,
                                                      parent_pid=os.getpid(), automatic_retry=False))
    with (output / (arm + '_WORKER.log')).open('xb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        receipt_path = output / (arm + '_WORKER_RESULT.json')
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        raise RuntimeError(f'{arm} qualification worker exited {result.returncode}: ' +
                           str(receipt.get('error', 'see child log')))
    return json.loads((output / (arm + '_WORKER_RESULT.json')).read_text())


def qualify(plan, root, *, worker_executor=None):
    root = Path(root)
    output = root / 'implementation_qualification'
    output.mkdir()
    ledger = WorkLedger(output / 'WORK.json')
    started = time.perf_counter()
    primary_error = None
    injected = worker_executor is not None
    executor = worker_executor or dispatch_worker
    receipt = dict(status='running', bindings={}, checks=[], work=dict(ledger.work), persistent_updates=0,
                   serving_generations=0, workers={}, execution_mode='injected_CPU_fixture' if injected else 'fresh_exec_children',
                   coordinator_pid=os.getpid(), coordinator_cuda_initialized=torch.cuda.is_initialized())
    write_json(root / 'IMPLEMENTATION_QUALIFICATION.json', receipt)
    try:
        if torch.cuda.is_initialized():
            raise ValueError('Qualification coordinator must remain CPU-only')
        if not injected and shutil.disk_usage(root).free < 40 * GIB + RESERVE_BYTES:
            raise RuntimeError('Qualification needs40GiB temporary/overlap writes plus8GiB reserve')
        selection = json.loads(Path(plan['qualification']).read_text())
        source = selection['source_state']
        bindings = dict(source_k8_sha256=file_identity(source)['sha256'],
                        qualification_inputs_sha256=hashlib.sha256(json.dumps(
                            selection, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                        execution_commit=plan['execution_source']['commit'])
        receipt['bindings'] = bindings
        if bindings['source_k8_sha256'] != selection['source_state_identity']['sha256']:
            raise ValueError('Original K8 identity differs before qualification')
        plan_path = output / 'WORKER_PLAN.json'
        write_json(plan_path, plan)
        for arm in ('K', 'KW'):
            receipt['active_worker'] = arm
            write_json(root / 'IMPLEMENTATION_QUALIFICATION.json', receipt)
            result = executor(arm, plan_path, root)
            if result.get('status') != 'passed':
                raise ValueError('Qualification worker did not pass: ' + arm)
            receipt['workers'][arm] = result
            receipt['checks'] = result.get('checks', [])
            receipt['work'] = dict(WorkLedger(output / 'WORK.json', resume=True).work)
            write_json(root / 'IMPLEMENTATION_QUALIFICATION.json', receipt)
        final = WorkLedger(output / 'WORK.json', resume=True)
        if final.work != EXPECTED_WORK or any(row['status'] != 'completed' for row in final.records):
            raise ValueError('Qualification work is incomplete or differs from8/8/14')
        k, kw = receipt['workers']['K'], receipt['workers']['KW']
        if not injected and (k['startup']['pid'] == kw['startup']['pid'] or any(
                not row.get('cuda_initialized') or row['startup'].get('cpu_fixture') for row in (k, kw))):
            raise ValueError('Require two distinct fresh GPU execution processes')
        if file_identity(source)['sha256'] != bindings['source_k8_sha256']:
            raise ValueError('Retained K8 state changed')
        receipt.update(status='passed', active_worker=None, native_neutral_exact=kw['native_neutral_exact'],
                       preview_isolation=kw['preview_isolation'], full_model_memory_checked=not injected,
                       ordinary_peak_allocated_bytes=k['peak_allocated_bytes'], ordinary_peak_reserved_bytes=k['peak_reserved_bytes'],
                       peak_allocated_bytes=kw['peak_allocated_bytes'], peak_reserved_bytes=kw['peak_reserved_bytes'],
                       controller_after_two_windows=kw['controller_after_two_windows'],
                       training_records={a: r['training_records']
                                         for a, r in receipt['workers'].items()},
                       transient_artifacts=k['artifacts'], transient_retention='Retained for review; explicit verified retirement required before production admission.')
        return receipt
    except BaseException as error:
        primary_error = error
        receipt.update(status='failed', error=repr(error),
                       traceback=traceback.format_exc(), automatic_retry=False)
        raise
    finally:
        final = WorkLedger(output / 'WORK.json', resume=True)
        receipt.update(work=final.work, work_unknown=[r for r in final.records if r['status'] != 'completed'],
                       wall_s=time.perf_counter() - started, coordinator_cuda_initialized=torch.cuda.is_initialized())
        for arm in ('K', 'KW'):
            path = output / (arm + '_WORKER_RESULT.json')
            if path.exists():
                worker = json.loads(path.read_text())
                receipt['workers'][arm] = worker
                receipt['checks'] = worker.get('checks', receipt['checks'])
                if worker.get('cleanup_errors'):
                    receipt.setdefault('cleanup_errors', []).extend(worker['cleanup_errors'])
        try:
            write_json(root / 'IMPLEMENTATION_QUALIFICATION.json', receipt)
        except BaseException as error:
            if primary_error is not None:
                primary_error.add_note(
                    'Could not write final qualification receipt: ' + repr(error))
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--worker', choices=('K', 'KW'))
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if args.worker:
        run_worker(plan, args.root, args.worker)
    else:
        qualify(plan, args.root)


if __name__ == '__main__':
    main()
