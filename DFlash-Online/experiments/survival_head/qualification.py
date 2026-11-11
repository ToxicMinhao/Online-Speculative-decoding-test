"""Four disposable native windows; no state initializes the main trajectories."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import traceback

import torch

from experiments.kplus_abt.qualification import endpoint, finite_endpoint
from experiments.functional_step_geometry.native_pilot import discarded_startup
from experiments.functional_step_geometry.qualify import cleanup, export_disposable_state
from experiments.full_k_gsm8k.training import live_state
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime, persist_comparison
from online_feedback.storage import save_torch, write_json
from .contract import ARMS, QUALIFICATION_CASES
from .data import read


def case_limits(case):
    if case not in QUALIFICATION_CASES:
        raise ValueError('Unknown qualification case')
    active = case in ('LOCAL', 'SURVIVAL')
    return dict(response_forwards=2, response_backwards=2, optimizer_steps=1,
                auxiliary_backwards=2 if active else 0, head_optimizer_steps=int(active))


def auxiliary_gradient_check(auxiliary, parameters, ledger):
    """Same clean graph, separately charged backward, exact existing .grad check."""
    before = [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
    values = ledger.call('auxiliary_backwards', torch.autograd.grad,
        auxiliary, parameters, retain_graph=True, allow_unused=True)
    finite = all(g is None or bool(torch.isfinite(g).all()) for g in values)
    norm_sq = sum(float(g.detach().double().square().sum()) for g in values if g is not None)
    unchanged = all((saved is None and p.grad is None) or
        (saved is not None and p.grad is not None and torch.equal(saved, p.grad.detach().cpu()))
        for saved, p in zip(before, parameters))
    return dict(finite=finite, norm_sq=norm_sq, nonzero=norm_sq > 0,
                native_gradient_buffer_unchanged=unchanged)


def verified_inputs(root):
    pool_path = Path(root)/'feedback/POOL.json'
    pool = read(pool_path)
    if len(pool.get('records', [])) != 128:
        raise ValueError('Qualification must reuse the sealed128-response pool')
    selected = pool['records'][:2]
    for row in selected:
        for key in ('feature', 'canonical'):
            expected = row[key]
            if identity(expected['path']) != {name:expected[name] for name in ('sha256', 'bytes')}:
                raise ValueError('Qualification packet identity changed')
        if read(row['canonical']['path'])['record_id'] != row['record_id']:
            raise ValueError('Qualification canonical request identity changed')
    return selected, dict(pool_identity=identity(pool_path), packets=selected)


def run_case(root, case):
    from specforge.training.strategies.base import StepContext
    from online_feedback.survival_head import response_bce
    from .training import SurvivalHeadTrainer, validate_head_state

    root = Path(root); plan = read(root/'sealed/PLAN.json')
    if case not in QUALIFICATION_CASES:
        raise ValueError('Unknown qualification case')
    output = root/'qualification'/case
    output.mkdir(parents=True, exist_ok=False)
    selected, input_evidence = verified_inputs(root)
    features = [Path(row['feature']['path']) for row in selected]
    arm = 'LOCAL' if case == 'DISABLED' else case
    ledger = WorkLedger(output/'WORK.jsonl', case_limits(case))
    trainer = None; failure = None; started = time.perf_counter()
    receipt = dict(status='running', case=case, binding=plan['binding'],
        execution_commit=plan['execution_source']['commit'], inputs=input_evidence,
        auxiliary_gradient_checks=[])
    gradients = {}
    try:
        pin_retained_graph_runtime(output)
        trainer = SurvivalHeadTrainer(plan['configs'][arm], output/'trainer', arm=arm,
            binding=plan['binding'], original_config_text=Path(plan['original_config']).read_text(),
            origin_state=plan['origin_state'], origin_export=plan['origin_export'])
        receipt['complete_state'] = read(trainer.output_root/'RESUME_CHECK.json')
        receipt['startup'] = discarded_startup(trainer, features, output/'startup')
        trainer.install_head(enabled=case != 'DISABLED')
        extension = getattr(trainer.bundle.model, 'survival_head_extension', None)
        if extension is not None:
            save_torch(output/'HEAD_INITIAL.pt', extension.state_dict())
            receipt['head_initial_identity'] = identity(output/'HEAD_INITIAL.pt')
            receipt['head_initial_check'] = validate_head_state(extension.state_dict(),
                arm=arm, expected_steps=0, native_recipe=trainer.backend.optimizer.state_dict()['recipe'])
            if not bool(extension.head.weight.ne(0).any()) or not bool(extension.head.bias.eq(0).all()):
                raise ValueError('Head initialization requires nonzero weights and zero bias')
            receipt['head_initial_check'].update(nonzero_weight=True, zero_bias=True)
            ledger.wrap(extension.optimizer.optimizer, 'step', 'head_optimizer_steps')
            augment = extension.augment_response
            parameters = [p for p in trainer.bundle.draft_model.parameters() if p.requires_grad]

            def checked_augment(native_loss, terms):
                # Same clean graph; this extra backward is diagnostic only.
                # autograd.grad never enters ordinary .grad or either optimizer.
                auxiliary = response_bce(terms)
                row = auxiliary_gradient_check(auxiliary, parameters, ledger)
                receipt['auxiliary_gradient_checks'].append(row)
                write_json(output/'AUXILIARY_CHECKS.json', receipt['auxiliary_gradient_checks'])
                if not all(row[name] for name in ('finite', 'nonzero', 'native_gradient_buffer_unchanged')):
                    raise ValueError('Auxiliary head must reach backbone with finite nonzero gradient')
                return augment(native_loss, terms)
            extension.augment_response = checked_augment
        trainer._active_event = 'qualification_'+case
        (trainer.events_root/trainer._active_event).mkdir()
        trainer.bundle.model.begin_event()
        ledger.wrap(trainer.core.strategy, 'forward_loss', 'response_forwards')
        ledger.wrap(trainer.backend, 'backward', 'response_backwards')
        ordinary_step = trainer.backend.optimizer.optimizer.step

        def step(*args, **kwargs):
            for name, p in trainer.bundle.draft_model.named_parameters():
                gradients[name] = None if p.grad is None else p.grad.detach().cpu().clone()
            return ledger.call('optimizer_steps', ordinary_step, *args, **kwargs)
        trainer.backend.optimizer.optimizer.step = step
        torch.cuda.reset_peak_memory_stats()
        for feature in features:
            raw = torch.load(feature, map_location='cpu', weights_only=True)
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(trainer._batch(raw), StepContext(
                global_step=trainer.global_step, total_steps=trainer.horizon, collect_detailed_metrics=True))
            result.materialize_metrics()
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
        saved = endpoint(trainer, gradients)
        save_torch(output/'ENDPOINT.pt', saved)  # Preserve before any endpoint gate.
        compact = {key:saved[key] for key in ('rng', 'anchors', 'global_step',
                    'presentations', 'micro', 'remainder', 'response_losses')}
        save_torch(output/'COMPACT_ENDPOINT.pt', compact)
        receipt.update(endpoint_identity=identity(output/'ENDPOINT.pt'),
            compact_endpoint_identity=identity(output/'COMPACT_ENDPOINT.pt'),
            observations=deepcopy(trainer.response_records), windows=deepcopy(trainer.window_records),
            physical_work=deepcopy(trainer.work))
        receipt['finite'] = finite_endpoint(saved)
        require_complete_work(read_work(ledger.path, ledger.limits), ledger.limits)
        if (trainer.global_step != 513 or trainer.presentations != 1026
                or trainer.core.accumulation_remainder or len(trainer.response_records) != 2):
            raise ValueError('Disposable window accounting differs')
        if case != 'KPLUS':
            reference = torch.load(root/'qualification/KPLUS/ENDPOINT.pt', map_location='cpu',
                                   weights_only=False, mmap=True)
            fields = tuple(saved) if case == 'DISABLED' else (
                'rng', 'anchors', 'global_step', 'presentations', 'micro', 'remainder')
            receipt['comparison'] = persist_comparison(output/'EXACT.json',
                {k:saved[k] for k in fields}, {k:reference[k] for k in fields})
            del reference
        if extension is not None:
            save_torch(output/'HEAD_FINAL.pt', extension.state_dict())
            receipt['head_final_identity'] = identity(output/'HEAD_FINAL.pt')
            receipt['head_final_check'] = validate_head_state(extension.state_dict(),
                arm=arm, expected_steps=1, native_recipe=trainer.backend.optimizer.state_dict()['recipe'])
            receipt['head_steps'] = extension.steps
            if extension.steps != 1:
                raise ValueError('Separate head must take one step per native window')
            if case == 'SURVIVAL':
                current = torch.load(output/'HEAD_INITIAL.pt', weights_only=False, map_location='cpu')
                local = torch.load(root/'qualification/LOCAL/HEAD_INITIAL.pt', weights_only=False, map_location='cpu')
                receipt['head_initialization_exact'] = persist_comparison(
                    output/'HEAD_INITIAL_EXACT.json', current['head'], local['head'])
        if case in ARMS:
            payload = live_state(trainer)
            temporary = output/'EXPORT_INPUT.pt'
            save_torch(temporary, payload)
            receipt['export'] = str(output/'export')
            receipt['export_check'] = export_disposable_state(trainer, temporary, output/'export', payload)
            temporary.unlink()
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        receipt.update(wall_s=time.perf_counter()-started, work=read_work(ledger.path, ledger.limits),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        cleanup(lambda:write_json(output/'RESULT.json', receipt), failure, receipt, 'qualification receipt')
        if trainer is not None:
            cleanup(trainer.close, failure, receipt, 'disposable trainer close')
    return receipt


def serving_case(root, case):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
    from experiments.o_half_matched_lr.archive import validate_record
    root = Path(root); plan = read(root/'sealed/PLAN.json')
    output = root/'qualification'/case
    checked = read(output/'RESULT.json')
    if case not in ARMS or checked['status'] != 'completed':
        raise ValueError('Serve only completed head-free disposable exports')
    for relative, expected in checked['export_check']['files'].items():
        if identity(Path(checked['export'])/relative) != expected:
            raise ValueError('Disposable export changed before native serving')
    selected, input_evidence = verified_inputs(root)
    ledger = WorkLedger(output/'SERVING_WORK.jsonl', dict(serving_generations=2))
    backend = None; failure = None; receipt = dict(status='running', checks=[], inputs=input_evidence,
        binding=plan['binding'], execution_commit=plan['execution_source']['commit'])
    try:
        timing_barrier()
        backend = SGLangDFlashBackend(plan['target'], checked['export'],
            worker_python=plan['worker_python'], output_root=output/'native',
            serving_gpu=str(plan['serving_gpu']), context_limit=4096, block_size=16)
        publication = backend.publish(checked['export'])
        weights = backend.check_weights(checked['export'])
        receipt['publication'] = publication
        receipt['weights'] = weights
        errors = publication_errors(publication, checked['export']) + weight_check_errors(weights, checked['export'])
        write_json(output/'SERVING_PUBLICATION.json', dict(publication=publication, weights=weights, errors=errors))
        if errors:
            raise ValueError('Disposable serving publication/weights failed: '+repr(errors))
        for index, row in enumerate(selected):
            canonical = read(row['canonical']['path'])
            path = output/f'SERVING_{index}.json'
            record = ledger.call('serving_generations', backend.generate, canonical['prompt_ids'],
                max_new_tokens=2048, request_id=canonical['record_id'], session_id='survival_qualification',
                request_index=index, purpose='evaluation', save_to=path)
            request = dict(canonical, panel_ordinal=index)
            errors = validate_record(record, request, checked['export'], canonical,
                label=canonical['record_id'], session='survival_qualification')
            native_path = Path(record['metadata']['native_trace_path'])
            if read(native_path) != record['metadata']['capture']:
                errors.append('native_sidecar_disagreement')
            receipt['checks'].append(dict(index=index, exact_output_and_termination=not errors,
                errors=errors, raw_path=str(path), raw_identity=identity(path),
                native_path=str(native_path), native_identity=identity(native_path)))
            write_json(output/'SERVING_CHECK.json', receipt)
            if errors:
                raise ValueError('Head-free disposable export changed canonical output')
        require_complete_work(read_work(ledger.path, ledger.limits), ledger.limits)
        receipt.update(status='passed', loading_s=backend.loading_s)
    except BaseException as error:
        failure = error; receipt.update(status='failed', error=repr(error)); raise
    finally:
        if backend is not None:
            cleanup(backend.close, failure, receipt, 'serving close')
        receipt['work'] = read_work(ledger.path, ledger.limits)
        cleanup(lambda:write_json(output/'SERVING_CHECK.json', receipt), failure, receipt, 'serving receipt')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('--case', choices=QUALIFICATION_CASES, required=True)
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    (serving_case if args.serve else run_case)(args.root, args.case)
