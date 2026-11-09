"""Eight disposable native updates across the two retained KPLUS4 states.

The ordinary reference, traced neutral hook, FSG and SCALAR each use the same
two newly collected responses. All temporary states are discarded; the main
events-5–6 pilot is deliberately not invoked by this module.
"""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import time
import traceback

import torch

from experiments.ok_continuation.resume import differences
from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime, persist_comparison, preserve_failure
from experiments.prefix_ranking.qualify import cpu_copy, override
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.functional_geometry import install_functional_geometry
from .training import FunctionalGeometryTrainer
from .work import QualificationWork, LIMITS


CASES = (('KPLUS', None), ('FSG_ZERO', 0.), ('FSG', 1.), ('SCALAR', 1.))
CACHE_FIELDS = ('_cached_block_mask', '_cached_seq_len', '_cached_bsz',
                'max_seq_len_cached', 'original_max_seq_len')


def cleanup(action, failure, receipt, label):
    """Preserve a scientific failure when its cleanup/reporting also fails."""
    if failure is None:
        action()
    else:
        preserve_failure(failure, action, receipt, label)


def runtime_state(model):
    """Snapshot bounded nonparameter runtime state for exact isolation checks."""
    return dict(modes={name: module.training for name, module in model.named_modules()},
        buffers={name: value.detach().cpu().clone() for name, value in model.named_buffers()},
        buffer_objects={name: dict(module._buffers) for name, module in model.named_modules()},
        nonpersistent={name: set(module._non_persistent_buffers_set)
                       for name, module in model.named_modules()},
        caches={name: {key: getattr(module, key) for key in CACHE_FIELDS if hasattr(module, key)}
                for name, module in model.named_modules()})


def restore_runtime(model, state):
    modules = dict(model.named_modules())
    device = next(model.parameters()).device
    with torch.no_grad():
        for name, original in state['buffer_objects'].items():
            module = modules[name]
            module._buffers.clear()
            module._buffers.update({key: None if value is None else value.to(device)
                                    for key, value in original.items()})
            module._non_persistent_buffers_set = set(state['nonpersistent'][name])
        for name, value in state['buffers'].items():
            owner, _, key = name.rpartition('.')
            module = modules[owner]
            module._buffers[key].copy_(value.to(device))
        for name, mode in state['modes'].items():
            modules[name].training = mode
        for name, fields in state['caches'].items():
            for key in CACHE_FIELDS:
                if key in fields:
                    setattr(modules[name], key, fields[key])
                elif hasattr(modules[name], key):
                    delattr(modules[name], key)


def scientific_state(trainer):
    return cpu_copy(dict(weights=trainer.bundle.draft_model.state_dict(),
        optimizer=trainer.backend.optimizer.state_dict(), rng=capture_rng(),
        global_step=trainer.global_step, presentations=trainer.presentations,
        micro=trainer.core._micro, accumulation=trainer.core.accumulation_remainder,
        gradients={name: parameter.grad for name, parameter in trainer.bundle.draft_model.named_parameters()},
        runtime=runtime_state(trainer.bundle.model)))


def restore_origin(trainer, baseline_runtime, root):
    # A deliberate pre-Adam stop leaves TrainerCore's metric totals pending.
    # Clear this disposable window before _restore validates a clean boundary.
    trainer.backend.optimizer.zero_grad()
    trainer.core._ratio_totals.clear()
    trainer.core._sum_totals.clear()
    trainer._anchor_observations.clear()
    trainer.response_records.clear()
    trainer.window_records.clear()
    trainer.work.clear()
    trainer._presentation_schedule = None
    trainer._restore(trainer.origin_path)
    restore_runtime(trainer.bundle.model, baseline_runtime)
    mismatch = differences(capture_rng(), trainer.origin['rng'], 'rng')
    if mismatch or trainer.global_step != 512 or trainer.presentations != 1024:
        raise ValueError('Qualification did not restore the exact original continuation boundary')
    write_json(root / 'RESTORED.json', dict(status='passed', global_step=512,
        presentations=1024, rng_exact=True, ordinary_sampling=True, optimizer_reset=False))


def set_observation_directory(trainer, label):
    trainer._active_event = label
    (trainer.events_root / label).mkdir(exist_ok=False)
    trainer.response_records.clear()
    trainer.window_records.clear()
    trainer.work.clear()


def physical_work(trainer):
    """Copy physical head/layer/chunk counts before a restoration clears them."""
    return dict(counts=dict(trainer.work), response_observations=len(trainer.response_records),
                optimizer_window_observations=len(trainer.window_records))


def warmup(trainer, features, work, baseline_runtime, root):
    """Two qualified startup backwards, no Adam; restore original RNG afterward."""
    from specforge.training.strategies.base import StepContext
    set_observation_directory(trainer, 'disposable_startup')
    before = scientific_state(trainer)
    trainer.bundle.model.begin_event()
    failure = None
    receipt = dict(status='running')
    try:
        for index, path in enumerate(features):
            raw = torch.load(path, map_location='cpu', weights_only=True)
            trainer._anchor_observations.clear()
            with work.operation('additional_forward', purpose='startup'):
                result = trainer.strategy.forward_loss(trainer._batch(raw), StepContext(
                    global_step=512, total_steps=trainer.horizon, collect_detailed_metrics=True))
            with work.operation('additional_backward', purpose='startup'):
                trainer.backend.backward(result.loss / 2, is_boundary=index == 1)
            del result, raw
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['physical_work'] = physical_work(trainer)
        cleanup(lambda: write_json(root / 'STARTUP_WORK.json', receipt), failure, receipt,
                'startup work receipt')
    trainer.backend.optimizer.zero_grad()
    restore_rng(before['rng'])
    restore_runtime(trainer.bundle.model, baseline_runtime)
    # Startup may warm code and cost counters, never weights or Adam tensors.
    persist_comparison(root / 'STARTUP_EXACT_RESTORATION.json', scientific_state(trainer), before)
    restore_origin(trainer, baseline_runtime, root)
    del before


def export_disposable_state(trainer, checkpoint_path, export_path, payload):
    """Export the decoded qualification payload, never a production pending file."""
    from experiments.ok_coding.training import _export_equal
    from specforge.export.to_hf import export_to_hf
    started = time.perf_counter()
    config_path = Path(trainer.cfg.model.draft_model_config)
    if config_path.is_dir():
        config_path /= 'config.json'
    export_to_hf(str(checkpoint_path), str(config_path), str(export_path),
        embedding_source=trainer.cfg.model.target_model_path,
        embedding_key=trainer.cfg.model.embedding_key)
    exported_at = time.perf_counter()
    _export_equal(payload, export_path)
    checked_at = time.perf_counter()
    receipt = dict(status='passed', exact_tensor_values_and_membership=True,
        export_s=exported_at-started, verification_s=checked_at-exported_at,
        files={str(path.relative_to(export_path)): identity(path)
               for path in sorted(Path(export_path).rglob('*')) if path.is_file()})
    write_json(Path(export_path).parent / 'EXPORT_EXACT.json', receipt)
    return receipt


def run_case(trainer, features, label, strength, work, baseline_runtime, root, reference,
             *, oracle_runner=None):
    from specforge.training.strategies.base import StepContext
    root.mkdir()
    work.context = dict(dataset=trainer.dataset, case=label)
    restore_origin(trainer, baseline_runtime, root)
    set_observation_directory(trainer, 'disposable_' + label.lower())
    trainer.bundle.model.begin_event()
    capture = {}
    hook = None
    started = time.perf_counter()

    def observe(phase, active_hook=None):
        if phase == 'pre':
            capture['gradients'] = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                    for name, p in trainer.bundle.draft_model.named_parameters()}
            if reference is not None:
                persist_comparison(root / 'NATIVE_GRADIENTS.json', capture['gradients'], reference['gradients'])
                del capture['gradients']
        elif phase == 'native':
            state = scientific_state(trainer)
            if reference is None:
                capture['native'] = state
            else:
                persist_comparison(root / 'NATIVE_ENDPOINT_AND_ADAM.json', state, reference['native'])
                del state
        elif phase == 'committed' and label == 'FSG':
            from .native_oracles import run_oracles
            capture['oracles'] = (oracle_runner or run_oracles)(active_hook, root / 'oracles')

    ordinary_step = trainer.backend.optimizer.step

    def reference_step():
        observe('pre')
        with work.operation('adam'):
            result = ordinary_step()
        observe('native')
        return result

    if strength is not None:
        hook = install_functional_geometry(trainer, arm='SCALAR' if label == 'SCALAR' else 'FSG',
            strength=strength, observer=observe, output_dir=root / 'geometry', work_callback=work.callback)
    backward = trainer.backend.backward
    forward = trainer.author_strategy.forward_loss

    def counted_forward(*args, **kwargs):
        with work.operation('ordinary_forward', presentation=trainer.presentations-1024):
            return forward(*args, **kwargs)

    def counted_backward(*args, **kwargs):
        with work.operation('ordinary_backward'):
            return backward(*args, **kwargs)

    rows = []
    receipt = dict(status='running', case=label, extra_adam_steps=0)
    failure = None
    try:
        with override(trainer.backend, 'backward', counted_backward), \
             override(trainer.author_strategy, 'forward_loss', counted_forward):
            if hook is None:
                trainer.backend.optimizer.step = reference_step
            for ordinal, path in enumerate(features):
                raw = torch.load(path, map_location='cpu', weights_only=True)
                trainer._anchor_observations.clear()
                result = trainer.core.train_step(trainer._batch(raw), StepContext(
                    global_step=trainer.global_step, total_steps=trainer.horizon,
                    collect_detailed_metrics=True))
                row = dict(record_id=raw['record_id'], sampled_anchors=deepcopy(trainer._anchor_observations),
                           metrics=result.materialize_metrics(), optimizer_stepped=result.optimizer_stepped)
                rows.append(row)
                trainer.presentations += 1
                trainer.global_step += int(result.optimizer_stepped)
                write_json(root / 'PRESENTATIONS.json', rows)
                del raw, result
        if trainer.global_step != 513 or trainer.presentations != 1026 or trainer.core.accumulation_remainder:
            raise ValueError('Disposable window counters differ')
        final = scientific_state(trainer)
        receipt['physical_work'] = physical_work(trainer)
        if reference is not None:
            persist_comparison(root / 'PRESENTATION_ANCHOR_LOSS_PARITY.json', rows, reference['rows'])
            if label == 'FSG_ZERO':
                persist_comparison(root / 'EXACT_NEUTRALITY.json', final, reference['final'])
            else:
                for key in ('optimizer', 'rng', 'global_step', 'presentations', 'micro',
                            'accumulation', 'gradients', 'runtime'):
                    persist_comparison(root / ('FINAL_' + key.upper() + '.json'), final[key], reference['final'][key])
        # Decode and restore the entire disposable scientific payload. This is
        # not a completed production event and is never reused by the pilot.
        with preserve_rng():
            decoded_path = root / 'disposable_state.pt'
            export_payload = dict(final, draft_state_dict=final['weights'])
            save_torch(decoded_path, export_payload)
            decoded = torch.load(decoded_path, map_location='cpu', weights_only=False)
            persist_comparison(root / 'DECODED_STATE.json', decoded, export_payload)
            restore_origin(trainer, baseline_runtime, root)
            trainer.bundle.draft_model.load_state_dict(decoded['weights'], strict=True)
            trainer.backend.optimizer.load_state_dict(decoded['optimizer'])
            trainer.global_step, trainer.presentations = decoded['global_step'], decoded['presentations']
            trainer.core._micro = decoded['micro']
            restore_runtime(trainer.bundle.model, decoded['runtime'])
            restore_rng(decoded['rng'])
            persist_comparison(root / 'DECODED_LIVE_RESTORE.json', scientific_state(trainer), final)
            export = root / 'export'
            receipt['export_verification'] = export_disposable_state(
                trainer, decoded_path, export, export_payload)
            decoded_identity = identity(decoded_path)
            decoded_path.unlink()
            write_json(root / 'DISPOSABLE_RETIREMENT.json', dict(path=str(decoded_path),
                identity=decoded_identity, reason='Decoded-state checks passed; never production recovery',
                lost_capability='Temporary tensor payload retired, not archived'))
        receipt.update(status='passed', export=str(export), optimizer_steps=1, presentations=2,
                       oracles=capture.get('oracles'))
        if reference is None:
            capture.update(final=final, rows=rows)
        else:
            del final
        return capture, receipt
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if 'physical_work' not in receipt:
            cleanup(lambda: receipt.update(physical_work=physical_work(trainer)),
                    failure, receipt, 'physical work capture')
        if hook is not None:
            cleanup(lambda: receipt.update(hook=hook.report()), failure, receipt, 'hook report')
            cleanup(hook.close, failure, receipt, 'hook close')
            if hasattr(trainer, 'functional_geometry'):
                delattr(trainer, 'functional_geometry')
        else:
            trainer.backend.optimizer.step = ordinary_step
        receipt.update(wall_s=time.perf_counter()-started, work=work.summary(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        cleanup(lambda: write_json(root / 'RESULT.json', receipt), failure, receipt, 'case result write')


def check_serving(plan, dataset, cases, prepared, work, root):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    backend = None
    records = []
    failure = None
    canonical = json.loads(Path(prepared['record_path']).read_text())
    try:
        backend = SGLangDFlashBackend(plan['target'], cases[0]['export'],
            worker_python=plan['worker_python'], output_root=root / 'native',
            serving_gpu='0', context_limit=4096, block_size=16)
        for ordinal, case in enumerate(cases):
            work.context = dict(dataset=dataset, case=case['case'])
            if ordinal:
                backend.publish(case['export'])
            with work.operation('serving_generation'):
                record = backend.generate(canonical['prompt_ids'], max_new_tokens=2048,
                    request_id='qualification_' + case['case'], session_id='fsg_qualification',
                    request_index=ordinal, purpose='evaluation',
                    save_to=root / (case['case'] + '.json'))
            if record['response_ids'] != canonical['response_ids'] or record['finish_reason'] != canonical['finish_reason']:
                raise ValueError('Native output/termination preservation failed: ' + case['case'])
            if case['case'] == 'FSG_ZERO':
                previous = records[0]
                if (record['metadata']['verification_calls'] != previous['metadata']['verification_calls']
                        or differences(record['metadata']['trace'], previous['metadata']['trace'])):
                    raise ValueError('Neutral export native proposals or verification trajectory differs')
            records.append(record)
        receipt = dict(status='passed', generations=len(records),
            loading_s=backend.loading_s, calls={r['record_id']:r['metadata']['verification_calls'] for r in records},
            exact_output_and_termination=True, exact_neutral_native_trajectory=True)
        write_json(root / 'SERVING_CHECK.json', receipt)
        return receipt
    except BaseException as error:
        failure = error
        raise
    finally:
        if backend is not None:
            cleanup(backend.close, failure, {}, 'serving backend close')


def authenticate_failed_ordinary(previous_root, root, plan, preparation):
    """Admit only the recorded post-Adam/pre-export failure, without resetting work."""
    previous_root, root = Path(previous_root), Path(root)
    if previous_root.resolve() == root.resolve():
        raise ValueError('Recovery requires a new directory; preserve the failed attempt')
    case = previous_root / 'qualification/gsm8k/KPLUS'
    original_plan = json.loads((previous_root / 'sealed/PLAN.json').read_text())
    original_preparation = json.loads((previous_root / 'qualification_feedback/PREPARATION.json').read_text())
    repair = plan['repair']
    if Path(repair['previous_root']).resolve() != previous_root.resolve():
        raise ValueError('Recovery root differs from the sealed repair receipt')
    for source in [repair['prior_work_ledger'], repair['shared_preparation'],
                   *repair['retained_reference'].values()]:
        if identity(source['path']) != source['identity']:
            raise ValueError('Sealed recovery artifact changed: ' + source['path'])
    result = json.loads((case / 'RESULT.json').read_text())
    decoded_check = json.loads((case / 'DECODED_STATE.json').read_text())
    startup_check = json.loads((previous_root / 'qualification/gsm8k/STARTUP_EXACT_RESTORATION.json').read_text())
    rows = json.loads((case / 'PRESENTATIONS.json').read_text())
    trace = result.get('traceback', '')
    if (result.get('status') != 'failed' or result.get('case') != 'KPLUS'
            or 'Native Adam counter differs:' not in result.get('error', '')
            or 'restore_origin(trainer, baseline_runtime, root)' not in trace
            or 'migrate_origin' not in trace or not decoded_check.get('exact_equal')
            or not startup_check.get('exact_equal')):
        raise ValueError('Retained attempt is not the authenticated post-Adam restoration failure')
    if (len(rows) != 2 or [row['optimizer_stepped'] for row in rows] != [False, True]
            or [row['record_id'] for row in rows] !=
               [row['request_id'] for row in original_preparation['datasets']['gsm8k']]):
        raise ValueError('Retained ordinary presentations do not match the prepared pair')
    for name in ('binding', 'geometry', 'serving', 'numerical_recipe', 'qualification_budget'):
        if differences(plan[name], original_plan[name]):
            raise ValueError('Recovery changes the scientific plan: ' + name)
    for dataset, source in original_plan['datasets'].items():
        for name in ('origin_state', 'origin_state_identity', 'origin_export', 'origin_weights_identity'):
            if differences(plan['datasets'][dataset][name], source[name]):
                raise ValueError('Recovery changes the original dataset checkpoint')
    if differences(preparation, original_preparation):
        raise ValueError('Recovery must reuse the complete original preparation unchanged')
    expected_counts = {kind + '_' + status: count
        for kind, count in (('adam', 1), ('ordinary_forward', 2), ('ordinary_backward', 2),
                            ('additional_forward', 2), ('additional_backward', 2))
        for status in ('attempted', 'completed')}
    previous_ledger = previous_root / 'QUALIFICATION_WORK.jsonl'
    if QualificationWork(previous_ledger).summary() != expected_counts:
        raise ValueError('Previous work is not exactly one ordinary step and two startup presentations')
    ledger_bytes = previous_ledger.read_bytes()
    destination = root / 'QUALIFICATION_WORK.jsonl'
    if destination.exists():
        raise ValueError('Recovery ledger already exists; no automatic recovery retry')
    with destination.open('xb') as stream:
        stream.write(ledger_bytes)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    if destination.read_bytes() != ledger_bytes:
        raise ValueError('Cumulative work prefix copy differs')
    payload_path = case / 'disposable_state.pt'
    observation_path = previous_root / 'qualification/gsm8k/trainer/events/disposable_kplus/RESPONSE_OBSERVATIONS.jsonl'
    files = (payload_path, case / 'PRESENTATIONS.json', case / 'RESULT.json',
             case / 'DECODED_STATE.json', observation_path, previous_ledger)
    receipt = dict(status='admitted', previous_root=str(previous_root),
        original_execution_commit=original_plan['execution_source']['commit'],
        evidence={str(path): identity(path) for path in files},
        ledger_prefix_bytes=len(ledger_bytes), ledger_prefix_identity=identity(previous_ledger),
        reused_adam_steps=1, remaining_adam_steps=7, recapture_forwards=2,
        recapture_backwards=2, recapture_adam_steps=0,
        original_physical_head_chunk_counts='unavailable; not reconstructed from replay counts')
    write_json(root / 'RECOVERY_ADMISSION.json', receipt)
    return dict(root=previous_root, case=case, payload=payload_path, rows=rows,
                observations=observation_path, receipt=receipt)


def retained_native_view(final):
    """The native observer runs before the caller advances its two host counters."""
    if (final['global_step'] != 513 or final['presentations'] != 1026
            or final['micro'] != 1026 or final['accumulation'] != 0
            or any(value is not None for value in final['gradients'].values())):
        raise ValueError('Retained ordinary endpoint is not a complete first qualification window')
    for moments in final['optimizer']['optimizer_state_dict']['state'].values():
        if float(moments['step']) != 513:
            raise ValueError('Retained ordinary Adam counters are not all 513')
    return dict(final, global_step=512, presentations=1025)


def recover_ordinary_reference(trainer, features, retained, work, baseline_runtime, root):
    """Recover raw gradients with two backwards, without repeating the saved Adam step."""
    from specforge.training.strategies.base import StepContext
    root.mkdir()
    restore_origin(trainer, baseline_runtime, root)
    set_observation_directory(trainer, 'retained_ordinary_gradient_replay')
    trainer.bundle.model.begin_event()
    before = scientific_state(trainer)
    payload = torch.load(retained['payload'], map_location='cpu', weights_only=False, mmap=True)
    persist_comparison(root / 'PAYLOAD_WEIGHT_ALIAS.json', payload['weights'], payload['draft_state_dict'])
    final = {key: value for key, value in payload.items() if key != 'draft_state_dict'}
    native = retained_native_view(final)
    expected_observations = [json.loads(line) for line in retained['observations'].read_text().splitlines()]
    work.context = dict(dataset=trainer.dataset, case='retained_ordinary_gradient_replay')
    gradients = {}
    replay_boundaries = 0
    forward, backward = trainer.author_strategy.forward_loss, trainer.backend.backward
    receipt = dict(status='running', optimizer_steps=0, response_forwards=2, backwards=2,
                   boundary_action='capture raw gradients, native clipping/norm, clear; no Adam')
    rows = []
    failure = None
    started = time.perf_counter()

    def count_forward(*args, **kwargs):
        with work.operation('additional_forward', purpose='retained_raw_gradient_replay'):
            return forward(*args, **kwargs)

    def count_backward(*args, **kwargs):
        with work.operation('additional_backward', purpose='retained_raw_gradient_replay'):
            return backward(*args, **kwargs)

    def capture_without_adam():
        nonlocal replay_boundaries
        replay_boundaries += 1
        gradients.update({name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                          for name, parameter in trainer.bundle.draft_model.named_parameters()})
        norm = torch.nn.utils.clip_grad_norm_(trainer.backend.optimizer.model_params,
            trainer.backend.optimizer.max_grad_norm, error_if_nonfinite=True)
        trainer.backend.optimizer.zero_grad()
        return norm.detach()

    try:
        with override(trainer.author_strategy, 'forward_loss', count_forward), \
             override(trainer.backend, 'backward', count_backward), \
             override(trainer.backend.optimizer, 'step', capture_without_adam):
            for path in features:
                raw = torch.load(path, map_location='cpu', weights_only=True)
                trainer._anchor_observations.clear()
                result = trainer.core.train_step(trainer._batch(raw), StepContext(
                    global_step=512, total_steps=trainer.horizon, collect_detailed_metrics=True))
                # optimizer_stepped is TrainerCore's boundary indicator here;
                # the intercepted boundary never calls Adam or advances its state.
                rows.append(dict(record_id=raw['record_id'],
                    sampled_anchors=deepcopy(trainer._anchor_observations),
                    metrics=result.materialize_metrics(), optimizer_stepped=result.optimizer_stepped))
                trainer.presentations += 1
                write_json(root / 'BOUNDARY_REPLAY_PRESENTATIONS.json', rows)
                del raw, result
        if replay_boundaries != 1 or trainer.global_step != 512:
            raise ValueError('Gradient replay did not retain the single intercepted boundary')
        receipt['physical_work'] = physical_work(trainer)
        persist_comparison(root / 'REPLAY_ANCHORS_LOSSES.json', rows, retained['rows'])
        actual_observations = [dict(row, event_id='disposable_kplus') for row in trainer.response_records]
        persist_comparison(root / 'REPLAY_RESPONSE_OBSERVATIONS.json', actual_observations, expected_observations)
        after = scientific_state(trainer)
        for name in ('weights', 'optimizer', 'gradients', 'global_step'):
            persist_comparison(root / ('REPLAY_UNCHANGED_' + name.upper() + '.json'), after[name], before[name])
        for name in ('rng', 'runtime'):
            persist_comparison(root / ('REPLAY_RETAINED_' + name.upper() + '.json'), after[name], final[name])
        del before, after
        # Restore and export the exact already-computed endpoint. No tensor
        # update is inferred from the newly recaptured gradients.
        with preserve_rng():
            restore_origin(trainer, baseline_runtime, root)
            trainer.bundle.draft_model.load_state_dict(final['weights'], strict=True)
            trainer.backend.optimizer.load_state_dict(deepcopy(final['optimizer']))
            trainer.global_step, trainer.presentations = final['global_step'], final['presentations']
            trainer.core._micro = final['micro']
            restore_runtime(trainer.bundle.model, final['runtime'])
            restore_rng(final['rng'])
            persist_comparison(root / 'RETAINED_LIVE_RESTORE.json', scientific_state(trainer), final)
            export = root / 'export'
            receipt['export_verification'] = export_disposable_state(trainer, retained['payload'], export, payload)
        receipt.update(status='passed', case='KPLUS', export=str(export),
            optimizer_steps=0, reused_optimizer_steps=1, presentations=2,
            original_case=str(retained['case']), exact_retained_endpoint=True)
        return dict(final=final, native=native, gradients=gradients, rows=retained['rows']), receipt
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if 'physical_work' not in receipt:
            cleanup(lambda: receipt.update(physical_work=physical_work(trainer)), failure, receipt,
                    'gradient replay physical work')
        receipt.update(wall_s=time.perf_counter()-started, work=work.summary())
        cleanup(lambda: write_json(root / 'RESULT.json', receipt), failure, receipt, 'gradient replay result')


def run(root, *, resume_from_failed_ordinary=None):
    root = Path(root)
    output = root / 'qualification'
    output.mkdir(exist_ok=False)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    preparation = json.loads((root / 'qualification_feedback/PREPARATION.json').read_text())
    if preparation['status'] != 'completed':
        raise ValueError('The sealed new-feedback preparation is incomplete')
    retained = (authenticate_failed_ordinary(resume_from_failed_ordinary, root, plan, preparation)
                if resume_from_failed_ordinary is not None else None)
    work = QualificationWork(root / 'QUALIFICATION_WORK.jsonl')
    summary = dict(status='running', execution_commit=plan['execution_source']['commit'],
                   limits=LIMITS, datasets={}, main_pilot_started=False, automatic_retry=False)
    if retained is not None:
        summary['recovery'] = retained['receipt']
    started = time.perf_counter()
    trainer = None
    failure = None
    try:
        pin_retained_graph_runtime(output)
        for dataset, item in plan['datasets'].items():
            directory = output / dataset
            directory.mkdir()
            torch.cuda.reset_peak_memory_stats()
            work.context = dict(dataset=dataset, case='startup')
            features = [row['feature_path'] for row in preparation['datasets'][dataset]]
            for row in preparation['datasets'][dataset]:
                if identity(row['feature_path']) != row['feature_identity']:
                    raise ValueError('Shared feedback packet changed')
            trainer = FunctionalGeometryTrainer(item['configs']['KPLUS'], directory / 'trainer',
                arm='KPLUS', dataset=dataset, binding=plan['binding'],
                original_config_text=Path(plan['original_config']).read_text(),
                origin_state=item['origin_state'], origin_export=item['origin_export'])
            baseline_runtime = runtime_state(trainer.bundle.model)
            warmup(trainer, features, work, baseline_runtime, directory)
            reference = None
            cases = []
            summary['datasets'][dataset] = dict(cases=cases)
            remaining_cases = CASES
            if retained is not None and dataset == 'gsm8k':
                reference, case = recover_ordinary_reference(trainer, features, retained, work,
                    baseline_runtime, directory / 'KPLUS')
                cases.append(case)
                remaining_cases = CASES[1:]
            for label, strength in remaining_cases:
                capture, case = run_case(trainer, features, label, strength, work,
                    baseline_runtime, directory / label, reference)
                if reference is None:
                    reference = capture
                cases.append(case)
                summary['work'] = work.summary()
                write_json(output / 'QUALIFICATION.json', summary)
            restore_origin(trainer, baseline_runtime, directory)
            trainer.close()
            trainer = None
            del reference, capture, baseline_runtime
            gc.collect()
            torch.cuda.empty_cache()
            summary['datasets'][dataset]['serving'] = check_serving(plan, dataset, cases,
                preparation['datasets'][dataset][0], work, directory / 'serving')
        summary['status'] = 'passed'
    except BaseException as error:
        failure = error
        summary.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            cleanup(trainer.close, failure, summary, 'qualification trainer close')
        summary.update(wall_s=time.perf_counter()-started, work=work.summary(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        cleanup(lambda: write_json(output / 'QUALIFICATION.json', summary), failure, summary,
                'qualification summary write')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--resume-from-failed-ordinary', type=Path,
                        help='Original failed qualification root; reuse its sole completed Adam endpoint')
    arguments = parser.parse_args()
    result = run(arguments.root, resume_from_failed_ordinary=arguments.resume_from_failed_ordinary)
    print(json.dumps({key: result[key] for key in ('status', 'wall_s', 'work')}))
