"""Exactly Q1..Q4: one temporary native Adam step per case, no serving work."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work, startup_metadata, fresh_worker_environment
from experiments.ok_coding import training as original
from experiments.ok_continuation import resume
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer
from .training import PrefixProtectionTrainer, live_state, validate_snapshot, verify_live
from .training_worker import original_contract

CASES = (
    ('Q1', dict(native_only=True), 0),
    ('Q2', dict(coefficient=0.), 2),
    ('Q3', dict(), 2),
    ('Q4', dict(forced_active=True), 2),
)
LIMITS = dict(forwards=8, backwards=8, reference_forwards=6, optimizer_steps=4)


def normalized_gradients(trainer):
    """Observe real accumulated/normalized gradients before the one native clip."""
    import torch
    values = {}
    for name, parameter in trainer.bundle.draft_model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.grad
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite normalized preclip gradient: ' + name)
        values[name] = None if value is None else value.detach().cpu().clone()
    if not values or all(value is None for value in values.values()):
        raise ValueError('Preclip check observed cleared/all-None gradients')
    return values


def tensor_comparison(left, right):
    """Exact native comparison plus bounded useful magnitude diagnostics."""
    import torch
    names = sorted(set(left) | set(right))
    changed = []
    coordinates = 0
    maximum = 0.
    rows = []
    for name in names:
        a, b = left.get(name), right.get(name)
        same = name in left and name in right and not resume.differences(a, b)
        if same:
            continue
        changed.append(name)
        row = dict(name=name, left_none=a is None, right_none=b is None)
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            row.update(left_dtype=str(a.dtype), right_dtype=str(b.dtype),
                       left_shape=list(a.shape), right_shape=list(b.shape))
            if a.shape == b.shape:
                count, error = 0, 0.
                aa, bb = a.detach().reshape(-1), b.detach().reshape(-1)
                for start in range(0, aa.numel(), 1 << 20):
                    ac, bc = aa[start:start+(1 << 20)].float(), bb[start:start+(1 << 20)].float()
                    count += int((ac != bc).sum())
                    error = max(error, float((ac-bc).abs().max()))
                row.update(changed_coordinates=count, max_abs_error=error)
                coordinates += count
                maximum = max(maximum, error)
        rows.append(row)
    return dict(exact=not changed, changed_parameters=len(changed), changed_coordinates=coordinates,
                max_abs_error=maximum, parameters=rows)


def compare_cases(baseline, candidate):
    """Q2 exactness uses decoded values, not checkpoint byte identity."""
    left, right = baseline['state'], candidate['state']
    state_keys = ('rng', 'global_step', 'presentations', 'events', 'reduction', 'accumulation_steps', 'horizon')
    state_differences = resume.differences({key: left[key] for key in state_keys}, {key: right[key] for key in state_keys})
    losses = [row['loss'] for row in baseline['responses']]
    candidate_losses = [row['loss'] for row in candidate['responses']]
    return dict(weights=tensor_comparison(left['draft_state_dict'], right['draft_state_dict']),
        gradients=tensor_comparison(baseline['gradients'], candidate['gradients']),
        optimizer_exact=not resume.differences(left['backend'], right['backend']),
        state_differences=state_differences,
        losses_exact=losses == candidate_losses,
        anchors_exact=baseline['anchors'] == candidate['anchors'],
        response_metrics_exact=all(
            a['loss'] == b['loss'] and a['ratio_metrics'] == b['ratio_metrics']
            and a['sum_metrics'] == b['sum_metrics']
            for a, b in zip(baseline['responses'], candidate['responses'])))


def require_neutral_equal(comparison):
    exact = (comparison['weights']['exact'] and comparison['gradients']['exact']
        and comparison['optimizer_exact'] and not comparison['state_differences']
        and comparison['losses_exact'] and comparison['anchors_exact']
        and comparison['response_metrics_exact'])
    if not exact:
        raise ValueError('Q2 neutral wrapper differs from native K; diagnostics retained')


def require_forced_active(baseline, candidate, comparison, receipt_path=None):
    def moments(state):
        values = state['backend']['optimizer']['optimizer_state_dict']['state']
        return {f'{parameter_id}.{name}': value[name]
                for parameter_id, value in values.items() for name in ('exp_avg', 'exp_avg_sq')}
    moment_comparison = tensor_comparison(moments(baseline['state']), moments(candidate['state']))
    result = dict(gradients=comparison['gradients'], moments=moment_comparison,
                  updated_bf16_parameters=comparison['weights'])
    if receipt_path is not None:
        write_json(receipt_path, result)
    if comparison['gradients']['exact'] or moment_comparison['exact']:
        raise ValueError('Q4 must change both accumulated preclip gradients and native Adam moments')
    return result


def frozen_snapshot(trainer):
    """Capture frozen tensors/modes only for the bounded qualification checks."""
    modules = dict(head=trainer.bundle.model.lm_head, embedding=trainer.bundle.model.embed_tokens)
    reference = getattr(trainer.bundle.model, 'released_reference', None)
    if reference is not None:
        modules['reference'] = reference.model.draft_model
    return {name: dict(training=module.training,
        values={key: value.detach().cpu().clone() for key, value in module.state_dict().items()})
        for name, module in modules.items()}


def _run_case(root, plan, name, options, reference_forwards, ledger):
    import torch
    from specforge.training.strategies.base import StepContext
    output = root/'qualification_001'/name
    output.mkdir(exist_ok=True)
    qualification = dict(case=name, source='original_K17_event4',
        completed_events=4, temporary_optimizer_steps=1, **options)
    source = torch.load(plan['qualification']['state'], map_location='cpu', weights_only=False)
    original.validate_snapshot(source, arm='K', seed=17, campaign_contract=original_contract(plan, 17),
        config_text=Path(plan['qualification']['original_config']).read_text(), completed_event=4)
    original._export_equal(source, plan['qualification']['export'])
    config = plan['configs']['17']['KP']
    trainer = None
    receipt = dict(case=name, status='running', startup=startup_metadata(), source_step=512, source_presentations=1024,
                   warmup_forwards=0, warmup_backwards=0, additional_optimizer_steps=0)
    started = time.perf_counter()
    try:
        trainer = PrefixProtectionTrainer(config, output/'runtime', arm='KP', seed=17,
            binding=plan['training_binding'], original_contract=original_contract(plan, 17),
            source_state=source, completed_event=4, qualification=qualification, **options)
        del source
        receipt['startup']['device_name'] = torch.cuda.get_device_name(0)
        receipt['startup']['device_uuid'] = str(torch.cuda.get_device_properties(0).uuid)
        initial = live_state(trainer)
        verify_live(trainer, initial, output/'INITIAL.json')
        frozen_before = frozen_snapshot(trainer)
        receipt['initialization_s'] = time.perf_counter()-started
        receipt['reference_setup_s'] = trainer.reference_setup_s
        receipt['loading_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        receipt['loading_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        paths = plan['qualification']['records']
        # This is a two-presentation diagnostic, not train_event or event5.
        features = SpecForgeEventTrainer._prepare_event(trainer, name, paths, 1)
        write_json(output/'FEATURES.json', features)
        trainer.bundle.model.begin_event()
        trainer.response_observer_path = output/'RESPONSE_OBSERVATIONS.jsonl'
        ledger.wrap(trainer.core.strategy, 'forward_loss', 'forwards')
        ledger.wrap(trainer.backend, 'backward', 'backwards')
        ledger.wrap(trainer.backend.optimizer.optimizer, 'step', 'optimizer_steps')
        if reference_forwards:
            ledger.wrap(trainer.bundle.model.released_reference, 'hidden', 'reference_forwards')
        ordinary_step = trainer.backend.optimizer.step
        captured = {}
        def capture_before_clip():
            observation_started = time.perf_counter()
            captured.update(normalized_gradients(trainer))
            save_torch(output/'NORMALIZED_PRECLIP_GRADIENTS.pt', captured)
            receipt['preclip_observation_and_persistence_s'] = time.perf_counter()-observation_started
            torch.cuda.synchronize()
            optimizer_started = time.perf_counter()
            result = ordinary_step()
            torch.cuda.synchronize()
            receipt['native_clip_optimizer_clear_s'] = time.perf_counter()-optimizer_started
            return result
        trainer.backend.optimizer.step = capture_before_clip
        anchors = []
        response_started = time.perf_counter()
        for feature in features:
            raw = torch.load(feature['path'], map_location='cpu', weights_only=True)
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(trainer._batch(raw), StepContext(
                global_step=trainer.global_step, total_steps=trainer.horizon, collect_detailed_metrics=True))
            trainer.presentations += 1
            if result.optimizer_stepped:
                trainer.global_step += 1
            # Materialization catches nonfinite diagnostics before case completion.
            metrics = result.materialize_metrics()
            if any(not __import__('math').isfinite(float(v)) for v in metrics.values()):
                raise ValueError('Nonfinite qualification metrics')
            anchors.append(deepcopy(trainer._anchor_observations))
            write_json(output/'RESPONSES.json', dict(responses=trainer.response_records, anchors=anchors))
        receipt['two_response_window_s'] = time.perf_counter()-response_started
        receipt['objective_work'] = trainer.bundle.model.end_event()
        state = live_state(trainer)
        validate_snapshot(state, arm='KP', seed=17, binding=plan['training_binding'],
            original_contract=original_contract(plan, 17), config_text=Path(config).read_text(),
            completed_event=4, qualification=qualification, qualification_steps=1)
        save_torch(output/'POST_STATE.pt', state)
        decoded = torch.load(output/'POST_STATE.pt', map_location='cpu', weights_only=False)
        if resume.differences(state, decoded):
            raise ValueError('Qualification decoded serialization differs')
        # Restore exactly this truthful disposable state, without creating a
        # completed event5, new forward, optimizer step or RNG draw.
        trainer._load_decoded(decoded)
        verify_live(trainer, decoded, output/'POST_RESTORE.json')
        frozen_after = frozen_snapshot(trainer)
        frozen_changes = resume.differences(frozen_before, frozen_after)
        train_ids = {id(p) for p in trainer.backend.optimizer.model_params}
        for module in (trainer.bundle.model.lm_head, trainer.bundle.model.embed_tokens):
            if train_ids & {id(p) for p in module.parameters()}:
                frozen_changes.append('target_head_or_embedding_in_optimizer')
        if reference_forwards:
            reference = trainer.bundle.model.released_reference.model.draft_model
            if train_ids & {id(p) for p in reference.parameters()} or reference.training:
                frozen_changes.append('reference_in_optimizer_or_training_mode')
        receipt['frozen_changes'] = frozen_changes
        if frozen_changes:
            raise ValueError('Frozen reference/head/embedding changed')
        receipt.update(status='completed', global_step=trainer.global_step,
            presentations=trainer.presentations, completed_events=len(trainer.events),
            serialized_restored_exact=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), device_name=torch.cuda.get_device_name(0),
            device_uuid=str(torch.cuda.get_device_properties(0).uuid))
        return dict(state=decoded, gradients=captured, responses=deepcopy(trainer.response_records), anchors=anchors)
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        if trainer is not None:
            receipt['trainer_status'] = trainer.status()
            trainer.close()
        receipt['wall_s'] = time.perf_counter()-started
        receipt['work'] = read_work(output/'WORK.jsonl', ledger.limits)
        write_json(output/'CASE.json', receipt)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def case_limits(reference_forwards):
    return dict(forwards=2, backwards=2, reference_forwards=reference_forwards, optimizer_steps=1)


def qualification_work(root):
    total = {phase: dict.fromkeys(LIMITS, 0) for phase in ('attempts', 'completed', 'unknown')}
    for name, _options, references in CASES:
        work = read_work(Path(root)/'qualification_001'/name/'WORK.jsonl', case_limits(references))
        for phase in total:
            for kind, count in work[phase].items():
                total[phase][kind] += count
    return total


def case_worker(root, name):
    """A cold case process; no compiler/model/optimizer state crosses cases."""
    root = Path(root)
    plan = json.loads((root/'sealed/PLAN.json').read_text())
    definition = next((row for row in CASES if row[0] == name), None)
    if definition is None:
        raise ValueError('Only Q1..Q4 are declared')
    _name, options, references = definition
    directory = root/'qualification_001'/name
    ledger = WorkLedger(directory/'WORK.jsonl', case_limits(references))
    _run_case(root, plan, name, options, references, ledger)
    require_complete_work(read_work(directory/'WORK.jsonl', ledger.limits), ledger.limits)


def load_case(directory):
    import torch
    rows = json.loads((directory/'RESPONSES.json').read_text())
    return dict(state=torch.load(directory/'POST_STATE.pt', map_location='cpu', weights_only=False),
        gradients=torch.load(directory/'NORMALIZED_PRECLIP_GRADIENTS.pt', map_location='cpu', weights_only=False),
        responses=rows['responses'], anchors=rows['anchors'])


def run(root):
    """CPU coordinator: four fresh sequential processes, then decoded comparisons."""
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Qualification coordinator must remain CPU-only')
    root = Path(root)
    output = root/'qualification_001'
    output.mkdir(exist_ok=False)
    receipt = dict(status='running', scope='four_cold_disposable_one_step_cases', budget=LIMITS,
                   automatic_retry=False, no_generations=True,
                   compiler_policy='Each case gets a fresh process and six empty cache directories')
    started = time.perf_counter()
    baseline = None
    device_uuid = None
    try:
        write_json(output/'QUALIFICATION.json', receipt)
        for name, _options, _references in CASES:
            directory = output/name
            directory.mkdir()
            environment = fresh_worker_environment(directory/'caches')
            command = [sys.executable, '-u', '-m', 'experiments.k_prefix_protection_v1.qualification',
                       str(root), '--case', name]
            launched = time.perf_counter()
            with (directory/'WORKER.log').open('w') as stream:
                process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            write_json(directory/'EXEC.json', dict(command=command, returncode=process.returncode,
                wall_s=time.perf_counter()-launched, automatic_retry=False))
            if process.returncode != 0:
                raise RuntimeError(name + ' failed; no qualification retry is authorized')
            case_receipt = json.loads((directory/'CASE.json').read_text())
            if case_receipt['status'] != 'completed':
                raise ValueError(name + ' process exited without complete case evidence')
            observed_device = case_receipt['startup']['device_uuid']
            if device_uuid is None:
                device_uuid = observed_device
            elif observed_device != device_uuid:
                raise ValueError('Qualification cases used different physical updater GPUs')
            candidate = load_case(directory)
            if name == 'Q1':
                baseline = candidate
            else:
                comparison = compare_cases(baseline, candidate)
                write_json(directory/'VERSUS_Q1.json', comparison)
                if not comparison['anchors_exact'] or comparison['state_differences']:
                    raise ValueError(name + ' stochastic state/counters/history/anchors differ')
                if name == 'Q2':
                    require_neutral_equal(comparison)
                elif name == 'Q4':
                    require_forced_active(baseline, candidate, comparison, directory/'FORCED_ACTIVE.json')
                del candidate
                gc.collect()
            if torch.cuda.is_initialized():
                raise RuntimeError('CPU qualification coordinator unexpectedly initialized CUDA')
        require_complete_work(qualification_work(root), LIMITS)
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['work'] = qualification_work(root)
        receipt['wall_s'] = time.perf_counter()-started
        write_json(output/'QUALIFICATION.json', receipt)
        write_json(output/'RESULT.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--case', choices=[row[0] for row in CASES])
    args = parser.parse_args()
    if args.case:
        case_worker(args.root, args.case)
    else:
        run(args.root)


if __name__ == '__main__':
    main()
