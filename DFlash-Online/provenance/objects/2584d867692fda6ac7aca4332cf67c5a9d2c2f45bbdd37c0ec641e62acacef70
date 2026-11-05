"""Reproduce the retained K first step, inspect one correction, and stop."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

import torch

from experiments.prefix_guard.qualify import (
    WorkLog, cpu_copy, named_gradients, observe_forwards, override,
    persist_comparison, pin_retained_graph_runtime, prepare_main,
    require_runtime_pin, run_startup, run_window,
)
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer


BUDGET = dict(optimizer_steps=1, training_presentations=2, maximum_direct_corrections=1,
              maximum_core_response_forwards=6, maximum_core_backward_equivalents=4,
              startup_response_evaluations=2, startup_backwards=2, startup_optimizer_steps=0,
              complete_response_generations=0, qualification_optimizer_steps=0)
PREFIX_METRICS = (
    'blocks', 'old_matching_prefix_tokens', 'new_matching_prefix_tokens',
    'net_matching_prefix_change', 'old_first_errors',
    'first_errors_repaired_with_prefix_intact', 'new_earlier_errors', 'old_prefix_tokens_lost',
)
RETAINED_TOTALS = dict(first_errors_repaired_with_prefix_intact=527,
                       new_earlier_errors=47, old_prefix_tokens_lost=153,
                       net_matching_prefix_change=1740)


def metric_totals(rows):
    return {key: sum(row[key] for row in rows) for key in PREFIX_METRICS}


def check_reproduction(post_k, reference, destination):
    """Fail before applying a correction if the original K case does not recur."""
    expected = [{key: row[key] for key in PREFIX_METRICS} for row in reference['responses']]
    actual = [{key: row[key] for key in PREFIX_METRICS} for row in post_k['responses']]
    expected_totals = metric_totals(expected)
    if any(expected_totals[key] != value for key, value in RETAINED_TOTALS.items()):
        raise ValueError('The sealed reference is not the original first-step case')
    persist_comparison(destination, actual, expected)
    return dict(exact_per_response_counts=True, actual=actual, expected=expected,
                totals=metric_totals(actual))


def check_presentations(records, ordinary, destination):
    keys = ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after')
    actual = [{key: row[key] for key in keys} for row in records]
    expected = [{key: row[key] for key in keys} for row in ordinary['records'][:2]]
    persist_comparison(destination, actual, expected)


def invariant_state(trainer):
    """State that neither the diagnostics nor the direct correction may change."""
    model = trainer.bundle.model
    return cpu_copy(dict(
        optimizer=trainer.backend.optimizer.state_dict(), rng=capture_rng(),
        global_step=trainer.global_step, presentations=trainer.presentations,
        accumulation_remainder=trainer.core.accumulation_remainder,
        training_modes={name: module.training for name, module in model.named_modules()},
        gradients=named_gradients(trainer),
        # Author mask caches contain runtime objects. Identity is the exact
        # restoration contract; the scalar keys are checked independently.
        mask_cache_identity=id(getattr(model, '_cached_block_mask', None)),
        mask_cache_seq_len=getattr(model, '_cached_seq_len', None),
        mask_cache_bsz=getattr(model, '_cached_bsz', None),
    ))


@contextmanager
def count_k_work(trainer, log):
    """Observe the unchanged K backward and actual Adam call, without extra work."""
    original_backward = trainer.backend.backward
    native = trainer.backend.optimizer.optimizer
    original_step = native.step

    def backward(*args, **kwargs):
        with log.work('k_backward'):
            return original_backward(*args, **kwargs)

    def step(*args, **kwargs):
        with log.work('adam_step'):
            return original_step(*args, **kwargs)

    with override(trainer.backend, 'backward', backward), override(native, 'step', step):
        yield


def validate_k_budget(log, trainer):
    actual = {key: log.counts[key] for key in ('k_backward_completed', 'adam_step_completed',
                                              'response_presentations')}
    expected = dict(k_backward_completed=2, adam_step_completed=1, response_presentations=2)
    if actual != expected or (trainer.global_step, trainer.presentations,
                               trainer.core.accumulation_remainder) != (1, 2, 0):
        raise AssertionError('The diagnostic must execute exactly one original K optimizer step')



def validate_core_budget(report, log, destination):
    work = report['work']
    forwards = sum(work[key] for key in ('original_response_forwards', 'post_k_response_forwards',
                                        'post_c_response_forwards'))
    backwards = log.counts['k_backward_completed'] + work['violation_backward_completed']
    receipt = dict(core_response_forwards=forwards, core_backward_equivalents=backwards,
                   adam_steps=log.counts['adam_step_completed'],
                   direct_corrections=work['direct_corrections'],
                   expected_response_forwards=6, maximum_backward_equivalents=4)
    write_json(destination, receipt)
    if (forwards != 6 or backwards > 4 or receipt['adam_steps'] != 1 or
            receipt['direct_corrections'] > 1 or
            work['original_response_forwards'] != 2 or work['post_k_response_forwards'] != 2 or
            work['post_c_response_forwards'] != 2):
        raise AssertionError('Core work exceeded or differs from the declared single-step diagnostic')
    return receipt


def diagnostic_json(value):
    """Keep NaN/Inf visible in failure records instead of losing the original error."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_json(item) for item in value]
    return value


def run(root):
    from online_feedback.prefix_correction import PrefixCorrection

    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    ordinary = json.loads(Path(plan['ordinary_receipt']).read_text())
    reference = json.loads(Path(plan['reference_probe']).read_text())
    if len(ordinary['features']) != 128 or len(ordinary['records']) != 256:
        raise ValueError('Use the complete retained event to reproduce its first two presentations')
    output = root / 'execution_001'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(status='running', phase='startup', plan=BUDGET, records=[],
                  complete_response_generations=0, checkpoint_exported=False, automatic_retry=False)
    trainer = correction = startup_log = log = None
    failure = None
    try:
        result['runtime_compatibility'] = pin_retained_graph_runtime(output)
        result['runtime_compatibility'].update(
            applied_before='disposable startup and fresh ordinary K construction',
            scope='startup, ordinary K, post-K violation and corrected-state diagnostic')
        write_json(output / 'RUNTIME_COMPATIBILITY.json', result['runtime_compatibility'])
        write_json(output / 'PROGRESS.json', result)
        pinned = torch.load(plan['pinned_initial'], map_location='cpu', weights_only=False, mmap=True)
        startup_log = WorkLog(output / 'startup')
        startup_log.trial = 'disposable_startup'
        require_runtime_pin()
        result['startup'] = run_startup(plan['config'], ordinary, pinned, startup_log, output / 'startup')
        if (result['startup']['status'] != 'passed' or result['startup']['optimizer_steps'] != 0 or
                result['startup']['response_evaluations'] != 2 or result['startup']['backwards'] != 2):
            raise AssertionError('Startup must perform two restored evaluations and no optimizer step')
        del pinned
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
        result['phase'] = 'ordinary_K'
        write_json(output / 'PROGRESS.json', result)
        loading_started = time.perf_counter()
        require_runtime_pin()
        trainer = SpecForgeEventTrainer(plan['config'], output / 'trainer', objective='osd_first_error',
                                       reduction='response_mean', ranking_lambda=0)
        result['initialization'] = prepare_main(trainer, plan['pinned_initial'], output / 'initialization')
        result['loading_and_initialization_s'] = time.perf_counter() - loading_started
        trainer.bundle.model.begin_event()
        correction = PrefixCorrection(trainer, output_dir=output / 'correction',
                                      record_ids=tuple(row['record_id'] for row in ordinary['records'][:2]))
        log = WorkLog(output / 'core')
        log.trial = 'ordinary_K'
        core_started = time.perf_counter()
        with observe_forwards(trainer, log), count_k_work(trainer, log), correction.capture_k():
            for row in run_window(trainer, ordinary, log):
                result['records'].append(row)
                write_json(output / 'PROGRESS.json', result)
        result['ordinary_K_wall_s'] = time.perf_counter() - core_started
        check_presentations(result['records'], ordinary, output / 'PRESENTATIONS.json')
        validate_k_budget(log, trainer)
        result['ordinary_K_peak_cuda'] = dict(
            allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            scope='fresh trainer loading and original ordinary K step')
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
        fixed_state = invariant_state(trainer)
        post_k_weights = cpu_copy(trainer.bundle.draft_model.state_dict())
        result['phase'] = 'post_K'
        write_json(output / 'PROGRESS.json', result)
        result['post_k'] = correction.measure_post_k()
        persist_comparison(output / 'POST_K_STATE_UNCHANGED.json', invariant_state(trainer), fixed_state)
        persist_comparison(output / 'POST_K_WEIGHTS_UNCHANGED.json',
                           cpu_copy(trainer.bundle.draft_model.state_dict()), post_k_weights)
        del post_k_weights
        result['reproduction'] = check_reproduction(result['post_k'], reference, output / 'REPRODUCTION.json')
        write_json(output / 'POST_K.json', result['post_k'])
        result['phase'] = 'direct_correction'
        write_json(output / 'PROGRESS.json', result)
        result['correction'] = correction.apply_once()
        persist_comparison(output / 'CORRECTION_STATE_UNCHANGED.json', invariant_state(trainer), fixed_state)
        result['phase'] = 'post_C'
        write_json(output / 'PROGRESS.json', result)
        corrected_weights = cpu_copy(trainer.bundle.draft_model.state_dict())
        result['post_c'] = correction.measure_post_c()
        persist_comparison(output / 'POST_C_STATE_UNCHANGED.json', invariant_state(trainer), fixed_state)
        persist_comparison(output / 'POST_C_WEIGHTS_UNCHANGED.json',
                           cpu_copy(trainer.bundle.draft_model.state_dict()), corrected_weights)
        del corrected_weights, fixed_state
        require_runtime_pin()
        result['core'] = correction.report()
        result['budget'] = validate_core_budget(result['core'], log, output / 'BUDGET.json')
        before, after = metric_totals(result['post_k']['responses']), metric_totals(result['post_c']['responses'])
        result['comparison'] = dict(ordinary_K=before, corrected_K=after,
            local_advancement=(after['old_prefix_tokens_lost'] < before['old_prefix_tokens_lost'] and
                               after['net_matching_prefix_change'] > before['net_matching_prefix_change']),
            interpretation='Local training-conditioned comparison; no held-out or native serving claim')
        result.update(status='completed', phase='finished')
    except BaseException as error:
        failure = error
        result.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        if startup_log is not None:
            result['startup_work_counts'] = dict(startup_log.counts)
        if log is not None:
            result['core_work_counts'] = dict(log.counts)
        if correction is not None:
            result['core'] = correction.report()
        if trainer is not None:
            result['completed_optimizer_steps'] = trainer.global_step
            result['completed_presentations'] = trainer.presentations
        current_peak = dict(
            allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        if 'ordinary_K_peak_cuda' in result:
            result['correction_peak_cuda'] = dict(current_peak,
                scope='post-K violation gradients, direct correction and post-C scoring; includes resident K model')
        result['peak_cuda_scope'] = 'max of ordinary K and correction phases; startup reported separately'
        result['peak_cuda_allocated_bytes'] = max(current_peak['allocated_bytes'],
            result.get('ordinary_K_peak_cuda', {}).get('allocated_bytes', 0))
        result['peak_cuda_reserved_bytes'] = max(current_peak['reserved_bytes'],
            result.get('ordinary_K_peak_cuda', {}).get('reserved_bytes', 0))
        for name, owner in (('correction', correction), ('trainer', trainer)):
            if owner is None:
                continue
            try:
                owner.close()
            except BaseException as error:
                result.setdefault('cleanup_errors', []).append(dict(owner=name, error=repr(error)))
                if failure is not None:
                    failure.add_note(f'{name} cleanup: {error!r}')
                else:
                    result.update(status='failed', phase='cleanup')
        result['wall_s'] = time.perf_counter() - started
        result['accounting'] = ('Startup is separate from the core six-forward/four-backward ceiling. '
            'The one Adam step belongs to ordinary K; the direct correction never advances Adam. '
            'Nested component timers are not additional work. No new teacher targets, generations or exports.')
        write_json(output / 'RESULT.json', diagnostic_json(result))
        if failure is None and result.get('cleanup_errors'):
            raise RuntimeError('Diagnostic cleanup failed; see RESULT.json')
    return result


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(sys.argv[1])
