"""Single retained G qualification and bounded backward diagnostics; never serves.

Private state/config/response paths are CLI inputs. No model-sized gradients are
written. Diagnostic backwards cannot publish a checkpoint or resume a campaign.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
from unittest.mock import patch

from . import training
from .gradient_diagnostics import durable_write, summarize_gradient_sum
from .gradient_probe import run_backward_probe, compare_gradients

BASE_COMMIT = '4a2e533d17596ab89e01357511c4ae05971c1071'
STATE_SHA = 'dbf2b1dab3b5dcbfe36164425d4391594fd4d5210e4f7c67b126ddbc2512b899'
RESPONSE_SHA = '985d4649c0b46241c790305b2e0c23cf9fea3ca2271cb07c9ef0e0a16e9ffab6'
STATE_ID = '8ce4d9e18a0b09c4b916ec37aac6087188a9255cb534d4b8e16f00d24dd8eca8'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify_restored(trainer, expected):
    """Inspect a restored but deliberately poisoned trainer without unpoisoning it."""
    failed = trainer.failed
    try:
        trainer.failed = False
        observed = training.state_digest(trainer.snapshot())
    finally:
        trainer.failed = failed
    training.require(observed == expected, 'Diagnostic did not restore exact initial state')
    return dict(status='passed', expected=expected, observed=observed,
                failed_flag_preserved=trainer.failed == failed)


@contextmanager
def eager_flex_backend():
    """Same FlexAttention API/BlockMask, Torch eager HOP backend; not SDPA."""
    import transformers.integrations.flex_attention as integration
    from torch.nn.attention.flex_attention import flex_attention
    original = integration.compile_friendly_flex_attention
    calls = []
    def eager(query, key, value, training=False, **kwargs):
        calls.append(dict(q_shape=list(query.shape), k_shape=list(key.shape), dtype=str(query.dtype),
                          kwargs=sorted(kwargs)))
        return flex_attention(query, key, value, **kwargs)
    integration.compile_friendly_flex_attention = eager
    try:
        yield calls
    finally:
        integration.compile_friendly_flex_attention = original


def execute(config_path, state_path, response_path, out):
    import torch
    from . import native
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(base_commit=BASE_COMMIT, status='initializing', main_campaign='paused',
                  serving_executions=0, diagnostic_checkpoints_exported=0,
                  persistent_updates=0, temporary_optimizer_steps={'ordinary_reference': 0, 'grouped_trial': 0},
                  variants={}, checks={}, configuration_changes='diagnostic observation only',
                  anchor_provenance='Reconstructed from verified retained RNG; historical per-pass anchor lists unavailable.',
                  teacher_provenance='Reconstructed from exact saved token IDs, no answer generation.')
    def save():
        report['elapsed_s'] = time.perf_counter() - started
        report['peak_host_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if torch.cuda.is_initialized():
            report['peak_cuda_allocated_bytes'] = max(report.get('peak_cuda_allocated_bytes', 0), torch.cuda.max_memory_allocated())
            report['peak_cuda_reserved_bytes'] = max(report.get('peak_cuda_reserved_bytes', 0), torch.cuda.max_memory_reserved())
        durable_write(out / 'RESULT.json', report)
    service = trainer = None
    initial = initial_identity = None
    try:
        training.require(file_sha(state_path) == STATE_SHA, 'Retained initial state file changed')
        training.require(file_sha(response_path) == RESPONSE_SHA, 'Retained response file changed')
        retained = torch.load(state_path, map_location='cpu', weights_only=False)
        training.require(training.state_digest(retained) == STATE_ID, 'Retained state content differs')
        training.require(not retained['includes_draft'] and not retained['optimizer']['state'] and
                         retained['optimizer_steps'] == retained['completed_events'] == 0,
                         'Expected released/fresh first-update state')
        response = json.loads(Path(response_path).read_text())
        config = json.loads(Path(config_path).read_text())
        training.require(config['agreement_arm'] == 'G' and config['agreement_qualification'] is True,
                         'Only the failed G qualification is allowed')
        # Never expose the retired RPC credential in new artifacts; no RPC server.
        config.pop('rpc_token', None)
        config['worker_out'] = str(out / 'training_worker')
        Path(config['worker_out']).mkdir()
        config['agreement_source_sha256'] = native.source_pins()
        report['source_pins'] = config['agreement_source_sha256']
        report['retained_bindings'] = dict(state_sha256=STATE_SHA, response_sha256=RESPONSE_SHA,
                                          retained_state_identity=STATE_ID)
        report['protocol'] = dict(group=training.GROUP_PROTOCOL, tolerances=training.EQUIVALENCE_TOLERANCES,
                                  numerics=training.core.NUMERICS)
        native.base.check_dependencies(config['workspace_root'])
        # Match original worker admission: physical device has already been narrowed.
        allocation = os.environ.pop('CUDA_VISIBLE_DEVICES')
        try:
            native.base.check_environment(config)
        finally:
            os.environ['CUDA_VISIBLE_DEVICES'] = allocation
        service = native.TrainerService(config)
        trainer = service.trainer
        released = {k: v.detach().cpu().clone() for k, v in trainer.draft.state_dict().items()}
        trainer.restore(retained, draft_state=released)
        initial = trainer.snapshot()
        initial_identity = training.state_digest(initial)
        report['checks']['retained_state_restored'] = (
            training.state_digest(trainer.snapshot(include_draft=False)) == STATE_ID)
        training.require(report['checks']['retained_state_restored'], 'Retained RNG/optimizer state not reproduced')
        del released, retained
        report['status'] = 'reproducing_original_sequence'
        save()

        case = {}
        original_loss = training.grouped_response_loss
        ordinary_event = training.qualified.PersistentPerTurnTrainer.train_event
        optimizer_step = trainer.optimizer.step
        phase = ['ordinary_reference']
        def count_step(*args, **kwargs):
            result = optimizer_step(*args, **kwargs)
            report['temporary_optimizer_steps'][phase[0]] += 1
            return result
        def ordinary(*args, **kwargs):
            result = ordinary_event(*args, **kwargs)
            durable_write(out / 'ORDINARY_REFERENCE.json', result)
            report['ordinary_reference_exposure_sha256'] = result['anchor_exposure_sha256']
            return result
        def capture_forward(*args, **kwargs):
            phase[0] = 'grouped_trial'
            case.clear()
            case.update(state=trainer.snapshot(), args=args, kwargs=dict(kwargs),
                        rng=training.rng_state(), anchor_rng=trainer.generator.get_state().cpu().clone())
            result = original_loss(*args, **kwargs)
            case['sampling'] = copy.deepcopy(result[3])
            case['losses'] = {k: float(v.detach()) for k, v in zip(('F', 'A', 'B'), result[:3])}
            return result
        def observe(summary, a, b, full):
            if summary['status'] != 'passed':
                case['summary'] = summary
                case['gradients'] = {label: {name: x.detach().to(device='cpu', dtype=torch.float32, copy=True)
                                            for name, x in row.items()}
                                     for label, row in (('A', a), ('B', b), ('F', full))}
        trainer.gradient_diagnostics_observer = observe
        original_error = None
        reproduction_started = time.perf_counter()
        try:
            with patch.object(trainer.optimizer, 'step', count_step), \
                 patch.object(training.qualified.PersistentPerTurnTrainer, 'train_event', ordinary), \
                 patch.object(training, 'grouped_response_loss', capture_forward):
                receipt = trainer.qualify_disabled_filter([response])
                durable_write(out / 'UNEXPECTED_QUALIFICATION_PASS.json', receipt)
        except Exception as error:
            original_error = str(error)
            report['original_failure'] = dict(type=type(error).__name__, message=str(error),
                                               traceback=traceback.format_exc())
        finally:
            trainer.gradient_diagnostics_observer = None
            report['checks']['reproduction_restoration'] = verify_restored(trainer, initial_identity)
            report['reproduction_wall_s'] = time.perf_counter() - reproduction_started
            report['original_failure_reproduced'] = (
                original_error == 'Grouped raw gradients differ beyond declared sum tolerance')
            save()
        if 'gradients' not in case or not report['original_failure_reproduced']:
            report['status'] = 'stopped_original_gate_not_reproduced'
            return report
        report['reproduced_gradient_check'] = case['summary']
        baseline = case['gradients']
        case_identity = training.state_digest(case['state'])
        report['failed_pass_state_identity'] = case_identity
        report['status'] = 'diagnostic_backwards'
        save()

        def restore_case():
            trainer.filter_enabled = case['state']['agreement']['filter_enabled']
            trainer.restore(case['state'])
            training.require(training.state_digest(trainer.snapshot()) == case_identity,
                             'Variant starting weights/optimizer/RNG differ')
            training.restore_rng(case['rng'])
            trainer.generator.set_state(case['anchor_rng'])

        def factory():
            value = original_loss(*case['args'], **case['kwargs'])
            training.require(value[3] == case['sampling'], 'Variant changed sampled anchors/groups/weights')
            return value

        def variant(name, order, *, eager=False):
            restore_case()
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            item = dict(status='running', order=list(order), diagnostic_only=True)
            report['variants'][name] = item
            save()
            try:
                if eager:
                    with eager_flex_backend() as calls:
                        result = run_backward_probe(factory, trainer.parameters, order=order)
                    training.require(len(calls) > 0, 'Equivalent eager wrapper was not reached')
                    item['eager_attention_calls'] = calls
                else:
                    result = run_backward_probe(factory, trainer.parameters, order=order)
                item.update({k: v for k, v in result.items() if k != 'gradients'})
                item['comparisons_to_original'] = {
                    label: compare_gradients(row, baseline[label if label in ('A', 'B') else 'F'], trainer.parameters)
                    for label, row in result['gradients'].items()}
                if set(('A', 'B', 'F')) <= result['gradients'].keys():
                    item['gradient_sum'] = summarize_gradient_sum(
                        result['gradients']['A'], result['gradients']['B'], result['gradients']['F'],
                        {k: case['state']['draft'][k] for k in trainer.parameters},
                        context=dict(trial_type=name, original_context=case['summary']['context']))
                if 'F1' in result['gradients'] and 'F2' in result['gradients']:
                    item['retained_graph_repeat'] = compare_gradients(
                        result['gradients']['F1'], result['gradients']['F2'], trainer.parameters)
                del result
                item['status'] = 'complete'
            except Exception as error:
                item.update(status='unavailable_or_failed', error=repr(error), traceback=traceback.format_exc())
            finally:
                item['wall_s'] = time.perf_counter() - tick
                item['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                item['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
                gc.collect()
                torch.cuda.empty_cache()
                restore_case()
                item['state_restoration'] = 'exact'
                save()
            return item

        repeated = variant('compiled_full_repeat', ('F1', 'F2'))
        ordered = variant('compiled_full_A_B', ('F', 'A', 'B'))
        if repeated['status'] != 'complete' or ordered['status'] != 'complete':
            report['status'] = 'stopped_native_probe_unavailable'
            return report
        exact = repeated['retained_graph_repeat']['global']['exactly_equal'] and all(
            row['global']['exactly_equal'] for item in (repeated, ordered)
            for row in item['comparisons_to_original'].values())
        report['compiled_repeat_and_order_exact'] = exact
        # Both stable and order-dependent native results leave the backend's
        # role unresolved. One equivalent eager case is the final comparison;
        # failure/OOM is recorded without changing anchors or precision.
        variant('eager_A_B_full', ('A', 'B', 'F'), eager=True)
        report['status'] = 'bounded_diagnostic_complete'
        return report
    except Exception as error:
        report.update(status='diagnostic_failed', error=repr(error), traceback=traceback.format_exc())
        return report
    finally:
        try:
            if trainer is not None and initial is not None:
                trainer.filter_enabled = initial['agreement']['filter_enabled']
                trainer.restore(initial)
                report['checks']['final_restoration'] = verify_restored(trainer, initial_identity)
        except Exception as error:
            report.update(status='diagnostic_failed_restoration', restoration_error=repr(error),
                          restoration_traceback=traceback.format_exc())
        finally:
            if trainer is not None:
                trainer.failed = True
            report['main_campaign'] = 'paused_no_automatic_resume'
            try:
                save()
            finally:
                if service is not None:
                    service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--response', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = execute(args.config, args.state, args.response, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_s', 'main_campaign')}), flush=True)
    return 0 if result['status'].startswith(('bounded_', 'stopped_')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
