"""Three explicit P8 trajectories; durable raw evidence precedes validity gates."""
from __future__ import annotations
import copy
import json
import time

from experiments.dflash_baseline_learning_curve.experiment import (
    require, digest, validate_measurement, preserve_invalid_numbers)
from experiments.dflash_kl_native_diagnostic.experiment import (
    _finite, validate_loss, validate_pair as validate_reference_pair, SAMPLING_FIELDS)
from .protocol import PLANNED, SERVING_CONTRACT
from .evidence import composite_identity, validate_pool


def validate_pair(pair):
    """Revalidate every original reference and the distinct pooled checkpoint."""
    try:
        reference = validate_reference_pair(pair)
        require(reference['status'] == 'passed', '; '.join(reference['errors']))
        require(pair['pool_id'] == 'p8_' + pair['domain'], 'P8 pool/domain identity differs')
        require(pair['pool_request_id'] == json.dumps(list(composite_identity(pair)),
                separators=(',', ':'), ensure_ascii=True), 'P8 composite identity differs')
        f = pair['F']['native']
        for label in ('F', 'S8'):
            require(all(pair[label]['native'][k] == pair[k] for k in ('domain','group_id','request_id')),
                    'Original group reference identity differs')
        row, loss = pair['P8']['native'], pair['P8']['kl']
        validate_measurement(row)
        require(all(row[k] == pair[k] for k in ('domain','group_id','request_id','pool_id','pool_request_id')),
                'P8 native ownership differs')
        require(row['checkpoint_id'] == 'P8' and
                row['checkpoint_identity_sha256'] == pair['checkpoint_bindings']['P8'], 'Wrong P8 checkpoint')
        require(row.get('serving_contract') == SERVING_CONTRACT and row.get('capture') is False and
                row.get('feedback_capture_enabled') is False, 'Serving conditions or capture differ')
        for key in ('input_ids','output_ids','finish_reason','cap_hit','eos_hit','role'):
            require(row.get(key) == f.get(key), 'P8 native comparison differs: ' + key)
        validate_loss(loss, row, pair['checkpoint_bindings']['P8'])
        require(loss.get('checkpoint_id') == 'P8' and loss.get('diagnostic_model_mode') == 'eval_no_grad' and
                loss.get('training_rng_preserved') is True and loss.get('role') == f['role'],
                'P8 diagnostic mode/role differs')
        require(all(loss.get(k) == pair[k] for k in ('group_id','request_id','pool_id','pool_request_id')),
                'P8 diagnostic owner differs')
        for key in SAMPLING_FIELDS + ('anchor_seed','sampling_and_weights_sha256','anchor_rng_after_sha256'):
            require(key in loss and loss[key] == pair['F']['kl'][key], 'P8 diagnostic exposure differs: ' + key)
        return dict(status='passed', errors=[])
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return dict(status='failed', errors=[str(error)])


def validate_training(receipt, pool):
    validate_pool(pool)
    require(receipt.get('status') == 'passed' and _finite(receipt), 'Incomplete/nonfinite P8 training')
    checkpoint, account = receipt['checkpoint'], receipt['accounting']
    require(checkpoint['checkpoint_id'] == 'P8' and checkpoint['pool_id'] == pool['pool_id'] and
            checkpoint['domain'] == pool['domain'] and checkpoint['group_ids'] == pool['group_ids'] and
            checkpoint['optimizer_steps'] == 8 and checkpoint['learning_rate'] == 3e-4 and
            checkpoint['checkpoint_identity_sha256'] == digest(checkpoint['files']), 'P8 checkpoint ownership differs')
    from .training import validate_training_accounting
    validate_training_accounting(account, pool)
    for population in ('source','heldout'):
        values, refs = receipt['losses'][population], pool['reference_pairs'][population]
        require(len(values) == len(refs) == 8, 'Eight read-only P8 losses per population required')
        for loss, ref in zip(values, refs):
            validate_loss(loss, ref['F']['native'], checkpoint['checkpoint_identity_sha256'])
            require(loss.get('checkpoint_id') == 'P8' and loss.get('diagnostic_model_mode') == 'eval_no_grad' and
                    loss.get('training_rng_preserved') is True and loss.get('role') == ref['F']['native']['role'],
                    'P8 diagnostic changed mode, RNG or role')
            require(all(loss.get(k) == ref[k] for k in ('group_id','request_id','pool_id','pool_request_id')),
                    'P8 diagnostic identity differs')
            for key in SAMPLING_FIELDS + ('anchor_seed','sampling_and_weights_sha256','anchor_rng_after_sha256'):
                require(key in loss and loss[key] == ref['F']['kl'][key], 'Original diagnostic binding differs: ' + key)
    return dict(status='passed', errors=[])


def _counts(result):
    pools = result['pools']
    receipts = [p.get('training') or {} for p in pools]
    result['optimizer_steps'] = sum((r.get('accounting') or {}).get('optimizer_steps_total', 0) for r in receipts)
    result['training_presentations'] = sum((r.get('accounting') or {}).get('response_presentations_total', 0) for r in receipts)
    result['actual_response_kl_evaluations'] = sum(len(v) for r in receipts for v in (r.get('losses') or {}).values())
    result['actual_native_generations'] = sum(len(p['native_raw']) for p in pools)
    result['attempted_native_generations'] = sum(p.get('attempted_native_generations',0) for p in pools)
    result['unknown_native_work_receipts'] = result['attempted_native_generations'] - result['actual_native_generations']
    result['unattempted_native_generations'] = 48 - result['attempted_native_generations']
    unknown_training = sum(p.get('training_requested',False) and not p.get('training') for p in pools)
    result['unknown_training_work_receipts'] = unknown_training
    # Entire unresolved worker allocations stay unknown, never presumed unexecuted.
    result['unknown_optimizer_steps'] = 8 * unknown_training + sum(
        (r.get('accounting') or {}).get('optimizer_steps_unknown', 0) for r in receipts)
    unknown_kl = 16 * unknown_training
    for receipt in receipts:
        diagnostics = dict(receipt.get('loss_diagnostics') or {})
        failed = receipt.get('failure_detail') or {}
        if failed.get('population') in ('source','heldout'):
            diagnostics[failed['population']] = failed
        for population, diagnostic in diagnostics.items():
            returned = len((receipt.get('losses') or {}).get(population, []))
            unknown_kl += diagnostic.get('unknown_response_calls',
                max(0, diagnostic.get('attempted_response_calls', returned) - returned))
    result['unknown_response_kl_evaluations'] = unknown_kl
    # A lost worker receipt makes its whole allocation unknown; it does not
    # establish that all sixteen diagnostic evaluations were attempted.
    result['attempted_response_kl_evaluations'] = result['actual_response_kl_evaluations'] + unknown_kl - 16 * unknown_training
    result['unresolved_response_kl_allocation_size'] = 16 * unknown_training
    result['unattempted_optimizer_steps'] = 24 - result['optimizer_steps'] - result['unknown_optimizer_steps']
    result['unattempted_response_kl_evaluations'] = 48 - result['actual_response_kl_evaluations'] - unknown_kl
    result['unknown_response_kl_worker_allocations'] = unknown_training


def run_campaign(pools, backend_factory, *, emit=lambda *args: None, snapshot=lambda value: None):
    require(len(pools) == 3 and len({p['pool_id'] for p in pools}) == 3, 'Exactly three dataset pools required')
    for pool in pools:
        validate_pool(pool)
    result = dict(status='running', pools=[], planned=dict(PLANNED), explicit_warmups=0, unexecuted_pools=[])
    started = time.monotonic()
    def persist():
        _counts(result)
        snapshot(preserve_invalid_numbers(result))
    persist()
    for pool in pools:
        tick = time.monotonic()
        current = dict(pool_id=pool['pool_id'], domain=pool['domain'], group_ids=pool['group_ids'],
            pool_payload_sha256=pool['pool_payload_sha256'], status='running', source=[], heldout=[],
            native_raw=[], attempted_native_generations=0, training_requested=False, costs={})
        result['pools'].append(current)
        backend = None
        try:
            backend = backend_factory(pool)
            current['startup'] = backend.start_pool(pool)
            persist()
            current['training_requested'] = True
            persist()
            current['training'] = backend.train_pool()
            emit('training_raw', preserve_invalid_numbers(current['training'])); persist()
            current['training_validation'] = validate_training(current['training'], pool)
            current['checkpoint'] = current['training']['checkpoint']
            current['trainer_stop'] = backend.stop_training()
            require(current['trainer_stop']['status'] == 'passed' and current['trainer_stop']['training_quiescent'] is True,
                    'Training must terminate before native evaluation')
            persist()
            for population in ('source','heldout'):
                for i, reference in enumerate(pool['reference_pairs'][population]):
                    current['attempted_native_generations'] += 1
                    persist()
                    native = backend.serve_request(copy.deepcopy(reference['F']['native']), population)
                    current['native_raw'].append(native)
                    emit('native_raw', preserve_invalid_numbers(native)); persist()
                    pair = copy.deepcopy(reference)
                    pair['checkpoint_bindings']['P8'] = current['checkpoint']['checkpoint_identity_sha256']
                    pair['P8'] = dict(native=native, kl=current['training']['losses'][population][i])
                    current[population].append(pair)
                    pair['validation'] = validate_pair(pair)
                    persist()
                    require(pair['validation']['status'] == 'passed', '; '.join(pair['validation']['errors']))
            current['status'] = 'complete'
        except Exception as error:
            current.update(status='failed', error=dict(type=type(error).__name__,message=str(error)))
            partial = getattr(error,'diagnostic_partial',None)
            if partial is not None:
                current['training'] = preserve_invalid_numbers(partial)
                current['work_incomplete'] = True
        finally:
            if backend is not None:
                try:
                    current['cleanup'] = backend.finish_pool()
                    require(current['cleanup']['status'] == 'passed', 'Owned-worker cleanup failed')
                except Exception as error:
                    current.update(status='failed',cleanup_error=repr(error))
            current['actual_wall_s'] = time.monotonic()-tick
            cleanup = current.get('cleanup',{})
            current['costs'] = dict(startup_s=cleanup.get('startup_s'),shutdown_s=cleanup.get('shutdown_s'),
                activation_s=cleanup.get('activation_s'),native_activation_s=cleanup.get('native_activation_s'),
                fresh_native_request_s=sum(r.get('request_s',0.) for r in current['native_raw']),
                cache_maintenance_s=sum(r.get('cache_maintenance_s',0.) for r in current['native_raw']),
                actual_pool_wall_s=current['actual_wall_s'],
                accounting='Components in training receipt; activation is outside inference; nested costs are not added twice. Historical/fresh timings cannot establish speedup.')
            persist()
        emit('pool_finished',preserve_invalid_numbers(current))
        if current['status'] != 'complete':
            result['status'] = 'failed'; break
        if len(result['pools']) == 1:
            result['first_pool_qualification'] = dict(status='passed',optimizer_steps=8,
                training_presentations=16,native_generations=16,kl_evaluations=16,
                scope='Correctness only; null/harmful results continue unchanged.')
    _counts(result)
    if len(result['pools']) == 3 and all(p['status'] == 'complete' for p in result['pools']):
        require((result['optimizer_steps'],result['training_presentations'],result['actual_native_generations'],
                 result['actual_response_kl_evaluations']) == (24,48,48,48), 'Bounded work accounting differs')
        result['status'] = 'complete'
    result['unexecuted_pools'] = [p['pool_id'] for p in pools[len(result['pools']):]]
    result['actual_wall_s'] = time.monotonic()-started
    persist()
    return preserve_invalid_numbers(result)
