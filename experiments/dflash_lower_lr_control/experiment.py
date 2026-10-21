"""One new lower-LR trajectory per original group; raw evidence precedes gates."""
from __future__ import annotations
import copy
import time

from experiments.dflash_baseline_learning_curve.experiment import (
    require, digest, validate_measurement, preserve_invalid_numbers)
from experiments.dflash_kl_native_diagnostic.experiment import (
    _finite, validate_loss, validate_pair as validate_reference_pair, SAMPLING_FIELDS)


def validate_pair(pair):
    try:
        previous = validate_reference_pair(pair)
        require(previous['status'] == 'passed', '; '.join(previous['errors']))
        row, loss = pair['L8']['native'], pair['L8']['kl']
        validate_measurement(row)
        require(row['request_id'] == pair['request_id'] and row['group_id'] == pair['group_id'] and
            row['domain'] == pair['domain'] and row['checkpoint_id'] == 'L8' and
            row['checkpoint_identity_sha256'] == pair['checkpoint_bindings']['L8'], 'L8 identity differs')
        frozen = pair['F']['native']
        for key in ('input_ids', 'output_ids', 'finish_reason', 'cap_hit', 'eos_hit', 'role'):
            require(row.get(key) == frozen.get(key), 'L8 native comparison differs: ' + key)
        validate_loss(loss, row, pair['checkpoint_bindings']['L8'])
        require(loss.get('checkpoint_id') == 'L8' and loss.get('diagnostic_model_mode') == 'eval_no_grad' and
            loss.get('training_rng_preserved') is True and loss.get('role') == frozen['role'],
            'L8 KL must remain read-only with its original response role')
        for key in SAMPLING_FIELDS + ('anchor_seed', 'sampling_and_weights_sha256', 'anchor_rng_after_sha256'):
            require(key in loss and loss[key] == pair['F']['kl'][key], 'L8 paired diagnostic anchors differ: ' + key)
        return dict(status='passed', errors=[])
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return dict(status='failed', errors=[str(error)])


def validate_training(receipt, group):
    require(receipt.get('status') == 'passed' and _finite(receipt), 'Incomplete/nonfinite L8 training')
    checkpoint, account = receipt['checkpoint'], receipt['accounting']
    require(checkpoint['checkpoint_id'] == 'L8' and checkpoint['group_id'] == group['group_id'] and
        checkpoint['optimizer_steps'] == 8 and checkpoint['learning_rate'] == 3e-5 and
        checkpoint['checkpoint_identity_sha256'] == digest(checkpoint['files']), 'Invalid genuine L8 checkpoint')
    require(account['episode_id'] == group['group_id'] and account['starting_step'] == 0 and
        account['target_step'] == account['optimizer_steps_total'] == account['optimizer_steps_this_event'] == 8 and
        account['response_presentations_total'] == account['response_presentations_this_event'] == 16 and
        account['checkpoint_id'] == 'L8' and account['source_diagnostic_steps'] == [] and
        account['optimizer_state_exported'] is False, 'Incorrect explicit 0→8 accounting')
    original = group['training']['updates'][0]['accounting']
    recipe = dict(original['recipe'], learning_rate=3e-5)
    require(account['recipe'] == recipe and account['support_bindings'] == original['support_bindings'] and
        account['support_sha256'] == original['support_sha256'] and
        account['feedback_checkpoint_sha256'] == group['checkpoints']['F']['checkpoint_identity_sha256'],
        'Training recipe or response-only fixed feedback differs')
    prior_batches = [b for u in group['training']['updates'] for b in u['accounting']['batches']]
    require(len(account['batches']) == 8 and account['original_exposure_match'] is True, 'Missing eight steps')
    fields = ('record_id', 'turn', 'response_start', 'response_end_exclusive', 'sequence_sha256',
        'loss_mask_sha256', *SAMPLING_FIELDS, 'sampling_and_weights_sha256')
    for i, (batch, prior) in enumerate(zip(account['batches'], prior_batches)):
        require(batch['optimizer_steps_before'] == i and batch['optimizer_steps_after'] == i+1 and
            batch['pass_index'] == i//2 and batch['effective_learning_rate'] == 3e-5 and
            batch['backward_response_scale'] == .5 and batch['original_exposure_match'] is True and
            batch['anchor_rng_after_sha256'] == prior['anchor_rng_after_sha256'] and
            len(batch['presentations']) == 2, 'Training schedule, LR or sampler RNG differs')
        for value, expected in zip(batch['presentations'], prior['presentations']):
            require(all(value[k] == expected[k] for k in fields), 'Training sampled exposure differs')
        require(batch['loss'] == sum(p['loss'] for p in batch['presentations'])/2, 'Loss must average two normalized responses')
    require(account['response_order'] == [r['record_id'] for r in group['support']]*4 and
        account['anchor_rng_before_sha256'] == original['anchor_rng_before_sha256'] and
        account['anchor_rng_after_sha256'] == prior_batches[-1]['anchor_rng_after_sha256'], 'Sampling trajectory changed')
    for population in ('source', 'heldout'):
        values = receipt['losses'][population]
        refs = group['reference_pairs'][population]
        require(len(values) == 4, 'Four read-only response losses required')
        for loss, ref in zip(values, refs):
            validate_loss(loss, ref['F']['native'], checkpoint['checkpoint_identity_sha256'])
            require(loss['checkpoint_id'] == 'L8' and loss['role'] == ref['F']['native']['role'] and
                loss['diagnostic_model_mode'] == 'eval_no_grad' and loss['training_rng_preserved'] is True,
                'Diagnostic roles or RNG preservation changed')
            for key in SAMPLING_FIELDS + ('anchor_seed', 'sampling_and_weights_sha256', 'anchor_rng_after_sha256'):
                require(loss[key] == ref['F']['kl'][key], 'Diagnostic sampled exposure differs: ' + key)
    return dict(status='passed', errors=[])


def _counts(result):
    receipts = [g.get('training') or {} for g in result['groups']]
    result['optimizer_steps'] = sum((r.get('accounting') or {}).get('optimizer_steps_total', 0) for r in receipts)
    result['training_presentations'] = sum((r.get('accounting') or {}).get('response_presentations_total', 0) for r in receipts)
    result['actual_response_kl_evaluations'] = sum(len(v) for r in receipts for v in (r.get('losses') or {}).values())
    result['actual_native_generations'] = sum(len(g['native_raw']) for g in result['groups'])


def run_campaign(groups, backend_factory, *, emit=lambda *args: None, snapshot=lambda value: None):
    require(len(groups) == 6 and len({g['group_id'] for g in groups}) == 6, 'Keep six original groups')
    result = dict(status='running', groups=[], optimizer_steps=0, training_presentations=0,
        actual_native_generations=0, actual_response_kl_evaluations=0, explicit_warmups=0, unexecuted_groups=[])
    started = time.monotonic()
    def persist():
        _counts(result)
        snapshot(preserve_invalid_numbers(result))
    for group in groups:
        tick = time.monotonic()
        current = dict(group_id=group['group_id'], domain=group['domain'], status='running',
            source=[], heldout=[], native_raw=[], costs={})
        result['groups'].append(current)
        backend = None
        try:
            backend = backend_factory(group)
            current['startup'] = backend.start_group(group)
            persist()
            current['training'] = backend.train_group()
            emit('training_raw', preserve_invalid_numbers(current['training']))
            persist()
            current['training_validation'] = validate_training(current['training'], group)
            current['checkpoint'] = current['training']['checkpoint']
            current['trainer_stop'] = backend.stop_training()
            require(current['trainer_stop']['status'] == 'passed' and current['trainer_stop']['training_quiescent'] is True,
                'Training must terminate before native evaluation')
            persist()
            for population in ('source', 'heldout'):
                for i, reference in enumerate(group['reference_pairs'][population]):
                    native = backend.serve_request(copy.deepcopy(reference['F']['native']), population)
                    current['native_raw'].append(native)
                    emit('native_raw', preserve_invalid_numbers(native)); persist()
                    pair = copy.deepcopy(reference)
                    pair['checkpoint_bindings']['L8'] = current['checkpoint']['checkpoint_identity_sha256']
                    pair['L8'] = dict(native=native, kl=current['training']['losses'][population][i])
                    current[population].append(pair)
                    pair['validation'] = validate_pair(pair)
                    persist()
                    require(pair['validation']['status'] == 'passed', '; '.join(pair['validation']['errors']))
            current['status'] = 'complete'
        except Exception as error:
            current.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
            partial = getattr(error, 'diagnostic_partial', None)
            if partial is not None:
                current['training'] = preserve_invalid_numbers(partial)
                current['work_incomplete'] = True
        finally:
            if backend is not None:
                try:
                    current['cleanup'] = backend.finish_group()
                    require(current['cleanup']['status'] == 'passed', 'Owned-worker cleanup failed')
                except Exception as error:
                    current.update(status='failed', cleanup_error=repr(error))
            current['actual_wall_s'] = time.monotonic()-tick
            cleanup = current.get('cleanup', {})
            current['costs'] = dict(startup_s=cleanup.get('startup_s'), shutdown_s=cleanup.get('shutdown_s'),
                activation_s=cleanup.get('activation_s'), native_activation_s=cleanup.get('native_activation_s'),
                fresh_native_request_s=sum(r.get('request_s', 0.) for r in current['native_raw']),
                cache_maintenance_s=sum(r.get('cache_maintenance_s', 0.) for r in current['native_raw']),
                actual_group_wall_s=current['actual_wall_s'],
                accounting='Training/evaluation/export components retained in training receipt; native activation is nested. No historical/fresh speedup comparison.')
            persist()
        emit('group_finished', preserve_invalid_numbers(current))
        if current['status'] != 'complete':
            result['status'] = 'failed'; break
        if len(result['groups']) == 1:
            result['first_group_qualification'] = dict(status='passed', optimizer_steps=8,
                training_presentations=16, native_generations=8, kl_evaluations=8,
                scope='Correctness only. Null/harmful results continue without retuning.')
    _counts(result)
    if len(result['groups']) == 6 and all(g['status'] == 'complete' for g in result['groups']):
        require((result['optimizer_steps'], result['training_presentations'], result['actual_native_generations'],
            result['actual_response_kl_evaluations']) == (48,96,48,48), 'Bounded work accounting differs')
        result['status'] = 'complete'
    result['unexecuted_groups'] = [g['group_id'] for g in groups[len(result['groups']):]]
    result['actual_wall_s'] = time.monotonic()-started
    persist()
    return preserve_invalid_numbers(result)
