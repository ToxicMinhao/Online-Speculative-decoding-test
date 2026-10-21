"""Read-only six-group diagnostic. Preserve each raw result before validation."""
from __future__ import annotations

import copy
import math
import time

from experiments.dflash_baseline_learning_curve.experiment import (
    digest, require, validate_measurement, preserve_invalid_numbers)
from .evidence import source_loss, heldout_records

SAMPLING_FIELDS = ('anchor_positions', 'block_keep_mask', 'label_positions', 'teacher_positions',
    'binary_loss_mask', 'position_weights', 'sampled_blocks', 'eligible_blocks', 'eligible_positions',
    'unique_supervised_positions', 'unique_supervised_position_ids', 'weighted_position_sum',
    'normalization_denominator', 'excluded_anchor_positions', 'zero_supervision',
    'backbone_forward_calls', 'head_projection_chunks')


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return 'nonfinite_float' not in value and all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def validate_loss(row, native, checkpoint):
    require(row.get('request_id', row.get('record_id')) == native['request_id'], 'KL request identity differs')
    require(row.get('checkpoint_identity_sha256') == checkpoint, 'KL checkpoint differs')
    require(row.get('input_ids_sha256') == digest(native['input_ids']) and
            row.get('output_ids_sha256') == digest(native['output_ids']), 'KL saved input/answer differs')
    require(_finite(row), 'Nonfinite KL or metadata')
    zero = row.get('zero_supervision') is True
    if zero:
        require(row.get('eligible_positions') == 0 and row.get('unique_supervised_positions') == 0 and
                row.get('excluded_from_loss_improvement') is True and row.get('loss') in (None, 0, 0.),
                'Zero supervision must be explicit and excluded')
    else:
        require(type(row.get('loss')) in (int, float) and math.isfinite(row['loss']) and
                type(row.get('eligible_positions')) is int and row['eligible_positions'] > 0,
                'Invalid nonempty response loss')
        require(row.get('excluded_from_loss_improvement') is not True, 'Unexpected loss exclusion')
    start, end = len(native['input_ids']), len(native['input_ids']) + len(native['output_ids'])
    positions = row.get('unique_supervised_position_ids')
    require(isinstance(positions, list) and len(set(positions)) == len(positions) == row['unique_supervised_positions'] and
            all(type(p) is int and start < p < end for p in positions), 'KL supervised foreign positions')
    # The unsampleable <2-token branch has no teacher or backbone forward.
    if row.get('sampled_blocks') == 0:
        require(zero and len(native['output_ids']) < 2, 'Unexpected empty anchor set')
        return
    require(all(key in row for key in SAMPLING_FIELDS), 'Missing KL anchor/exposure metadata')
    require(row['sampling_and_weights_sha256'] == digest({k: row[k] for k in SAMPLING_FIELDS}),
            'KL sampling/weight hash differs')
    require(type(row.get('anchor_seed')) is int and type(row['normalization_denominator']) in (int, float) and
            row['normalization_denominator'] > 0, 'Invalid anchor seed/denominator')
    require(len(row['anchor_positions']) == row['sampled_blocks'] and
            all(start <= p < end for p in row['anchor_positions']), 'KL anchor outside response')


def validate_pair(pair):
    try:
        require(pair.get('population') in ('source', 'heldout'), 'Unknown request population')
        for label in ('F', 'S8'):
            row, loss = pair[label]['native'], pair[label]['kl']
            validate_measurement(row)
            require(row['request_id'] == pair['request_id'] and row['checkpoint_id'] == label,
                    'Native request/checkpoint differs')
            require(row['checkpoint_identity_sha256'] == pair['checkpoint_bindings'][label], 'Wrong group checkpoint')
            validate_loss(loss, row, pair['checkpoint_bindings'][label])
        f, s = pair['F']['native'], pair['S8']['native']
        require(f['input_ids'] == s['input_ids'] and f['output_ids'] == s['output_ids'], 'Native input/output mismatch')
        require(f['finish_reason'] == s['finish_reason'] and f['cap_hit'] == s['cap_hit'] and
                f.get('eos_hit') == s.get('eos_hit'), 'Native termination differs')
        a, b = pair['F']['kl'], pair['S8']['kl']
        for key in SAMPLING_FIELDS + ('anchor_seed', 'sampling_and_weights_sha256'):
            require(key in a and key in b and a[key] == b[key], 'Paired KL anchors/exposure differ: ' + key)
        if pair['population'] == 'heldout':
            for value in (a, b):
                require(value.get('diagnostic_model_mode') == 'eval_no_grad' and value.get('training_rng_preserved') is True,
                        'Held-out evaluation is not read-only/RNG-preserving')
        return dict(status='passed', errors=[])
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return dict(status='failed', errors=[str(error)])


def _pair(group, population, frozen, adapted, f_loss, s_loss):
    return dict(group_id=group['group_id'], domain=group['domain'], population=population,
        request_id=frozen['request_id'], checkpoint_bindings={k: group['checkpoints'][k]['checkpoint_identity_sha256'] for k in ('F', 'S8')},
        F=dict(native=copy.deepcopy(frozen), kl=copy.deepcopy(f_loss)),
        S8=dict(native=copy.deepcopy(adapted), kl=copy.deepcopy(s_loss)))


def run_campaign(groups, backend_factory, *, emit=lambda *args: None, snapshot=lambda value: None):
    require(len(groups) == 6, 'Keep the six original groups')
    result = dict(status='running', groups=[], planned_source_generations=24,
        planned_response_kl_evaluations=48, optimizer_steps=0, actual_source_generations=0,
        actual_response_kl_evaluations=0, explicit_warmups=0, unexecuted_groups=[])
    start = time.monotonic()
    for group in groups:
        group_start = time.monotonic()
        current = dict(group_id=group['group_id'], domain=group['domain'], status='running', source=[], heldout=[],
                       source_raw=[], heldout_loss_raw=[], costs={})
        result['groups'].append(current)
        backend = None
        try:
            backend = backend_factory(group)
            current['startup'] = backend.start_group(group)
            snapshot(preserve_invalid_numbers(result))
            for frozen in group['feedback']:
                native = backend.serve_source(copy.deepcopy(frozen))
                current['source_raw'].append(native)
                result['actual_source_generations'] += 1
                emit('source_generation_raw', preserve_invalid_numbers(native))
                snapshot(preserve_invalid_numbers(result))
                pair = _pair(group, 'source', frozen, native, source_loss(group, 'F', frozen), source_loss(group, 'S8', frozen))
                current['source'].append(pair)
                pair['validation'] = validate_pair(pair)
                snapshot(preserve_invalid_numbers(result))
                require(pair['validation']['status'] == 'passed', '; '.join(pair['validation']['errors']))
            records = heldout_records(group)
            losses = {}
            for label in ('F', 'S8'):
                value = backend.evaluate_heldout(copy.deepcopy(records), label)
                current['heldout_loss_raw'].append(value)
                result['actual_response_kl_evaluations'] += len(value.get('response_losses', []))
                emit('heldout_loss_raw', preserve_invalid_numbers(value))
                snapshot(preserve_invalid_numbers(result))
                require(value.get('status') == 'passed' and value.get('checkpoint_id') == label and
                        value.get('checkpoint_identity_sha256') == group['checkpoints'][label]['checkpoint_identity_sha256'],
                        'Wrong held-out loss checkpoint/status')
                require(len(value['response_losses']) == 4 and
                        [r.get('request_id', r.get('record_id')) for r in value['response_losses']] ==
                        [r['request_id'] for r in records], 'Missing/reordered held-out loss')
                losses[label] = value['response_losses']
            for i, (frozen, adapted) in enumerate(zip(records, heldout_records(group, 'S8'))):
                pair = _pair(group, 'heldout', frozen, adapted, losses['F'][i], losses['S8'][i])
                current['heldout'].append(pair)
                pair['validation'] = validate_pair(pair)
                snapshot(preserve_invalid_numbers(result))
                require(pair['validation']['status'] == 'passed', '; '.join(pair['validation']['errors']))
            current['status'] = 'complete'
        except Exception as error:
            current['status'] = 'failed'
            current['error'] = dict(type=type(error).__name__, message=str(error))
            partial = getattr(error, 'diagnostic_partial', None)
            if partial is not None:
                current['failed_loss_partial'] = preserve_invalid_numbers(partial)
                result['actual_response_kl_evaluations'] += len(partial.get('response_losses', []))
                current['loss_work_incomplete'] = True
        finally:
            if backend is not None:
                try:
                    current['cleanup'] = backend.finish_group()
                    require(current['cleanup'].get('status') == 'passed', 'Native cleanup failed')
                except Exception as error:
                    current['status'] = 'failed'
                    current['cleanup_error'] = repr(error)
            current['actual_wall_s'] = time.monotonic() - group_start
            cleanup = current.get('cleanup', {})
            loss_receipts = current['heldout_loss_raw'] + ([current['failed_loss_partial']] if 'failed_loss_partial' in current else [])
            current['costs'] = dict(
                startup_s=cleanup.get('startup_s'), shutdown_s=cleanup.get('shutdown_s'),
                activation_s=cleanup.get('activation_s'), native_activation_s=cleanup.get('native_activation_s'),
                source_replay_request_s=sum(row.get('request_s', 0.) for row in current['source_raw']),
                source_cache_maintenance_s=sum(row.get('cache_maintenance_s', 0.) for row in current['source_raw']),
                heldout_loss_receipts=[dict(checkpoint_id=r.get('checkpoint_id'), costs=r.get('costs', {}),
                    diagnostic_wall_s=r.get('diagnostic_wall_s'), response_phase_seconds=[v.get('phase_seconds', {})
                    for v in r.get('response_losses', [])]) for r in loss_receipts],
                actual_group_wall_s=current['actual_wall_s'], optimizer_steps=0, checkpoint_exports=0,
                accounting='Diagnostic work only. Teacher preparation is nested in loss wall time; native activation is nested in activation. No cross-run latency comparison.')
            snapshot(preserve_invalid_numbers(result))
        emit('group_finished', preserve_invalid_numbers(current))
        if current['status'] != 'complete':
            result['status'] = 'failed'
            break
        if len(result['groups']) == 1:
            result['first_group_qualification'] = dict(status='passed', source_generations=4, kl_evaluations=8,
                scope='Correctness only; null/harmful results continue without retuning.')
    if len(result['groups']) == 6 and all(g['status'] == 'complete' for g in result['groups']):
        require(result['actual_source_generations'] == 24 and result['actual_response_kl_evaluations'] == 48,
                'Diagnostic execution count differs')
        result['status'] = 'complete'
    result['unexecuted_groups'] = [g['group_id'] for g in groups[len(result['groups']):]]
    result['actual_wall_s'] = time.monotonic() - start
    snapshot(preserve_invalid_numbers(result))
    return preserve_invalid_numbers(result)
