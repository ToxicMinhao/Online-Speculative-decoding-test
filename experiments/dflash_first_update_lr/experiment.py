"""Fixed-input first-update comparison; never generates training/history data."""
from __future__ import annotations

import copy
import time
from experiments.dflash_per_turn.experiment import (
    EXPECTED_STREAMS, digest, require, valid_ids, validate_measurement,
)

LABELS = ('D1', 'L1', 'F')
RATES = {'D1': 3e-4, 'L1': 1e-4}


def validate_reference(reference):
    require(reference['episode_id'] in EXPECTED_STREAMS, 'Unapproved stream')
    require([r['turn'] for r in reference['canonical']] == [1, 2], 'Need first two fixed requests')
    for row in reference['canonical']:
        valid_ids(row['input_ids']); valid_ids(row['output_ids'], output=True)
        require(row['input_ids_sha256'] == digest(row['input_ids']), 'Saved input digest differs')
    require(reference['canonical'][0]['checkpoint_id'] == 'F', 'Response one must use released draft')
    require(reference['original_update']['accounting']['optimizer_steps_this_event'] == 2,
            'Original first update differs')
    for turn in (1, 2):
        for label in (('F',) if turn == 1 else ('F', 'D1')):
            rows = [r for r in reference['original_scores'] if r['turn'] == turn and r['checkpoint_id'] == label]
            require(len(rows) == 3 and {r['repeat'] for r in rows} == {0, 1, 2}, 'Historical repeats missing')
            require(all(r['same_generation_valid'] and r['input_ids'] == reference['canonical'][turn-1]['input_ids']
                        and r['output_ids'] == reference['canonical'][turn-1]['output_ids'] for r in rows),
                    'Historical input/output validity failed')
            require(len({r['verification_calls'] for r in rows}) == 1, 'Original verification counts unstable')
    return True


def benchmark_order(turn, repeat):
    require(turn in (1, 2) and repeat in range(3), 'Outside bounded matrix')
    # First request-2 run is the original-rate reproduction gate, included in 108.
    offset = (repeat + (0 if turn == 2 else 1)) % 3
    return list(LABELS[offset:] + LABELS[:offset])


def check_branch(update, label, support):
    require(update['checkpoint']['checkpoint_id'] == label, 'Wrong branch checkpoint')
    a = update['accounting']; recipe = a['recipe']
    require(recipe['learning_rate'] == RATES[label] and recipe['seed'] == 17 and
            recipe['passes'] == 2 and recipe['responses_per_step'] == 1 and
            recipe['loss'] == 'forward_KL', 'Learning recipe differs')
    require(a['optimizer_steps_total'] == a['optimizer_steps_this_event'] == 2 and
            a['response_presentations_total'] == a['response_presentations_this_event'] == 2 and
            a['support_turns'] == [1], 'Not an independent two-step first update')
    require(update.get('fresh_initialization_verified') is True and
            update.get('original_exposure_match') is True, 'Independence/exposure not verified')
    start = len(support['input_ids']); end = start + len(support['output_ids'])
    mask = [0] * start + [1] * (end-start) + [0] * 9
    require(len(a['response_audits']) == 1 and a['response_audits'][0]['loss_mask_sha256'] == digest(mask),
            'Newest response mask differs')
    return True


def run_stream(reference, backend, *, emit=lambda kind, data: None, snapshot=lambda r: None):
    validate_reference(reference)
    result = dict(schema_version=1, mode='first_update_lr', status='running',
        episode_id=reference['episode_id'], source=reference['source'], synthetic=reference['synthetic'],
        canonical=copy.deepcopy(reference['canonical']), reference_binding=reference['original_result_binding'],
        updates=[], scores=[], seed=17, output_cap=1024, block_size=10,
        reproduction_check={'status': 'pending'}, exposure_checks={'status': 'pending'})
    started = time.monotonic()
    try:
        result['startup'] = backend.start_episode(result['episode_id'])
        support = backend.prepare_support(copy.deepcopy(result['canonical'][0]))
        require(support['input_ids'] == result['canonical'][0]['input_ids'] and
                support['output_ids'] == result['canonical'][0]['output_ids'], 'Saved feedback changed')
        result['support'] = support
        for label in ('D1', 'L1'):
            begin = time.monotonic()
            update = backend.train_branch(label, RATES[label], support, reference['original_update'])
            update['observed_branch_call_s'] = time.monotonic() - begin
            result['updates'].append(update)
            emit('update', update); snapshot(result)
            check_branch(update, label, support)
        a, b = (u['accounting'] for u in result['updates'])
        require(a['anchor_exposure_sha256'] == b['anchor_exposure_sha256'] ==
                reference['original_update']['accounting']['anchor_exposure_sha256'], 'Anchors differ')
        result['exposure_checks'] = dict(status='passed', independent_branches=2,
            optimizer_steps=4, original_anchor_exposure_sha256=a['anchor_exposure_sha256'])
        historical = {label: next(r for r in reference['original_scores'] if r['turn'] == 2 and
                      r['checkpoint_id'] == label)['verification_calls'] for label in ('F', 'D1')}
        for turn in (2, 1):
            item = result['canonical'][turn-1]
            for repeat in range(3):
                order = benchmark_order(turn, repeat)
                emit('method_order', dict(turn=turn, repeat=repeat, checkpoints=order))
                for label in order:
                    expected = backend.checkpoints[label]['checkpoint_identity_sha256']
                    output = backend.serve(list(item['input_ids']), label, capture=False,
                                           turn_index=turn, repeat=repeat)
                    row = {**output, 'episode_id': result['episode_id'], 'source': result['source'],
                        'turn': turn, 'probe': 'next_request_transfer' if turn == 2 else 'in_sample_fit',
                        'checkpoint_id': label, 'repeat': repeat, 'training_data': False, 'capture': False,
                        'context_tokens': len(item['input_ids']), 'output_tokens': len(output['output_ids']),
                        'input_match': output.get('input_ids') == item['input_ids'] and
                            output.get('input_ids_sha256') == digest(item['input_ids']),
                        'canonical_output_match': output.get('output_ids') == item['output_ids'],
                        'checkpoint_identity_match': output.get('checkpoint_identity_sha256') == expected}
                    row['same_generation_valid'] = all(row[k] for k in
                        ('input_match', 'canonical_output_match', 'checkpoint_identity_match'))
                    if turn == 2 and label in historical:
                        row['original_verification_count_match'] = output['verification_calls'] == historical[label]
                    result['scores'].append(row); emit('score', row); snapshot(result)
                    validate_measurement(output)
                    require(row['same_generation_valid'], 'Input/output/weight mismatch: preserve record, investigate')
                    if turn == 2 and label in historical:
                        require(row['original_verification_count_match'],
                                'Original request-2 verification counts not reproduced: investigate before comparison')
                    if turn == 2 and label == 'D1' and repeat == 0:
                        result['reproduction_check'] = dict(status='passed',
                            historical_calls=historical['D1'], observed_calls=output['verification_calls'],
                            output_ids_match=True, input_ids_match=True,
                            scope='Original 3e-4 request 2 reproduced; historical timing is not required to match.')
                        emit('reproduction_gate', result['reproduction_check']); snapshot(result)
        require(len(result['scores']) == 18, 'Incomplete matrix')
        require(all(len({r['verification_calls'] for r in result['scores'] if r['turn'] == t and
                    r['checkpoint_id'] == l}) == 1 for t in (1, 2) for l in LABELS),
                'Fresh repeated verification counts unstable')
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error)); emit('failure', {'error': repr(error)})
    finally:
        try:
            result['cleanup'] = backend.finish_episode()
            require(result['cleanup']['status'] == 'passed', 'Worker shutdown failed')
        except Exception as error:
            result.update(status='failed', cleanup_error=repr(error))
        result['actual_wall_s'] = time.monotonic() - started
        snapshot(result)
    return result
