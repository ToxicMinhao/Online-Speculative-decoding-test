"""Execute each sealed native location/arm once; retain failures before gating.

No optimizer, feedback capture, full generation or loss evaluation is constructed
here. Historical target verification forwards reconstruct the saved schedule;
they are separately budgeted and never scored as new verification outcomes. The first group gates correctness, not efficacy.
"""
from __future__ import annotations

import copy
import time

from experiments.dflash_baseline_learning_curve.experiment import preserve_invalid_numbers
from .protocol import ARMS, ADAPTED_ARMS, MAX_BLOCK_EVALUATIONS, require, validate_row, compare_anchor
from .evidence import validate_plan


def _count(result):
    locations = [loc for group in result['groups'] for loc in group['locations']]
    result['attempted_arm_evaluations'] = sum(len(loc['attempted_arms']) for loc in locations)
    rows = [row for loc in locations for row in loc['rows'].values()]
    counts = [row.get('draft_block_evaluations') for row in rows]
    result['actual_draft_block_evaluations'] = sum(n for n in counts if type(n) is int and n >= 0)
    result['unknown_work_receipts'] = sum(n is None for n in counts) + sum(
        len(set(loc['attempted_arms']) - set(loc['rows'])) for loc in locations)
    result['unattempted_arm_evaluations'] = result['planned_arm_evaluations'] - result['attempted_arm_evaluations']
    result['valid_pair_comparisons'] = sum(
        pair.get('valid') is True for loc in locations for pair in loc['comparisons'].values())
    for key in ('target_verifications', 'reconstruction_target_verifications',
                'reconstruction_target_verify_attempts', 'scored_target_verifications',
                'regenerated_response_tokens', 'training_steps'):
        result[key] = sum(row[key] for row in rows if type(row.get(key)) is int and row[key] >= 0)
    result['unknown_reconstruction_work_receipts'] = sum(
        type(row.get('reconstruction_target_verifications')) is not int for row in rows) + sum(
        len(set(loc['attempted_arms']) - set(loc['rows'])) for loc in locations)
    result['unattempted_reconstruction_target_verifications'] = sum(
        len(set(ARMS) - set(loc['attempted_arms'])) * loc['anchor']['reconstruction']['target_verification_count']
        for loc in locations)
    # A failed probe can stop partway through its saved verification schedule.
    # Count its known remaining rounds as well as wholly unattempted arms. A
    # missing/unknown receipt cannot establish which rounds remain unattempted.
    result['unattempted_reconstruction_target_verifications'] += sum(
        max(0, loc['anchor']['reconstruction']['target_verification_count'] - count)
        for loc in locations for arm in loc['attempted_arms']
        for count in [loc['rows'].get(arm, {}).get('reconstruction_target_verifications')]
        if type(count) is int and count >= 0)

    require(result['attempted_arm_evaluations'] <= result['planned_arm_evaluations'] <= MAX_BLOCK_EVALUATIONS,
            'Sealed diagnostic budget exceeded')


def run_campaign(plan, backend_factory, *, emit=lambda *args: None, snapshot=lambda value: None):
    validate_plan(plan)
    groups = plan['groups']
    result = dict(status='running', plan=copy.deepcopy(plan), groups=[],
        selected_locations=sum(len(r['anchors']) for g in groups for r in g['requests']),
        planned_arm_evaluations=3*sum(len(r['anchors']) for g in groups for r in g['requests']),
        planned_reconstruction_target_verifications=sum(len(ARMS) * anchor['reconstruction']['target_verification_count']
            for group in groups for request in group['requests'] for anchor in request['anchors']),
        attempted_arm_evaluations=0, actual_draft_block_evaluations=0,
        explicit_warmups=0, automatic_retries=0, response_kl_evaluations=0,
        first_group_qualification={'status': 'not_attempted'}, unexecuted_groups=[])
    for group in groups:
        result['groups'].append(dict(group_id=group['group_id'], domain=group['domain'], status='not_attempted',
            requests=copy.deepcopy(group['requests']), costs={}, locations=[
                dict(anchor=copy.deepcopy(anchor), rows={}, validations={}, comparisons={}, attempted_arms=[],
                     status='not_attempted') for request in group['requests'] for anchor in request['anchors']]))
    started = time.monotonic()

    def persist():
        _count(result)
        result['unexecuted_groups'] = [g['group_id'] for g in result['groups'] if g['status'] == 'not_attempted']
        # Rows are sanitized at admission. Let the caller retain full snapshots
        # or write a compact progress index; immutable raw rows are emitted first.
        snapshot(result)

    persist()
    for index, (group, current) in enumerate(zip(groups, result['groups'])):
        tick = time.monotonic()
        current['status'] = 'running'
        backend = None
        cleanup_failed = False
        try:
            # Keep zero-round requests in the plan/report, never fabricate probes.
            if current['locations']:
                backend = backend_factory(group)
                native_group = copy.deepcopy(group)
                native_group['anchors'] = [loc['anchor'] for loc in current['locations']]
                current['startup'] = backend.start(native_group)
                persist()
            # Same sealed prefixes and independently reconstructed caches, but only three checkpoint
            # activations per group rather than switching after every block.
            for arm in ARMS:
                for loc in current['locations']:
                    anchor = loc['anchor']
                    loc['status'] = 'running'
                    loc['attempted_arms'].append(arm)
                    emit('probe_claimed', dict(anchor_id=anchor['anchor_id'], arm=arm))
                    persist()
                    try:
                        row = backend.probe(copy.deepcopy(anchor), arm)
                    except Exception as error:
                        partial = getattr(error, 'diagnostic_partial', None)
                        if partial is not None:
                            loc['rows'][arm] = preserve_invalid_numbers(partial)
                            emit('probe_raw', loc['rows'][arm])
                            loc['validations'][arm] = validate_row(anchor, arm, loc['rows'][arm])
                        loc['error'] = dict(arm=arm, type=type(error).__name__, message=str(error))
                        loc['status'] = 'failed'
                        persist()
                        raise
                    loc['rows'][arm] = preserve_invalid_numbers(row)
                    emit('probe_raw', loc['rows'][arm])
                    persist()  # durable raw evidence precedes every acceptance check
                    loc['validations'][arm] = validate_row(anchor, arm, loc['rows'][arm])
                    persist()
                    require(loc['validations'][arm]['status'] == 'passed',
                            '; '.join(loc['validations'][arm]['errors']))
                    if arm in ADAPTED_ARMS:
                        pair, positions = compare_anchor(anchor, loc['rows']['F'], loc['rows'][arm], arm)
                        pair['positions'] = positions
                        loc['comparisons'][arm] = pair
                        persist()
                        require(pair['valid'], 'Matched-condition comparison invalid: ' + repr(pair.get('invalid_reasons')))
                    if len(loc['rows']) == len(ARMS):
                        loc['status'] = 'complete'
                    persist()
            current['status'] = 'complete'
        except Exception as error:
            current.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
            for loc in current['locations']:
                if loc['status'] == 'running':
                    loc['status'] = 'failed'
        finally:
            if backend is not None:
                try:
                    current['cleanup'] = backend.close()
                    require(current['cleanup']['status'] == 'passed', 'Owned-worker cleanup failed')
                    current['costs'] = current['cleanup'].get('costs', current['cleanup'])
                except Exception as error:
                    cleanup_failed = True
                    current.update(status='failed', cleanup_error=dict(type=type(error).__name__, message=str(error)))
            current['actual_wall_s'] = time.monotonic()-tick
            current['unattempted_arms'] = [dict(anchor_id=loc['anchor']['anchor_id'], arms=[
                arm for arm in ARMS if arm not in loc['attempted_arms']]) for loc in current['locations']
                if len(loc['attempted_arms']) != len(ARMS)]
            persist()
        emit('group_finished', preserve_invalid_numbers(current))
        if index == 0:
            passed = current['status'] == 'complete' and bool(current['locations'])
            result['first_group_qualification'] = dict(status='passed' if passed else 'failed',
                selected_locations=len(current['locations']),
                scope='Correctness only; qualification probes are included in the sealed budget.',
                reason=None if passed else 'First complete group did not supply valid native reconstruction evidence.')
            persist()
            if not passed:
                break
        # Later independent groups remain eligible after an affected group stops,
        # unless worker cleanup failed and independence can no longer be ensured.
        if cleanup_failed:
            break
    _count(result)
    result['status'] = 'complete' if all(g['status'] == 'complete' for g in result['groups']) and \
        result['first_group_qualification']['status'] == 'passed' else 'incomplete'
    if result['status'] == 'complete':
        require(result['actual_draft_block_evaluations'] == result['planned_arm_evaluations'] and
                result['unknown_work_receipts'] == 0, 'Completed native work differs from sealed plan')
        require(result['target_verifications'] == result['reconstruction_target_verifications'] ==
                result['reconstruction_target_verify_attempts'] == result['planned_reconstruction_target_verifications'] and
                result['unknown_reconstruction_work_receipts'] == 0 and result['scored_target_verifications'] == 0,
                'Completed reconstruction work differs from sealed plan')
    result['actual_wall_s'] = time.monotonic()-started
    persist()
    return preserve_invalid_numbers(result)
