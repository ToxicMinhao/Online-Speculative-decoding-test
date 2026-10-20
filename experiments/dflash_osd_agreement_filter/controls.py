"""Predeclared numerical validity of N; no precision or update-rule changes.

The 1% bound is a numerical control-admission limit fixed before GPU outcomes,
not an efficacy threshold. No absolute floor may admit a nonzero intended update
rounded to zero. A zero intended norm requires an exactly zero applied norm.
"""
from __future__ import annotations

import math

N_NORM_POLICY = {
    'version': 'n_actual_norm_match_v1',
    'relative_error_max': 0.01,
    'relative_error': 'abs(actual_norm - intended_norm) / intended_norm',
    'zero_intended_rule': 'actual_norm == 0 exactly',
    'scope': 'Every N optimizer step; all ancestral steps must qualify',
    'failure_action': 'Preserve evidence and stop; no selective-filter attribution',
}


def norm_match(actual_norm, intended_norm):
    good = all(type(x) in (int, float) and math.isfinite(x) and x >= 0
               for x in (actual_norm, intended_norm))
    relative = None
    reason = None
    if not good:
        reason = 'missing_nonfinite_or_negative_norm'
    elif intended_norm == 0:
        if actual_norm != 0:
            reason = 'nonzero_actual_for_zero_intended'
    else:
        relative = abs(actual_norm - intended_norm) / intended_norm
        if not math.isfinite(relative) or relative > N_NORM_POLICY['relative_error_max']:
            reason = 'relative_norm_error_exceeds_limit'
    return dict(status='passed' if reason is None else 'failed', policy=dict(N_NORM_POLICY),
                actual_norm=actual_norm if good else None,
                intended_norm=intended_norm if good else None,
                relative_error=relative if relative is None or math.isfinite(relative) else None,
                error=reason)


def n_update_control(update):
    """Recompute both steps; a cached 'passed' status never establishes validity."""
    event = update.get('event')
    batches = update.get('accounting', {}).get('batches', [])
    errors, steps = [], []
    if type(event) is not int or event not in (1, 2):
        errors.append('invalid_event')
    if len(batches) != 2:
        errors.append('missing_or_extra_step_records')
    for index, batch in enumerate(batches):
        stats = batch.get('agreement', {})
        check = norm_match(stats.get('applied_step_norm'), stats.get('hypothetical_filtered_step_norm'))
        if (type(event) is int and event in (1, 2) and
                (batch.get('pass_index') != index or stats.get('optimizer_step_calls') != 1 or
                 stats.get('optimizer_steps_before') != 2 * (event - 1) + index or
                 stats.get('optimizer_steps_after') != 2 * (event - 1) + index + 1)):
            errors.append(f'step{index + 1}:optimizer_step_identity_mismatch')
        if stats.get('mode') != 'N':
            errors.append(f'step{index + 1}:wrong_arm')
        if stats.get('uniform_control') != check:
            errors.append(f'step{index + 1}:missing_or_inconsistent_control_receipt')
        if check['status'] != 'passed':
            errors.append(f'step{index + 1}:' + check['error'])
        steps.append(dict(pass_index=index, **check))
    return dict(status='passed' if not errors else 'failed', event=event, steps=steps,
                errors=errors, policy=dict(N_NORM_POLICY))


def n_checkpoint_control(result, turn):
    """N at request t depends on every write before t, not just its latest write."""
    errors, checks = [], []
    updates = result.get('arms', {}).get('N', {}).get('updates', [])
    previous_identity = None
    for event in range(1, turn):
        matches = [u for u in updates if u.get('event') == event]
        if len(matches) != 1:
            errors.append(f'event{event}:missing_or_duplicate_update')
            continue
        update = matches[0]
        checkpoint = update.get('checkpoint', {})
        identity = checkpoint.get('checkpoint_identity_sha256')
        if (checkpoint.get('checkpoint_id') != f'D{event}' or
                not isinstance(identity, str) or not identity):
            errors.append(f'event{event}:missing_checkpoint_identity')
        if event > 1 and update.get('accounting', {}).get('capture_checkpoint_sha256') != previous_identity:
            errors.append(f'event{event}:predecessor_checkpoint_mismatch')
        previous_identity = identity
        check = n_update_control(update)
        checks.append(check)
        if check['status'] != 'passed':
            errors.append(f'event{event}:invalid_N_norm_control')
    scored = [row for row in result.get('scores', []) if row.get('arm') == 'N' and row.get('turn') == turn]
    if (len(scored) != 1 or scored[0].get('checkpoint_id') != f'D{turn - 1}' or
            not previous_identity or scored[0].get('checkpoint_identity_sha256') != previous_identity):
        errors.append('evaluated_checkpoint_not_bound_to_update_evidence')
    return dict(status='passed' if not errors and turn in (2, 3) else 'failed',
                through_event=turn - 1, updates=checks, errors=errors, policy=dict(N_NORM_POLICY))
