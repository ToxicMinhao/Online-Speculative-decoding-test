"""Bounded fresh-F/S8 evaluation of sealed new requests; no training path."""
from __future__ import annotations

import copy
import math
import time

from experiments.dflash_baseline_learning_curve.experiment import (
    digest, preserve_invalid_numbers, require, validate_measurement, _all_finite)
from .evidence import validate_group
from .protocol import PLANNED, SERVING_CONTRACT, EXPECTED_PARENT_SUFFIXES

ARMS = ('F', 'S8')
ROLE = 'related_instance'
IDENTITY = ('request_id', 'parent_request_id', 'group_id', 'domain', 'role')


def validate_read_only_receipt(receipt):
    require(isinstance(receipt, dict) and receipt.get('status') == 'passed' and _all_finite(receipt),
            'Incomplete/nonfinite read-only runtime receipt')
    for key in ('optimizer_steps', 'training_presentations', 'response_kl_evaluations', 'new_checkpoints',
                'checkpoint_exports', 'explicit_warmups', 'explicit_generation_warmups', 'warmup_generations', 'retries'):
        if key in receipt:
            require(type(receipt[key]) is int and receipt[key] == 0, 'Forbidden extra work: ' + key)
    for key in ('optimizer_created', 'optimizer_loaded'):
        if key in receipt:
            require(receipt[key] is False, 'Read-only execution created/loaded an optimizer')


def validate_native(row, child, checkpoint_bindings, label):
    try:
        require(isinstance(row, dict) and _all_finite(row), 'Nonfinite or malformed native evidence')
        validate_measurement(row)
        require(label in ARMS and row['checkpoint_id'] == label and
                row['checkpoint_identity_sha256'] == checkpoint_bindings[label], 'Wrong checkpoint identity')
        require(child['role'] == ROLE and all(row.get(k) == child.get(k) for k in IDENTITY),
                'Child/parent/group/domain/role identity differs')
        require(row['input_ids'] == child['input_ids'] and row['input_ids_sha256'] == child['input_ids_sha256']
                == digest(child['input_ids']), 'Sealed child prompt differs')
        require(row.get('serving_contract') == SERVING_CONTRACT and
                row.get('feedback_capture_enabled') is False, 'Serving contract or capture differs')
        require(type(row.get('eos_hit')) is bool, 'Missing exact EOS status')
        return dict(status='passed', errors=[])
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return dict(status='failed', errors=[str(error)])


def validate_pair(pair):
    try:
        child = pair['child']
        require(all(pair.get(k) == child.get(k) for k in IDENTITY), 'Pair differs from sealed child identity')
        require(pair.get('canonical_reference') == 'fresh_F', 'Fresh F must supply the canonical output')
        require(set(pair['checkpoint_bindings']) == set(ARMS), 'Exactly F and own S8 are required')
        for label in ARMS:
            checked = validate_native(pair[label]['native'], child, pair['checkpoint_bindings'], label)
            require(checked['status'] == 'passed', '; '.join(checked['errors']))
        for key in ('input_ids', 'output_ids', 'finish_reason', 'cap_hit', 'eos_hit'):
            require(pair['S8']['native'][key] == pair['F']['native'][key], 'Fresh F/S8 differs: ' + key)
        return dict(status='passed', errors=[])
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return dict(status='failed', errors=[str(error)])


def repeat_flags(child, frozen, feedback_answers):
    """Postgeneration token equality only; this does not establish semantic novelty."""
    require(len(feedback_answers) == 4 and len({r['request_id'] for r in feedback_answers}) == 4,
            'Exactly four distinct original feedback answers required')
    for row in feedback_answers:
        require(row['group_id'] == child['group_id'] and row['domain'] == child['domain'] and
                row['output_ids_sha256'] == digest(row['output_ids']), 'Feedback answer binding differs')
    parent = next((r for r in feedback_answers if r['request_id'] == child['parent_request_id']), None)
    require(parent is not None, 'Parent answer is absent from own-group feedback')
    matches = [r['request_id'] for r in feedback_answers if frozen['output_ids'] == r['output_ids']]
    return dict(matches_parent_answer=frozen['output_ids'] == parent['output_ids'],
                matches_any_feedback_answer=bool(matches), matching_feedback_request_ids=matches,
                non_exact_repeat=not matches, child_F_output_ids_sha256=digest(frozen['output_ids']),
                parent_output_ids_sha256=parent['output_ids_sha256'],
                feedback_output_ids_sha256={r['request_id']: r['output_ids_sha256'] for r in feedback_answers},
                interpretation='Exact token repetition only; token inequality does not prove semantic novelty.')


def _context_row(group, population, request_id, frozen, learned):
    row = dict(group_id=group['group_id'], domain=group['domain'], population=population,
               request_id=request_id, origin='retained_PR9_PR10', execution='not_rerun',
               matched_causal_comparison_to_children=False, available=learned is not None)
    for label, native in (('F', frozen), ('S8', learned)):
        row[label + '_native_sha256'] = digest(native) if native else None
        row[label + '_checkpoint_identity_sha256'] = native.get('checkpoint_identity_sha256') if native else None
        for key in ('verification_calls', 'accepted_draft_tokens', 'verification_committed_tokens', 'output_tokens'):
            row[label + '_' + key] = native.get(key) if native else None
    row['S8_vs_F_verification_calls_saved'] = (frozen['verification_calls'] - learned['verification_calls']) if learned else None
    if learned is None:
        row['unavailable_reason'] = 'Retained S8 parent-source native generation was not supplied; KL is not a native-call substitute.'
    return row


def retained_context(group):
    """Descriptive retained measurements only; called after new generation ends."""
    contexts = []
    refs = group.get('reference_pairs', {})
    feedback = {r['request_id']: r for r in group['feedback']}
    for child in group['children']:
        rid = child['parent_request_id']
        prior = next((p for p in refs.get('source', []) if p['request_id'] == rid), None)
        contexts.append(_context_row(group, 'parent_source', rid,
            prior['F']['native'] if prior else feedback[rid], prior['S8']['native'] if prior else None))
    if refs.get('heldout'):
        contexts.extend(_context_row(group, 'ordinary_heldout', p['request_id'], p['F']['native'], p['S8']['native'])
                        for p in refs['heldout'])
    else:
        scores = {(r['request_id'], r['checkpoint_id']): r for r in group.get('scores', [])}
        for request in group.get('evaluation_requests', []):
            rid = request['request_id']
            if (rid, 'F') in scores:
                contexts.append(_context_row(group, 'ordinary_heldout', rid, scores[rid, 'F'], scores.get((rid, 'S8'))))
    return contexts


def accounting(result):
    attempts = [a for g in result['groups'] for a in g['native_attempts'] if a.get('dispatched')]
    statuses = [a['status'] for a in attempts]
    require(all(s in ('attempted', 'unknown', 'returned_valid', 'returned_invalid', 'returned_unvalidated')
                for s in statuses), 'Unknown native attempt status')
    returned = sum(s.startswith('returned_') for s in statuses)
    counts = dict(attempted_native_generations=len(attempts), actual_native_generations=returned,
                  returned_native_generations=returned,
                  returned_valid_native_generations=statuses.count('returned_valid'),
                  returned_invalid_native_generations=statuses.count('returned_invalid'),
                  returned_unvalidated_native_generations=statuses.count('returned_unvalidated'),
                  unknown_native_work_receipts=sum(s in ('attempted', 'unknown') for s in statuses),
                  unattempted_native_generations=24 - len(attempts), optimizer_steps=0,
                  training_presentations=0, actual_response_kl_evaluations=0, new_checkpoints=0,
                  explicit_warmups=0, retries=0)
    require(counts['unattempted_native_generations'] >= 0 and
            counts['attempted_native_generations'] == returned + counts['unknown_native_work_receipts'],
            'Native work budget does not reconcile')
    return counts


def _safe_sum(rows, field):
    values = [r.get(field) if isinstance(r, dict) else None for r in rows]
    valid = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    return dict(sum_s=sum(valid), recorded_values=len(valid), missing_or_invalid_values=len(values) - len(valid))


def run_campaign(groups, backend_factory, *, emit=lambda *args: None, snapshot=lambda value: None):
    require(len(groups) == 6 and len({g['group_id'] for g in groups}) == 6, 'Exactly six original groups required')
    require([g['group_id'] for g in groups] == list(EXPECTED_PARENT_SUFFIXES), 'Original manifest group order differs')
    for group in groups:
        validate_group(group)
    require(len({c['request_id'] for g in groups for c in g['children']}) == 12, 'Twelve unique sealed children required')
    result = dict(status='running', planned=copy.deepcopy(PLANNED), groups=[],
                  selected_groups=[dict(group_id=g['group_id'], domain=g['domain'],
                      children=copy.deepcopy(g['children']), group_payload_sha256=g['group_payload_sha256'],
                      sealed_manifest_sha256=g['sealed_manifest_sha256'],
                      checkpoint_bindings={k: g['checkpoints'][k]['checkpoint_identity_sha256'] for k in ARMS}) for g in groups],
                  unexecuted_groups=[], persistence_errors=[])
    started = time.monotonic()

    def persist(*, final=False):
        result.update(accounting(result))
        try:
            snapshot(preserve_invalid_numbers(copy.deepcopy(result)))
        except Exception as error:
            result['persistence_errors'].append(dict(type=type(error).__name__, message=str(error)))
            result['status'] = 'failed'
            if not final:
                raise

    persist()
    for group in groups:
        tick = time.monotonic()
        current = dict(group_id=group['group_id'], domain=group['domain'], status='running',
                       group_payload_sha256=group['group_payload_sha256'], children=copy.deepcopy(group['children']),
                       sealed_manifest_sha256=group['sealed_manifest_sha256'],
                       checkpoint_bindings={k: group['checkpoints'][k]['checkpoint_identity_sha256'] for k in ARMS},
                       native_raw=[], native_attempts=[], pairs=[], feedback_answers=[], historical_context=[])
        result['groups'].append(current)
        backend, active = None, None
        pairs = {c['request_id']: dict(**{k: c[k] for k in IDENTITY}, child=copy.deepcopy(c),
                    checkpoint_bindings=copy.deepcopy(current['checkpoint_bindings']), canonical_reference='fresh_F')
                 for c in group['children']}
        current['pairs'] = list(pairs.values())
        try:
            backend = backend_factory(copy.deepcopy(group))
            current['startup'] = backend.start_group(copy.deepcopy(group))
            persist()
            validate_read_only_receipt(current['startup'])
            for label in ARMS:
                for child in group['children']:
                    active = dict(ordinal=len(current['native_attempts']), request_id=child['request_id'],
                                  checkpoint_id=label, dispatched=False, status='scheduled')
                    current['native_attempts'].append(active)
                    persist()
                    active.update(dispatched=True, status='attempted')
                    try:
                        persist()
                    except Exception:
                        active.update(dispatched=False, status='unattempted')
                        raise
                    try:
                        native = backend.serve_request(copy.deepcopy(child), label)
                    except Exception as error:
                        partial = getattr(error, 'native_partial', None)
                        if isinstance(partial, dict) and partial.get('generation_returned') is True:
                            raw = copy.deepcopy(partial.get('raw_receipt', partial))
                            active.update(status='returned_invalid', raw_index=len(current['native_raw']),
                                          failure_detail=copy.deepcopy(partial),
                                          validation=dict(status='failed', errors=[str(error)]))
                            current['native_raw'].append(raw)
                            emit('native_raw', preserve_invalid_numbers(raw))
                        else:
                            active.update(status='unknown', error=dict(type=type(error).__name__, message=str(error)))
                        persist()
                        raise
                    active.update(status='returned_unvalidated', raw_index=len(current['native_raw']))
                    current['native_raw'].append(copy.deepcopy(native))
                    pairs[child['request_id']][label] = dict(native=copy.deepcopy(native))
                    persist()
                    emit('native_raw', preserve_invalid_numbers(copy.deepcopy(native)))
                    active['validation'] = validate_native(native, child, current['checkpoint_bindings'], label)
                    active['status'] = 'returned_valid' if active['validation']['status'] == 'passed' else 'returned_invalid'
                    persist()
                    require(active['validation']['status'] == 'passed', '; '.join(active['validation']['errors']))
                    if label == 'S8':
                        pair = pairs[child['request_id']]
                        pair['validation'] = validate_pair(pair)
                        if pair['validation']['status'] != 'passed':
                            active.update(status='returned_invalid', validation=copy.deepcopy(pair['validation']))
                        persist()
                        require(pair['validation']['status'] == 'passed', '; '.join(pair['validation']['errors']))
            current['status'] = 'complete'
        except Exception as error:
            current.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
            if active and active['status'] == 'returned_unvalidated':
                active.update(status='returned_invalid', validation=dict(status='failed', errors=['Raw persistence/validation incomplete']))
            if getattr(error, 'native_start', None) is not None:
                current['startup_failure'] = preserve_invalid_numbers(error.native_start)
        finally:
            if backend is not None:
                try:
                    current['cleanup'] = backend.finish_group()
                    validate_read_only_receipt(current['cleanup'])
                    require(current['cleanup'].get('status') == 'passed' and
                            current['cleanup'].get('checkpoint_weights_unchanged') is True and
                            current['cleanup'].get('training_quiescent') is True and _all_finite(current['cleanup']),
                            'Checkpoint immutability/quiescence/cleanup failed')
                except Exception as error:
                    current.update(status='failed', cleanup_error=dict(type=type(error).__name__, message=str(error)))
                    if getattr(error, 'native_finish', None) is not None:
                        current['cleanup'] = preserve_invalid_numbers(error.native_finish)
            try:
                # Sealed construction and all generation decisions are complete before answers are examined.
                current['feedback_answers'] = [dict(request_id=r['request_id'], group_id=group['group_id'],
                    domain=group['domain'], output_ids=copy.deepcopy(r['output_ids']),
                    output_ids_sha256=digest(r['output_ids'])) for r in group['feedback']]
                for pair in current['pairs']:
                    if 'F' in pair:
                        pair['repetition'] = repeat_flags(pair['child'], pair['F']['native'], current['feedback_answers'])
                current['historical_context'] = retained_context(group)
            except Exception as error:
                current.update(status='failed', postgeneration_error=dict(type=type(error).__name__, message=str(error)))
            current['actual_wall_s'] = time.monotonic() - tick
            current['costs'] = dict(actual_group_wall_s=current['actual_wall_s'],
                fresh_request_timing=_safe_sum(current['native_raw'], 'request_s'),
                cache_maintenance_timing=_safe_sum(current['native_raw'], 'cache_maintenance_s'),
                per_request_scopes=[dict(request_id=r.get('request_id'), checkpoint_id=r.get('checkpoint_id'),
                    reported_fields={k: copy.deepcopy(v) for k, v in r.items() if k.endswith('_s') or
                        'timing' in k or 'initialization' in k or 'prefill' in k or k == 'costs'})
                    for r in current['native_raw'] if isinstance(r, dict)],
                startup=copy.deepcopy(current.get('startup')), cleanup=copy.deepcopy(current.get('cleanup')),
                accounting='Raw scopes retained: request time includes prefill/drafting/verification. Loading, '
                'activation and cache work may contain nested timers; do not add overlapping scopes. '
                'One fixed F-then-S8 execution per request does not establish speedup.')
            cleanup = current.get('cleanup') or {}
            integrity_valid = (not current.get('cleanup_error') and cleanup.get('status') == 'passed' and
                cleanup.get('checkpoint_weights_unchanged') is True and cleanup.get('training_quiescent') is True
                and _all_finite(cleanup))
            for pair in current['pairs']:
                # A later independent generation failure does not erase an earlier
                # complete valid pair when cleanup proves checkpoint integrity.
                pair['group_execution_valid'] = bool(integrity_valid)
            persist(final=True)
        try:
            emit('group_finished', preserve_invalid_numbers(copy.deepcopy(current)))
        except Exception as error:
            current.update(status='failed', event_error=dict(type=type(error).__name__, message=str(error)))
        if current['status'] != 'complete' or result['persistence_errors']:
            result['status'] = 'failed'
            break
        if len(result['groups']) == 1:
            result['first_group_qualification'] = dict(status='passed', native_generations=4, children=2,
                cleanup_passed=True, checkpoint_weights_unchanged=True,
                scope='Correctness only, inside the fixed budget; null/harmful outcomes continue unchanged.')
    if len(result['groups']) == 6 and all(g['status'] == 'complete' for g in result['groups']) and not result['persistence_errors']:
        result['status'] = 'complete'
    result['unexecuted_groups'] = [g['group_id'] for g in groups[len(result['groups']):]]
    result['actual_wall_s'] = time.monotonic() - started
    persist(final=True)
    return preserve_invalid_numbers(result)
