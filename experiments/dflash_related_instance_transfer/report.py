"""Summed paired counts with explicit selected and non-exact-repeat denominators."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

from experiments.dflash_baseline_learning_curve.experiment import preserve_invalid_numbers, _all_finite
from .experiment import ARMS, IDENTITY, accounting, repeat_flags, validate_pair, validate_read_only_receipt
from .protocol import PLANNED, EXPECTED_PARENT_SUFFIXES

BOUNDARY = (
    'Twelve constructed requests and six retained S8 adaptation trajectories test transfer in this '
    'constructed setting. Exact output-token inequality does not prove semantic novelty or exclude partial '
    'memorization; novelty depends on the sealed prompt-pair review. Target-answer correctness is not the '
    'speculative-decoding metric. Parent-source and ordinary held-out measurements are retained descriptive '
    'context, not matched causal comparisons to children. One fixed-order F-then-S8 execution per request '
    'does not establish speedup. These results do not identify the cause of prior regressions, establish '
    'broad generalization or real conversation benefit, or justify automatic retuning/expansion. Any later '
    'selected method requires fresh confirmation.'
)
COUNTERS = ('verification_calls', 'accepted_draft_tokens', 'verification_committed_tokens', 'output_tokens')


def _ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if numerator is not None and denominator else None


def _group_errors(group, selected):
    errors = []
    for key in ('group_id', 'domain', 'children', 'group_payload_sha256', 'sealed_manifest_sha256', 'checkpoint_bindings'):
        if group.get(key) != selected.get(key):
            errors.append('group_binding_differs:' + key)
    cleanup = group.get('cleanup') or {}
    try:
        validate_read_only_receipt(cleanup)
        validate_read_only_receipt(group.get('startup'))
    except (ValueError, TypeError, KeyError) as error:
        errors.append('read_only_receipt_failed:' + str(error))
    if (cleanup.get('status') != 'passed' or cleanup.get('checkpoint_weights_unchanged') is not True or
            cleanup.get('training_quiescent') is not True or not _all_finite(cleanup)):
        errors.append('cleanup_or_checkpoint_immutability_not_verified')
    expected = [(arm, c['request_id']) for arm in ARMS for c in selected['children']]
    attempts = group.get('native_attempts', [])
    if len(attempts) > 4 or any((a.get('checkpoint_id'), a.get('request_id')) != expected[i] or
            a.get('ordinal') != i for i, a in enumerate(attempts[:4])):
        errors.append('generation_order_or_identity_differs')
    returned = [a for a in attempts if a.get('dispatched') and str(a.get('status', '')).startswith('returned_')]
    if [a.get('raw_index') for a in returned] != list(range(len(group.get('native_raw', [])))):
        errors.append('raw_receipts_not_bound_once_to_returned_calls')
    return errors


def comparison(pair, child, group, selected, *, extra_errors=()):
    errors = list(extra_errors)
    pair = pair or {}
    checked = validate_pair(pair)
    errors.extend(checked['errors'])
    if pair.get('validation') != dict(status='passed', errors=[]):
        errors.append('recorded_pair_validation_failed_or_missing')
    if pair.get('group_execution_valid') is not True:
        errors.append('group_execution_not_valid')
    if pair.get('child') != child or pair.get('checkpoint_bindings') != selected['checkpoint_bindings']:
        errors.append('pair_differs_from_selected_child_or_checkpoints')
    repetition = None
    try:
        repetition = repeat_flags(child, pair['F']['native'], group['feedback_answers'])
        if pair.get('repetition') != repetition:
            errors.append('postgeneration_repeat_flags_differ')
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        errors.append('repeat_analysis_incomplete:' + str(error))
    attempted = any(a.get('request_id') == child['request_id'] and a.get('dispatched')
                    for a in group.get('native_attempts', []))
    for label in ARMS:
        matches = [a for a in group.get('native_attempts', []) if a.get('request_id') == child['request_id']
                   and a.get('checkpoint_id') == label and a.get('dispatched')]
        if len(matches) != 1 or matches[0].get('status') != 'returned_valid':
            errors.append('native_attempt_missing_invalid_or_unknown:' + label)
        else:
            try:
                if group['native_raw'][matches[0]['raw_index']] != pair[label]['native']:
                    errors.append('paired_native_differs_from_durable_raw:' + label)
            except (KeyError, IndexError, TypeError):
                errors.append('raw_binding_missing:' + label)
    row = {key: child[key] for key in IDENTITY}
    row.update(valid=not errors, invalid_reasons=sorted(set(errors)), attempted=attempted,
               pair_returned=all(label in pair for label in ARMS),
               non_exact_repeat=repetition['non_exact_repeat'] if repetition else None,
               matches_parent_answer=repetition['matches_parent_answer'] if repetition else None,
               matches_any_feedback_answer=repetition['matches_any_feedback_answer'] if repetition else None,
               matching_feedback_request_ids=repetition['matching_feedback_request_ids'] if repetition else None,
               construction={key: child.get(key) for key in ('text', 'construction_method', 'changes_made',
                   'why_required_answer_differs', 'semantic_review')},
               sealed_input_ids_sha256=child.get('input_ids_sha256'))
    for label in ARMS:
        native = (pair.get(label) or {}).get('native') or {}
        if not isinstance(native, dict):
            native = {}
        for key in COUNTERS + ('request_s', 'native_generation_s', 'cache_maintenance_s', 'prefill_accounting',
                              'timing_scope', 'initialization_policy', 'cap_hit', 'eos_hit', 'finish_reason',
                              'checkpoint_identity_sha256'):
            row[label + '_' + key] = native.get(key)
        row[label + '_calls_per_1000_output_tokens'] = _ratio(native.get('verification_calls'), native.get('output_tokens'), 1000) if row['valid'] else None
    saved = row['F_verification_calls'] - row['S8_verification_calls'] if row['valid'] else None
    row.update(S8_vs_F_verification_calls_saved=saved,
               S8_vs_F_verification_call_reduction_percent=_ratio(saved, row['F_verification_calls'], 100) if row['valid'] else None,
               S8_vs_F_native_outcome='invalid' if saved is None else 'better' if saved > 0 else 'worse' if saved < 0 else 'tied')
    return row


def pool(rows, subset='all_valid'):
    selected = list(rows)
    included = selected if subset == 'all_valid' else [r for r in selected if r['non_exact_repeat'] is True]
    valid = [r for r in included if r['valid']]
    row = dict(subset=subset, selected_requests=len(selected), subset_requests=len(included),
               attempted_requests=sum(r['attempted'] for r in included), valid_requests=len(valid),
               invalid_or_incomplete_requests=sum(r['attempted'] and not r['valid'] for r in included),
               unattempted_requests=sum(not r['attempted'] for r in included),
               repeat_unclassified_requests=sum(r['non_exact_repeat'] is None for r in selected),
               valid_non_exact_repeat_requests=sum(r['valid'] and r['non_exact_repeat'] is True for r in selected),
               valid_exact_repeat_requests=sum(r['valid'] and r['non_exact_repeat'] is False for r in selected))
    for label in ARMS:
        for key in COUNTERS:
            row[label + '_' + key] = sum(r[label + '_' + key] for r in valid) if valid else None
        calls, tokens = row[label + '_verification_calls'], row[label + '_output_tokens']
        row[label + '_calls_per_1000_output_tokens'] = _ratio(calls, tokens, 1000)
        row[label + '_accepted_draft_tokens_per_call'] = _ratio(row[label + '_accepted_draft_tokens'], calls)
        row[label + '_committed_progress_per_call'] = _ratio(row[label + '_verification_committed_tokens'], calls)
        row[label + '_emitted_tokens_per_call'] = _ratio(tokens, calls)
    saved = row['F_verification_calls'] - row['S8_verification_calls'] if valid else None
    row['S8_vs_F_verification_calls_saved'] = saved
    row['S8_vs_F_verification_call_reduction_percent'] = _ratio(saved, row['F_verification_calls'], 100)
    for outcome in ('better', 'worse', 'tied'):
        row['S8_vs_F_native_' + outcome + '_requests'] = sum(r['S8_vs_F_native_outcome'] == outcome for r in valid)
    return row


def _interpret(primary, nonrepeat, *, execution_complete=True):
    if not execution_complete or primary['valid_requests'] != primary['selected_requests']:
        return ['Evidence is incomplete or invalid. Valid-subset counts remain descriptive; no complete-setting transfer conclusion.']
    if primary['S8_vs_F_verification_calls_saved'] > 0:
        if nonrepeat['valid_requests'] and nonrepeat['S8_vs_F_verification_calls_saved'] > 0:
            return ['S8 uses fewer calls on these prevalidated new instances, including the non-exact-repeat subset. '
                    'This supports transfer in this constructed setting and does not exclude partial memorization.']
        return ['The all-valid aggregate improves, but an improvement on the non-exact-repeat subset is not established. '
                'Gains confined to repeated answers do not establish that transfer.']
    return ['S8 has no aggregate call benefit on these related new instances. Retained parent gains, where present, '
            'would indicate a narrower benefit than this control requires; this does not identify a cause or prove all online adaptation ineffective.']


def summarize(result):
    selected = result.get('selected_groups', [])
    groups = result.get('groups', [])
    group_counts = Counter(g.get('group_id') for g in groups)
    selected_counts = Counter(c['request_id'] for g in selected for c in g['children'])
    comparisons = []
    for owner in selected:
        group = next((g for g in groups if g.get('group_id') == owner['group_id']), {})
        shared_errors = _group_errors(group, owner)
        if group_counts[owner['group_id']] > 1:
            shared_errors.append('duplicate_executed_group')
        for child in owner['children']:
            matches = [p for p in group.get('pairs', []) if p.get('request_id') == child['request_id']]
            errors = list(shared_errors)
            if len(matches) != 1:
                errors.append('missing_or_duplicate_selected_pair')
            if selected_counts[child['request_id']] != 1:
                errors.append('duplicate_selected_child')
            comparisons.append(comparison(matches[0] if matches else None, child, group, owner, extra_errors=errors))
    scopes = [('overall', 'all', comparisons)]
    scopes += [('domain', domain, [r for r in comparisons if r['domain'] == domain])
               for domain in dict.fromkeys(g['domain'] for g in selected)]
    scopes += [('group', g['group_id'], [r for r in comparisons if r['group_id'] == g['group_id']]) for g in selected]
    pooled = [dict(scope=scope, scope_id=scope_id, **pool(rows, subset))
              for scope, scope_id, rows in scopes for subset in ('all_valid', 'non_exact_repeat')]
    accounting_errors = []
    try:
        actual = accounting(result)
        accounting_errors.extend(key + '_differs_from_attempt_receipts' for key, value in actual.items() if result.get(key) != value)
    except (ValueError, KeyError, TypeError) as error:
        actual = {}
        accounting_errors.append('attempt_accounting_failed:' + str(error))
    if result.get('planned') != PLANNED:
        accounting_errors.append('planned_scope_differs')
    if len(selected) != 6 or len(comparisons) != 12 or len(selected_counts) != 12:
        accounting_errors.append('selected_scope_is_not_six_groups_twelve_unique_children')
    if [g.get('group_id') for g in selected] != list(EXPECTED_PARENT_SUFFIXES):
        accounting_errors.append('selected_manifest_group_order_differs')
    if [g.get('group_id') for g in groups] != [g.get('group_id') for g in selected[:len(groups)]]:
        accounting_errors.append('executed_group_order_differs')
    unknown_groups = [g.get('group_id') for g in groups if g.get('group_id') not in {o['group_id'] for o in selected}]
    if unknown_groups:
        accounting_errors.append('unselected_executed_group')
    for group in groups:
        known = {c['request_id'] for o in selected if o['group_id'] == group.get('group_id') for c in o['children']}
        if any(p.get('request_id') not in known for p in group.get('pairs', [])):
            accounting_errors.append('unselected_pair')
    if result.get('status') == 'complete' and (actual.get('returned_valid_native_generations') != 24 or
            actual.get('unknown_native_work_receipts') != 0 or actual.get('unattempted_native_generations') != 0):
        accounting_errors.append('completed_scope_does_not_reconcile')
    complete = (result.get('status') == 'complete' and not accounting_errors and len(groups) == 6
                and len(comparisons) == 12 and all(r['valid'] for r in comparisons)
                and result.get('first_group_qualification', {}).get('status') == 'passed'
                and not result.get('persistence_errors'))
    interpretations = [dict(scope=scope, scope_id=scope_id,
        patterns=_interpret(pool(rows), pool(rows, 'non_exact_repeat'), execution_complete=complete))
        for scope, scope_id, rows in scopes]
    costs = [dict(group_id=g['group_id'], domain=g['domain'], status=g['status'],
                  components=g.get('costs'), startup=g.get('startup'), cleanup=g.get('cleanup')) for g in groups]
    context = [r for g in groups for r in g.get('historical_context', [])]
    context_pooled = []
    for scope, scope_id, _ in scopes:
        scoped = [r for r in context if scope == 'overall' or r.get('domain' if scope == 'domain' else 'group_id') == scope_id]
        for population in ('parent_source', 'ordinary_heldout'):
            rows = [r for r in scoped if r.get('population') == population]
            available = [r for r in rows if r.get('available') is True]
            f = sum(r['F_verification_calls'] for r in available) if available else None
            s = sum(r['S8_verification_calls'] for r in available) if available else None
            context_pooled.append(dict(scope=scope, scope_id=scope_id, population=population,
                recorded_requests=len(rows), available_pairs=len(available), F_verification_calls=f,
                S8_verification_calls=s, S8_vs_F_verification_calls_saved=f-s if available else None,
                S8_vs_F_verification_call_reduction_percent=_ratio(f-s, f, 100) if available else None,
                use='Retained descriptive context only; different request sets are not matched causal comparisons.'))
    return preserve_invalid_numbers(dict(status='complete' if complete else 'partial_or_failed' if groups else 'not_executed',
        execution_status=result.get('status'), planned=result.get('planned'), accounting=actual,
        accounting_errors=sorted(set(accounting_errors)), selected_requests=len(comparisons),
        valid_requests=sum(r['valid'] for r in comparisons), comparisons=comparisons, pooled=pooled,
        historical_context=context, historical_context_pooled=context_pooled, costs=costs,
        actual_wall_s=result.get('actual_wall_s'), interpretations=interpretations, interpretation_boundary=BOUNDARY,
        unexecuted_groups=result.get('unexecuted_groups'), first_group_qualification=result.get('first_group_qualification'),
        selected_groups=selected, provenance=dict(
            source_manifest=result.get('source_manifest'), request_manifest=result.get('request_manifest'),
            plan_sha256=result.get('plan_sha256'), additional=result.get('bindings', result.get('provenance', {}))),
        groups=[{k: g.get(k) for k in ('group_id', 'domain', 'status', 'error', 'cleanup_error', 'postgeneration_error')} for g in groups],
        raw_evidence='RESULT.json, raw native receipts and trajectories retain exact output IDs, per-round offsets, validation failures and costs.'))


def _csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                         for k, v in row.items()} for row in rows)


def write_reports(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(result)
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    for name, rows in (('PAIRED_REQUESTS.csv', summary['comparisons']), ('POOLED.csv', summary['pooled']),
                       ('RETAINED_CONTEXT.csv', summary['historical_context']),
                       ('RETAINED_CONTEXT_POOLED.csv', summary['historical_context_pooled']), ('COSTS.csv', summary['costs'])):
        _csv(out / name, rows)
    lines = ['# DFlash related-instance transfer', '', 'Status: **' + summary['status'] + '**.', '', BOUNDARY, '',
             'All valid children remain in the primary results. The non-exact-repeat subset matches none of the four '
             'original feedback answers in exact F output tokens. No case is replaced after generation.', '',
             '| Scope | Subset | Valid / subset / selected | F calls | S8 calls | Calls saved | Better / worse / tied |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in summary['pooled']:
        counts = '/'.join(str(row['S8_vs_F_native_' + name + '_requests']) for name in ('better', 'worse', 'tied'))
        values = [str(row[key]) if row[key] is not None else '—'
                  for key in ('F_verification_calls', 'S8_verification_calls', 'S8_vs_F_verification_calls_saved')]
        lines.append('| %s:%s | %s | %s/%s/%s | %s | %s |' % (row['scope'], row['scope_id'], row['subset'],
            row['valid_requests'], row['subset_requests'], row['selected_requests'], ' | '.join(values), counts))
    lines += ['', 'Call changes use sums over identical-output valid pairs, never a mean of request percentages. '
              'Invalid and incomplete rows remain visible and contribute no gains.', '',
              '## Work and costs', '', json.dumps(summary['accounting'], sort_keys=True), '',
              'The budget is 24 natural generations, zero training, KL evaluations, new checkpoints, warmups or retries. '
              'Runtime initialization is allowed and is separate from generation. COSTS.csv preserves raw timer scopes; '
              'nested loading/activation/cache timers must not be added blindly. No serving-speedup estimate is calculated.', '',
              '## Retained context and interpretation', '',
              'RETAINED_CONTEXT.csv and RETAINED_CONTEXT_POOLED.csv contain parent-source and ordinary held-out '
              'measurements without rerunning them. They are separate populations and descriptive context only.', '']
    for row in summary['interpretations']:
        lines.append('- ' + row['scope'] + ':' + row['scope_id'] + ': ' + ' '.join(row['patterns']))
    if summary['accounting_errors']:
        lines += ['', 'Accounting errors: ' + '; '.join(summary['accounting_errors'])]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary
