"""Pure-CPU, recorded-metric analysis of all eight fixed GSM8K FSG windows."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import time

COMMIT = '6440e8ff5af1a845f830d65c884f6711d7e4a687'
WINDOWS = (1, 32, 65, 128)
CANDIDATES = ('prestate', 'native', 'raw_fsg', 'feasible_fsg', 'committed', 'scalar',
              'uniform_raw', 'uniform', 'expanded_raw', 'expanded')
COUNTS = ('valid_occurrences', 'protected_occurrences', 'tie_occurrences', 'wrong_ties',
    'blocks', 'old_matching_prefix_tokens', 'new_matching_prefix_tokens',
    'net_matching_prefix_change', 'old_first_errors', 'first_errors_repaired_with_prefix_intact',
    'new_earlier_errors', 'old_prefix_tokens_lost', 'original_first_errors', 'K_useful_repairs',
    'C_useful_repairs', 'useful_repairs_retained', 'useful_repairs_lost', 'useful_repairs_newly_obtained',
    'gained_prefix_occurrences', 'native_gained_prefix_occurrences', 'native_prefix_gains_retained',
    'native_prefix_gains_lost', 'prefix_gains_beyond_native', 'canonical_margin_increases',
    'canonical_margin_decreases', 'canonical_margin_ties')
CONTRASTS = (('feasible_fsg', 'native'), ('committed', 'native'), ('feasible_fsg', 'scalar'),
             ('feasible_fsg', 'raw_fsg'), ('uniform', 'feasible_fsg'), ('expanded', 'feasible_fsg'))


class Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}

    def read(self, path):
        full = self.root / path
        data = full.read_bytes()
        self.files[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return json.loads(data)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_response(row, before, native):
    for name in COUNTS:
        require(type(row[name]) is int, 'Noninteger occurrence: ' + name)
        require(name == 'net_matching_prefix_change' or row[name] >= 0, 'Negative occurrence: ' + name)
    require(row['old_matching_prefix_tokens'] == row['protected_occurrences'], 'Protected support differs')
    require(row['new_matching_prefix_tokens']-row['old_matching_prefix_tokens'] == row['net_matching_prefix_change'], 'Net prefix equation differs')
    require(row['gained_prefix_occurrences']-row['old_prefix_tokens_lost'] == row['net_matching_prefix_change'], 'Gain/loss equation differs')
    require(row['C_useful_repairs'] == row['first_errors_repaired_with_prefix_intact'] ==
            row['useful_repairs_retained']+row['useful_repairs_newly_obtained'], 'Candidate repairs differ')
    require(row['K_useful_repairs'] == row['useful_repairs_retained']+row['useful_repairs_lost'] ==
            native['first_errors_repaired_with_prefix_intact'], 'Native repair identity accounting differs')
    require(row['original_first_errors'] == row['old_first_errors'], 'First-error denominator differs')
    require(row['native_gained_prefix_occurrences'] == row['native_prefix_gains_retained']+
            row['native_prefix_gains_lost'], 'Native gained-prefix accounting differs')
    require(row['gained_prefix_occurrences'] == row['native_prefix_gains_retained']+
            row['prefix_gains_beyond_native'], 'Candidate gained-prefix accounting differs')
    require(row['wrong_ties'] <= row['tie_occurrences'] <= row['valid_occurrences'], 'Tie accounting differs')
    for name in ('blocks', 'old_first_errors', 'protected_occurrences', 'valid_occurrences', 'old_matching_prefix_tokens'):
        require(row[name] == before[name], 'Fixed support differs: ' + name)
    for role in ('prefix', 'first_error'):
        teacher = row['teacher_kl'][role]
        reference = before['teacher_kl'][role]
        require(teacher['denominator'] == reference['denominator'] and teacher['occurrences'] == reference['occurrences'], 'Teacher support differs')
        require(math.isfinite(teacher['numerator']) and teacher['denominator'] >= 0, 'Teacher values are nonfinite')
        expected = teacher['numerator']/teacher['denominator'] if teacher['denominator'] else None
        require(teacher['mean'] == expected, 'Saved teacher mean differs')
    require(row['teacher_kl']['prefix']['occurrences'] == row['protected_occurrences'], 'Teacher prefix occurrence mismatch')
    require(row['teacher_kl']['first_error']['occurrences'] == row['old_first_errors'], 'Teacher first-error occurrence mismatch')
    require(math.isfinite(row['risk_unaveraged']), 'Nonfinite protected KL')


def expected_binding(records, event, window):
    start = (window-1)*2
    pass_index = (window-1)//64
    other = pass_index*128 + (((window-1)%64+32)%64)*2
    return dict(dataset='gsm8k', event=event, window=window, pass_index=pass_index,
        fitting_ids=[r['record_id'] for r in records[start:start+2]],
        off_fit_ids=[r['record_id'] for r in records[other:other+2]],
        diagnostic_rng_key=[20261108, 0, event, window],
        scope='off-fit buffered training data, not held-out evaluation')


def candidate_control(name, receipt, production):
    candidate = receipt['candidates'].get(name, {})
    if name == 'prestate':
        stats = dict(norm=0., native_norm=production['native']['native_norm'], direction_cosine=None)
    elif name == 'native' or candidate.get('bypass') == 'zero_production_sigma':
        stats = production['native']
    elif name == 'committed':
        stats = production['committed']
    else:
        stats = candidate['candidate']
    decision = (production['decision'] if name == 'committed' else candidate.get('decision', {}))
    return dict(realized_norm=stats['norm'], native_norm=stats['native_norm'],
        norm_ratio_to_native=stats['norm']/stats['native_norm'] if stats['native_norm'] else None,
        direction_cosine=stats.get('direction_cosine'),
        continuous_norm=stats.get('continuous_norm'),
        scalar_target_norm=stats.get('target_norm'),
        scalar_relative_norm_mismatch=stats.get('relative_norm_mismatch'),
        candidate_guard_accepted=decision.get('accepted'),
        candidate_guard_reason=decision.get('reason'),
        guarded_score=candidate.get('guarded_score'),
        exact_score_reuse=candidate.get('exact_score_reuse'),
        diagnostic_only=name not in ('committed', 'prestate', 'native'),
        guard_scope='fitting_pair_A_only; D outcome is not guarded')


def aggregate_pair(rows, risk, control, *, event, window, role, candidate):
    result = dict(dataset='gsm8k', event=event, window=window, pair=role, candidate=candidate,
                  response_denominator=2, **control)
    result.update({name: sum(row[name] for row in rows) for name in COUNTS})
    raw = .5*math.fsum(row['risk_unaveraged'] for row in rows)
    require(risk['raw'] == raw, 'Stored pair protected KL differs from its two responses')
    require(risk['value'] == max(0., raw), 'Stored protected-KL deadband representation differs')
    result.update(protected_old_draft_kl=risk['value'], protected_old_draft_kl_raw=raw,
                  protected_old_draft_kl_eta=risk['eta'], kl_response_average_denominator=2)
    for teacher_role in ('prefix', 'first_error'):
        terms = [row['teacher_kl'][teacher_role] for row in rows]
        numerator = math.fsum(term['numerator'] for term in terms)
        denominator = math.fsum(term['denominator'] for term in terms)
        result.update({f'teacher_{teacher_role}_numerator': numerator,
            f'teacher_{teacher_role}_denominator': denominator,
            f'teacher_{teacher_role}_mean': numerator/denominator if denominator else None,
            f'teacher_{teacher_role}_occurrences': sum(term['occurrences'] for term in terms)})
    result['useful_repair_fraction'] = result['C_useful_repairs']/result['old_first_errors'] if result['old_first_errors'] else None
    result['native_repair_retention_fraction'] = result['useful_repairs_retained']/result['K_useful_repairs'] if result['K_useful_repairs'] else None
    result['old_prefix_loss_fraction'] = result['old_prefix_tokens_lost']/result['old_matching_prefix_tokens'] if result['old_matching_prefix_tokens'] else None
    return result


def summarize_events(pairs):
    grouped = defaultdict(list)
    for row in pairs:
        grouped[row['event'], row['pair'], row['candidate']].append(row)
    result = []
    for (event, role, candidate), rows in sorted(grouped.items()):
        value = dict(event=event, pair=role, candidate=candidate, window_denominator=len(rows),
                     response_occurrence_denominator=2*len(rows))
        value.update({name: sum(row[name] for row in rows) for name in COUNTS})
        value['mean_pair_protected_old_draft_kl'] = math.fsum(row['protected_old_draft_kl'] for row in rows)/len(rows)
        for kind in ('prefix', 'first_error'):
            numerator = math.fsum(row[f'teacher_{kind}_numerator'] for row in rows)
            denominator = math.fsum(row[f'teacher_{kind}_denominator'] for row in rows)
            value.update({f'teacher_{kind}_numerator': numerator, f'teacher_{kind}_denominator': denominator,
                          f'teacher_{kind}_mean': numerator/denominator if denominator else None})
        result.append(value)
    return result


def contrasts(pairs):
    grouped = defaultdict(dict)
    for row in pairs:
        grouped[row['event'], row['window'], row['pair']][row['candidate']] = row
    result = []
    for (event, window, role), candidates in sorted(grouped.items()):
        for left, right in CONTRASTS:
            if left not in candidates or right not in candidates:
                continue
            a, b = candidates[left], candidates[right]
            row = dict(event=event, window=window, pair=role, comparison=left+'_minus_'+right,
                       left=left, right=right, left_norm=a['realized_norm'], right_norm=b['realized_norm'],
                       norm_relative_difference=(a['realized_norm']/b['realized_norm']-1) if b['realized_norm'] else None)
            for key in ('C_useful_repairs', 'net_matching_prefix_change', 'old_prefix_tokens_lost',
                        'new_earlier_errors', 'protected_old_draft_kl', 'teacher_prefix_mean', 'teacher_first_error_mean'):
                row[key+'_delta'] = a[key]-b[key] if a[key] is not None and b[key] is not None else None
            result.append(row)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contrast_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['event'], row['pair'], row['comparison']].append(row)
    summaries = []
    for (event, role, comparison), values in sorted(grouped.items()):
        result = dict(event=event, pair=role, comparison=comparison, windows=len(values))
        for key in ('C_useful_repairs', 'net_matching_prefix_change', 'old_prefix_tokens_lost', 'new_earlier_errors'):
            changes = [row[key+'_delta'] for row in values]
            result.update({key+'_delta':sum(changes), key+'_positive_windows':sum(v>0 for v in changes),
                           key+'_negative_windows':sum(v<0 for v in changes), key+'_tied_windows':sum(v==0 for v in changes)})
        result['lower_KL_windows'] = sum(row['protected_old_draft_kl_delta']<0 for row in values)
        result['lower_KL_fewer_repairs_lower_net_windows'] = sum(row['protected_old_draft_kl_delta']<0
            and row['C_useful_repairs_delta']<0 and row['net_matching_prefix_change_delta']<0 for row in values)
        result['max_absolute_relative_norm_difference'] = max((abs(row['norm_relative_difference'])
            for row in values if row['norm_relative_difference'] is not None), default=None)
        summaries.append(result)
    return summaries


def analyze(root, output):
    started = time.perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(root)
    plan = inputs.read('sealed/PLAN.json')
    require(plan['execution_source']['commit'] == COMMIT, 'Execution source identity differs')
    pairs, responses, windows, errors = [], [], [], []
    for event in (5, 6):
        base = Path(f'campaign_001/gsm8k/event_{event:03d}')
        try:
            native = inputs.read(base/'KPLUS/RESULT.json')
            fsg = inputs.read(base/'FSG/RESULT.json')
            bindings = inputs.read(base/'FSG/diagnostics/BINDINGS.json')
            observers = inputs.read(base/'FSG/diagnostics/RESULTS.json')
            require([row['window'] for row in bindings] == list(WINDOWS), 'Diagnostic binding coverage differs')
            require([row['window'] for row in observers] == list(WINDOWS), 'Diagnostic observer coverage differs')
            require(len(native['records']) == len(fsg['records']) == 256, 'Training presentation count differs')
            require([(r['record_id'], r['pass'], r['sampled_anchors']) for r in native['records']] ==
                    [(r['record_id'], r['pass'], r['sampled_anchors']) for r in fsg['records']], 'A trajectory anchors or order differ')
        except (OSError, KeyError, ValueError, TypeError) as error:
            errors.append(dict(event=event, error=str(error)))
            continue
        for window, binding, observer in zip(WINDOWS, bindings, observers):
            try:
                folder = base/'FSG/diagnostics'/f'window_{window:03d}'
                receipt = inputs.read(folder/'RESULT.json')
                anchors = inputs.read(folder/'ANCHORS.json')
                production = inputs.read(base/'FSG/geometry'/f'step_{(event-1)*128+window:06d}.json')
                expected = expected_binding(native['records'], event, window)
                require(binding == expected == receipt['binding'], 'A/D cyclic binding differs')
                require(len(set(binding['fitting_ids']+binding['off_fit_ids'])) == 4, 'A/D identities overlap')
                require(receipt['status'] == observer['status'] == production['status'] == 'completed', 'Incomplete fixed window')
                require(receipt['restoration_exact'] and receipt['physical_restore_hidden_exact'] and receipt['extra_adam_steps'] == 0, 'Window state restoration differs')
                require(observer['measurements'] == receipt['measurements'], 'Duplicate observer metrics differ')
                require(production['global_step'] == (event-1)*128+window, 'Production step identity differs')
                expected_pairs = [('A', key) for key in binding['fitting_ids']]+[('D', key) for key in binding['off_fit_ids']]
                require([(a['role'], a['record_id']) for a in anchors] == expected_pairs, 'Anchor identities differ')
                for local, a in enumerate(anchors[:2]):
                    original = fsg['records'][(window-1)*2+local]
                    require(len(original['sampled_anchors']) == 1, 'Multiple fitting anchor draws')
                    draw = original['sampled_anchors'][0]
                    require(a['anchors'] == draw['positions'] and a['keep'] == draw['keep_mask'], 'Fitting anchors differ')
                measured = receipt['measurements']
                expected_candidates = set(CANDIDATES)
                if receipt['candidates'].get('expanded', {}).get('bypass') == 'zero_production_sigma':
                    expected_candidates.remove('expanded_raw')
                require(set(measured) == expected_candidates, 'Candidate coverage differs')
                local_pairs, local_responses = [], []
                for name in CANDIDATES:
                    if name not in measured:
                        continue
                    current = measured[name]
                    require([(r['role'], r['record_id']) for r in current] == expected_pairs, 'Response/pair ordering differs')
                    control = candidate_control(name, receipt, production)
                    for i, row in enumerate(current):
                        verify_response(row, measured['prestate'][i], measured['native'][i])
                        flattened = dict(event=event, window=window, candidate=name, role=row['role'], record_id=row['record_id'])
                        flattened.update({key: row[key] for key in COUNTS})
                        flattened.update(risk_unaveraged=row['risk_unaveraged'])
                        for role in ('prefix', 'first_error'):
                            flattened.update({f'teacher_{role}_{key}': value for key, value in row['teacher_kl'][role].items()})
                        local_responses.append(flattened)
                    for role, subset in (('A', current[:2]), ('D', current[2:])):
                        local_pairs.append(aggregate_pair(subset, receipt['role_risks'][name][role], control,
                            event=event, window=window, role=role, candidate=name))
                pairs.extend(local_pairs); responses.extend(local_responses)
                windows.append(dict(event=event, window=window, status='valid', candidates=len(measured),
                    response_occurrences=4, production_guard=production['decision'],
                    state_restored=True, observer_wall_s=observer.get('observer_wall_s')))
            except (OSError, KeyError, ValueError, TypeError) as error:
                errors.append(dict(event=event, window=window, error=str(error)))
                windows.append(dict(event=event, window=window, status='invalid_or_unavailable'))
    event_rows = summarize_events(pairs)
    contrast_rows = contrasts(pairs)
    contrast_totals = contrast_summary(contrast_rows)
    for name, rows in (('window_pair_candidates', pairs), ('response_candidates', responses),
                       ('event_pair_candidates', event_rows), ('same_state_contrasts', contrast_rows),
                       ('event_contrast_summary', contrast_totals)):
        write_csv(output/(name+'.csv'), rows)
    summary = dict(status='analyzed' if len(windows)==8 and not errors else 'incomplete_or_invalid',
        execution_commit=COMMIT, binding=plan['binding'], windows=windows, errors=errors, pair_candidate_rows=len(pairs),
        event_pair_candidates=event_rows, same_state_contrasts=contrast_rows,
        work=dict(model_loads=0, forwards=0, backwards=0, optimizer_steps=0, generations=0),
        cpu_wall_s=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        interpretation='Fixed same-prestate recorded diagnostics; D is buffered training data; no independent-token or seed inference')
    for name, value in (('SUMMARY.json', summary), ('INPUTS.json', inputs.files)):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    (output/'CONTRAST_SUMMARY.json').write_text(json.dumps(contrast_totals, indent=2, allow_nan=False)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.evidence, args.output)
    print(json.dumps({key:result[key] for key in ('status', 'pair_candidate_rows', 'errors', 'cpu_wall_s')}))
