"""Local, scalar-only numerical/accounting supplement; no model or network use."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
EXECUTION = 'c9461345189b91d241ff21590027500e6ac7dab4'
ANALYSIS = '220c62e2c52489cf114125b679ae7188eea09218'
WINDOWS = (1, 16, 32, 48, 64, 65, 80, 96, 112, 128)
CANDIDATES = ('R_FULL', 'R_HALF', 'D_FULL')
INPUTS = {}


def source(path):
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data


def read_json(path):
    return json.loads(source(path))


def read_csv(path):
    return list(csv.DictReader(source(path).decode().splitlines()))


def summary(values, reasons=()):
    present = [value for value in values if value is not None]
    return dict(n=len(values), defined=len(present), undefined=len(values)-len(present),
        min=min(present) if present else None,
        median=statistics.median(present) if present else None,
        max=max(present) if present else None,
        undefined_reasons=dict(Counter(reason for reason in reasons if reason)))


def quotient(numerator, denominator, name):
    return (numerator/denominator, None) if denominator else (None, name + '_is_zero')


def numerical(rows):
    output = []
    per_window = []
    value_names = ('V_before', 'formal_ideal', 'ideal', 'proposed_linearized_V',
        'realized_linearized_V', 'actual_V', 'proposed_minus_formal_ideal',
        'realized_minus_proposed', 'actual_minus_realized_linearized',
        'proposed_norm', 'realized_norm')
    for event in (2, 4):
        for candidate in CANDIDATES:
            selected = [row for row in rows if int(row['event']) == event and row['candidate'] == candidate]
            assert len(selected) == 10
            values = {name: [] for name in value_names}
            ratios = defaultdict(list)
            ratio_reasons = defaultdict(list)
            for row in selected:
                numbers = {name: float(row[name]) if row[name] else None for name in value_names}
                for name, value in numbers.items():
                    values[name].append(value)
                proposed_dot = float(row['g_dot_proposed_fp32_displacement'])
                realized_dot = float(row['g_dot_realized_bf16_displacement'])
                v = numbers['V_before']
                specifications = {
                    'formal_ideal_over_V_before': (numbers['formal_ideal'], v, 'V_before'),
                    'proposed_linearized_over_V_before': (numbers['proposed_linearized_V'], v, 'V_before'),
                    'realized_linearized_over_V_before': (numbers['realized_linearized_V'], v, 'V_before'),
                    'actual_over_V_before': (numbers['actual_V'], v, 'V_before'),
                    'realized_norm_over_proposed_norm': (numbers['realized_norm'], numbers['proposed_norm'], 'proposed_norm'),
                    'realized_dot_over_proposed_dot': (realized_dot, proposed_dot, 'proposed_dot'),
                    'actual_hinge_reduction_fraction': (v-numbers['actual_V'], v, 'V_before'),
                    'actual_minus_realized_linearized_over_V_before': (numbers['actual_minus_realized_linearized'], v, 'V_before'),
                }
                row_ratios = {}
                for name, arguments in specifications.items():
                    value, reason = quotient(*arguments)
                    ratios[name].append(value)
                    ratio_reasons[name].append(reason)
                    row_ratios[name] = dict(value=value, null_reason=reason)
                per_window.append(dict(event=event, pass_index=int(row['pass_index']),
                    local_step=int(row['local_step']), candidate=candidate, fitted_role=row['fitted_role'],
                    branch_status=row['branch_status'], values=numbers, ratios=row_ratios))
            output.append(dict(event=event, candidate=candidate, fitted_role=selected[0]['fitted_role'],
                branch_counts=dict(Counter(row['branch_status'] for row in selected)),
                values={name: summary(data, [row['ideal_null_reason'] for row in selected]
                    if name == 'ideal' else ()) for name, data in values.items()},
                ratios={name: summary(data, ratio_reasons[name]) for name, data in ratios.items()}))
    return output, per_window


def add_numbers(target, row):
    for key, value in row.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value


def costs_and_work(raw, compact, windows):
    result = dict(campaign_wall_s=compact['actual_campaign_wall_s'],
        work_ledger=compact['work_ledger'], events=[])
    outer_keys = ('pre_capture_snapshot_s', 'pre_capture_s', 'post_K_parameter_snapshot_s',
                  'full_continuation_snapshot_s', 'candidate_sequence_s', 'final_restore_and_check_s')
    for event in (2, 4):
        name = f'event_{event:03d}'
        episode = raw['episodes'][name]
        original = episode['correction']
        window_rows = windows[event]
        timer_names = sorted({key for window in window_rows for key in window['costs']})
        diagnostic = {key: dict(sum_s=sum(window['costs'][key] for window in window_rows),
                              **summary([window['costs'][key] for window in window_rows])) for key in timer_names}
        candidate_costs = {}
        candidate_children_sum = 0.
        for candidate in (*CANDIDATES, 'K_ONLY'):
            names = sorted({key for window in window_rows for key in window['candidates'][candidate]['costs']})
            candidate_costs[candidate] = {}
            for key in names:
                data = [window['candidates'][candidate]['costs'][key] for window in window_rows]
                candidate_costs[candidate][key] = dict(sum_s=sum(data), **summary(data))
                candidate_children_sum += sum(data)
        candidate_children_sum += diagnostic['validation_and_block_metrics_s']['sum_s']
        sequence_sum = diagnostic['candidate_sequence_s']['sum_s']
        assert sequence_sum >= candidate_children_sum
        phase_calls = defaultdict(dict)
        for key, count in original['forward_calls'].items():
            phase, module = key.rsplit('/', 1)
            phase_calls[phase][module] = count
        phase_summary = []
        for phase, counts in phase_calls.items():
            layers = {key: value for key, value in counts.items() if key.startswith('draft_layer_')}
            phase_summary.append(dict(phase=phase, draft_forwards=counts.get('draft', 0),
                vocabulary_head_calls_including_recompute=counts.get('vocabulary_head', 0),
                layer_calls=sum(layers.values()), layer_calls_by_name=layers,
                layer_calls_above_one_per_layer_per_draft=sum(layers.values())-len(layers)*counts.get('draft', 0)))
        assert sum(row['draft_forwards'] for row in phase_summary) == 1068
        counters = {role: {} for role in ('A', 'R', 'D')}
        mechanism_work = {}
        for window in window_rows:
            add_numbers(mechanism_work, window['work'])
            for role in counters:
                add_numbers(counters[role], window['work']['projection_chunks_and_recomputation'][role])
        result['events'].append(dict(event=event,
            enclosing_times_s={key: episode[key] for key in ('parent_validation_s',
                'loading_and_live_resume_s', 'endpoint_comparison_s', 'episode_wall_s')},
            post_episode_timer_s={'verified_replay_retirement_s': episode.get('verified_replay_retirement_s')},
            startup=compact['episodes'][name]['startup'],
            event_costs_s=episode['event'], objective_component_times_s=episode['objective_work'],
            original_reference_correction_times_s=original['costs'],
            diagnostic_window_timer_distributions=diagnostic,
            candidate_timer_distributions=candidate_costs,
            diagnostic_disjoint_outer_timer_sum_s=sum(diagnostic[key]['sum_s'] for key in outer_keys),
            candidate_sequence_named_children_sum_s=candidate_children_sum,
            candidate_sequence_other_instrumentation_s=sequence_sum-candidate_children_sum,
            standalone_D_gradient_time_s=None,
            standalone_D_gradient_time_null_reason='Only D_FULL construction_and_gradient_s was recorded; it also includes correction and gradient alignment.',
            original_response_and_backward_work=original['work'],
            diagnostic_response_and_backward_work=mechanism_work,
            diagnostic_chunk_work_by_role=counters,
            forward_phase_module_counts=phase_summary,
            R_extra_post_K_head_evaluations=original['work']['post_k_reference_chunks']-original['work']['pre_reference_chunks'],
            D_extra_post_K_head_evaluations=counters['D']['post_k_objective_chunks']-counters['D']['pre_reference_chunks'],
            peak_allocated_bytes=episode['peak_allocated_bytes'], peak_reserved_bytes=episode['peak_reserved_bytes']))
    return result


def tie_detail(windows):
    observations = []
    for event in (2, 4):
        for window in windows[event]:
            step = window['step']
            baseline = window['candidates']['K_ONLY']['roles']['R']['V']
            current = {}
            for candidate in CANDIDATES:
                role = window['candidates'][candidate]['roles']['R']
                totals = role['metrics']['summary']
                current[candidate] = dict(event=event, pass_index=(step-1)//64, local_step=step,
                    candidate=candidate, R_V_before=baseline, R_V_after=role['V'],
                    R_hinge_decreased=role['V'] < baseline, R_hinge_zero=role['V'] == 0,
                    valid_R_blocks=totals['valid_blocks'], protected_R_positions=totals['protected_positions'],
                    R_wrong_tie_positions=totals['candidate_wrong_native_tie_errors'],
                    R_strict_negative_margin_positions=totals['candidate_strict_negative_margin_errors'],
                    R_prefix_change=totals['prefix_change_sum'],
                    blocking_protected_wrong_tie_blocks=0,
                    old_prefix_tokens_lost_in_blocking_tie_blocks=0)
            path = ROOT / 'raw/campaign_001' / f'event_{event:03d}/mechanism/window_{step:03d}_BLOCKS.jsonl'
            digest = hashlib.sha256()
            reconstructed_ties = Counter()
            with path.open('rb') as stream:
                for line in stream:
                    digest.update(line)
                    block = json.loads(line)
                    if block['role'] != 'R' or block['candidate'] not in current:
                        continue
                    offset = block['candidate_prefix_tokens'] + 1
                    weights = block['protected_weights']
                    reconstructed_ties[block['candidate']] += sum(
                        weight > 0 and prediction != canonical and margin == 0
                        for weight, prediction, canonical, margin in zip(weights,
                            block['candidate_predictions'], block['canonical_tokens'],
                            block['candidate_canonical_margins']))
                    if (offset < len(weights) and weights[offset] > 0
                            and block['candidate_predictions'][offset] != block['canonical_tokens'][offset]
                            and block['candidate_canonical_margins'][offset] == 0):
                        row = current[block['candidate']]
                        row['blocking_protected_wrong_tie_blocks'] += 1
                        row['old_prefix_tokens_lost_in_blocking_tie_blocks'] += block['old_prefix_tokens_lost']
            INPUTS[str(path.relative_to(ROOT))] = digest.hexdigest()
            for candidate, row in current.items():
                assert reconstructed_ties[candidate] == row['R_wrong_tie_positions']
            observations.extend(current.values())
    summaries = []
    for event in (2, 4):
        for candidate in CANDIDATES:
            selected = [row for row in observations if row['event'] == event and row['candidate'] == candidate]
            decreases = [row for row in selected if row['R_hinge_decreased']]
            summaries.append(dict(event=event, candidate=candidate, total_windows=len(selected),
                R_hinge_decreased_windows=len(decreases),
                decreased_windows_with_R_wrong_ties=sum(row['R_wrong_tie_positions'] > 0 for row in decreases),
                decreased_windows_with_blocking_protected_ties=sum(row['blocking_protected_wrong_tie_blocks'] > 0 for row in decreases),
                decreased_windows_with_zero_R_hinge=sum(row['R_hinge_zero'] for row in decreases),
                decreased_windows_with_positive_R_prefix_change=sum(row['R_prefix_change'] > 0 for row in decreases),
                **{key: sum(row[key] for row in decreases) for key in ('valid_R_blocks', 'protected_R_positions',
                    'R_wrong_tie_positions', 'R_strict_negative_margin_positions', 'R_prefix_change',
                    'blocking_protected_wrong_tie_blocks', 'old_prefix_tokens_lost_in_blocking_tie_blocks')}))
    return dict(condition='Strict decrease of freshly scored R V versus same-window K_ONLY R V; not D fitted V.',
        interpretation='Protected wrong-tie positions and currently blocking protected-tie blocks are distinct. Counts are overlapping block occurrences, not requests or independent replications.',
        summaries=summaries, per_window=observations)


def triple(row):
    return ' / '.join('null' if row[key] is None else f'{row[key]:.6g}' for key in ('min', 'median', 'max'))


def markdown(result):
    lines = ['# Numerical predictions and diagnostic accounting', '',
        f"Execution: `{EXECUTION}`. Validated analysis: `{ANALYSIS}`. This supplement contains descriptive numerical and accounting checks only; it makes no mechanism recommendation.", '',
        '## Fitted-objective values', '',
        'Every cell is **minimum / median / maximum across ten fixed windows** of that event. R candidates fit R; D_FULL fits D. Formal ideal is (1−alpha)V even for an inactive branch; the applicable ideal and its null reason are retained separately in JSON. Negative linearized predictions are retained.', '',
        '| Event | Candidate | Before V | Formal ideal | Proposed linear V | Realized linear V | Actual V |',
        '|---:|---|---|---|---|---|---|']
    for row in result['numerical_summaries']:
        lines.append(f"| {row['event']} | {row['candidate']} | " + ' | '.join(triple(row['values'][key]) for key in
            ('V_before', 'formal_ideal', 'proposed_linearized_V', 'realized_linearized_V', 'actual_V')) + ' |')
    lines += ['', '## Per-window ratios', '',
        'Cells again show minimum / median / maximum, computed per window before summarizing. Undefined denominators yield null with an explicit reason; they are never replaced by zero. Ratios are descriptive and can exceed one or be negative.', '',
        '| Event | Candidate | Actual V / before V | Realized/proposed dot | Realized/proposed norm | Actual reduction fraction |',
        '|---:|---|---|---|---|---|']
    for row in result['numerical_summaries']:
        lines.append(f"| {row['event']} | {row['candidate']} | " + ' | '.join(triple(row['ratios'][key]) for key in
            ('actual_over_V_before', 'realized_dot_over_proposed_dot', 'realized_norm_over_proposed_norm', 'actual_hinge_reduction_fraction')) + ' |')
    lines += ['', 'The JSON also retains proposed/realized linear V divided by before V, each discrepancy, applicable-ideal nulls, all zero-denominator reasons, and every window’s scalar values. Proposed-versus-realized differences describe rounding of the actual displacement; actual-versus-realized residuals do not uniquely separate nonlinear/nonsmooth effects from finite-precision gradient/forward effects.', '',
        '## Wrong ties after R hinge decreases', '',
        '| Event | Candidate | R hinge falls: windows | Those windows with wrong ties | Protected wrong-tie positions | Blocking protected-tie blocks | Old-prefix tokens lost in those blocks | R prefix delta |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in result['R_wrong_tie_detail']['summaries']:
        lines.append(f"| {row['event']} | {row['candidate']} | {row['R_hinge_decreased_windows']}/10 | "
            f"{row['decreased_windows_with_R_wrong_ties']} | {row['R_wrong_tie_positions']} | "
            f"{row['blocking_protected_wrong_tie_blocks']} | {row['old_prefix_tokens_lost_in_blocking_tie_blocks']} | {row['R_prefix_change']:+} |")
    lines += ['', 'A blocking tie is the candidate’s first native mismatch, with zero canonical margin, inside the originally protected prefix. These counts establish local remaining errors, not a causal explanation of all missed progress or a serving-speed claim. D_FULL is selected here by its cross-reference R hinge, not its fitted D hinge.', '',
        '## Phase costs', '', f"Campaign wall time: **{result['accounting']['campaign_wall_s']:.3f} s**.", '',
        '| Timer / scope (seconds) | Event 2 | Event 4 |', '|---|---:|---:|']
    events = result['accounting']['events']
    selectors = [
        ('Episode through endpoint comparison', lambda e: e['enclosing_times_s']['episode_wall_s']),
        ('Parent-state validation, inside episode', lambda e: e['enclosing_times_s']['parent_validation_s']),
        ('Disposable startup wall, inside episode', lambda e: e['startup']['wall_s']),
        ('Actual load/live resume, inside episode', lambda e: e['enclosing_times_s']['loading_and_live_resume_s']),
        ('Training-event wall before commit', lambda e: e['event_costs_s']['event_wall_s_before_commit']),
        ('Training windows, nested in event', lambda e: e['event_costs_s']['training_s']),
        ('Checkpoint save, nested in event', lambda e: e['event_costs_s']['checkpoint_save_s']),
        ('Export, nested in event', lambda e: e['event_costs_s']['export_s']),
        ('Decoded endpoint comparison, inside episode', lambda e: e['enclosing_times_s']['endpoint_comparison_s']),
        ('Verified replay retirement, after episode timer', lambda e: e['post_episode_timer_s']['verified_replay_retirement_s']),
    ]
    for key in ('pre_reference_s', 'post_k_s', 'correction_s', 'integrity_s', 'pre_reference_integrity_s'):
        selectors.append(('Original ' + key + ', nested in training', lambda e, key=key: e['original_reference_correction_times_s'][key]))
    for key in events[0]['diagnostic_window_timer_distributions']:
        selectors.append(('Diagnostic ' + key + ', sum over ten windows', lambda e,key=key: e['diagnostic_window_timer_distributions'][key]['sum_s']))
    selectors += [('Diagnostic disjoint outer timers (not added again)', lambda e: e['diagnostic_disjoint_outer_timer_sum_s']),
                  ('Candidate-sequence other instrumentation', lambda e: e['candidate_sequence_other_instrumentation_s'])]
    for label, getter in selectors:
        values = [getter(event) for event in events]
        lines.append('| ' + label + ' | ' + ' | '.join('not recorded / not applicable' if value is None else f'{value:.3f}' for value in values) + ' |')
    lines += ['', '| Event | Candidate | Restore sum | Construction/gradient sum | State hash sum | Scoring sum |',
              '|---:|---|---:|---:|---:|---:|']
    for event in events:
        for candidate, timings in event['candidate_timer_distributions'].items():
            lines.append(f"| {event['event']} | {candidate} | " + ' | '.join(f"{timings[key]['sum_s']:.3f}" for key in
                ('restore_s', 'construction_and_gradient_s', 'state_identity_s', 'scoring_s')) + ' |')
    lines += ['', *['- ' + line for line in result['timer_definitions']], '', '## Work and recomputation', '',
        '| Event | Phase | Draft forwards | Head calls including recomputation | Layer calls | Layer excess over one/layer/draft |',
        '|---:|---|---:|---:|---:|---:|']
    for event in events:
        for row in event['forward_phase_module_counts']:
            lines.append(f"| {event['event']} | {row['phase']} | {row['draft_forwards']} | "
                f"{row['vocabulary_head_calls_including_recompute']} | {row['layer_calls']} | {row['layer_calls_above_one_per_layer_per_draft']} |")
        lines.append(f"| {event['event']} | separate disposable startup | {event['startup']['work_counts']['draft_forwards']} | "
            f"{event['startup']['work_counts']['head_forwards_including_recompute']} | {event['startup']['work_counts']['layer_forwards_including_recompute']} | 0 |")
    ledger = result['accounting']['work_ledger']
    lines += ['', f"Recorded core work: {ledger['core_response_forwards']} response forwards, {ledger['core_response_backwards']} backward equivalents, "
        f"{ledger['ordinary_Adam_steps']} ordinary Adam steps and {ledger['ordinary_presentations']} presentations. Separate startup: "
        f"{ledger['startup_response_forwards']} forwards / {ledger['startup_response_backwards']} backwards / {ledger['startup_Adam_steps']} Adam. Serving and feedback generations: zero.", '',
        'The head/layer calls above are internal work, not additional response equivalents. The D_FULL hook phase includes its D gradient replay plus six candidate-scoring responses per window. R extra post-K head evaluations (beyond its same-layout pre-capture count) are ' +
        '/'.join(str(event['R_extra_post_K_head_evaluations']) for event in events) + ' for events 2/4; corresponding D extra evaluations are ' +
        '/'.join(str(event['D_extra_post_K_head_evaluations']) for event in events) + '. Ordinary K head calls include its own objective/teacher work and recomputation; this receipt does not separate all those contributions.', '',
        'All original/candidate counters, candidate timing distributions, startup counters, byte-based memory peaks and scalar objective component timers remain in the JSON. Earlier KRC1 local non-monotonicity was already documented; this supplement makes no first-discovery claim.', '']
    return '\n'.join(lines)


def main():
    analysis = ROOT / 'analysis'
    existing = read_json(analysis / 'SUMMARY.json')
    assert existing['execution_git_commit'] == EXECUTION and existing['included_candidate_slots'] == 80
    predictions = read_csv(analysis / 'PREDICTIONS.csv')
    compact = read_json(analysis / 'COSTS_WORK.json')
    raw_costs = read_json(ROOT / 'raw/campaign_001/COSTS.json')
    assert raw_costs['work_ledger'] == compact['work_ledger']
    windows = {event: [read_json(ROOT / 'raw/campaign_001' / f'event_{event:03d}/mechanism/window_{step:03d}.json')
                       for step in WINDOWS] for event in (2, 4)}
    for rows in windows.values():
        assert all(row['status'] == 'completed' and row['restoration_passed'] and row['k_only_witness_passed'] for row in rows)
    by_identity = {(event, window['step']): window for event, rows in windows.items() for window in rows}
    for row in predictions:
        entry = by_identity[int(row['event']), int(row['local_step'])]['candidates'][row['candidate']]
        for key, value in entry['prediction'].items():
            if key not in row:
                continue
            actual = None if not row[key] else float(row[key]) if isinstance(value, (float, int)) else row[key]
            assert actual == value, (row['event'], row['local_step'], row['candidate'], key)
    values, per_window = numerical(predictions)
    accounting = costs_and_work(raw_costs, compact, windows)
    ties = tie_detail(windows)
    result = dict(execution_git_commit=EXECUTION, validated_analysis_git_commit=ANALYSIS,
        supplement_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_summaries=values, numerical_per_window=per_window, R_wrong_tie_detail=ties,
        accounting=accounting,
        timer_definitions=[
            'Episode wall is recorded before replay-state identity hashing and event-2 tensor retirement; campaign wall includes those later operations. Do not assume episode sums cover the whole campaign.',
            'Startup wall includes startup loading. Event wall contains training, checkpoint save and export; objective and original correction component timers nest inside training.',
            'The disjoint diagnostic outer sum contains pre-capture snapshot, pre-capture, post-K parameter snapshot, full-continuation snapshot, candidate sequence and final restore/check. Validation/block metrics are inside candidate sequence, so they are not added again.',
            'Candidate restore, construction/gradient, state identity and scoring are inside candidate sequence. Its residual also includes direction-evidence reductions, recorder setup, receipt writes and other instrumentation; it is not a separately measured kernel category.',
            'D_FULL construction_and_gradient_s includes its gradient replay, direct correction and gradient alignment. A standalone D-gradient timer is unavailable and remains null.',
            'R_FULL construction timing is bookkeeping only: the actual original full correction was already applied and charged to original correction_s.',
            'These are recorded host wall timers. Not every subphase is separately CUDA-synchronized; a later CPU copy/hash can absorb preceding asynchronous work. Do not interpret construction/hash components as isolated GPU-kernel costs.',
            'Candidate and diagnostic component timers overlap their enclosing sequence/training/event/campaign timers. No overlapping timer totals are added.',
        ],
        scope='Descriptive scalar supplement; no model passes, GPU/network use, raw-array publication or mechanism recommendation.',
        source_files_sha256=INPUTS)
    forbidden = {'canonical_tokens', 'valid_positions', 'protected_weights', 'pre_K_predictions',
        'K_ONLY_predictions', 'candidate_predictions', 'all_margins', 'anchors', 'sampled_anchors',
        'hidden_states', 'input_ids', 'response_ids', 'prompt_ids'}
    def assert_public(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                assert_public(item)
        elif isinstance(value, list):
            for item in value:
                assert_public(item)
    assert_public(result)
    (analysis / 'NUMERICAL_COST_DETAIL.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (analysis / 'NUMERICAL_COST_DETAIL.md').write_text(markdown(result))
    print(json.dumps(dict(numerical_groups=len(values), numerical_windows=len(per_window),
        cost_events=len(accounting['events']), tie_groups=len(ties['summaries']),
        output_json=str(analysis/'NUMERICAL_COST_DETAIL.json'), output_markdown=str(analysis/'NUMERICAL_COST_DETAIL.md')), indent=2))


if __name__ == '__main__':
    main()
