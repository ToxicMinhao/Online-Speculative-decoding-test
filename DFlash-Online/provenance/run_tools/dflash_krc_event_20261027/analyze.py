"""Summarize retained KRC records locally, with no model or remote execution."""
import csv
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('kc_analysis_helpers',
    HERE.parent / 'dflash_kc_event_20261027/analyze.py')
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
read, write_csv = BASE.read, BASE.write_csv
finite_number, show, ratio = BASE.finite_number, BASE.show, BASE.ratio
ARMS = ('K', 'KRC')
FORWARD_KEYS = (
    'original_response_forwards', 'pre_reference_response_forwards',
    'post_k_reference_response_forwards', 'post_c_reference_response_forwards',
    'post_k_learning_response_forwards', 'post_c_learning_response_forwards',
)


def rows(path):
    if not path.is_file():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def reduction(reference, candidate):
    return 100 * (1 - candidate / reference) if reference else None


def native_summary(raw):
    """Re-pool validated paired counts and request means, never timing samples."""
    directory = raw / 'evaluation/report'
    source = read(directory / 'SUMMARY.json', {})
    per_request = rows(directory / 'PER_REQUEST.csv')
    valid = [row for row in per_request if row['valid'] == 'True']
    result = {}
    for arm in ARMS:
        calls = sum(int(row[f'{arm}_verification_calls']) for row in valid)
        tokens = sum(int(row[f'{arm}_output_tokens']) for row in valid)
        accepted = sum(int(row[f'{arm}_accepted_draft_tokens']) for row in valid)
        progress = sum(int(row[f'{arm}_committed_progress']) for row in valid)
        latency = sum(float(row[f'{arm}_mean_request_latency_s']) for row in valid)
        result[arm] = dict(verification_calls=calls, output_tokens=tokens,
            accepted_draft_tokens=accepted, committed_progress=progress,
            summed_mean_request_latency_s=latency,
            calls_per_1000_output_tokens=1000 * calls / tokens if tokens else None,
            accepted_draft_tokens_per_call=ratio(accepted, calls),
            committed_progress_per_call=ratio(progress, calls))
        if source and len(valid):
            for name in ('verification_calls', 'output_tokens', 'accepted_draft_tokens', 'committed_progress'):
                if result[arm][name] != source[arm][name]:
                    raise ValueError(f'Paired CSV and serving summary disagree: {arm}/{name}')
            if abs(latency - source[arm]['summed_mean_request_latency_s']) > 1e-9:
                raise ValueError(f'Paired CSV and serving mean latency disagree: {arm}')
    differences = [int(row['K_verification_calls']) - int(row['KRC_verification_calls']) for row in valid]
    comparison = dict(
        call_reduction_pct=reduction(result['K']['verification_calls'], result['KRC']['verification_calls']),
        request_time_reduction_pct=reduction(result['K']['summed_mean_request_latency_s'],
                                            result['KRC']['summed_mean_request_latency_s']),
        improved=sum(value > 0 for value in differences),
        regressed=sum(value < 0 for value in differences), tied=sum(value == 0 for value in differences),
        gross_saved_calls=sum(max(value, 0) for value in differences),
        gross_added_calls=sum(max(-value, 0) for value in differences),
        net_saved_calls=sum(differences),
    )
    repeat_totals = rows(directory / 'REPEAT_TOTALS.csv')
    repeat_comparisons = []
    for repeat in range(3):
        selected = {row['method']: row for row in repeat_totals if int(row['repeat']) == repeat}
        if set(selected) != set(ARMS):
            continue
        k, krc = selected['K'], selected['KRC']
        repeat_comparisons.append(dict(repeat=repeat, requests=int(k['requests']),
            K_latency_s=float(k['request_latency_s']), KRC_latency_s=float(krc['request_latency_s']),
            request_time_reduction_pct=reduction(float(k['request_latency_s']), float(krc['request_latency_s'])),
            K_calls=int(k['verification_calls']), KRC_calls=int(krc['verification_calls'])))
    return dict(**result, comparison=comparison, valid_requests=len(valid),
                source_summary=source, repeat_comparisons=repeat_comparisons), per_request


def role_probe_tables(manager):
    aggregates, responses, identities, effects = [], [], [], []
    for role, key in (('A_learning', 'learning'), ('R_reference', 'reference')):
        selected = [{'step': row['step'], **row[key]} for row in manager.get('probes', []) if row.get(key)]
        aggregate, response, identity = BASE.probe_tables({'probes': selected})
        for collection in (aggregate, response, identity):
            renamed = []
            for row in collection:
                values = {name.replace('KC', 'KRC'): value for name, value in row.items()}
                if 'candidate' in values:
                    values['candidate'] = values['candidate'].replace('KC', 'KRC')
                renamed.append({'role': role, **values})
            collection[:] = renamed
        aggregates.extend(aggregate)
        responses.extend(response)
        identities.extend(identity)
        for step in sorted({row['step'] for row in identity}):
            blocks = [row for row in identity if row['step'] == step]
            effects.append(dict(role=role, step=step, blocks=len(blocks),
                blocks_improved=sum(row['KRC_minus_K_prefix'] > 0 for row in blocks),
                blocks_regressed=sum(row['KRC_minus_K_prefix'] < 0 for row in blocks),
                blocks_tied=sum(row['KRC_minus_K_prefix'] == 0 for row in blocks),
                prefix_gain=sum(max(row['KRC_minus_K_prefix'], 0) for row in blocks),
                prefix_loss=sum(max(-row['KRC_minus_K_prefix'], 0) for row in blocks),
                repairs_retained=sum(row['repair_retained'] for row in blocks),
                repairs_lost=sum(row['repair_lost'] for row in blocks),
                repairs_new=sum(row['repair_new'] for row in blocks)))
    return aggregates, responses, identities, effects


def reference_exposure(raw):
    exposure = []
    for path in sorted((raw / 'correction').glob('window_*.json')):
        record = read(path)
        details = record['details']
        reference = details['reference']
        binding = details['schedule_binding']
        metadata = reference['responses']
        post = reference.get('post_k') or {}
        protection = post.get('totals', {})
        exposure.append(dict(
            step=record['summary']['step'], status=record['summary']['status'],
            pass_index=binding['pass_index'], batch_index=binding['batch_index'],
            reference_batch_index=binding['reference_batch_index'],
            learning_record_ids='|'.join(binding['learning_record_ids']),
            reference_record_ids='|'.join(binding['reference_record_ids']),
            captured_reference_responses=len(metadata),
            sampled_valid_anchors=sum(sum(bool(value) for value in BASE.flatten(row['keep'])) for row in metadata),
            available_proposal_occurrences=sum(sum(value > 0 for block in row['validity_weights'][0]
                                                   for value in block) for row in metadata),
            protected_occurrences=protection.get('protected_positions'),
            protected_weight_sum=protection.get('protected_weight'),
            strict_negative_margin_occurrences=protection.get('strict_margin_violations'),
            wrong_tie_occurrences=protection.get('wrong_at_tie_positions'),
            V_backward_completed=reference['work']['violation_backward_completed'],
            pre_reference_integrity_passed=(bool(details.get('reference_pre_integrity'))
                and all(details['reference_pre_integrity'].values())),
        ))
    return exposure


def activation_table(raw):
    path = raw / 'evaluation/ACTIVATIONS.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    output = []
    for ordinal, record in enumerate(records):
        publication = record.get('native_status', {}).get('publication', {})
        output.append(dict(
            activation_ordinal=ordinal, panel_ordinal=record['panel_ordinal'],
            repeat=record['repeat'], method=record['method'], checkpoint_id=record['checkpoint_id'],
            activation_s=record['activation_s'], native_activation_s=record.get('native_activation_s'),
            native_load_s=publication.get('native_load_s'), helper_refresh_s=publication.get('helper_refresh_s'),
            cache_maintenance_s=publication.get('cache_maintenance_s'),
            errors='; '.join(record.get('errors', [])),
        ))
    return output


def correction_summary(steps):
    complete = [row for row in steps if row['window_status'] == 'completed']
    active = [row for row in complete if row['active']]
    ratios = [row['realized_correction_to_K_norm_ratio'] for row in complete
              if finite_number(row.get('realized_correction_to_K_norm_ratio'))]
    active_ratios = [row['realized_correction_to_K_norm_ratio'] for row in active
                     if finite_number(row.get('realized_correction_to_K_norm_ratio'))]
    return dict(completed_windows=len(complete), observed_windows=len(steps),
        applied=len(active), realized_nonzero=sum(row['BF16_nonzero_correction'] for row in complete),
        applied_rounded_zero=sum(not row['BF16_nonzero_correction'] for row in active),
        positive_V=sum(finite_number(row.get('V')) and row['V'] > 0 for row in complete),
        zero_V=sum(finite_number(row.get('V')) and row['V'] == 0 for row in complete),
        branches={name: sum(row['correction_status'] == name for row in steps)
                  for name in sorted({row['correction_status'] for row in steps if row['correction_status']})},
        median_realized_to_K_norm=median(ratios) if ratios else None,
        active_median_realized_to_K_norm=median(active_ratios) if active_ratios else None,
        minimum_realized_to_K_norm=min(ratios) if ratios else None,
        maximum_realized_to_K_norm=max(ratios) if ratios else None)


def cost_summary(costs, event, manager, serving, activations):
    component = manager.get('costs', {})
    intrinsic = sum(component.get(name, 0) for name in ('pre_reference_s', 'post_k_s', 'correction_s'))
    diagnostics = sum(component.get(name, 0) for name in
                      ('post_c_reference_s', 'post_k_learning_s', 'post_c_learning_s'))
    integrity = sum(component.get(name, 0) for name in
                    ('integrity_s', 'pre_reference_integrity_s', 'first_learning_gate_s'))
    event_wall = costs.get('event_wall_s')
    execution = serving['source_summary'].get('execution', {})
    activation_s = sum(row['activation_s'] for row in activations)
    if activations and abs(activation_s - execution['activation_s']) > 1e-9:
        raise ValueError('Activation rows and execution total disagree')
    return dict(
        startup_s=costs.get('startup', {}).get('wall_s'),
        fresh_loading_and_initialization_s=costs.get('loading_and_initialization_s'),
        event_wall_s=event_wall, training_windows_s=event.get('costs', {}).get('training_s'),
        intrinsic_reference_and_correction_s=intrinsic,
        pre_reference_s=component.get('pre_reference_s'), post_K_reference_s=component.get('post_k_s'),
        correction_s=component.get('correction_s'), diagnostic_rechecks_s=diagnostics,
        post_K_A_s=component.get('post_k_learning_s'), post_C_A_s=component.get('post_c_learning_s'),
        post_C_R_s=component.get('post_c_reference_s'), integrity_s=integrity,
        checkpoint_persistence_s=event.get('costs', {}).get('checkpoint_save_s'),
        export_s=event.get('costs', {}).get('export_s'), backend_loading_s=costs.get('backend_loading_s'),
        benchmark_activation_count=len(activations), all_benchmark_activation_s=activation_s,
        benchmark_activation_count_by_arm={arm: sum(row['method'] == arm for row in activations) for arm in ARMS},
        benchmark_activation_s_by_arm={arm: sum(row['activation_s'] for row in activations if row['method'] == arm) for arm in ARMS},
        first_measured_KRC_activation_s=next((row['activation_s'] for row in activations if row['method'] == 'KRC'), None),
        benchmark_weight_validation_s=execution.get('weight_validation_s'),
        serving_panel_time_saved_s=(serving['K']['summed_mean_request_latency_s']
                                   - serving['KRC']['summed_mean_request_latency_s']),
        event_wall_excluding_diagnostic_replays_s=event_wall-diagnostics if finite_number(event_wall) else None,
        actual_campaign_wall_s=costs.get('actual_campaign_wall_s'),
        no_fresh_matched_K_training_time=True,
        timer_scope='Nested timers must not be summed into event time again. Event minus replays still includes integrity checks and sparse A prefix capture.')


def analyze(root=HERE, raw=None):
    root = Path(root)
    raw = Path(raw) if raw else root / 'raw/campaign_001'
    if not raw.is_dir():
        raise FileNotFoundError('Retained raw records have not arrived: ' + str(raw))
    output = root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    status = read(raw / 'STATUS.json', {'status': 'unavailable'})
    costs = read(raw / 'COSTS.json', {})
    event = read(raw / 'KRC/events/event_001/RESULT.json', {})
    manager = read(raw / 'CORRECTION_EVIDENCE.json', costs.get('correction', {}))
    gate = read(raw / 'FIRST_WINDOW_REPRODUCTION.json', {})
    serving, requests = native_summary(raw)
    steps = BASE.step_table(manager)
    probes, responses, identities, effects = role_probe_tables(manager)
    learning = BASE.exposure_table(event)
    reference = reference_exposure(raw)
    by_step = {row['step']: row for row in reference}
    for row in steps:
        binding = by_step.get(row['step'], {})
        row['learning_record_ids'] = binding.get('learning_record_ids', row.get('record_ids'))
        row['reference_record_ids'] = binding.get('reference_record_ids')
        row['protection_role'] = 'R_reference'
        row['V_and_post_K_totals_role'] = 'R_reference'
    activations = activation_table(raw)
    work, counts = manager.get('work', {}), manager.get('counts', {})
    history = read(HERE.parent / 'dflash_kc_event_20261027/analysis/SUMMARY.json', {})
    historical_kc_calls = history.get('serving', {}).get('KC', {}).get('verification_calls')
    historical_kc = dict(verification_calls=historical_kc_calls,
        source='../dflash_kc_event_20261027/analysis/SUMMARY.json',
        scope='Historical same-panel call context only; no historical timing comparison')
    summary = dict(
        status=status, first_learning_window_exact=gate.get('exact_equal'),
        serving=serving, correction=correction_summary(steps),
        counts=counts, work=work, forward_calls=manager.get('forward_calls', {}),
        core_response_forwards=sum(work.get(name, 0) for name in FORWARD_KEYS),
        core_backward_equivalents=counts.get('K_backward_completed', 0) + work.get('violation_backward_completed', 0),
        diagnostic_response_forwards=sum(work.get(name, 0) for name in
            ('post_c_reference_response_forwards', 'post_k_learning_response_forwards', 'post_c_learning_response_forwards')),
        startup=costs.get('startup', {}), startup_work=costs.get('startup_work_counts', {}),
        qualification_optimizer_steps=costs.get('new_qualification_optimizer_steps'),
        event_optimizer_steps=event.get('optimizer_steps'), event_presentations=event.get('response_presentations'),
        learning_exposure=event.get('selection_totals', {}),
        learning_sampled_anchors=sum(row['sampled_valid_anchors'] for row in learning),
        reference_exposure=dict(
            windows=len(reference), responses=sum(row['captured_reference_responses'] for row in reference),
            sampled_valid_anchors=sum(row['sampled_valid_anchors'] for row in reference),
            available_proposal_occurrences=sum(row['available_proposal_occurrences'] for row in reference),
            protected_occurrences=sum(row['protected_occurrences'] or 0 for row in reference),
            protected_weight_sum=sum(row['protected_weight_sum'] or 0 for row in reference),
            negative_margin_occurrences=sum(row['strict_negative_margin_occurrences'] or 0 for row in reference),
            wrong_tie_occurrences=sum(row['wrong_tie_occurrences'] or 0 for row in reference),
            all_pre_reference_integrity_passed=bool(reference) and all(row['pre_reference_integrity_passed'] for row in reference)),
        probes=probes, probe_effects=effects, costs=cost_summary(costs, event, manager, serving, activations),
        peak_allocated_bytes=costs.get('peak_allocated_bytes'), peak_reserved_bytes=costs.get('peak_reserved_bytes'),
        historical_KC=historical_kc,
        scope='One event, one seed, reused development panel; reference responses are additional training exposure.')
    for name, values, fallback in (
        ('STEPS.csv', steps, ['step', 'window_status']),
        ('PROBE_SUMMARY.csv', probes, ['step', 'role', 'candidate']),
        ('PROBE_RESPONSES.csv', responses, ['step', 'role', 'record_id']),
        ('REPAIR_IDENTITIES.csv', identities, ['step', 'role', 'record_id', 'block']),
        ('PROBE_EFFECTS.csv', effects, ['step', 'role']),
        ('LEARNING_EXPOSURE.csv', learning, ['optimizer_step']),
        ('REFERENCE_EXPOSURE.csv', reference, ['step']),
        ('PER_REQUEST.csv', requests, ['request_id', 'valid']),
        ('REPEAT_COMPARISON.csv', serving['repeat_comparisons'], ['repeat']),
        ('ACTIVATIONS.csv', activations, ['activation_ordinal', 'method', 'activation_s']),
    ):
        write_csv(output / name, values, fallback)
    (output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (root / 'REPORT.md').write_text(render_report(summary))
    return summary


def render_report(s):
    serving, correction, costs = s['serving'], s['correction'], s['costs']
    comparison = serving['comparison']
    source = serving['source_summary']
    complete = (s['status'].get('status') == 'completed' and serving['valid_requests'] == 64
                and source.get('valid_timed_generations') == 384)
    lookup = {(row['step'], row['role'], row['candidate']): row for row in s['probes']}
    before_r = lookup.get((64, 'R_reference', 'ordinary_K_candidate'))
    after_r = lookup.get((64, 'R_reference', 'corrected_KRC'))
    effect_a = next((row for row in s['probe_effects']
                     if row['step'] == 64 and row['role'] == 'A_learning'), None)
    local_note = ''
    if before_r and after_r and effect_a:
        direction = ('rises' if after_r['V'] > before_r['V'] else
                     'falls' if after_r['V'] < before_r['V'] else 'is unchanged')
        local_note = (
            f"At step 64, reference V {direction}; lost reference-prefix occurrences change from "
            f"{before_r['old_prefix_tokens_lost']} to {after_r['old_prefix_tokens_lost']}. On learning A "
            f"at that step, the correction loses {effect_a['repairs_lost']} useful repairs and adds "
            f"{effect_a['repairs_new']}. These local measurements do not establish a monotonic preservation "
            'guarantee; the regressions remain visible separately from the final native result.'
        )
    lines = ['# KRC: one-event reference-response correction', '']
    if complete:
        calls = comparison['call_reduction_pct']
        outcome = ('improved drafting on this panel' if calls > 0 else
                   'did not improve drafting over retained K1')
        lines += [f"**KRC {outcome}: {serving['K']['verification_calls']:,} → "
                  f"{serving['KRC']['verification_calls']:,} verification calls "
                  f"({calls:+.4f}% call reduction).** Requests improve/regress/tie: "
                  f"**{comparison['improved']}/{comparison['regressed']}/{comparison['tied']}**.", '']
    else:
        lines += [f"**Status: {s['status'].get('status', 'unavailable')}.** The full screen is incomplete; "
                  'available measurements do not establish a completed-screen outcome.', '']
    lines += [
        'KRC starts from released DFlash with the fresh seed-17 optimizer and uses the original 128 responses, '
        'two passes and unchanged K learning objective. The correction protects the next disjoint response pair '
        'in each pass, using its original anchors and freshly captured prefixes. The reference responses are '
        '**additional training exposure**, not held-out evaluation. The correction formula, BF16 path, optimizer '
        'and serving settings remain unchanged. No new feedback or F/O/K retraining was performed.', '',
        f"First learning-window ordinary-K reproduction: **{s['first_learning_window_exact']}**. "
        f"Completed update windows: **{correction['completed_windows']}/128**; event presentations/Adam steps: "
        f"**{s['event_presentations']}/{s['event_optimizer_steps']}**.", '',
        '## Fresh native K1/KRC comparison', '',
        f"Validated paired requests: **{serving['valid_requests']}/64**; valid timed executions: "
        f"**{source.get('valid_timed_generations')}/384**. Tokens and termination are checked against retained F; "
        'each checkpoint must have repeat-consistent counts. Count totals use one trajectory per request. '
        'Latency pools the three-repeat mean for each request.', '',
        '| Arm | Verification calls | Accepted draft tokens | Calls / 1,000 tokens | Progress / call | Summed mean time (s) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        row = serving[arm]
        lines.append(f"| {arm} | {row['verification_calls']} | {row['accepted_draft_tokens']} | "
                     f"{show(row['calls_per_1000_output_tokens'])} | {show(row['committed_progress_per_call'])} | "
                     f"{show(row['summed_mean_request_latency_s'],6)} |")
    lines += ['', f"Pooled request-time reduction: **{show(comparison['request_time_reduction_pct'])}%**. "
              f"Gross calls saved/added: **{comparison['gross_saved_calls']}/{comparison['gross_added_calls']}**; "
              f"net saved: **{comparison['net_saved_calls']}**. Serving latency excludes training, reference passes, "
              'loading and checkpoint switching. This is the reused development panel, not a fresh-input confirmation.', '',
              '| Repeat | K time (s) | KRC time (s) | Time reduction |', '|---:|---:|---:|---:|']
    for row in serving['repeat_comparisons']:
        lines.append(f"| {row['repeat']} | {show(row['K_latency_s'],6)} | {show(row['KRC_latency_s'],6)} | "
                     f"{show(row['request_time_reduction_pct'])}% |")
    historical = s['historical_KC']['verification_calls']
    lines += ['', f"Earlier KC used **{historical}** calls on this panel; that is historical count context only. "
              'The primary comparison is the fresh K1/KRC pair above. Historical KC timings are not combined '
              'with this run.', '',
              '## Local behavior on learning A and reference R', '',
              f"Correction branches: `{correction['branches']}`. Applied/nonzero-after-BF16 corrections: "
              f"**{correction['applied']}/{correction['realized_nonzero']}** out of {correction['completed_windows']}; "
              f"applied but rounded to zero: **{correction['applied_rounded_zero']}**. Median realized correction/K-step "
              f"norm ratio: **{show(correction['median_realized_to_K_norm'],6)}**, or "
              f"**{show(correction['active_median_realized_to_K_norm'],6)}** among applied corrections. Every window's "
              'V, norms, no-op branch and costs remain in STEPS.csv.', '',
              '| Step | Role | Candidate | V | Useful repairs | Earlier errors / blocks | Lost old prefix | Net prefix gain | Negative margins | Wrong ties |',
              '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in s['probes']:
        lines.append(f"| {row['step']} | {row['role']} | {row['candidate']} | {show(row['V'],9)} | "
                     f"{row['first_errors_repaired_with_prefix_intact']}/{row['old_first_errors']} | "
                     f"{row['new_earlier_errors']}/{row['blocks']} | "
                     f"{row['old_prefix_tokens_lost']}/{row['old_matching_prefix_tokens']} | "
                     f"{row['net_matching_prefix_change']:+d} | {row['strict_margin_violations']} | {row['wrong_at_tie_positions']} |")
    lines += ['', '| Step | Role | Repairs retained / lost / new | Blocks improved / regressed / tied | Prefix gain / loss vs K candidate |',
              '|---:|---|---:|---:|---:|']
    for row in s['probe_effects']:
        lines.append(f"| {row['step']} | {row['role']} | {row['repairs_retained']}/{row['repairs_lost']}/{row['repairs_new']} | "
                     f"{row['blocks_improved']}/{row['blocks_regressed']}/{row['blocks_tied']} | "
                     f"{row['prefix_gain']}/{row['prefix_loss']} |")
    lines += ['', 'A is the current learning pair; R is the separate correction pair. Each ordinary K candidate '
              'starts from that window\'s current KRC weights. The later candidates are not historical K64/K128 '
              'checkpoints. All local counts concern overlapping sampled-block occurrences, not independent '
              'requests or realized serving gains. Lower reference V alone does not establish successful transfer.', '',
              local_note, '',
              '## Exposure and costs', '',
              f"K sampled anchors: **{s['learning_sampled_anchors']}**; reference sampled anchors: "
              f"**{s['reference_exposure']['sampled_valid_anchors']}** across "
              f"**{s['reference_exposure']['responses']}** additional reference-response presentations, each evaluated "
              'once before and once after K (512 intrinsic reference forwards in the complete run). Reference protection '
              f"covers **{s['reference_exposure']['protected_occurrences']}** prefix occurrences, with triangular "
              f"weight sum **{show(s['reference_exposure']['protected_weight_sum'],0)}**. Ordinary K exposure is counted "
              'once per optimizer-boundary row. Reference exposure is recorded separately; this is not an equal-compute '
              'or equal-exposure comparison with K.', '',
              '| Cost | Seconds | Scope |', '|---|---:|---|']
    for key, title, scope in (
        ('startup_s', 'Disposable startup', 'Separate; two forward/backwards, zero Adam'),
        ('fresh_loading_and_initialization_s', 'Fresh initialization', 'Before main event'),
        ('event_wall_s', 'Full KRC event', 'Training, references, correction, probes, checks and publication'),
        ('training_windows_s', 'Training windows', 'Nested in event'),
        ('intrinsic_reference_and_correction_s', 'Reference capture/gradients and correction', 'Intrinsic added work; nested in event'),
        ('pre_reference_s', 'Pre-step reference capture', 'Includes retained feature loading/capture'),
        ('post_K_reference_s', 'Post-K reference gradients', 'Nested intrinsic work'),
        ('correction_s', 'Direct parameter correction', 'Nested intrinsic work'),
        ('diagnostic_rechecks_s', 'Eighteen A/R diagnostic replays', 'Diagnostic only'),
        ('integrity_s', 'State-integrity and reproduction checks', 'Instrumentation'),
        ('checkpoint_persistence_s', 'Checkpoint persistence', 'Nested in event'),
        ('export_s', 'Final export', 'Nested in event'),
        ('backend_loading_s', 'Serving backend loading', 'Benchmark setup'),
        ('all_benchmark_activation_s', 'All benchmark checkpoint activations', 'Benchmark-only switching; excluded from request latency'),
        ('benchmark_weight_validation_s', 'Benchmark weight-identity checks', 'Separate from activation and serving latency'),
        ('actual_campaign_wall_s', 'Campaign process wall', 'All execution including timing repeats'),
    ):
        lines.append(f"| {title} | {show(costs.get(key),3)} | {scope} |")
    lines += ['', f"Core work: **{s['core_response_forwards']}/786 response forwards**, "
              f"**{s['core_backward_equivalents']}/≤512 backward equivalents**, "
              f"**{s['counts'].get('Adam_completed')}/128 Adam steps**, and "
              f"**{s['work'].get('direct_corrections')}/≤128 corrections**. The "
              f"**{s['diagnostic_response_forwards']}/18** local diagnostic forwards are included. "
              f"New qualification optimizer steps: **{s['qualification_optimizer_steps']}**. Startup and actual "
              'head/layer recomputation counts are retained separately in SUMMARY.json.', '',
              f"The benchmark made **{costs['benchmark_activation_count']}** checkpoint publications "
              f"(`{costs['benchmark_activation_count_by_arm']}`), not one per generation: consecutive executions "
              'can reuse the active checkpoint. Its activation sum is benchmark overhead, not a deployment charge '
              f"per request. The first measured KRC activation took **{show(costs['first_measured_KRC_activation_s'],3)}s**. "
              'Every publication is retained in ACTIVATIONS.csv.', '',
              f"Event time excluding diagnostic replays: **{show(costs['event_wall_excluding_diagnostic_replays_s'],3)}s**; "
              'this still includes integrity checks and sparse A prefix capture. Component timers are nested and '
              'must not be added to the event total again. No fresh matched K training-cost measurement exists. '
              'Reusing feedback/features incurs no new collection or teacher-preparation work here; it does not '
              'erase their historical cost.', '',
              f"The ready-checkpoint serving gain on this panel is **{show(costs['serving_panel_time_saved_s'],3)}s**, "
              f"while measured intrinsic reference/correction work is **{show(costs['intrinsic_reference_and_correction_s'],3)}s**. "
              'This result does not establish an adaptation-inclusive speedup. There is no fresh matched K training-cost '
              'control, and no benefit is extrapolated to future requests.', '',
              f"Peak allocated/reserved GPU memory: **{show(s['peak_allocated_bytes']/1e9 if s['peak_allocated_bytes'] else None,3)} / "
              f"{show(s['peak_reserved_bytes']/1e9 if s['peak_reserved_bytes'] else None,3)} GB**.", '',
              '## Scope and retained tables', '',
              'This one-seed, one-event screen separates local preservation from final serving performance. It does '
              'not establish feedback-window robustness, generalization to a fresh panel, or an adaptation-inclusive '
              'online speedup. No tuning, subsequent event or replacement mechanism is implied.', '',
              '- `analysis/SUMMARY.json`, `STEPS.csv`, `PROBE_SUMMARY.csv`, `PROBE_RESPONSES.csv`, `PROBE_EFFECTS.csv`, and `REPAIR_IDENTITIES.csv`.',
              '- `analysis/LEARNING_EXPOSURE.csv`, `REFERENCE_EXPOSURE.csv`, `PER_REQUEST.csv`, `REPEAT_COMPARISON.csv`, and `ACTIVATIONS.csv`.',
              '- Every timing repeat and native record remains under `raw/campaign_001/evaluation/`.',
              '- Source/result audit, first-step reproduction and storage records are reported separately.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    analyze(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE,
            Path(sys.argv[2]) if len(sys.argv) > 2 else None)
