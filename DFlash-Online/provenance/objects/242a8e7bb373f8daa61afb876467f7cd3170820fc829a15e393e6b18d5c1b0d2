"""Build the KC report from retained records; no model or remote operations."""
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write_csv(path, rows, fallback):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or fallback
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def show(value, digits=4):
    return f'{value:.{digits}f}' if finite_number(value) else 'unavailable'


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def flatten(rows):
    return [value for row in rows for value in row]


def step_table(manager):
    rows = []
    for record in manager.get('steps', []):
        row = {key: record.get(key) for key in
               ('step', 'status', 'actual_Adam_steps_completed', 'window_wall_s', 'integrity_s')}
        row['record_ids'] = '|'.join(record.get('record_ids', []))
        row.update(record.get('correction') or {})
        row['window_status'] = record.get('status')
        row['correction_status'] = (record.get('correction') or {}).get('status')
        row['active'] = row['correction_status'] == 'applied'
        row['BF16_nonzero_correction'] = (
            finite_number(row.get('realized_correction_norm')) and row['realized_correction_norm'] > 0)
        row.update({'post_K_' + key: value for key, value in (record.get('post_K_totals') or {}).items()})
        row.update({'cost_' + key: value for key, value in record.get('costs', {}).items()})
        row.update({'work_' + key: value for key, value in record.get('work', {}).items()})
        row['integrity_passed'] = all(
            all(value.values()) if isinstance(value, dict) else bool(value)
            for value in record.get('integrity', {}).values())
        rows.append(row)
    return rows


def probe_tables(manager):
    aggregate, responses, identities = [], [], []
    for probe in manager.get('probes', []):
        step = probe['step']
        for state_name, candidate in (('post_k', 'ordinary_K_candidate'), ('post_c', 'corrected_KC')):
            state = probe.get(state_name)
            if not state:
                continue
            row = dict(step=step, candidate=candidate, V=state['V'],
                       violation_numerator=state['numerator'], protection_denominator=state['denominator'],
                       **state['totals'])
            row['earlier_error_block_fraction'] = ratio(row['new_earlier_errors'], row['blocks'])
            row['lost_old_prefix_fraction'] = ratio(row['old_prefix_tokens_lost'], row['old_matching_prefix_tokens'])
            row['useful_repair_fraction'] = ratio(row['first_errors_repaired_with_prefix_intact'], row['old_first_errors'])
            if state_name == 'post_c':
                row.update(state.get('repair_totals', {}))
            aggregate.append(row)
            for ordinal, response in enumerate(state['responses']):
                responses.append(dict(step=step, candidate=candidate, response_ordinal=ordinal,
                    **{key: value for key, value in response.items() if not isinstance(value, (list, dict))}))
        if not probe.get('post_k') or not probe.get('post_c'):
            continue
        for ordinal, (metadata, k, kc) in enumerate(zip(
                probe['responses'], probe['post_k']['responses'], probe['post_c']['responses'], strict=True)):
            old_lengths = [sum(weight > 0 for weight in block) for block in metadata['protected_weights'][0]]
            k_lengths = flatten(k['matching_prefix_lengths'])
            kc_lengths = flatten(kc['matching_prefix_lengths'])
            anchors = flatten(metadata['anchors'])
            for block, (anchor, old, ordinary, corrected) in enumerate(zip(
                    anchors, old_lengths, k_lengths, kc_lengths, strict=True)):
                repaired_k, repaired_kc = ordinary > old, corrected > old
                identities.append(dict(step=step, response_ordinal=ordinal, record_id=metadata['record_id'],
                    block=block, anchor=anchor, original_prefix_length=old,
                    K_prefix_length=ordinary, KC_prefix_length=corrected,
                    K_useful_repair=repaired_k, KC_useful_repair=repaired_kc,
                    repair_retained=repaired_k and repaired_kc,
                    repair_lost=repaired_k and not repaired_kc,
                    repair_new=not repaired_k and repaired_kc,
                    K_earlier_error=ordinary < old, KC_earlier_error=corrected < old,
                    K_old_prefix_lost=max(old-ordinary, 0), KC_old_prefix_lost=max(old-corrected, 0),
                    KC_minus_K_prefix=corrected-ordinary))
    return aggregate, responses, identities


def exposure_table(event):
    rows, window = [], []
    for record in event.get('records', []):
        window.append(record)
        if not record['optimizer_stepped']:
            continue
        anchors = sum(sum(bool(value) for value in flatten(draw['keep_mask']))
                      for presentation in window for draw in presentation['sampled_anchors'])
        metrics = record['metrics']
        retained = {key: value for key, value in metrics.items()
                    if key.startswith(('available_', 'selected_', 'first_error_position_')) or key in (
                        'supervised_position_exposure', 'main_loss_numerator', 'loss_denominator',
                        'empty_blocks', 'no_error_blocks', 'response_presentations')}
        rows.append(dict(optimizer_step=record['optimizer_step_after'],
                         record_ids='|'.join(item['record_id'] for item in window),
                         presentations=len(window), sampled_valid_anchors=anchors, **retained))
        window = []
    return rows


def cost_summary(costs, event, manager, serving, raw):
    event_costs = event.get('costs', costs.get('event', {}))
    correction = manager.get('costs', {})
    event_wall = costs.get('event_wall_s')
    probe = correction.get('post_c_s')
    intrinsic = (correction.get('post_k_s', 0) + correction.get('correction_s', 0)) if correction else None
    recheck_excluded = event_wall-probe if finite_number(event_wall) and finite_number(probe) else None
    activation_rows = []
    path = raw/'evaluation/ACTIVATIONS.jsonl'
    if path.is_file():
        activation_rows = [json.loads(line) for line in path.read_text().splitlines()]
    first_kc_activation = next((row['activation_s'] for row in activation_rows if row['method']=='KC'), None)
    kc_serving = serving.get('KC', {}).get('summed_mean_request_latency_s')
    k_serving = serving.get('K', {}).get('summed_mean_request_latency_s')
    illustrative = (recheck_excluded+first_kc_activation+kc_serving
                    if all(finite_number(v) for v in (recheck_excluded,first_kc_activation,kc_serving)) else None)
    return dict(
        startup_s=costs.get('startup',{}).get('wall_s'),
        fresh_loading_and_initialization_s=costs.get('loading_and_initialization_s'),
        event_wall_s=event_wall, training_windows_s=event_costs.get('training_s'),
        intrinsic_post_K_and_correction_s=intrinsic,
        post_K_s=correction.get('post_k_s'), correction_s=correction.get('correction_s'),
        corrected_probe_s=probe, integrity_s=correction.get('integrity_s'),
        partial_capture_s=correction.get('prefix_chunk_capture_s'),
        checkpoint_persistence_s=event_costs.get('checkpoint_save_s'), export_s=event_costs.get('export_s'),
        event_wall_excluding_corrected_rechecks_s=recheck_excluded,
        backend_loading_s=costs.get('backend_loading_s'),
        all_benchmark_publications_s=serving.get('execution',{}).get('activation_s'),
        first_measured_KC_publication_s=first_kc_activation,
        K_serving_panel_s=k_serving, KC_serving_panel_s=kc_serving,
        checkpoint_ready_panel_time_saved_s=k_serving-kc_serving if all(finite_number(v) for v in (k_serving,kc_serving)) else None,
        illustrative_KC_event_activation_and_panel_s=illustrative,
        illustrative_total_scope=('One measured event minus six corrected-state rechecks, plus one measured '
            'KC activation and one 64-request serving pass reconstructed from repeat means. '
            'Includes integrity instrumentation/persistence/export; excludes historical feedback/teacher preparation, '
            'startup/loading and benchmark-only repetitions/switches. Not a matched net-online comparison.'),
        actual_campaign_wall_s=costs.get('actual_campaign_wall_s'),
        nested_cost_warning='Component timers are nested in training/event totals and must not be added again.',
        no_fresh_matched_K_training_time=True,
    )


def analyze(root, raw=None):
    root=Path(root); raw=Path(raw) if raw else root/'raw/campaign_001'
    output=root/'analysis'; output.mkdir(parents=True,exist_ok=True)
    status=read(raw/'STATUS.json',{'status':'unavailable'})
    event=read(raw/'KC/events/event_001/RESULT.json',{})
    costs=read(raw/'COSTS.json',{})
    manager=read(raw/'CORRECTION_EVIDENCE.json',costs.get('correction',{}))
    serving=read(raw/'evaluation/report/SUMMARY.json',{})
    first_gate=read(raw/'FIRST_WINDOW_REPRODUCTION.json',{})
    steps=step_table(manager)
    probes,responses,identities=probe_tables(manager)
    exposure=exposure_table(event)
    completed=[row for row in steps if row['window_status']=='completed']
    active=[row for row in completed if row['active']]
    realized=[row for row in completed if row['BF16_nonzero_correction']]
    ratios=[row['realized_correction_to_K_norm_ratio'] for row in completed
            if finite_number(row.get('realized_correction_to_K_norm_ratio'))]
    active_ratios=[row['realized_correction_to_K_norm_ratio'] for row in active
                   if finite_number(row.get('realized_correction_to_K_norm_ratio'))]
    branch_counts={name:sum(row['correction_status']==name for row in steps)
                   for name in sorted({row['correction_status'] for row in steps if row['correction_status']})}
    probe_effects=[]
    for step in sorted({row['step'] for row in identities}):
        selected=[row for row in identities if row['step']==step]
        probe_effects.append(dict(step=step,blocks=len(selected),
            blocks_improved=sum(row['KC_minus_K_prefix']>0 for row in selected),
            blocks_regressed=sum(row['KC_minus_K_prefix']<0 for row in selected),
            blocks_tied=sum(row['KC_minus_K_prefix']==0 for row in selected),
            prefix_gain=sum(max(row['KC_minus_K_prefix'],0) for row in selected),
            prefix_loss=sum(max(-row['KC_minus_K_prefix'],0) for row in selected),
            repairs_retained=sum(row['repair_retained'] for row in selected),
            repairs_lost=sum(row['repair_lost'] for row in selected),
            repairs_new=sum(row['repair_new'] for row in selected)))
    measured_costs=cost_summary(costs,event,manager,serving,raw)
    counts=manager.get('counts',{})
    summary=dict(status=status, first_window_exact=first_gate.get('exact_equal'),
        completed_windows=len(completed),observed_windows=len(steps),active_corrections=len(active),
        activation_fraction=ratio(len(active),len(completed)),branch_counts=branch_counts,
        BF16_nonzero_corrections=len(realized),
        BF16_nonzero_fraction=ratio(len(realized),len(completed)),
        applied_but_rounded_zero=sum(not row['BF16_nonzero_correction'] for row in active),
        positive_V_windows=sum(finite_number(row.get('V')) and row['V'] > 0 for row in completed),
        zero_V_windows=sum(finite_number(row.get('V')) and row['V'] == 0 for row in completed),
        median_realized_correction_to_K_norm_ratio=median(ratios) if ratios else None,
        active_median_realized_correction_to_K_norm_ratio=median(active_ratios) if active_ratios else None,
        min_realized_correction_to_K_norm_ratio=min(ratios) if ratios else None,
        max_realized_correction_to_K_norm_ratio=max(ratios) if ratios else None,
        actual_Adam_steps=counts.get('Adam_completed'), native_counts=counts,
        event_optimizer_steps=event.get('optimizer_steps'),event_presentations=event.get('response_presentations'),
        work=manager.get('work',{}),forward_calls=manager.get('forward_calls',{}),
        startup=costs.get('startup',{}),startup_work=costs.get('startup_work_counts',{}),
        new_qualification_optimizer_steps=costs.get('new_qualification_optimizer_steps'),
        supervised_exposure=event.get('selection_totals',{}),
        exposure_scope='Optimizer-boundary metrics counted once; sampled-block occurrences may overlap.',
        sampled_valid_anchors=sum(row['sampled_valid_anchors'] for row in exposure),
        correction_protected_occurrences=sum(row.get('post_K_protected_positions',0) for row in completed),
        correction_protected_weight_sum=sum(row.get('post_K_protected_weight',0) for row in completed),
        probes=probes,probe_effects=probe_effects,serving=serving,costs=measured_costs,
        peak_allocated_bytes=costs.get('peak_allocated_bytes'),peak_reserved_bytes=costs.get('peak_reserved_bytes'),
        source_root=str(raw),scope='One event, one seed and reused development panel; no fresh-panel confirmation or tuning.')
    write_csv(output/'STEPS.csv',steps,['step','window_status','correction_status'])
    write_csv(output/'PROBE_SUMMARY.csv',probes,['step','candidate'])
    write_csv(output/'PROBE_RESPONSES.csv',responses,['step','candidate','record_id'])
    write_csv(output/'REPAIR_IDENTITIES.csv',identities,['step','record_id','block'])
    write_csv(output/'PROBE_EFFECTS.csv',probe_effects,['step','blocks'])
    write_csv(output/'EXPOSURE.csv',exposure,['optimizer_step','presentations'])
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    report=render_report(summary)
    (root/'REPORT.md').write_text(report)
    return summary


def render_report(s):
    status=s['status']; serving=s['serving']; costs=s['costs']; probes=s['probes']
    comparison=serving.get('comparisons',{}).get('KC_vs_K',{})
    complete=(status.get('status')=='completed' and serving.get('valid_requests')==64
              and serving.get('valid_timed_generations')==384)
    lines=['# KC: one-event K-plus-correction experiment','']
    if complete:
        calls=comparison['call_reduction_pct']; timing=comparison['request_time_reduction_pct']
        change=(f'reduced verification calls by {calls:.4f}%' if calls>0 else
                f'increased verification calls by {-calls:.4f}%' if calls<0 else
                'left verification calls unchanged')
        conclusion=('The primary drafting outcome improved on this panel.' if calls>0 else
                    'The experiment did not demonstrate better drafting than ordinary K.')
        lines += [f"**KC {change}: {serving['K']['verification_calls']:,} calls for K1 versus "
                  f"{serving['KC']['verification_calls']:,} for KC.** {conclusion} "
                  f"The signed call-reduction metric is {calls:+.4f}%.", '',
                  f"Summed mean request times are {serving['K']['summed_mean_request_latency_s']:.6f}s "
                  f"for K1 and {serving['KC']['summed_mean_request_latency_s']:.6f}s for KC "
                  f"(signed reduction {timing:+.4f}%). "
                  'This descriptive timing difference is not evidence of a deployment speedup. '
                  'These are fresh comparisons with available checkpoints on the reused development panel.','']
    else:
        lines += [f"**Status: {status.get('status','unavailable')}.** The full screen is not complete. "
                  'Any available serving measurements below are a partial valid cohort; no completed-screen claim is made.','']
    lines += ['KC starts from released DFlash with the qualified fresh seed-17 optimizer, uses the original 128 '
        'responses/features and exact two-pass schedule, and applies the tested correction after each ordinary K step. '
        'K1 is the unchanged retained one-event checkpoint. No new feedback or F/O/K retraining is included. '
        'Models, BF16 arithmetic, native AdamW, constant LR 3e-4, weight decay 0.01, clipping 1, response accumulation 2, '
        'Flex attention, full history and greedy block-size-16/2,048-token-or-EOS serving remain fixed.', '',
        f"First-window exact reproduction: **{s['first_window_exact']}**. Completed windows: "
        f"**{s['completed_windows']}/128**; actual native Adam calls: **{s['actual_Adam_steps']}**; "
        f"committed training presentations: **{s['event_presentations']}**.", '']
    if serving:
        lines += ['## Fresh native comparison','',
            f"Valid requests: {serving.get('valid_requests')}/64; valid timed generations: "
            f"{serving.get('valid_timed_generations')}/384. Outputs and termination are checked against retained F; "
            'counts must be consistent across repeats. Training closes before timed evaluation.','',
            '| Arm | Verification calls | Sum of mean request time (s) | Accepted draft tokens | Calls / 1,000 tokens | Committed progress / call |',
            '|---|---:|---:|---:|---:|---:|']
        for arm in ('K','KC'):
            row=serving.get(arm,{})
            lines.append(f"| {arm} | {row.get('verification_calls')} | {show(row.get('summed_mean_request_latency_s'),6)} | "
                         f"{row.get('accepted_draft_tokens')} | {show(row.get('calls_per_1000_output_tokens'))} | "
                         f"{show(row.get('committed_progress_per_call'))} |")
        d=comparison.get('call_request_directions',{})
        lines += ['',f"Requests improved/regressed/tied on verification calls: **{d.get('improved')}/"
            f"{d.get('regressed')}/{d.get('tied')}**. Counts use one validated trajectory per request; "
            'latency uses three per-request repeats and pooled sums. Publication/loading and training are excluded '
            'from request latency. Historical F is the output reference, not a fresh latency comparator.','']
        outputs=serving.get('canonical_outputs',{})
        lines += [f"Canonical outputs: **{outputs.get('output_tokens')} tokens**; "
                  f"EOS requests: **{outputs.get('eos_requests')}**; "
                  f"output-cap hits: **{outputs.get('output_cap_hits')}**. "
                  'Per-request lengths, all repeats and invalid/incomplete slots remain in the raw evaluation tables.','']
    lines += ['## Correction behavior','',
        f"Applied correction decisions: **{s['active_corrections']}/{s['completed_windows']}** "
        f"({show(100*s['activation_fraction'] if s['activation_fraction'] is not None else None,2)}%). "
        f"Actual nonzero BF16 corrections: **{s['BF16_nonzero_corrections']}/{s['completed_windows']}**; "
        f"applied but rounded entirely to zero: **{s['applied_but_rounded_zero']}**. "
        f"Positive-V/zero-V windows: **{s['positive_V_windows']}/{s['zero_V_windows']}**. "
        f"Branches: `{s['branch_counts']}`. Median realized correction/K-step norm ratio across all windows: "
        f"**{show(s['median_realized_correction_to_K_norm_ratio'],6)}** "
        f"(range {show(s['min_realized_correction_to_K_norm_ratio'],6)}–"
        f"{show(s['max_realized_correction_to_K_norm_ratio'],6)}); among applied corrections: "
        f"**{show(s['active_median_realized_correction_to_K_norm_ratio'],6)}**.",'',
        'V is measured at the actual post-K candidate on the fixed pre-step prefix. It is not a held-out loss. '
        'Post-correction V and prefix outcomes are measured only at steps 1, 64 and 128. Every step, no-op, '
        'zero-gradient outcome, BF16-realized norm and component cost remains in STEPS.csv.','']
    if probes:
        lines += ['| Step | Candidate | V | Useful repairs | Earlier errors / blocks | Lost old-prefix tokens / original prefix | Net prefix gain | Wrong ties |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|']
        for p in probes:
            lines.append(f"| {p['step']} | {p['candidate']} | {show(p['V'],9)} | "
                f"{p['first_errors_repaired_with_prefix_intact']} / {p['old_first_errors']} | "
                f"{p['new_earlier_errors']} / {p['blocks']} | "
                f"{p['old_prefix_tokens_lost']} / {p['old_matching_prefix_tokens']} | "
                f"{p['net_matching_prefix_change']:+d} | {p['wrong_at_tie_positions']} |")
        lines += ['', '| Step | Useful repairs retained / lost / new | Blocks improve / regress / tie versus K | Prefix gain / loss versus K |',
                  '|---:|---:|---:|---:|']
        for p in s['probe_effects']:
            lines.append(f"| {p['step']} | {p['repairs_retained']} / {p['repairs_lost']} / {p['repairs_new']} | "
                f"{p['blocks_improved']} / {p['blocks_regressed']} / {p['blocks_tied']} | {p['prefix_gain']} / {p['prefix_loss']} |")
    lines += ['', 'Later probe K candidates start from the same current KC pre-step weights; they are not historical '
        'K64/K128 checkpoints. These are overlapping sampled-block occurrences on training responses, not independent '
        'examples or native serving results. REPAIR_IDENTITIES.csv identifies each retained/lost/new repair by '
        'step, response, block and anchor. A lower V does not by itself demonstrate useful learning. '
        'The repair table also exposes lost useful repairs; net improvement does not imply that every repair survives.', '',
        '## Exposure and cost','',
        f"K supervision: **{s['sampled_valid_anchors']} sampled valid anchors**; "
        f"available/selected occurrences **{show(s['supervised_exposure'].get('available_position_occurrences'),0)}/"
        f"{show(s['supervised_exposure'].get('selected_position_occurrences'),0)}**. Correction protection covers "
        f"**{s['correction_protected_occurrences']} old-prefix occurrences**, with total triangular weight "
        f"{show(s['correction_protected_weight_sum'],0)}. Exposure totals use optimizer-boundary rows once; "
        'intermediate accumulation rows are not double-counted.','',
        '| Cost | Seconds | Scope |','|---|---:|---|']
    for key,label,scope in (
        ('startup_s','Disposable startup','Two forward/backwards, zero Adam; separate from core budget'),
        ('fresh_loading_and_initialization_s','Fresh model and exact initialization','Separate setup'),
        ('event_wall_s','Complete KC event','Training, correction, probes, checks, persistence and export'),
        ('training_windows_s','Training windows','Nested inside event'),
        ('intrinsic_post_K_and_correction_s','Post-K gradients and direct corrections','Intrinsic added work; nested inside training'),
        ('corrected_probe_s','Six corrected-state rechecks','Diagnostic only; nested inside training'),
        ('integrity_s','State integrity and step-1 reproduction','Instrumentation nested inside training'),
        ('checkpoint_persistence_s','Checkpoint persistence','Nested inside event'),
        ('export_s','Final checkpoint export','Nested inside event'),
        ('backend_loading_s','Native backend loading','Benchmark setup'),
        ('all_benchmark_publications_s','All benchmark checkpoint switches','Benchmark only, excluded from serving'),
        ('actual_campaign_wall_s','Actual campaign wall','All current execution work, including repetitions')):
        lines.append(f"| {label} | {show(costs.get(key),3)} | {scope} |")
    lines += ['', f"For explicit cost accounting, the measured KC event excluding corrected-state rechecks costs "
        f"**{show(costs['event_wall_excluding_corrected_rechecks_s'],3)}s**. Adding one measured KC activation "
        f"({show(costs['first_measured_KC_publication_s'],3)}s) and one reconstructed 64-request KC serving pass "
        f"({show(costs['KC_serving_panel_s'],3)}s) gives **{show(costs['illustrative_KC_event_activation_and_panel_s'],3)}s**. "
        'This arithmetic retains integrity/persistence/export overhead, excludes runtime setup and historical feedback '
        'preparation, and excludes all benchmark-only extra repetitions/switches. It is not a matched end-to-end '
        'online speedup estimate: retained K has no fresh matched training-time measurement.', '',
        'The original feedback and teacher features are reused: zero new collection/preparation in this run does not '
        'mean those artifacts had zero historical cost. Intrinsic correction timers and capture timers are nested; '
        'do not add them to event time again. Historical K training time must not be presented as a contemporaneous '
        'measure of KC overhead.', '',
        f"Training-process peak allocated/reserved GPU memory: **{show(s['peak_allocated_bytes']/1e9 if s['peak_allocated_bytes'] else None,3)} / "
        f"{show(s['peak_reserved_bytes']/1e9 if s['peak_reserved_bytes'] else None,3)}GB**. "
        f"Actual core counters: `{s['native_counts']}`; response/backward work: `{s['work']}`. "
        f"New qualification optimizer steps: {s['new_qualification_optimizer_steps']}. "
        'Startup/chunk/recomputation counters are retained separately in SUMMARY.json.', '',
        '## Limits and artifacts','',
        'This is one prespecified event with one seed and a previously used development panel. Local repair preservation '
        'and final native gains remain distinct measurements. The sampled local effects do not establish transfer or '
        'robustness, and this screen does not identify the cause of any transfer limitation. No tuning or follow-up '
        'run is implied by this report.', '',
        '- `analysis/STEPS.csv`, `PROBE_SUMMARY.csv`, `PROBE_RESPONSES.csv`, `PROBE_EFFECTS.csv`, '
        '`REPAIR_IDENTITIES.csv`, `EXPOSURE.csv` and `SUMMARY.json`.',
        '- `raw/campaign_001/evaluation/report/PER_REQUEST.csv`, `ALL_REPEATS.csv` and `REPEAT_TOTALS.csv`.',
        '- `raw/campaign_001/FIRST_WINDOW_REPRODUCTION.json`, `CORRECTION_EVIDENCE.json`, training receipts and raw native records.',
        '- Independent source/result audit and integrity/storage records are reported separately.','']
    return '\n'.join(lines)


if __name__=='__main__':
    analyze(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent,
            Path(sys.argv[2]) if len(sys.argv)>2 else None)
