"""Compact, evidence-bound G report; no training or inference imports."""
import csv
import json
from pathlib import Path
from statistics import median
import sys


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def write_csv(path, rows, empty_fields):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or empty_fields
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shown(value, digits=4):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def response_identity(event, presentation):
    records = event.get('records', [])
    if 1 <= presentation <= len(records):
        row = records[presentation - 1]
        return dict(record_id=row['record_id'], pass_index=row['pass'], presentation=presentation)
    return dict(record_id=None, pass_index=None, presentation=presentation)


def analyze(root, rawroot=None):
    root = Path(root)
    raw = Path(rawroot) if rawroot is not None else root / 'raw/campaign_001'
    status = read_json(raw / 'STATUS.json') or {'status': 'unavailable'}
    qualification = read_json(raw / 'qualification/RESULT.json') or {}
    event = read_json(raw / 'G/events/event_001/RESULT.json') or {}
    costs = read_json(raw / 'COSTS.json') or {}
    guard = read_json(raw / 'GUARD_EVIDENCE.json') or costs.get('guard') or event.get('prefix_guard') or {}
    serving = read_json(raw / 'evaluation/report/SUMMARY.json') or {}
    output = root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)

    steps = []
    for source in guard.get('steps', []):
        row = dict(source)
        native = row['native_update_norm']
        row['realized_to_native_norm_ratio'] = fraction(row['realized_update_norm'], native)
        row['ideal_to_native_norm_ratio'] = fraction(row['projected_fp32_update_norm'], native)
        ideal_correction = row.get('ideal_correction_norm', row['projection_coefficient'] * row['h_norm'])
        row['ideal_correction_to_native_norm_ratio'] = fraction(ideal_correction, native)
        realized_correction = row.get('realized_correction_norm')
        row['realized_correction_to_native_norm_ratio'] = (
            fraction(realized_correction, native) if realized_correction is not None else None)
        steps.append(row)
    responses = [{**row, **response_identity(event, row['presentation'])}
                 for row in guard.get('responses', [])]
    probes, probe_summary = [], []
    for source in guard.get('probes', []):
        common = {key: value for key, value in source.items() if key != 'responses'}
        for index, response in enumerate(source['responses']):
            presentation = (source['optimizer_step'] - 1) * 2 + index + 1
            probes.append({**common, 'response_ordinal_in_step': index + 1,
                           **response_identity(event, presentation), **response,
                           'H_change': response['h_after'] - response['h_before']})
        observations = source['responses']
        summary = {**common, 'response_count': len(observations)}
        for field in ('h_before', 'h_after'):
            summary['mean_' + field] = sum(row[field] for row in observations) / len(observations)
        for field in ('first_errors_repaired_with_prefix_intact', 'new_earlier_errors',
                      'old_prefix_tokens_lost', 'net_matching_prefix_change'):
            summary[field] = sum(row[field] for row in observations)
        probe_summary.append(summary)
    write_csv(output / 'guard_steps.csv', steps, ['optimizer_step', 'active'])
    write_csv(output / 'guard_responses.csv', responses, ['presentation', 'record_id', 'h_before'])
    write_csv(output / 'local_probes.csv', probes,
              ['optimizer_step', 'candidate', 'response_ordinal_in_step', 'h_before', 'h_after'])
    write_csv(output / 'local_probe_summary.csv', probe_summary, ['optimizer_step', 'candidate'])

    active = [row for row in steps if row['active']]
    ratios = [row['realized_to_native_norm_ratio'] for row in steps
              if row['realized_to_native_norm_ratio'] is not None]
    guard_summary = dict(
        observed_steps=len(steps), active_projections=len(active),
        activation_fraction=fraction(len(active), len(steps)),
        active_positive_realized_derivative=sum(row['realized_directional_derivative'] > 0 for row in active),
        median_realized_to_native_norm_ratio=median(ratios) if ratios else None,
        minimum_realized_to_native_norm_ratio=min(ratios) if ratios else None,
        maximum_realized_to_native_norm_ratio=max(ratios) if ratios else None,
        response_measurements=len(responses), local_probe_response_evaluations=len(probes),
        work=guard.get('work', {}), forward_calls=guard.get('forward_calls', {}),
        costs=guard.get('costs', {}))
    event_costs = event.get('costs', costs.get('event', {}))
    training = event_costs.get('training_s')
    probe_time = guard.get('costs', {}).get('probes_s')
    training_without_probes = training - probe_time if training is not None and probe_time is not None else None
    failed_trainer = status.get('trainer_at_failure', {})
    main_steps = event.get('optimizer_steps', failed_trainer.get('global_step', status.get('main_optimizer_steps')))
    main_presentations = event.get('response_presentations', failed_trainer.get('presentations', status.get('main_presentations')))
    summary = dict(status=status, qualification_status=qualification.get('status', 'unavailable'),
        qualification_counts=qualification.get('counts', qualification.get('work_counts', {})),
        main_optimizer_steps=main_steps, main_response_presentations=main_presentations,
        serving=serving, guard=guard_summary, probes=probe_summary, costs=costs,
        training_s_excluding_local_probes=training_without_probes,
        scope='one prespecified development screen; original training-held-out panel reused',
        source_root=str(raw))
    (output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

    lines = ['# Prefix-preservation guard on K: one-event screen', '',
        '## Material Passport', '',
        f"Status: **{status['status']}**. GPU qualification: **{qualification.get('status', 'unavailable')}**. "
        'ARS experiment-agent execution and evidence review; project stage unchanged.', '',
        'G starts from released DFlash with a fresh seed-17 optimizer and reuses the original '
        '128 feedback responses and anchor/presentation schedule. K is the retained event-1 checkpoint. '
        'The K objective and native AdamW step remain unchanged; G projects only the component of '
        'the proposed displacement predicted to increase the current strict-prefix protection quantity.', '',
        '**This development panel is held out from feedback training but was already used in development.** '
        'It is not a fresh confirmation panel. No feedback is collected or learned from during replay.', '']
    if qualification.get('status') != 'passed':
        lines.extend(['**Qualification did not pass; no efficacy conclusion is supported.**', '',
            f"Recorded qualification error: `{qualification.get('error', 'unavailable')}`.", '',
            'The failed exact checks and their diagnostics remain in `qualification/`; '
            'a failure does not show that the guard helps or harms drafting.', ''])
    elif serving:
        valid = serving['valid_requests']
        lines.extend(['## Native serving comparison', '',
            f"Valid common requests: **{valid}/{serving['planned_requests']}**; "
            f"valid timed generations: **{serving['valid_timed_generations']}**. "
            'Each count uses one validated trajectory; timings use the mean of three fresh repeats per request. '
            'Exact output tokens and termination are checked against retained F.', '',
            '| Arm | Verification calls | Sum of mean request times (s) | Accepted draft tokens |',
            '|---|---:|---:|---:|'])
        for arm in ('K', 'G'):
            row = serving[arm]
            lines.append(f"| {arm} | {row['verification_calls']} | {shown(row['summed_mean_request_latency_s'], 6)} | {row['accepted_draft_tokens']} |")
        comparison = serving['comparisons']['G_vs_K']
        directions = comparison['call_request_directions']
        lines.extend(['', f"**G versus K:** pooled verification-call reduction "
            f"**{shown(comparison['call_reduction_pct'])}%**; request-time reduction "
            f"**{shown(comparison['request_time_reduction_pct'])}%**. "
            f"Requests improved/regressed/tied: **{directions['improved']}/{directions['regressed']}/{directions['tied']}**.", '',
            'Positive reductions favor G. Ratios use summed paired measurements, not mean request percentages. '
            'Serving time includes prefill, drafting, verification and the qualified request cache work, '
            'with the checkpoint available; training, probes, loading and publication are excluded.', ''])
        if status['status'] != 'completed' or valid != serving['planned_requests']:
            lines.extend(['**These are incomplete-run descriptive results.** Invalid and incomplete requests '
                          'remain visible and are excluded from gain claims; no completed-screen conclusion is drawn.', ''])
    else:
        lines.extend(['**No native serving results are available yet.** No improvement or regression claim is supported.', ''])

    lines.extend(['## Guard behavior and local measurements', '',
        f"Observed main steps: **{len(steps)}**; active projections: **{len(active)}** "
        f"({shown(100 * len(active) / len(steps) if steps else None, 2)}%). "
        f"Active steps with positive realized directional derivative: **{guard_summary['active_positive_realized_derivative']}**.", '',
        f"Median realized/native update norm: **{shown(guard_summary['median_realized_to_native_norm_ratio'], 6)}** "
        f"(range {shown(guard_summary['minimum_realized_to_native_norm_ratio'], 6)}–"
        f"{shown(guard_summary['maximum_realized_to_native_norm_ratio'], 6)}). "
        'All analytic, FP32-proposal and realized BF16 derivatives/norms are retained in `guard_steps.csv`. '
        'Positive realized derivatives are recorded without rollback; the first-order projection is not a finite-step guarantee.', ''])
    if probe_summary:
        lines.extend(['| Step | Candidate | Mean H before → after | Useful first-error repairs | New earlier errors | Lost old-prefix tokens | Net prefix tokens |',
                      '|---:|---|---:|---:|---:|---:|---:|'])
        for row in probe_summary:
            lines.append(f"| {row['optimizer_step']} | {row['candidate']} | "
                f"{shown(row['mean_h_before'], 6)} → {shown(row['mean_h_after'], 6)} | "
                f"{row['first_errors_repaired_with_prefix_intact']} | {row['new_earlier_errors']} | "
                f"{row['old_prefix_tokens_lost']} | {row['net_matching_prefix_change']:+d} |")
        lines.append('')
    lines.extend(['These are same-step local counterfactuals on training-conditioned blocks with fixed pre-update '
        'prefix membership and weights. They are not native serving measurements or independently trained arms. '
        'The norm-scaled candidate targets the projected FP32 norm; its realized BF16 norm is reported separately. '
        'Lower H alone is insufficient. Native benefit alongside retained repair progress supports the candidate; '
        'benefit dominated by near-zero updates may instead reflect suppressed learning. '
        'A local norm-scaled comparison does not replace a trained shrinkage control for a later causal claim.', '',
        '## Costs and scope accounting', '',
        '| Measurement | Seconds | Scope |', '|---|---:|---|',
        f"| Qualification | {shown(qualification.get('wall_s'), 3)} | Disposable work; excluded from deployment adaptation |",
        f"| G loading and initial-state check | {shown(costs.get('loading_and_initialization_s'), 3)} | Separate from training and serving |",
        f"| Complete G event | {shown(costs.get('event_wall_s'), 3)} | Contains training, local probes, persistence/export and event overhead |",
        f"| Training windows including local probes | {shown(training, 3)} | Nested inside complete event |",
        f"| Local probes | {shown(probe_time, 3)} | Diagnostic work nested inside training windows |",
        f"| Training windows excluding local probes | {shown(training_without_probes, 3)} | Subtraction only; not a second cost |",
        f"| H backward | {shown(guard.get('costs', {}).get('h_backward_s'), 3)} | Nested component timer |",
        f"| Projection | {shown(guard.get('costs', {}).get('guard_projection_s'), 3)} | Nested component timer |",
        f"| Checkpoint persistence | {shown(event_costs.get('checkpoint_save_s'), 3)} | Nested inside complete event |",
        f"| Export | {shown(event_costs.get('export_s'), 3)} | Nested inside complete event |",
        f"| Native backend loading | {shown(costs.get('backend_loading_s'), 3)} | Separate benchmark setup |",
        f"| Benchmark publications | {shown(serving.get('execution', {}).get('activation_s'), 3)} | Benchmark-only switches, excluded from serving latency |",
        f"| Actual campaign wall time | {shown(costs.get('actual_campaign_wall_s'), 3)} | Includes qualification, benchmark setup and all repetitions |", '',
        'Nested timers must not be added again to event wall time. H backward plus projection measures only '
        'those components; it does not capture all added forward/snapshot/bookkeeping overhead or establish a '
        'fresh matched training-cost difference against K. Retained feedback/features have zero new collection '
        'or preparation in this run, not zero historical cost.', '',
        f"Completed main training: {summary['main_response_presentations']} presentations / "
        f"{summary['main_optimizer_steps']} steps. Recorded local probe responses: {len(probes)} of 18 planned. "
        'The separate qualification plan is three temporary optimizer steps and six training presentations, '
        'plus two declared startup forward/backward evaluations with no optimizer step. '
        'Actual forward, chunk, backward and startup counters remain in `analysis/SUMMARY.json` and qualification records.', '',
        'Independent review checked the response-wide normalization, separate H-gradient buffer, unchanged K/Adam '
        'state, global projection and probe restoration. CPU tests exercise those contracts; only the native '
        'qualification receipts establish GPU integration. Raw invalid/incomplete work is retained.', '',
        'No tuning, additional arm, feedback window or later event is authorized by this report.', '',
        'Files: `analysis/guard_steps.csv`, `analysis/guard_responses.csv`, `analysis/local_probes.csv`, '
        '`analysis/local_probe_summary.csv`, `analysis/SUMMARY.json`; serving per-request and repeat records '
        'remain under the raw evaluation report directory.', ''])
    (root / 'REPORT.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    print(json.dumps(analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), indent=2))
