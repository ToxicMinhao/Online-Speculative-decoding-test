"""Revalidate saved MathQA evidence and report the fixed latency contrasts."""
import argparse
import csv
import json
from pathlib import Path

from online_feedback.storage import write_json
from .contract import BUDGET, METHODS, validate_plan
from .statistics import reduce_final, summarize


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def audit_endpoint(root, plan):
    # Resolve native sidecars relative to the production root, not evaluation/.
    from .evaluation import audit_stage
    return audit_stage(root / 'campaign_001/evaluation/final',
        plan['evaluation_requests'], plan['checkpoints'], evidence_root=root)


def audit_controls(root, plan):
    from .evaluation import audit_stage
    control = dict(canonical=plan['control_canonical'], counts=plan['control_counts'])
    return audit_stage(root / 'campaign_001/evaluation/controls',
        [plan['control_request']], plan['checkpoints'], evidence_root=root, control=control)


def complete_work_errors(actual):
    """The completed endpoint must account for controls and every timed request."""
    count = BUDGET['total_generations']
    expected = dict(planned_generations=count, attempted_generations=count,
        returned_generations=count, completed_valid_generations=count,
        invalid_generations=0, pending_canonical_generations=0,
        unknown_generation_attempts=0, unattempted_generations=0,
        optimizer_steps=0, training_presentations=0, feedback_collections=0,
        teacher_preparations=0, extra_warmup_generations=0)
    return ['actual_work_' + key for key, value in expected.items()
            if type(actual.get(key)) is not int or actual[key] != value]


def format_effect(value):
    low, high = value['ci95_pct']
    return f"{value['point_pct']:+.4f}% [{low:+.4f}%, {high:+.4f}%]"


def markdown(summary):
    """Keep latency claims first, distinguishing primary and secondary gates."""
    lines = ['# O58 / K58 / K+58 MathQA serving transfer', '',
        '## Material Passport', '',
        'ARS experiment-agent run → validate. Frozen GSM8K-trained checkpoints on new MathQA requests. '
        'Saved-record audit; not an independent model reproduction.', '',
        'Execution commit: `' + summary['execution_source']['commit'] + '`.', '',
        'Status: **' + summary['status'] + '**. Zero training, updates, feedback capture or teacher preparation.']
    endpoint = summary['endpoint']
    if endpoint is not None:
        totals = endpoint['method_totals']
        effects = endpoint['comparisons']
        lines += ['', '## Complete-request time', '',
            '| Checkpoint | Total request seconds¹ | Mean ms/request | K+ reduction versus this checkpoint [95% CI] | Native-generation seconds¹ | Cache-maintenance seconds¹ | Verification calls² | Output tokens² | Cap hits² |',
            '|---|---:|---:|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            row = totals[method]
            effect = '—' if method == 'KPLUS' else format_effect(effects['KPLUS_vs_' + method]['request_latency_s_reduction'])
            lines.append(f"| {method}58 | {row['request_latency_s']:.4f} | {row['mean_request_ms']:.4f} | {effect} | "
                f"{row['native_generate_s']:.4f} | {row['cache_maintenance_s']:.4f} | {row['verification_calls']} | "
                f"{row['output_tokens']} | {row['cap_hits']} |")
        lines += ['', '¹ Sum of per-request means over three fresh repeats. Complete-request time includes cache maintenance; '
            'loading and checkpoint switching are outside it. The two listed components need not exhaust the complete timer.', '',
            '² One validated trajectory per request after repeat consistency checks, not three times the same output.', '',
            endpoint['interpretation']['primary']['conclusion'], '',
            endpoint['interpretation']['secondary']['conclusion'] + ' Secondary role: **' +
            endpoint['interpretation']['secondary']['role'] + '**.', '',
            'The prespecified primary is K+ versus OSD (O). K+ versus K is confirmatory only if the primary '
            'complete-time interval has lower endpoint above zero. Both estimates remain visible. '
            'Call and component-time intervals are descriptive.', '',
            '## Three timing repeats', '',
            '| Repeat | O complete / native / cache seconds | K complete / native / cache seconds | K+ complete / native / cache seconds |',
            '|---|---:|---:|---:|']
        for repeat, values in endpoint['repeat_totals'].items():
            cells = [' / '.join(f"{values[method][key]:.4f}" for key in
                     ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')) for method in METHODS]
            lines.append('| ' + str(int(repeat) + 1) + ' | ' + ' | '.join(cells) + ' |')
        lines += ['', '## Drafting counts', '']
        for reference in ('O', 'K'):
            effect = effects['KPLUS_vs_' + reference]
            lines += [f"K+ versus {reference}: {effect['call_reduction']['point_pct']:+.4f}% pooled call reduction; "
                f"{effect['request_wins']} request improvements, {effect['request_regressions']} regressions, "
                f"{effect['request_ties']} ties. Gross calls saved/added: "
                f"{effect['gross_calls_saved']}/{effect['gross_calls_added']}.", '']
        lines += ['## Analysis and interpretation limits', '',
            'All 1,024 paired request identities are included. The same 10,000 PCG64 seed-20261103 resamples '
            'are used for both contrasts; each draw resamples 1,024 request identities carrying every method. '
            'Intervals use linear 2.5/97.5 percentiles. Timing repeats are measurements of a request, not independent samples.', '',
            'Canonical outputs are each request’s fresh O58 repeat-0 tokens and termination under the native verifier. '
            'This is not a separate target-only reference. Valid cap hits and slow requests remain included.', '',
            'This tests cross-dataset serving transfer for these fixed GSM8K-trained checkpoints. It does not '
            'establish training-seed robustness, MathQA adaptation, training-inclusive speedup, or a causal '
            'explanation for a difference. Historical GSM8K times are not pooled with these measurements.']
    else:
        lines += ['', '**No gain claim is available:** correctness, identity or completeness checks failed. '
            'No requests were removed to manufacture a valid subset.']
    lines += ['', '## Work, costs and retained evidence', '',
        'Budget: 9,216 timed generations plus three untimed historical controls = 9,219 total. '
        'The controls check each export on historical panel ordinal 0; they do not enter endpoint statistics.', '',
        'Actual work and campaign/loading/switching costs are preserved in SUMMARY.json and COSTS.json. '
        'They are separate from request latency; nested clocks must not be added.', '',
        'Companion files: PAIRED_REQUESTS.csv, ALL_REPEATS.csv, RAW_REPEAT_ROWS.csv, REPEAT_TOTALS.csv, '
        'VALIDITY.json, SUMMARY.json and COSTS.json. Native raw token/trace files remain in the run directory.', '',
        'No tuning, replacement requests, selective retiming, extra arms or automatic subsequent experiment.']
    if summary['errors']:
        lines += ['', 'Failures:', '', *['- ' + error for error in summary['errors']]]
    return '\n'.join(lines) + '\n'


def build_report(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / 'final_report'
    output.mkdir(parents=True, exist_ok=False)
    plan = read(root / 'sealed/PLAN.json')
    campaign = read(root / 'campaign_001/STATUS.json')
    errors, audit, endpoint, control, admission = [], {}, None, {}, {}
    evaluation = root / 'campaign_001/evaluation/final'
    raw_path = evaluation / 'MEASUREMENTS.jsonl'
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
    write_csv(output / 'RAW_REPEAT_ROWS.csv', raw_rows)
    try:
        validate_plan(plan)
        for stage in ('PREFLIGHT', 'POSTFLIGHT'):
            admission[stage] = read(root / (stage + '.json'))
            if admission[stage]['status'] != 'passed' or admission[stage]['model_work'] != 0:
                errors.append(stage.lower() + '_not_passed')
        saved_control = read(root / 'campaign_001/CONTROL_AUDIT.json')
        if (saved_control['status'] != 'passed' or saved_control['errors']
                or len(saved_control['rows']) != BUDGET['historical_control_generations']
                or {row['method'] for row in saved_control['rows']} != set(METHODS)):
            errors.append('original_control_audit_not_passed')
        control = audit_controls(root, plan)
        if (control['status'] != 'passed' or control['errors']
                or len(control['rows']) != BUDGET['historical_control_generations']
                or {row['method'] for row in control['rows']} != set(METHODS)):
            errors.append('historical_controls_not_complete_and_valid')
        audit = audit_endpoint(root, plan)
        errors.extend(audit['errors'])
        saved_audit = read(root / 'campaign_001/FINAL_AUDIT.json')
        if saved_audit['status'] != 'passed' or saved_audit['errors']:
            errors.append('original_final_audit_not_passed')
        if campaign['status'] != 'completed' or campaign.get('cleanup_errors'):
            errors.append('campaign_not_completed_cleanly')
        errors.extend(complete_work_errors(campaign.get('actual_work', {})))
        if audit['status'] != 'passed':
            errors.append('reconstructed_endpoint_audit_not_passed')
        if not errors:
            endpoint = summarize(audit['rows'], plan['evaluation_requests'])
            paired = reduce_final(audit['rows'], plan['evaluation_requests'])
            write_csv(output / 'PAIRED_REQUESTS.csv', [dict(panel_ordinal=row['panel_ordinal'],
                request_id=row['request_id'], **{method + '_' + key: value
                for method in METHODS for key, value in row[method].items()}) for row in paired])
            write_csv(output / 'REPEAT_TOTALS.csv', [dict(repeat=int(repeat), method=method, **values)
                for repeat, methods in endpoint['repeat_totals'].items() for method, values in methods.items()])
    except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
        errors.append(repr(error))
    rows = audit.get('rows', [])
    write_csv(output / 'ALL_REPEATS.csv', rows)
    write_json(output / 'VALIDITY.json', dict(
        endpoint={key: value for key, value in audit.items() if key != 'rows'},
        historical_controls=control, admission=admission, errors=errors))
    # Preserve the campaign's complete cost/accounting scope rather than infer a
    # nonoverlapping kernel breakdown from aggregate request clocks.
    costs = {key: value for key, value in campaign.items()
             if key.endswith('_s') or key.endswith('_costs') or key in ('actual_work', 'work', 'evaluation', 'controls')}
    write_json(output / 'COSTS.json', costs)
    summary = dict(status='valid_complete' if endpoint is not None and not errors else 'invalid_or_incomplete',
        material_passport=dict(workflow='ARS experiment-agent run → validate',
            readiness='ANALYZED' if endpoint is not None and not errors else 'PARTIAL',
            scope='Frozen-checkpoint MathQA serving transfer, no training'),
        execution_source=plan['execution_source'], budget=BUDGET, endpoint=endpoint,
        campaign=campaign, costs=costs,
        admission=admission,
        evaluation_audit={key: value for key, value in audit.items() if key != 'rows'},
        historical_controls={key: value for key, value in control.items() if key != 'rows'},
        raw_repeat_rows_preserved=len(raw_rows), errors=errors,
        limitations=['Paired prompt uncertainty does not measure training-seed robustness.',
            'Fresh O58 canonical references are native-verifier comparisons, not target-only decodes.',
            'Fixed GSM8K-trained exports; no new MathQA learning or adaptation-inclusive speed claim.',
            'Ordered primary/secondary total-time tests; all other interval summaries descriptive.',
            'No pooling with historical GSM8K latency or counting timing repeats as independent prompts.'])
    write_json(output / 'SUMMARY.json', summary)
    (output / 'REPORT.md').write_text(markdown(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output')
    args = parser.parse_args()
    summary = build_report(args.root, args.output)
    print(json.dumps(dict(status=summary['status'], errors=summary['errors'])))
    raise SystemExit(0 if summary['status'] == 'valid_complete' else 1)
