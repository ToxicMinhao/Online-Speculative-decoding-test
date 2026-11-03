"""Compact outcome, work, exposure and treatment-activity report for fixed K+."""
import argparse
import csv
import json
from pathlib import Path

from online_feedback.storage import write_json
from .campaign import ARM_LIMITS
from .contract import NEW_BUDGET
from .evaluation import audit_stage
from .statistics import summarize, reduce_final


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def training_summary(root):
    """Sum only original response observations, not accumulation-window metrics."""
    from .training import validate_event_receipt
    base = root / 'campaign_001/arms/KPLUS'
    status = read(base / 'STATUS.json')
    work = status['actual_work']
    if (status['status'] != 'completed' or status['events_completed'] != 58 or status['tail_completed'] != 49
            or work['attempts'] != ARM_LIMITS or work['completed'] != ARM_LIMITS or any(work['unknown'].values())):
        raise ValueError('Incomplete or extra one-arm training work')
    rows, resources, anchor_comparisons, objective_work = [], [], [], []
    for event in range(1, 59):
        event_id = f'event_{event:03d}'
        result = read(base / 'trainer/events' / event_id / 'RESULT.json')
        validate_event_receipt(result, event_number=event)
        boundary = read(base / 'boundaries' / f'{event_id}.json')
        if boundary['status'] != 'passed' or boundary['event'] != event:
            raise ValueError('Missing complete scientific boundary')
        for pass_index in (0, 1):
            records = [row for row in result['records'] if row['pass'] == pass_index]
            if len(records) != 128:
                raise ValueError('Wrong response presentation count')
            metrics = [row['response_observation']['sum_metrics'] for row in records]
            fields = set(metrics[0])
            if any(set(value) != fields for value in metrics):
                raise ValueError('Per-response diagnostic schema changed')
            row = dict(event=event, pass_index=pass_index, presentations=128,
                prompt_tokens=sum(value['prompt_tokens'] for value in records),
                response_tokens=sum(value['response_tokens'] for value in records),
                active_anchors=sum(value['active_anchors'] for value in records))
            row.update({field: sum(value[field] for value in metrics) for field in sorted(fields)})
            row['mixed_response_frequency'] = row['kplus_mixed_response'] / row['kplus_response_count']
            weight = row['kplus_W1'] + row['kplus_W2']
            row['pooled_error_weight_moved_share'] = row['kplus_error_weight_moved'] / weight if weight else 0.
            row['mean_response_moved_share'] = row['kplus_error_weight_moved_share'] / row['kplus_response_count']
            row['maximum_absolute_conservation_error'] = max(abs(value['kplus_conservation_error']) for value in metrics)
            rows.append(row)
        objective_work.append(dict(event=event, **result['objective_work']))
        resources.append(dict(event=event, **read(base / 'events' / event_id / 'RESOURCE_COSTS.json')))
        anchor_comparisons.append(dict(event=event, **read(base / 'events' / event_id / 'ORIGINAL_ANCHOR_COMPARISON.json')))
    totals = {key: sum(row[key] for row in rows) for key in rows[0]
              if key not in ('event', 'pass_index', 'mixed_response_frequency',
                             'pooled_error_weight_moved_share', 'mean_response_moved_share',
                             'maximum_absolute_conservation_error')}
    totals['mixed_response_frequency'] = totals['kplus_mixed_response'] / totals['kplus_response_count']
    weight = totals['kplus_W1'] + totals['kplus_W2']
    totals['pooled_error_weight_moved_share'] = totals['kplus_error_weight_moved'] / weight if weight else 0.
    totals['mean_response_moved_share'] = totals['kplus_error_weight_moved_share'] / totals['kplus_response_count']
    totals['maximum_absolute_conservation_error'] = max(row['maximum_absolute_conservation_error'] for row in rows)
    return dict(status='passed', actual_work=work, exposure=totals, pass_rows=rows,
        event_costs=status['event_costs'], resources=resources, objective_work=objective_work,
        original_anchor_comparisons=anchor_comparisons,
        worker_wall_s=status['worker_wall_s'], backend_loading_s=status['backend_loading_s'],
        trainer_loading_s=status['trainer_loading_s'], initial_validation_s=status['initial_validation_s'],
        final_state_check=read(base / 'FINAL_STATE_CHECK.json'),
        initial_check=read(base / 'trainer/INITIAL_CHECK.json'),
        final_boundary=read(base / 'boundaries/event_058.json'))


def build_report(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / 'final_report'
    output.mkdir(parents=True, exist_ok=True)
    plan = read(root / 'sealed/PLAN.json')
    campaign = read(root / 'campaign_001/STATUS.json')
    errors = []
    training = endpoint = None
    audit = {}
    try:
        training = training_summary(root)
        checkpoints = dict(K=plan['references']['K'], KPLUS=read(root / 'campaign_001/arms/KPLUS/STATUS.json')['final_checkpoint'])
        audit = audit_stage(root / 'campaign_001/evaluation/final', plan['evaluation_requests'], checkpoints,
            canonicals=read(plan['canonicals']), baseline_counts=read(plan['baseline_counts']))
        errors.extend(audit['errors'])
        if audit['status'] == 'passed':
            endpoint = summarize(audit['rows'])
            paired = reduce_final(audit['rows'])
            write_csv(output / 'MEASUREMENTS.csv', audit['rows'])
            write_csv(output / 'PAIRED_REQUESTS.csv', [dict(panel_ordinal=row['panel_ordinal'], request_id=row['request_id'],
                **{method + '_' + key: value for method in ('K', 'KPLUS') for key, value in row[method].items()})
                for row in paired])
        if campaign['status'] != 'completed':
            errors.append('campaign_not_completed')
    except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
        errors.append(repr(error))
    if training is not None:
        write_csv(output / 'TRAINING_EXPOSURE.csv', training['pass_rows'])
        write_json(output / 'TRAINING.json', training)
    summary = dict(status='valid_complete' if endpoint is not None and not errors else 'invalid_or_incomplete',
        material_passport=dict(workflow='ARS experiment-agent run → validate',
            material='Fixed K+ November3 handoff, pinned code/config and retained native raw records',
            readiness='ANALYZED' if endpoint is not None and not errors else 'PARTIAL',
            review_scope='Raw-record reconstruction and CPU implementation review; not an independent GPU reproduction'),
        execution_source=plan['execution_source'], budget=NEW_BUDGET, endpoint=endpoint,
        training=training, campaign=campaign, evaluation_audit={key: value for key, value in audit.items() if key != 'rows'},
        errors=errors, limitations=[
            'One fitted seed and a previously used benchmark; request bootstrap is not training-seed robustness.',
            'KPLUS was trained from released initialization on its own feedback; K58 was only a retained serving control.',
            'Original K intermediate feature tensors/states were retired; this is not identical-feature replay.',
            'Native anchor comparisons are observations, never forced schedules; identical tokens do not establish identical hidden tensors.',
            'Training/component/compiler timers can nest; final serving gain is not training-inclusive or asynchronous deployment speedup.',
            'Treatment activity is not evidence of endpoint efficacy; no tuning, extra seed or rescue run is authorized.'])
    write_json(output / 'SUMMARY.json', summary)
    lines = ['# K+ full GSM8K endpoint', '', 'Execution: `' + plan['execution_source']['commit'] + '`.', '',
        'Material Passport: ARS experiment-agent run → validate; ' + summary['material_passport']['readiness'] +
        '. Source: fixed November3 handoff, execution seal and native raw records. '
        'Record reconstruction is not an independent GPU reproduction.', '',
        'Status: **' + summary['status'] + '**. One new KPLUS arm; retained K58 is read-only.', '',
        'The two arms use their own online feedback trajectories. K58 did not generate K+ training data or initialize K+.']
    if endpoint:
        effect = endpoint['comparisons']['KPLUS_vs_K']
        lines += ['', '| Endpoint | K58 | K+58 |', '|---|---:|---:|']
        for field in ('verification_calls', 'output_tokens', 'request_latency_s', 'accepted_tokens_per_call', 'committed_progress_per_call'):
            lines.append(f"| {field} | {endpoint['method_totals']['K'][field]} | {endpoint['method_totals']['KPLUS'][field]} |")
        for field in ('call_reduction', 'request_latency_s_reduction'):
            value = effect[field]
            lines += ['', f"{field}: {value['point_pct']:.6f}%; paired 95% interval {value['ci95_pct']}%."]
        lines += ['', endpoint['interpretation']['endpoint_label'] + '. ' + endpoint['interpretation']['recommendation'], '',
                  f"Wins/regressions/ties: {effect['request_wins']}/{effect['request_regressions']}/{effect['request_ties']}; "
                  f"gross calls saved/added: {effect['gross_calls_saved']}/{effect['gross_calls_added']}.", '',
                  'Count totals use one repeat-stable trajectory per request; time uses three-repeat means. '
                  'PCG64 seed 20261102, 10,000 paired request bootstrap draws, linear 2.5/97.5 percentiles. '
                  'Per-repeat timings and complete paired rows are preserved in the companion files.']
    if training:
        exposure = training['exposure']
        lines += ['', f"K+ adaptation worker wall: {training['worker_wall_s']:.3f} s. "
            f"Mixed-response frequency: {exposure['mixed_response_frequency']:.6f}; "
            f"pooled share of first-error weight moved: {exposure['pooled_error_weight_moved_share']:.6f}.", '',
            'TRAINING.json preserves nested costs, per-event allocator peaks and compiler timers, initial/final scientific-state '
            'checks, and observed historical anchor matches. TRAINING_EXPOSURE.csv aggregates each original forward once; '
            'group numerator/mass sums and response-average moved share have distinct meanings.']
    lines += ['', *['- ' + value for value in summary['limitations']]]
    if errors:
        lines += ['', 'Failures:', '', *['- ' + value for value in errors]]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = build_report(args.root, args.output)
    print(json.dumps(dict(status=result['status'], errors=result['errors'])))
    raise SystemExit(0 if result['status'] == 'valid_complete' else 1)
