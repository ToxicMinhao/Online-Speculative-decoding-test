"""Fixed paired serving decision and parent-frozen mechanism descriptions."""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from experiments.repair_conditional_kl_pilot.analysis import summarize as paired_summary
from experiments.repair_conditional_kl_pilot.analysis import TIMES
from experiments.repair_conditional_kl_pilot.evaluation import COUNT_FIELDS
from experiments.frontier_preservation_pilot.point3_analysis import exported_parameters_identity
from online_feedback.storage import write_json

METHODS = ('BASE', 'TAIL')
SEED, DRAWS = 20261126, 10000


def complete_generation_accounting(rows):
    """Bind all warmups and scored rows to the twelve prescribed activations."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['activation']].append(row)
    if len(rows) != 792 or set(grouped) != set(range(12)):
        raise ValueError('Require all768 scored executions and24 warmups')
    warmup_ids = {row['request_id'] for row in rows if row['phase'] == 'warmup'}
    if len(warmup_ids) != 2:
        raise ValueError('Require the original two warmups throughout')
    group_ids = {}
    for activation, records in grouped.items():
        repeat, group = divmod(activation // 2, 2)
        shift, position = (repeat + group) % 2, activation % 2
        method = (METHODS[shift:] + METHODS[:shift])[position]
        if any((r['repeat'], r['group'], r['method']) != (repeat, group, method) for r in records):
            raise ValueError('Activation differs from the fixed alternating schedule')
        warmup = [r for r in records if r['phase'] == 'warmup']
        scored = [r for r in records if r['phase'] == 'scored']
        if (len(warmup) != 2 or {r['request_id'] for r in warmup} != warmup_ids
                or len(scored) != 64 or len({r['request_id'] for r in scored}) != 64):
            raise ValueError('Activation lost or duplicated a warmup/scored request')
        selected = frozenset(r['request_id'] for r in scored)
        if group in group_ids and selected != group_ids[group]:
            raise ValueError('Request group changed across arms/repeats')
        group_ids[group] = selected
    if (group_ids[0] & group_ids[1] or len(group_ids[0] | group_ids[1]) != 128
            or (group_ids[0] | group_ids[1]) & warmup_ids):
        raise ValueError('Require disjoint scored groups and separate warmups')
    return dict(returned=792, valid=792, scored=768, warmups=24, activations=12)


def summarize(rows, *, expected_requests=128):
    result = paired_summary(rows, expected_requests=expected_requests, methods=METHODS,
                            bootstrap_seed=SEED)
    result['generation_accounting'] = complete_generation_accounting(rows) if expected_requests == 128 else None
    result['bootstrap']['shared_indices_for_latency_and_calls'] = True
    result['bootstrap']['scope'] = 'Fixed checkpoints, reused development panel; no training-seed replication'
    for key in TIMES:
        delta = [row['BASE'][key] - row['TAIL'][key] for row in result['paired']]
        result['contrasts']['TAIL_vs_BASE'][key + '_request_outcomes'] = dict(
            wins=sum(x > 0 for x in delta), regressions=sum(x < 0 for x in delta),
            ties=sum(x == 0 for x in delta))
    return result


def freeze_decision(report, *, distinct_weights, active, correctness_complete):
    """Apply the handoff to unrounded estimates; identity overrides noisy timings."""
    if not correctness_complete:
        return dict(label='BLOCKED', selected='BASE', reason='Incomplete or invalid comparison')
    if not distinct_weights:
        return dict(label='NO_DEMONSTRATED_BENEFIT', selected='BASE',
                    reason='Identical serving parameters cannot establish a speedup')
    contrast = report['contrasts']['TAIL_vs_BASE']
    latency = contrast['request_latency_s_reduction_pct']
    calls = contrast['verification_call_reduction_pct']
    if latency['value'] is None or latency['ci95'] is None or calls['value'] is None:
        return dict(label='BLOCKED', selected='BASE', reason='Primary comparison is undefined')
    lower, upper = latency['ci95']
    if lower > 0 and calls['value'] >= 0:
        return dict(label='SUPPORTED_FOR_CONFIRMATION', selected='TAIL',
                    reason='Positive latency lower bound, nonnegative call estimate, distinct weights',
                    automatic_adoption=False)
    if active and latency['value'] > 0 and calls['value'] > 0 and lower <= 0 <= upper and lower > -.5:
        return dict(label='TENTATIVE', selected='BASE',
                    reason='Favorable points with unresolved latency inside the declared pilot tolerance')
    return dict(label='NO_DEMONSTRATED_BENEFIT', selected='BASE',
                reason='The completed fixed pilot meets neither benefit criterion')


MECHANISM_FIELDS = ('raw_repair', 'useful_repair', 'reachable_origin_followers',
                    'lost_prefix', 'damaged_prefix_positions', 'prefix_damaged',
                    'surviving_origin_followers', 'positive_progress', 'Y', 'uncapped_progress')


def mechanism_summary(base_rows, tail_rows, *, cohort):
    """Compare endpoint behavior on identical parent-defined blocks, per response."""
    base = {(r['record_id'], r['anchor']): r for r in base_rows}
    tail = {(r['record_id'], r['anchor']): r for r in tail_rows}
    if len(base) != len(base_rows) or len(tail) != len(tail_rows) or base.keys() != tail.keys():
        raise ValueError('Mechanism rows must preserve every unique matched response/anchor')
    grouped = defaultdict(lambda: {method: defaultdict(int) for method in METHODS})
    for key, before in base.items():
        after = tail[key]
        for field in ('parent', 'e', 'L', 'K', 'm', 'high'):
            if before[field] != after[field]:
                raise ValueError('Mechanism origin changed: ' + field)
        for method, row in (('BASE', before), ('TAIL', after)):
            if row['Y'] != int(row['useful_repair']) + row['reachable_origin_followers'] - row['lost_prefix']:
                raise ValueError('Signed fixed-parent progress identity failed')
            group = grouped[key[0]][method]
            group['blocks'] += 1
            group['origin_errors'] += int(row['e'] >= 1)
            for field in MECHANISM_FIELDS:
                group[field] += int(row[field])
    records = []
    for record_id in sorted(grouped):
        entry = dict(record_id=record_id, cohort=cohort)
        for method in METHODS:
            entry.update({method + '_' + field: value for field, value in grouped[record_id][method].items()})
        entry.update({'TAIL_minus_BASE_' + field: entry['TAIL_' + field] - entry['BASE_' + field]
                      for field in MECHANISM_FIELDS})
        records.append(entry)
    totals = {method: {field: sum(row[method + '_' + field] for row in records)
                       for field in ('blocks', 'origin_errors', *MECHANISM_FIELDS)} for method in METHODS}
    return dict(cohort=cohort, responses=len(records), matched_blocks=len(base), totals=totals,
                differences={field: totals['TAIL'][field] - totals['BASE'][field] for field in MECHANISM_FIELDS},
                per_response=records,
                inference='Training/retention description; no independent block trials or serving claim')


def csv_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_public_tables(root, public, serving, mechanism):
    """Only anonymous ordinals enter the reviewer tables; raw evidence stays private."""
    public = Path(public)
    public.mkdir(parents=True, exist_ok=True)
    identifiers = {row['request_id']: i for i, row in enumerate(serving['paired'])}
    per_request = []
    for row in serving['paired']:
        result = dict(request_ordinal=identifiers[row['request_id']])
        for method in METHODS:
            result.update({method + '_' + key: row[method][key] for key in (*TIMES, *COUNT_FIELDS, 'finish_reason')})
        per_request.append(result)
    csv_table(public / 'PER_REQUEST.csv', per_request)
    rows = [json.loads(line) for line in (Path(root) / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
    warmups = {key: i for i, key in enumerate(sorted({r['request_id'] for r in rows if r['phase'] == 'warmup'}))}
    for phase, filename in (('scored', 'ALL_REPEATS.csv'), ('warmup', 'WARMUPS.csv')):
        table = []
        for row in rows:
            if row['phase'] == phase:
                lookup = identifiers if phase == 'scored' else warmups
                table.append(dict(request_ordinal=lookup[row['request_id']],
                    **{key: row[key] for key in ('activation', 'repeat', 'group', 'method', *TIMES, *COUNT_FIELDS,
                                                 'finish_reason', 'valid_measurement')}))
        csv_table(public / filename, table)
    write_json(public / 'SERVING_SUMMARY.json', {k: v for k, v in serving.items() if k not in ('paired', 'bootstrap')})
    write_json(public / 'BOOTSTRAP.json', {k: v for k, v in serving['bootstrap'].items() if k != 'input_order'})
    for cohort, result in mechanism.items():
        table = [{('response_ordinal' if key == 'record_id' else key): (ordinal if key == 'record_id' else value)
                  for key, value in row.items()} for ordinal, row in enumerate(result['per_response'])]
        csv_table(public / (cohort.upper() + '_MECHANISM.csv'), table)
    write_json(public / 'MECHANISM_SUMMARY.json', {name: {k: v for k, v in result.items() if k != 'per_response'}
                                                for name, result in mechanism.items()})



def pool_activity_table(training):
    """Use original response observations once, never accumulated boundary metrics."""
    records = training['training']['records']
    identifiers = {key: i for i, key in enumerate(sorted({row['record_id'] for row in records}))}
    output = []
    for ordinal, row in enumerate(records):
        observed = row['response_observation']
        if observed['record_id'] != row['record_id']:
            raise ValueError('Pool activity lost its original response identity')
        metrics = observed['continuation_repair']
        if any(type(value) not in (int, float, bool) or not math.isfinite(float(value))
               for value in metrics.values()):
            raise ValueError('Pool activity must be finite compact scalar evidence')
        output.append(dict(presentation_ordinal=ordinal, response_ordinal=identifiers[row['record_id']],
            pass_index=row['pass'], optimizer_step_after=row['optimizer_step_after'], **metrics))
    if sum(row['continuation_changed_high_positions'] for row in output) != training['activity']['continuation_changed_high_positions']:
        raise ValueError('Coefficient activity total double-counts or drops a presentation')
    return output


def report_provenance(plan, training, readout, clock, protocol, rejected_preparation=None):
    """Separate launched processes, model forwards and the source of each phase."""
    processes = [row for row in clock['processes'] if row['phase'] == 'readout']
    if readout.get('resumed_unexecuted'):
        forwards = [readout['prior_attempt_response_forwards']]
        if rejected_preparation is not None:
            forwards.append(rejected_preparation['new_forward_evidence'])
        forwards.append(readout['attempt_work']['response_reads_completed'])
    else:
        forwards = [readout['work']['response_reads_completed']]
    if len(forwards) != len(processes) or sum(forwards) != readout['work']['response_reads_completed']:
        raise ValueError('Readout process attempts and completed-forward receipts do not reconcile')
    return dict(execution_commit=protocol['execution_commit'],
        training_execution_commit=training['execution_commit'],
        serving_execution_commit=protocol['execution_commit'],
        analysis_execution_commit=plan.get('analysis_execution_commit', plan['execution_commit']),
        readout_attempts=len(processes), readout_model_forward_attempts=sum(count > 0 for count in forwards),
        readout_forward_counts_by_process=forwards)


def run_analysis(plan, public, *, raw_root=None, analysis_output_root=None):
    """Reduce completed serving and mechanism evidence without another model call."""
    root = Path(raw_root) if raw_root is not None else Path(plan['root'])
    output = Path(analysis_output_root) if analysis_output_root is not None else root
    output.mkdir(parents=True, exist_ok=True)
    serving = json.loads((root / 'evaluation/SUMMARY.json').read_text())
    evaluation = json.loads((root / 'evaluation/STATUS.json').read_text())
    readout = json.loads((root / 'mechanism/STATUS.json').read_text())
    training = json.loads((root / 'arms/TAIL/STATUS.json').read_text())
    mechanism = json.loads((root / 'mechanism/SUMMARY.json').read_text())
    complete = (evaluation['status'] == 'completed' and evaluation['completed_valid'] == 792
                and all(evaluation[key] == 0 for key in ('invalid', 'unknown', 'unattempted'))
                and readout['status'] == 'completed' and training['status'] == 'completed')
    activity = training['activity']
    decision = freeze_decision(serving, distinct_weights=serving['distinct_serving_weights'],
        active=activity['continuation_changed_high_positions'] > 0, correctness_complete=complete)
    if decision['label'] == 'BLOCKED':
        write_json(output / 'REPORT_BLOCKED.json', decision)
        raise ValueError('An incomplete comparison cannot produce a gain report')
    serving['decision'] = decision
    rejection_path = root / 'resume_002/REPAIR.json'
    rejected = (json.loads(rejection_path.read_text())['rejected_preparation']
                if rejection_path.exists() else None)
    serving.update(report_provenance(plan, training, readout,
        json.loads((root / 'CLOCK.json').read_text()),
        json.loads((root / 'evaluation/PROTOCOL.json').read_text()), rejected))
    serving['activity'] = activity
    write_json(output / 'FINAL_ANALYSIS.json', dict(serving=serving, mechanism=mechanism))
    write_public_tables(root, public, serving, mechanism)
    csv_table(Path(public) / 'POOL_ACTIVITY.csv', pool_activity_table(training))
    write_json(Path(public) / 'TRAINING_COSTS.json', dict(
        candidate=training['training']['costs'], loading_s=training['loading_s'],
        total_event_process_wall_s=training['wall_s'],
        startup_wall_s=training['startup']['wall_s'], startup_optimizer_steps=training['startup']['optimizer_steps'],
        peak_allocated_bytes=training['peak_allocated_bytes'], peak_reserved_bytes=training['peak_reserved_bytes'],
        historical_BASE_training='Retained context only; no freshly matched BASE training overhead estimate'))
    contrast = serving['contrasts']['TAIL_vs_BASE']
    latency = contrast['request_latency_s_reduction_pct']
    calls = contrast['verification_call_reduction_pct']
    interval = lambda value: '[' + ', '.join(f'{x:+.5f}%' for x in value['ci95']) + ']'
    lines = ['# Continuation repair allocation pilot', '',
        f"Training execution commit: `{serving['training_execution_commit']}`. Serving execution commit: `{serving['serving_execution_commit']}`. CPU analysis commit: `{serving['analysis_execution_commit']}`.", '',
        f"Completed fixed comparison: **{decision['label']}**. Recommended frozen configuration: **{decision['selected']}**.", '',
        'The intervention refines the existing HIGH repair budget within each response. BASE is the authenticated retained native C+K+PAIR event-two endpoint; TAIL alone receives the fixed continuation refinement. PR64 remains an inconclusive observational diagnostic.', '',
        '| Metric | BASE | TAIL | Reduction, TAIL versus BASE | 95% paired interval |',
        '|---|---:|---:|---:|---:|',
        f"| Summed per-request mean latency (s) | {serving['totals']['BASE']['request_latency_s']:.6f} | {serving['totals']['TAIL']['request_latency_s']:.6f} | {latency['value']:+.5f}% | {interval(latency)} |",
        f"| Verification calls | {serving['totals']['BASE']['verification_calls']:.0f} | {serving['totals']['TAIL']['verification_calls']:.0f} | {calls['value']:+.5f}% | {interval(calls)} |", '',
        f"Calls: {contrast['wins']} request wins, {contrast['regressions']} regressions and {contrast['ties']} ties; {contrast['gross_calls_saved']} calls saved and {contrast['gross_calls_added']} added.", '',
        'All128 reused development requests, two endpoints, three timing repeats and24 included warmups are retained. Exact canonical outputs/termination and repeat-consistent counts are required. The primary interval resamples whole requests10,000 times with PCG64 seed20261126, sharing indices for calls and latency; it averages each request’s repeats first.', '',
        f"Serving parameter tensors distinct: **{serving['distinct_serving_weights']}**. Changed HIGH coefficient occurrences: **{activity['continuation_changed_high_positions']}**. Activity alone does not establish benefit.", '',
        '| Fixed-parent readout cohort | BASE signed progress | TAIL signed progress | TAIL−BASE |',
        '|---|---:|---:|---:|']
    for cohort, values in mechanism.items():
        lines.append(f"| {cohort} | {values['totals']['BASE']['Y']} | {values['totals']['TAIL']['Y']} | {values['differences']['Y']:+d} |")
    lines += ['', 'Mechanism tables aggregate matched blocks within responses and separately retain raw/useful repairs, reachable followers, lost consecutive-prefix occurrences and damaged individual positions. These are training-buffer/retention descriptions, not independent block trials or serving measurements. No PR64 regression was repeated. The historical new-buffer witness overlaps these training responses and is not an independent cohort.', '',
        'Fresh native-generation/cache times, all repeats, loading/switching/warmup costs and parameter identities are in SERVING_SUMMARY.json and the per-request/repeat tables. Historical BASE training time is context, not a fresh matched overhead estimate. Qualification/startup/committed work and storage admission remain separately recorded in the run receipts.', '',
        'This fixed event on a reused development panel does not establish training-seed robustness, fresh-panel generalization, optimal/calibrated repair utility or training-inclusive speedup. No label automatically authorizes adoption, tuning or another run. Independent audit and final all-location cost/validity receipts accompany publication.', '']
    if readout.get('resumed_unexecuted'):
        lines += [f"Readout processes: {serving['readout_attempts']}; processes completing model forwards: {serving['readout_model_forward_attempts']}; completed forwards by process: {serving['readout_forward_counts_by_process']}.", 'The first readout attempt stopped on a concurrent storage-receipt write after73 primary responses and one included repeat. The original failure and costs remain recorded. The repaired orchestration reused those74 completed forwards and executed only55 remaining primary plus eight retention readouts; no training, completed readout or qualification forward was repeated.', '']
    Path(public, 'REPORT.md').write_text('\n'.join(lines))
    return decision


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--analysis-output-root', type=Path)
    parser.add_argument('--analysis-commit')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if args.analysis_commit is not None:
        plan = dict(plan, analysis_execution_commit=args.analysis_commit)
    run_analysis(plan, args.public, raw_root=args.raw_root, analysis_output_root=args.analysis_output_root)
