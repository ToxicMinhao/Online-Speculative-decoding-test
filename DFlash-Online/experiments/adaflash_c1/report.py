"""Compact three-arm decision plus actual exposure, clipping and complete work."""
import argparse
import json
from pathlib import Path

from online_feedback.storage import write_json
from experiments.kplus_full_gsm8k.report import write_csv
from .campaign import ARM_LIMITS
from .contract import NEW_BUDGET
from .evaluation import audit_stage
from .statistics import summarize, reduce_final


def read(path):
    return json.loads(Path(path).read_text())


def training_summary(root):
    """Add each original response observation once, never optimizer-window copies."""
    from .training import validate_event_receipt
    base = root / 'campaign_001/arms/A'
    status = read(base / 'STATUS.json')
    work = status['actual_work']
    if (status['status'] != 'completed' or status['events_completed'] != 58 or status['tail_completed'] != 49
            or work['attempts'] != ARM_LIMITS or work['completed'] != ARM_LIMITS or any(work['unknown'].values())):
        raise ValueError('Incomplete or extra A trajectory work')
    rows, resources, objective_work, anchor_comparisons = [], [], [], []
    for event in range(1, 59):
        event_id = f'event_{event:03d}'
        result = read(base / 'trainer/events' / event_id / 'RESULT.json')
        validate_event_receipt(result, event_number=event)
        boundary = read(base / 'boundaries' / f'{event_id}.json')
        if boundary['status'] != 'passed' or boundary['event'] != event:
            raise ValueError('Missing complete own scientific boundary')
        for pass_index in (0, 1):
            records = [record for record in result['records'] if record['pass'] == pass_index]
            metrics = [record['response_observation']['sum_metrics'] for record in records]
            if len(records) != 128 or any(set(value) != set(metrics[0]) for value in metrics):
                raise ValueError('Response count or diagnostic schema changed')
            row = dict(event=event, pass_index=pass_index, presentations=len(records),
                prompt_tokens=sum(record['prompt_tokens'] for record in records),
                response_tokens=sum(record['response_tokens'] for record in records),
                active_anchors=sum(record['active_anchors'] for record in records))
            row.update({field: sum(value[field] for value in metrics) for field in sorted(metrics[0])})
            rows.append(row)
        objective_work.append(dict(event=event, **result['objective_work']))
        resources.append(dict(event=event, **read(base / 'events' / event_id / 'RESOURCE_COSTS.json')))
        anchor_comparisons.append(dict(event=event,
            **read(base / 'events' / event_id / 'ORIGINAL_ANCHOR_COMPARISON.json')))
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key not in ('event', 'pass_index')}
    return dict(status='passed', actual_work=work, exposure=totals, pass_rows=rows,
        event_costs=status['event_costs'], resources=resources, objective_work=objective_work,
        original_anchor_comparisons=anchor_comparisons, worker_wall_s=status['worker_wall_s'],
        backend_loading_s=status['backend_loading_s'], trainer_loading_s=status['trainer_loading_s'],
        initial_validation_s=status['initial_validation_s'],
        numerical_qualification=read(base / 'LOSS_QUALIFICATION.json'),
        initial_check=read(base / 'trainer/INITIAL_CHECK.json'),
        final_check=read(base / 'FINAL_STATE_CHECK.json'))


def build_report(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / 'results'
    output.mkdir(parents=True, exist_ok=False)
    plan = read(root / 'sealed/PLAN.json')
    errors, training, endpoint = [], None, None
    campaign_path = root / 'campaign_001/STATUS.json'
    campaign = read(campaign_path) if campaign_path.exists() else {'status': 'unavailable'}
    arm_path = root / 'campaign_001/arms/A/STATUS.json'
    arm = read(arm_path) if arm_path.exists() else {'status': 'unavailable'}
    checkpoints = dict(plan['references'], A=str(root / 'campaign_001/arms/A/trainer/events/event_058/export'))
    audit = audit_stage(root / 'campaign_001/evaluation/final', plan['evaluation_requests'], checkpoints,
        canonicals=read(plan['canonicals']), evidence_root=root)
    write_json(output / 'VALIDITY.json', {key: value for key, value in audit.items() if key != 'rows'})
    write_csv(output / 'ALL_REPEATS.csv', audit['rows'])
    if audit['status'] == 'passed':
        endpoint = summarize(audit['rows'])
        paired = reduce_final(audit['rows'])
        flat = [dict(seed=row['seed'], event=row['event'], panel_ordinal=row['panel_ordinal'],
            request_id=row['request_id'], **{method + '_' + key: value
                for method in ('O', 'KPLUS', 'A') for key, value in row[method].items()}) for row in paired]
        write_csv(output / 'PAIRED_REQUESTS.csv', flat)
        write_json(output / 'REPEAT_TOTALS.json', endpoint['repeat_totals'])
    else:
        errors.extend(audit['errors'])
    try:
        training = training_summary(root)
        write_json(output / 'TRAINING.json', training)
        write_csv(output / 'TRAINING_EXPOSURE.csv', training['pass_rows'])
        write_csv(output / 'EVENT_COSTS.csv', training['event_costs'])
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        errors.append('training_evidence: ' + repr(error))
    if campaign.get('status') != 'completed':
        errors.append('campaign_not_completed')
    summary = dict(status='valid_complete' if not errors else 'invalid_or_incomplete',
        material_passport=dict(workflow='ARS experiment-agent run -> validate',
            readiness='ANALYZED' if not errors else 'PARTIAL',
            review_scope='Retained native evidence reconstruction; not independent GPU replication'),
        execution_source=plan['execution_source'], budget=NEW_BUDGET,
        configuration=dict(binding=plan['binding'], configs=plan['configs'],
            original_config=plan['original_config'], loss=plan['loss'],
            model_pin='sealed/MODEL_PIN.json', runtime_pin='sealed/RUNTIME_PIN.json',
            training_pin='sealed/TRAINING_PIN.json'),
        checkpoints=checkpoints,
        endpoint=endpoint, training=training, campaign=campaign, arm_progress=arm,
        evaluation_audit={key: value for key, value in audit.items() if key != 'rows'}, errors=errors)
    write_json(output / 'SUMMARY.json', summary)
    lines = ['# Paper-defined AdaFlash-OPD versus K+ and ordinary OSD', '',
        '## Material Passport', '',
        'ARS experiment-agent run -> validate; status **' + summary['status'] + '**. '
        'This is analysis of retained native records, not independent GPU replication.', '',
        'Execution commit: `' + plan['execution_source']['commit'] + '`.', '',
        'Configuration binding: `' + plan['binding'] + '`. Exact checkpoint paths, configuration paths '
        'and model/runtime pin locations appear in SUMMARY.json and the sealed source manifest.', '',
        'Planned new training: A58 only, with 58 own-feedback events, 7,424 Adam steps and 14,848 response presentations. '
        'O58 and K+58 are retained controls measured freshly together with A58.', '',
        'A uses 0.8 target-argmax CE + 0.2 entry-wise upper-clipped reverse KL at temperature 1. '
        'Each vocabulary contribution is capped at 0.01 before summation; negative entries remain unchanged. '
        'All valid proposals receive loss with the matched OSD sampler, position decay and response normalization. '
        'There is no K+ prefix/error mask and no adaptive-length head.', '',
        'This implements the paper formula. The pinned released trainer instead normalizes summed vocabulary '
        'divergence before applying a position-level cap; it is not the loss tested here.']
    if endpoint:
        lines += ['', '| Measurement | O58 | K+58 | A58 |', '|---|---:|---:|---:|']
        fields = [('verification_calls', 'Verification calls'), ('output_tokens', 'Output tokens'),
            ('request_latency_s', 'Complete-request seconds'), ('native_generate_s', 'Native-generation seconds'),
            ('cache_maintenance_s', 'Cache-maintenance seconds'),
            ('accepted_tokens_per_call', 'Accepted draft tokens / verification'),
            ('committed_progress_per_call', 'Committed output tokens / verification')]
        for field, label in fields:
            lines.append('| ' + label + ' | ' + ' | '.join(
                str(endpoint['method_totals'][method][field]) for method in ('O', 'KPLUS', 'A')) + ' |')
        for name, effect in endpoint['comparisons'].items():
            timing = effect['request_latency_s_reduction']
            calls = effect['call_reduction']
            lines += ['', f"**{name}:** complete-request reduction {timing['point_pct']:.6f}% "
                f"[95% paired CI {timing['ci95_pct']}]; call reduction {calls['point_pct']:.6f}%. "
                f"Request call wins/regressions/ties {effect['request_wins']}/"
                f"{effect['request_regressions']}/{effect['request_ties']}; "
                f"gross calls saved/added {effect['gross_calls_saved']}/{effect['gross_calls_added']}."]
        lines += ['', 'Primary decision: **' + endpoint['interpretation']['endpoint_label'] + '**. '
            'The sign of K+/A complete-request-time interval controls this decision; '
            'call counts and native-generation time are separate outcomes.', '',
            'Time totals sum each request\'s three-repeat mean. Count totals use one validated trajectory '
            'per request after repeat consistency. PCG64 seed 20261102, 10,000 paired-request bootstrap '
            'draws, linear 2.5/97.5 percentiles. Repeats are not independent requests.']
    if training:
        lines += ['', f"A training/serving worker wall: {training['worker_wall_s']:.3f} seconds. "
            'TRAINING.json and EVENT_COSTS.csv keep training, teacher preparation, feature persistence, '
            'activation and publication costs separate; nested timers must not be added. '
            'TRAINING_EXPOSURE.csv records clipping activity and actual anchors/positions once per original forward.']
        lines += ['', 'LOSS_QUALIFICATION.json separately records the synthetic loss/gradient checks '
            '(12 tiny loss forwards/backwards on success, zero model forwards and zero optimizer steps). '
            'Four inherited native correctness completions are separate from timed serving.']
    lines += ['', 'Limitations: one training seed, a previously used train-disjoint panel, endpoint serving latency. '
        'Arms collected their own feedback; this is a recipe comparison rather than identical-feature loss isolation. '
        'Faster endpoint serving does not establish training-inclusive or asynchronous deployment speedup. '
        'No tuning, extra arm, adaptive length or follow-up run is authorized.']
    if errors:
        lines += ['', 'Incomplete/invalid work remains visible:', '', *['- ' + value for value in errors]]
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
