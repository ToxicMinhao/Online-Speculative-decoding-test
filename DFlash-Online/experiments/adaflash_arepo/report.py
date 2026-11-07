"""Four-checkpoint endpoint report with qualification and full work accounting."""
import argparse
import csv
import json
from pathlib import Path

from online_feedback.storage import write_json
from .contract import METHODS, NEW_BUDGET
from .evaluation import audit_stage
from .statistics import reduce_final, summarize


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def training_summary(root):
    """Count original response observations once, without accumulation duplicates."""
    base = Path(root) / 'campaign_001/arms/AREPO'
    status = read(base / 'STATUS.json')
    if status['status'] != 'completed':
        raise ValueError('A_repo trajectory did not complete')
    from .campaign import ARM_LIMITS
    from .training import validate_event_receipt
    work = status['actual_work']
    if (status['status'] != 'completed' or status['events_completed'] != 58
            or status['tail_completed'] != 49 or work['attempts'] != ARM_LIMITS
            or work['completed'] != ARM_LIMITS or any(work['unknown'].values())):
        raise ValueError('Incomplete or extra A_repo trajectory work')
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
    totals = {key: sum(row[key] for row in rows) for key in rows[0]
              if key not in ('event', 'pass_index')}
    return dict(status='passed', actual_work=work, exposure=totals, pass_rows=rows,
        event_costs=status['event_costs'], resources=resources, objective_work=objective_work,
        original_anchor_comparisons=anchor_comparisons, worker_wall_s=status['worker_wall_s'],
        backend_loading_s=status['backend_loading_s'], trainer_loading_s=status['trainer_loading_s'],
        initial_validation_s=status['initial_validation_s'],
        initial_check=read(base / 'trainer/INITIAL_CHECK.json'),
        final_check=read(base / 'FINAL_STATE_CHECK.json'))


def returned_measurements(folder):
    """Retain invalid and partial rows even when the strict endpoint audit fails."""
    path = folder / 'MEASUREMENTS.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def build_report(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / 'results'
    output.mkdir(parents=True, exist_ok=False)
    plan = read(root / 'sealed/PLAN.json')
    errors, training, endpoint, qualification = [], None, None, None
    campaign_path = root / 'campaign_001/STATUS.json'
    campaign = read(campaign_path) if campaign_path.exists() else {'status': 'unavailable'}
    arm_path = root / 'campaign_001/arms/AREPO/STATUS.json'
    arm = read(arm_path) if arm_path.exists() else {'status': 'unavailable'}
    checkpoints = dict(plan['references'],
        AREPO=str(root / 'campaign_001/arms/AREPO/trainer/events/event_058/export'))
    folder = root / 'campaign_001/evaluation/final'
    audit = audit_stage(folder, plan['evaluation_requests'], checkpoints,
        canonicals=read(plan['canonicals']), evidence_root=root)
    write_json(output / 'VALIDITY.json', {key: value for key, value in audit.items() if key != 'rows'})
    raw_rows = returned_measurements(folder)
    write_csv(output / 'ALL_REPEATS.csv', raw_rows)
    if audit['status'] == 'passed':
        endpoint = summarize(audit['rows'])
        paired = reduce_final(audit['rows'])
        flat = [dict(seed=row['seed'], event=row['event'], panel_ordinal=row['panel_ordinal'],
            request_id=row['request_id'], **{method + '_' + key: value
                for method in METHODS for key, value in row[method].items()}) for row in paired]
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
    try:
        qualification = read(root / 'qualification/QUALIFICATION.json')
        details = root / 'qualification/RESULT.json'
        if details.exists():
            qualification = dict(qualification, details=read(details))
        if qualification['status'] != 'passed':
            errors.append('source_loss_qualification_not_passed')
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append('qualification_evidence: ' + repr(error))
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
        checkpoints=checkpoints, qualification=qualification, endpoint=endpoint,
        training=training, campaign=campaign, arm_progress=arm,
        evaluation_audit={key: value for key, value in audit.items() if key != 'rows'}, errors=errors,
        retained_training_cost_scope='Historical O58/KPLUS58/paper-A58 costs, not fresh matched measurements.')
    write_json(output / 'SUMMARY.json', summary)
    (output / 'REPORT.md').write_text(render_report(summary) + '\n')
    return summary


def render_report(summary):
    endpoint, training = summary['endpoint'], summary['training']
    lines = ['# Released AdaFlash loss under our matched online protocol', '',
        '## Material Passport', '',
        'ARS experiment-agent run -> validate; status **' + summary['status'] + '**. '
        'This analyzes retained native records and is not independent GPU replication.', '',
        'Execution commit: `' + summary['execution_source']['commit'] + '`.', '',
        'Configuration binding: `' + summary['configuration']['binding'] + '`. '
        'SUMMARY.json records checkpoint identities, source/configuration pins, qualification and work.', '',
        '## Fixed comparison', '',
        'A_repo is the only newly trained arm: released DFlash, fresh seed-17 BF16 AdamW, '
        '58 own-feedback events, 7,424 Adam steps and 14,848 response presentations. '
        'O58, K+58 and paper-A58 are retained checkpoints, measured freshly together with A_repo58.', '',
        'The loss ports AdaFlash 478bf0ff7a3a5232b13d0f14e752d5fe1da0b718: '
        '0.8 teacher-argmax CE + 0.2 reverse KL, temperature 1. It preserves the source probability '
        'operations and intermediate dtypes, sums RKL over vocabulary, divides by the full response '
        'weight sum plus 1e-6, caps normalized positions at 0.01, then applies weights. '
        'Separately reduced CE and RKL are mixed last. All valid proposals receive loss; '
        'there is no K+ mask or adaptive-length head. This is not PR30\'s FP32-cap/rescaling proxy.', '',
        'Each learning recipe collected its own feedback; this is an online-recipe endpoint comparison, '
        'not identical-feature isolation or a full AdaFlash system reproduction.']
    if endpoint:
        lines += ['', '## Fresh endpoint measurements', '',
            '| Measurement | O58 | K+58 | Paper-A58 | A_repo58 |', '|---|---:|---:|---:|---:|']
        fields = [('verification_calls', 'Verification calls'), ('output_tokens', 'Output tokens'),
            ('request_latency_s', 'Complete-request seconds'), ('native_generate_s', 'Native-generation seconds'),
            ('cache_maintenance_s', 'Cache-maintenance seconds'),
            ('native_accepted_tokens_per_call', 'Runtime accepted counter / verification'),
            ('matched_accepted_tokens_per_call', 'Matched accepted draft tokens / verification'),
            ('committed_progress_per_call', 'Committed progress / verification')]
        for field, label in fields:
            lines.append('| ' + label + ' | ' + ' | '.join(
                str(endpoint['method_totals'][method][field]) for method in METHODS) + ' |')
        lines += ['', 'The native accepted counter is committed progress minus one per verification round. '
            'Matched acceptance clips the canonical-matching draft prefix at the stopping boundary. '
            'EOS/caps can make these counters differ. Committed progress excludes each request\'s '
            'initial target-prefill token. Every ratio uses summed counts; all counters remain in the tables.']
        for name, effect in endpoint['comparisons'].items():
            timing, calls = effect['request_latency_s_reduction'], effect['call_reduction']
            lines += ['', f"**{name}:** complete-request reduction {timing['point_pct']:.6f}% "
                f"[95% paired CI {timing['ci95_pct']}]; call reduction {calls['point_pct']:.6f}%. "
                f"Request wins/regressions/ties {effect['request_wins']}/{effect['request_regressions']}/"
                f"{effect['request_ties']}; gross calls saved/added "
                f"{effect['gross_calls_saved']}/{effect['gross_calls_added']}."]
        lines += ['', 'Primary decision: **' + endpoint['interpretation']['endpoint_label'] + '**. '
            'The paired K+/A_repo complete-request-time interval determines this label. '
            'Other intervals are unadjusted descriptive comparisons.', '',
            'Time totals sum per-request means over three repeats. Count totals use one validated '
            'trajectory per request after repeat consistency. The bootstrap uses 10,000 paired-request '
            'draws, PCG64 seed 20261102 and linear 2.5/97.5 percentiles. Repeats are not independent requests.']
    lines += ['', '## Costs, validity and limits', '',
        'Qualification is separately retained in SUMMARY.json: at most six disposable AdamW steps '
        'before complete scientific-state reset. It does not count toward the 7,424 main steps. '
        'Four inherited native correctness completions remain separate from the 12,288 timed completions.']
    if training:
        lines += ['', f"A_repo worker wall: {training['worker_wall_s']:.3f} seconds. "
            'TRAINING.json and EVENT_COSTS.csv separate training, teacher preparation, persistence, '
            'activation and publication; nested timers must not be added. Exposure records count '
            'original response observations once, excluding checkpoint recomputation and accumulation copies.']
    lines += ['', 'O/K+/paper-A training costs are historical context, not fresh matched training-overhead '
        'measurements. Fresh endpoint serving gains do not establish training-inclusive or asynchronous '
        'deployment speedup. Loading/switching remain outside request latency. No detailed kernel breakdown '
        'is inferred from aggregate request times.', '',
        'The result concerns one seed and a reused train-disjoint mathematics panel. It does not establish '
        'other-domain or seed robustness, full-system AdaFlash performance, or an isolated causal explanation '
        'of any difference. All measured regressions remain in the paired totals. No tuning or follow-up is authorized.']
    if summary['errors']:
        lines += ['', '**Invalid or incomplete:** no complete endpoint conclusion is supported.', '',
            *['- ' + error for error in summary['errors']]]
    lines += ['', 'Deliverables: SUMMARY.json, VALIDITY.json, ALL_REPEATS.csv, PAIRED_REQUESTS.csv '
        '(only for a valid complete endpoint), REPEAT_TOTALS.json, TRAINING.json, '
        'TRAINING_EXPOSURE.csv and EVENT_COSTS.csv. Raw failures and unattempted/unknown counts '
        'remain in the original status and validity evidence.']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = build_report(args.root, args.output)
    print(json.dumps(dict(status=result['status'], errors=result['errors'])))
    raise SystemExit(0 if result['status'] == 'valid_complete' else 1)
