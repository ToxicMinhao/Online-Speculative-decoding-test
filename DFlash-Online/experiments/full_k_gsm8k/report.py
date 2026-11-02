"""Validate final evidence and report the fixed full-horizon comparison."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from online_feedback.storage import write_json
from .contract import ARMS, NEW_BUDGET, validate_plan
from .evaluation import (
    METHODS, PANEL_SIZE, INTERMEDIATE_EVENTS, INTERMEDIATE_SIZE,
    audit_stage, final_schedule, schedule_balance,
)
from .statistics import (
    COUNT_FIELDS, TIMINGS, COMPARISONS, intermediate_summary,
    interpret, reduce_final, summarize,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict,list)) else value
                             for key,value in row.items()})


def validate_training_audit(audit, plan):
    """Require the orchestrator's raw-state/work audit, not a completion label alone."""
    errors = list(audit.get('errors', []))
    if audit.get('status') != 'passed' or audit.get('budget') != NEW_BUDGET:
        errors.append('training_audit_or_fixed_budget')
    expected = dict(events_completed=58, optimizer_steps=7424, response_presentations=14848,
        online_generations=7473, terminal_preparation_forwards=7424,
        correctness_generations=4, intermediate_generations=384)
    for arm in ARMS:
        row = audit.get('arms', {}).get(arm, {})
        if any(row.get(key) != value for key,value in expected.items()):
            errors.append(arm + ': incomplete_own_trajectory')
        if row.get('final_checkpoint') != plan['checkpoints'][arm]:
            errors.append(arm + ': endpoint_checkpoint_identity')
        for key in ('actual_work','costs','exposure'):
            if key not in row:
                errors.append(arm + ': missing_' + key)
    return errors


def evaluation_accounting(stages):
    fields = ('planned_generations','attempted_generations','returned_generations',
              'completed_valid_generations','invalid_generations','pending_canonical_generations',
              'unknown_generation_attempts','unattempted_generations')
    totals = {field:0 for field in fields}
    receipts = []
    for stage in stages:
        progress = stage['progress']
        planned = PANEL_SIZE*4*3 if stage['phase'] == 'final' else INTERMEDIATE_SIZE
        receipt = dict(phase=stage['phase'],arm=stage['arm'],event=stage['event'],audit_status=stage['status'])
        for field in fields:
            if field == 'planned_generations':
                value = planned
            elif field == 'unattempted_generations':
                value = progress.get(field, planned-progress.get('attempted_generations',0))
            else:
                value = progress.get(field,0)
            receipt[field] = value
            totals[field] += value
        receipt.update({key:progress.get(key) for key in ('activation_s','weight_validation_s','evaluation_wall_s')})
        receipts.append(receipt)
    return dict(totals=totals,stages=receipts,
                scope='Intermediate and final evaluation only; online/correctness generations are in the training audit.')


def length_summary(paired):
    output = {}
    for method in METHODS:
        values = [row[method] for row in paired]
        lengths = [row['output_tokens'] for row in values]
        output[method] = dict(requests=len(values),minimum_output_tokens=min(lengths),
            maximum_output_tokens=max(lengths),mean_output_tokens=sum(lengths)/len(lengths),
            cap_hits=sum(row['cap_hit'] for row in values),
            termination_counts=dict(Counter(row['finish_reason'] for row in values)))
    return output


def resource_summary(training_audit):
    """Keep memory and named compiler timers separate from elapsed-time totals."""
    result = {}
    for arm in ARMS:
        source = training_audit.get('arms', {}).get(arm, {})
        records = source.get('resource_records', [])
        compiler = {}
        for record in records:
            for name, timer in record['training_compiler_timers'].items():
                total = compiler.setdefault(name, dict(calls=0, total_s=0.))
                total['calls'] += timer['calls']
                total['total_s'] += timer['total_s']
        result[arm] = dict(
            peak_allocated_bytes=source.get('peak_allocated_bytes'),
            peak_reserved_bytes=source.get('peak_reserved_bytes'),
            event_records=records, compiler_by_timer_name=compiler,
            unavailable_costs=source.get('unavailable_costs', []),
            memory_scope='Trainer-process CUDA peak reset immediately before controller.update; '
                'observed through preparation, training, save/export and publication. '
                'Excludes initial loading, later boundary validation and the separate serving worker.',
            compiler_scope='TorchDynamo timers from that update window; each name is summed across events. '
                'Different timer names can nest and must not be added together or on top of update time.',
            serving_worker_scope='Compilation is included in worker loading/serving, but is not separately '
                'identified. Trainer peaks do not measure serving-worker or total-system memory.')
    return result


def material_passport():
    return dict(origin_skill='ars-codex:academic-research-suite / experiment-agent',
                origin_mode='validate', origin_date=datetime.now(timezone.utc).isoformat(),
                verification_status='ANALYZED', version_label='full_k_gsm8k_report_v1',
                evidence='Sealed protocol, raw native records, work ledgers and recorded live-state checks.',
                verification_scope='CPU record reconstruction and statistical analysis; '
                    'not an independent GPU reproduction or reinspection of retired model/feature tensors.')


def derived_final_curve(rows, canonicals):
    """A same-panel subset of existing final records, not additional evaluation."""
    curves = []
    for arm in ARMS:
        selected = [dict(row,phase='intermediate') for row in rows
                    if row['method'] == arm and row['repeat'] == 0 and row['panel_ordinal'] < INTERMEDIATE_SIZE]
        curve = intermediate_summary(selected,canonicals)
        curve['source'] = 'first128_subset_of_final_repeat0; no additional generations'
        curves.append(curve)
    return curves


def format_effect(effect):
    point, ci = effect['point_pct'], effect['ci95_pct']
    if point is None:
        return 'undefined (zero reference total)', 'undefined'
    return f'{point:+.4f}%', f'[{ci[0]:+.4f}, {ci[1]:+.4f}]%' if ci else 'undefined'


def render_report(summary):
    passport = summary['material_passport']
    lines = ['# Full GSM8K K/K_HALF online-training comparison', '',
        'Execution source: `' + summary['execution_source']['commit'] + '`.', '',
        '## Material Passport', '',
        '- Origin Skill: ' + passport['origin_skill'],
        '- Origin Mode: ' + passport['origin_mode'],
        '- Origin Date: ' + passport['origin_date'],
        '- Verification Status: ' + passport['verification_status'],
        '- Version Label: ' + passport['version_label'], '',
        passport['verification_scope'], '',
        'The fixed protocol trains only K and K_HALF, each from released DFlash with fresh seed-17 Adam state. '
        'Each is assigned its own original 7,473-request stream, 58 updates and 7,424 Adam steps, '
        'with the final 49 requests served without another update. F and completed O58 are retained controls. '
        'This is an arm-specific online-feedback comparison; old O feedback features were not reconstructed.', '',
        'The target is pinned Qwen3-8B and the draft is Qwen3-8B-DFlash-b16. K uses constant LR 3e-4; '
        'K_HALF uses LR 1.5e-4. Both retain the qualified current-prefix-plus-first-error KL objective, '
        'selected-weight normalization, BF16 AdamW, response-only supervision, two passes, microbatch 1 '
        'and accumulation 2. Greedy block-16 serving uses 2,048 tokens or EOS, context 4,096, concurrency 1 '
        'and full per-request cache flushing.', '']
    if summary['status'] != 'valid':
        lines += ['**The experiment is invalid or incomplete; no efficacy conclusion is issued.**', '',
                  *['- ' + error for error in summary['errors']], '',
                  'Completed, invalid, unknown and unattempted work remains in the audit and raw records. '
                  'No unfavorable request is replaced or partial successful subset promoted to the primary result.']
        return '\n'.join(lines)+'\n'
    analysis = summary['analysis']
    lines += ['**' + analysis['interpretation']['endpoint_label'].capitalize() + '.**', '',
        'Primary: K_HALF58 versus O58. This compares different learning recipes; K58/O58 is the '
        'same-LR full-updater control. The benchmark was used previously and only seed 17 is fitted here.', '',
        '| Method | Calls | Output tokens | Calls / 1,000 tokens | Accepted tokens | Progress / call |',
        '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        row = analysis['method_totals'][method]
        lines.append(f"| {method} | {row['verification_calls']} | {row['output_tokens']} | "
                     f"{row['calls_per_1000_output_tokens']:.4f} | {row['accepted_draft_tokens']} | "
                     f"{row['committed_progress_per_call']:.4f} |")
    lines += ['', '| Comparison | Call reduction | Paired 95% interval | Wins / regressions / ties | Gross saved / added |',
              '|---|---:|---:|---:|---:|---:|']
    for candidate, baseline in COMPARISONS:
        name = candidate+'_vs_'+baseline
        row = analysis['comparisons'][name]
        point,ci = format_effect(row['call_reduction'])
        lines.append(f"| {candidate}/{baseline} | {point} | {ci} | {row['request_wins']} / "
                     f"{row['request_regressions']} / {row['request_ties']} | "
                     f"{row['gross_calls_saved']} / {row['gross_calls_added']} |")
    lines += ['', '| Comparison | Complete request-time reduction | Paired 95% interval | Native-generation reduction | Cache-time reduction |',
              '|---|---:|---:|---:|---:|']
    for candidate,baseline in COMPARISONS:
        row = analysis['comparisons'][candidate+'_vs_'+baseline]
        point,ci = format_effect(row['request_latency_s_reduction'])
        native,_ = format_effect(row['native_generate_s_reduction'])
        cache,_ = format_effect(row['cache_maintenance_s_reduction'])
        lines.append(f'| {candidate}/{baseline} | {point} | {ci} | {native} | {cache} |')
    lines += ['', 'Calls use one repeat-consistent trajectory per request; timing uses three-repeat request means. '
              'All ratios divide summed measurements. The 10,000 paired-request bootstrap uses PCG64 seed 20261102 '
              'and linear percentile endpoints; its unit is one of 1,024 requests carrying all methods, '
              'not a timing repeat. Intervals are conditional on this fitted seed and previously used benchmark. '
              'Secondary intervals are descriptive/unadjusted. No equivalence threshold was introduced.', '',
              '| Repeat | F request seconds | O request seconds | K request seconds | K_HALF request seconds |',
              '|---|---:|---:|---:|---:|']
    for repeat, methods in analysis['repeat_totals'].items():
        lines.append('| '+str(int(repeat)+1)+' | '+' | '.join(f"{methods[m]['request_latency_s']:.3f}" for m in METHODS)+' |')
    lines += ['', '| Method versus F | Gross gain calls | Gross harm calls | Net saved calls |',
              '|---|---:|---:|---:|']
    for method, row in analysis['gain_harm_vs_F'].items():
        lines.append(f"| {method} | {row['gross_gain_calls']} | {row['gross_harm_calls']} | {row['net_calls_saved']} |")
    lines += ['', 'Gross gains and harms retain request identities. They describe useful and harmful endpoint '
              'changes, not a unique cause such as memorization or gradient conflict.', '',
              '## Fixed learning-curve observations', '',
              '| Arm | Event | Requests | Calls | Retained F calls | Reduction vs F | Wins / regressions / ties |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['learning_curve']:
        lines.append(f"| {row['arm']} | {row['event']} | {row['requests']} | {row['verification_calls']} | "
                     f"{row['frozen_verification_calls']} | {row['call_reduction_vs_F_pct']:+.4f}% | "
                     f"{row['request_wins']} / {row['request_regressions']} / {row['request_ties']} |")
    lines += ['', 'Events 14/29/44 use the fixed first 128 requests once. Event 58 in this table uses those same '
              '128 requests from the final repeat-zero records, with no extra generation. The curve uses retained '
              'F count context; the full 1,024-request final comparisons above use fresh F. These observations '
              'did not select checkpoints or stop training. No matched intermediate O measurements are supplied, '
              'so this is not evidence of faster learning than O.', '',
              '## Work, exposure and costs', '',
              'All 116 training events, 14,848 Adam steps, 29,696 response presentations, 14,946 online responses, '
              '14,848 terminal-feature preparations, 768 intermediate evaluations, 12,288 final evaluations and '
              'eight inherited correctness generations are separately accounted for. Complete generations total 28,010. '
              'No disposable Adam qualification step or new efficacy warmup is included.', '',
              '| Arm | Training seconds | Blocking updates seconds | Controller serving seconds | Loading seconds | Online wall seconds |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        costs = summary['training_audit']['arms'][arm]['costs']
        fields = ('training_s','blocking_update_s','controller_serving_s','loading_s','online_wall_s')
        values = ['unavailable' if costs.get(field) is None else f'{costs[field]:.3f}' for field in fields]
        lines.append('| '+arm+' | '+' | '.join(values)+' |')
    lines += ['', 'Teacher preparation, capture, feature persistence, optimizer work, checkpoint/export, publication '
              'and retention costs are retained separately in COSTS.json. Training/preparation/persistence/publication '
              'are nested within blocking updates; capture is nested within serving. Do not add nested timers twice. '
              'Actual anchor and supervised-position occurrence exposure appears in EXPOSURE.json; repeated positions '
              'are exposure, not unique tokens. F/O58 historical adaptation costs are not fresh matched costs.', '',
              '| Arm | Update-window peak allocated GiB | Update-window peak reserved GiB |',
              '|---|---:|---:|']
    for arm in ARMS:
        resource = summary['resources'][arm]
        values = ['unavailable' if resource[key] is None else f'{resource[key]/2**30:.3f}'
                  for key in ('peak_allocated_bytes','peak_reserved_bytes')]
        lines.append('| '+arm+' | '+' | '.join(values)+' |')
    lines += ['', 'These are trainer-process CUDA peaks, reset before each controller update and measured through '
              'preparation, training, checkpoint/export and publication. They exclude initial loading, subsequent '
              'boundary validation and the separate serving worker; they are not whole-system memory peaks.', '',
              'RESOURCES.json preserves each event and compiler timer by name. TorchDynamo timers can nest: '
              'do not sum different timer names or add them to training/update wall time. Serving-worker compilation '
              'is included in loading/serving but is not separately identifiable. Final worker loading and campaign '
              'wall time are separate fields in COSTS.json, when their receipts are available.', '',
              'Final request timing is contemporaneous checkpoint-ready serving. It does not establish net '
              'asynchronous deployment or training-inclusive speedup. Aggregate timers are not a kernel-time breakdown.', '',
              '## One next recommendation', '', analysis['interpretation']['next_action'], '',
              'This recommendation authorizes no further event, tuning, panel expansion or new mechanism. '
              'A positive endpoint is not training-seed robustness, convergence or conference novelty. '
              'Valid negative and inconclusive outcomes complete the fixed experiment.', '',
              'Files: [summary](SUMMARY.json), [paired requests](PAIRED_REQUESTS.csv), '
              '[all evaluation records](ALL_EVALUATIONS.csv), [learning curve](LEARNING_CURVE.csv), '
              '[raw-evidence audit](EVALUATION_AUDIT.json), [work accounting](ACCOUNTING.json), '
              '[costs](COSTS.json), [resources](RESOURCES.json), [exposure](EXPOSURE.json).', '']
    return '\n'.join(lines)


def write_campaign_report(root, output=None, *, training_audit=None):
    root = Path(root)
    output = Path(output) if output is not None else root/'final_report'
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(root/'sealed/PLAN.json')
    errors = []
    try:
        validate_plan(plan)
    except (OSError,KeyError,TypeError,ValueError) as error:
        errors.append('sealed_plan: '+repr(error))
    requests = plan['evaluation_requests']
    canonicals = read_json(root/'sealed/CANONICALS.json')
    if training_audit is None:
        try:
            training_audit = read_json(root/'campaign_001/TRAINING_AUDIT.json')
        except (OSError,ValueError) as error:
            training_audit = dict(status='missing',errors=[repr(error)],arms={})
    errors.extend(validate_training_audit(training_audit,plan))
    stages = []
    for arm in ARMS:
        for event in INTERMEDIATE_EVENTS:
            arm_root = root/'campaign_001/arms'/arm
            checkpoint = str(Path(plan['checkpoints'][arm]).parent.parent/f'event_{event:03d}'/'export')
            stage = audit_stage(arm_root/'evaluation'/f'event_{event:03d}',requests,{arm:checkpoint},
                canonicals=canonicals,phase='intermediate',arm=arm,event=event,evidence_root=root)
            stages.append(stage)
    stages.append(audit_stage(root/'campaign_001/evaluation/final',requests,plan['checkpoints'],
        canonicals=canonicals,phase='final',evidence_root=root))
    for stage in stages:
        errors.extend(f"{stage['phase']}/{stage['arm']}/{stage['event']}: {error}" for error in stage['errors'])
    rows = [row for stage in stages for row in stage['rows']]
    analysis,paired,curves = None,[],[]
    if not errors:
        final = stages[-1]['rows']
        analysis = summarize(final)
        paired = reduce_final(final)
        if len(paired) != PANEL_SIZE:
            errors.append('final_panel_incomplete')
            analysis = None
        else:
            curves = [intermediate_summary(stage['rows'],canonicals) for stage in stages[:-1]]
            curves.extend(derived_final_curve(final,canonicals))
            curves.sort(key=lambda row:(ARMS.index(row['arm']),row['event']))
    accounting = evaluation_accounting(stages)
    resources = resource_summary(training_audit)
    costs = dict(arms={arm:training_audit.get('arms',{}).get(arm,{}).get('costs',{}) for arm in ARMS},
                 evaluations=accounting['stages'],scope='Named scopes overlap; do not add nested timers twice.')
    for name,filename in (('final_loading','FINAL_LOADING.json'),('campaign','STATUS.json')):
        path = root/'campaign_001'/filename
        costs[name] = read_json(path) if path.exists() else dict(status='unavailable',source=str(path))
    summary = dict(status='valid' if analysis is not None and not errors else 'invalid_or_incomplete',
        errors=errors,execution_source=plan['execution_source'],analysis=analysis,
        material_passport=material_passport(),resources=resources,
        interpretation=analysis['interpretation'] if analysis else interpret({},valid=False),
        training_audit=training_audit,evaluation_accounting=accounting,learning_curve=curves,
        schedule_balance=schedule_balance(final_schedule(requests)),
        lengths_and_termination=length_summary(paired) if paired else {},costs=costs)
    write_json(output/'SUMMARY.json',summary)
    write_json(output/'EVALUATION_AUDIT.json',stages)
    write_json(output/'ACCOUNTING.json',dict(training=training_audit,evaluation=accounting,budget=NEW_BUDGET))
    write_json(output/'COSTS.json',costs)
    write_json(output/'RESOURCES.json',resources)
    write_json(output/'EXPOSURE.json',{arm:training_audit.get('arms',{}).get(arm,{}).get('exposure',{}) for arm in ARMS})
    write_csv(output/'ALL_EVALUATIONS.csv',rows)
    write_csv(output/'PAIRED_REQUESTS.csv',[{key:value for key,value in row.items() if key not in METHODS} |
        {method+'_'+key:value for method in METHODS for key,value in row[method].items()} for row in paired])
    write_csv(output/'LEARNING_CURVE.csv',curves)
    (output/'REPORT.md').write_text(render_report(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('output',nargs='?')
    args = parser.parse_args()
    result = write_campaign_report(args.root,args.output)
    print(json.dumps({'status':result['status'],'errors':result['errors']}))
    raise SystemExit(0 if result['status'] == 'valid' else 1)
