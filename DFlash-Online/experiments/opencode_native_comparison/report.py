"""Publish compact numerical evidence only after independent raw validation."""
import argparse
import csv
import json
from pathlib import Path

from . import analysis, raw_audit
from .contract import METHODS, ARMS

PUBLIC_REPEAT_FIELDS = ('request_id', 'source_id', 'request_ordinal', 'method', 'checkpoint_id',
    'repeat', 'activation', 'group', 'phase', 'physical_gpu_uuid', 'output_tokens',
    'verification_calls', 'accepted_draft_tokens', 'native_accepted_draft_tokens',
    'committed_progress', 'finish_reason', 'request_latency_s', 'native_generate_s',
    'cache_maintenance_s', 'valid_measurement')


def table(path, rows, fields=None):
    rows = list(rows)
    fields = fields or sorted({key for row in rows for key in row})
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def format_interval(value, *, percent=True):
    if value is None or value['value'] is None:
        return 'undefined'
    suffix = '%' if percent else '×'
    lo, hi = value['ci95']
    return f"{value['value']:.4f}{suffix} (95% CI {lo:.4f}, {hi:.4f})"


def conclusion(summary):
    primary = summary['contrasts']['C_CODE58_vs_AREPO_CODE58']['request_latency_s_reduction_pct']
    if primary['ci95'] is None:
        text='The primary latency contrast is undefined; no C/AREPO benefit can be established.'
    elif primary['ci95'][0] > 0:
        text=('C reduces complete-request latency against AREPO on this fixed CODE panel. '
              'This supports the relative endpoint comparison in this one-seed setting, not an isolated loss-component explanation.')
    elif primary['ci95'][1] < 0:
        text='C increases complete-request latency against AREPO on this fixed CODE panel; the primary comparison is unfavorable.'
    else:
        text=('The primary C/AREPO latency interval includes zero. A latency benefit is not demonstrated, '
              'and this does not establish equivalence or preservation.')
    frozen=[summary['contrasts'][arm+'_vs_F'][metric] for arm in ARMS
            for metric in ('request_latency_s_reduction_pct','verification_call_reduction_pct')]
    if all(item['value'] is not None and item['value']<0 for item in frozen):
        text+=' All three adapted endpoints use more verification calls and take longer than frozen F in the pooled measurements.'
        if all(item['ci95'] is not None and item['ci95'][1]<0 for item in frozen):
            text+=(' All six descriptive intervals support those regressions. Adaptation is harmful in this setting, '
                   'even if C is better than the other adapted endpoints.')
        else:
            text+=(' These are numerical regressions; the intervals do not establish every regression. '
                   'An advantage among adapted endpoints must not be described as an overall benefit over F.')
    return text


def build_report(plan, output):
    """Keep the full audit private; ``public`` is the only publication directory."""
    output = Path(output)
    public = output/'public'
    public.mkdir(parents=True, exist_ok=False)
    audit = raw_audit.audit_run(plan, output/'private')
    evaluation = audit.get('evaluation', {})
    rows = evaluation.get('rows', [])
    summary = None
    if audit['status']=='passed':
        try:
            summary=analysis.summarize(rows)
            raw_audit.compare_reductions(summary,evaluation['independent_summary'])
        except ValueError as error:
            audit['status']='failed'
            audit['errors'].append(str(error))
            raw_audit.write(output/'private/REDUCTION_FAILURE.json',dict(status='failed',error=str(error)))
    table(public/'ALL_REPEATS.csv', ({key:row.get(key) for key in PUBLIC_REPEAT_FIELDS} for row in rows), PUBLIC_REPEAT_FIELDS)
    training = audit.get('training', {})
    table(public/'TRAINING_COSTS.csv', training.get('costs', []), ('arm','event','metric','value'))
    table(public/'TRAINING_EXPOSURE.csv', training.get('exposure', []), ('arm','event','pass_index','metric','value'))
    table(public/'ENGINE_ACTIVATION_COSTS.csv', audit.get('overhead',{}).get('costs', []), ('source','metric','value'))
    table(public/'PHASE_COSTS.csv',audit.get('overhead',{}).get('phases',[]),('phase','status','wall_s','started_epoch'))
    table(public/'OWN_FEEDBACK_CURVES.csv', training.get('event_curves', []),
          ('arm','event','requests','pre_update_checkpoint',*raw_audit.COUNTS,*raw_audit.TIMES))
    table(public/'WORK.csv', audit.get('work', []),
          ('kind','limit','attempted','completed','unknown','unattempted','unused_allowance'))
    status = dict(status=audit['status'], execution_commit=plan['execution_commit'],
        report_commit='Recorded by the publishing Git commit; distinct from execution commit',
        raw_audit_scope=audit['scope'], raw_audit_errors=len(audit['errors']),
        evaluation_status=evaluation.get('status','unavailable'),
        independently_validated=evaluation.get('independently_validated',0),
        raw_evaluation_failures=len(evaluation.get('errors',[])),
        additional_model_work=0, planned_scored_generations=15360, planned_warmups=240,
        methods=list(METHODS), training_arms=training.get('arms',{}))
    raw_audit.write(public/'AUDIT_SUMMARY.json',status)
    execution=evaluation.get('execution_status',{})
    raw_audit.write(public/'VALIDITY.json',{key:execution.get(key) for key in
        ('status','planned','attempted','returned','completed_valid','invalid','unknown','unattempted','automatic_retry')})
    clock=audit.get('clock',{})
    observed=audit.get('audit_observed_epoch')
    raw_audit.write(public/'CAMPAIGN_COSTS.json',dict(
        audit_wall_s=audit.get('audit_wall_s'),occupied_card_s=clock.get('completed_card_s'),
        elapsed_until_audit_s=(observed-clock['first_model_load_epoch'] if observed and clock else None),
        elapsed_scope='Original model-load clock through this CPU audit; final report writing/publication follows.',
        nested_timer_scope=training.get('timer_scope'),
        activation_timer_scope=audit.get('overhead',{}).get('overlap'),
        final_publication_s=None))
    if 'endpoint_identities' in audit:
        raw_audit.write(public/'ENDPOINT_IDENTITIES.json',audit['endpoint_identities'])
    lines = [f"Execution commit: `{plan['execution_commit']}`. Report revision is recorded separately by its Git commit.",
             '',f"Status: **{audit['status']}**. This report adds no model executions.",'']
    if audit['status'] == 'passed':
        raw_audit.write(public/'RESULTS.json',summary)
        paired = []
        for item in summary['paired']:
            for method in METHODS:
                paired.append(dict(request_id=item['request_id'],method=method,
                    **{key:value for key,value in item[method].items() if key!='repeat_times'}))
        table(public/'PER_REQUEST.csv',paired)
        comparisons = []
        for name, item in summary['contrasts'].items():
            for metric, value in item.items():
                if isinstance(value,dict) and 'ci95' in value:
                    comparisons.append(dict(contrast=name,metric=metric,value=value['value'],
                        lower95=None if value['ci95'] is None else value['ci95'][0],
                        upper95=None if value['ci95'] is None else value['ci95'][1],status=value['status']))
        table(public/'CONTRASTS.csv',comparisons)
        raw_audit.write(public/'BOOTSTRAP.json',summary['bootstrap'])
        lines += [conclusion(summary),'',
            'Primary C/AREPO complete-request time reduction: **'+format_interval(
                summary['contrasts']['C_CODE58_vs_AREPO_CODE58']['request_latency_s_reduction_pct'])+'**.','',
            '| Method | Calls | Native accepted/call | Committed/call | Complete time (s) | Native time (s) | Tokens/s |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for method in METHODS:
            row=summary['totals'][method]
            calls='N/A' if row['verification_calls'] is None else str(int(row['verification_calls']))
            accepted='N/A' if row['native_accepted_per_call'] is None else f"{row['native_accepted_per_call']:.5f}"
            progress='N/A' if row['committed_per_call'] is None else f"{row['committed_per_call']:.5f}"
            lines.append(f"| {method} | {calls} | {accepted} | {progress} | {row['request_latency_s']:.3f} | {row['native_generate_s']:.3f} | {row['tokens_per_s']:.3f} |")
        lines += ['', '| Contrast | Complete-request reduction | Call reduction | Wins / regressions / ties |',
                  '|---|---:|---:|---:|']
        for name,item in summary['contrasts'].items():
            count='N/A' if item['baseline']=='AR' else format_interval(item['verification_call_reduction_pct'])
            outcomes='N/A' if item['baseline']=='AR' else f"{item['wins']} / {item['regressions']} / {item['ties']}"
            lines.append(f"| {name} | {format_interval(item['request_latency_s_reduction_pct'])} | {count} | {outcomes} |")
        lines += ['', 'All five endpoints preserve the target-only AR canonical tokens and termination. '
            'All repeats have consistent native counts. The independent reducer reproduces the request means, '
            'summed ratios, 10,000 paired bootstrap draws and intervals. Secondary intervals are descriptive and unadjusted.']
    else:
        lines += ['The declared experiment is incomplete or has a correctness failure. '
                  'No full-panel effect estimate or gain claim is produced. Returned records remain visible in ALL_REPEATS.csv; '
                  'WORK.csv distinguishes unknown work from unattempted work and unused qualification allowance.','',
                  '| Arm | Status | Completed events / 58 |','|---|---|---:|']
        for arm in ARMS:
            state=training.get('arms',{}).get(arm,{})
            lines.append(f"| {arm} | {state.get('status','unavailable')} | {state.get('events_completed','unavailable')} |")
    lines += ['',
        'The custom panel comes from the prospectively bounded OpenCodeInstruct reservoir, not the original author 1,024-row slice. '
        'All three online arms start from release with seed 17 and use their own feedback. Training context is 4,096 for this study; '
        'the output cap remains 2,048. Freshness is relative to the authenticated inventory, with historical coverage limits disclosed in READINESS.md.', '',
        'Counts use one validated trajectory per request; three repetitions estimate timing, not independent samples. '
        'Native acceptance is progress minus verification calls; stop-clipped matched drafts are counted directly from the trace. '
        'The initial prefill token is excluded from committed progress. Zero-call requests remain in output and latency totals. '
        'AR speculative counts are N/A.', '',
        'Training cost fields are individually preserved in TRAINING_COSTS.csv. Nested timers overlap and must not be summed arbitrarily. '
        'Own-feedback curves are descriptive pre-update trajectories, not controlled endpoint comparisons. '
        'The event and engine costs include initialization/publication work separately from scored request latency; '
        'no kernel-time breakdown is inferred from aggregate latency.', '',
        'Endpoint serving performance does not establish training-inclusive or asynchronous throughput. '
        'AREPO implements the released loss in the common framework, not the complete AdaFlash concurrent-serving system. '
        'One trained seed per arm does not establish seed robustness. No result triggers tuning or an additional run.', '',
        'Audit scope: independent CPU reconstruction of retained raw evidence by a collaborating agent with shared project context. '
        'This is not independent GPU replication. Private prompts, generated tokens and native traces remain outside the publication directory.']
    (public/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(status=audit['status'],public_directory=str(public),private_audit=str(output/'private/RAW_AUDIT.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(build_report(raw_audit.read(args.plan),args.output),sort_keys=True))


if __name__=='__main__':
    main()
