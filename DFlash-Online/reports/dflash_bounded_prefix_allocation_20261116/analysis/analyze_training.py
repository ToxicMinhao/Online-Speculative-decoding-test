"""Read retained C7/C8 JSON evidence only; never import model/runtime packages.

Prefix summaries describe the implemented coefficient change, not its efficacy.
Only RESPONSE_OBSERVATIONS rows are reduced; embedded copies and accumulation
metrics are cross-checks, never additional samples. No historical/fresh timer
comparison is made by this reader.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


FIELDS = ('blocks', 'valid_positions', 'prefix_positions', 'changed_blocks', 'changed_positions',
          'prefix_weight_before', 'prefix_weight_after', 'weight_moved', 'mass_error_abs_sum',
          'mass_violations', 'bounds_violations', 'support_violations', 'nonfinite_values', 'delta_numerator')
VIOLATIONS = ('mass_violations', 'bounds_violations', 'support_violations', 'nonfinite_values')
SLOT_FIELDS = ('occurrences', 'native_weight_sum', 'candidate_weight_sum',
               'signed_weight_change', 'absolute_weight_change')
EXPECTED_WORK = dict(adam=261, response_forward=528, response_backward=528,
    training_initialization=3, feedback_generation=128, teacher_preparation=128,
    qualification_generation=8, warmup_generation=72, scored_generation=1152)


class Audit:
    def __init__(self, root):
        self.root = Path(root)
        self.errors, self.inputs, self.checks = [], {}, 0

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def read(self, relative, lines=False):
        path = self.root / relative
        raw = path.read_bytes()
        self.inputs[str(relative)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return [json.loads(line) for line in raw.splitlines() if line] if lines else json.loads(raw)

    def near(self, left, right, scale=1):
        # Same pinned allocation tolerance, summed over the contributing rows.
        return math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-7*max(1, scale))


def summarize_observations(rows, audit, event):
    totals, slots, lengths, responses = Counter(), defaultdict(Counter), Counter(), []
    for ordinal, row in enumerate(rows):
        tag = f'C{event}/presentation{ordinal}'
        audit.check(row['scope'] == 'one_original_response_forward', tag+': original-forward scope')
        audit.check(row['event_id'] == f'event_{event:03d}' and row['presentation_ordinal'] == ordinal
                    and row['pass_index'] == ordinal//128, tag+': event/pass/ordinal')
        summary, native = row['prefix_allocation'], row['sum_metrics']
        finite = all(math.isfinite(float(summary[key])) for key in FIELDS)
        audit.check(finite, tag+': finite allocation summaries')
        if not finite:
            continue
        for key in VIOLATIONS:
            audit.check(summary[key] == 0, tag+': '+key)
        counts = summary['prefix_length_counts']
        audit.check(len(counts) == 17 and counts[-1] == 0, tag+': block16, known seed excluded')
        audit.check(sum(counts) == summary['blocks'], tag+': block histogram denominator')
        audit.check(sum(i*n for i,n in enumerate(counts)) == summary['prefix_positions'],
                    tag+': prefix occurrence denominator')
        audit.check(audit.near(summary['prefix_weight_after'], summary['prefix_weight_before'], summary['blocks']),
                    tag+': aggregate prefix mass')
        groups = summary['by_prefix_length_and_slot']
        seen = set()
        for group in groups:
            length, slot = group['prefix_length'], group['slot']
            key = length, slot
            audit.check(key not in seen and 1 <= slot <= length <= 15, tag+': distinct valid prefix slot')
            seen.add(key)
            audit.check(group['occurrences'] == counts[length], tag+': slot occurrence denominator')
            before, after = group['native_weight_sum'], group['candidate_weight_sum']
            tol = 2e-7*group['occurrences'] + 2e-6*abs(before)
            audit.check(.75*before-tol <= after <= 1.25*before+tol, tag+': aggregate relative bounds')
            audit.check(audit.near(after-before, group['signed_weight_change'], group['occurrences']),
                        tag+': signed movement consistency')
            for name in SLOT_FIELDS:
                slots[key][name] += group[name]
        audit.check(sum(g['occurrences'] for g in groups) == summary['prefix_positions'], tag+': complete slot coverage')
        audit.check(audit.near(sum(g['native_weight_sum'] for g in groups), summary['prefix_weight_before'],
                               summary['prefix_positions']), tag+': slot native mass')
        audit.check(audit.near(sum(g['candidate_weight_sum'] for g in groups), summary['prefix_weight_after'],
                               summary['prefix_positions']), tag+': slot candidate mass')
        audit.check(audit.near(sum(g['absolute_weight_change'] for g in groups)/2, summary['weight_moved'],
                               summary['prefix_positions']), tag+': slot moved mass')
        audit.check(native['kplus_response_count'] == 1, tag+': one separately normalized response')
        w1,w2 = native['kplus_W1'],native['kplus_W2']
        expected1 = 1-.5*w2/(w1+2*w2) if w1>0 and w2>0 else 1.
        expected2 = 1+.5*w1/(w1+2*w2) if w1>0 and w2>0 else 1.
        audit.check(audit.near(native['kplus_coefficient1'],expected1)
                    and audit.near(native['kplus_coefficient2'],expected2), tag+': native first-error formula')
        audit.check(audit.near(native['kplus_first_error_weight_after'],w1+w2), tag+': native first-error mass')
        for name in FIELDS:
            totals[name] += summary[name]
        for length,count in enumerate(counts):
            lengths[length] += count
        responses.append(dict(event=event, presentation=ordinal, pass_index=row['pass_index'],
            record_id=row['record_id'], loss=row['loss'], **{k:summary[k] for k in FIELDS},
            first_error_coefficient1=native['kplus_coefficient1'], first_error_coefficient2=native['kplus_coefficient2'],
            first_error_weight_before=w1+w2, first_error_weight_after=native['kplus_first_error_weight_after']))
    slot_rows = []
    for (length,slot),values in sorted(slots.items()):
        count = values['occurrences']
        slot_rows.append(dict(event=event,prefix_length=length,slot=slot,**values,
            mean_native_coefficient=values['native_weight_sum']/count,
            mean_candidate_coefficient=values['candidate_weight_sum']/count,
            mean_signed_change=values['signed_weight_change']/count,
            mean_absolute_change=values['absolute_weight_change']/count))
    return dict(event=event,responses=len(rows),totals=dict(totals),
        prefix_length_counts=dict(lengths), changed_block_fraction=totals['changed_blocks']/totals['blocks']
            if totals['blocks'] else None,
        moved_prefix_mass_share=totals['weight_moved']/totals['prefix_weight_before']
            if totals['prefix_weight_before'] else None), responses, slot_rows


def inspect_event(audit, event, plan):
    base = f'tasks/C{event}'
    task = audit.read(base+'/RESULT.json')
    status = audit.read(base+'/STATUS.json')
    receipt = audit.read(base+f'/trainer/events/event_{event:03d}/RESULT.json')
    observations = audit.read(base+f'/trainer/events/event_{event:03d}/RESPONSE_OBSERVATIONS.jsonl', lines=True)
    windows = audit.read(base+f'/trainer/events/event_{event:03d}/WINDOW_OBSERVATIONS.jsonl', lines=True)
    audit.check(task['status']==status['status']==receipt['status']=='completed', f'C{event}: complete receipts')
    audit.check(task['execution_commit']==plan['execution_commit'] and task['binding']==plan['binding'], f'C{event}: code/binding')
    audit.check(len(observations)==256 and len(windows)==128 and len(receipt['records'])==256, f'C{event}: exact event work')
    audit.check(receipt['start_step']==(event-1)*128 and receipt['end_step']==event*128
                and receipt['optimizer_steps']==128 and task['endpoint']['presentations']==event*256,
                f'C{event}: persistent counters')
    audit.check(task['boundary']['export_exact'] is True, f'C{event}: complete recovery/export check')
    for row, observed in zip(receipt['records'],observations):
        audit.check(row['response_observation']==observed and row['sampled_anchors']==observed['sampled_anchors']
                    and row['record_id']==observed['record_id'], f'C{event}: original-row identity')
    for i, window in enumerate(windows):
        audit.check(window['window']==i+1 and window['global_step']==(event-1)*128+i+1
            and window['record_ids']==[r['record_id'] for r in observations[2*i:2*i+2]], f'C{event}: window identity')
        audit.check(window['learning_rate']==.0003, f'C{event}: unchanged LR')
    audit.check(len({r['record_id'] for r in observations})==128
                and all(n==2 for n in Counter(r['record_id'] for r in observations).values()), f'C{event}: two passes')
    summary, rows, slots = summarize_observations(observations,audit,event)
    audit.check(task['startup']['optimizer_steps']==0 and task['startup']['status']=='completed', f'C{event}: disposable startup')
    cost = dict(event=event, loading_s=task['loading_s'], startup_s=task['startup']['wall_s'],
        task_wall_s=status['wall_s'], event_wall_s=task['event_wall_s'],
        **task['costs'], boundary_check_s=task['boundary']['wall_s'],
        publication_wait_s=task['publication_wait_s'], physical_work=task['physical_work'],
        memory=audit.read(base+'/MEMORY.json'), sampled_process_tree_peak_rss_bytes=status['sampled_process_tree_peak_rss_bytes'])
    return summary,rows,slots,cost


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(root, output):
    audit = Audit(root)
    plan = audit.read('sealed/PLAN.json')
    events, responses, slots, costs = [], [], [], []
    for event in (7,8):
        summary, rows, groups, cost = inspect_event(audit,event,plan)
        events.append(summary);responses.extend(rows);slots.extend(groups);costs.append(cost)
    qualification = audit.read('qualification/training/STATUS.json')
    expected_cases = ['native','cap_zero','active','continued_next_pair','reloaded_next_pair']
    audit.check(qualification['status']=='passed' and [c['case'] for c in qualification['cases']]==expected_cases,
                'qualification: all five prespecified cases')
    for key in ('exact_native_cap0','exact_own_resume','actual_intervention','origin_restored'):
        audit.check(qualification[key] is True, 'qualification: '+key)
    active = qualification['cases'][2]
    audit.check(active['unchanged_native_inputs_and_first_error_terms'] is True,
                'qualification: direct native input and first-error equality')
    native, zero, active, continued, reloaded = qualification['cases']
    for key in ('forward','preclip','endpoint','block_inputs'):
        audit.check(native[key]==zero[key], 'qualification native/cap0 raw equality: '+key)
    for key in ('forward','preclip','endpoint','block_inputs','allocation'):
        audit.check(continued[key]==reloaded[key], 'qualification own-resume raw equality: '+key)
    audit.check(active['block_inputs']==native['block_inputs'], 'qualification active native block inputs')
    for actual, reference in zip(active['forward'], native['forward']):
        keep=lambda row:{k:v for k,v in row['sums'].items() if k!='main_loss_numerator'}
        audit.check(keep(actual)==keep(reference), 'qualification active unchanged first-error/support sums')
    qualification_costs = [dict(case=row['case'],status=row['status'],wall_s=row['wall_s'],
        preclip_observer_s=row['preclip_observer_s'], block_observer_s=row['block_observer_s'],
        pair_wall_excluding_observers_s=row['pair_wall_excluding_observers_s']) for row in qualification['cases']]
    audit.check(all(row['status']=='completed' for row in qualification_costs), 'qualification complete case receipts')
    work = audit.read('WORK.json');clock=audit.read('CLOCK.json')
    audit.check(work['attempted']==work['completed']==EXPECTED_WORK and not work['pending'], 'exact completed work including startup')
    audit.check(not clock['active'], 'all occupied-card leases closed')
    audit.check(audit.near(sum(c['occupied_card_s'] for c in clock['finished']),clock['completed_card_s']),
                'occupied-card accounting sums leases, not overlapping wall time')
    report = dict(status='passed' if not audit.errors else 'failed', execution_commit=plan['execution_commit'],
        checks=audit.checks, errors=audit.errors, events=events, qualification=dict(
            status=qualification['status'], cases=expected_cases, optimizer_applications=qualification['optimizer_applications'],
            response_presentations=qualification['response_presentations'],startup_forward_backward_each=2,
            loading_s=qualification['loading_s'],startup_s=qualification['startup_s'],wall_s=qualification['wall_s'],
            peak_gpu_allocated_bytes=qualification['peak_gpu_allocated_bytes'],
            peak_gpu_reserved_bytes=qualification['peak_gpu_reserved_bytes'],
            case_costs=qualification_costs,
            unchanged_native_inputs_and_first_error_terms=active['unchanged_native_inputs_and_first_error_terms']),
        training_costs=costs, work=work, clock=clock,
        limits=['This reader reports only retained training/qualification evidence, not serving efficacy.',
                'Mass/bounds/support checks use recorded per-block violation counts and independent aggregate consistency.',
                'The active qualification fixture directly checks native first-error terms. Production zero support violations are implementation receipts.',
                'Per-event task/event/training/publication timers are nested; do not sum them as independent costs.',
                'Historical N/S training time is not a fresh matched overhead control.'], inputs=audit.inputs)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'TRAINING_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_csv(output/'RESPONSE_ALLOCATIONS.csv',responses)
    write_csv(output/'PREFIX_LENGTH_SLOT.csv',slots)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=analyze(args.root,args.output)
    print(json.dumps(dict(status=result['status'],checks=result['checks'],errors=result['errors'])))
    raise SystemExit(0 if result['status']=='passed' else 1)
