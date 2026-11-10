"""Independent CPU audit of completed native verification-transition evidence.

Does not import the experiment analyzer or run any model. Intentionally refuses
partial campaigns: selected but missing rows must never become zero effects.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import time

import numpy as np


EXPECTED_COMMIT = 'd7b1311e7b2f910e935bf81b4e40282767d7540c'
DATASETS = ('gsm8k', 'mathqa')


def load(path):
    return json.loads(Path(path).read_text())


def equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f'{label}: {actual!r} != {expected!r}')


def full_walk(record, request, canonical):
    equal(record['prompt_ids'], request['prompt_ids'], 'prompt identity')
    equal(record['response_ids'], canonical['response_ids'], 'output identity')
    equal(record['finish_reason'], canonical['finish_reason'], 'termination identity')
    equal(record['max_new_tokens'], request['max_new_tokens'], 'original output cap')
    equal(record['stop_token_ids'], request['stop_token_ids'], 'stop IDs')
    traces = record['metadata']['trace']
    equal(record['metadata']['verification_calls'], len(traces), 'calls vs rounds')
    output = canonical['response_ids']
    assert 1 <= len(output) <= request['max_new_tokens']
    assert not any(t in request['stop_token_ids'] for t in output[:-1])
    equal('eos' if output[-1] in request['stop_token_ids'] else 'length',
          canonical['finish_reason'], 'EOS/cap rule')
    if canonical['finish_reason'] == 'length':
        equal(len(output), request['max_new_tokens'], 'cap termination')
    position = 0
    for i, block in enumerate(traces):
        equal(block['output_start'], position, 'contiguous start')
        equal(block['anchor_token'], output[position], 'known seed')
        amount = block['committed_progress']
        assert type(amount) is int and 1 <= amount <= 16
        equal(block['committed_tokens'], output[position+1:position+1+amount], 'committed suffix')
        equal(block['finished'], i == len(traces)-1, 'terminal round')
        equal(len(block['draft_tokens']), 15, 'all proposals retained')
        equal(len(block['target_argmax']), 16, 'all verifier predictions retained')
        position += amount
    equal(position, len(output)-1, 'complete walk')
    return traces


def derive_row(item, directory):
    record4 = load(directory/'KPLUS4_native.json')
    record6 = load(directory/'KPLUS6_native.json')
    canonical = item['canonical']
    output = canonical['response_ids']
    t4 = full_walk(record4, item['request'], canonical)
    t6 = full_walk(record6, item['request'], canonical)
    row = dict(dataset=item['dataset'], request_id=item['request_id'],
               selection_rank=item['selection_rank'], status='completed',
               total_calls4=len(t4), total_calls6=len(t6),
               endpoint_call_difference=len(t6)-len(t4))
    differing = next((i for i, pair in enumerate(zip(t4, t6))
                      if pair[0]['committed_progress'] != pair[1]['committed_progress']), None)
    prior = [i+1 for i, pair in enumerate(zip(t4, t6))
             if (differing is None or i < differing)
             and pair[0]['draft_tokens'] != pair[1]['draft_tokens']]
    row['prior_proposal_difference_rounds'] = prior
    row['prior_proposal_difference_count'] = len(prior)
    if differing is None:
        equal(len(t4), len(t6), 'no-divergence round count')
        row.update(transition_kind='none', divergence_round=None, boundary=None, interior=None,
                   first_rejected_position_repaired=None, repair_check='not_applicable',
                   C44=None,C46=None,C64=None,C66=None,D=0,R4=0,R6=0,M4=0,M6=0,
                   both_restart_positive=False)
        return row
    b4, b6 = t4[differing], t6[differing]
    equal(b4['output_start'], b6['output_start'], 'common transition prefix')
    round_number = differing+1
    progress4, progress6 = b4['committed_progress'], b6['committed_progress']
    end4 = b4['output_start']+1+progress4
    end6 = b6['output_start']+1+progress6
    remaining4, remaining6 = len(output)-end4, len(output)-end6
    row.update(transition_kind='advance' if progress6>progress4 else 'regression',
               divergence_round=round_number, progress4=progress4, progress6=progress6,
               successor_output_tokens4=end4, successor_output_tokens6=end6,
               remaining_tokens4=remaining4, remaining_tokens6=remaining6,
               terminal4=remaining4==0, terminal6=remaining6==0,
               boundary=min(remaining4,remaining6)<=16, interior=min(remaining4,remaining6)>16,
               C44=len(t4)-round_number, C66=len(t6)-round_number)
    start = b4['output_start']+1
    valid = min(15, len(output)-start)
    first_rejection = next((j for j in range(valid)
                            if b4['draft_tokens'][j] != b4['target_argmax'][j]), None)
    if first_rejection is None:
        row.update(first_rejected_position_repaired=None, preceding_matches_preserved=None,
                   first_rejected_proposal_position=None,
                   first_rejected_absolute_output_position=None, repair_check='no_valid_first_rejection')
    else:
        j=first_rejection
        equal(b4['draft_tokens'][:j],output[start:start+j],'pre-rejection canonical matches')
        equal(b4['target_argmax'][j],output[start+j],'first rejection canonical target')
        preceding=b6['draft_tokens'][:j]==output[start:start+j]
        repaired=preceding and b6['draft_tokens'][j]==output[start+j]
        row.update(first_rejected_position_repaired=repaired, preceding_matches_preserved=preceding,
                   first_rejected_proposal_position=j+1,
                   first_rejected_absolute_output_position=start+j,repair_check='checked')
    for key, offset in (('C46',end6),('C64',end4)):
        if offset==len(output):
            assert not (directory/f'{key}.json').exists(), 'terminal successor should not launch replay'
            row[key]=0
            continue
        raw=load(directory/f'{key}.json')
        equal(raw['status'],'completed','crossed execution status')
        assert raw['output_matches_canonical'] and raw['termination_matches']
        record=raw['record']
        for field in ('prompt_ids','max_new_tokens','stop_token_ids'):
            equal(record[field],item['request'][field],f'crossed {field}')
        equal(record['response_ids'],output,'crossed complete output')
        equal(record['finish_reason'],canonical['finish_reason'],'crossed termination')
        traces=record['metadata']['trace']
        equal(record['metadata']['verification_calls'],len(traces),'crossed trace length')
        equal(raw['verification_calls'],len(traces),'crossed receipt calls')
        equal(raw['original_remaining_budget'],item['request']['max_new_tokens']-offset,'remaining cap')
        position=offset-1
        for i,block in enumerate(traces):
            equal(block['output_start'],position,'crossed absolute position')
            equal(block['anchor_token'],output[position],'crossed seed')
            n=block['committed_progress']
            assert type(n) is int and 1<=n<=16
            equal(block['committed_tokens'],output[position+1:position+1+n],'crossed canonical progress')
            equal(block['finished'],i==len(traces)-1,'crossed terminal flag')
            position+=n
        equal(position,len(output)-1,'crossed suffix completes')
        row[key]=len(traces)
    row.update(D=row['C66']-row['C44'], R4=row['C46']-row['C44'], R6=row['C66']-row['C64'],
               M4=row['C64']-row['C44'], M6=row['C66']-row['C46'])
    equal(row['D'],row['endpoint_call_difference'],'native diagonal difference')
    equal(row['D'],row['R4']+row['M6'],'decomposition via K4')
    equal(row['D'],row['M4']+row['R6'],'decomposition via K6')
    row['both_restart_positive']=row['R4']>0 and row['R6']>0
    return row


def interval(values):
    finite=np.asarray(values,dtype=float)
    finite=finite[np.isfinite(finite)]
    return np.percentile(finite,[2.5,97.5],method='linear').tolist() if len(finite) else None


def audit_summary(rows, reported):
    rows=sorted(rows,key=lambda r:r['selection_rank'])
    equal(len(rows),64,'full sample size')
    equal([r['selection_rank'] for r in rows],list(range(64)),'full sealed rank set')
    equal(reported['status'],'completed','summary completion')
    equal(reported['transitions'],dict(Counter(r['transition_kind'] for r in rows)),'transition counts')
    for arm in (4,6):
        equal(reported[f'native_endpoint_total_calls{arm}'],sum(r[f'total_calls{arm}'] for r in rows),'pooled endpoint')
    equal(reported['endpoint_call_ties'],sum(r['endpoint_call_difference']==0 for r in rows),'endpoint ties')
    sample=np.random.Generator(np.random.PCG64(20261110)).integers(0,64,size=(10000,64))
    answer={'endpoint_calls4':sum(r['total_calls4'] for r in rows),
            'endpoint_calls6':sum(r['total_calls6'] for r in rows),
            'bootstrap_indices_sha256':hashlib.sha256(sample.tobytes()).hexdigest(),
            'transitions':dict(Counter(r['transition_kind'] for r in rows)),'strata':{}}
    for name in ('all_advances','interior_advances'):
        eligible=np.array([r['transition_kind']=='advance' and (name=='all_advances' or r['interior']) for r in rows])
        count=int(eligible.sum()); reported_stratum=reported['strata'][name]
        equal(reported_stratum['eligible_requests'],count,'eligible count')
        both=sum(bool(e and r['R4']>0 and r['R6']>0) for r,e in zip(rows,eligible))
        repaired=sum(bool(e and r['first_rejected_position_repaired'] is True) for r,e in zip(rows,eligible))
        equal(reported_stratum['both_restart_positive_count'],both,'both positive')
        equal(reported_stratum['repair_verified_count'],repaired,'verified repair count')
        equal(reported_stratum['repair_not_verified_count'],count-repaired,'unverified advance count')
        stratum={'n':count,'both_positive':both,'verified_repairs':repaired,'unverified_advances':count-repaired}
        sizes=eligible[sample].sum(axis=1)
        for effect in ('R4','R6'):
            values=np.array([r[effect] for r in rows],dtype=np.int64)
            contribution=values*eligible
            totals=contribution[sample].sum(axis=1)
            conditional=np.array([total/n if n else np.nan for total,n in zip(totals,sizes)])
            ci=interval(conditional) if count>=2 else None
            unconditional_ci=interval(totals/64)
            result=reported_stratum[effect]
            equal(result['conditional_ci95'],ci,f'{name} {effect} conditional bootstrap')
            equal(result['unconditional_ci95'],unconditional_ci,f'{name} {effect} unconditional bootstrap')
            equal(result['empty_stratum_resamples'],int(np.count_nonzero(sizes==0)),'empty strata')
            selected=values[eligible]
            mean=float(selected.mean()) if count else None
            equal(result['conditional']['mean'],mean,'conditional point')
            equal(result['unconditional']['mean'],float(contribution.mean()),'unconditional point')
            if count:
                equal(result['conditional']['median'],float(np.median(selected)),'conditional median')
                equal(result['conditional']['minimum'],float(selected.min()),'minimum')
                equal(result['conditional']['maximum'],float(selected.max()),'maximum')
                equal(result['conditional']['q25'],float(np.percentile(selected,25)),'Q1')
                equal(result['conditional']['q75'],float(np.percentile(selected,75)),'Q3')
            abs_total=int(np.abs(contribution).sum());pos_total=int(np.maximum(contribution,0).sum())
            abs_share=float(np.abs(contribution).max()/abs_total) if abs_total else None
            pos_share=float(np.maximum(contribution,0).max()/pos_total) if pos_total else None
            equal(result['concentration']['largest_absolute_share'],abs_share,'absolute concentration')
            equal(result['concentration']['largest_positive_share'],pos_share,'positive concentration')
            loo=[]
            for i,row in enumerate(rows):
                n=count-int(eligible[i]);total=int(contribution.sum()-contribution[i])
                value=float(total/n) if n else None
                equal(result['leave_one_out'][i]['request_id'],row['request_id'],'LOO identity')
                equal(result['leave_one_out'][i]['conditional_mean'],value,'LOO mean')
                equal(result['leave_one_out'][i]['unconditional_mean'],float(total/63),'LOO unconditional mean')
                loo.append(value)
            signs={'positive':sum(x is not None and x>0 for x in loo),
                   'negative':sum(x is not None and x<0 for x in loo),
                   'zero':sum(x==0 for x in loo),'empty':sum(x is None for x in loo)}
            equal(result['loo_conditional_signs'],signs,'LOO signs')
            stratum[effect]={'mean':mean,'ci95':ci,'unconditional_mean':float(contribution.mean()),
                             'unconditional_ci95':unconditional_ci,'empty_draws':int(np.count_nonzero(sizes==0)),
                             'largest_absolute_share':abs_share,'largest_positive_share':pos_share,'loo_signs':signs}
        answer['strata'][name]=stratum
    return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    status=load(args.evidence/'campaign/STATUS.json')
    equal(status['status'],'completed','Only completed evidence may receive final inference')
    equal(status['completed_requests'],128,'completed selected sample')
    assert status['failure'] is None
    plan=load(args.evidence/'PLAN.json')
    equal(plan['execution_commit'],EXPECTED_COMMIT,'scientific execution identity')
    rows=load(args.evidence/'campaign/ROWS.json')
    equal(len(rows),128,'all rows retained')
    by_id={(r['dataset'],r['request_id']):r for r in rows};equal(len(by_id),128,'unique requests')
    derived=[]
    for item in plan['requests']:
        directory=args.evidence/'campaign/raw'/item['dataset']/f"{item['selection_rank']:03d}"
        row=derive_row(item,directory)
        reported=by_id[(row['dataset'],row['request_id'])]
        for key,value in row.items():equal(reported[key],value,f"{row['dataset']}/{row['selection_rank']}/{key}")
        derived.append(row)
    summaries=load(args.evidence/'campaign/report/SUMMARY.json')
    result={d:audit_summary([r for r in derived if r['dataset']==d],summaries[d]) for d in DATASETS}
    result.update(status='passed',checked_requests=128,
                  independent_implementation=True,independent_review_team=False,
                  disclosure='Same-team review; auditor authored production analysis, but this independent arithmetic does not import it.',
                  model_work=0,wall_s=time.perf_counter()-started,source_commit=EXPECTED_COMMIT)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'DERIVED_ROWS.json').write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
