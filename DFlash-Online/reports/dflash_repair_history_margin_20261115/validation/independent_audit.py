"""Independent CPU reduction of this handoff's completed native evidence.

No production preparation, matching, analysis or report functions are imported.
Token-bearing records and the full-vocabulary QA tensor remain private.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def compact_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def fp32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def verify_original_v1_payload(seal):
    """Reconstruct only the declared v1 numeric histogram for its old hash.

    The file-level receipt remains a separate byte check. This does not rewrite
    the seal, its fingerprint, labels, pairs or histogram in the retained file.
    """
    body={k:v for k,v in seal.items() if k!='payload_sha256'}
    def hashed(value):
        return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
            separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    plain=hashed(body)
    if body.get('schema')!='repair_history_margin.pre_k6.v1':
        raise ValueError('Independent audit only supports the declared original v1 seal')
    original=body['coverage']['unmatched_depth_counts']
    numeric={}
    for key,count in original.items():
        if type(key) not in (int,str) or str(key) not in {str(j) for j in range(1,16)}:
            raise ValueError('Malformed original v1 histogram depth')
        depth=int(key)
        if depth in numeric or type(count) is not int or count<0:
            raise ValueError('Malformed original v1 histogram count or duplicate depth')
        numeric[depth]=count
    restored=dict(body,coverage=dict(body['coverage'],unmatched_depth_counts=numeric))
    typed=hashed(restored)
    return dict(original_payload_sha256=seal['payload_sha256'],
        plain_string_key_sha256=plain,plain_string_key_hash_passed=plain==seal['payload_sha256'],
        typed_v1_sha256=typed,typed_v1_hash_passed=typed==seal['payload_sha256'],
        declared_type_reconstruction='coverage.unmatched_depth_counts keys restored to integers1–15 for original v1 hash only',
        retained_payload_rewritten=False)


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []

    def equal(self, name, observed, expected):
        self.total += 1
        if observed != expected:
            self.failures.append(dict(name=name, observed=observed, expected=expected))

    def near(self, name, observed, expected):
        self.total += 1
        if observed is None or expected is None:
            good = observed == expected
        else:
            good = bool(np.isclose(observed, expected, rtol=1e-12, atol=1e-14))
        if not good:
            self.failures.append(dict(name=name, observed=observed, expected=expected))


def native_measure(block, parent, checks, label):
    start = block['output_start']
    suffix = parent['response_ids'][start+1:start+16]
    for i, token in enumerate(suffix):
        if token in parent['stop_token_ids']:
            suffix = suffix[:i+1]
            break
    if block['anchor_token'] in parent['stop_token_ids']:
        suffix = []
    proposals = block['draft_tokens']
    prefix = next((i for i,(p,y) in enumerate(zip(proposals,suffix)) if p != y), len(suffix))
    accepted = block['accepted_draft_tokens']
    checks.equal(label+':prefix_accepted',proposals[:accepted],suffix[:accepted])
    checks.equal(label+':accepted_within_boundary',accepted <= len(suffix),True)
    checks.equal(label+':canonical_commit',block['committed_tokens'],parent['response_ids'][start+1:start+1+len(block['committed_tokens'])])
    checks.equal(label+':commit_count',block['committed_progress'],len(block['committed_tokens']))
    genuine = prefix < len(suffix) and prefix == accepted
    return dict(accepted=accepted, limit=len(suffix), genuine=genuine, direct=[p == y for p,y in zip(proposals,suffix)])


def outcome(measure, j):
    if not 1 <= j <= measure['limit']:
        raise ValueError('Audit outcome outside saved boundary')
    a = measure['accepted']
    if a < j and not measure['genuine']:
        raise ValueError('Audit finds noncanonical/non-genuine native rejection')
    kind = 'accepts_through' if a >= j else 'exact_slot_rejection' if a == j-1 else 'earlier_rejection'
    return kind, measure['direct'][j-1]


def bootstrap(rows):
    by_request = defaultdict(list)
    for source, value in rows:
        by_request[source].append(value)
    if not by_request:
        return None
    groups = [by_request[s] for s in sorted(by_request)]
    numerators = np.array([sum(g) for g in groups],dtype=np.float64)
    denominators = np.array([len(g) for g in groups],dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(20261115))
    scores=[]
    for first in range(0,10000,500):
        sample=rng.integers(len(groups),size=(min(500,10000-first),len(groups)))
        scores.extend((numerators[sample].sum(1)/denominators[sample].sum(1)).tolist())
    return [float(x) for x in np.quantile(scores,[.025,.975])]


def audit(folder, qa_path=None):
    folder=Path(folder)
    checks=Checks()
    plan=read(folder/'PLAN.json')
    seal=read(folder/'PRE_K6.json')
    analysis=read(folder/'ANALYSIS.json')
    hash_validation=verify_original_v1_payload(seal)
    checks.equal('pre_K6_original_typed_v1_payload_hash',hash_validation['typed_v1_hash_passed'],True)
    receipt_path=folder/'PRE_K6_SEAL.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        for name,expected in receipt['files'].items():
            data=(folder/name).read_bytes()
            checks.equal('pre_K6_file:'+name,dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),expected)
        checks.equal('pre_K6_not_loaded_at_seal',receipt['k6_loaded'],False)
    requests=plan['requests']
    starts,raw_margin_rows,all_measures={},[],{}
    expected_labels={(r['source_id'],r['start_offset']):r for r in seal['labels']}
    all_native_fields=('output_start','anchor_token','draft_tokens','target_argmax','positions','accepted_draft_tokens',
        'native_accepted_draft_tokens','committed_progress','committed_tokens','finished','stop_clipped')
    for request in requests:
        source=request['source_id']
        current=folder/'requests'/request['request_id']
        parent=read(current/'PARENT.json')
        margins=read(current/'MARGINS.json')['starts']
        for ordinal, block in enumerate(parent['metadata']['trace']):
            start=block['output_start']; key=(source,start)
            n5=read(current/f'{ordinal:04d}_K5.json')['blocks'][0]
            n4=read(current/f'{ordinal:04d}_K4.json')['blocks'][0]
            for field in all_native_fields:
                checks.equal(f'{source}:{start}:neutral:{field}',n5[field],block[field])
            k5=native_measure(n5,parent,checks,f'{source}:{start}:K5')
            k4=native_measure(n4,parent,checks,f'{source}:{start}:K4')
            rows=margins[str(start)]
            checks.equal(f'{source}:{start}:margin_count',len(rows),k5['accepted'])
            values={}
            for row in rows:
                j=row['slot'];canonical=parent['response_ids'][start+j]
                checks.equal(f'{source}:{start}:{j}:chosen',row['chosen_id'],canonical)
                checks.equal(f'{source}:{start}:{j}:canonical',row['canonical_id'],canonical)
                checks.equal(f'{source}:{start}:{j}:margin',row['margin'],fp32(row['canonical_score']-row['competitor_score']))
                checks.equal(f'{source}:{start}:{j}:nonnegative',row['margin']>=0,True)
                checks.equal(f'{source}:{start}:{j}:qa_hash',row['qa_hash'],compact_hash([20261115,source,start,j]))
                values[j]=row['margin'];raw_margin_rows.append(row)
            recently=k4['genuine'] and k5['accepted'] >= k4['accepted']+1
            item=dict(source_id=source,start_offset=start,a4=k4['accepted'],a5=k5['accepted'],valid_proposal_count=k5['limit'],
                k4_first_error=k4['genuine'],recently_repaired=recently,j=k4['accepted']+1 if recently else None,
                margins=[dict(slot=j,margin=values[j]) for j in sorted(values)])
            checks.equal(f'{source}:{start}:label',item,expected_labels.get(key))
            starts[key]=item
            outcome_path=current/f'K6_{start:05d}.json'
            if outcome_path.exists():
                k6=read(outcome_path)['blocks'][0]
                all_measures[key]=native_measure(k6,parent,checks,f'{source}:{start}:K6')
                for field in ('output_start','anchor_token','positions'):
                    checks.equal(f'{source}:{start}:K6_conditioning:{field}',k6[field],block[field])
    checks.equal('label_count',len(starts),len(seal['labels']))
    raw_outcomes=read(folder/'OUTCOMES.json')
    checks.equal('recorded_K6_count',len(all_measures),len(raw_outcomes))
    for recorded in raw_outcomes:
        key=(recorded['source_id'],recorded['start_offset']); measured=all_measures[key]
        checks.equal(str(key)+':a6',recorded['a6'],measured['accepted'])
        checks.equal(str(key)+':valid_limit',recorded['valid_proposal_count'],measured['limit'])
        checks.equal(str(key)+':genuine',recorded['k6_first_error'],measured['genuine'])
        checks.equal(str(key)+':direct',recorded['direct_correct'],{str(j):v for j,v in enumerate(measured['direct'],1)})
    by_request=defaultdict(list)
    for row in starts.values(): by_request[row['source_id']].append(row)
    independently_paired=[];structural={};all_repairs=0
    for row in starts.values():
        if not row['recently_repaired']:continue
        all_repairs+=1;j=row['j'];m=row['margins'][j-1]['margin'];source=row['source_id'];start=row['start_offset']
        controls=[c for c in by_request[source] if c['start_offset']!=start and c['a5']==row['a5'] and min(c['a4'],c['a5'],c['valid_proposal_count'])>=j]
        structural[source,start]=controls
        eligible=[c for c in controls if abs(c['margins'][j-1]['margin']-m)<=.25 and (c['margins'][j-1]['margin']==0)==(m==0)]
        if eligible:
            chosen=min(eligible,key=lambda c:(abs(c['margins'][j-1]['margin']-m),compact_hash([20261115,source,start,c['start_offset'],j])))
            independently_paired.append(dict(source_id=source,repaired_start=start,comparator_start=chosen['start_offset'],j=j,
                repaired_margin=m,comparator_margin=chosen['margins'][j-1]['margin'],absolute_margin_difference=abs(chosen['margins'][j-1]['margin']-m),pair_weight=1))
    for pool in seal['structural_pools']:
        source,start,j=pool['source_id'],pool['repaired_start'],pool['j']
        expected=[dict(start_offset=c['start_offset'],margin=c['margins'][j-1]['margin']) for c in structural[source,start]]
        checks.equal(str((source,start))+':structural_pool',expected,pool['comparators'])
    checks.equal('all_repairs',all_repairs,seal['coverage']['all_repairs'])
    sealed_pairs=seal['matched_pairs']
    checks.equal('pair_count',len(independently_paired),len(sealed_pairs))
    for i,pair in enumerate(independently_paired):
        for field,value in pair.items():checks.equal(f'pair:{i}:{field}',value,sealed_pairs[i][field])
        pool=next(p for p in seal['structural_pools'] if (p['source_id'],p['repaired_start'])==(pair['source_id'],pair['repaired_start']))
        checks.equal(f'pair:{i}:all_structural_starts',[c['start_offset'] for c in structural[pair['source_id'],pair['repaired_start']]], [c['start_offset'] for c in pool['comparators']])
    if independently_paired:
        r=np.array([p['repaired_margin'] for p in independently_paired]);c=np.array([p['comparator_margin'] for p in independently_paired])
        sd=float(np.sqrt((np.var(r,ddof=1)+np.var(c,ddof=1))/2)) if len(r)>1 else 0.0
        smd=float(abs(r.mean()-c.mean())/sd) if sd else 0.0 if r.mean()==c.mean() else None
        checks.near('margin_SMD',smd,seal['balance']['absolute_smd'])
        reuse=Counter((p['source_id'],p['comparator_start']) for p in independently_paired)
        checks.equal('distinct_controls',len(reuse),seal['coverage']['distinct_comparator_starts'])
        checks.equal('max_reuse',max(reuse.values()),seal['coverage']['maximum_comparator_reuse'])
    deltas=[];structural_deltas=[];early=[];accept=[];direct=[]
    role_counts=[Counter(),Counter()]
    for pair in independently_paired:
        source=pair['source_id'];start=pair['repaired_start'];j=pair['j']
        repair,repair_direct=outcome(all_measures[source,start],j)
        control,control_direct=outcome(all_measures[source,pair['comparator_start']],j)
        role_counts[0][repair]+=1;role_counts[1][control]+=1
        deltas.append((source,int(repair=='exact_slot_rejection')-int(control=='exact_slot_rejection')))
        early.append(int(repair=='earlier_rejection')-int(control=='earlier_rejection'))
        accept.append(int(repair=='accepts_through')-int(control=='accepts_through'))
        direct.append(int(repair_direct)-int(control_direct))
        old=[outcome(all_measures[source,c['start_offset']],j)[0] for c in structural[source,start]]
        structural_deltas.append((source,int(repair=='exact_slot_rejection')-sum(x=='exact_slot_rejection' for x in old)/len(old)))
    n=len(deltas)
    if n:
        for name,values in [('delta',[x[1] for x in deltas]),('earlier_break_difference',early),('acceptance_through_difference',accept),('direct_correct_difference',direct)]:
            checks.near('primary:'+name,sum(values)/n,analysis['primary'][name])
        for role,counts in zip(('repaired','established'),role_counts):
            for kind in ('exact_slot_rejection','earlier_rejection','accepts_through'):
                checks.equal(role+':'+kind,counts[kind],analysis['primary'][role]['counts'][kind])
        ci=bootstrap(deltas);sci=bootstrap(structural_deltas)
        for name,value in zip(('lower','upper'),ci):checks.near('primary_CI:'+name,value,analysis['primary']['interval'][name])
        for name,value in zip(('lower','upper'),sci):checks.near('structural_CI:'+name,value,analysis['structural']['interval'][name])
        checks.near('structural_delta',sum(v for _,v in structural_deltas)/n,analysis['structural']['delta'])
    else:ci=sci=None
    qa_status='unavailable'
    if qa_path:
        import torch
        samples=torch.load(qa_path,map_location='cpu',weights_only=False)['rows']
        expected=sorted(raw_margin_rows,key=lambda row:row['qa_hash'])[:32]
        checks.equal('qa_bottom_hash_selection',[s['qa_hash'] for s in samples],[r['qa_hash'] for r in expected])
        by_hash={r['qa_hash']:r for r in raw_margin_rows}
        for sample in samples:
            row=by_hash[sample['qa_hash']]
            for field in ('source_id','start_offset','slot','canonical_id','chosen_id','canonical_score','competitor_score','competitor_id','margin','native_dtype'):
                checks.equal(sample['qa_hash']+':row_binding:'+field,sample[field],row[field])
            checks.equal(sample['qa_hash']+':tensor_dtype',str(sample['native_scores'].dtype),'torch.bfloat16')
            checks.equal(sample['qa_hash']+':finite_scores',bool(torch.isfinite(sample['native_scores']).all()),True)
            values=sample['native_scores'].tolist();y=sample['canonical_id']
            winning=max(range(len(values)),key=lambda i:(values[i],-i))
            rival=max((i for i in range(len(values)) if i!=y),key=lambda i:(values[i],-i))
            checks.equal(sample['qa_hash']+':native_argmax',winning,sample['chosen_id'])
            checks.equal(sample['qa_hash']+':rival',rival,sample['competitor_id'])
            checks.equal(sample['qa_hash']+':canonical_score',values[y],sample['canonical_score'])
            checks.equal(sample['qa_hash']+':competing_score',values[rival],sample['competitor_score'])
            checks.equal(sample['qa_hash']+':qa_margin',fp32(values[y]-values[rival]),sample['margin'])
        qa_status='checked'
    return dict(status='passed' if not checks.failures else 'failed',checks=checks.total,failures=checks.failures,
        native_requests=len(requests),native_starts=len(starts),accepted_margin_rows=len(raw_margin_rows),
        original_v1_payload_hash_validation=hash_validation,
        repairs=all_repairs,matched_pairs=n,primary_delta=sum(v for _,v in deltas)/n if n else None,
        primary_interval=ci,structural_interval=sci,margin_full_vocabulary_qa=qa_status,
        independence='Separate formulas/code; no production analysis imports. Same executor team; no extra GPU run.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True)
    parser.add_argument('--qa-tensor')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit(args.folder,args.qa_tensor)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2))
    raise SystemExit(0 if result['status']=='passed' else 1)
