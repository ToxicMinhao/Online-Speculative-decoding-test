"""Publication-only reduction; deliberately imports no production analyzer."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

TASK = Path('/home/minhao2/EAGLE_SPEC/research/dflash_repair_history_margin_20261115')
RAW = TASK/'delivery/retained/campaign_001'
PUBLIC = RAW/'final_report'
checks, failures = 0, []
def check(ok, label):
    global checks
    checks += 1
    if not ok: failures.append(label)
def read(path): return json.loads(path.read_text())
def identity(path):
    raw=path.read_bytes(); return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

a = read(RAW/'ANALYSIS.json'); summary=read(PUBLIC/'SUMMARY.json'); b=summary['result']; seal=read(RAW/'PRE_K6.json')
seal_before=identity(RAW/'PRE_K6.json'); seal_receipt=read(RAW/'PRE_K6_SEAL.json')
check(seal_before==seal_receipt['files']['PRE_K6.json'],'original file seal matches retained bytes')
check(seal_before['sha256']=='f7bd53131c9358d4bbab59a83be12a299744f8313fd86247590780ac0e4a7678','original PRE_K6 sha preserved')
original_payload={k:v for k,v in seal.items() if k!='payload_sha256'}
untyped_hash=fingerprint(original_payload)
original_payload['coverage']=dict(original_payload['coverage'])
original_payload['coverage']['unmatched_depth_counts']={int(k):v for k,v in original_payload['coverage']['unmatched_depth_counts'].items()}
recovered_hash=fingerprint(original_payload)
check(recovered_hash==seal['payload_sha256']==a['pre_k6_payload_sha256']==summary['pre_k6_payload_sha256'],'legacy type reconstruction reproduces ORIGINAL payload fingerprint')
check(untyped_hash!=recovered_hash,'actual retained JSON reproduces original deserialization problem')
check((TASK/'delivery/retained/EXIT_CODE').read_text().strip()=='1','original EXIT=1 preserved')
check('ValueError: Pre-K+6 matching seal changed' in (TASK/'delivery/retained/supervisor.log').read_text(),'original report-builder failure preserved')
check(read(RAW/'RESULT.json')['status']=='completed','native RESULT completed before publication error')
check(read(RAW/'STATUS.json')['status']=='completed','native STATUS completed')

rounding=[]
def compare(x,y,path='result'):
    if isinstance(x,dict) and isinstance(y,dict):
        check(x.keys()==y.keys(),path+'.keys')
        for k in x.keys()&y.keys(): compare(x[k],y[k],path+'.'+str(k))
    elif isinstance(x,list) and isinstance(y,list):
        check(len(x)==len(y),path+'.length')
        for i,(xx,yy) in enumerate(zip(x,y)): compare(xx,yy,path+f'[{i}]')
    elif x!=y:
        allowed={'result.structural.delta':'exact_slot_rejection_difference',
                 'result.structural.earlier_break_difference':'earlier_rejection_difference'}
        if path in allowed:
            independently_summed=math.fsum(row[allowed[path]] for row in a['structural_outcomes'])/len(a['structural_outcomes'])
            check(x==independently_summed,path+'.native agrees with independent fsum')
            ordinary_sum = 0.0
            for row in a['structural_outcomes']:
                ordinary_sum += row[allowed[path]]
            ordinary_sum /= len(a['structural_outcomes'])
            check(y==ordinary_sum,path+'.report agrees exactly with Python3.11 ordered sum')
            rounding.append(dict(path=path,native=x,report=y,absolute_difference=abs(x-y),binary64_ulps=abs(x-y)/math.ulp(x),independent_fsum=independently_summed,independent_ordered_sum=ordinary_sum))
        else: check(False,path+'.values differ')
    else: check(True,path)
compare(a,b)

def cells(name,expected):
    with (PUBLIC/name).open(newline='') as f:
        reader=csv.DictReader(f); rows=list(reader); columns=reader.fieldnames
    check(len(rows)==len(expected),name+'.row_count')
    for i,(observed,wanted) in enumerate(zip(rows,expected)):
        for key in columns:
            value=wanted.get(key)
            text='' if value is None else json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else str(value)
            check(observed[key]==text,f'{name}[{i}].{key}')
    return rows

pair_rows=cells('MATCHED_PAIRS.csv',a['matched_pair_outcomes'])
cells('STRUCTURAL_OUTCOMES.csv',a['structural_outcomes'])
cells('PER_REQUEST.csv',a['per_request'])
cells('ALL_STARTS.csv',seal['labels'])
cells('REPAIR_COVERAGE.csv',seal['repairs'])
cells('K5_MARGINS.csv',[dict(source_id=r['source_id'],start_offset=r['start_offset'],**m) for r in seal['labels'] for m in r['margins']])
pool_rows=[dict(source_id=p['source_id'],repaired_start=p['repaired_start'],j=p['j'],a5=p['a5'],comparator_start=c['start_offset'],comparator_margin=c['margin']) for p in seal['structural_pools'] for c in p['comparators']]
cells('STRUCTURAL_POOLS.csv',pool_rows)
reuse=Counter((r['source_id'],int(r['comparator_start'])) for r in pair_rows)
reuse_rows=[dict(source_id=k[0],start_offset=k[1],uses=v) for k,v in sorted(reuse.items())]
cells('COMPARATOR_REUSE.csv',reuse_rows)
check(len(reuse)==a['coverage']['distinct_comparator_starts']==223,'distinct comparator reuse')
check(max(reuse.values())==a['coverage']['maximum_comparator_reuse']==2,'maximum reuse')
check(Counter(reuse.values())=={1:211,2:12},'reuse distribution')

lookup={(r['source_id'],r['start_offset']):r for r in read(RAW/'OUTCOMES.json')}
counts={'repaired':Counter(),'established':Counter()}; direct=Counter(); clusters=defaultdict(list)
for pair in seal['matched_pairs']:
    observed={}
    for name,offset_name in [('repaired','repaired_start'),('established','comparator_start')]:
        row=lookup[(pair['source_id'],pair[offset_name])]; accepted=row['a6'];j=pair['j']
        check(row['valid'] and 1<=j<=row['valid_proposal_count'],'joined outcome valid')
        label='accepts_through' if accepted>=j else 'exact_slot_rejection' if accepted==j-1 else 'earlier_rejection'
        if accepted<j:check(row['k6_first_error'],'genuine rejection when classified')
        counts[name][label]+=1;direct[name]+=int(row['direct_correct'][str(j)]);observed[name]=label
    clusters[pair['source_id']].append(int(observed['repaired']=='exact_slot_rejection')-int(observed['established']=='exact_slot_rejection'))
n=len(pair_rows);delta=(counts['repaired']['exact_slot_rejection']-counts['established']['exact_slot_rejection'])/n
check(n==235,'235 pairs')
for name in counts:
    check(dict(counts[name])==a['primary'][name]['counts'],name+' risk counts')
    check(sum(counts[name].values())==n,name+' full denominator')
    check(direct[name]==a['primary'][name]['direct_correct'],name+' direct correct count')
check(delta==a['primary']['delta'],'primary delta from independent raw outcome join')
keys=sorted(clusters);totals=np.array([sum(clusters[k]) for k in keys]);sizes=np.array([len(clusters[k]) for k in keys]);rng=np.random.Generator(np.random.PCG64(20261115))
selected=rng.integers(len(keys),size=(10000,len(keys)));draws=totals[selected].sum(axis=1)/sizes[selected].sum(axis=1);interval=np.quantile(draws,[.025,.975])
check(float(interval[0])==a['primary']['interval']['lower'],'independent bootstrap lower')
check(float(interval[1])==a['primary']['interval']['upper'],'independent bootstrap upper')
check(np.isfinite(draws).all() and a['primary']['interval']['undefined_draws']==0,'undefined bootstrap count')
check(a['coverage']['matched_repairs']>=100 and a['coverage']['matched_requests']>=40 and a['balance']['absolute_smd']<=.10,'fixed support criteria pass')
check(interval[0]>.05 and a['decision']=='additional_risk_above_five_percentage_points_supported','fixed decision consistent')

work=read(RAW/'BUDGET.json'); rows=list(csv.DictReader((PUBLIC/'WORK_ACCOUNTING.csv').open()))
for row in rows:
    kind=row['operation']
    for column in ('attempted','completed','failed','unknown'):
        check(int(row[column])==work.get(column,{}).get(kind,0),'work '+kind+' '+column)
    if row['planned']:
        check(int(row['planned'])-int(row['attempted'])==int(row['unattempted']),'work '+kind+' unattempted')
check(all(int(r['unattempted'])==0 for r in rows if r['unattempted']),'prescribed unattempted zero')
check(not work['failed'] and not work['unknown'],'no failed or unknown native work')
check(identity(RAW/'PRE_K6.json')==seal_before,'original seal bytes unchanged after audit')
receipt=dict(status='passed' if not failures else 'failed',checks=checks,failures=failures,
    scope='Independent publication/table/raw-outcome reduction; no production analysis imports or GPU work',
    author_overlap='Reviewer authored production analysis/report; separate implementation, not independent GPU replication',
    native_analysis_and_report='All fields exact except two explicitly documented binary64 structural aggregate differences',
    structural_binary64_differences=rounding,original_pre_k6_identity=seal_before,
    original_payload_sha256=seal['payload_sha256'],untyped_deserialized_hash=untyped_hash,
    recovered_original_typed_payload_hash=recovered_hash,original_exit_code=1,
    original_publication_failure_preserved=True,native_execution_status='completed',
    primary=dict(pairs=n,requests=len(keys),counts={k:dict(v) for k,v in counts.items()},direct_correct=dict(direct),delta=delta,bootstrap_interval=list(map(float,interval))),
    python=sys.version,numpy=np.__version__)
(TASK/'reviews/REPORT_TABLE_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','checks','failures','structural_binary64_differences')},indent=2))
if failures:raise SystemExit(1)
