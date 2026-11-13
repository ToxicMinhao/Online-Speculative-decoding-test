"""Independent reduction of published scalar rows; never imports experiment code."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import numpy as np


def file_identity(path):
    return dict(name=path.name, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def audit(report_dir):
    root = Path(report_dir)
    paths = {name: root / name for name in (
        'paired_occurrences.csv', 'slots.csv', 'per_request.csv', 'SUMMARY.json')}
    def rows(name):
        with paths[name].open() as handle:
            return list(csv.DictReader(handle))
    observations = rows('paired_occurrences.csv')
    slots_output = rows('slots.csv')
    requests_output = rows('per_request.csv')
    summary = json.loads(paths['SUMMARY.json'].read_text())
    checks, failures = 0, []

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)

    def equal(actual, expected, label):
        check(math.isclose(float(actual), float(expected), rel_tol=1e-13, abs_tol=1e-12), label)

    grouped = defaultdict(list)
    seen = set()
    outcomes = ('earlier_break', 'first_mismatch_at_tag', 'accepts_through')
    matrix = Counter()
    passes = Counter()
    for row in observations:
        sid = (row['request_id'], int(row['canonical_start']), int(row['proposal_slot']))
        key = sid + (int(row['presentation_id']),)
        check(key not in seen, f'unique occurrence {key}')
        seen.add(key)
        passes[int(row['pass_index'])] += 1
        check(int(row['pass_index']) == int(row['presentation_id']) // 128, f'pass {key}')
        check(float(row['floor']) == .5 * min(float(row['parent_margin']), 1), f'floor {key}')
        native, recent = [], []
        for arm, target in (('native', native), ('recent', recent)):
            prefix, slot = int(row[arm+'_correct_prefix']), sid[2]
            kind = outcomes[0] if prefix + 1 < slot else outcomes[1] if prefix + 1 == slot else outcomes[2]
            check(row[arm+'_outcome'] == kind, f'outcome {arm} {key}')
            accepted = int(prefix >= slot)
            check(int(row[arm+'_accepts_through']) == accepted, f'acceptance {arm} {key}')
            direct = int(row[arm+'_direct_correct'])
            check(not accepted or direct == 1, f'accept/direct {arm} {key}')
            check(kind != outcomes[1] or direct == 0, f'mismatch/direct {arm} {key}')
            margin = float(row[arm+'_margin'])
            check(math.isfinite(margin), f'finite {arm} {key}')
            check(not (margin > 0 and not direct) and not (margin < 0 and direct), f'sign/direct {arm} {key}')
            check(int(row[arm+'_below_floor']) == int(margin < float(row['floor'])), f'below floor {arm} {key}')
            check(int(row[arm+'_wrong_at_tie']) == int(margin == 0 and not direct), f'wrong tie {arm} {key}')
            target.extend([accepted, kind, direct, margin])
        rescue, loss = int(recent[0] and not native[0]), int(native[0] and not recent[0])
        for name, value in [('rescue', rescue), ('loss', loss), ('net', rescue-loss)]:
            check(int(row[name]) == value, f'{name} {key}')
        grouped[sid].append((key[-1], int(row['pass_index']), native, recent, rescue, loss))
        matrix[(native[1], recent[1])] += 1

    check(len(observations) == 1629, '1629 occurrences')
    check(passes == Counter({0:814, 1:815}), 'pass counts')
    check(len(grouped) == 815, '815 slots')
    check(Counter(map(len, grouped.values())) == Counter({2:814, 1:1}), 'slot multiplicity')
    check([k for k,v in grouped.items() if len(v)==1] == [('train_00786',91,13)], 'singleton identity')
    slot_rows = {(r['request_id'], int(r['canonical_start']), int(r['proposal_slot'])):r for r in slots_output}
    check(set(slot_rows)==set(grouped), 'output slot keys')
    req = defaultdict(lambda: [0,Fraction(0),Fraction(0),Fraction(0),Fraction(0)])
    total = [Fraction(0)]*4
    for sid, values in grouped.items():
        count = len(values)
        check(len({v[1] for v in values}) == count, f'distinct passes {sid}')
        n, r, rescue, loss = [sum(Fraction(v[index][0]) if index in (2,3) else Fraction(v[index]) for v in values)/count for index in (2,3,4,5)]
        for i,value in enumerate((n,r,rescue,loss)):
            total[i] += value
            req[sid[0]][i+1] += value
        req[sid[0]][0] += 1
        target=slot_rows[sid]
        for name,value in [('native_accepts_through',n),('recent_accepts_through',r),('rescue',rescue),('loss',loss),('net',r-n)]:
            equal(target[name],value,f'slot {name} {sid}')
    check(len(req)==128, '128 requests')
    n,r,rescue,loss=total
    net=r-n
    check(net==rescue-loss,'net=rescue-loss exact')
    check(net <= 815-n,'net<=native headroom exact')
    primary=summary['primary']
    for name,value in [('native_retention_pct',100*n/815),('recent_retention_pct',100*r/815),('native_headroom_pct',100*(815-n)/815),('rescues',rescue),('losses',loss),('rescue_pct',100*rescue/815),('loss_pct',100*loss/815),('net_difference_pp',100*net/815)]:
        equal(primary[name],value,'primary '+name)
    exact=primary['exact_net_difference_pp']
    check(Fraction(exact['numerator'],exact['denominator'])==100*net/815,'exact primary fraction')
    check(sum(matrix.values())==1629,'outcome table total')
    request_rows={r['request_id']:r for r in requests_output}
    check(set(request_rows)==set(req),'request output keys')
    for rid,v in req.items():
        for name,value in zip(('slots','native_retained_equivalents','recent_retained_equivalents','rescue_equivalents','loss_equivalents'),v):
            equal(request_rows[rid][name],value,f'request {name} {rid}')
        equal(request_rows[rid]['net_equivalents'],v[2]-v[1],f'request net {rid}')
    order=sorted(req)
    check(summary['primary_interval']['request_order']==order,'bootstrap sorted request order')
    # Independent multiplicity reduction rather than the production indexed sum.
    rng=np.random.Generator(np.random.PCG64(1707))
    draws=rng.integers(0,len(order),size=(10000,len(order)))
    counts=np.array([req[k][0] for k in order],dtype=np.int64)
    # Every slot has one or two occurrences: twice the request net is integer.
    twice_net=np.array([int(2*(req[k][2]-req[k][1])) for k in order],dtype=np.int64)
    values=[]
    for draw in draws:
        multiplicity=np.bincount(draw,minlength=len(order))
        denominator=int(multiplicity@counts)
        check(denominator>0,'bootstrap denominator')
        values.append(50*int(multiplicity@twice_net)/denominator)
    ordered=sorted(values)
    def linear_quantile(p):
        x=p*(len(ordered)-1); lo=int(math.floor(x)); hi=int(math.ceil(x))
        return ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)
    lower,upper=linear_quantile(.025),linear_quantile(.975)
    equal(summary['primary_interval']['lower_pp'],lower,'bootstrap lower')
    equal(summary['primary_interval']['upper_pp'],upper,'bootstrap upper')
    check(summary['primary_interval']['draws']==10000,'10000 draws')
    check(summary['primary_interval']['undefined_draws']==0,'no undefined draws')
    result=dict(status='passed' if not failures else 'failed', checks=checks, failures=failures,
        inputs=[file_identity(p) for p in paths.values()],model_work=0,
        method='independent stdlib/Fraction grouping plus request multiplicities; no production analysis import',
        full_cohort=dict(native_retained_equivalents=float(n),recent_retained_equivalents=float(r),
            rescue_equivalents=float(rescue),loss_equivalents=float(loss),net_pp=float(100*net/815),
            native_headroom_pp=float(100*(815-n)/815), exact_net_pp=str(100*net/815)),
        primary_interval=dict(lower_pp=lower,upper_pp=upper,draws=10000,seed=1707),
        occurrence_outcome_table=[dict(native=a,recent=b,count=matrix[(a,b)]) for a in outcomes for b in outcomes])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--report',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); result=audit(args.report)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','failures','full_cohort','primary_interval')},indent=2))
    raise SystemExit(bool(result['failures']))
