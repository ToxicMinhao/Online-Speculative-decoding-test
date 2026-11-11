#!/usr/bin/env python3
"""Independent scalar audit of raw repeats. No production or model imports.

Successor-control analysis requires its sealed plan and panel. Exact native token/weight validity
remains a separate audit; this script validates scalar identities and arithmetic.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np

ARMS = ('K', 'TRUE', 'SHUFFLED')
CONTRASTS = (('TRUE','SHUFFLED'), ('TRUE','K'), ('SHUFFLED','K'))
COUNTS = ('verification_calls','accepted_draft_tokens','native_accepted_draft_tokens','committed_progress','output_tokens')
TIMES = ('request_latency_s','native_generate_s','cache_maintenance_s')


def schedule(n):
    result = []
    for repeat in (0,1):
        for block in range(n//8):
            arms = list(ARMS[block%3:] + ARMS[:block%3])
            if repeat: arms.reverse()
            for position, arm in enumerate(arms):
                for ordinal in range(block*8, block*8+8):
                    result.append((ordinal, arm, repeat, block, position))
    return result


def publication_methods(slots):
    # Native activate() forces a first publication regardless of the initially loaded export.
    result = []
    for _, arm, _, _, _ in slots:
        if not result or result[-1] != arm: result.append(arm)
    return result


def boolean(value):
    if type(value) is bool: return value
    if value == 'True': return True
    if value == 'False': return False
    raise ValueError('Invalid boolean: '+repr(value))


def integer(value):
    if type(value) is int: return value
    if isinstance(value, str) and value and str(int(value)) == value: return int(value)
    raise ValueError('Invalid integer: '+repr(value))


def normalize(row):
    result = dict(row)
    for key in (*COUNTS,'panel_ordinal','repeat','block_ordinal','position','seed','event',
                'first_round_accepted_prefix_tokens','first_round_raw_prefix_tokens'):
        result[key] = integer(row[key])
    for key in TIMES:
        result[key] = float(row[key])
        assert math.isfinite(result[key]) and result[key] >= 0, key
        if key != 'cache_maintenance_s': assert result[key] > 0, key
    for key in ('cap_hit','valid_measurement','first_round_present'):
        result[key] = boolean(row[key])
    for key in ('errors','structural_errors'):
        value = row[key]
        result[key] = json.loads(value) if isinstance(value,str) else value
        assert result[key] == [], key
    for key in ('first_round_start_digest','first_round_output_start'):
        if result[key] in ('','None','null'): result[key] = None
    if result['first_round_output_start'] is not None:
        result['first_round_output_start'] = integer(result['first_round_output_start'])
    assert result['seed'] == 17 and result['event'] == 5 and result['dataset'] == 'gsm8k'
    assert result['validation_status'] == 'valid' and result['valid_measurement'] is True
    assert result['phase'] == 'final' and result['method'] in ARMS
    assert all(result[key] >= 0 for key in COUNTS)
    assert 0 < result['output_tokens'] <= 2048
    assert result['committed_progress'] == result['output_tokens'] - 1
    assert result['native_accepted_draft_tokens'] == result['committed_progress'] - result['verification_calls']
    assert result['accepted_draft_tokens'] <= result['committed_progress']
    assert result['finish_reason'] in ('eos','length')
    assert result['cap_hit'] == (result['finish_reason'] == 'length')
    if result['cap_hit']: assert result['output_tokens'] == 2048
    for key in ('first_round_accepted_prefix_tokens','first_round_raw_prefix_tokens'):
        assert 0 <= result[key] <= 15
    assert result['first_round_accepted_prefix_tokens'] <= result['first_round_raw_prefix_tokens']
    if result['first_round_present']:
        assert result['first_round_output_start'] == 0
        assert isinstance(result['first_round_start_digest'],str) and len(result['first_round_start_digest']) == 64
        assert result['verification_calls'] > 0
    else:
        assert result['verification_calls'] == 0 and result['output_tokens'] == 1 and result['finish_reason'] == 'eos'
        assert result['first_round_start_digest'] is None and result['first_round_output_start'] is None
        assert result['first_round_accepted_prefix_tokens'] == result['first_round_raw_prefix_tokens'] == 0
    return result


def compare_report_rows(raw_rows, csv_text):
    """Require the report CSV to preserve every raw field and execution order."""
    report_rows = list(csv.DictReader(csv_text.splitlines()))
    assert len(report_rows) == len(raw_rows), 'Report/raw timing coverage differs'
    checks = 0
    for ordinal, (raw, reported) in enumerate(zip(raw_rows, report_rows)):
        assert set(raw) == set(reported), ('Report/raw field coverage differs', ordinal)
        for key, value in raw.items():
            rendered = (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                        else '' if value is None else str(value))
            assert reported[key] == rendered, ('Report/raw value differs', ordinal, key)
            checks += 1
    return dict(status='exact_match', rows=len(report_rows), field_checks=checks,
                execution_order='identical_to_raw', fields='all_raw_fields_preserved')


def compute(rows, n, requests=None):
    assert type(n) is int and 8 <= n <= 256 and n % 8 == 0
    assert len(rows) == 6*n, 'Not complete6N coverage; no complete-case analysis'
    planned = schedule(n)
    observed = [(r['panel_ordinal'],r['method'],r['repeat'],r['block_ordinal'],r['position']) for r in rows]
    assert observed == planned, 'Sealed cyclic/reversed schedule changed'
    by_key = {(r['panel_ordinal'],r['method'],r['repeat']):r for r in rows}
    assert len(by_key) == len(rows), 'Duplicated original timing execution'
    pairs, ids = [], []
    for ordinal in range(n):
        item = dict(panel_ordinal=ordinal)
        shared = set()
        for arm in ARMS:
            first, second = [by_key[ordinal,arm,r] for r in (0,1)]
            identity = ('request_id','checkpoint_id','finish_reason','cap_hit','first_round_present',
                        'first_round_start_digest','first_round_output_start','first_round_accepted_prefix_tokens',
                        'first_round_raw_prefix_tokens',*COUNTS)
            assert all(first[k] == second[k] for k in identity), 'Repeat identity or count disagreement'
            if requests is not None:
                assert first['request_id'] == requests[ordinal]['request_id']
                assert ordinal == requests[ordinal]['panel_ordinal']
            item[arm] = {k:first[k] for k in (*COUNTS,'checkpoint_id','finish_reason','cap_hit',
                        'first_round_present','first_round_accepted_prefix_tokens')}
            item[arm].update({k:(first[k]+second[k])/2 for k in TIMES})
            shared.add((first['request_id'],first['output_tokens'],first['finish_reason'],
                        first['first_round_present'],first['first_round_start_digest']))
        assert len(shared) == 1, 'Cross-arm request/output/first-round identity disagreement'
        item['request_id'] = next(iter(shared))[0]
        ids.append(item['request_id']); pairs.append(item)
    assert len(set(ids)) == n
    checkpoints = {arm:{p[arm]['checkpoint_id'] for p in pairs} for arm in ARMS}
    assert all(len(v) == 1 for v in checkpoints.values())
    assert len({next(iter(v)) for v in checkpoints.values()}) == 3
    arrays = {a:{k:np.array([p[a][k] for p in pairs],dtype=np.float64) for k in (*COUNTS,*TIMES)} for a in ARMS}
    totals = {a:{k:float(v.sum()) for k,v in values.items()} for a,values in arrays.items()}
    for arm in ARMS:
        for key in COUNTS: totals[arm][key] = int(totals[arm][key])
        totals[arm]['cap_hit_requests'] = sum(p[arm]['cap_hit'] for p in pairs)
    indices = np.random.Generator(np.random.PCG64(20261111)).integers(0,n,size=(10000,n))
    repeats = {str(r):{a:{k:sum(row[k] for row in rows if row['repeat']==r and row['method']==a)
                         for k in (*COUNTS,*TIMES)} for a in ARMS} for r in (0,1)}
    comparisons = {}
    for a,b in CONTRASTS:
        metrics = {}
        for key in ('verification_calls',*TIMES):
            av,bv = arrays[a][key],arrays[b][key]
            denominator = bv[indices].sum(axis=1)
            ratios = np.full(len(indices),np.nan)
            np.divide(av[indices].sum(axis=1),denominator,out=ratios,where=denominator>0)
            draws = 100*(1-ratios)
            finite = draws[np.isfinite(draws)]
            metrics[key] = dict(point_pct=float(100*(1-av.sum()/bv.sum())) if bv.sum() else None,
                candidate_sum=float(av.sum()),reference_sum=float(bv.sum()),
                ci95_pct=np.quantile(finite,[.025,.975],method='linear').tolist() if len(finite) else None,
                undefined_bootstrap_draws=len(draws)-len(finite))
        delta=arrays[b]['verification_calls']-arrays[a]['verification_calls']
        prefix=np.array([p[a]['first_round_accepted_prefix_tokens']-p[b]['first_round_accepted_prefix_tokens'] for p in pairs])
        comparisons[a+'_vs_'+b] = dict(metrics=metrics,
            calls_saved=int(np.maximum(delta,0).sum()),calls_added=int(np.maximum(-delta,0).sum()),
            call_wins=int((delta>0).sum()),call_losses=int((delta<0).sum()),call_ties=int((delta==0).sum()),
            sweep_latency_reduction_pct=[100*(1-repeats[str(r)][a]['request_latency_s']/repeats[str(r)][b]['request_latency_s']) for r in (0,1)],
            first_round=dict(wins=int((prefix>0).sum()),losses=int((prefix<0).sum()),ties=int((prefix==0).sum()),
                gained=int(np.maximum(prefix,0).sum()),lost=int(np.maximum(-prefix,0).sum()),
                seed_eos_requests=sum(not p[a]['first_round_present'] for p in pairs)))
    primary = comparisons['TRUE_vs_SHUFFLED']['metrics']['verification_calls']['ci95_pct']
    versus_k = comparisons['TRUE_vs_K']['metrics']['verification_calls']['ci95_pct']
    assert primary is not None and versus_k is not None, 'Undefined primary call contrast'
    outcome = ('informative_placement_supported' if primary[0] > 0 else
               'prioritization_unfavorable_for_this_fork' if primary[1] < 0 else
               'successor_placement_unresolved')
    latency = {}
    for name,effect in comparisons.items():
        ci = effect['metrics']['request_latency_s']['ci95_pct']
        sweeps = effect['sweep_latency_reduction_pct']
        latency[name] = dict(benefit_supported=ci[0]>0 and all(v>0 for v in sweeps),
                            slowdown_supported=ci[1]<0 and all(v<0 for v in sweeps))
        a,b = name.split('_vs_')
        delta = arrays[b]['request_latency_s']-arrays[a]['request_latency_s']
        effect['request_latency_directions'] = dict(wins=int((delta>0).sum()),
            losses=int((delta<0).sum()),ties=int((delta==0).sum()))
    return dict(request_count=n,timing_rows=len(rows),method_totals=totals,comparisons=comparisons,
        repeat_totals=repeats,paired_requests=pairs,
        means_per_request_s={a:{k:totals[a][k]/n for k in TIMES} for a in ARMS},
        classification=dict(outcome=outcome,incremental_verifier_benefit_vs_K=versus_k[0]>0,
            informative_placement_and_incremental_verifier_benefit=primary[0]>0 and versus_k[0]>0,
            latency=latency,automatic_followup_authorized=False),
        expected_publications=len(publication_methods(planned)))


MANIP_TOTALS = ('eligible_errors', 'ineligible_errors', 'mixed_strata', 'flipped_bits',
                'uniform_expected_flips', 'reassigned_weight', 'projection_calls',
                'projection_s', 'assignment_s')


def check_manipulation(observation):
    """Rebuild donor maps from compact block rows; no native tensor reconstruction."""
    row = observation['successor_shuffle']
    assert row['schema'] == 'kplus_successor_shuffle_v1' and row['status'] == 'passed'
    assert row['seed'] == 20261111 and row['identity'] is False
    assert row['response_id'] == observation['record_id']
    for flag in ('class_counts_exact', 'mass_preserved', 'selection_preserved', 'denominator_preserved'):
        assert row[flag] is True, flag
    assert 1 <= row['projection_calls'] <= 64 and row['projection_completed'] == row['projection_calls']
    assert [b['anchor_offset'] for b in row['blocks']] == row['anchor_offsets']
    errors = [b for b in row['blocks'] if b['first_error_position'] is not None]
    blocks = {b['anchor_offset']:b for b in errors}
    assert len(blocks) == len(errors), 'Duplicate first-error anchor'
    for block in row['blocks']:
        if block['first_error_position'] is None:
            assert block['eligible_successor'] is False
            assert block['true_priority'] is block['assigned_priority'] is block['donor_anchor_offset'] is None
    grouped = {}
    ineligible = 0
    for anchor, block in blocks.items():
        pos = block['first_error_position']
        if block['eligible_successor']:
            assert type(pos) is int and 1 <= pos <= 14
            assert block['true_priority'] in (1, 2) and block['assigned_priority'] in (1, 2)
            grouped.setdefault(pos, []).append(anchor)
        elif pos is not None:
            ineligible += 1
            assert block['true_priority'] == block['assigned_priority'] == 1
    strata = {s['position']:s for s in row['strata']}
    assert len(strata) == len(row['strata']) and set(strata) == set(grouped)
    eligible, mixed, flips, benchmark = 0, 0, 0, 0.0
    for pos, anchors in grouped.items():
        recipients = sorted(anchors)
        def key(anchor):
            encoded = json.dumps([20261111,row['response_id'],pos,anchor],
                                 ensure_ascii=False,separators=(',',':')).encode('utf-8')
            return hashlib.sha256(encoded).digest(), anchor
        donors = sorted(anchors, key=key) if len(anchors) > 1 else recipients
        n = len(anchors)
        k = sum(blocks[a]['true_priority'] == 2 for a in anchors)
        changed = 0
        for recipient, donor in zip(recipients, donors):
            b = blocks[recipient]
            assert b['donor_anchor_offset'] == donor
            assert b['assigned_priority'] == blocks[donor]['true_priority']
            changed += b['assigned_priority'] != b['true_priority']
        expected = dict(n=n,k=k,recipients=recipients,donors=donors,true_high_count=k,
                        assigned_high_count=k,flipped_bits=changed,uniform_expected_flips=2.0*k*(n-k)/n)
        assert all(strata[pos][name] == value for name,value in expected.items()), (pos,expected)
        eligible += n; mixed += 0 < k < n; flips += changed
        benchmark += expected['uniform_expected_flips']
    assert (eligible,ineligible,mixed,flips) == tuple(row[k] for k in MANIP_TOTALS[:4])
    assert math.isclose(benchmark,row['uniform_expected_flips'],rel_tol=1e-12,abs_tol=1e-12)
    for name in ('W1','W2','coefficient1','coefficient2','error_mass'):
        assert math.isclose(row['true_'+name],row['assigned_'+name],rel_tol=2e-6,abs_tol=2e-7), name
    sums = observation['sum_metrics']
    assert sums['shuffle_response_count'] == 1
    for name in (*MANIP_TOTALS,'coefficient_l1_weight','true_loss','assigned_loss'):
        value = row[name]
        assert isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        if name not in ('true_loss','assigned_loss'): assert value >= 0
        assert sums['shuffle_'+name] == struct.unpack('f',struct.pack('f',value))[0], name
    for name in ('W1','W2','coefficient1','coefficient2'):
        assert sums['kplus_'+name] == row['assigned_'+name]
    assert sums['kplus_S'] == row['selected_weight']
    assert sums['kplus_first_error_weight_after'] == row['assigned_error_mass']
    assert observation['loss'] == row['assigned_loss']
    return row


def check_ledger(rows, expected):
    started, completed = {k:set() for k in expected}, {k:set() for k in expected}
    for row in rows:
        key, ordinal = row['kind'], integer(row['ordinal'])
        assert key in expected and 1 <= ordinal <= expected[key]
        if row['phase'] == 'started':
            assert ordinal == len(started[key])+1
            started[key].add(ordinal)
        else:
            assert row['phase'] == 'completed' and ordinal in started[key] and ordinal not in completed[key]
            completed[key].add(ordinal)
    assert {k:len(v) for k,v in started.items()} == expected
    assert {k:len(v) for k,v in completed.items()} == expected
    return dict(attempts=expected,completed=expected,unknown={k:0 for k in expected})


def check_training(results, baseline, manipulation, checkpoints, ledgers):
    expected_work = dict(optimizer_steps=128,response_forwards=256,response_backwards=256)
    schedules = {}
    for arm in ('K','SHUFFLED'):
        result = results[arm]
        expected = dict(status='completed',method=arm,dataset='gsm8k',event_id='event_005',
                        start_step=512,end_step=640,optimizer_steps=128,response_presentations=256,passes=2)
        assert all(result[k] == value for k,value in expected.items()), arm
        assert result['successor_work'] == dict(ordinary_presentations=256,**expected_work)
        assert result['features'] == baseline['features']
        assert result['weights_identity'] == checkpoints[arm]['weights_identity']
        rows = result['records']; assert len(rows) == 256
        assert len(set(r['record_id'] for r in rows[:128])) == 128
        assert {r['record_id'] for r in rows[:128]} == {r['record_id'] for r in rows[128:]}
        for i,row in enumerate(rows):
            assert row['pass'] == i//128
            assert row['optimizer_stepped'] is bool(i%2)
            assert row['optimizer_step_after'] == 512+(i+1)//2
            assert row['learning_rate'] == (0.0003 if i%2 else None)
            observation = row['response_observation']
            assert observation['record_id'] == row['record_id'] and observation['pass_index'] == row['pass']
            assert observation['presentation_ordinal'] == i
            assert observation['sampled_anchors'] == row['sampled_anchors']
        keys = ('record_id','pass','optimizer_step_after','optimizer_stepped','learning_rate','sampled_anchors')
        schedules[arm] = [{key:r[key] for key in keys} for r in rows]
        assert schedules[arm] == [{key:r[key] for key in keys} for r in baseline['records']]
        work = check_ledger(ledgers[arm],expected_work)
        assert all(result['work_ledger'][phase] == values for phase,values in work.items())
    assert schedules['K'] == schedules['SHUFFLED']
    rows = [check_manipulation(r['response_observation']) for r in results['SHUFFLED']['records']]
    assert manipulation['status'] == 'passed' and manipulation['presentations'] == 256
    assert manipulation['rows'] == rows
    totals = {name:sum(row[name] for row in rows) for name in MANIP_TOTALS}
    assert all(manipulation[name] == value for name,value in totals.items())
    changed = results['SHUFFLED']['weights_identity'] != checkpoints['TRUE']['weights_identity']
    assert changed and manipulation['final_export_changed'] is True
    assert totals['flipped_bits'] > 0 and manipulation['labels_changed'] is True
    assert manipulation['all_invariants_passed'] is True and manipulation['actuation'] == 'active'
    return dict(status='passed',main_optimizer_steps=256,main_presentations=512,
                main_response_forwards=512,main_response_backwards=512,
                TRUE_new_main_steps=0,per_new_arm=expected_work,manipulation_totals=totals,
                labels_changed=True,final_export_changed=True,
                benchmark_scope='Descriptive expectation under uniform permutation; realized map is one fixed hash permutation.',
                tensor_scope='Compact recorded native FP32 masses checked within tolerance; no tensor/gradient replay.')


def compare_summary(result, reported):
    expected = reported['endpoint']['summary']
    checks = 0
    for arm, values in result['method_totals'].items():
        for key,value in values.items():
            assert value == expected['method_totals'][arm][key], (arm,key)
            checks += 1
    assert result['repeat_totals'] == expected['repeat_totals']
    for name,effect in result['comparisons'].items():
        other = expected['comparisons'][name]
        for key,metric in effect['metrics'].items():
            target = other['call_reduction' if key == 'verification_calls' else key+'_reduction']
            for field,value in metric.items():
                assert value == target[field], (name,key,field,value,target[field])
                checks += 1
        for ours,theirs in (('calls_saved','gross_calls_saved'),('calls_added','gross_calls_added'),
                            ('call_wins','request_wins'),('call_losses','request_regressions'),('call_ties','request_ties')):
            assert effect[ours] == other[theirs], (name,ours)
            checks += 1
        for key,theirs in (('wins','wins'),('losses','regressions'),('ties','ties')):
            assert effect['request_latency_directions'][key] == other['request_latency_'+theirs]
            checks += 1
        for key,theirs in (('wins','wins'),('losses','losses'),('ties','ties'),
                          ('gained','gross_prefix_tokens_gained'),('lost','gross_prefix_tokens_lost'),
                          ('seed_eos_requests','seed_eos_requests')):
            assert effect['first_round'][key] == other['first_native_round'][theirs]
            checks += 1
    for key,value in result['classification'].items():
        if key == 'latency':
            for name,claims in value.items():
                assert all(v == reported['decision'][key][name][k] for k,v in claims.items())
        else: assert value == reported['decision'][key], key
    return dict(status='exact_match',numerical_fields=checks,repeat_totals='exact',classification='exact')


def numeric_resources(value, prefix=''):
    """Retain clocks at their original levels; nested clocks are never added."""
    rows = {}
    if isinstance(value, dict):
        for key,item in value.items():
            path = prefix+'.'+key if prefix else key
            if type(item) in (int,float) and (key.endswith('_s') or any(
                    word in key for word in ('steps','calls','generations','presentations','bytes','memory','requests','blocks'))):
                rows[path] = item
            elif isinstance(item,dict): rows.update(numeric_resources(item,path))
    return rows


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Local retained execution root')
    parser.add_argument('--baseline',type=Path,required=True,help='Authenticated PR37 native TRUE RESULT.json')
    parser.add_argument('--failed-status',type=Path,action='append',default=[],help='Separate prior failed-launch STATUS; repeat if needed')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior audit: '+str(args.output))
    inputs = {}
    def read(path,lines=False,text=False):
        path = Path(path)
        if not path.is_absolute(): path = args.root/path
        data = path.read_bytes()
        inputs[str(path.resolve())] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        decoded = data.decode()
        return decoded if text else [json.loads(line) for line in decoded.splitlines()] if lines else json.loads(decoded)
    output = dict(status='running',command=sys.argv,utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,
        verification_status='ANALYZED',model_work=0,remote_operations=0,
        independence='Separately implemented arithmetic, with same-team implementation-author overlap disclosed.',
        scope='Raw scalar rows, compact training/manipulation records and work counts; no native model/token/tensor replay.',
        bootstrap=dict(draws=10000,generator='PCG64',seed=20261111,method='linear',percentiles=[2.5,97.5],
                       unit='whole paired request jointly across all arms',intervals_adjusted=False))
    try:
        plan, panel = read('sealed/PLAN.json'), read('sealed/PANEL.json')
        n = integer(plan['panel_size']); assert 8 <= n <= 256 and n%8 == 0
        requests = panel['requests']; assert len(requests) == n
        analysis = plan['analysis']
        contract = dict(resamples=10000,seed=20261111,generator='PCG64',percentile_method='linear',
                        primary=[['TRUE','SHUFFLED']],controls=[['TRUE','K'],['SHUFFLED','K']],
                        intervals_adjusted=False,uncertainty_unit='paired_request',
                        repeats_reduction='within_request_mean',effect='percent_reduction_of_paired_totals')
        assert all(analysis[k] == v for k,v in contract.items())
        assert plan['true_reuse']['enabled'] is True
        true_auth = read('sealed/TRUE_AUTHENTICATION.json')
        assert true_auth['status'] == 'passed' and true_auth['binding'] == plan['binding']
        assert true_auth['execution_commit'] == plan['execution_source']['commit']
        assert all(true_auth[k] is True for k in ('origin','packets','schedule','optimizer','runtime','step640','export'))
        assert plan['budget']['optimizer_steps'] == 256 and plan['budget']['training_presentations'] == 512
        planned = plan['evaluation_schedule']
        assert [(r['panel_ordinal'],r['method'],r['repeat'],r['block_ordinal'],r['position']) for r in planned] == schedule(n)
        assert all(r['request_id'] == requests[r['panel_ordinal']]['request_id'] for r in planned)
        raw = read('evaluation/MEASUREMENTS.jsonl',lines=True)
        result = compute([normalize(row) for row in raw],n,requests)
        output['results'] = result
        output['raw_to_report_timing_comparison'] = compare_report_rows(raw,read('final_report/TIMING_REPEATS.csv',text=True))
        output['published_table_comparison'] = compare_summary(result,read('final_report/SUMMARY.json'))
        audit = read('evaluation/AUDIT.json')
        assert audit['status'] == 'passed' and not audit.get('errors')
        if 'rows' in audit: assert audit['rows'] == raw
        output['native_integrity'] = 'Supplied native raw audit passed; scalar auditor does not independently replay native tokens.'
        activations = read('evaluation/ACTIVATIONS.jsonl',lines=True)
        assert [r['method'] for r in activations] == publication_methods(schedule(n))
        output['publication_receipt_count'] = len(activations)
        checkpoints = read('CHECKPOINTS.json')
        assert checkpoints['TRUE']['weights_identity']['sha256'] == '9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795'
        for arm in ARMS:
            assert all(p[arm]['checkpoint_id'] == checkpoints[arm]['checkpoint_path'] for p in result['paired_requests'])
        training = {a:read('campaign_001/gsm8k/event_005/'+a+'/RESULT.json') for a in ('K','SHUFFLED')}
        ledgers = {a:read('campaign_001/gsm8k/event_005/'+a+'/WORK.jsonl',lines=True) for a in ('K','SHUFFLED')}
        baseline = read(args.baseline.resolve())
        assert inputs[str(args.baseline.resolve())] == plan['true_reuse']['result_identity']
        assert baseline['weights_identity'] == checkpoints['TRUE']['weights_identity']
        output['training_audit'] = check_training(training,baseline,read('manipulation/RESULT.json'),checkpoints,ledgers)
        startup_work = dict(response_forwards=2,response_backwards=2,optimizer_steps=0)
        for arm in ('K','SHUFFLED'):
            check_ledger(read('campaign_001/gsm8k/event_005/'+arm+'/startup/WORK.jsonl',lines=True),startup_work)
        output['training_audit']['main_discarded_startup'] = dict(response_forwards=4,response_backwards=4,optimizer_steps=0)
        admission = read('qualification/ADMISSION.json')
        assert admission['status'] == 'passed' and admission['optimizer_steps'] == 3
        assert admission['response_forwards'] == admission['response_backwards'] == 12
        assert admission['binding'] == plan['binding'] and admission['execution_commit'] == plan['execution_source']['commit']
        assert admission['next_anchor_draws'] == 3 and admission['qualification_states_not_used_for_training'] is True
        for case in ('TRUE','IDENTITY','SHUFFLED'):
            check_ledger(read('qualification/'+case+'/WORK.jsonl',lines=True),
                         dict(response_forwards=2,response_backwards=2,optimizer_steps=1,next_anchor_draws=1))
            check_ledger(read('qualification/'+case+'/startup/WORK.jsonl',lines=True),startup_work)
        output['qualification'] = dict(optimizer_steps=3,response_forwards=12,response_backwards=12,
            scope='Separate disposable work including startup; native identity parity supported by supplied qualification receipt.')
        resources = read('RESOURCES.json'); run = read('STATUS.json')
        assert run['status'] == 'completed' and read('POSTFLIGHT.json')['status'] == 'passed'
        assert run['execution_commit'] == plan['execution_source']['commit']
        training_status = {a:read('campaign_001/gsm8k/event_005/'+a+'/STATUS.json') for a in ('K','SHUFFLED')}
        assert all(v['status'] == 'completed' for v in training_status.values())
        failed_walls = []
        for path in args.failed_status:
            failed = read(path.resolve()); assert failed['status'] == 'failed'
            assert math.isfinite(failed['wall_s']) and failed['wall_s'] >= 0
            failed_walls.append(dict(path=str(path.resolve()),wall_s=failed['wall_s']))
        output['resources'] = dict(current_run_wall_s=run['wall_s'],timers=numeric_resources(resources),
            training={a:numeric_resources(v) for a,v in training.items()},
            training_worker={a:numeric_resources(v) for a,v in training_status.items()},
            supplied_prior_failed_launches=failed_walls,
            wall_including_supplied_failed_launches_s=run['wall_s']+sum(v['wall_s'] for v in failed_walls),
            prior_failure_inventory='Only explicitly supplied failed-launch receipts; absence is not proof of zero failures.',
            accounting='Nested timers retained separately; only distinct run walls are added, excluding staging/analysis gaps.')
        output.update(status='passed',inference_permitted=True)
    except (AssertionError,KeyError,ValueError,TypeError,OSError,ZeroDivisionError,IndexError) as error:
        output.update(status='failed',error=repr(error),inference_permitted=False)
        if 'results' in output:
            output['results']['classification'] = dict(outcome='invalid_or_incomplete',automatic_followup_authorized=False)
    output['elapsed_s'] = time.perf_counter()-started
    output['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:output[k] for k in ('status','elapsed_s')},indent=2))
    raise SystemExit(0 if output['status'] == 'passed' else 1)


if __name__ == '__main__': main()
