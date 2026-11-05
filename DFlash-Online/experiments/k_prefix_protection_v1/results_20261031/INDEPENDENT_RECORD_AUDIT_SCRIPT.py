#!/usr/bin/env python3
"""Independent compact-record audit. Standard Python/NumPy only; no model execution.

Usage: python final_record_audit.py RUN_ROOT --output AUDIT_DIRECTORY
Original O schedules: RUN_ROOT/audit_inputs/original_schedules/SEED_EVENT.json.
No experiment statistics/report helper is imported. This is a record analysis,
not independent GPU execution or decoding of remotely retained model tensors.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np

SEEDS = (17, 29)
METHODS = ('F', 'K', 'K_HALF', 'KP')
ARMS = ('K_HALF', 'KP')
COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
TIME_FIELDS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COMPARISONS = (('KP', 'K'), ('KP', 'F'), ('KP', 'K_HALF'),
               ('K_HALF', 'K'), ('K_HALF', 'F'), ('K', 'F'))


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}
        self.checks = 0
        self.raw_records = 0
        self.trace_blocks = 0
        self.proposal_blocks = 0

    def require(self, value, description):
        self.checks += 1
        if not value:
            raise ValueError(description)

    def text(self, path):
        path = Path(path)
        payload = path.read_bytes()
        self.inputs[str(path)] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        return payload.decode()

    def read(self, path):
        return json.loads(self.text(path))

    def rows(self, path):
        return [json.loads(line) for line in self.text(path).splitlines() if line]

    def number_equal(self, actual, expected, label, *, atol=1e-10, rtol=1e-12):
        if expected is None or actual is None:
            self.require(actual is expected, label + ': undefined mismatch')
            return
        self.require(math.isfinite(actual) and math.isfinite(expected)
                     and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), label)


def order(ordinal, repeat):
    methods = list(METHODS if (ordinal // 4) % 2 == 0 else ('F', 'KP', 'K_HALF', 'K'))
    rotation = (ordinal + repeat) % 4
    return methods[rotation:] + methods[:rotation]


def schedule(requests, seed):
    return [dict(seed=seed, event=4, request_id=request['request_id'], panel_ordinal=i,
                 repeat=repeat, method=method, position=position)
            for i, request in enumerate(requests) for repeat in range(3)
            for position, method in enumerate(order(i, repeat))]


def native_record(audit, record, request, checkpoint, label, ordinal, canonical):
    meta = record['metadata']
    output = record['response_ids']
    trace = meta['trace']
    audit.require(record['record_id'] == label and record['request_index'] == ordinal
                  and record['session_id'] == 'k_prefix_protection_v1_evaluation', label + ': request identity')
    audit.require(record['checkpoint_id'] == checkpoint and record['purpose'] == 'evaluation', label + ': checkpoint/purpose')
    audit.require(record['prompt_ids'] == request['prompt_ids'] and record['stop_token_ids'] == request['stop_token_ids']
                  and record['max_new_tokens'] == request['max_new_tokens'] == 2048, label + ': input/settings')
    audit.require(output == canonical['response_ids'] and record['finish_reason'] == canonical['finish_reason'], label + ': output reference')
    audit.require(1 <= len(output) <= 2048 and all(type(token) is int and token >= 0 for token in output), label + ': output IDs')
    stops = set(request['stop_token_ids'])
    audit.require(not any(token in stops for token in output[:-1]), label + ': continued after EOS')
    finish = 'eos' if output[-1] in stops else 'length'
    audit.require(record['finish_reason'] == finish and (finish == 'eos' or len(output) == 2048), label + ': termination')
    capture = meta['capture']
    audit.require(meta['runtime'] == 'pinned_SGLang_FA3' and meta['block_size'] == 16 and meta['context_limit'] == 4096
                  and meta['cache_policy'] == 'idle_full_flush_before_every_request_and_publication', label + ': runtime')
    audit.require(meta['feedback_capture_in_timed_request'] is False and capture['collect_features'] is False
                  and capture['feature_bytes'] == 0 and capture['native_capture_validated'] is True, label + ': capture')
    if 'blocks' in capture:
        audit.require(capture['blocks'] == trace, label + ': native/wrapper trace')
    offset, accepted = 0, 0
    for number, block in enumerate(trace):
        context = label + ': block' + str(number)
        progress = block['committed_progress']
        audit.require(type(progress) is int and 1 <= progress <= 16 and offset < len(output), context + ': progress')
        audit.require(block['output_start'] == offset and block['anchor_token'] == output[offset], context + ': anchor')
        committed = output[offset + 1:offset + 1 + progress]
        audit.require(block['committed_tokens'] == committed and len(committed) == progress, context + ': committed tokens')
        count = block['accepted_draft_tokens']
        audit.require(type(count) is int and 0 <= count <= progress, context + ': acceptance bound')
        # Count full native proposal evidence separately and require coverage
        # of every retained block at the complete-serving boundary.
        if 'draft_tokens' in block:
            audit.proposal_blocks += 1
            proposals, target = block['draft_tokens'], block['target_argmax']
            width = block['verify_size']
            audit.require(width == len(target) == len(proposals) + 1 and width <= 16, context + ': width')
            first = next((j for j in range(1, width) if proposals[j - 1] != target[j - 1]), width)
            expected = proposals[:first - 1] + [target[first - 1]]
            audit.require(progress <= first and committed == expected[:progress]
                          and committed == target[:progress], context + ': posterior agreement')
            audit.require(count == min(first - 1, progress), context + ': accepted prefix')
            audit.require(block['positions'] == list(range(len(request['prompt_ids']) + offset,
                          len(request['prompt_ids']) + offset + width)), context + ': absolute positions')
            audit.require(block['native_accepted_draft_tokens'] == progress - 1
                          and block['raw_accepted_draft_tokens_before_stop'] == first - 1
                          and block['stop_clipped'] == (progress < first), context + ': native counters')
        offset += progress
        accepted += count
    audit.require(offset == len(output) - 1 and meta['committed_progress'] == offset, label + ': total progress')
    audit.require(meta['verification_calls'] == len(trace) and meta['accepted_draft_tokens'] == accepted, label + ': call totals')
    result = dict(verification_calls=len(trace), accepted_draft_tokens=accepted,
                  committed_progress=offset, output_tokens=len(output), prompt_tokens=len(request['prompt_ids']),
                  cap_hit=finish == 'length', finish_reason=finish, checkpoint_id=checkpoint,
                  request_latency_s=meta['request_latency_s'], native_generate_s=meta['native_generate_s'],
                  cache_maintenance_s=capture['cache_maintenance_s'])
    for field in TIME_FIELDS:
        value = result[field]
        audit.require(type(value) in (int, float) and math.isfinite(value)
                      and (value >= 0 if field == 'cache_maintenance_s' else value > 0), label + ': ' + field)
    audit.raw_records += 1
    audit.trace_blocks += len(trace)
    return result


def serving(audit, plan):
    requests = plan['evaluation_requests']
    audit.require(len(requests) == 128 and [r['panel_ordinal'] for r in requests] == list(range(128)), 'sealed panel size/order')
    audit.require(len({r['request_id'] for r in requests}) == 128, 'unique panel IDs')
    wanted = [slot for seed in SEEDS for slot in schedule(requests, seed)]
    audit.require(plan['evaluation_schedule'] == wanted, 'sealed full schedule')
    reference = audit.read(audit.root/'campaign_001/EVALUATION_CANONICAL.json')
    audit.require(set(reference) == {r['request_id'] for r in requests}, 'complete global canonical panel')
    paired, repeats = [], []
    for seed in SEEDS:
        folder = audit.root/'campaign_001'/f'seed_{seed}/event_004/evaluation'
        rows = audit.rows(folder/'MEASUREMENTS.jsonl')
        planned = schedule(requests, seed)
        audit.require(len(rows) == 1536, f'{seed}: complete cell')
        audit.require([{k:row.get(k) for k in slot} for row, slot in zip(rows, planned)] == planned,
                      f'{seed}: exact chronological schedule')
        protocol = audit.read(folder/'PROTOCOL.json')
        checkpoints = {'F':plan['draft'], 'K':plan['original_k4'][str(seed)]['export']}
        for arm in ARMS:
            result = audit.read(audit.root/'campaign_001'/f'seed_{seed}'/arm/'events/event_004/RESULT.json')
            checkpoints[arm] = result['export_path']
        audit.require(protocol['checkpoints'] == checkpoints and protocol['schedule'] == planned, f'{seed}: checkpoint/schedule')
        gate = audit.read(folder/'QUALIFICATION.json')
        audit.require(gate['status'] == 'passed' and gate['requests'] == 8 and gate['generations'] == 96
                      and gate['extra_generations'] == 0, f'{seed}: included gate')
        status = audit.read(folder/'STATUS.json')
        for key, expected in dict(status='completed', attempted_generations=1536, returned_generations=1536,
                completed_valid_generations=1536, invalid_generations=0, pending_canonical_generations=0,
                unknown_generation_attempts=0, unattempted_generations=0).items():
            audit.require(status[key] == expected, f'{seed}: status {key}')
        for method in METHODS:
            weights = audit.read(folder/f'WEIGHTS_{method}.json')
            w = weights['native_reply']
            audit.require(not weights['errors'] and w['status'] == 'passed' and w['all_equal'] is True
                          and w['tensors'] == w['source_tensor_count'] == 58 and w['native_tensor_count'] == 43
                          and w['checkpoint_path'] == checkpoints[method], f'{seed}/{method}: weight receipt')
        by_request = defaultdict(lambda: defaultdict(list))
        for row, slot in zip(rows, planned):
            ordinal, repeat, method = slot['panel_ordinal'], slot['repeat'], slot['method']
            request = requests[ordinal]
            label = f'{ordinal:04d}_r{repeat}_{method}'
            audit.require(row['label'] == label and row['validation_status'] == 'valid'
                          and row['valid_measurement'] is True and not row['errors'] and not row['structural_errors'], label + ': validity')
            canonical = reference[request['request_id']]
            audit.require(canonical['source_cell'] == [17,4] and canonical['source_method'] == 'F'
                          and canonical['source_repeat'] == 0 and canonical['F_checkpoint'] == plan['draft'], label + ': reference origin')
            record = audit.read(folder/(label+'.json'))
            values = native_record(audit, record, request, checkpoints[method], label, ordinal, canonical)
            for field, value in values.items():
                audit.require(row[field] == value, label + ': reconstructed ' + field)
            if method == 'F':
                audit.require(all(values[f] == canonical['F_counts'][f] for f in COUNT_FIELDS), label + ': global F counts')
                if seed == 17 and repeat == 0:
                    audit.require(canonical['source_record_path'].endswith('/seed_17/event_004/evaluation/'+label+'.json'), label + ': canonical path')
            by_request[ordinal][method].append(values)
            repeats.append({**slot, **values})
        for i, request in enumerate(requests):
            value = dict(seed=seed, event=4, panel_ordinal=i, request_id=request['request_id'])
            for method in METHODS:
                items = by_request[i][method]
                audit.require(len(items) == 3 and all(tuple(x[k] for k in COUNT_FIELDS) == tuple(items[0][k] for k in COUNT_FIELDS)
                              for x in items), f'{seed}/{i}/{method}: repeat counts')
                value[method] = {k:items[0][k] for k in COUNT_FIELDS}
                value[method].update({k:sum(x[k] for x in items)/3 for k in TIME_FIELDS})
            paired.append(value)
    audit.require(audit.raw_records == 3072 and len(paired) == 256, 'complete serving/reduced totals')
    audit.require(audit.proposal_blocks == audit.trace_blocks, 'all native blocks retain full proposal evidence')
    return paired, repeats


def published_tables(audit, paired, repeats):
    raw_rows = list(csv.DictReader(audit.text(audit.root/'final_report/ALL_REPEATS.csv').splitlines()))
    paired_rows = list(csv.DictReader(audit.text(audit.root/'final_report/PAIRED_REQUESTS.csv').splitlines()))
    audit.require(len(raw_rows) == 3072 and len(paired_rows) == 256, 'published CSV row totals')
    for stored, reconstructed in zip(raw_rows, repeats):
        for field in ('seed','event','panel_ordinal','repeat','position',*COUNT_FIELDS,*TIME_FIELDS):
            audit.number_equal(float(stored[field]), reconstructed[field], 'all-repeat CSV '+field)
        for field in ('request_id','method','finish_reason','checkpoint_id'):
            audit.require(stored[field] == str(reconstructed[field]), 'all-repeat CSV '+field)
        audit.require(stored['valid_measurement'] == 'True' and stored['errors'] == '[]', 'all-repeat CSV validity')
    for stored, reconstructed in zip(paired_rows, paired):
        audit.require(stored['request_id'] == reconstructed['request_id']
                      and int(stored['seed']) == reconstructed['seed']
                      and int(stored['panel_ordinal']) == reconstructed['panel_ordinal'], 'paired CSV identity')
        for method in METHODS:
            for field in (*COUNT_FIELDS,*TIME_FIELDS):
                audit.number_equal(float(stored[method+'_'+field]), reconstructed[method][field], 'paired CSV '+method+'/'+field)


def evaluation_costs(audit, plan, repeats):
    """Keep switching, equality checks and model loading outside serving times."""
    campaign = audit.read(audit.root/'campaign_001/COSTS.json')
    published = audit.read(audit.root/'final_report/EVALUATION_COSTS.json')
    audit.require(len(published) == 2, 'two published evaluation-cost cells')
    totals = []
    for seed in SEEDS:
        folder = audit.root/'campaign_001'/f'seed_{seed}/event_004/evaluation'
        status = audit.read(folder/'STATUS.json')
        expected_switches, previous = [], None
        for slot in schedule(plan['evaluation_requests'], seed):
            if slot['method'] != previous:
                expected_switches.append({key:slot[key] for key in ('panel_ordinal','repeat','method')})
                previous = slot['method']
        activations = audit.rows(folder/'ACTIVATIONS.jsonl')
        audit.require([{key:row[key] for key in ('panel_ordinal','repeat','method')} for row in activations]
                      == expected_switches, f'{seed}: exact checkpoint-switch schedule')
        checkpoints = audit.read(folder/'PROTOCOL.json')['checkpoints']
        for row in activations:
            native = row['native_status']; publication = native['publication']
            checkpoint = checkpoints[row['method']]
            audit.require(not row['errors'] and row['checkpoint_structure_valid']
                          and row['checkpoint_id'] == checkpoint and native['status'] == 'passed'
                          and native['poisoned'] is None and native['capture_pending'] is False
                          and native['cache_tokens'] == publication['cache_after_tokens'] == 0
                          and native['attention_backend'] == 'fa3' and native['block_size'] == 16
                          and publication['status'] == 'passed' and publication['checkpoint_path'] == checkpoint
                          and all(publication[key] is True for key in ('helper_matches_current_layers',
                              'helper_norm_matches_current_layers','fused_enabled')), f'{seed}: valid checkpoint activation')
            audit.require(math.isfinite(row['activation_s']) and row['activation_s'] >= 0, f'{seed}: activation time')
        switching = sum(row['activation_s'] for row in activations)
        checks = sum(audit.read(folder/f'WEIGHTS_{method}.json')['validation_s'] for method in METHODS)
        audit.number_equal(switching, status['activation_s'], f'{seed}: summed activation cost')
        audit.number_equal(checks, status['weight_validation_s'], f'{seed}: summed weight-check cost')
        audit.require(campaign[f'{seed}/evaluation'] == status, f'{seed}: campaign evaluation cost receipt')
        audit.require(next(row for row in published if row['seed'] == seed) == {'seed':seed,'event':4,**status},
                      f'{seed}: published evaluation cost receipt')
        loading = campaign[f'{seed}/native_loading_s']
        audit.require(all(math.isfinite(value) and value >= 0 for value in
                      (checks, loading, status['evaluation_wall_s'])), f'{seed}: finite overhead costs')
        totals.append(dict(seed=seed,event=4,activation_count=len(activations),activation_s=switching,
                           weight_validation_s=checks,native_loading_s=loading,
                           evaluation_wall_s=status['evaluation_wall_s'],
                           serving_repeat_sums={field:sum(row[field] for row in repeats if row['seed']==seed)
                                                for field in TIME_FIELDS}))
    return totals


def percentile(values):
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [.025,.975], method='linear').tolist() if len(finite) else None


def numerical_analysis(audit, paired, published):
    arrays = {method:{field:np.array([[r[method][field] for r in paired if r['seed'] == seed]
                                    for seed in SEEDS], dtype=np.float64)
                      for field in (*COUNT_FIELDS,*TIME_FIELDS)} for method in METHODS}
    draws = np.random.Generator(np.random.PCG64(20261102)).integers(0,128,(10000,128))
    analysis = {}
    for scope, which in [('combined',slice(None)),('17',slice(0,1)),('29',slice(1,2))]:
        expected = published['combined'] if scope == 'combined' else published['per_seed'][scope]
        output = {}
        for candidate, baseline in COMPARISONS:
            name = candidate+'_vs_'+baseline
            metrics = {}
            for field in ('verification_calls',*TIME_FIELDS):
                c = arrays[candidate][field][which].sum(axis=0)
                b = arrays[baseline][field][which].sum(axis=0)
                point = 100*(b.sum()-c.sum())/b.sum() if b.sum() else None
                boot_b = b[draws].sum(axis=1)
                boot_c = c[draws].sum(axis=1)
                boot = np.full(10000,np.nan)
                np.divide(100*(boot_b-boot_c),boot_b,out=boot,where=boot_b>0)
                ci = percentile(boot)
                key = 'call_reduction' if field == 'verification_calls' else field+'_reduction'
                other = expected[name][key]
                audit.number_equal(point,other['point_pct'],scope+'/'+name+'/'+field+'/point')
                if ci is None:
                    audit.require(other['ci95_pct'] is None,'undefined interval')
                else:
                    for i in range(2): audit.number_equal(ci[i],other['ci95_pct'][i],scope+'/'+name+'/'+field+'/interval')
                metrics[field] = dict(point_pct=point,ci95_pct=ci,candidate_sum=float(c.sum()),reference_sum=float(b.sum()))
            delta = arrays[baseline]['verification_calls'][which] - arrays[candidate]['verification_calls'][which]
            counts = dict(request_seed_wins=int((delta>0).sum()),request_seed_regressions=int((delta<0).sum()),
                          request_seed_ties=int((delta==0).sum()),gross_calls_saved=int(np.maximum(delta,0).sum()),
                          gross_calls_added=int(np.maximum(-delta,0).sum()))
            audit.require(all(expected[name][k] == v for k,v in counts.items()),scope+'/'+name+'/win and gross counts')
            output[name] = {**metrics,**counts}
        for candidate in ARMS:
            f = arrays['F']['verification_calls'][which]
            k = arrays['K']['verification_calls'][which]
            c = arrays[candidate]['verification_calls'][which]
            kg,kh,cg,ch = np.maximum(f-k,0),np.maximum(k-f,0),np.maximum(f-c,0),np.maximum(c-f,0)
            retained = np.minimum(kg,cg)
            parts = dict(K_gross_gain=kg,K_gross_harm=kh,candidate_gross_gain=cg,candidate_gross_harm=ch,
                         retained_gain=retained,lost_gain=kg-retained,added_gain=cg-retained,
                         prevented_harm=np.maximum(kh-ch,0),added_harm=np.maximum(ch-kh,0))
            audit.require(np.array_equal(k-c,parts['added_gain']-parts['lost_gain']+parts['prevented_harm']-parts['added_harm']),
                          scope+'/'+candidate+'/gain-harm identity')
            expected_parts = published['gain_harm'][scope][candidate]
            values = {}
            for name, part in parts.items():
                point = int(part.sum()); samples = part.sum(axis=0)[draws].sum(axis=1)
                ci = percentile(samples)
                audit.number_equal(point,expected_parts[name]['point'],scope+'/'+candidate+'/'+name)
                for i in range(2): audit.number_equal(ci[i],expected_parts[name]['ci95'][i],scope+'/'+candidate+'/'+name+'/interval')
                values[name] = dict(point=point,ci95=ci)
            den = kg.sum(axis=0)[draws].sum(axis=1)
            fractions = np.full(10000,np.nan)
            np.divide(retained.sum(axis=0)[draws].sum(axis=1),den,out=fractions,where=den>0)
            fraction = float(retained.sum()/kg.sum()) if kg.sum() else None
            audit.number_equal(fraction,expected_parts['retained_gain_fraction']['point'],scope+'/'+candidate+'/retained fraction')
            ci = percentile(fractions)
            audit.require((ci is None) == (expected_parts['retained_gain_fraction']['ci95'] is None),'retention interval definedness')
            if ci is not None:
                for i in range(2): audit.number_equal(ci[i],expected_parts['retained_gain_fraction']['ci95'][i],'retention interval')
            output[candidate+'_gain_harm'] = dict(parts=values,retained_fraction=fraction,retained_fraction_ci95=ci)
        analysis[scope] = output
    return analysis


def work_ledger(audit, path, expected):
    started = {k:[] for k in expected}; finished = {k:[] for k in expected}
    for row in audit.rows(path):
        kind,ordinal,phase = row['kind'],row['ordinal'],row['phase']
        audit.require(kind in expected and type(ordinal) is int and 1<=ordinal<=expected[kind],str(path)+': work kind/ordinal')
        if phase == 'started':
            audit.require(ordinal == len(started[kind])+1,str(path)+': contiguous start')
            started[kind].append(ordinal)
        else:
            audit.require(phase == 'completed' and ordinal in started[kind] and ordinal not in finished[kind],str(path)+': completion')
            finished[kind].append(ordinal)
    audit.require({k:len(v) for k,v in started.items()} == expected
                  and {k:len(v) for k,v in finished.items()} == expected,str(path)+': complete budget')
    return expected


def training_and_costs(audit, plan, published):
    totals = Counter(); costs = defaultdict(lambda:Counter()); pids = set(); devices = set(); cache_paths = set()
    original_hashes = audit.read(audit.root/'sealed/ORIGINAL_ARTIFACT_AUDIT.json')['files']
    for seed in SEEDS:
        for event in range(1,5):
            original = audit.root/'audit_inputs/original_schedules'/f'{seed}_{event}.json'
            retained = plan['original_schedules'][str(seed)][str(event)]['path']
            ordinary = audit.read(original)
            audit.require(audit.inputs[str(original)]['sha256'] == original_hashes[retained]['sha256'],str(original)+': original schedule bytes')
            for arm in ARMS:
                base = audit.root/'campaign_001'/f'seed_{seed}'
                folder = base/f'resume_{event:03d}_{arm}'
                result = audit.read(base/arm/'events'/f'event_{event:03d}/RESULT.json')
                expected = dict(forwards=256,backwards=256,optimizer_steps=128)
                if arm == 'KP': expected['reference_forwards'] = 256
                totals.update(work_ledger(audit,folder/'WORK.jsonl',expected))
                audit.require(result['status'] == 'completed' and result['start_step'] == (event-1)*128
                              and result['end_step'] == event*128 and result['response_presentations'] == 256
                              and result['optimizer_steps'] == 128 and result['passes'] == 2, str(folder)+': event accounting')
                audit.require(result['features'] == ordinary['features'] and len(result['records']) == 256,str(folder)+': shared immutable pool')
                for i,(row,old) in enumerate(zip(result['records'],ordinary['records'])):
                    audit.require(all(row[k] == old[k] for k in ('record_id','pass','sampled_anchors')),str(folder)+': natural anchor/order'+str(i))
                    obs = row['response_observation']
                    audit.require(obs['record_id'] == row['record_id'] and obs['scope'] == 'one_original_response_forward',str(folder)+': response scope')
                    audit.require(row['optimizer_stepped'] == (i%2==1)
                                  and row['optimizer_step_after'] == (event-1)*128+(i+1)//2,str(folder)+': step progression')
                    audit.require(math.isfinite(obs['loss']),str(folder)+': finite response loss')
                    if arm == 'KP':
                        p=obs['prefix_protection']
                        audit.require(p['finite'] and p['protection_evaluated'] and not p['qualification_forced_active'],str(folder)+': production penalty')
                        audit.number_equal(obs['loss'],p['total_loss'],str(folder)+': total response loss',atol=1e-7,rtol=1e-6)
                        audit.require(p['active_positions'] <= p['protected_positions']
                                      and sum(p['protected_prefix_histogram']) == p['protected_prefix_histogram_denominator'],str(folder)+': exposure')
                worker = audit.read(folder/'WORKER.json'); closed = audit.read(folder/'CLOSED.json')
                audit.require(worker['status']=='completed' and closed['closed'] and not closed['failed']
                              and closed['global_step']==event*128 and closed['presentations']==event*256
                              and closed['accumulation_remainder']==0,str(folder)+': closed boundary')
                startup = worker['startup']; pids.add(startup['pid']);devices.add(startup['device_uuid'])
                audit.require(startup['cuda_initialized_on_entry'] is False and startup['warmup_model_work']==0,str(folder)+': cold startup')
                for item in startup['caches'].values():
                    audit.require(item['initially_empty'],str(folder)+': empty compiler cache')
                    cache_paths.add(item['path'])
                for filename in ('LIVE.json','LIVE_BOUNDARY.json'):
                    audit.require(audit.read(folder/filename)['exact_resume'],str(folder)+': exact restoration receipt')
                endpoint=audit.read(folder/'SAVED_ENDPOINT.json')
                audit.require(endpoint['status']=='passed' and endpoint['export_exact'],str(folder)+': decoded state/export receipt')
                c=audit.read(folder/'COSTS.json')
                audit.number_equal(sum(r['training_s'] for r in result['records']),c['training_s'],str(folder)+': training time')
                key=f'{seed}/{arm}'
                for field in ('training_s','event_wall_s','loading_initialization_resume_validation_s','frozen_reference_setup_s',
                              'checkpoint_save_s','export_s','state_validation_hash_s'):
                    audit.require(math.isfinite(c[field]) and c[field]>=0,str(folder)+': cost '+field)
                    costs[key][field]+=c[field]
                for field in ('peak_allocated_bytes','peak_reserved_bytes','loading_peak_allocated_bytes','loading_peak_reserved_bytes'):
                    costs[key][field]=max(costs[key][field],c[field])
                if arm=='KP':
                    audit.require(c['objective_work']['work']['reference_response_forwards']==256,str(folder)+': reference accounting')
    audit.require(dict(totals)==dict(forwards=4096,backwards=4096,optimizer_steps=2048,reference_forwards=2048),'production work totals')
    audit.require(len(pids)==16 and len(devices)==1 and len(cache_paths)==96,'fresh processes/device/caches')
    for trajectory, values in costs.items():
        reported=published['costs']['current'][trajectory]
        for field,value in values.items(): audit.number_equal(value,reported[field],trajectory+'/'+field)
    qualified=Counter()
    for case,references in (('Q1',0),('Q2',2),('Q3',2),('Q4',2)):
        folder=audit.root/'qualification_001'/case
        expected=dict(forwards=2,backwards=2,optimizer_steps=1,reference_forwards=references)
        qualified.update(work_ledger(audit,folder/'WORK.jsonl',expected))
        receipt=audit.read(folder/'CASE.json')
        audit.require(receipt['status']=='completed' and not receipt['frozen_changes'] and receipt['serialized_restored_exact']
                      and receipt['global_step']==513 and receipt['presentations']==1026,case+': disposable state')
    audit.require(dict(qualified)==dict(forwards=8,backwards=8,optimizer_steps=4,reference_forwards=6),'qualification budget')
    return dict(production=dict(totals),qualification=dict(qualified),costs={k:dict(v) for k,v in costs.items()},
                scope='Parent and nested timers remain separate; original K times are historical, not fresh matched costs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    audit=Audit(args.root)
    result=dict(status='running',scope='Independent compact-record analysis, not GPU reexecution or tensor-state replay',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numerical_comparison_atol=1e-10,
                numerical_comparison_rtol=1e-12,errors=[])
    try:
        plan=audit.read(audit.root/'sealed/PLAN.json')
        result['execution_commit']=plan['execution_source']['commit']
        published=audit.read(audit.root/'final_report/SUMMARY.json')
        audit.require(published['status']=='valid' and not published['errors'],'published report must be valid')
        paired,repeats=serving(audit,plan)
        published_tables(audit,paired,repeats)
        result['evaluation_costs']=evaluation_costs(audit,plan,repeats)
        result['analysis']=numerical_analysis(audit,paired,published['analysis'])
        result['training']=training_and_costs(audit,plan,published)
        result['raw_measurements']=audit.raw_records
        result['trace_blocks']=audit.trace_blocks
        result['full_proposal_blocks']=audit.proposal_blocks
        result['paired_seed_requests']=len(paired)
        result['bootstrap']=dict(request_identities=128,both_seeds_and_all_methods_together=True,
                                 draws=10000,generator='PCG64',seed=20261102,quantile_method='linear')
        rows=[{k:v for k,v in row.items() if k not in METHODS} |
              {method+'_'+field:value for method in METHODS for field,value in row[method].items()} for row in paired]
        with (args.output/'INDEPENDENT_PAIRED_REQUESTS.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        result['status']='passed'
    except Exception as error:
        result['status']='failed';result['errors']=[repr(error)];result['traceback']=traceback.format_exc()
    finally:
        result['checks']=audit.checks
        result['raw_records_checked']=audit.raw_records
        result['evidence_limits']=['Native GPU execution was not repeated.','Model/Adam tensors were not decoded locally; retained exact restoration/export receipts were inspected.',
                                  'No model outputs, requests, endpoints or observations were selected or replaced by this audit.']
        (args.output/'AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (args.output/'INPUT_IDENTITIES.json').write_text(json.dumps(audit.inputs,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','raw_records_checked','errors')}))
    raise SystemExit(0 if result['status']=='passed' else 2)


if __name__=='__main__':
    main()
