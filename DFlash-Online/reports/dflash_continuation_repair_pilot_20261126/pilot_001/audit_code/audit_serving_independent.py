"""Independent retained-evidence audit of the fixed BASE/TAIL serving comparison.

Uses only Python/NumPy. No production validators, reducers or bootstrap helpers
are imported. Shared context and objective authorship are disclosed in output.
This script performs no model, optimizer, generation or network operation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np

ARMS = ('BASE', 'TAIL')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNTS = ('output_tokens', 'verification_calls', 'accepted_draft_tokens',
          'native_accepted_draft_tokens', 'committed_progress')
SEED, DRAWS, REQUESTS = 20261126, 10000, 128


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.files = {}

    def require(self, condition, description):
        self.checks += 1
        if not condition:
            raise ValueError(description)

    def equal(self, actual, expected, label):
        """Exact discrete comparisons; bounded FP64 roundoff for arithmetic replay."""
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict), label + ': object required')
            for key, value in expected.items():
                self.require(key in actual, label + ': missing field ' + key)
                self.equal(actual[key], value, label + '/' + key)
        elif isinstance(expected, list):
            self.require(isinstance(actual, list) and len(actual) == len(expected), label + ': list size')
            for index, (value, reference) in enumerate(zip(actual, expected)):
                self.equal(value, reference, label + '/' + str(index))
        elif isinstance(expected, float):
            self.require(type(actual) in (int, float) and math.isfinite(actual)
                         and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-11), label + ': numeric mismatch')
        else:
            self.require(actual == expected, label + ': mismatch')

    def read(self, path, *, lines=False):
        path = Path(path)
        data = path.read_bytes()
        self.files[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return [json.loads(row) for row in data.splitlines() if row] if lines else json.loads(data)


def expected_schedule(panel, warmups):
    """Literal three repeats × two groups × alternating two checkpoints."""
    entries, rows = [], []
    for repeat in range(3):
        for group in range(2):
            order = ('BASE', 'TAIL') if (repeat + group) % 2 == 0 else ('TAIL', 'BASE')
            selected = panel[64*group:64*(group+1)]
            for position, arm in enumerate(order):
                activation = len(entries)
                entries.append(dict(activation=activation, repeat=repeat, group=group,
                    position=position, method=arm, warmups=2,
                    request_ids=[request['request_id'] for request in selected]))
                for phase, request in [('warmup', q) for q in warmups] + [('scored', q) for q in selected]:
                    rows.append(dict(activation=activation, repeat=repeat, group=group,
                        method=arm, phase=phase, request_ordinal=request['request_index'],
                        request_id=request['request_id']))
    return entries, rows


def check_record(audit, record, request, reference, slot, checkpoint, ordinal):
    """Reconstruct tokens and native acceptance directly from saved verifier blocks."""
    label = f'record {ordinal}'
    expected_label = f"a{slot['activation']:03d}_{slot['phase']}_{slot['request_ordinal']:04d}_{slot['method']}"
    audit.equal({k: record[k] for k in ('record_id', 'session_id', 'request_index', 'checkpoint_id', 'purpose')},
                dict(record_id=expected_label, session_id='opencode_final58',
                     request_index=request['request_index'], checkpoint_id=checkpoint, purpose='evaluation'), label)
    for key in ('prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens'):
        audit.equal(record[key], reference[key], label + '/canonical/' + key)
    response, prompt = record['response_ids'], record['prompt_ids']
    audit.require(0 < len(response) <= 2048 and len(prompt) + 2048 <= 4096, label + ': lengths')
    audit.require(all(type(token) is int and token >= 0 for token in response + prompt), label + ': token types')
    audit.require(not any(token in record['stop_token_ids'] for token in response[:-1]), label + ': premature EOS')
    finish = 'eos' if response[-1] in record['stop_token_ids'] else 'length'
    audit.require(record['finish_reason'] == finish and (finish == 'eos' or len(response) == 2048), label + ': termination')
    meta, capture = record['metadata'], record['metadata']['capture']
    audit.equal(meta, dict(block_size=16, context_limit=4096, runtime='pinned_SGLang_FA3',
        feedback_capture_in_timed_request=False,
        cache_policy='idle_full_flush_before_every_request_and_publication'), label + '/runtime')
    audit.equal(capture, dict(collect_features=False, feature_bytes=0, native_capture_validated=True,
                              prompt_ids=prompt, response_ids=response, blocks=meta['trace']), label + '/capture')
    offset = accepted_total = native_total = 0
    for index, block in enumerate(meta['trace']):
        name = label + '/block ' + str(index)
        width, proposal, target = block['verify_size'], block['draft_tokens'], block['target_argmax']
        progress = block['committed_progress']
        audit.require(type(width) is int and width == len(proposal)+1 == len(target)
                      and 1 <= width <= 16, name + ': width')
        audit.require(type(progress) is int and 1 <= progress <= width, name + ': progress')
        audit.require(all(type(token) is int and token >= 0 for token in proposal + target), name + ': tokens')
        accepted = 0
        while accepted < len(proposal) and proposal[accepted] == target[accepted]:
            accepted += 1
        audit.require(progress <= accepted+1, name + ': progress beyond rejection')
        audit.equal(block, dict(output_start=offset, anchor_token=response[offset],
            positions=list(range(len(prompt)+offset, len(prompt)+offset+width)),
            committed_tokens=response[offset+1:offset+1+progress],
            accepted_draft_tokens=min(accepted, progress), native_accepted_draft_tokens=progress-1,
            raw_accepted_draft_tokens_before_stop=accepted, stop_clipped=progress < accepted+1), name)
        audit.equal(block['committed_tokens'], (proposal[:accepted]+[target[accepted]])[:progress], name + '/decision')
        audit.equal(block['committed_tokens'], target[:progress], name + '/posterior')
        offset += progress
        accepted_total += min(accepted, progress)
        native_total += progress-1
    audit.require(offset == len(response)-1, label + ': reconstruction length')
    counts = dict(output_tokens=len(response), verification_calls=len(meta['trace']),
                  accepted_draft_tokens=accepted_total, native_accepted_draft_tokens=native_total,
                  committed_progress=offset)
    for key in ('verification_calls', 'accepted_draft_tokens', 'committed_progress'):
        audit.equal(meta[key], counts[key], label + '/' + key)
    audit.require(native_total == offset-len(meta['trace']), label + ': native acceptance')
    times = dict(request_latency_s=meta['request_latency_s'], native_generate_s=meta['native_generate_s'],
                 cache_maintenance_s=capture['cache_maintenance_s'])
    for key, value in times.items():
        audit.require(type(value) in (int, float) and math.isfinite(value)
                      and value >= 0 and (key == 'cache_maintenance_s' or value > 0), label + '/' + key)
    audit.require(times['native_generate_s'] <= times['request_latency_s'], label + ': nested generation time')
    return {**counts, **times, 'finish_reason': finish, 'checkpoint_id': checkpoint,
            'prompt_tokens': len(prompt), 'cap_hit': finish == 'length'}, len(meta['trace'])


def independent_reduction(rows, audit, *, expected_requests=REQUESTS):
    groups = defaultdict(list)
    for row in rows:
        audit.require(row['valid_measurement'] is True and not row['errors'], 'Invalid row cannot enter arithmetic')
        if row['phase'] == 'scored':
            groups[row['request_id'], row['method']].append(row)
    identities = sorted({key[0] for key in groups})
    audit.require(len(identities) == expected_requests and len(groups) == 2*expected_requests, 'Paired panel coverage')
    paired = []
    for request_id in identities:
        item = dict(request_id=request_id)
        for arm in ARMS:
            repeated = sorted(groups[request_id, arm], key=lambda row: row['repeat'])
            audit.require([row['repeat'] for row in repeated] == [0, 1, 2], 'Missing/duplicate repeat')
            for field in (*COUNTS, 'finish_reason', 'checkpoint_id', 'physical_gpu_uuid', 'source_id'):
                audit.require(all(row[field] == repeated[0][field] for row in repeated), 'Repeat identity/count: ' + field)
            item[arm] = {key: repeated[0][key] for key in (*COUNTS, 'finish_reason')}
            item[arm]['repeat_times'] = {key: [row[key] for row in repeated] for key in TIMES}
            item[arm].update({key: math.fsum(row[key] for row in repeated)/3 for key in TIMES})
        audit.require(all(item['BASE'][key] == item['TAIL'][key] for key in ('output_tokens', 'finish_reason')), 'Paired output agreement')
        paired.append(item)
    totals = {}
    vectors = {arm: {key: np.array([row[arm][key] for row in paired], dtype=np.float64)
                     for key in (*TIMES, *COUNTS)} for arm in ARMS}
    for arm in ARMS:
        value = {key: math.fsum(float(x) for x in vector) for key, vector in vectors[arm].items()}
        calls = value['verification_calls']
        value.update({name: value[key]/calls if calls else None for name, key in (
            ('native_accepted_per_call', 'native_accepted_draft_tokens'),
            ('stop_clipped_accepted_per_call', 'accepted_draft_tokens'), ('committed_per_call', 'committed_progress'))})
        value['tokens_per_s'] = value['output_tokens']/value['request_latency_s']
        value['repeat_totals'] = {key: [math.fsum(row[arm]['repeat_times'][key][repeat] for row in paired)
                                      for repeat in range(3)] for key in TIMES}
        value['termination_counts'] = {kind: sum(row[arm]['finish_reason'] == kind for row in paired) for kind in ('eos', 'length')}
        lengths = sorted(int(x) for x in vectors[arm]['output_tokens'])
        value['output_length'] = dict(min=lengths[0], max=lengths[-1], mean=sum(lengths)/len(lengths),
                                     median=float(np.median(lengths)))
        totals[arm] = value
    # Independently draw one common whole-request index matrix for every metric.
    random = np.random.Generator(np.random.PCG64(SEED))
    draws = random.integers(0, expected_requests, size=(DRAWS, expected_requests))
    contrast = dict(candidate='TAIL', baseline='BASE')
    for key in (*TIMES, 'verification_calls'):
        name = 'verification_call_reduction_pct' if key == 'verification_calls' else key + '_reduction_pct'
        base, tail = vectors['BASE'][key], vectors['TAIL'][key]
        if float(base.sum()) == 0:
            contrast[name] = dict(value=None, ci95=None, status='undefined')
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = 100*(1-tail[draws].sum(axis=1)/base[draws].sum(axis=1))
        if not np.isfinite(ratios).all():
            contrast[name] = dict(value=None, ci95=None, status='undefined')
        else:
            contrast[name] = dict(value=100*(1-totals['TAIL'][key]/totals['BASE'][key]),
                ci95=np.quantile(ratios, [0.025, .975], method='linear').tolist(),
                status='prespecified_primary' if key == 'request_latency_s' else 'descriptive_unadjusted')
    for key in ('verification_calls', *TIMES):
        changes = [row['BASE'][key]-row['TAIL'][key] for row in paired]
        outcomes = dict(wins=sum(x > 0 for x in changes), regressions=sum(x < 0 for x in changes), ties=sum(x == 0 for x in changes))
        if key == 'verification_calls':
            contrast.update(outcomes, gross_calls_saved=int(sum(max(x, 0) for x in changes)),
                            gross_calls_added=int(sum(max(-x, 0) for x in changes)), net_calls_saved=int(sum(changes)))
        else:
            contrast[key + '_request_outcomes'] = outcomes
    return dict(requests=expected_requests, paired=paired, totals=totals,
        contrasts={'TAIL_vs_BASE': contrast}, bootstrap=dict(draws=DRAWS, seed=SEED, generator='PCG64',
            input_order=identities, input_order_digest=digest(identities),
            draw_sha256=hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest(),
            shared_indices_for_latency_and_calls=True))


def independent_decision(report, *, distinct, active, complete):
    if not complete:
        return dict(label='BLOCKED', selected='BASE')
    if not distinct:
        return dict(label='NO_DEMONSTRATED_BENEFIT', selected='BASE')
    contrast = report['contrasts']['TAIL_vs_BASE']
    latency, calls = contrast['request_latency_s_reduction_pct'], contrast['verification_call_reduction_pct']
    if latency['value'] is None or latency['ci95'] is None or calls['value'] is None:
        return dict(label='BLOCKED', selected='BASE')
    lower, upper = latency['ci95']
    if lower > 0 and calls['value'] >= 0:
        return dict(label='SUPPORTED_FOR_CONFIRMATION', selected='TAIL')
    if active and latency['value'] > 0 and calls['value'] > 0 and -.5 < lower <= 0 <= upper:
        return dict(label='TENTATIVE', selected='BASE')
    return dict(label='NO_DEMONSTRATED_BENEFIT', selected='BASE')


def check_csv(audit, path, expected):
    payload = Path(path).read_bytes()
    audit.files[str(path)] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    actual = list(csv.DictReader(payload.decode().splitlines()))
    audit.require(len(actual) == len(expected), Path(path).name + ': row count')
    for index, (row, reference) in enumerate(zip(actual, expected)):
        audit.require(set(row) == set(reference), Path(path).name + ': columns')
        for key, value in reference.items():
            cell = row[key]
            if isinstance(value, bool):
                audit.require(cell == str(value), f'{Path(path).name}/{index}/{key}')
            elif isinstance(value, int):
                audit.require(int(cell) == value, f'{Path(path).name}/{index}/{key}')
            elif isinstance(value, float):
                audit.equal(float(cell), value, f'{Path(path).name}/{index}/{key}')
            else:
                audit.require(cell == value, f'{Path(path).name}/{index}/{key}')


def audit_elapsed_extension(root, plan, repaired, receipt, audit):
    """An explicit approved elapsed-only change never resets the scientific clock."""
    original, effective = plan['limits'], repaired['limits']
    audit.require(set(effective) == set(original), 'Same resource-limit fields')
    for key, value in original.items():
        if key != 'elapsed_s':
            audit.require(effective[key] == value, 'Unchanged resource limit/'+key)
    path = root/'sealed/ELAPSED_EXTENSION_AUTHORIZATION.json'
    authorization = audit.read(path)
    audit.equal(repaired['elapsed_extension_authorization_identity'], audit.files[str(path)], 'Approved elapsed receipt identity')
    audit.equal(receipt['elapsed_extension_authorization_identity'], audit.files[str(path)], 'Repair elapsed receipt identity')
    audit.require(receipt['elapsed_extension_authorization'] == repaired['elapsed_extension_authorization'],
                  'Elapsed authorization locator')
    audit.equal(authorization, dict(kind='elapsed_limit_extension', original_plan_binding=plan['binding'],
        original_clock_epoch=receipt['original_clock_epoch'], previous_elapsed_s=14400,
        approved_elapsed_s=43200, occupied_card_s=14400, authorized_scope='remaining readouts and evaluation',
        user_answer='Extend elapsed limit to 12 hours'), 'Explicit elapsed-only approval')
    audit.require(original['elapsed_s'] == 14400 and effective['elapsed_s'] == 43200
                  and original['occupied_card_s'] == effective['occupied_card_s'] == 14400,
                  'Exact approved limits')
    audit.equal(receipt['original_limits'], original, 'Receipt original resource limits')
    audit.equal(receipt['effective_limits'], effective, 'Receipt effective resource limits')
    audit.equal(receipt['elapsed_extension'],dict(
        authorization_path=repaired['elapsed_extension_authorization'],
        authorization_identity=audit.files[str(path)], original_limits=original,effective_limits=effective,
        original_clock_epoch=receipt['original_clock_epoch'],
        prior_preflight_source_identity=authorization['prior_preflight_source_identity'],
        only_elapsed_limit_changed=True), 'Detailed elapsed extension receipt')
    return authorization


def audit_complete_clock(root, clocks, plans, authorization, audit):
    """Retain both failed process intervals and charge the human-waiting interval."""
    first, second = clocks
    final = audit.read(root/'CLOCK.json')
    for name, clock in (('original', first), ('failed_pre_model', second), ('final', final)):
        audit.require(clock['first_gpu_action_epoch'] == first['first_gpu_action_epoch'], name+'/original clock')
        processes = clock['processes']
        audit.require(all(row['end_epoch'] is not None and row['end_epoch'] >= row['start_epoch']
                          for row in processes), name+'/closed process intervals')
        occupied = math.fsum(row['end_epoch']-row['start_epoch'] for row in processes)
        audit.require(math.isclose(clock['occupied_card_s'], occupied, rel_tol=0., abs_tol=1e-6),
                      name+'/all occupied time retained')
    audit.equal(second['processes'][:-1], first['processes'], 'Prior native work unchanged')
    audit.equal(final['processes'][:-2], second['processes'], 'Both failed processes retained')
    audit.require([row['phase'] for row in final['processes']] ==
                  ['qualification','training','readout','readout','readout','evaluation'],
                  'No qualification/training/readout repetition')
    audit.require([row['exit_code'] for row in final['processes']] == [0,0,1,1,0,0], 'Exact process outcomes')
    for left, right in zip(final['processes'], final['processes'][1:]):
        audit.require(right['start_epoch'] >= left['end_epoch'], 'Serial owned processes')
    audit.require(second['processes'][-1]['execution_commit'] == plans[0]['execution_commit'],
                  'Failed preparation source identity')
    audit.require(all(row['execution_commit'] == plans[1]['execution_commit']
                      for row in final['processes'][-2:]), 'Final readout/serving source identity')
    elapsed = final['processes'][-1]['end_epoch'] - final['first_gpu_action_epoch']
    audit.require(math.isclose(final['elapsed_s'], elapsed, rel_tol=0., abs_tol=1e-6), 'Elapsed clock includes waiting')
    audit.require(final['elapsed_s'] <= plans[1]['limits']['elapsed_s']
                  and final['occupied_card_s'] <= plans[1]['limits']['occupied_card_s'], 'Original/effective time ceilings')
    if authorization is not None:
        audit.require(authorization['recorded_epoch'] <= final['processes'][-2]['start_epoch'],
                      'Elapsed approval predates remaining execution')
    return dict(first_gpu_action_epoch=final['first_gpu_action_epoch'], elapsed_s=final['elapsed_s'],
        occupied_card_s=final['occupied_card_s'], failed_pre_model_s=second['processes'][-1]['end_epoch']-second['processes'][-1]['start_epoch'],
        original_elapsed_limit_s=14400, effective_elapsed_limit_s=plans[1]['limits']['elapsed_s'],
        occupied_limit_s=plans[1]['limits']['occupied_card_s'])


def audit_source_provenance(root, protocol, plan, audit):
    """Check each explicit repair against the immutable original scientific plan."""
    original_commit, serving_commit = plan['execution_commit'], protocol['execution_commit']
    directories = [root/name for name in ('resume_001','resume_002') if (root/name).exists()]
    if not directories:
        audit.require(serving_commit == original_commit, 'Source binding without continuation')
        return dict(training_commit=original_commit, serving_commit=serving_commit, readout_attempts=1)
    audit.require(directories[0].name == 'resume_001' and not (root/'resume_003').exists(), 'Declared repair sequence')
    plan_identity = audit.files[str(root/'sealed/PLAN.json')]
    required_roots = set(plan.get('prior_charged_roots', [])) | {str(Path(plan['source_manifest']).parent)}
    clocks, plans, receipts = [], [], []
    authorization = None
    for directory in directories:
        receipt = audit.read(directory/'REPAIR.json')
        repaired = audit.read(directory/'EXECUTION_PLAN.json')
        receipts.append(receipt); plans.append(repaired)
        audit.equal(receipt['original_plan_identity'], plan_identity, 'Original PLAN identity')
        audit.equal(repaired['original_plan_identity'], plan_identity, 'Continuation PLAN identity')
        audit.require(receipt['original_plan'] == repaired['original_plan'], 'Original PLAN locator')
        audit.require(receipt['original_training_commit'] == original_commit
                      and repaired['training_execution_commit'] == original_commit
                      and receipt['orchestration_execution_commit'] == repaired['execution_commit'], 'Phase commit identities')
        audit.require(repaired['source_root'] not in {prior['source_root'] for prior in [plan,*plans[:-1]]},
                      'Separate retained source checkouts')
        audit.equal(receipt['original_source_identity'], plan['source_identity'], 'Original source manifest binding')
        audit.equal(receipt['repair_source_identity'], repaired['source_identity'], 'Repair source manifest binding')
        audit.equal(receipt['binding'], plan['binding'], 'Same scientific input binding')
        audit.require(receipt['additional_adam_applications'] == 0 and receipt['original_settings_unchanged'] is True,
                      'No additional optimizer work or changed settings')
        allowed_changes = {'source_root','source_manifest','source_identity','execution_commit',
            'training_execution_commit','original_plan','original_plan_identity','prior_charged_roots'}
        if repaired.get('limits') != plan.get('limits'):
            audit.require(directory.name == 'resume_002', 'Only final approved continuation may extend elapsed limit')
            authorization = audit_elapsed_extension(root, plan, repaired, receipt, audit)
            allowed_changes |= {'limits','elapsed_extension_authorization','elapsed_extension_authorization_identity'}
            required_roots.add(str(Path(authorization['prior_preflight_source_manifest']).parent))
        audit.require(set(repaired) == set(plan) | allowed_changes, 'No hidden continuation PLAN fields')
        for key, value in plan.items():
            if key not in allowed_changes:
                audit.require(repaired[key] == value, 'Immutable PLAN field/'+key)
        audit.require(required_roots <= set(repaired['prior_charged_roots']), 'All earlier staged sources charged')
        required_roots |= set(repaired['prior_charged_roots']) | {str(Path(repaired['source_manifest']).parent)}
        allowed_files = {'experiments/continuation_repair_pilot/'+name for name in
                         ('launch.py','readout.py','analysis.py','README.md')}
        allowed_files.add('reports/acl2027_method_development/SEQUENCE.md')
        audit.require(bool(receipt['changed_files']) and all(path in allowed_files
            or (path.startswith('tests/test_continuation_repair_') and path.endswith('.py'))
            or path.startswith('reports/dflash_continuation_repair_pilot_20261126/')
            for path in receipt['changed_files']), 'Repair file scope')
        for relative, expected in receipt['preserved_failures'].items():
            path = directory/'prior'/relative
            data = path.read_bytes()
            actual = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            audit.files[str(path)] = actual
            audit.equal(actual, expected, 'Retained failure evidence/'+relative)
        prior_clock = audit.read(directory/'prior/CLOCK.json'); clocks.append(prior_clock)
        audit.require(receipt['original_clock_epoch'] == prior_clock['first_gpu_action_epoch']
                      and receipt['prior_occupied_card_s'] == prior_clock['occupied_card_s'],
                      'Original clock and failed work retained')
        old_adam = audit.read(directory/'prior/ADAM_LEDGER.json')
        current_adam = audit.read(root/'ADAM_LEDGER.json')
        audit.require(old_adam == current_adam and len(old_adam['calls']) == receipt['original_adam_applications'],
                      'No optimizer applications after resume')
    audit.require(plans[-1]['execution_commit'] == serving_commit, 'Final serving source identity')
    result = dict(training_commit=original_commit,serving_commit=serving_commit,readout_attempts=1+len(directories),
        readout_model_forward_attempts=2,
        original_plan_identity=plan_identity,additional_adam_applications=0,
        original_clock_epoch=receipts[0]['original_clock_epoch'],orchestration_repairs=len(directories),
        historical_individual_readout_byte_equality_proven=False)
    if len(directories) == 2:
        rejected = receipts[1]['rejected_preparation']
        audit.equal(rejected, dict(status='passed',original_clock_unchanged=True,
            retained_aggregate_hashes_unchanged=True,existing_readouts=74,new_forward_evidence=0,
            historical_individual_byte_equality_proven=False,rejected_execution_commit=plans[0]['execution_commit']),
            'Preserved pre-model rejection evidence')
        scope = audit.read(root/'sealed/RESUME2_AUTHORIZATION.json')
        audit.equal(scope['scope'],dict(readouts=63,scored_generations=768,warmups=24,
            additional_training_steps=0,repeat_completed_readouts=False),'Explicit remaining-work authorization')
        audit.require(scope['user_authorization']=='approve to resume the remaining readouts and evaluation'
                      and scope['original_clock_preserved'] is True,'Explicit resume decision')
        result['clock'] = audit_complete_clock(root,clocks,plans,authorization,audit)
        audit.require(scope['recorded_epoch'] <= audit.read(root/'CLOCK.json')['processes'][-2]['start_epoch'],
                      'Remaining-work approval predates execution')
        manifests = [('source.json',plan['source_identity'],original_commit),
                     ('repair_source.json',plans[0]['source_identity'],plans[0]['execution_commit']),
                     ('execution_source.json',plans[1]['source_identity'],serving_commit)]
        if authorization is not None:
            manifests.append(('preflight_source.json',authorization['prior_preflight_source_identity'],scope['source_commit']))
        identities=[]
        for name, expected_identity, commit in manifests:
            path=root/'audit_inputs'/name
            manifest=audit.read(path)
            audit.equal(audit.files[str(path)],expected_identity,'Retained manifest identity/'+name)
            audit.require(manifest['commit']==commit and manifest['clean'] is True,'Clean source identity/'+name)
            identities.append(dict(role=name,commit=commit,identity=expected_identity))
        result['source_manifests']=identities
        result['source_authentication_scope']='Authenticates retained source manifests; full live checkout bytes require companion postflight.'
        result['elapsed_extension'] = None if authorization is None else dict(
            previous_elapsed_s=authorization['previous_elapsed_s'],approved_elapsed_s=authorization['approved_elapsed_s'],
            occupied_card_s=authorization['occupied_card_s'],user_answer=authorization['user_answer'])
    return result


def audit_run(root, public, audit, *, analysis_root=None):
    root, public = Path(root), Path(public)
    inputs = audit.read(root/'evaluation/INPUTS.json')
    references = audit.read(root/'evaluation/REFERENCES.json')
    protocol = audit.read(root/'evaluation/PROTOCOL.json')
    plan = audit.read(root/'sealed/PLAN.json')
    provenance = audit_source_provenance(root, protocol, plan, audit)
    panel, warmups = inputs['panel'], inputs['warmup']
    audit.require(len(panel) == 128 and len(warmups) == 2, 'Fixed inputs')
    all_requests = panel+warmups
    audit.require(len({q['request_id'] for q in all_requests}) == 130
                  and len({tuple(q['prompt_ids']) for q in all_requests}) == 130, 'Distinct inputs')
    for group_name, requests in (('panel', panel), ('warmup', warmups)):
        audit.require(set(references[group_name]) == {q['request_id'] for q in requests}, 'Canonical coverage')
        for ordinal, request in enumerate(requests):
            ref = references[group_name][request['request_id']]
            audit.require(request['request_index'] == ordinal, 'Sealed request order')
            audit.require(request['max_new_tokens'] == 2048 and request['stop_token_ids'] == [151645,151643], 'Pinned stopping policy')
            audit.require(ref['reference_kind'] == 'frozen_DFlash' and ref['verified'] is True, 'Frozen reference provenance')
            audit.equal(ref['provenance'], protocol['canonical_provenance'], 'Canonical provenance')
            audit.require(ref['binding_digest'] == digest({k:v for k,v in ref.items() if k != 'binding_digest'}), 'Canonical digest')
            for key in ('request_id', 'prompt_ids', 'stop_token_ids', 'max_new_tokens'):
                audit.equal(ref[key], request[key], 'Canonical input binding/' + key)
    entries, slots = expected_schedule(panel, warmups)
    audit.equal(protocol['schedule'], entries, 'Exact activation plan')
    audit.require(protocol['gpu'] == plan['devices']['A'], 'Physical card binding')
    rows = audit.read(root/'evaluation/MEASUREMENTS.jsonl', lines=True)
    audit.require(len(rows) == len(slots) == 792, 'Complete declared generation count')
    requests_by_id = {q['request_id']:q for q in all_requests}
    native_blocks = 0
    seen_paths = set()
    repeated_counts = {}
    for index, (row, slot) in enumerate(zip(rows, slots)):
        audit.equal(row, slot, 'Chronological row ' + str(index))
        audit.require(row['valid_measurement'] is True and row['errors'] == [], 'Raw validity row ' + str(index))
        audit.equal(row['physical_gpu_uuid'], protocol['gpu'], 'Physical serving GPU')
        request = requests_by_id[slot['request_id']]
        audit.equal(row['source_id'], request['source_id'], 'Source request binding')
        expected_name = f"a{slot['activation']:03d}_{slot['phase']}_{slot['request_ordinal']:04d}_{slot['method']}.json"
        audit.require(Path(row['raw_path']).name == expected_name, 'Raw file slot binding')
        relative = 'evaluation/raw/' + expected_name
        seen_paths.add(relative)
        reference = references['warmup' if slot['phase'] == 'warmup' else 'panel'][slot['request_id']]
        measured, blocks = check_record(audit, audit.read(root/relative), request, reference, slot,
                                       protocol['checkpoints'][slot['method']], index)
        audit.equal(row, measured, 'Measured/raw agreement ' + str(index))
        repeat_key = (slot['phase'], slot['method'], slot['request_id'])
        current_counts = tuple(measured[key] for key in (*COUNTS, 'finish_reason', 'checkpoint_id'))
        if repeat_key in repeated_counts:
            audit.require(current_counts == repeated_counts[repeat_key], 'Warmup/scored repeat counts')
        repeated_counts[repeat_key] = current_counts
        native_blocks += blocks
    audit.require(len(seen_paths) == 792 and len(list((root/'evaluation/raw').glob('*.json'))) == 792, 'Exactly792 native records')
    status = audit.read(root/'evaluation/STATUS.json')
    audit.equal(status, dict(status='completed', planned=792, attempted=792, returned=792,
        completed_valid=792, invalid=0, pending=None, automatic_retry=False, unknown=0, unattempted=0), 'Serving status')
    work = audit.read(root/'evaluation_work.json')
    audit.equal(work, dict(attempted={'scored_generation':768,'warmup_generation':24},
                          completed={'scored_generation':768,'warmup_generation':24}, pending={}, operations=792), 'Serving ledger')
    result = independent_reduction(rows, audit)
    summary = audit.read(root/'evaluation/SUMMARY.json')
    audit.equal(summary, result, 'Production summary')
    accounting = dict(returned=792, valid=792, scored=768, warmups=24, activations=12)
    audit.equal(summary['generation_accounting'], accounting, 'Generation accounting')
    activations = audit.read(root/'evaluation/ACTIVATIONS.jsonl', lines=True)
    audit.require(len(activations) == 12, 'Activation receipts')
    for index, (actual, expected) in enumerate(zip(activations, entries)):
        audit.equal(actual, expected, 'Activation ' + str(index))
        publication = actual['publication']
        native = publication['native_status']
        emitted = native['publication']
        checkpoint = protocol['checkpoints'][expected['method']]
        audit.equal(publication, dict(checkpoint_structure_valid=True, checkpoint_id=checkpoint), 'Publication')
        audit.equal(native, dict(status='passed', poisoned=None, capture_pending=False, cache_tokens=0,
                                 attention_backend='fa3', block_size=16), 'Publication runtime')
        audit.equal(emitted, dict(status='passed', checkpoint_path=checkpoint, cache_after_tokens=0,
            helper_matches_current_layers=True, helper_norm_matches_current_layers=True, fused_enabled=True), 'Publication cache/helpers')
    for arm in ARMS:
        weight = audit.read(root/f'evaluation/WEIGHTS_{arm}.json')
        audit.equal(weight, dict(errors=[]), 'Serving weight errors')
        audit.equal(weight['receipt'], dict(status='passed', all_equal=True, tensors=58,
            source_tensor_count=58, native_tensor_count=43, checkpoint_path=protocol['checkpoints'][arm],
            helper_matches_current_layers=True, helper_norm_matches_current_layers=True, fused_enabled=True), 'Serving loaded weights')
    costs = dict(loading_s=math.fsum(row['loading_s'] for row in activations),
        activation_s=math.fsum(row['publication']['activation_s'] for row in activations),
        warmup_request_latency_s=math.fsum(row['request_latency_s'] for row in rows if row['phase']=='warmup'),
        scored_request_latency_s=math.fsum(row['request_latency_s'] for row in rows if row['phase']=='scored'))
    audit.equal(summary['serving_costs'], costs, 'Serving costs')
    identities = audit.read(root/'SERVING_PARAMETERS.json')
    audit.equal(summary['parameter_identities'], identities, 'Tensor-content identities')
    distinct = identities['BASE']['sha256'] != identities['TAIL']['sha256']
    audit.require(summary['distinct_serving_weights'] is distinct, 'Distinct tensor-content endpoints')
    training = audit.read(root/'arms/TAIL/STATUS.json')
    mechanism = audit.read(root/'mechanism/STATUS.json')
    unchanged = audit.read(root/'evaluation/EXPORTS_UNCHANGED.json')
    audit.equal(unchanged['BASE'], plan['base_export_identity'], 'BASE export unchanged')
    audit.equal(unchanged['TAIL'], training['checkpoint_identity'], 'TAIL export unchanged')
    activity = sum(row['response_observation']['continuation_repair']['continuation_changed_high_positions']
                   for row in training['training']['records'])
    audit.equal(training['activity']['continuation_changed_high_positions'], activity, 'Once-per-presentation activity')
    complete = training['status'] == mechanism['status'] == 'completed'
    decision = independent_decision(result, distinct=distinct, active=activity>0, complete=complete)
    derived = Path(analysis_root) if analysis_root is not None else root
    final = audit.read(derived/'FINAL_ANALYSIS.json')['serving']
    audit.equal(final, result, 'Final analysis arithmetic')
    audit.equal(final['decision'], decision, 'Decision')
    audit.equal(final['execution_commit'], protocol['execution_commit'], 'Final source identity')
    if provenance.get('orchestration_repairs'):
        audit.equal(final['training_execution_commit'], provenance['training_commit'], 'Final training source')
        audit.equal(final['readout_attempts'], provenance['readout_attempts'], 'Final readout process count')
    if provenance.get('orchestration_repairs') == 2:
        first_readout = audit.read(root/'mechanism/attempts/001/STATUS.json')
        rejected = audit.read(root/'resume_002/REPAIR.json')['rejected_preparation']
        forward_counts = [first_readout['work']['response_reads_completed'], rejected['new_forward_evidence'],
                          mechanism['attempt_work']['response_reads_completed']]
        audit.require(forward_counts == [74,0,63] and sum(forward_counts) == mechanism['work']['response_reads_completed'],
                      'All model and zero-forward readout attempts reconciled')
        audit.equal(final['readout_forward_counts_by_process'], forward_counts, 'Final per-process model forwards')
        audit.equal(final['readout_model_forward_attempts'], sum(value>0 for value in forward_counts),
                    'Final model-forward attempt count')
    public_summary = audit.read(public/'SERVING_SUMMARY.json')
    audit.equal(public_summary, {key:value for key,value in result.items() if key not in ('paired','bootstrap')}, 'Public summary')
    audit.equal(public_summary['decision'], decision, 'Public decision')
    if provenance.get('orchestration_repairs'):
        audit.equal(public_summary['execution_commit'], provenance['serving_commit'], 'Public serving source')
        audit.equal(public_summary['training_execution_commit'], provenance['training_commit'], 'Public training source')
        audit.equal(public_summary['readout_attempts'], provenance['readout_attempts'], 'Public readout process count')
    if provenance.get('orchestration_repairs') == 2:
        audit.equal(public_summary['readout_forward_counts_by_process'], forward_counts, 'Public per-process model forwards')
        audit.equal(public_summary['readout_model_forward_attempts'], 2, 'Public model-forward attempts')
    if analysis_root is not None:
        source = audit.read(derived/'ANALYSIS_SOURCE.json')
        view_path = derived/'ANALYSIS_PLAN.json'
        view = audit.read(view_path)
        audit.equal(source['local_analysis_view_identity'], audit.files[str(view_path)], 'CPU analysis view identity')
        audit.require(source['clean'] is True and len(source['analysis_execution_commit']) == 40
                      and all(value in '0123456789abcdef' for value in source['analysis_execution_commit']),
                      'Pinned clean CPU analysis commit')
        audit.equal(source['serving_execution_commit'], provenance['serving_commit'], 'CPU source receipt serving commit')
        audit.equal(source['training_execution_commit'], provenance['training_commit'], 'CPU source receipt training commit')
        audit.equal(view['analysis_execution_commit'], source['analysis_execution_commit'], 'CPU analysis view commit')
        execution_plan_path = root/('resume_002/EXECUTION_PLAN.json' if provenance.get('orchestration_repairs') == 2
                                   else 'resume_001/EXECUTION_PLAN.json' if provenance.get('orchestration_repairs')
                                   else 'sealed/PLAN.json')
        execution_plan = audit.read(execution_plan_path)
        audit.require({key:value for key,value in view.items() if key!='analysis_execution_commit'} == execution_plan,
                      'CPU view preserves every scientific execution-plan field')
        code = (derived/'analysis.py').read_bytes()
        audit.require(hashlib.sha256(code).hexdigest() == source['analysis_py_sha256'], 'Retained CPU analysis file identity')
        audit.files[str(derived/'analysis.py')] = dict(bytes=len(code),sha256=source['analysis_py_sha256'])
        for label, table in (('final', final),('public', public_summary)):
            audit.equal(table['analysis_execution_commit'], source['analysis_execution_commit'], label+'/CPU analysis commit')
            audit.equal(table['serving_execution_commit'], provenance['serving_commit'], label+'/serving execution commit')
        provenance['analysis_source'] = source
    audit.equal(audit.read(public/'BOOTSTRAP.json'), {key:value for key,value in result['bootstrap'].items() if key!='input_order'}, 'Public bootstrap')
    ordinals = {row['request_id']:i for i,row in enumerate(result['paired'])}
    expected_requests = [dict(request_ordinal=i, **{arm+'_'+key:row[arm][key]
        for arm in ARMS for key in (*TIMES,*COUNTS,'finish_reason')}) for i,row in enumerate(result['paired'])]
    check_csv(audit, public/'PER_REQUEST.csv', expected_requests)
    warmup_ordinals = {key:i for i,key in enumerate(sorted(q['request_id'] for q in warmups))}
    for phase, name in (('scored','ALL_REPEATS.csv'), ('warmup','WARMUPS.csv')):
        lookup = ordinals if phase=='scored' else warmup_ordinals
        expected = [dict(request_ordinal=lookup[row['request_id']], **{key:row[key]
            for key in ('activation','repeat','group','method',*TIMES,*COUNTS,'finish_reason','valid_measurement')})
            for row in rows if row['phase']==phase]
        check_csv(audit, public/name, expected)
    return dict(status='passed', source_commit=protocol['execution_commit'], source_provenance=provenance, generation_accounting=accounting,
        native_blocks=native_blocks, totals=result['totals'], contrasts=result['contrasts'],
        bootstrap={key:value for key,value in result['bootstrap'].items() if key!='input_order'},
        decision=decision, distinct_serving_weights=distinct, changed_high_occurrences=activity,
        serving_costs=costs,
        statistical_fallacy_scan=dict(coverage='11/11 considered',
            simpson='Prespecified summed request means/counts; per-request outcomes and repeat totals retained. No subgroup generalization.',
            ecological='Whole requests are resampling units; timing repeats and training blocks are not independent replications.',
            berkson='Reused selected development panel, explicitly not a fresh-population confirmation.',
            collider='No fitted adjustment or output/performance-based request exclusion.',
            base_rate='Both full call totals and gross saved/added counts retained.',
            regression_to_mean='Fixed endpoints and sealed panel; one pilot cannot establish training-seed robustness.',
            survivorship='Every768 scored execution and24 warmups required; no invalid/dropout removal.',
            look_elsewhere='One prespecified primary timing interval; other intervals descriptive. This does not correct the project-wide history of mechanism exploration.',
            forking_paths='Fixed PCG64 seed,10000 paired draws,linear percentiles,all declared requests; no tuning after outcomes.',
            causal='Endpoint comparison does not isolate a theoretical causal mechanism or establish training-inclusive speedup.',
            reverse_causality='Fixed checkpoints precede evaluation; mechanism readouts remain descriptive training evidence.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis-root', type=Path,
                        help='Separate derived CPU analysis directory; leaves retrieved raw artifacts unchanged')
    args = parser.parse_args()
    audit = Audit()
    result = dict(status='running')
    try:
        result = audit_run(args.raw_root, args.public, audit, analysis_root=args.analysis_root)
    except Exception as error:
        result = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(checks=audit.checks, files_read=len(audit.files),
            input_files_digest=digest(sorted((str(Path(k).relative_to(args.raw_root)) if Path(k).is_relative_to(args.raw_root)
                else 'analysis/'+str(Path(k).relative_to(args.analysis_root)) if args.analysis_root is not None and Path(k).is_relative_to(args.analysis_root)
                else 'public/'+Path(k).name, value) for k,value in audit.files.items())),
            audit_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            numerical_comparison='FP64 arithmetic replay atol1e-11/rtol2e-12 only; scientific decision uses unrounded results',
            independence='Standalone arithmetic and validity code, no production reducer/bootstrap imports; shared context/model family and prior objective authorship disclosed.',
            limits='Authenticates retained canonical/token/trace agreement, not a new verifier run or independent target-only reference. Full parameter-content authentication and server quiet/resource evidence require companion postflight audit.',
            new_model_work=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
