#!/usr/bin/env python3
"""Assemble the fixed BASE/TAIL full-run report after execution and audits finish.

CPU-only, standard library, no experimental or model imports. This is report
assembly, not an independent statistical replication. It consumes the retained
analysis and requires separately supplied passing serving/work/resource audits.
Original private files and seals are read only; public tables use ordinals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ARMS = ('BASE', 'TAIL')
METHODS = ('BASE', 'TAIL', 'AREPO', 'F')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNTS = ('output_tokens', 'verification_calls', 'accepted_draft_tokens',
          'native_accepted_draft_tokens', 'committed_progress')
CONTRASTS = ('TAIL_vs_BASE', 'BASE_vs_AREPO', 'TAIL_vs_AREPO', 'BASE_vs_F', 'TAIL_vs_F')
WORK = dict(feedback_generation=14848, teacher_preparation=14848,
            response_forward=29696, response_backward=29696, adam=14848,
            correctness_generation=8, canonical_generation=1024,
            scored_generation=12288, warmup_generation=96)
EXPECTED_WITNESSES = ('released',) + tuple(f'{arm}_{event:03d}' for arm in ARMS for event in (1, 16, 32, 58))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    result = float(value)
    require(math.isfinite(result), f'Nonfinite {label}')
    return result


def total(values, label):
    return math.fsum(finite(v, label) for v in values)


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def identity(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha256(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def csv_table(path, rows):
    require(bool(rows), f'No rows for {Path(path).name}')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decision(report):
    """Three separate unrounded conclusions; counts are not an adoption gate."""
    result = report['contrasts']['TAIL_vs_BASE']['request_latency_s_degradation_pct']
    point = finite(result['value'], 'primary estimate')
    lower, upper = [finite(x, 'primary confidence limit') for x in result['ci95']]
    require(lower <= upper, 'Inverted confidence interval')
    return dict(selected='TAIL' if point <= .1 else 'BASE',
                practical_rule_met=point <= .1,
                degradation_bound_supported=upper <= .1,
                speed_improvement_supported=upper < 0,
                observed_degradation_pct=point, ci95=[lower, upper], threshold_pct=.1)


def completed_evidence(raw, plan):
    status, work, clock = [read(raw / name) for name in ('STATUS.json', 'WORK.json', 'CLOCK.json')]
    require(status['status'] == 'completed', 'Campaign is not complete')
    require(not work['pending'] and not clock['active'], 'Unknown work or unresolved GPU lease remains')
    for key, count in WORK.items():
        require(work['completed'].get('production.' + key, 0) == count, f'Incomplete production {key}')
    require(work['completed'].get('production.canonical_warmup_generation', 0) == 0,
            'This plan reuses both authenticated warmup references')
    qualification = read(raw / 'qualification/QUALIFICATION.json')
    require(qualification['status'] == 'passed', 'Qualification has not passed')
    require(qualification['execution_commit'] == plan['execution_commit'], 'Qualification source differs from plan')
    for kind, ceiling in (('adam', 8), ('generation', 16)):
        require(work['attempted'].get('qualification.' + kind, 0) <= ceiling,
                f'Qualification {kind} ceiling exceeded')
    statuses = {}
    for arm in ARMS:
        state = read(raw / 'campaign_001/arms' / arm / 'STATUS.json')
        require((state['status'], state['events_completed'], state['global_step'], state['presentations']) ==
                ('completed', 58, 7424, 14848), f'{arm} is not a complete 58-event trajectory')
        require(state['binding'] == plan['binding'] and state['execution_commit'] == plan['execution_commit'],
                f'{arm} provenance mismatch')
        require([r['event'] for r in state['event_costs']] == list(range(1, 59)), f'{arm} event cost gaps')
        require(not state.get('cleanup_errors'), f'{arm} worker cleanup failed')
        statuses[arm] = state
    canonical = read(raw / 'canonicals/STATUS.json')
    evaluation = read(raw / 'evaluation/STATUS.json')
    require(canonical['status'] == evaluation['status'] == 'completed', 'Serving phases incomplete')
    require(canonical['completed'] == 1024 and evaluation['completed_valid'] == 12384, 'Serving slots incomplete')
    require(not any(row[k] for row in (canonical, evaluation) for k in ('invalid', 'unknown', 'unattempted')),
            'Invalid, unknown, or unattempted serving slots remain')
    activation_count = sum(len((raw / phase / 'ACTIVATIONS.jsonl').read_text().splitlines())
                           for phase in ('canonicals', 'evaluation'))
    require(work['completed'].get('production.activation', 0) == activation_count,
            'Global canonical/scored activation work does not reconcile')
    require(status['clock']['elapsed_s'] <= 72 * 3600 and clock['completed_card_s'] <= 120 * 3600,
            'Elapsed or GPU-card time cap exceeded; report authorized revision separately before assembly')
    return statuses, work, clock, status, qualification


def checked_analysis(raw, plan):
    report = read(raw / 'FINAL_ANALYSIS.json')
    require(report['execution_commit'] == plan['execution_commit'], 'Analysis execution provenance differs')
    require(report['requests'] == 1024 and len(report['paired']) == 1024, 'Wrong analysis panel size')
    bootstrap = report['bootstrap']
    require((bootstrap['draws'], bootstrap['generator'], bootstrap['seed'], bootstrap['percentiles']) ==
            (10000, 'PCG64', 20261126, 'linear 2.5/97.5'), 'Bootstrap contract changed')
    require(bootstrap['shared_indices_across_endpoints_metrics'] is True, 'Bootstrap indices were not shared')
    require(bootstrap['input_order'] == [r['request_id'] for r in report['paired']], 'Bootstrap request order mismatch')
    require(len(set(bootstrap['input_order'])) == 1024, 'Duplicate paired requests')
    selected = decision(report)
    require(report['decision']['status'] == 'completed', 'Analysis decision is blocked')
    for key, value in selected.items():
        require(report['decision'][key] == value, f'Changed decision field: {key}')
    for method in METHODS:
        for key in (*TIMES, *COUNTS):
            observed = total((row[method][key] for row in report['paired']), method + '/' + key)
            require(math.isclose(observed, report['totals'][method][key], rel_tol=1e-12, abs_tol=1e-9),
                    'Paired table and summary totals disagree')
    for name in CONTRASTS:
        contrast = report['contrasts'][name]
        require(contrast['wins'] + contrast['regressions'] + contrast['ties'] == 1024, 'Request outcome denominator changed')
        for key in (*[k + '_degradation_pct' for k in TIMES], 'verification_call_reduction_pct'):
            finite(contrast[key]['value'], name + '/' + key)
            limits = [finite(x, name + '/' + key) for x in contrast[key]['ci95']]
            require(limits[0] <= limits[1], 'Invalid comparison interval')
    return report


def archived_event(raw, arm, event, zstd):
    """Stream once, verify every member, retain only compact report inputs."""
    archive = raw / 'compact' / arm / f'event_{event:03d}.tar.zst'
    index = read(str(archive) + '.index.json')
    require(index['binding_digest'] == digest({k: v for k, v in index.items() if k != 'binding_digest'}),
            'Compact index binding mismatch')
    require(identity(archive) == dict(bytes=index['archive_bytes'], sha256=index['archive_sha256']),
            'Compact archive identity mismatch')
    prefix = f'campaign_001/arms/{arm}'
    result_name = f'{prefix}/trainer/events/event_{event:03d}/RESULT.json'
    resource_name = f'{prefix}/events/event_{event:03d}/RESOURCE_COSTS.json'
    serving_prefix = f'{prefix}/events/event_{event:03d}/loop/serving/'
    selected, seen = {}, set()
    process = subprocess.Popen([zstd, '-q', '-d', '-c', str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|', ignore_zeros=True) as stream:
            for member in stream:
                name = member.name
                require(member.isfile() and name in index['members'] and name not in seen, 'Unexpected compact member')
                expected = index['members'][name]
                require(member.size == expected['bytes'], 'Compact member size mismatch')
                keep = name in (result_name, resource_name) or name.startswith(serving_prefix)
                checksum, pieces = hashlib.sha256(), []
                with stream.extractfile(member) as payload:
                    for chunk in iter(lambda: payload.read(1024 * 1024), b''):
                        checksum.update(chunk)
                        if keep:
                            pieces.append(chunk)
                require(checksum.hexdigest() == expected['sha256'], 'Compact member hash mismatch')
                if keep:
                    selected[name] = json.loads(b''.join(pieces))
                seen.add(name)
        process.stdout.close()
        stderr = process.stderr.read().decode('utf-8', 'replace')
        require(process.wait() == 0, 'Archive decompression failed: ' + stderr)
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    require(seen == set(index['members']), 'Compact archive missing members')
    feedback_costs = [v for k, v in selected.items() if k.startswith(serving_prefix) and k.endswith('.costs.json')]
    feedback = [v for k, v in selected.items() if k.startswith(serving_prefix) and not k.endswith('.costs.json')]
    require(len(feedback_costs) == len(feedback) == 128, 'Feedback record/cost coverage differs from 128')
    retirement = read(str(archive) + '.retirement.json')
    require(retirement['status'] == 'completed', 'Completed-event compaction is unresolved')
    return selected[result_name], selected[resource_name], feedback_costs, feedback, dict(
        arm=arm, event=event, archive_bytes=index['archive_bytes'], archive_sha256=index['archive_sha256'],
        original_member_count=len(index['members']), removed_loose_files=len(retirement['removed_paths']),
        removed_loose_allocated_bytes=retirement['removed_allocated_bytes'], wall_s=retirement['wall_s'])


def summarize_event(arm, event, result, resource, feedback_costs, feedback, costs):
    require((result['method'], result['event_id'], result['status'], result['response_presentations'],
             result['optimizer_steps'], result['start_step'], result['end_step']) ==
            (arm, f'event_{event:03d}', 'completed', 256, 128, (event - 1) * 128, event * 128), 'Event accounting mismatch')
    records = result['records']
    require(len(records) == 256, 'Missing original response presentations')
    observations = [r['response_observation'] for r in records]
    keys = [(r['record_id'], r['pass_index']) for r in observations]
    require(len(set(keys)) == 256 and len({r['record_id'] for r in observations}) == 128, 'Duplicate or missing original response observations')
    component_rows = [r['effective_components'] for r in observations]
    for row in component_rows:
        denominator = finite(row['denominator'], 'response denominator')
        require(denominator > 0, 'Nonpositive response denominator')
        residual = finite(row['effective_FKL_numerator'], 'effective FKL') + finite(row['effective_PAIR_numerator'], 'effective PAIR') - finite(row['effective_total_numerator'], 'effective total')
        require(abs(residual) <= row['reconciliation_tolerance'], 'Effective loss components do not reconcile')
        require(residual == row['reconciliation_error'], 'Stored component residual differs')
    activity = [r.get('continuation_repair') for r in observations]
    require(all(v is not None for v in activity) if arm == 'TAIL' else all(v is None for v in activity),
            'Continuation activity belongs only to TAIL')
    active = [r for r in activity if r is not None]
    response_lengths = [len(row['response_ids']) for row in feedback]
    calls = sum(row['metadata']['verification_calls'] for row in feedback)
    accepted = sum(row['metadata']['accepted_draft_tokens'] for row in feedback)
    native_accepted = sum(block['native_accepted_draft_tokens'] for row in feedback for block in row['metadata']['trace'])
    progress = sum(row['metadata']['committed_progress'] for row in feedback)
    metrics = dict(arm=arm, event=event, feedback_responses=128, response_tokens=sum(response_lengths),
        response_length_min=min(response_lengths), response_length_max=max(response_lengths),
        response_length_mean=sum(response_lengths) / 128, presentations=256, optimizer_steps=128,
        feedback_verification_calls=calls, feedback_accepted_draft_tokens=accepted,
        feedback_native_accepted_draft_tokens=native_accepted, feedback_committed_progress=progress,
        feedback_accepted_per_call=accepted / calls, feedback_native_accepted_per_call=native_accepted / calls,
        feedback_progress_per_call=progress / calls,
        start_step=(event-1)*128, end_step=event*128,
        effective_FKL_mean=total((r['effective_FKL'] for r in component_rows), 'normalized FKL') / 256,
        effective_PAIR_mean=total((r['effective_PAIR'] for r in component_rows), 'normalized PAIR') / 256,
        effective_total_mean=total((r['effective_total'] for r in component_rows), 'normalized total') / 256,
        max_component_error=max(abs(r['reconciliation_error']) for r in component_rows),
        component_checks=256,
        continuation_active_presentations=sum(bool(r['continuation_active_response']) for r in active),
        continuation_changed_high_positions=sum(int(r['continuation_changed_high_positions']) for r in active),
        continuation_moved_weight=total((r['continuation_moved_weight'] for r in active), 'moved weight'),
        continuation_high_occurrences=sum(int(r['continuation_high_count']) for r in active),
        continuation_low_occurrences=sum(int(r['continuation_low_count']) for r in active),
        continuation_kappa_mean=total((r['continuation_kappa'] for r in active), 'kappa') / 256,
        continuation_fk_correction_sum=total((r['continuation_fk_correction'] for r in active), 'FKL correction'),
        continuation_pair_correction_sum=total((r['continuation_pair_correction'] for r in active), 'PAIR correction'))
    timing = dict(arm=arm, event=event)
    for key in ('training_s', 'checkpoint_save_s', 'export_s', 'update_call_s', 'teacher_preparation_s',
                'feature_persistence_s', 'retirement_s', 'compact_persistence_s'):
        timing[key] = finite(costs[key], key)
    timing['activation_s'] = finite(costs['activation']['activation_s'], 'activation time')
    for key in ('serve_call_s', 'buffer_persistence_s', 'generation_validation_and_recording_s', 'native_request_latency_s'):
        timing['feedback_' + key] = total((r[key] for r in feedback_costs), key)
    timing['feedback_native_generate_s'] = total((r['metadata']['native_generate_s'] for r in feedback), 'native generation time')
    timing['feedback_cache_maintenance_s'] = total((r['metadata']['capture']['cache_maintenance_s'] for r in feedback), 'feedback cache time')
    timing['training_peak_allocated_bytes'] = resource['training_peak_allocated_bytes']
    timing['training_peak_reserved_bytes'] = resource['training_peak_reserved_bytes']
    return metrics, timing


def witness_tables(raw):
    rows = []
    metrics = ('Y', 'raw_repair', 'useful_repair', 'reachable_origin_followers', 'lost_prefix',
               'damaged_prefix_positions', 'prefix_damaged')
    for endpoint in EXPECTED_WITNESSES:
        receipt = read(raw / 'witness' / endpoint / 'STATUS.json')
        require(receipt['status'] == 'completed' and receipt['training_state_unchanged'] and receipt['state']['status'] == 'passed',
                'Witness readout or training-state restoration failed')
        require(len(receipt['responses']) == 8, 'Witness response coverage changed')
        blocks = receipt['paired_blocks']
        for block in blocks:
            require(block['Y'] == block['useful_repair'] + block['reachable_origin_followers'] - block['lost_prefix'],
                    'Witness decomposition mismatch')
        row = dict(endpoint=endpoint, event=receipt['event'], responses=8, paired_blocks=len(blocks),
                   actual_response_forwards=receipt['work']['draft_forwards_completed'],
                   wall_s=receipt['wall_s'], training_state_unchanged=True)
        row.update({key: sum(int(block[key]) for block in blocks) for key in metrics})
        rows.append(row)
    require(sum(r['actual_response_forwards'] for r in rows) <= 72, 'Witness forward ceiling exceeded')
    return rows


def public_serving_tables(output, report, rows):
    lookup = {row['request_id']: index for index, row in enumerate(report['paired'])}
    require(len(rows) == 12384 and all(r['valid_measurement'] for r in rows), 'Serving table is incomplete or invalid')
    csv_table(output / 'PER_REQUEST.csv', [dict(request_ordinal=i, **{
        method + '_' + key: row[method][key] for method in METHODS for key in (*TIMES, *COUNTS, 'finish_reason')
    }) for i, row in enumerate(report['paired'])])
    for phase, filename in (('scored', 'ALL_REPEATS.csv'), ('warmup', 'WARMUPS.csv')):
        chosen = [row for row in rows if row['phase'] == phase]
        mapping = lookup if phase == 'scored' else {value: index for index, value in enumerate(sorted({row['request_id'] for row in chosen}))}
        require(len(chosen) == (12288 if phase == 'scored' else 96), 'Wrong per-phase table size')
        csv_table(output / filename, [dict(request_ordinal=mapping[row['request_id']], **{
            key: row[key] for key in ('activation', 'repeat', 'group', 'method', *TIMES, *COUNTS, 'finish_reason', 'valid_measurement')
        }) for row in chosen])
    repeat_rows = [dict(method=m, repeat=r, **{key: report['totals'][m]['repeat_totals'][key][r] for key in TIMES})
                   for m in METHODS for r in range(3)]
    csv_table(output / 'REPEAT_TOTALS.csv', repeat_rows)
    contrasts = []
    for name in CONTRASTS:
        value = report['contrasts'][name]
        row = {key: value[key] for key in ('candidate', 'baseline', 'wins', 'regressions', 'ties', 'gross_calls_saved', 'gross_calls_added', 'net_calls_saved')}
        for metric in (*[key + '_degradation_pct' for key in TIMES], 'verification_call_reduction_pct'):
            item = value[metric]
            row.update({metric: item['value'], metric + '_lower95': item['ci95'][0], metric + '_upper95': item['ci95'][1]})
        contrasts.append(row)
    csv_table(output / 'CONTRASTS.csv', contrasts)


def fmt(value):
    return f'{value:+.5f}%'


def interval(value):
    return f"{fmt(value['value'])} [{fmt(value['ci95'][0])}, {fmt(value['ci95'][1])}]"


def statistical_review(commit):
    checks = [
        ('Simpson’s paradox', 'The primary ratio pools paired request costs. Per-request values and the four balanced serving groups remain available. No claim that all groups or requests improve, and no claim that aggregation reversal is universally absent.'),
        ('Ecological fallacy', 'Pooled latency and call savings describe this fixed panel; individual wins/regressions/ties are reported separately.'),
        ('Berkson’s paradox', 'Eligibility and exposure exclusions are fixed before outputs. Selection limits the population; unmeasured selection effects are not proved absent.'),
        ('Collider bias', 'No output-dependent filtering or post-treatment covariate adjustment enters the primary estimate. The mechanism observations do not establish causal identification; structural collider effects remain unresolved.'),
        ('Base-rate neglect', 'Report all 1,024 request outcomes, including ties, and all scored/invalid/unknown/unattempted counts. Acceptance and repair observations retain their denominators.'),
        ('Regression to the mean', 'Use fixed final endpoints and a panel chosen without scores. PR65’s earlier outcome is retained and is not reclassified by this new run.'),
        ('Survivorship bias', 'A completed report requires every declared request/repeat and both full trajectories. Failed attempts and consumed costs are not removed.'),
        ('Look-elsewhere effect', 'TAIL versus BASE is the single primary contrast. All four secondary contrasts are descriptive with unadjusted intervals, not independent confirmatory discoveries.'),
        ('Garden of forking paths', 'The plan fixes the panel, endpoints, bootstrap and unrounded 0.1% rule. This builder does not alter fits, remove unfavorable requests or choose checkpoints.'),
        ('Correlation versus causation', 'The two complete algorithms are compared, but their own evolving responses can differ. Witness/activity associations do not isolate which component caused an endpoint difference, and serving gains do not establish online training-inclusive speedup.'),
        ('Reverse causality', 'Training precedes serving evaluation. Within-training activity and witness changes are descriptive; their association with serving performance does not identify a causal direction.'),
    ]
    return '\n'.join(['# Statistical interpretation review', '', f'Execution commit: `{commit}`.', '',
        'Coverage: 11/11 ARS statistical fallacy categories checked at report assembly. These are scoped interpretation checks, not a proof that all structural biases are absent. Numerical reconstruction is covered by the separate serving audit.', '',
        '| Check | Finding and scope |', '|---|---|',
        *[f'| {name} | {finding} |' for name, finding in checks], ''])


def prose(plan, report, arm_costs, activity, witness, work, clock, campaign, audit_refs):
    chosen = decision(report)
    primary = report['contrasts']['TAIL_vs_BASE']
    timing = primary['request_latency_s_degradation_pct']
    yes = lambda flag: 'Yes' if flag else 'No'
    identical = report['decision']['identical_serving_weights'] or report['decision']['identical_native_trajectories']
    lines = [
        '# BASE versus TAIL: complete 58-event comparison', '',
        '## Material Passport', '',
        f"Experiment: PR66, fixed full-run confirmation. Execution commit: `{plan['execution_commit']}`. Sealed plan binding: `{plan['binding']}`.",
        'Status: complete execution, with the separately referenced CPU audits passed. Report assembly is not a second GPU replication.', '',
        f"**Retain {chosen['selected']} under the user's prespecified practical rule.** TAIL's complete-request latency degradation versus BASE is {interval(timing)} (paired 95% interval). Positive values mean slower. This recommendation uses the unrounded estimate, not the displayed rounding.", '',
        '| Question | Prespecified criterion | Result |', '|---|---|---|',
        f"| Observed practical rule | Degradation ≤ 0.1% | {yes(chosen['practical_rule_met'])} |",
        f"| Evidence for the 0.1% degradation bound | Upper 95% limit ≤ 0.1% | {yes(chosen['degradation_bound_supported'])} |",
        f"| Evidence for faster serving | Upper 95% limit < 0 | {yes(chosen['speed_improvement_supported'])} |", '',
    ]
    if chosen['practical_rule_met'] and not chosen['degradation_bound_supported']:
        lines += ['The point estimate satisfies the user’s preference, but the 0.1% bound remains unresolved. This does not establish equivalence.', '']
    if identical:
        lines += ['The retained tensor or native-trajectory identity check found BASE and TAIL identical in that respect. Timing differences alone therefore cannot establish a mechanism benefit.', '']
    lines += [
        f"Verification-call reduction for TAIL versus BASE is {interval(primary['verification_call_reduction_pct'])}. Request wins/regressions/ties are {primary['wins']}/{primary['regressions']}/{primary['ties']}; gross calls saved/added are {primary['gross_calls_saved']}/{primary['gross_calls_added']}. Calls are a separate outcome, not an additional adoption gate.", '',
        '## Final serving comparison', '',
        'The fixed fresh panel contains 1,024 OpenCode requests. All 12,288 scored executions and 96 warmups passed the native frozen-F output/termination and repeat-consistent call-count checks. The 1,024 fresh F canonical generations are separate from scored timing; both original warmup references were authenticated and reused.', '',
        '| Contrast | Complete-request degradation, 95% CI | Call reduction, 95% CI |', '|---|---:|---:|',
    ]
    for name in CONTRASTS:
        row = report['contrasts'][name]
        lines.append(f"| {row['candidate']} versus {row['baseline']} | {interval(row['request_latency_s_degradation_pct'])} | {interval(row['verification_call_reduction_pct'])} |")
    lines += ['', 'TAIL versus BASE is primary. AREPO and F comparisons are descriptive, with unadjusted intervals. AREPO is the AdaFlash loss implemented in the common framework; this does not reproduce its complete asynchronous system.', '',
        'Each request contributes its mean over three timing repeats; call totals use one validated trajectory. The 10,000 paired whole-request bootstrap draws use PCG64 seed 20261126 and linear 2.5/97.5 percentiles, with shared draws across endpoints and metrics.', '',
        'See [per-request results](PER_REQUEST.csv), [all scored repeats](ALL_REPEATS.csv), [warmups](WARMUPS.csv), [repeat timing totals](REPEAT_TOTALS.csv), [all contrasts](CONTRASTS.csv), and [bootstrap metadata](BOOTSTRAP.json). Native accepted draft tokens and stop-clipped acceptance remain separate in these tables.', '',
        '## Training and mechanism observations', '',
        'Both arms completed 58 events from released weights and fresh seed-17 optimizers: 7,424 AdamW steps and 14,848 response presentations per arm. Each generated its own 7,424 feedback responses from the same prompt stream. The two trajectories therefore need not share generated answers, anchors or later training exposure.', '',
        '| Arm | Active presentations | Changed HIGH coefficient occurrences | Effective component checks |', '|---|---:|---:|---:|',
    ]
    for arm in ARMS:
        rows = [r for r in activity if r['arm'] == arm]
        lines.append(f"| {arm} | {sum(r['continuation_active_presentations'] for r in rows):,} / 14,848 | {sum(r['continuation_changed_high_positions'] for r in rows):,} | {sum(r['component_checks'] for r in rows):,} |")
    lines += ['', 'These are original-forward observations counted once per presentation. Effective FKL and PAIR components reconcile with the actual loss; response means are averaged after each response’s own normalization. Coefficient activity does not by itself establish benefit.', '',
        'The shared witness uses eight responses generated by BASE before its first update, with at most eight independently selected blocks per response. The released origin and events 1, 16, 32 and 58 are observed without parameter updates. This is BASE-training evidence, not held-out evidence or identical exposure for TAIL.', '',
        '| Endpoint | Useful repairs | Reachable followers | Lost old-prefix occurrences | Damaged prefix positions | Signed prefix gain |', '|---|---:|---:|---:|---:|---:|',
    ]
    for row in witness:
        lines.append(f"| {row['endpoint']} | {row['useful_repair']} | {row['reachable_origin_followers']} | {row['lost_prefix']} | {row['damaged_prefix_positions']} | {row['Y']:+d} |")
    lines += ['', 'Every paired witness block satisfies signed progress = useful repair + reachable original followers − lost old-prefix occurrences. See [witness observations](WITNESS.csv) and [per-event exposure/activity](EVENT_TRAINING.csv). These observations describe training-conditioned changes; they are not an independent serving test.', '',
        '## Costs and execution accounting', '',
        '| Arm | Worker wall (h) | Training (h) | Teacher preparation (h) | Feedback serving (h) | Export + state save (h) |', '|---|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        c = arm_costs[arm]
        lines.append(f"| {arm} | {c['worker_wall_s']/3600:.3f} | {c['training_s']/3600:.3f} | {c['teacher_preparation_s']/3600:.3f} | {c['feedback_serve_call_s']/3600:.3f} | {(c['export_s']+c['checkpoint_save_s'])/3600:.3f} |")
    training_cost_change = 100 * (arm_costs['TAIL']['training_s'] / arm_costs['BASE']['training_s'] - 1)
    worker_cost_change = 100 * (arm_costs['TAIL']['worker_wall_s'] / arm_costs['BASE']['worker_wall_s'] - 1)
    lines += ['', f"TAIL's observed training time differs from BASE by {fmt(training_cost_change)}; its complete arm worker time differs by {fmt(worker_cost_change)}. These are fresh complete-trajectory cost comparisons. Because the arms generate their own evolving feedback, they do not isolate the cost of one auxiliary calculation at identical training inputs."]
    lines += ['', f"Total elapsed time on the original clock is {campaign['clock']['elapsed_s']/3600:.3f} h; occupied GPU-card time is {clock['completed_card_s']/3600:.3f} h. The fixed limits are 72 elapsed hours and 120 GPU-card-hours. Witness work totals {sum(r['actual_response_forwards'] for r in witness)} response forwards and {sum(r['wall_s'] for r in witness):.3f} s; it is diagnostic work, not an extra optimizer update.", '',
        'Worker, update, training and component timers have nested scopes and must not be added together. Teacher preparation, feature persistence, export, activation, checkpoint saving and archive maintenance are listed separately in [event costs](EVENT_COSTS.csv) and [cost summaries](COSTS.json). Live feature capture is included in feedback-serving time; the retained receipts do not isolate a capture-only causal overhead. Compiler/recomputation/startup work is covered by the work audit, not silently included in committed Adam counts. Backend loading and disposable-startup timers are reported separately; a standalone trainer-construction timer was not recorded.', '',
        f"Qualification attempted {work['attempted'].get('qualification.adam',0)} disposable Adam steps and {work['attempted'].get('qualification.generation',0)} generations. All attempts, failures, startup and committed work are retained in [work accounting](WORK_ACCOUNTING.json). New canonical warmup generations: 0 (the optional ceiling of 2 was unused).", '',
        f"Recorded failed work operations: {len(work['failures'])}. Their identities, process attempts and any authorized resumes are retained by the work audit. A finite negative or mixed scientific result is not a correctness failure.", '',
        'The storage audit covers the 90 GB new-allocation ceiling and 400 GB A800 reserve. [Retirement summary](RETIREMENT_SUMMARY.json) distinguishes lossless compacting from deletion of bulky recovery/features. Retired tensors are not described as archived or resumable. Final event-57/58 recoveries and event-58 exports must pass the separate postflight audit.', '',
        '## Audit scope and limitations', '',
        'Required CPU audit receipts: ' + ', '.join(f'[{key}]({value})' for key, value in audit_refs.items()) + '.',
        'The [statistical interpretation review](STATISTICAL_REVIEW.md) covers all 11 ARS fallacy categories, with structural and causal limitations stated explicitly.',
        'The report assembler uses no model or production analysis imports. Its arithmetic checks and exact member hashes are additional consistency checks; the independent audits establish the scope of raw reconstruction. Shared implementation context and compact-evidence limitations remain disclosed in those audits.', '',
        'Freshness is relative to the audited retained project history through PR65 and the recorded current-job inventory. This does not prove absence of unrecorded or deleted historical exposure, or of semantic near-duplicates. Inputs were selected without generated lengths or outcomes.', '',
        'This is one dataset and one training seed. Request bootstrap intervals and repeated timing do not establish training-seed or cross-dataset robustness. Serving-only differences do not demonstrate training-inclusive online speedup. Additional TAIL activity or better training-witness values do not establish a causal mechanism benefit on their own.', '',
        'PR65 remains a negative pilot under its own prespecified decision rule; this full-run authorization does not rewrite that result. No algorithm tuning, extra seed, endpoint selection or automatic further run follows this report.', '',
        'Source, plan, audit and artifact identities are in [provenance](PROVENANCE.json). Private prompts, source IDs, tokens, sampled anchors, features and model states are excluded from this public package.', '',
    ]
    return '\n'.join(lines)


def build(args):
    started = time.perf_counter()
    raw, output = Path(args.raw_root), Path(args.output)
    require(not output.exists(), 'Use a new report output directory; preserve previous attempts')
    output.mkdir(parents=True)
    stage = output / '.assembly'
    stage.mkdir()
    try:
        plan = read(args.plan)
        require(plan['binding'] == digest({k: v for k, v in plan.items() if k != 'binding'}), 'Original plan seal mismatch')
        statuses, work, clock, campaign, qualification = completed_evidence(raw, plan)
        report = checked_analysis(raw, plan)
        required = dict(serving=args.serving_audit, work=args.work_audit,
                        resource=args.resource_audit, postflight=args.postflight)
        audit_refs, audit_identities = {}, {}
        for name, path in required.items():
            receipt = read(path)
            require(receipt['status'] == 'passed', name + ' audit has not passed')
            require(receipt['plan_binding'] == plan['binding'] and receipt['execution_commit'] == plan['execution_commit'],
                    name + ' audit does not bind the actual plan/source')
            filename = name.upper() + '_AUDIT.json'
            # Audit owners provide public-safe receipts; never copy arbitrary raw
            # nested details or an entire retained plan into the public report.
            audit_refs[name] = filename
            audit_identities[name] = identity(path)
            write(stage / filename, dict(status='passed', original_receipt_identity=identity(path),
                scope='Original audit receipt retained privately; independently authored public narrative accompanies publication',
                execution_commit=plan['execution_commit'], plan_binding=plan['binding']))
        events, event_costs, compact = [], [], []
        for arm in ARMS:
            for event, entry in enumerate(statuses[arm]['event_costs'], 1):
                result, resource, costs, feedback, archive = archived_event(raw, arm, event, args.zstd)
                activity, timing = summarize_event(arm, event, result, resource, costs, feedback, entry['costs'])
                events.append(activity)
                event_costs.append(timing)
                compact.append(archive)
        witness = witness_tables(raw)
        require(sum(r['actual_response_forwards'] for r in witness) == work['completed'].get('production.witness_response_forward', 0),
                'Witness work does not match durable ledger')
        rows = [json.loads(line) for line in (raw / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
        public_serving_tables(stage, report, rows)
        csv_table(stage / 'EVENT_TRAINING.csv', events)
        csv_table(stage / 'EVENT_COSTS.csv', event_costs)
        csv_table(stage / 'WITNESS.csv', witness)
        csv_table(stage / 'COMPACT_EVENTS.csv', compact)
        costs = {}
        for arm in ARMS:
            chosen = [r for r in event_costs if r['arm'] == arm]
            startup = [read(path) for path in sorted((raw / 'campaign_001/arms' / arm).glob('startup_*/RESULT.json'))]
            require(bool(startup), 'Disposable startup receipts are missing')
            costs[arm] = {key: total((r[key] for r in chosen), key) for key in chosen[0] if key.endswith('_s')}
            costs[arm].update(worker_wall_s=statuses[arm]['worker_wall_s'],
                backend_loading_s=statuses[arm]['backend_loading_s'],
                disposable_startup_wall_s=total((row['wall_s'] for row in startup), 'startup time'),
                disposable_startup_attempts=len(startup),
                disposable_startup_failed_attempts=sum(row['status'] != 'completed' for row in startup),
                trainer_construction_only_s=None,
                max_training_allocated_bytes=max(r['training_peak_allocated_bytes'] for r in chosen),
                max_training_reserved_bytes=max(r['training_peak_reserved_bytes'] for r in chosen))
        cost_contrasts = {key: dict(tail_minus_base_s=costs['TAIL'][key] - costs['BASE'][key],
            tail_degradation_pct=100 * (costs['TAIL'][key] / costs['BASE'][key] - 1))
            for key in ('worker_wall_s', 'training_s', 'update_call_s', 'feedback_serve_call_s', 'teacher_preparation_s')}
        write(stage / 'COSTS.json', dict(arms=costs, tail_versus_base=cost_contrasts,
            comparison_scope='Fresh complete trajectories with own evolving feedback; not a same-input isolated overhead estimate',
            final_serving=report['serving_costs'],
            elapsed_s=campaign['clock']['elapsed_s'], occupied_card_s=clock['completed_card_s'],
            witness_wall_s=total((r['wall_s'] for r in witness), 'witness time'),
            capture_only_overhead_s=None,
            capture_scope='Included in feedback serving; no isolated capture-only timing was recorded',
            timer_scope='Nested timers, not an additive decomposition', execution_commit=plan['execution_commit']))
        write(stage / 'WORK_ACCOUNTING.json', dict(attempted=work['attempted'], completed=work['completed'],
            pending_count=len(work['pending']), failed_operations=len(work['failures']),
            process_receipts=len(list((raw / 'processes').glob('*.json'))),
            original_model_load_epoch=clock['first_model_load_epoch'],
            finished_card_intervals=len(clock['finished']), qualification_limits=work['qualification_limits'],
            production_targets=work['production_targets'], execution_commit=plan['execution_commit']))
        cleanup = None
        if args.cleanup:
            receipt = read(args.cleanup)
            require(receipt['status'] == 'completed', 'Historical cleanup is not complete')
            cleanup = dict(identity=identity(args.cleanup), status=receipt.get('status'),
                           removed_files=len(receipt['removed']),
                           removed_allocated_bytes=receipt['removed_allocated_bytes'],
                           free_before_bytes=receipt['free_before_bytes'], free_after_bytes=receipt['free_after_bytes'],
                           scope='Historical pre-run cleanup; exact paths and lost capability remain in the retained ledger')
        write(stage / 'RETIREMENT_SUMMARY.json', dict(compact_events=len(compact),
            removed_redundant_loose_files=sum(r['removed_loose_files'] for r in compact),
            removed_redundant_loose_bytes=sum(r['removed_loose_allocated_bytes'] for r in compact),
            archives_bytes=sum(r['archive_bytes'] for r in compact), historical_cleanup=cleanup,
            bulk_retirement_scope='See passed work/resource/postflight audits; hashes do not restore retired tensors',
            execution_commit=plan['execution_commit']))
        write(stage / 'BOOTSTRAP.json', {**{k: v for k, v in report['bootstrap'].items() if k != 'input_order'},
            'execution_commit': plan['execution_commit']})
        write(stage / 'SERVING_SUMMARY.json', {k: v for k, v in report.items() if k not in ('paired', 'bootstrap')})
        references = dict(plan=identity(args.plan), final_analysis=identity(raw / 'FINAL_ANALYSIS.json'),
            measurements=identity(raw / 'evaluation/MEASUREMENTS.jsonl'),
            work=identity(raw / 'WORK.json'), clock=identity(raw / 'CLOCK.json'),
            qualification=identity(raw / 'qualification/QUALIFICATION.json'),
            report_builder=identity(__file__), audits=audit_identities)
        (stage / 'REPORT.md').write_text(prose(plan, report, costs, events, witness, work, clock, campaign, audit_refs))
        (stage / 'STATISTICAL_REVIEW.md').write_text(statistical_review(plan['execution_commit']))
        input_names = ('original_config', 'runtime_pin', 'training_pin', 'model_pin', 'manifest',
                       'evaluation_inputs', 'data_seal', 'exposure_inventory', 'witness_selection', 'warmup_canonicals')
        write(stage / 'PROVENANCE.json', dict(execution_commit=plan['execution_commit'], plan_binding=plan['binding'],
            report_assembly='CPU-only report assembly; no scientific fitting or model work',
            evidence=references, endpoint_parameter_identities=report['parameter_identities'],
            source_identity=plan['source_identity'],
            named_input_identities={name: plan['sealed_files'][plan[name]] for name in input_names},
            configuration_identities={arm: plan['sealed_files'][plan['configs'][arm]] for arm in ARMS},
            formula_code_identities=plan['formula_code_identities'],
            initialization=plan['initialization'], original_training_order_digest=plan['original_training_order_digest'],
            serving=plan['serving'], resource_limits=plan['limits'],
            public_privacy='Ordinal tables only; no prompts, source IDs, tokens, features or anchors',
            report_artifacts={p.name: identity(p) for p in stage.iterdir() if p.is_file()}))
        for path in stage.iterdir():
            path.replace(output / path.name)
        stage.rmdir()
        write(output / 'BUILD_STATUS.json', dict(status='completed', wall_s=time.perf_counter() - started,
            execution_commit=plan['execution_commit'], plan_binding=plan['binding'],
            scientific_work='zero', report=identity(output / 'REPORT.md')))
        return output
    except BaseException as error:
        # No success report is published from incomplete/missing evidence.
        write(output / 'BUILD_STATUS.json', dict(status='blocked', error=repr(error),
            scope='Report assembly only; raw evidence unchanged', wall_s=time.perf_counter() - started))
        raise


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'raw-root', 'output', 'serving-audit', 'work-audit', 'resource-audit', 'postflight'):
        command.add_argument('--' + name, required=True)
    command.add_argument('--cleanup')
    command.add_argument('--zstd', default=shutil.which('zstd') or '/home/minhao2/miniconda3/bin/zstd')
    return command


if __name__ == '__main__':
    print(build(parser().parse_args()))
