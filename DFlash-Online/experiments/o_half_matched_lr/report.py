"""Audit and report the count-focused O/K matched-learning-rate experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from online_feedback.storage import write_json
from experiments.ok_coding.report import write_csv
from experiments.fresh_confirmation.protocol import COUNT_FIELDS, repeat_errors
from .statistics import METHODS, SEEDS, interpret, summarize

TIMINGS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
ARCHIVED_METHODS = ('F', 'K', 'K_HALF')
NEW_METHODS = ('O', 'O_HALF')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def join_counts(archive, new_rows):
    """Use archived repeat zero and validated new repeat zero, never bridge rows."""
    requests = archive['requests']
    expected = {(seed, request['request_id'], method, repeat)
                for seed in SEEDS for request in requests
                for method in NEW_METHODS for repeat in (0, 1)}
    keys = [(row['seed'], row['request_id'], row['method'], row['repeat']) for row in new_rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('New final measurements do not cover exactly two repeats of the sealed panel')
    if any(row.get('phase') != 'final' or row.get('event') != 4
           or row.get('valid_measurement') is not True or row.get('errors')
           or row.get('validation_status') != 'valid' for row in new_rows):
        raise ValueError('Bridge, invalid or nonfinal rows cannot enter performance analysis')
    grouped = defaultdict(list)
    for row in new_rows:
        grouped[(row['seed'], row['request_id'], row['method'])].append(row)
    joined = []
    for seed in SEEDS:
        for request in requests:
            rid, ordinal = request['request_id'], request['panel_ordinal']
            reference = archive['canonicals'][rid]
            value = dict(seed=seed, event=4, request_id=rid, panel_ordinal=ordinal)
            for method in ARCHIVED_METHODS:
                counts = archive['counts'][str(seed)][method][rid]
                value[method] = {name: counts[name] for name in COUNT_FIELDS}
                value[method].update(
                    cap_hit=reference['finish_reason'] == 'length',
                    finish_reason=reference['finish_reason'], provenance='archived_PR21_repeat_0',
                    count_source=str(archive['source_root']), verified_repeats=3,
                    checkpoint_id=archive['checkpoints'][str(seed)][method])
            for method in NEW_METHODS:
                selected = sorted(grouped[(seed, rid, method)], key=lambda row: row['repeat'])
                if repeat_errors(selected, method) or any(row['panel_ordinal'] != ordinal for row in selected):
                    raise ValueError('New repeated counts or paired request ordinal differ')
                first = selected[0]
                value[method] = {name: first[name] for name in COUNT_FIELDS}
                value[method].update(cap_hit=first['cap_hit'], finish_reason=first['finish_reason'],
                    provenance='new_final_repeat_0', count_source=first['record_path'],
                    verified_repeats=2, checkpoint_id=first['checkpoint_id'])
            for method in METHODS:
                if (value[method]['output_tokens'] != len(reference['response_ids'])
                        or value[method]['finish_reason'] != reference['finish_reason']):
                    raise ValueError('Joined counts do not bind to the immutable canonical response')
            joined.append(value)
    return joined


def response_observations(result, *, seed, event):
    """Ordinary O loss observations precede accumulation merging; no K mask metrics."""
    responses, windows = [], []
    for ordinal, row in enumerate(result['records']):
        observation = row['response_observation']
        if (observation['record_id'] != row['record_id']
                or observation['scope'] != 'one_original_response_forward'
                or observation.get('prefix_protection') is not None):
            raise ValueError('Ordinary O response observation identity or scope differs')
        if not math.isfinite(observation['loss']):
            raise ValueError('Nonfinite ordinary response loss')
        for numerator, denominator in observation['ratio_metrics'].values():
            if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator < 0:
                raise ValueError('Invalid response numerator/denominator')
        if not all(math.isfinite(value) for value in observation['sum_metrics'].values()):
            raise ValueError('Nonfinite response exposure')
        base = dict(seed=seed, event=event, arm='O_HALF', pass_index=row['pass'],
                    window=ordinal // 2 + 1, response_in_window=ordinal % 2,
                    presentation_ordinal=ordinal, record_id=row['record_id'])
        responses.append({**base, 'training_s': row['training_s'],
                          'sampled_anchors': row['sampled_anchors'], **observation})
        if row['optimizer_stepped']:
            gradient = row['metrics']['grad_norm']
            if not math.isfinite(gradient):
                raise ValueError('Nonfinite ordinary window gradient norm')
            windows.append({**base, 'scope': 'optimizer_window',
                'optimizer_step_after': row['optimizer_step_after'], 'gradient_norm': gradient,
                'clip_threshold': 1., 'clipping_active': gradient > 1.,
                'learning_rate': row['learning_rate']})
    if len(responses) != 256 or len(windows) != 128:
        raise ValueError('Ordinary event response/window counts differ')
    return responses, windows


def audit_training(root, plan):
    """Keep every failed boundary visible; aggregate only validated complete events."""
    from experiments.ok_coding.campaign import read_work, require_complete_work
    from experiments.ok_coding.training import validate_event_receipt

    events, errors, processes, devices, caches = [], [], [], [], []
    for seed in SEEDS:
        for event in range(1, 5):
            seed_root = Path(root) / 'campaign_001' / f'seed_{seed}'
            folder = seed_root / f'resume_{event:03d}_O_HALF'
            entry = dict(seed=seed, event=event, arm='O_HALF', valid=False, errors=[])
            try:
                costs_path = folder / 'COSTS.json'
                if costs_path.exists():
                    entry['costs'] = read_json(costs_path)
                result = read_json(seed_root / 'O_HALF' / 'events' / f'event_{event:03d}' / 'RESULT.json')
                schedule = plan['original_schedules'][str(seed)][str(event)]
                schedule = schedule['path'] if isinstance(schedule, dict) else schedule
                validate_event_receipt(result, event_number=event, objective='osd',
                                       reference_schedule=read_json(schedule))
                limits = dict(forwards=256, backwards=256, optimizer_steps=128)
                work = read_work(folder / 'WORK.jsonl', limits)
                entry['work'] = work
                require_complete_work(work, limits)
                worker = read_json(folder / 'WORKER.json')
                if (worker['status'] != 'completed' or worker['work'] != work
                        or worker['additional_qualification_steps'] != 0):
                    raise ValueError('Ordinary worker completion/work differs')
                execution = read_json(folder / 'EXEC.json')
                if execution['returncode'] != 0:
                    raise ValueError('Training interpreter did not finish successfully')
                entry['costs'].update(worker_wall_s=worker['worker_wall_s'],
                                      subprocess_wall_s=execution['wall_s'])
                startup = worker['startup']
                if (startup['cuda_initialized_on_entry'] or startup['warmup_model_work'] != 0
                        or any(not cache['initially_empty'] for cache in startup['caches'].values())):
                    raise ValueError('Fresh cold startup differs')
                live, boundary = read_json(folder / 'LIVE.json'), read_json(folder / 'LIVE_BOUNDARY.json')
                endpoint, closed = read_json(folder / 'SAVED_ENDPOINT.json'), read_json(folder / 'CLOSED.json')
                if (not live['exact_resume'] or not boundary['exact_resume']
                        or endpoint['status'] != 'passed' or not endpoint['export_exact']
                        or endpoint['global_step'] != event * 128):
                    raise ValueError('Own scientific restoration or export equality failed')
                if (not closed['closed'] or closed['failed'] or closed['accumulation_remainder'] != 0
                        or closed['global_step'] != event * 128 or closed['presentations'] != event * 256):
                    raise ValueError('Incomplete ordinary accumulation boundary')
                if event == 1:
                    migration = read_json(folder / 'MIGRATION.json')
                    if (migration['status'] != 'passed'
                            or not migration['source_weights_rng_moments_counters_history_exact']):
                        raise ValueError('Original O initialization migration failed')
                costs = entry['costs']
                if costs['response_observations'] != 256:
                    raise ValueError('Response-cost observation denominator differs')
                if any(costs.get(name, 0.) != 0 for name in ('frozen_reference_setup_s', 'reference_operations')):
                    raise ValueError('O_HALF contains forbidden reference work')
                objective = costs.get('objective_work', {})
                if (any(objective.get(name, 0.) != 0 for name in
                        ('reference_setup_s', 'reference_forward_s', 'reference_head_s', 'extra_loss_s', 'reference_operations'))
                        or objective.get('work', {}).get('reference_response_forwards', 0) != 0):
                    raise ValueError('Ordinary objective contains reference/auxiliary work')
                responses, windows = response_observations(result, seed=seed, event=event)
                if any(row['learning_rate'] != (1.5e-4 if row['optimizer_stepped'] else None)
                       for row in result['records']):
                    raise ValueError('O_HALF learning rate differs')
                entry.update(valid=True, response_records=responses, window_records=windows, startup=startup)
                processes.append(startup['pid'])
                devices.append((startup['cuda_visible_devices'], startup['device_uuid']))
                caches.extend(value['path'] for value in startup['caches'].values())
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                entry['errors'].append(repr(error))
                errors.append(f'{seed}/{event}:' + repr(error))
            events.append(entry)
    if len(processes) != 8 or len(set(processes)) != 8 or len(set(devices)) != 1:
        errors.append('eight_fresh_processes_or_same_updater')
    if len(caches) != 48 or len(set(caches)) != 48:
        errors.append('separate_empty_compiler_caches')
    return events, errors


def aggregate_training(events):
    groups = defaultdict(list)
    for event in events:
        if event['valid']:
            for row in event['response_records']:
                key = f"{row['seed']}/{row['event']}"
                for group in (key, key + f"/pass_{row['pass_index']}"):
                    groups[group].append(row)
    output = {}
    for name, rows in groups.items():
        fields = set().union(*(row['sum_metrics'] for row in rows))
        output[name] = dict(responses=len(rows),
            mean_normalized_response_loss=sum(row['loss'] for row in rows) / len(rows),
            summed_exposure={key: sum(row['sum_metrics'].get(key, 0) for row in rows) for key in sorted(fields)})
    return output


def cost_summary(events, plan):
    """New O_HALF costs stand alone; all reused-arm training times are historical."""
    current = {}
    for seed in SEEDS:
        entries = [row['costs'] for row in events if row['seed'] == seed and row['valid']]
        if not entries:
            continue
        current[str(seed)] = {'events': len(entries)}
        for field in ('training_s', 'event_wall_s', 'loading_initialization_resume_validation_s',
                      'checkpoint_save_s', 'export_s', 'state_validation_hash_s',
                      'worker_wall_s', 'subprocess_wall_s'):
            current[str(seed)][field] = sum(row.get(field, 0.) for row in entries)
        for field in ('peak_allocated_bytes', 'peak_reserved_bytes', 'loading_peak_allocated_bytes',
                      'loading_peak_reserved_bytes'):
            current[str(seed)][field] = max(row.get(field, 0) for row in entries)
        names = set().union(*(row.get('objective_work', {}) for row in entries)) - {'work'}
        current[str(seed)]['objective_subtimers'] = {
            key: sum(row.get('objective_work', {}).get(key, 0.) for row in entries)
            for key in names if all(isinstance(row.get('objective_work', {}).get(key, 0), (int, float)) for row in entries)}
    historical = {}
    for plan_key, methods in (('original_run', ('O', 'K')), ('pr21_run', ('K_HALF',))):
        source = Path(plan[plan_key]) / 'final_report/TRAINING_COSTS.csv'
        if not source.exists():
            historical[plan_key] = {'status': 'unavailable', 'source': str(source)}
            continue
        with source.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for seed in SEEDS:
            for method in methods:
                selected = [row for row in rows if int(row['seed']) == seed and row['arm'] == method]
                if len(selected) != 4 or {int(row['event']) for row in selected} != {1, 2, 3, 4}:
                    historical[f'{seed}/{method}'] = {'status': 'incomplete', 'source': str(source)}
                else:
                    historical[f'{seed}/{method}'] = {'source': str(source), 'status': 'historical_unmatched',
                        **{field: sum(float(row[field]) for row in selected) for field in
                           ('training_s', 'event_wall_s', 'loading_initialization_resume_validation_s',
                            'checkpoint_save_s', 'export_s')}}
    return {'current_O_HALF': current, 'historical_unmatched': historical,
            'scope': 'No fresh matched updater-cost comparison against retained O/K/K_HALF.',
            'timer_overlap': 'Event time includes training/save/export; objective subtimers overlap training/loading. Do not sum nested timers.',
            'comparative_latency': 'Not supported: old/new measurements, checkpoint batching and unequal repeat counts.'}


def flatten_record(row):
    return {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()}


def length_summary(joined):
    output = {}
    for seed in SEEDS:
        rows = [row for row in joined if row['seed'] == seed]
        if not rows:
            continue
        output[str(seed)] = {}
        for method in METHODS:
            lengths = [row[method]['output_tokens'] for row in rows]
            output[str(seed)][method] = dict(requests=len(rows), minimum_output_tokens=min(lengths),
                maximum_output_tokens=max(lengths), mean_output_tokens=sum(lengths)/len(lengths),
                cap_hits=sum(row[method]['cap_hit'] for row in rows),
                termination_counts=dict(Counter(row[method]['finish_reason'] for row in rows)),
                provenance=rows[0][method]['provenance'])
    return output


def audit_archive_reuse(plan):
    """Recheck authoritative archived records; public scalar CSVs are insufficient."""
    import hashlib
    from .archive import audit_archive

    binding = plan['archive']
    payload = Path(binding['path']).read_bytes()
    if (len(payload) != binding['identity']['bytes']
            or hashlib.sha256(payload).hexdigest() != binding['identity']['sha256']):
        raise ValueError('Sealed archive audit identity changed')
    retained = json.loads(payload)
    current = audit_archive(plan['pr21_run'])
    for key in ('status', 'requests', 'canonicals', 'counts', 'checkpoints', 'files',
                'source_commit', 'panel_binding', 'canonical_identity'):
        if retained[key] != current[key]:
            raise ValueError('Archived source evidence changed: ' + key)
    if current['status'] != 'passed':
        raise ValueError('Archived native evidence did not pass')
    return current


def audit_new_evaluations(root, plan, archive):
    from .evaluation import audit_stage

    stages = []
    for phase in ('bridge', 'final'):
        for seed in SEEDS:
            folder = (Path(root) / 'campaign_001' / f'bridge_seed_{seed}' if phase == 'bridge'
                      else Path(root) / 'campaign_001' / f'seed_{seed}' / 'event_004/evaluation')
            try:
                if phase == 'bridge':
                    checkpoints = archive['checkpoints'][str(seed)]
                else:
                    endpoint = read_json(Path(root) / 'campaign_001' / f'seed_{seed}' /
                                         'O_HALF/events/event_004/RESULT.json')
                    checkpoints = {'O': plan['original_endpoints'][str(seed)]['O']['export'],
                                   'O_HALF': endpoint['export_path']}
                audit = audit_stage(folder, archive['requests'], checkpoints,
                                    seed=seed, phase=phase, archive=archive)
            except (OSError, ValueError, KeyError, TypeError) as error:
                audit = dict(status='invalid_or_incomplete', rows=[], errors=[repr(error)], progress={})
            stages.append(dict(seed=seed, phase=phase, folder=str(folder), **audit))
    return stages


def evaluation_accounting(root):
    """Return actual new records/timing costs, including invalid partial work."""
    all_rows, costs = [], []
    for phase in ('bridge', 'final'):
        for seed in SEEDS:
            folder = (Path(root) / 'campaign_001' / f'bridge_seed_{seed}' if phase == 'bridge'
                      else Path(root) / 'campaign_001' / f'seed_{seed}' / 'event_004/evaluation')
            rows = read_rows(folder / 'MEASUREMENTS.jsonl')
            all_rows.extend(rows)
            status = read_json(folder / 'STATUS.json') if (folder / 'STATUS.json').exists() else {'status': 'unattempted'}
            costs.append({'seed': seed, 'phase': phase, **status,
                'returned_timing_totals': {name: sum(row[name] for row in rows
                    if isinstance(row.get(name), (int, float)) and math.isfinite(row[name])) for name in TIMINGS},
                'latency_comparison_supported': False})
    return all_rows, costs


def validate_retention_receipts(root, events):
    """Odd-event tensor retirement is legitimate only with its sealed proof."""
    from .storage import validate_retired
    errors = []
    for event in events:
        if not event['valid']:
            continue
        seed, number = event['seed'], event['event']
        folder = Path(root) / 'campaign_001' / f'seed_{seed}' / f'resume_{number:03d}_O_HALF'
        try:
            identities = read_json(folder / 'STATE_IDENTITIES.json')
            for saved, identity in identities.items():
                path = Path(saved)
                if not path.exists():
                    validate_retired(root, seed, number, path, identity)
        except (OSError, KeyError, TypeError, ValueError) as error:
            errors.append(f'{seed}/{number}:retention:' + repr(error))
    return errors


def render_report(summary):
    analysis = summary['analysis']
    decision = summary['interpretation']
    lines = ['# Ordinary OSD at matched lower learning rate: four-event coding comparison', '',
        'Execution source: `' + summary['execution_source']['commit'] + '`.', '',
        'Question: does K retain useful drafting benefit over ordinary OSD at the same LR1.5e-4? '
        'Before execution, O_HALF was expected potentially to recover part of ordinary O’s deficit; '
        'the relative outcome was unknown. Only O_HALF was trained.', '',
        '**Decision ' + decision['branch'] + ': ' + decision['label'] + '.**', '',
        'This reuses the PR21 development panel that already informed the choice of K_HALF. '
        'It is development/ablation evidence, not fresh confirmation. F/K/K_HALF counts are verified '
        'archived repeat-zero trajectories; O/O_HALF counts come from new two-repeat evaluation. '
        'The compatibility bridge is excluded from the performance sample.', '']
    if analysis is not None:
        lines += ['| Scope | Method | LR | Count provenance | Calls | Output tokens | Calls / 1,000 tokens | Accepted tokens | Progress / call |',
                  '|---|---|---:|---|---:|---:|---:|---:|---:|']
        for scope, methods in analysis['method_totals'].items():
            for method, value in methods.items():
                lr = 'released' if method == 'F' else ('1.5e-4' if method.endswith('_HALF') else '3e-4')
                provenance = 'archived PR21' if method in ARCHIVED_METHODS else 'new two-repeat'
                lines.append(f"| {scope} | {method} | {lr} | {provenance} | {value['verification_calls']} | "
                    f"{value['output_tokens']} | {value['calls_per_1000_output_tokens']:.4f} | "
                    f"{value['accepted_draft_tokens']} | {value['committed_progress_per_call']:.4f} |")
        lines += ['', '| Scope | Comparison | Call reduction | Paired 95% interval | Wins / regressions / ties | Gross saved / added |',
                  '|---|---|---:|---:|---:|---:|']
        for scope, comparisons in [('combined', analysis['combined']), *analysis['per_seed'].items()]:
            for name, value in comparisons.items():
                effect = value['call_reduction']; low, high = effect['ci95_pct']
                lines.append(f"| {scope} | {name} | {effect['point_pct']:+.4f}% | [{low:+.4f}, {high:+.4f}]% | "
                    f"{value['request_seed_wins']} / {value['request_seed_regressions']} / {value['request_seed_ties']} | "
                    f"{value['gross_calls_saved']} / {value['gross_calls_added']} |")
        lines += ['', 'The reverse O_HALF/K_HALF reduction is calculated with K_HALF calls as its denominator:', '']
        for scope, value in analysis['reverse_primary'].items():
            effect = value['call_reduction']; low, high = effect['ci95_pct']
            lines.append(f"- {scope}: {effect['point_pct']:+.4f}% [{low:+.4f}, {high:+.4f}]%.")
        lines += ['', 'Objective-by-LR interaction uses the common F-call denominator; positive values mean '
                  'K’s call advantage over O is greater at lower LR. It is not a subtraction of differently normalized reductions.', '']
        for scope, value in analysis['objective_lr_interaction'].items():
            low, high = value['ci95_pct']
            lines.append(f"- {scope}: {value['point_pct']:+.4f}% [{low:+.4f}, {high:+.4f}]%; "
                f"numerator {value['numerator_call_difference']} calls / F {value['denominator_F_calls']}.")
        lines += ['', 'The 0.5% target is a research-priority point-estimate threshold, not deployment break-even. '
                  'A positive interval alone does not satisfy that threshold or prove the true effect exceeds it. '
                  'Practical equivalence requires the entire primary interval within ±0.5 percentage points; '
                  'the provisional-base decision also requires both seed points within that band.', '',
                  'All intervals resample 128 joint request identities carrying both fixed seeds and all methods, '
                  'using 10,000 PCG64 seed20261102 draws and linear percentiles. The 256 seed/request outcomes '
                  'and repeats are not independent samples. Intervals are nominal/descriptive, conditional on these '
                  'fitted trajectories and reused panel; they do not adjust for earlier K_HALF selection. '
                  'Secondary intervals are unadjusted.', '',
                  '| Scope / candidate vs comparator | Retained gain | Lost gain | Added gain | Prevented harm | Added harm | Net saved |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for scope, values in analysis['gain_harm'].items():
            for name, value in values.items():
                columns = [value[key]['point'] for key in ('retained_gain', 'lost_gain', 'added_gain', 'prevented_harm', 'added_harm')]
                lines.append(f"| {scope} / {name} | " + ' | '.join(str(int(v)) for v in columns)
                             + f" | {value['net_calls_saved_vs_comparator']} |")
        lines += ['', 'Positive gains and harms are formed within seed/request before pooling. '
                  'The identity added gain − lost gain + prevented harm − added harm equals net saved calls '
                  'is checked at every paired outcome. Gross gains/harms, retained fractions and paired intervals '
                  'are retained in SUMMARY.json. This accounting does not identify gradient conflict, memorization '
                  'or a preservation mechanism.']
    lines += ['', '## Work and cost', '',
              'New training budget: 1,024 Adam steps, 2,048 ordinary response presentations/forward-backward '
              'equivalents, zero reference/diagnostic passes and zero disposable optimizer qualification steps. '
              'New serving budget: 48 compatibility-bridge plus 1,024 O/O_HALF executions = 1,072; '
              'all 112 gate/bridge executions are included. Actual completion/failure counters appear in SUMMARY.json.', '',
              '| O_HALF seed | Valid events | Training seconds | Event wall seconds | Loading/resume seconds | Peak allocated GiB |',
              '|---|---:|---:|---:|---:|---:|']
    for seed, value in summary['costs']['current_O_HALF'].items():
        lines.append(f"| {seed} | {value['events']} | {value['training_s']:.3f} | {value['event_wall_s']:.3f} | "
                     f"{value['loading_initialization_resume_validation_s']:.3f} | {value['peak_allocated_bytes']/2**30:.3f} |")
    lines += ['', 'Historical O/K/K_HALF updater costs are unmatched context in COST_SUMMARY.json. '
              'Nested forward/head/compile/setup timers are not additive to their parent totals. '
              'Response losses are separately normalized observations captured before accumulation merging; '
              'window gradient/clipping records are counted once. O and K have different masks/normalization, '
              'so their raw losses are not evidence of serving improvement.', '',
              '**No comparative serving latency or throughput result is supported.** New evaluations batch '
              'checkpoints and use two repeats; reused methods were measured previously with three. '
              'Request, native-generation, cache, loading and activation times are operational accounting only. '
              'No old/new timing pooling, call-count-to-speedup conversion, or training-inclusive speedup is claimed.']
    if summary['errors']:
        lines += ['', 'Evidence failures; completed/invalid/unattempted work remains retained:', '',
                  *['- ' + error for error in summary['errors']]]
    lines += ['', '## Scope and one next action', '', decision['next_action'], '',
              'This recommendation does not authorize a follow-on experiment or a project-stage change. '
              'The fixed O-generated feedback stream is shared replay, not an O_HALF-controlled online deployment. '
              'Equal LR neither matches Adam displacement nor isolates K’s selection from its normalization. '
              'Lower LR also changes decoupled weight decay. No general learning-rate optimum, convergence, '
              'domain-shift/stochastic-decoding benefit or novelty claim follows.', '',
              'Files: [paired counts](PAIRED_REQUESTS.csv), [all new evaluations](ALL_NEW_EVALUATIONS.csv), '
              '[summary](SUMMARY.json), [bridge audit](BRIDGE_REPORT.json), [training audit](TRAINING_AUDIT.json), '
              '[costs](COST_SUMMARY.json), [training costs](TRAINING_COSTS.csv), '
              '[evaluation accounting](EVALUATION_COSTS.json).', '']
    return '\n'.join(lines)


def write_campaign_report(root, output, *, execution_commit=None):
    from .contract import validate_plan

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(root / 'sealed/PLAN.json')
    errors, archive, stages, joined = [], None, [], []
    try:
        validate_plan(plan)
        if execution_commit is not None and execution_commit != plan['execution_source']['commit']:
            raise ValueError('Report execution revision differs from the sealed source')
    except (ValueError, KeyError, TypeError) as error:
        errors.append('plan:' + repr(error))
    try:
        archive = audit_archive_reuse(plan)
        stages = audit_new_evaluations(root, plan, archive)
        for stage in stages:
            errors.extend(f"{stage['phase']}/{stage['seed']}:" + error for error in stage['errors'])
        final_rows = [row for stage in stages if stage['phase'] == 'final' and not stage['errors'] for row in stage['rows']]
        if all(not stage['errors'] for stage in stages) and len(stages) == 4:
            joined = join_counts(archive, final_rows)
            if len(joined) != 256:
                raise ValueError('Final report requires all128 requests carrying both seeds')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        errors.append('archived_or_new_evaluation:' + repr(error))
    events, training_errors = audit_training(root, plan)
    errors.extend(training_errors)
    errors.extend(validate_retention_receipts(root, events))
    status, reconstructed = {}, {}
    try:
        from .campaign import reconcile, require_complete
        status = read_json(root / 'campaign_001/STATUS.json')
        reconstructed = reconcile(root)
        require_complete(reconstructed)
        if status['status'] != 'completed' or any(status.get(key) != value for key, value in reconstructed.items()):
            errors.append('campaign_completion_or_counter_reconstruction')
    except (OSError, ValueError, KeyError, TypeError) as error:
        status = {**status, 'report_completion_error': repr(error)}
        errors.append('campaign:' + repr(error))
    new_rows, evaluation_costs = evaluation_accounting(root)
    analysis = summarize(joined) if joined and not errors else None
    decision = analysis['interpretation'] if analysis else interpret({}, {}, valid=False)
    summary = dict(status='valid' if not errors else 'invalid_or_incomplete', errors=errors,
        execution_source=plan['execution_source'], campaign_status=status,
        reconstructed_campaign_counts=reconstructed, interpretation=decision,
        analysis=analysis, lengths_and_termination=length_summary(joined),
        costs=cost_summary(events, plan),
        archive_reuse={'status': archive['status'], 'source_root': archive['source_root'],
                      'source_commit': archive['source_commit'], 'panel_binding': archive['panel_binding']}
                      if archive else {'status': 'not_validated'},
        training_scope='Only validated complete events enter aggregates; failed raw receipts and costs remain visible.',
        latency_comparison_supported=False)
    write_json(output / 'SUMMARY.json', summary)
    write_json(output / 'BRIDGE_REPORT.json', {'stages': [v for v in stages if v['phase'] == 'bridge'],
                                               'performance_sample': False})
    write_json(output / 'EVALUATION_AUDIT.json', stages)
    write_json(output / 'EVALUATION_COSTS.json', {'stage_receipts': evaluation_costs,
        'campaign_operational_costs': read_json(root / 'campaign_001/COSTS.json')
            if (root / 'campaign_001/COSTS.json').exists() else {'status': 'unavailable'},
        'latency_comparison_supported': False})
    write_json(output / 'TRAINING_AUDIT.json', {'errors': training_errors, 'events': events})
    write_json(output / 'TRAINING_AGGREGATES.json', aggregate_training(events))
    write_json(output / 'COST_SUMMARY.json', summary['costs'])
    write_csv(output / 'ALL_NEW_EVALUATIONS.csv', [flatten_record(row) for row in new_rows])
    write_csv(output / 'PAIRED_REQUESTS.csv', [
        {key: value for key, value in row.items() if key not in METHODS} |
        {method + '_' + key: value for method in METHODS for key, value in row[method].items()}
        for row in joined])
    write_csv(output / 'TRAINING_COSTS.csv', [dict(seed=v['seed'], event=v['event'], arm='O_HALF',
               valid=v['valid'], **flatten_record(v['costs'])) for v in events if 'costs' in v])
    write_csv(output / 'TRAINING_RESPONSES.csv', [flatten_record(row) for v in events if v['valid'] for row in v['response_records']])
    write_csv(output / 'TRAINING_WINDOWS.csv', [row for v in events if v['valid'] for row in v['window_records']])
    (output / 'REPORT.md').write_text(render_report(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--execution-commit')
    args = parser.parse_args()
    summary = write_campaign_report(args.root, args.output or str(Path(args.root) / 'final_report'),
                                    execution_commit=args.execution_commit)
    print(json.dumps({'status': summary['status'], 'errors': summary['errors']}))
    raise SystemExit(0 if summary['status'] == 'valid' else 2)
