#!/usr/bin/env python3
"""Summarize retrieved Track2 cost/resource/retention receipts, without model work.

This is descriptive accounting by the trainer author, not an independent
reproduction or the separate endpoint/statistics audit. Remote absolute paths
remain evidence strings; only compact files inside --run-root are opened.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ARMS = ('K', 'U', 'R')
EXECUTION = '8a4cbcc8b78adfee845a0a1409db4a0d29c54058'
EXPOSURES = ('presentations', 'replay_presentations', 'prompt_tokens',
             'response_tokens', 'active_anchors',
             'response_forward_backward_presentations', 'optimizer_steps')
TIMER_NOTES = {
    'campaign_wall_s': 'Outer campaign wall, including cleanup; excludes separate preflight/postflight and report built after this timer.',
    'adaptation_ready_wall_s': 'Nested in campaign wall: initialization, serving backend load, eight adaptation events and their checks.',
    'event_wall_s': 'Disjoint sequential event scopes; each includes pre-update serving, preparation, selection, training workers, maintenance, exports and checks.',
    'worker_wall_s': 'Inside event wall: construction/restore, feature admission, native loops, checkpoint commit, validation, state hashing and cleanup; excludes Python process startup.',
    'process_wall_s': 'Outer subprocess wall including startup and worker wall; never add to worker wall.',
    'construction_and_restore_s': 'Sum scope of construction_s and restore_s; do not add all three.',
    'feature_admission_io_and_validation_s': 'Worker subcomponent before native pass loops.',
    'total_loop_s': 'Includes training_s and input_load_and_loop_bookkeeping_s; do not add loop total to its children.',
    'training_s': 'Native per-response timer sums, including forward/backward/optimizer and observations; excludes input file loading.',
    'checkpoint_save_s': 'Worker subcomponent; export_s in trainer RESULT is deferred zero, not the later materialized-export cost.',
    'export_and_verify_s': 'Post-eviction native export, load and exact-weight verification, outside training workers but inside event/campaign wall.',
    'initialization_s': 'Sum of three initialization workers; separate process-wall scopes include startup.',
    'backend_loading_s': 'One serving-backend load within campaign/adaptation wall.',
    'activation_s': 'Pre-update checkpoint publication and validation inside serving/event wall; excludes final-evaluation activations.',
    'teacher_preparation_s': 'Shared K feedback preparation includes native/terminal loading, binding checks, concatenation and terminal teacher forward; not isolated GPU compute or separately attributable to each arm.',
    'feature_persistence_s': 'Prepared-packet save inside event wall, not all native capture I/O.',
    'descriptor_s': 'Shared CPU descriptor construction within event wall.',
    'selection_s': 'Shared CPU pass selection within event wall.',
    'memory_maintenance_s': 'Reservoir commit/eviction and associated retirement within event wall.',
    'resource_check_s': 'Disk/cache scan sum inside event/evaluation/campaign scopes; RESOURCE_CHECKS.jsonl is canonical, STATUS copies are not additional samples.',
    'evaluation_wall_s': 'Final capture-OFF evaluation scope, including its publication, checks and barriers; inside campaign wall.',
    'compiler_timers': 'Per-name sums over sequential workers; names can nest, and are already inside native training time. Never sum across timer names.',
}
LOSSES = {
    'native_feature_tensor': 'Original native capture tensor removed; compact conditioning evidence survives. Prepared supervision remains only while retained in reservoir.',
    'terminal_feature_tensor': 'Separate terminal teacher tensor removed; compact evidence survives. Prepared supervision remains only while retained in reservoir.',
    'prepared_packet': 'Exact evicted supervision tensor is unavailable for direct replay. Descriptor, source identity and compact membership/validation receipts survive.',
    'predecessor_full_state': 'Direct exact resume from this intermediate full model/Adam/RNG boundary is lost; compact validation survives. Immediate successors can also be retired; later retained boundaries are listed separately.',
    'preceding_export': 'This intermediate serving-weight file is unavailable; compact serving checks/configuration remain.',
    'unclassified': 'Capability loss requires manual interpretation of the recorded path and reason.',
}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def aggregate(rows, key):
    values = [row[key] for row in rows if number(row.get(key))]
    return {'sum': sum(values) if values else None, 'observations': len(values),
            'expected_observations': len(rows)}


def extrema(rows, key):
    available = [(i, row[key]) for i, row in enumerate(rows) if number(row.get(key))]
    if not available:
        return {'min': None, 'max': None, 'observations': 0}
    low, high = min(available, key=lambda pair: pair[1]), max(available, key=lambda pair: pair[1])
    return {'min': low[1], 'max': high[1], 'observations': len(available),
            'min_source': rows[low[0]].get('source'), 'max_source': rows[high[0]].get('source')}


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files, self.missing, self.issues = {}, [], []

    def read(self, relative, *, required=False, lines=False):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('Evidence symlink/path escapes the retrieved run root')
        if not path.is_file():
            if required:
                self.missing.append(str(relative))
            return [] if lines else {}
        data = path.read_bytes()
        self.files[str(relative)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        return [json.loads(line) for line in data.splitlines() if line.strip()] if lines else json.loads(data)

    def glob(self, pattern):
        return [str(path.relative_to(self.root)) for path in sorted(self.root.glob(pattern))]


def category(name):
    for suffix, label in (('_preceding_export.json', 'preceding_export'),
                          ('_predecessor_state.json', 'predecessor_full_state'),
                          ('_prepared.json', 'prepared_packet'),
                          ('_terminal.json', 'terminal_feature_tensor'),
                          ('_native.json', 'native_feature_tensor')):
        if name.endswith(suffix):
            return label
    return 'unclassified'


def read_training(evidence, expected_events):
    rows, initial = [], []
    for arm in ARMS:
        base = f'campaign_001/arms/{arm}'
        worker = evidence.read(f'{base}/INITIAL_WORKER.json', required=True)
        process = evidence.read(f'{base}/initial_PROCESS.json', required=True)
        initial.append(dict(arm=arm, status=worker.get('status'),
                            worker_wall_s=worker.get('worker_wall_s'), process_wall_s=process.get('wall_s'),
                            process_returncode=process.get('returncode')))
        for event in range(1, expected_events + 1):
            label = f'event_{event:03d}'
            prefix = f'{base}/trainer/events/{label}'
            result = evidence.read(f'{prefix}/RESULT.json', required=True)
            worker = evidence.read(f'{base}/events/{label}/WORKER.json', required=True)
            process = evidence.read(f'{base}/{label}_PROCESS.json', required=True)
            resource = evidence.read(f'{base}/events/{label}/RESOURCE_COSTS.json', required=True)
            export = evidence.read(f'{prefix}/EXPORT_CHECK.json', required=True)
            boundary = evidence.read(f'{base}/boundaries/{label}.json', required=True)
            row = dict(arm=arm, event=event, result_status=result.get('status'),
                worker_status=worker.get('status'), process_returncode=process.get('returncode'),
                worker_wall_s=worker.get('worker_wall_s'), process_wall_s=process.get('wall_s'),
                source=f'{prefix}/RESULT.json', pass_exposure=result.get('pass_exposure'),
                result_export_status=result.get('export_status'), boundary_export_status=boundary.get('export_status'),
                export_status=export.get('status'), export_exact=export.get('export_exact'),
                export_and_verify_s=export.get('export_and_verify_s'),
                allocator_peak_allocated_bytes=resource.get('training_peak_allocated_bytes'),
                allocator_peak_reserved_bytes=resource.get('training_peak_reserved_bytes'),
                compiler_timers=resource.get('training_compiler_timers', {}))
            for field in ('training_s', 'checkpoint_save_s', 'event_wall_s_before_commit'):
                row[field] = result.get('costs', {}).get(field)
            for field in ('construction_s', 'restore_s', 'construction_and_restore_s',
                          'feature_admission_io_and_validation_s'):
                row[field] = result.get('runtime_costs', {}).get(field)
            loops = result.get('pass_loop_costs', [])
            for field in ('total_loop_s', 'input_load_and_loop_bookkeeping_s'):
                summary = aggregate(loops, field)
                row[field] = summary['sum'] if len(loops) == 2 and summary['observations'] == 2 else None
            passes = result.get('pass_exposure', [])
            for field in EXPOSURES:
                summary = aggregate(passes, field)
                row[field] = summary['sum'] if len(passes) == 2 and summary['observations'] == 2 else None
            row['fresh_presentations'] = (row['presentations'] - row['replay_presentations']
                if row['presentations'] is not None and row['replay_presentations'] is not None else None)
            # Observation only: work-ledger reconstruction belongs to the separate auditor.
            row['producer_work'] = worker.get('work')
            rows.append(row)
    sums = (*EXPOSURES, 'fresh_presentations', 'worker_wall_s', 'process_wall_s',
            'training_s', 'checkpoint_save_s', 'event_wall_s_before_commit',
            'construction_s', 'restore_s', 'construction_and_restore_s',
            'feature_admission_io_and_validation_s', 'total_loop_s',
            'input_load_and_loop_bookkeeping_s', 'export_and_verify_s')
    arms = {}
    for arm in ARMS:
        subset = [row for row in rows if row['arm'] == arm]
        timers = {}
        for row in subset:
            for name, value in row['compiler_timers'].items():
                timers.setdefault(name, []).append(value)
        arms[arm] = dict(events_with_result=sum(row['result_status'] == 'completed' for row in subset),
            expected_events=expected_events, additive_across_events={key: aggregate(subset, key) for key in sums},
            allocator_peaks={key: extrema(subset, key) for key in ('allocator_peak_allocated_bytes', 'allocator_peak_reserved_bytes')},
            compiler_timers_by_name={name: {'total_s': aggregate(values, 'total_s'),
                'calls': aggregate(values, 'calls')} for name, values in timers.items()})
    return {'events': rows, 'arms': arms, 'initialization': initial,
            'exposure_scope': 'Presentation-weighted token/anchor totals, not unique tokens or FLOPs; replay repeats are intentionally counted.',
            'allocator_scope': 'Maximum of per-worker CUDA allocator peaks reset after construction/restore. Window includes train_event loops and checkpoint persistence/commit; excludes later verify_boundary, state hashing and cleanup. This is not whole-job GPU, device, or host-memory peak.'}


def read_resources(evidence, status, plan):
    canonical = 'campaign_001/RESOURCE_CHECKS.jsonl'
    if (evidence.root / canonical).is_file():
        samples = [dict(row, source=f'{canonical}:{i+1}') for i, row in enumerate(evidence.read(canonical, lines=True))]
        origin = canonical
    else:
        samples = [dict(row.get('resource', row), source=f'campaign_001/STATUS.json:resource_checks[{i}]')
                   for i, row in enumerate(status.get('resource_checks', []))]
        origin = 'STATUS fallback only; may cover fewer checks than canonical JSONL'
        evidence.missing.append(canonical)
    preflight = evidence.read('PREFLIGHT.json', required=True)
    postflight = evidence.read('POSTFLIGHT.json', required=True)
    admission = evidence.read('campaign_001/STORAGE_ADMISSION.json', required=True)
    return dict(sample_origin=origin, samples=samples,
        sampled_extrema={key: extrema(samples, key) for key in (
            'free_bytes', 'compiler_growth_bytes', 'compact_allocated_bytes', 'pending_write_bytes')},
        scope='Discrete filesystem observations, not continuous peaks. Compiler/compact allocated blocks are inode-deduplicated by producer. pending_write_bytes is the sampled gate argument (campaign uses default zero), not observed pending allocation; separate require_write_space checks are not logged here. No host or whole-device peak was measured here.',
        storage_plan=plan.get('storage'), admission=admission,
        preflight=dict(status=preflight.get('status'), host_available_bytes=preflight.get('host_available_bytes'),
                       hardware=preflight.get('hardware'), storage=preflight.get('storage')),
        postflight=dict(status=postflight.get('status'), storage=postflight.get('storage')),
        resource_check_timer=aggregate(samples, 'resource_check_wall_s'))


def read_retention(evidence, last_event):
    if not (evidence.root / 'campaign_001/RETIREMENTS').is_dir():
        evidence.missing.append('campaign_001/RETIREMENTS/')
    rows = [dict(evidence.read(path), source=path, category=category(path))
            for path in evidence.glob('campaign_001/RETIREMENTS/*.json')]
    retired = [row for row in rows if row.get('status') == 'retired']
    paths = Counter(row.get('path') for row in retired)
    duplicate_paths = sorted(str(path) for path, count in paths.items() if count > 1)
    if duplicate_paths:
        evidence.issues.append('Duplicate retired-path receipts: do not interpret their sum as distinct files')
    groups = {}
    for name in sorted({row['category'] for row in rows}):
        subset = [row for row in retired if row['category'] == name]
        groups[name] = dict(retired_receipts=len(subset), distinct_paths=len({row.get('path') for row in subset}),
            nominal_logical_bytes=aggregate(subset, 'bytes'), nominal_allocated_bytes=aggregate(subset, 'allocated_bytes'),
            hardlinks_greater_than_one=sum(number(row.get('hardlinks')) and row['hardlinks'] > 1 for row in subset),
            lost_capability=LOSSES[name])
    retained = []

    def add(kind, path, source, identity=None, **details):
        if not path:
            return
        conflict = path in paths
        if conflict:
            evidence.issues.append('Retained artifact is also named in a retirement receipt: ' + path)
        retained.append(dict(kind=kind, path=path, evidence=source, identity=identity,
            retired_path_conflict=conflict, existence_proof='Producer validation/commit only; not a fresh final filesystem existence check', **details))

    for arm in ARMS:
        base = f'campaign_001/arms/{arm}'
        initial = evidence.read(f'{base}/trainer/INITIAL_CHECK.json', required=True)
        add('shared_initial_state' if arm == 'K' else 'initial_reference', initial.get('state_path'),
            f'{base}/trainer/INITIAL_CHECK.json', arm=arm, producer_status=initial.get('status'))
        if arm != 'K':
            reference = evidence.read(f'{base}/trainer/initial_reference.json', required=True)
            add('referenced_shared_initial_state', reference.get('state_path'),
                f'{base}/trainer/initial_reference.json', {'sha256': reference.get('sha256')}, arm=arm)
        boundaries = []
        for source in evidence.glob(f'{base}/boundaries/event_???.json'):
            boundary = evidence.read(source)
            if boundary.get('status') == 'passed' and type(boundary.get('event')) is int:
                boundaries.append((boundary['event'], source, boundary))
        if not boundaries and last_event:
            evidence.missing.append(f'{base}/boundaries/event_{last_event:03d}.json')
        if boundaries:
            arm_event, source, boundary = max(boundaries, key=lambda row: row[0])
            add('last_committed_full_state', boundary.get('state'), source, boundary.get('state_identity'),
                arm=arm, event=arm_event, producer_status=boundary.get('status'),
                retained_capability='Complete model, Adam, counters and RNG at this committed boundary, as validated by production receipts')
        for arm_event, source, boundary in sorted(boundaries, reverse=True):
            if (boundary.get('export_status') == 'materialized' and boundary.get('checkpoint_path')
                    and boundary['checkpoint_path'].rstrip('/') + '/model.safetensors' not in paths):
                add('last_materialized_export', boundary['checkpoint_path'].rstrip('/') + '/model.safetensors',
                    source, boundary.get('weights_identity'), arm=arm, event=arm_event)
                break
    memory = evidence.read('campaign_001/MEMORY.json', required=True)
    for slot in memory.get('slots', []):
        event = slot.get('event_index')
        rid = slot.get('record_id')
        if type(event) is not int or not isinstance(rid, str) or '/' in rid:
            evidence.issues.append('Unrecognized reservoir slot identity')
            continue
        source = f'campaign_001/events/event_{event:03d}/packets/{rid}.json'
        packet = evidence.read(source, required=True)
        add('final_reservoir_packet', packet.get('binding', {}).get('path'), source,
            packet.get('identity'), record_id=rid, global_ordinal=slot.get('global_ordinal'))
    return dict(receipts=rows, retired_receipts=len(retired), distinct_retired_paths=len(paths),
        incomplete_or_other_receipts=[row['source'] for row in rows if row.get('status') != 'retired'],
        duplicate_retired_paths=duplicate_paths, categories=groups, retained=retained,
        global_events_completed=last_event, reservoir_completed_event=memory.get('completed_event'),
        notes=['Retired tensors are deleted, not archived.',
               'Receipt sizes count removed directory entries. Allocated-byte sums are nominal and are not measured reclaimed space: hardlinks and open references may retain blocks.',
               'No inode-only deduplication: receipts lack filesystem device IDs; inode reuse can also occur.',
               'Retained evidence means production validated these artifacts and no matching retirement is recorded. Compact-only retrieval does not rehash their bulk bytes or prove current remote presence.',
               'trainer/training_state.pt is a producer-maintained latest hardlink alias; it is not an additional independently retained payload.'])


def summarize(root):
    evidence = Evidence(root)
    plan = evidence.read('sealed/PLAN.json', required=True)
    status = evidence.read('campaign_001/STATUS.json', required=True)
    if not status:
        raise ValueError('No retrieved campaign STATUS.json; preparation alone is not result evidence')
    expected_events = plan.get('events', 8)
    if type(expected_events) is not int or not 1 <= expected_events <= 8:
        raise ValueError('Unexpected planned event count')
    completed = status.get('events_completed', 0)
    if type(completed) is not int or not 0 <= completed <= expected_events:
        raise ValueError('Unexpected committed event count')
    training = read_training(evidence, expected_events)
    event_rows, selection_rows = [], []
    for event in range(1, expected_events + 1):
        source = f'campaign_001/events/event_{event:03d}/COMPLETE.json'
        row = evidence.read(source, required=True)
        event_rows.append(dict(event=event, status=row.get('status'), event_wall_s=row.get('event_wall_s'), source=source))
        selection_source = f'campaign_001/events/event_{event:03d}/SELECTION.json'
        selection = evidence.read(selection_source, required=True)
        selection_rows.append(dict(event=event, source=selection_source, **selection.get('cost', {})))
    resources = read_resources(evidence, status, plan)
    retention = read_retention(evidence, completed)
    if completed == 8 and status.get('status') == 'completed':
        expected_retirements = dict(native_feature_tensor=1024, terminal_feature_tensor=1024,
                                   prepared_packet=896, predecessor_full_state=21, preceding_export=21)
        retention['complete_protocol_expected_distinct_paths'] = expected_retirements
        for name, expected in expected_retirements.items():
            actual = retention['categories'].get(name, {}).get('distinct_paths', 0)
            if actual != expected:
                evidence.issues.append(f'Retirement coverage {name}: {actual} distinct paths, expected {expected} for completed eight-event protocol')
    evaluation = evidence.read('campaign_001/final_evaluation/STATUS.json', required=True)
    source = plan.get('execution_source')
    commit = source.get('commit') if isinstance(source, dict) else source
    if commit != EXECUTION:
        evidence.issues.append('Execution source differs from schema-pinned source; review compatibility before interpreting totals')
    return dict(schema='track2-cost-resource-retention-v1',
        generated_utc=datetime.now(timezone.utc).isoformat(), run_root=str(evidence.root),
        execution_source=source, schema_execution_source=EXECUTION,
        disclosure='Prepared by the author of the additive trainer and seal. This is source-informed compact-receipt accounting, not an independent trainer review, full-state redecoding, GPU reproduction or endpoint/bootstrap audit.',
        verification_status='ANALYZED', campaign_status=status.get('status'),
        phase=status.get('phase'), events_completed=completed,
        costs=dict(recorded_components=status.get('costs', {}),
                   campaign_wall_s=status.get('campaign_wall_s'),
                   adaptation_ready_wall_s=status.get('adaptation_ready_wall_s'),
                   event_scopes=event_rows, event_scope_sum=aggregate(event_rows, 'event_wall_s'),
                   evaluation_wall_s=evaluation.get('evaluation_wall_s'),
                   evaluation_status=evaluation.get('status'),
                   evaluation_subcomponents={key: evaluation.get(key) for key in ('activation_s', 'weight_validation_s')},
                   timer_scope_notes=TIMER_NOTES,
                   additive_rule='Only sum the same timer across disjoint events/workers. Do not sum different scopes into a total; campaign wall is the outer measured total.'),
        training=training, resources=resources, retention=retention,
        selection_costs=dict(events=selection_rows,
            sequential_sums={key: aggregate(selection_rows, key) for key in ('total_wall_seconds', 'scoring_seconds', 'greedy_seconds')},
            array_working_bytes=extrema(selection_rows, 'array_working_bytes'),
            scope='Recorded CPU scoring/greedy timings nest inside selection total. Array working bytes describe explicitly counted NumPy arrays, not process RSS or measured host peak; retain each producer working_bytes_scope.'),
        evidence=dict(files=evidence.files, missing=sorted(set(evidence.missing)), issues=evidence.issues,
                      completeness='required_receipts_present_inventory_not_verified' if not evidence.missing else 'partial_receipt_coverage',
                      coverage_scope='Required fixed-path receipts and expected final retirement category counts; no authoritative retrieval inventory comparison. Absence of unindexed files cannot be ruled out.'),
        extractor=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def display(value):
    return 'unavailable' if value is None else f'{value:,.3f}' if isinstance(value, float) else str(value)


def markdown(summary):
    lines = ['# Track2 cost, resource and retention accounting', '', summary['disclosure'], '',
        f"Producer status: **{summary['campaign_status']}**, committed events: {summary['events_completed']}. "
        f"Receipt coverage: {summary['evidence']['completeness']}. This extraction is ANALYZED, not an independent GPU verification.", '',
        '## Timing', '', summary['costs']['additive_rule'], '', '| Recorded scope | Seconds |', '|---|---:|']
    for field in ('campaign_wall_s', 'adaptation_ready_wall_s', 'evaluation_wall_s'):
        lines.append(f"| {field} | {display(summary['costs'][field])} |")
    for key, value in summary['costs']['recorded_components'].items():
        lines.append(f'| {key} (component; nested) | {display(value)} |')
    lines += ['', '## Arm totals', '', summary['training']['exposure_scope'], '',
        '| Arm | Event receipts | Presentations | Replay | Prompt tokens | Response tokens | Active anchors | Adam steps | Native training s | Worker s | Export s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in summary['training']['arms'].items():
        cells = [arm, f"{row['events_with_result']}/{row['expected_events']}"]
        cells += [display(row['additive_across_events'][key]['sum']) for key in (
            'presentations', 'replay_presentations', 'prompt_tokens', 'response_tokens',
            'active_anchors', 'optimizer_steps', 'training_s', 'worker_wall_s', 'export_and_verify_s')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Each sum has observation coverage in JSON; incomplete coverage is not filled with zero.', '',
        '## Resources', '', summary['training']['allocator_scope'], '', summary['resources']['scope'], '',
        '| Discrete observation | Minimum bytes | Maximum bytes | Samples |', '|---|---:|---:|---:|']
    for key, values in summary['resources']['sampled_extrema'].items():
        lines.append(f"| {key} | {display(values['min'])} | {display(values['max'])} | {values['observations']} |")
    lines += ['', '| Arm | Maximum allocator allocated bytes | Maximum allocator reserved bytes |', '|---|---:|---:|']
    for arm, row in summary['training']['arms'].items():
        peaks = row['allocator_peaks']
        lines.append(f"| {arm} | {display(peaks['allocator_peak_allocated_bytes']['max'])} | {display(peaks['allocator_peak_reserved_bytes']['max'])} |")
    selection = summary['selection_costs']
    lines += ['', selection['scope'], '',
        '| Selection measurement | Value |', '|---|---:|']
    for key, value in selection['sequential_sums'].items():
        lines.append(f"| {key} | {display(value['sum'])} |")
    lines.append(f"| Maximum persistent array bytes | {display(selection['array_working_bytes']['max'])} |")
    lines += ['', '## Retention', '',
        f"Completed retirement receipts: {summary['retention']['retired_receipts']}; distinct retired paths: {summary['retention']['distinct_retired_paths']}.", '',
        '| Category | Paths | Nominal logical bytes | Lost capability |', '|---|---:|---:|---|']
    for category_name, row in summary['retention']['categories'].items():
        lines.append(f"| {category_name} | {row['distinct_paths']} | {display(row['nominal_logical_bytes']['sum'])} | {row['lost_capability']} |")
    lines += ['', *['- ' + note for note in summary['retention']['notes']], '',
        'Exact paths, reasons, retained evidence, state/export identities and final reservoir membership appear in the JSON and CSV companions.', '',
        '## Scope notes', '', *['- **' + key + '**: ' + value for key, value in TIMER_NOTES.items()]]
    if summary['evidence']['missing'] or summary['evidence']['issues']:
        lines += ['', '## Evidence limitations', '', *['- ' + value for value in summary['evidence']['issues']],
                  *['- Missing: ' + value for value in summary['evidence']['missing']]]
    return '\n'.join(lines) + '\n'


def write_csv(path, rows):
    fields = sorted({field for row in rows for field in row})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New local output directory; refuses overwrite')
    args = parser.parse_args()
    summary = summarize(args.run_root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'COST_RESOURCE_RETENTION.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (args.output / 'COST_RESOURCE_RETENTION.md').write_text(markdown(summary))
    write_csv(args.output / 'TRAINING_COST_ROWS.csv', summary['training']['events'])
    write_csv(args.output / 'RETIREMENT_PATHS.csv', summary['retention']['receipts'])
    write_csv(args.output / 'RETAINED_ARTIFACTS.csv', summary['retention']['retained'])
    print(json.dumps({'output': str(args.output.resolve()), 'coverage': summary['evidence']['completeness'],
                      'issues': len(summary['evidence']['issues']), 'missing': len(summary['evidence']['missing'])}))


if __name__ == '__main__':
    main()
