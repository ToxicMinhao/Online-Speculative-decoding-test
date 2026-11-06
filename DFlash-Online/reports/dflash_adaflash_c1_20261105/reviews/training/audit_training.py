"""Read-only stdlib audit of the completed A58 training evidence on A800."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ROOT = Path('/data/sources/minhao/runs/dflash_adaflash_c1_20261105/production_001')
BASE = ROOT / 'campaign_001/arms/A'
COMMIT = '681336c5e5c5caa0aa67560e46ffe83ce81e9351'
started = time.perf_counter()
checks = 0
errors = []


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        errors.append(label)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def finite(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, label + '.' + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, label + '.' + str(index))
    elif isinstance(value, (int, float)):
        check(math.isfinite(value), label + '.nonfinite')


plan = read(ROOT / 'sealed/PLAN.json')
reported = read(ROOT / 'results/TRAINING.json')
status = read(BASE / 'STATUS.json')
manifest = read(plan['manifest'])
check(plan['execution_source']['commit'] == COMMIT and not plan['execution_source']['dirty'], 'execution_source')
check(status['status'] == 'completed' and status['events_completed'] == 58 and status['tail_completed'] == 49,
      'completed_trajectory')
expected_work = dict(online_generations=7473, correctness_generations=4,
    terminal_preparation_forwards=7424, response_forwards=14848,
    response_backwards=14848, optimizer_steps=7424)
work = defaultdict(lambda: defaultdict(list))
for item in lines(BASE / 'WORK.jsonl'):
    work[item['kind']][item['phase']].append(item['ordinal'])
for kind, count in expected_work.items():
    check(set(work[kind]) == {'started', 'completed'}, 'work_phase.' + kind)
    for phase in ('started', 'completed'):
        check(work[kind][phase] == list(range(1, count + 1)), 'work_order.' + kind + '.' + phase)
    check(status['actual_work']['attempts'][kind] == count == status['actual_work']['completed'][kind],
          'work_status.' + kind)
    check(status['actual_work']['unknown'][kind] == 0, 'work_unknown.' + kind)
check(set(work) == set(expected_work), 'work_kinds')
check(reported['actual_work'] == status['actual_work'], 'report_actual_work')

pass_rows, event_rows = [], []
costs = Counter()
objective_work = Counter()
all_ids = []
all_removed = set()
retirement_counts, retirement_bytes = Counter(), Counter()
evidence_files = {}
anchors_comparison = Counter()
peaks = Counter()
collection = Counter()
preparation_count = 0
supervision_from_anchors = 0
label_mismatch_responses = 0
loss_negative_responses = 0
minimum_loss = float('inf')


def retirement(path):
    receipt = read(path)
    proposal = read(path.with_suffix('.plan.json'))
    check(receipt['status'] == 'completed', str(path) + '.completed')
    check(receipt['files'] == receipt['removed'] == proposal['files'], str(path) + '.removed_matches_plan')
    check(receipt['compact_evidence'] == proposal['compact_evidence'], str(path) + '.evidence_matches_plan')
    check(sum(item['bytes'] for item in receipt['removed']) == receipt['removed_bytes'], str(path) + '.byte_sum')
    for item in receipt['removed']:
        name = item['path']
        check(name not in all_removed and not Path(name).exists(), 'retired_unique_absent.' + name)
        all_removed.add(name)
        retirement_counts[item['kind']] += 1
        retirement_bytes[item['kind']] += item['bytes']
    for name, identity in receipt['compact_evidence'].items():
        if name in evidence_files:
            check(evidence_files[name] == identity, 'retirement_evidence_identity.' + name)
        evidence_files[name] = identity


for event in range(1, 59):
    event_id = f'event_{event:03d}'
    folder = BASE / 'events' / event_id
    trainer = BASE / 'trainer/events' / event_id
    result = read(trainer / 'RESULT.json')
    observed = lines(trainer / 'RESPONSE_OBSERVATIONS.jsonl')
    windows = lines(trainer / 'WINDOW_OBSERVATIONS.jsonl')
    receipt = read(folder / 'loop/events' / (event_id + '.json'))
    check(receipt['training'] == result, event_id + '.loop_trainer_agreement')
    check(result['status'] == 'completed' and result['method'] == 'A' and result['objective'] == 'osd', event_id + '.method')
    check(result['paper_loss'] == plan['loss'], event_id + '.loss_contract')
    check(result['start_step'] == (event - 1) * 128 and result['end_step'] == event * 128
          and result['optimizer_steps'] == 128 and result['response_presentations'] == 256, event_id + '.training_budget')
    check(result['counters'] == dict(global_step=event * 128, presentations=event * 256, completed_events=event), event_id + '.counters')
    check(len(result['records']) == len(observed) == 256 and len(windows) == 128, event_id + '.observations')
    features = {feature['record_id']: feature for feature in result['features']}
    expected_ids = [row['request_id'] for row in manifest['train_requests'][(event - 1) * 128:event * 128]]
    check(list(features) == expected_ids, event_id + '.feedback_order')
    all_ids.extend(features)
    boundary = read(BASE / 'boundaries' / (event_id + '.json'))
    scientific = boundary['scientific_check']
    check(boundary['status'] == 'passed' and boundary['event'] == event and boundary['arm'] == 'A', event_id + '.boundary')
    check(scientific['status'] == 'passed' and scientific['global_step'] == event * 128
          and scientific['presentations'] == event * 256 and scientific['adam_state_entries'] == 58
          and scientific['export_exact'] and scientific['live']['exact_resume']
          and scientific['live']['mismatches'] == [] and scientific['extra_optimizer_steps'] == 0, event_id + '.scientific_boundary_receipt')
    resources = read(folder / 'RESOURCE_COSTS.json')
    check(all(resources[name] == 0 for name in ('extra_forwards', 'extra_backwards', 'extra_optimizer_steps')), event_id + '.passive_resources')
    for name in ('training_peak_allocated_bytes', 'training_peak_reserved_bytes'):
        peaks[name] = max(peaks[name], resources[name])
    check(reported['resources'][event - 1] == dict(event=event, **resources), event_id + '.reported_resources')
    obj = result['objective_work']
    check(obj['reference_operations'] == 0 and all(obj[name] == 0 for name in
          ('reference_setup_s', 'reference_forward_s', 'reference_head_s', 'projection_s')), event_id + '.no_extra_mechanism')
    check(obj['work']['response_forwards'] == 256 and obj['work']['draft_forwards'] == 256,
          event_id + '.actual_forward_count')
    objective_work.update(obj['work'])
    costs['teacher_head_s_nested_training'] += obj['teacher_head_s']
    check(reported['objective_work'][event - 1] == dict(event=event, **obj), event_id + '.reported_objective_work')
    comparison = read(folder / 'ORIGINAL_ANCHOR_COMPARISON.json')
    check(comparison['status'] == 'compared' and comparison['forced'] is False, event_id + '.natural_anchors')
    for name in ('presentations', 'identity_matches', 'exact_anchor_matches', 'presentation_order_matches'):
        anchors_comparison[name] += comparison[name]

    for ordinal, (row, observation) in enumerate(zip(result['records'], observed)):
        label = event_id + '.' + str(ordinal)
        check(row['response_observation'] == observation, label + '.observation_exact')
        check(observation['scope'] == 'one_original_response_forward' and observation['presentation_ordinal'] == ordinal,
              label + '.observation_scope')
        check(row['pass'] == observation['pass_index'] == ordinal // 128 and row['record_id'] == observation['record_id'], label + '.identity')
        check(row['sampled_anchors'] == observation['sampled_anchors'], label + '.anchors')
        check(row['optimizer_stepped'] == (ordinal % 2 == 1)
              and row['optimizer_step_after'] == (event - 1) * 128 + (ordinal + 1) // 2, label + '.Adam_counters')
        check(row['learning_rate'] == (3e-4 if row['optimizer_stepped'] else None), label + '.learning_rate')
        feature = features[row['record_id']]
        active = [position for draw in row['sampled_anchors'] for position, keep in
                  zip(draw['positions'][0], draw['keep_mask'][0]) if keep]
        check(len(active) == len(set(active)) == row['active_anchors'] == min(512, feature['response_tokens'] - 1), label + '.anchor_count')
        check(all(feature['prompt_length'] <= anchor <= feature['length'] - 2 for anchor in active), label + '.response_only_anchors')
        positions = sum(min(15, feature['length'] - anchor - 1) for anchor in active)
        sums = observation['sum_metrics']
        check(sums['adaflash_available_positions'] == sums['supervised_position_exposure'] == positions, label + '.supervised_positions')
        check(sums['adaflash_eligible_entries'] == positions * 151936, label + '.vocabulary_denominator')
        check(sums['adaflash_response_count'] == 1, label + '.response_count_once')
        check(0 <= sums['adaflash_clipped_entries'] <= sums['adaflash_eligible_entries'] and
              0 <= sums['adaflash_negative_entries'] <= sums['adaflash_eligible_entries'] and
              0 <= sums['adaflash_clipped_positions'] <= positions and
              0 <= sums['adaflash_teacher_argmax_canonical_mismatches'] <= positions, label + '.clipping_bounds')
        check(math.isclose(observation['loss'], sums['adaflash_loss_numerator'] / sums['adaflash_loss_denominator'], rel_tol=1e-6, abs_tol=1e-7), label + '.loss_reduction')
        finite(observation, label)
        supervision_from_anchors += positions
        label_mismatch_responses += sums['adaflash_teacher_argmax_canonical_mismatches'] > 0
        loss_negative_responses += observation['loss'] < 0
        minimum_loss = min(minimum_loss, observation['loss'])

    for pass_index in (0, 1):
        records = result['records'][128 * pass_index:128 * (pass_index + 1)]
        check(sorted(row['record_id'] for row in records) == sorted(features), event_id + '.pass_coverage.' + str(pass_index))
        sums = [row['response_observation']['sum_metrics'] for row in records]
        row = dict(event=event, pass_index=pass_index, presentations=128,
            prompt_tokens=sum(item['prompt_tokens'] for item in records),
            response_tokens=sum(item['response_tokens'] for item in records),
            active_anchors=sum(item['active_anchors'] for item in records))
        row.update({key: sum(value[key] for value in sums) for key in sums[0]})
        check(row == reported['pass_rows'][(event - 1) * 2 + pass_index], event_id + '.published_pass_exposure.' + str(pass_index))
        pass_rows.append(row)
    for index, window in enumerate(windows):
        pair = observed[2 * index:2 * index + 2]
        check(window['global_step'] == (event - 1) * 128 + index + 1 and window['window'] == index + 1
              and window['record_ids'] == [row['record_id'] for row in pair] and window['learning_rate'] == 3e-4,
              event_id + '.optimizer_window.' + str(index))
        check(window['mean_response_loss'] == sum(row['loss'] for row in pair) / 2, event_id + '.separate_response_mean.' + str(index))
        finite(window, event_id + '.window')

    saved_cost = status['event_costs'][event - 1]['costs']
    check(status['event_costs'][event - 1] == reported['event_costs'][event - 1], event_id + '.reported_event_cost')
    for name, value in result['costs'].items():
        check(saved_cost[name] == value, event_id + '.trainer_cost.' + name)
        costs[name] += value
    check(result['costs']['training_s'] == sum(row['training_s'] for row in result['records']), event_id + '.training_time_from_presentations')
    preparation = receipt['preparation']
    check([item['record_id'] for item in preparation] == expected_ids, event_id + '.prepared_order')
    for name in ('teacher_preparation_s', 'feature_persistence_s'):
        total = sum(item[name] for item in preparation)
        check(total == saved_cost[name], event_id + '.prepare_cost.' + name)
        costs[name] += total
    for name in ('update_call_s', 'publication_validation_s', 'retention_s'):
        costs[name] += saved_cost[name]
    check(saved_cost['activation'] == receipt['activation'], event_id + '.activation_receipt')
    for name in ('activation_s', 'native_activation_s'):
        costs[name] += saved_cost['activation'][name]
    finite(saved_cost, event_id + '.costs')
    for prepared in preparation:
        check(prepared['status'] == 'prepared' and prepared['response_only_loss'] and
              prepared['native_capture_validated'] and prepared['label_mismatches'] == [], event_id + '.teacher_binding.' + prepared['record_id'])
        feature = features[prepared['record_id']]
        source = read(folder / 'loop/serving' / (prepared['record_id'] + '.json'))
        request = manifest['train_requests'][(event - 1) * 128 + expected_ids.index(prepared['record_id'])]
        expected_checkpoint = plan['draft'] if event == 1 else str(BASE / 'trainer/events' / f'event_{event - 1:03d}' / 'export')
        check(source['record_id'] == request['request_id'] and source['prompt_ids'] == request['prompt_ids']
              and source['checkpoint_id'] == expected_checkpoint and source['purpose'] == 'serving', event_id + '.own_causal_response')
        check(prepared['input_ids'] == source['prompt_ids'] + source['response_ids'] and
              prepared['target_argmax_ids'] == source['response_ids'], event_id + '.exact_saved_training_tokens')
        check(source['metadata']['capture']['collect_features'] is True and source['max_new_tokens'] == 2048,
              event_id + '.feedback_capture_enabled')
        for name in ('request_latency_s', 'native_generate_s'):
            collection[name] += source['metadata'][name]
        for name in ('capture_host_s', 'feature_persistence_s', 'cache_maintenance_s'):
            collection[name] += source['metadata']['capture'][name]
        collection['responses'] += 1
        collection['cap_hits'] += source['finish_reason'] == 'length'
        collection['output_tokens'] += len(source['response_ids'])
        check(prepared['prompt_length'] == feature['prompt_length'] and prepared['response_length'] == feature['response_tokens'], event_id + '.prepared_shape')
        check(prepared['terminal_feature']['target_forward_count'] == 1, event_id + '.terminal_forward')
        equivalence = read(folder / 'preparation_retirement' / (prepared['record_id'] + '.equivalence.json'))
        check(equivalence['status'] == 'passed' and equivalence['exact_decoded_rows'] and equivalence['additional_model_forwards'] == 0, event_id + '.feature_equivalence_receipt')
        retirement(folder / 'preparation_retirement' / (prepared['record_id'] + '.json'))
        preparation_count += 1
    retirement(folder / 'RETIREMENT.json')
    event_rows.append(dict(event=event, start_step=result['start_step'], end_step=result['end_step'],
        training_s=result['costs']['training_s'], **{name: saved_cost[name] for name in
        ('teacher_preparation_s', 'feature_persistence_s', 'update_call_s', 'publication_validation_s', 'retention_s')},
        activation_s=saved_cost['activation']['activation_s'], **{key: resources[key] for key in peaks}))

check(len(set(all_ids)) == len(all_ids) == 7424, 'distinct_training_feedback')
totals = {key: sum(row[key] for row in pass_rows) for key in pass_rows[0] if key not in ('event', 'pass_index')}
check(totals == reported['exposure'], 'published_exposure_totals')
check(supervision_from_anchors == totals['supervised_position_exposure'], 'anchor_reconstructed_exposure')
tail = [read(path) for path in sorted((BASE / 'tail').glob('*.json'))]
check(len(tail) == 49 and [row['record_id'] for row in tail] == [row['request_id'] for row in manifest['train_requests'][7424:]], 'tail_identity')
for row in tail:
    check(row['checkpoint_id'] == status['final_checkpoint'] and row['metadata']['capture']['collect_features'] is False, 'tail_no_learning')
qualification = read(BASE / 'LOSS_QUALIFICATION.json')
check(qualification['status'] == 'passed' and qualification['work'] == dict(synthetic_loss_forwards=12,
    synthetic_loss_backwards=12, model_forwards=0, optimizer_steps=0, rng_draws=0), 'synthetic_qualification_work')
for name in ('INITIAL_CHECK.json',):
    check(read(BASE / 'trainer' / name)['status'] == 'passed', name)
check(read(BASE / 'FINAL_STATE_CHECK.json')['status'] == 'passed', 'final_state_receipt')
initial = BASE / 'trainer/initial_state.pt'
final = BASE / 'trainer/events/event_058/training_state.pt'
export = BASE / 'trainer/events/event_058/export/model.safetensors'
check(initial.is_file() and final.is_file() and export.is_file(), 'retained_initial_final')
check(retirement_counts == dict(native_capture=7424, terminal_feature=7424, prepared_feature=7424,
    previous_state=57, previous_export=57), 'retirement_counts')
# Authenticate compact evidence once per unique file. No model/state tensor is loaded.
for name, identity in evidence_files.items():
    path = Path(name)
    check(path.is_file(), 'retained_evidence_exists.' + name)
    if path.is_file():
        data = path.read_bytes()
        check(len(data) == identity['bytes'] and hashlib.sha256(data).hexdigest() == identity['sha256'], 'retained_evidence_identity.' + name)

summary = dict(status='passed' if not errors else 'failed', execution_commit=COMMIT,
    audit='Independent stdlib reconstruction of saved records; authorship overlap disclosed.',
    utc=datetime.now(timezone.utc).isoformat(), checks=checks, errors=errors, actual_work=expected_work,
    exposure=totals, supervised_occurrences_reconstructed_from_anchors=supervision_from_anchors,
    clipping_entry_rate=totals['adaflash_clipped_entries'] / totals['adaflash_eligible_entries'],
    clipping_position_rate=totals['adaflash_clipped_positions'] / totals['adaflash_available_positions'],
    teacher_argmax_canonical_mismatch_responses=label_mismatch_responses,
    teacher_argmax_canonical_mismatch_rate=totals['adaflash_teacher_argmax_canonical_mismatches'] / totals['adaflash_available_positions'],
    minimum_response_loss=minimum_loss, negative_response_losses=loss_negative_responses,
    costs_s=dict(costs), cost_scope='Timers may nest; teacher head is within training; update_call contains multiple components.',
    feedback_collection=dict(collection),
    worker_wall_s=status['worker_wall_s'], backend_loading_s=status['backend_loading_s'], trainer_loading_s=status['trainer_loading_s'],
    initial_validation_s=status['initial_validation_s'], peaks_bytes=dict(peaks), objective_work=dict(objective_work),
    natural_anchor_comparison=dict(anchors_comparison), event_rows=event_rows,
    retirement=dict(counts=dict(retirement_counts), logical_bytes=dict(retirement_bytes),
        total_paths=len(all_removed), compact_evidence_files=len(evidence_files),
        retained_initial_final=True, lost='All feature tensors and intermediate event1..57 optimizer states/exports were retired, not archived. Exact intermediate tensor replay/resume is unavailable.'),
    limitations=['No new model forwards, optimizer steps, generation or tensor loading.',
        'Full tensor/Adam/RNG equality relies on saved qualification and boundary receipts; this audit independently reconstructs accounting and compact evidence.',
        'Clipping event counts inherit native FP32 logging; loss/weights were not recomputed from retired feature tensors.',
        'No historical training timing is a contemporaneous matched-cost comparison.'],
    runtime_s=time.perf_counter() - started, free_bytes_after=shutil.disk_usage(ROOT).free)
print(json.dumps(summary, indent=2, allow_nan=False))
