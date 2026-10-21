"""Bind retained evidence and seal outcome-independent native block locations.

This module performs file reads and ordinary Python validation only. It never
loads a model, generates missing trajectories or changes historical evidence.
Plans contain private token IDs and must be saved outside the public checkout.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.dflash_baseline_learning_curve.experiment import digest, require
from experiments.dflash_kl_native_diagnostic.evidence import checked_file, file_binding
from experiments.dflash_lower_lr_control.evidence import load_evidence as load_references
from experiments.dflash_lower_lr_control.experiment import validate_pair, validate_training

HERE = Path(__file__).resolve().parent
ARMS = ('F', 'S8', 'L8')
MAX_LOCATIONS, MAX_BLOCKS = 240, 720
BLOCK_SIZE, CONTEXT_LIMIT = 10, 32768
SAMPLE_MANIFEST_SHA256 = '1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d'
LOWER_LR_COMMIT = '673e75efb72595f718b41c03fb2c7aa0e71fc40d'
SELECTION = dict(source='F length ascending; original manifest order tiebreak; second of four',
                 rounds='floor(i*(n-1)/(min(n,8)-1)); all when n<=1',
                 max_locations_per_request=8, outcome_dependent=False)
RECONSTRUCTION_MODE = 'recorded_target_verification_schedule'


def _json(path):
    return json.loads(Path(path).read_text())


def remap_trace(root, original_root, row):
    """Resolve only that group's retained trace and verify bytes before reading."""
    root, original_root = Path(root).resolve(), Path(original_root)
    original = Path(row['trajectory']['path'])
    require(original.is_relative_to(original_root / 'native' / row['group_id']),
            'Retained trace belongs to another root/group')
    local = (root / original.relative_to(original_root)).resolve()
    require(local.is_relative_to(root), 'Retained trace escapes local evidence root')
    binding = checked_file(local, row['trajectory'])
    return _json(local), dict(original=copy.deepcopy(row['trajectory']), local=binding)


def _termination(row, eos_ids, output_cap):
    output, finish = row['output_ids'], row['finish_reason']
    require(isinstance(output, list) and output, 'Retained response is empty')
    require(type(row['cap_hit']) is bool and type(row['eos_hit']) is bool,
            'Missing retained EOS/cap flags')
    require(len(output) <= output_cap and row['cap_hit'] == (len(output) == output_cap),
            'Saved output-length cap flag differs')
    if finish.get('type') == 'stop':
        require(row['eos_hit'] and finish.get('matched') in eos_ids and
                output[-1] == finish['matched'], 'Saved EOS termination differs')
        boundary = 'eos'
    else:
        require(finish == dict(type='length', length=output_cap) and row['cap_hit'] and
                not row['eos_hit'] and len(output) == output_cap, 'Saved cap termination differs')
        boundary = 'cap'
    require(not any(token in eos_ids for token in output[:-1]) and
            (boundary == 'eos' or output[-1] not in eos_ids), 'Tokens occur beyond saved EOS')
    return dict(finish_reason=copy.deepcopy(finish), cap_hit=row['cap_hit'], eos_hit=row['eos_hit'],
                response_boundary=boundary, output_cap=output_cap, output_tokens=len(output))


def frozen_prefill(document, row):
    """Bind the original prompt and its first output, not a cold saved prefix."""
    events = [event for event in document['events'] if event['kind'] == 'target_prefill']
    require(len(events) == 1 and document['events'][0] == events[0],
            'Exactly one original target-prefill emission must precede verification')
    event = events[0]
    require(row['initial_output_tokens'] == 1 and event['output_offset'] == 0 and
            event['position'] == len(row['input_ids']) and
            event['committed_token_ids'] == row['output_ids'][:1] and
            event['committed_provenance'] == ['target_prefill'],
            'Original prefill prompt/seed identity differs')
    return dict(input_ids_sha256=digest(row['input_ids']), prompt_tokens=len(row['input_ids']),
                seed_token_id=row['output_ids'][0], event_sha256=digest(event))


def frozen_rounds(document, row):
    """Require complete raw nine-proposal records, including clipped terminal rounds."""
    from experiments.dflash_baseline_learning_curve.native import normalize_rounds
    normalized, initial = normalize_rounds(document, row, row['input_ids'])
    require(normalized == row['rounds'] and initial == row['initial_output_tokens'],
            'Normalized frozen rounds changed')
    require(document['finish_reason'] == row['finish_reason'], 'Raw trace termination differs')
    frozen_prefill(document, row)
    events = [event for event in document['events'] if event['kind'] == 'verification']
    require(len(events) == len(normalized), 'Missing raw frozen proposal records')
    starts = []
    for index, event in enumerate(events):
        start, proposals = event['output_offset'], event['draft_proposed_token_ids']
        require(type(start) is int and 1 <= start < len(row['output_ids']), 'Invalid frozen output offset')
        require(isinstance(proposals, list) and len(proposals) == 9 and
                all(type(token) is int and token >= 0 for token in proposals),
                'Complete nine-token frozen proposal record required')
        absolute = len(row['input_ids']) + start
        require(event['verification_input_token_ids'] == [row['output_ids'][start-1]] + proposals,
                'Raw frozen seed/proposal alignment differs')
        require(event['verification_input_positions'] == list(range(absolute-1, absolute+9)) and
                event['draft_proposed_positions'] == list(range(absolute, absolute+9)),
                'Raw frozen absolute positions differ')
        require(event['context']['input_ids_sha256'] == digest(row['input_ids']) and
                event['context']['committed_output_prefix_length'] == start and
                event['context']['verification_input_start_position'] == absolute-1,
                'Raw frozen canonical context differs')
        require(event['round_index'] == index, 'Raw frozen round order differs')
        predictions = event['target_predicted_token_ids']
        require(isinstance(predictions, list) and len(predictions) == BLOCK_SIZE and
                all(type(token) is int and token >= 0 for token in predictions) and
                event['target_prediction_positions'] == list(range(absolute, absolute+BLOCK_SIZE)),
                'Complete original target prediction IDs/positions required')
        accepted = next((i for i, (proposal, target) in enumerate(zip(proposals, predictions))
                         if proposal != target), 9)
        require(event['native_acceptance_length_before_stop'] == accepted and
                event['rejection_boundary_proposal_index'] == (None if accepted == 9 else accepted),
                'Original acceptance differs from recorded target predictions')
        require(type(event['stop_clipped']) is bool and type(event['finished']) is bool,
                'Original verification stop state missing')
        if index < len(events)-1:
            require(event['stop_clipped'] is False and event['finished'] is False and
                    event['committed_token_ids'] == proposals[:accepted] + [predictions[accepted]],
                    'Intermediate original verification must commit its full accepted prefix and next seed')
        starts.append(start)
    require(len(starts) == len(set(starts)), 'Duplicate frozen block starts')
    return sorted(copy.deepcopy(events), key=lambda event: event['output_offset'])


def load_evidence(baseline_root, diagnostic_root, lower_lr_root):
    """Load exact PR9/10/11 compact mirrors; tensors remain an execution-time gate."""
    bundle = load_references(baseline_root, diagnostic_root)
    root, pins = Path(lower_lr_root).resolve(), _json(HERE / 'reference_bindings.json')
    require(pins['lower_lr_commit'] == LOWER_LR_COMMIT and
            pins['sample_manifest_sha256'] == SAMPLE_MANIFEST_SHA256 and
            bundle['bindings']['files']['SAMPLE_MANIFEST.json']['sha256'] == SAMPLE_MANIFEST_SHA256,
            'Required PR11 revision/sealed sample manifest differs')
    files = {name: checked_file(root / name, pin) for name, pin in pins['files'].items()}
    result, source, config = (_json(root / name) for name in
                             ('RESULT.json', 'SOURCE_MANIFEST.json', 'RUNTIME_CONFIG.json'))
    require(source['git']['commit'] == LOWER_LR_COMMIT and source['git']['dirty'] is False,
            'PR11 executed revision differs')
    require(result['status'] == 'complete' and result['unexecuted_groups'] == [] and
            (result['optimizer_steps'], result['training_presentations'], result['actual_native_generations'],
             result['actual_response_kl_evaluations'], result['explicit_warmups']) == (48, 96, 48, 48, 0),
            'Completed bounded PR11 evidence required')
    for key, value in bundle['runtime_config'].items():
        if key != 'output_root':
            require(config[key] == value, 'PR9/10/11 runtime configuration differs: ' + key)
    require([g['group_id'] for g in result['groups']] == [g['group_id'] for g in bundle['groups']],
            'PR11 group identities/order differ')
    baseline_pins = _json(HERE.parent / 'dflash_kl_native_diagnostic/baseline_bindings.json')
    traces, runtime = [], None
    for group, prior in zip(bundle['groups'], result['groups']):
        require(prior['status'] == 'complete' and prior['domain'] == group['domain'] and
                prior['training_validation']['status'] == 'passed' and
                prior['trainer_stop']['training_quiescent'] is True, 'PR11 group incomplete/invalid')
        validate_training(prior['training'], group)
        checkpoint = copy.deepcopy(prior['checkpoint'])
        expected_path = Path(pins['original_root']) / 'native' / group['group_id'] / 'checkpoints/L8'
        require(checkpoint == prior['training']['checkpoint'] and Path(checkpoint['path']) == expected_path,
                'PR11 checkpoint identity/path differs')
        group['checkpoints'] = {arm: group['checkpoints'][arm] for arm in ('F', 'S8')}
        group['checkpoints']['L8'] = checkpoint
        require(len({v['checkpoint_identity_sha256'] for v in group['checkpoints'].values()}) == 3,
                'Three independently identified checkpoints required')
        native_config = _json(root / 'native' / group['group_id'] / 'NATIVE_CONFIG.json')
        for key in config:
            require(native_config[key] == config[key], 'PR11 native runtime differs: ' + key)
        native_runtime = {key: native_config[key] for key in
                          ('target', 'released', 'tokenizer_path', 'effective_limits')}
        if runtime is None:
            runtime = native_runtime
        require(native_runtime == runtime and native_config['episode_id'] == group['group_id'] and
                native_config['released'] == group['checkpoints']['F'], 'Native model/group runtime differs')
        target = runtime['target']['identity']
        require(target['dtype'] == 'bfloat16' and target['target_frozen'] is True and
                target['artifact']['repo_id'] == 'meta-llama/Llama-3.1-8B-Instruct' and
                runtime['effective_limits']['total_context_tokens'] == CONTEXT_LIMIT and
                runtime['effective_limits']['proposal_block_tokens'] == BLOCK_SIZE and
                runtime['effective_limits']['output_tokens'] == 1024, 'Qualified native contract differs')
        eos_ids = target['stop_token_ids']
        require(isinstance(eos_ids, list) and eos_ids and len(set(eos_ids)) == len(eos_ids) and
                all(type(token) is int and token >= 0 for token in eos_ids), 'Missing exact EOS definitions')
        group['eos_ids'], group['frozen_trajectories'] = list(eos_ids), {}
        for population in ('source', 'heldout'):
            originals, pairs = group['reference_pairs'][population], prior[population]
            require(len(pairs) == 4 and [p['request_id'] for p in pairs] ==
                    [p['request_id'] for p in originals], 'PR11 response population/order differs')
            for pair, old in zip(pairs, originals):
                require(validate_pair(pair)['status'] == 'passed' and pair['validation']['status'] == 'passed',
                        'Invalid retained F/S8/L8 comparison')
                require(all(pair[key] == old[key] for key in
                            ('group_id', 'domain', 'population', 'request_id', 'F', 'S8')),
                        'Retained PR9/10 comparison changed in PR11')
                require(pair['checkpoint_bindings'] == {arm: group['checkpoints'][arm]['checkpoint_identity_sha256']
                                                       for arm in ARMS}, 'Wrong group checkpoint binding')
                for arm in ARMS:
                    _termination(pair[arm]['native'], eos_ids, 1024)
                document, binding = remap_trace(root, pins['original_root'], pair['L8']['native'])
                from experiments.dflash_baseline_learning_curve.native import normalize_rounds
                rounds, initial = normalize_rounds(document, pair['L8']['native'], pair['L8']['native']['input_ids'])
                require(rounds == pair['L8']['native']['rounds'] and initial == pair['L8']['native']['initial_output_tokens']
                        and document['finish_reason'] == pair['L8']['native']['finish_reason'] and
                        document['max_new_tokens'] == 1024, 'PR11 trace/termination/limit differs')
                traces.append(binding)
                frozen, frozen_binding = remap_trace(baseline_root, baseline_pins['original_root'], pair['F']['native'])
                group['frozen_trajectories'][pair['request_id']] = dict(
                    events=frozen_rounds(frozen, pair['F']['native']),
                    prefill=frozen_prefill(frozen, pair['F']['native']), binding=frozen_binding)
            group['reference_pairs'][population] = copy.deepcopy(pairs)
    bundle['native_runtime'], bundle['runtime_config'] = copy.deepcopy(runtime), config
    bundle['bindings']['lower_lr'] = dict(files=files, trajectories=traces, lower_lr_commit=LOWER_LR_COMMIT,
                                         complete_groups=6, pairs=48)
    return bundle


def selected_round_indices(n, maximum=8):
    require(type(n) is int and n >= 0 and maximum == 8, 'Keep the fixed up-to-eight round sampler')
    if n <= 1:
        return list(range(n))
    k = min(n, maximum)
    return [i * (n-1) // (k-1) for i in range(k)]


def select_source_control(feedback):
    require(len(feedback) == 4 and len({r['request_id'] for r in feedback}) == 4,
            'Four distinct original source requests required')
    return sorted(enumerate(feedback), key=lambda item: (len(item[1]['output_ids']), item[0]))[1][1]


def retained_measurements(pair):
    return {arm: dict(verification_calls=pair[arm]['native']['verification_calls'],
                     output_tokens=len(pair[arm]['native']['output_ids']),
                     accepted_draft_tokens=pair[arm]['native']['accepted_draft_tokens'],
                     retained_response_kl=pair[arm]['kl']['loss'],
                     finish_reason=copy.deepcopy(pair[arm]['native']['finish_reason']),
                     cap_hit=pair[arm]['native']['cap_hit'], eos_hit=pair[arm]['native']['eos_hit'],
                     checkpoint_identity_sha256=pair['checkpoint_bindings'][arm]) for arm in ARMS}


def reconstruction_digest(reconstruction):
    return digest({key: value for key, value in reconstruction.items() if key != 'schedule_sha256'})


def validate_reconstruction(anchor):
    """Validate every cache-building round before the selected measured block.

    The original target forward schedule is fixed by the hashed native trace.
    Its saved proposals are replay inputs, never newly generated draft blocks.
    """
    reconstruction = anchor['reconstruction']
    require(reconstruction['mode'] == RECONSTRUCTION_MODE and
            reconstruction['schedule_sha256'] == reconstruction_digest(reconstruction),
            'Reconstruction mode/schedule identity differs')
    ids, output = anchor['input_ids'], anchor['output_ids']
    prefill, rounds = reconstruction['prefill'], reconstruction['rounds']
    require(prefill['input_ids_sha256'] == digest(ids) and prefill['prompt_tokens'] == len(ids) and
            prefill['seed_token_id'] == output[0] and isinstance(prefill['event_sha256'], str) and
            len(prefill['event_sha256']) == 64, 'Original prompt-prefill identity differs')
    require(isinstance(rounds, list) and reconstruction['target_verification_count'] == len(rounds) ==
            anchor['retained_round_index'], 'Reconstruction target-work count/selected round differs')
    cursor = 1
    for index, event in enumerate(rounds):
        proposals, predictions = event['draft_proposed_token_ids'], event['target_predicted_token_ids']
        require(event['round_index'] == index and event['output_offset'] == cursor,
                'Reconstruction rounds are not original and contiguous')
        require(isinstance(proposals, list) and len(proposals) == 9 and
                isinstance(predictions, list) and len(predictions) == BLOCK_SIZE and
                all(type(token) is int and token >= 0 for token in proposals+predictions),
                'Reconstruction requires all original proposals and target predictions')
        absolute = len(ids)+cursor
        require(event['verification_input_token_ids'] == [output[cursor-1]]+proposals and
                event['verification_input_positions'] == list(range(absolute-1, absolute+9)) and
                event['target_prediction_positions'] == list(range(absolute, absolute+BLOCK_SIZE)),
                'Reconstruction seed/input/absolute positions differ')
        accepted = next((i for i, (proposal, target) in enumerate(zip(proposals, predictions))
                         if proposal != target), 9)
        committed = proposals[:accepted]+[predictions[accepted]]
        require(type(event['commit_length']) is int and event['commit_length'] == len(committed) and
                event['committed_token_ids'] == committed == output[cursor:cursor+len(committed)] and
                event['next_seed_token_id'] == committed[-1] and
                event['native_acceptance_length_before_stop'] == accepted,
                'Reconstruction commit length/tokens/next seed differ')
        require(event['stop_clipped'] is False and event['finished'] is False and
                isinstance(event['raw_event_sha256'], str) and len(event['raw_event_sha256']) == 64,
                'Reconstruction must use complete nonfinal retained rounds')
        cursor += len(committed)
        require(cursor <= anchor['block_output_start'], 'Reconstruction includes the measured block or later feedback')
    require(cursor == anchor['block_output_start'] and output[cursor-1] == anchor['seed_token_id'],
            'Reconstruction does not reach the selected canonical prefix/seed exactly')
    return reconstruction


def _reconstruction(trajectory, selected_index):
    rounds = []
    for event in trajectory['events'][:selected_index]:
        record = {key: copy.deepcopy(event[key]) for key in
                  ('round_index', 'output_offset', 'verification_input_token_ids',
                   'verification_input_positions', 'draft_proposed_token_ids', 'target_predicted_token_ids',
                   'target_prediction_positions', 'committed_token_ids',
                   'native_acceptance_length_before_stop', 'stop_clipped', 'finished')}
        record.update(commit_length=len(event['committed_token_ids']),
                      next_seed_token_id=event['committed_token_ids'][-1], raw_event_sha256=digest(event))
        rounds.append(record)
    reconstruction = dict(mode=RECONSTRUCTION_MODE, prefill=copy.deepcopy(trajectory['prefill']),
                          rounds=rounds, target_verification_count=len(rounds))
    reconstruction['schedule_sha256'] = reconstruction_digest(reconstruction)
    return reconstruction


def validate_anchor(anchor):
    ids, output, start = anchor['input_ids'], anchor['output_ids'], anchor['block_output_start']
    require(isinstance(ids, list) and ids and isinstance(output, list) and output and
            all(type(token) is int and token >= 0 for token in ids + output), 'Exact nonempty token IDs required')
    require(type(start) is int and 1 <= start < len(output), 'Block starts after its known seed')
    require(anchor['canonical_prefix_ids'] == ids + output[:start] and
            anchor['canonical_prefix_sha256'] == digest(ids + output[:start]) and
            anchor['seed_token_id'] == output[start-1] and
            anchor['canonical_next_ids'] == output[start:start+9], 'Canonical prefix/seed alignment differs')
    require(anchor['expected_absolute_positions'] == list(range(len(ids)+start, len(ids)+start+9)),
            'Absolute proposal positions differ')
    proposals = anchor['expected_frozen_proposal_ids']
    require(isinstance(proposals, list) and len(proposals) == 9 and
            all(type(token) is int and token >= 0 for token in proposals), 'All nine frozen proposals required')
    require(0 < len(anchor['canonical_prefix_ids'])-1 and
            len(anchor['canonical_prefix_ids'])-1+BLOCK_SIZE <= CONTEXT_LIMIT, 'Native context exceeds qualified limit')
    require(anchor['population'] in ('source', 'heldout') and set(anchor['checkpoints']) == set(ARMS),
            'Wrong request population or checkpoint arms')
    for arm, checkpoint in anchor['checkpoints'].items():
        require(checkpoint['checkpoint_identity_sha256'] == digest(checkpoint['files']), 'Checkpoint identity differs')
        require(checkpoint['checkpoint_id'] == {'F': 'F', 'S8': 'U2', 'L8': 'L8'}[arm], 'Wrong checkpoint label')
    termination = _termination(anchor, anchor['eos_ids'], anchor['output_cap'])
    require(all(anchor[key] == value for key, value in termination.items()), 'Termination metadata differs')
    require(anchor['reaches_response_boundary'] == (start+len(anchor['canonical_next_ids']) == len(output)) and
            anchor['censored_suffix_positions'] == 9-len(anchor['canonical_next_ids']) and
            anchor['suffix_censored'] == (len(anchor['canonical_next_ids']) < 9), 'Censoring metadata differs')
    validate_reconstruction(anchor)
    return dict(status='passed', errors=[])


def _make_anchor(group, pair, event, trajectory):
    row, start = pair['F']['native'], event['output_offset']
    output, ids = row['output_ids'], row['input_ids']
    anchor = dict(anchor_id=f"{group['group_id']}/{pair['population']}/{pair['request_id']}/{start}",
                  group_id=group['group_id'], domain=group['domain'], population=pair['population'],
                  request_id=pair['request_id'], source_control=pair['population'] == 'source',
                  input_ids=list(ids), output_ids=list(output), block_output_start=start,
                  canonical_prefix_ids=ids+output[:start], canonical_prefix_sha256=digest(ids+output[:start]),
                  seed_token_id=output[start-1],
                  expected_frozen_proposal_ids=list(event['draft_proposed_token_ids']),
                  canonical_next_ids=output[start:start+9], expected_absolute_positions=list(event['draft_proposed_positions']),
                  eos_ids=list(group['eos_ids']), checkpoints=copy.deepcopy(group['checkpoints']),
                  retained_round_index=event['round_index'], frozen_raw_event_sha256=digest(event),
                  reconstruction=_reconstruction(trajectory, event['round_index']),
                  frozen_trajectory=copy.deepcopy(trajectory['binding']),
                  reaches_response_boundary=start+min(9, len(output)-start) == len(output),
                  censored_suffix_positions=max(0, 9-(len(output)-start)), suffix_censored=len(output)-start < 9,
                  **_termination(row, group['eos_ids'], 1024))
    validate_anchor(anchor)
    return anchor


def plan_digest(plan):
    return digest({key: value for key, value in plan.items() if key != 'plan_sha256'})


def build_plan(bundle):
    plan = dict(schema_version=2, scope='read_only_matched_prefix', arms=list(ARMS), groups=[],
                reconstruction_mode=RECONSTRUCTION_MODE,
                evaluation_order='group_checkpoint_then_location', arm_order=list(ARMS),
                sample_manifest_sha256=SAMPLE_MANIFEST_SHA256, lower_lr_commit=LOWER_LR_COMMIT,
                evidence_bindings=copy.deepcopy(bundle['bindings']),
                native_runtime=copy.deepcopy(bundle['native_runtime']),
                runtime_config_sha256=digest(bundle['runtime_config']),
                selection=copy.deepcopy(SELECTION),
                maximum_locations=MAX_LOCATIONS, maximum_block_evaluations=MAX_BLOCKS,
                first_group_qualification_included=True, extra_warmups=0, automatic_retries=0)
    for group in bundle['groups']:
        chosen = select_source_control(group['feedback'])['request_id']
        pairs = group['reference_pairs']['heldout'] + [p for p in group['reference_pairs']['source']
                                                     if p['request_id'] == chosen]
        current = dict(group_id=group['group_id'], domain=group['domain'],
                       checkpoints=copy.deepcopy(group['checkpoints']), requests=[], anchors=[],
                       source_selection=[dict(request_id=row['request_id'], output_tokens=len(row['output_ids']),
                                              manifest_order=i) for i, row in enumerate(group['feedback'])],
                       selected_source_request_id=chosen,
                       retained_campaign_measurements={pop: [dict(request_id=p['request_id'],
                            retained_measurements=retained_measurements(p)) for p in group['reference_pairs'][pop]]
                                                        for pop in ('source', 'heldout')})
        for pair in pairs:
            trajectory = group['frozen_trajectories'][pair['request_id']]
            events = sorted(trajectory['events'], key=lambda event: event['output_offset'])
            require(len({event['output_offset'] for event in events}) == len(events), 'Duplicate frozen block starts')
            indices = selected_round_indices(len(events))
            anchors = [_make_anchor(group, pair, events[index], trajectory) for index in indices]
            request = dict(group_id=group['group_id'], domain=group['domain'], request_id=pair['request_id'],
                           population=pair['population'], source_control=pair['population'] == 'source',
                           input_ids_sha256=digest(pair['F']['native']['input_ids']),
                           output_ids_sha256=digest(pair['F']['native']['output_ids']),
                           available_round_count=len(events), selected_round_indices=indices,
                           selected_location_count=len(anchors), no_rounds=len(events) == 0,
                           planned_reconstruction_target_verifications=3*sum(
                               a['reconstruction']['target_verification_count'] for a in anchors),
                           anchors=anchors, retained_measurements=retained_measurements(pair),
                           **_termination(pair['F']['native'], group['eos_ids'], 1024))
            current['requests'].append(request)
            current['anchors'].extend(anchors)
        current['selected_location_count'] = len(current['anchors'])
        current['planned_block_evaluations'] = 3*len(current['anchors'])
        current['planned_reconstruction_target_verifications'] = sum(
            r['planned_reconstruction_target_verifications'] for r in current['requests'])
        plan['groups'].append(current)
    requests = [request for group in plan['groups'] for request in group['requests']]
    plan.update(selected_request_count=len(requests), heldout_request_count=sum(r['population'] == 'heldout' for r in requests),
                source_control_request_count=sum(r['population'] == 'source' for r in requests),
                zero_round_requests=[r['request_id'] for r in requests if r['no_rounds']],
                selected_location_count=sum(g['selected_location_count'] for g in plan['groups']),
                planned_block_evaluations=sum(g['planned_block_evaluations'] for g in plan['groups']),
                planned_reconstruction_target_verifications=sum(
                    g['planned_reconstruction_target_verifications'] for g in plan['groups']))
    plan['plan_sha256'] = plan_digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan, bundle=None):
    """Validate a sealed plan; optionally require exact regeneration from evidence."""
    require(plan['plan_sha256'] == plan_digest(plan), 'Sealed plan digest changed')
    require(plan['schema_version'] == 2 and plan['scope'] == 'read_only_matched_prefix' and
            plan['reconstruction_mode'] == RECONSTRUCTION_MODE and
            plan['arms'] == list(ARMS) and plan['sample_manifest_sha256'] == SAMPLE_MANIFEST_SHA256 and
            plan['lower_lr_commit'] == LOWER_LR_COMMIT, 'Plan identity/arms differ')
    require(plan['evaluation_order'] == 'group_checkpoint_then_location' and plan['arm_order'] == list(ARMS),
            'Sealed checkpoint-major execution order changed')
    require(len(plan['groups']) == 6 and len({g['group_id'] for g in plan['groups']}) == 6,
            'Exactly six original groups required')
    require(plan['maximum_locations'] == MAX_LOCATIONS and plan['maximum_block_evaluations'] == MAX_BLOCKS and
            plan['first_group_qualification_included'] is True and plan['extra_warmups'] == plan['automatic_retries'] == 0,
            'Bounded work/qualification contract changed')
    require(plan['selection'] == SELECTION, 'Fixed outcome-independent sampling contract changed')
    requests, anchors = [], []
    for group in plan['groups']:
        current_requests = group['requests']
        require(len(current_requests) == 5 and [r['population'] for r in current_requests] == ['heldout']*4+['source'],
                'All four held-out requests and one source control required')
        controls = group['source_selection']
        require(len(controls) == 4 and [r['manifest_order'] for r in controls] == list(range(4)) and
                len({r['request_id'] for r in controls}) == 4, 'Original source manifest order missing')
        chosen = sorted(controls, key=lambda row: (row['output_tokens'], row['manifest_order']))[1]['request_id']
        require(chosen == group['selected_source_request_id'] == current_requests[-1]['request_id'],
                'Source control is not outcome-independent lower median')
        flat = []
        for request in current_requests:
            require(request['group_id'] == group['group_id'] and request['domain'] == group['domain'],
                    'Request group/domain differs')
            indices = selected_round_indices(request['available_round_count'])
            require(request['selected_round_indices'] == indices and request['selected_location_count'] ==
                    len(indices) == len(request['anchors']) and request['no_rounds'] == (len(indices) == 0),
                    'Deterministic round selection/count changed')
            require(request['source_control'] == (request['population'] == 'source'), 'Source population label differs')
            require(request['planned_reconstruction_target_verifications'] == 3*sum(
                    a['reconstruction']['target_verification_count'] for a in request['anchors']),
                    'Request reconstruction target-work budget differs')
            offsets = []
            for selected_index, anchor in zip(indices, request['anchors']):
                validate_anchor(anchor)
                require(all(anchor[key] == request[key] for key in
                            ('group_id', 'domain', 'request_id', 'population', 'source_control', 'finish_reason',
                             'cap_hit', 'eos_hit', 'response_boundary', 'output_tokens', 'output_cap')) and
                        anchor['checkpoints'] == group['checkpoints'], 'Anchor/request pairing differs')
                require(digest(anchor['input_ids']) == request['input_ids_sha256'] and
                        digest(anchor['output_ids']) == request['output_ids_sha256'], 'Canonical request IDs differ')
                require(anchor['retained_round_index'] == selected_index, 'Selected frozen round identity changed')
                offsets.append(anchor['block_output_start'])
            require(offsets == sorted(set(offsets)), 'Selected starts must be unique and ordered')
            flat.extend(request['anchors'])
        require(group['anchors'] == flat and group['selected_location_count'] == len(flat) and
                group['planned_block_evaluations'] == 3*len(flat), 'Group flat anchors/work count differs')
        require(group['planned_reconstruction_target_verifications'] == sum(
                r['planned_reconstruction_target_verifications'] for r in current_requests),
                'Group reconstruction target-work budget differs')
        requests.extend(current_requests); anchors.extend(flat)
    require(len({r['request_id'] for r in requests}) == len(requests) == plan['selected_request_count'] == 30 and
            plan['heldout_request_count'] == 24 and plan['source_control_request_count'] == 6, 'Selected population differs')
    require(len({a['anchor_id'] for a in anchors}) == len(anchors) == plan['selected_location_count'] <= MAX_LOCATIONS and
            plan['planned_block_evaluations'] == 3*len(anchors) <= MAX_BLOCKS and
            plan['zero_round_requests'] == [r['request_id'] for r in requests if r['no_rounds']], 'Plan budget/count differs')
    require(plan['planned_reconstruction_target_verifications'] == sum(
            g['planned_reconstruction_target_verifications'] for g in plan['groups']),
            'Total reconstruction target-work budget differs')
    if bundle is not None:
        require(plan == build_plan(bundle), 'Sealed plan differs from original evidence/deterministic selection')
    return dict(status='passed', errors=[], selected_requests=len(requests), selected_locations=len(anchors),
                planned_block_evaluations=3*len(anchors),
                planned_reconstruction_target_verifications=plan['planned_reconstruction_target_verifications'],
                plan_sha256=plan['plan_sha256'])
