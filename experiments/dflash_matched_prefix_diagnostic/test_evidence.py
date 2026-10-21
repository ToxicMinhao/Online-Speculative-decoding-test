"""Synthetic CPU tests: no private records, tokenizer or model loading."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dflash_baseline_learning_curve.experiment import digest
from experiments.dflash_matched_prefix_diagnostic import evidence as E


def checkpoint(arm, group):
    files = {'model.safetensors': dict(bytes=1, sha256=digest([arm, group]))}
    return dict(path='/synthetic/'+group+'/'+arm, files=files, checkpoint_identity_sha256=digest(files),
                checkpoint_id={'F': 'F', 'S8': 'U2', 'L8': 'L8'}[arm])


def trajectory(row):
    ids, output = row['input_ids'], row['output_ids']
    events = [dict(kind='target_prefill', output_offset=0, position=len(ids),
                   committed_token_ids=output[:1], committed_provenance=['target_prefill'])]
    for i, start in enumerate(range(1, len(output))):
        proposals = [777]*9
        position = len(ids)+start
        events.append(dict(kind='verification', round_index=i, output_offset=start,
                           draft_proposed_token_ids=proposals,
                           draft_proposed_positions=list(range(position, position+9)),
                           verification_input_token_ids=[output[start-1]]+proposals,
                           verification_input_positions=list(range(position-1, position+9)),
                           target_predicted_token_ids=[output[start]]+[777]*9,
                           target_prediction_positions=list(range(position, position+10)),
                           context=dict(input_ids_sha256=digest(ids), committed_output_prefix_length=start,
                                        verification_input_start_position=position-1),
                           rejection_boundary_proposal_index=0, committed_token_ids=output[start:start+1],
                           native_acceptance_length_before_stop=0, finished=start == len(output)-1,
                           committed_provenance=['target_correction'], stop_clipped=start == len(output)-1))
    return dict(events=events, input_token_ids=ids, final_output_token_ids=output, max_new_tokens=1024,
                finish_reason=row['finish_reason'], reconciliation=dict(status='passed'))


def make_row(group, population, i, length):
    row = dict(group_id=group, request_id=f'{group}_{population}_{i}', input_ids=[11, 12],
               output_ids=list(range(20, 20+length-1))+[99], accepted_draft_tokens=0,
               verification_calls=length-1, verification_committed_tokens=length-1,
               initial_output_tokens=1, finish_reason=dict(type='stop', matched=99), cap_hit=False, eos_hit=True)
    from experiments.dflash_baseline_learning_curve.native import normalize_rounds
    row['rounds'] = normalize_rounds(trajectory(row), row, row['input_ids'])[0]
    return row


def synthetic_bundle():
    groups = []
    for g in range(6):
        gid = f'group_{g}'
        cps = {arm: checkpoint(arm, gid) for arm in E.ARMS}
        group = dict(group_id=gid, domain='synthetic', checkpoints=cps, feedback=[],
                     reference_pairs=dict(source=[], heldout=[]), eos_ids=[99], frozen_trajectories={})
        for population in ('source', 'heldout'):
            for i in range(4):
                row = make_row(gid, population, i, [6, 4, 4, 8][i])
                pair = dict(group_id=gid, domain='synthetic', population=population, request_id=row['request_id'],
                            checkpoint_bindings={arm: value['checkpoint_identity_sha256'] for arm, value in cps.items()},
                            **{arm: dict(native=copy.deepcopy(row), kl=dict(loss=1.0)) for arm in E.ARMS})
                group['reference_pairs'][population].append(pair)
                if population == 'source':
                    group['feedback'].append(row)
                group['frozen_trajectories'][row['request_id']] = dict(
                    events=E.frozen_rounds(trajectory(row), row),
                    prefill=E.frozen_prefill(trajectory(row), row), binding=dict(synthetic=True))
        groups.append(group)
    return dict(groups=groups, bindings={}, native_runtime={}, runtime_config={})


class SelectionTests(unittest.TestCase):
    def test_zero_and_single_rounds(self):
        self.assertEqual(E.selected_round_indices(0), [])
        self.assertEqual(E.selected_round_indices(1), [0])

    def test_endpoints_and_distinct_floor_indices(self):
        for n in range(2, 1000):
            actual = E.selected_round_indices(n)
            self.assertEqual(len(actual), min(8, n))
            self.assertEqual(actual, sorted(set(actual)))
            self.assertEqual((actual[0], actual[-1]), (0, n-1))
        self.assertEqual(E.selected_round_indices(10), [0, 1, 2, 3, 5, 6, 7, 9])

    def test_no_sampling_budget_override(self):
        for n, budget in [(-1, 8), (True, 8), (8, 9)]:
            with self.assertRaises(ValueError):
                E.selected_round_indices(n, budget)

    def test_source_lower_median_manifest_tiebreak(self):
        rows = [dict(request_id=str(i), output_ids=[0]*length) for i, length in enumerate([6, 4, 4, 8])]
        self.assertEqual(E.select_source_control(rows)['request_id'], '2')

    def test_selectors_do_not_use_adapted_measurements(self):
        bundle = synthetic_bundle()
        before = E.build_plan(bundle)
        for group in bundle['groups']:
            for population in ('source', 'heldout'):
                for pair in group['reference_pairs'][population]:
                    pair['S8']['native']['verification_calls'] = 12345
                    pair['L8']['kl']['loss'] = 876.0
        after = E.build_plan(bundle)
        self.assertEqual([[a['anchor_id'] for a in g['anchors']] for g in before['groups']],
                         [[a['anchor_id'] for a in g['anchors']] for g in after['groups']])


class FrozenTraceTests(unittest.TestCase):
    def setUp(self):
        self.row = make_row('group', 'heldout', 0, 4)
        self.trace = trajectory(self.row)

    def test_all_nine_saved_proposals_retained_at_terminal(self):
        events = E.frozen_rounds(self.trace, self.row)
        self.assertEqual(len(events[-1]['draft_proposed_token_ids']), 9)
        self.assertEqual(events[-1]['output_offset'], len(self.row['output_ids'])-1)

    def test_missing_terminal_proposals_are_not_substituted(self):
        self.trace['events'][-1]['draft_proposed_token_ids'].pop()
        with self.assertRaises(ValueError):
            E.frozen_rounds(self.trace, self.row)

    def test_incorrect_seed_is_rejected(self):
        self.trace['events'][1]['verification_input_token_ids'][0] = 7
        with self.assertRaises(ValueError):
            E.frozen_rounds(self.trace, self.row)

    def test_wrong_absolute_positions_rejected(self):
        self.trace['events'][1]['draft_proposed_positions'][0] += 1
        with self.assertRaises(ValueError):
            E.frozen_rounds(self.trace, self.row)

    def test_duplicate_offset_rejected(self):
        self.trace['events'][2]['output_offset'] = 1
        with self.assertRaises(ValueError):
            E.frozen_rounds(self.trace, self.row)

    def test_original_prefill_seed_and_position_are_required(self):
        for key, value in [('position', 0), ('committed_token_ids', [7]), ('output_offset', 1)]:
            trace = copy.deepcopy(self.trace)
            trace['events'][0][key] = value
            with self.assertRaises(ValueError):
                E.frozen_rounds(trace, self.row)

    def test_reconstruction_requires_complete_target_predictions_and_round_identity(self):
        for key, value in [('target_predicted_token_ids', [20]*9),
                           ('target_prediction_positions', [1]*10),
                           ('native_acceptance_length_before_stop', 1), ('round_index', 1),
                           ('stop_clipped', True), ('finished', True)]:
            trace = copy.deepcopy(self.trace)
            trace['events'][1][key] = value
            with self.assertRaises(ValueError, msg=key):
                E.frozen_rounds(trace, self.row)

    def test_trace_hash_and_private_path_remapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'native/group/request.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(self.trace))
            self.row['trajectory'] = dict(E.file_binding(path), path='/private/original/native/group/request.json')
            document, binding = E.remap_trace(directory, '/private/original', self.row)
            self.assertEqual(document, self.trace)
            self.assertEqual(binding['local']['path'], str(path))
            self.assertEqual(binding['original']['path'], self.row['trajectory']['path'])
            path.write_text('{}')
            with self.assertRaises(ValueError):
                E.remap_trace(directory, '/private/original', self.row)

    def test_trace_other_group_path_rejected(self):
        self.row['trajectory'] = dict(path='/private/original/native/other/request.json')
        with self.assertRaises(ValueError):
            E.remap_trace('/tmp', '/private/original', self.row)

    def test_trace_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            path = Path(outside)/'request.json'; path.write_text('{}')
            local = Path(root)/'native/group'; local.mkdir(parents=True)
            (local/'request.json').symlink_to(path)
            self.row['trajectory'] = dict(E.file_binding(path), path='/original/native/group/request.json')
            with self.assertRaises(ValueError):
                E.remap_trace(root, '/original', self.row)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.bundle = synthetic_bundle()
        self.plan = E.build_plan(self.bundle)

    def test_deterministic_serializable_and_input_immutable(self):
        original = copy.deepcopy(self.bundle)
        repeated = E.build_plan(self.bundle)
        self.assertEqual(repeated, self.plan)
        self.assertEqual(json.loads(json.dumps(repeated)), self.plan)
        self.assertEqual(original, self.bundle)
        self.assertEqual(E.validate_plan(self.plan, self.bundle)['status'], 'passed')
        self.assertEqual(self.plan['selected_request_count'], 30)
        self.assertEqual(self.plan['heldout_request_count'], 24)
        self.assertEqual(self.plan['source_control_request_count'], 6)
        self.assertEqual(self.plan['evaluation_order'], 'group_checkpoint_then_location')
        self.assertEqual(self.plan['arm_order'], ['F', 'S8', 'L8'])

    def test_terminal_short_block_keeps_full_f_proposals_and_marks_censoring(self):
        anchor = self.plan['groups'][0]['requests'][0]['anchors'][-1]
        self.assertEqual(anchor['canonical_next_ids'], [99])
        self.assertEqual(len(anchor['expected_frozen_proposal_ids']), 9)
        self.assertEqual(anchor['censored_suffix_positions'], 8)
        self.assertEqual(anchor['response_boundary'], 'eos')
        self.assertTrue(anchor['reaches_response_boundary'])

    def test_zero_round_request_is_reported_not_replaced(self):
        group = self.bundle['groups'][0]
        pair = group['reference_pairs']['heldout'][0]
        row = make_row(group['group_id'], 'heldout', 0, 1)
        for arm in E.ARMS:
            pair[arm]['native'] = copy.deepcopy(row)
        group['frozen_trajectories'][pair['request_id']]['events'] = E.frozen_rounds(trajectory(row), row)
        plan = E.build_plan(self.bundle)
        request = plan['groups'][0]['requests'][0]
        self.assertEqual(request['anchors'], [])
        self.assertEqual(plan['zero_round_requests'], [request['request_id']])
        self.assertEqual(plan['selected_request_count'], 30)

    def test_cap_censoring_is_not_eos(self):
        anchor = copy.deepcopy(self.plan['groups'][0]['anchors'][-1])
        anchor.update(output_ids=[20]*1024, block_output_start=1023, output_tokens=1024,
                      canonical_prefix_ids=anchor['input_ids']+[20]*1023, seed_token_id=20,
                      canonical_prefix_sha256=digest(anchor['input_ids']+[20]*1023),
                      canonical_next_ids=[20], expected_absolute_positions=list(range(1025, 1034)),
                      finish_reason=dict(type='length', length=1024), cap_hit=True, eos_hit=False,
                      response_boundary='cap', reaches_response_boundary=True,
                      censored_suffix_positions=8, suffix_censored=True)
        row = dict(anchor, initial_output_tokens=1)
        trace = trajectory(row)
        anchor['retained_round_index'] = 1022
        anchor['reconstruction'] = E._reconstruction(dict(events=trace['events'][1:],
            prefill=E.frozen_prefill(trace, row)), 1022)
        self.assertEqual(E.validate_anchor(anchor)['status'], 'passed')

    def test_original_prompt_and_prior_schedule_are_sealed_separately(self):
        anchor = self.plan['groups'][0]['anchors'][2]
        reconstruction = E.validate_reconstruction(anchor)
        self.assertEqual(reconstruction['prefill']['prompt_tokens'], len(anchor['input_ids']))
        self.assertEqual(reconstruction['prefill']['seed_token_id'], anchor['output_ids'][0])
        self.assertEqual([r['output_offset'] for r in reconstruction['rounds']], [1, 2])
        self.assertEqual(reconstruction['target_verification_count'], 2)
        self.assertEqual(reconstruction['rounds'][-1]['next_seed_token_id'], anchor['seed_token_id'])
        self.assertEqual(reconstruction['schedule_sha256'], E.reconstruction_digest(reconstruction))
        first = self.plan['groups'][0]['anchors'][0]
        self.assertEqual(first['reconstruction']['rounds'], [])

    def test_reconstruction_rejects_discontinuous_and_wrong_commit_schedule(self):
        original = self.plan['groups'][0]['anchors'][2]
        mutations = [('output_offset', 2), ('commit_length', 2), ('committed_token_ids', [1]),
                     ('next_seed_token_id', 1), ('verification_input_token_ids', [0]*10),
                     ('target_predicted_token_ids', [777]*10), ('finished', True),
                     ('stop_clipped', True), ('round_index', 1)]
        for key, value in mutations:
            anchor = copy.deepcopy(original)
            reconstruction = anchor['reconstruction']
            reconstruction['rounds'][0][key] = value
            reconstruction['schedule_sha256'] = E.reconstruction_digest(reconstruction)
            with self.assertRaises(ValueError, msg=key):
                E.validate_reconstruction(anchor)

    def test_reconstruction_cannot_omit_round_or_include_measured_block(self):
        original = self.plan['groups'][0]['anchors'][2]
        for delta in (-1, 1):
            anchor = copy.deepcopy(original)
            if delta == -1:
                anchor['reconstruction']['rounds'].pop()
            else:
                anchor['reconstruction']['rounds'].append(copy.deepcopy(anchor['reconstruction']['rounds'][-1]))
            reconstruction = anchor['reconstruction']
            reconstruction['target_verification_count'] = len(reconstruction['rounds'])
            reconstruction['schedule_sha256'] = E.reconstruction_digest(reconstruction)
            with self.assertRaises(ValueError):
                E.validate_reconstruction(anchor)

    def test_multitoken_commits_reach_exact_original_cache_boundary(self):
        anchor = copy.deepcopy(self.plan['groups'][0]['anchors'][0])
        anchor.update(output_ids=list(range(20, 38)), block_output_start=15,
                      retained_round_index=4, seed_token_id=34)
        reconstruction = anchor['reconstruction']
        reconstruction['rounds'] = []
        cursor = 1
        for index, commit_length in enumerate([7, 1, 1, 5]):
            accepted = commit_length-1
            committed = anchor['output_ids'][cursor:cursor+commit_length]
            proposals = committed[:-1]+[777]*(9-accepted)
            predictions = committed+[777]*(10-commit_length)
            absolute = len(anchor['input_ids'])+cursor
            reconstruction['rounds'].append(dict(round_index=index, output_offset=cursor,
                verification_input_token_ids=[anchor['output_ids'][cursor-1]]+proposals,
                verification_input_positions=list(range(absolute-1, absolute+9)),
                draft_proposed_token_ids=proposals, target_predicted_token_ids=predictions,
                target_prediction_positions=list(range(absolute, absolute+10)),
                committed_token_ids=committed, commit_length=commit_length,
                next_seed_token_id=committed[-1], native_acceptance_length_before_stop=accepted,
                stop_clipped=False, finished=False, raw_event_sha256=digest([index, cursor])))
            cursor += commit_length
        reconstruction['target_verification_count'] = 4
        reconstruction['schedule_sha256'] = E.reconstruction_digest(reconstruction)
        self.assertEqual(E.validate_reconstruction(anchor), reconstruction)
        self.assertEqual([r['output_offset'] for r in reconstruction['rounds']], [1, 8, 9, 10])
        self.assertEqual(reconstruction['rounds'][-1]['next_seed_token_id'], anchor['seed_token_id'])

    def test_reconstruction_budget_is_distinct_from_measured_draft_blocks(self):
        expected = 3*sum(a['reconstruction']['target_verification_count']
                         for g in self.plan['groups'] for a in g['anchors'])
        self.assertEqual(self.plan['planned_reconstruction_target_verifications'], expected)
        self.assertEqual(self.plan['planned_block_evaluations'], 3*self.plan['selected_location_count'])
        for path in ('request', 'group', 'plan'):
            plan = copy.deepcopy(self.plan)
            obj = {'request': plan['groups'][0]['requests'][0], 'group': plan['groups'][0], 'plan': plan}[path]
            obj['planned_reconstruction_target_verifications'] += 1
            plan['plan_sha256'] = E.plan_digest(plan)
            with self.assertRaises(ValueError):
                E.validate_plan(plan)

    def test_eos_at_exact_cap_retains_eos_boundary_and_length_cap_flag(self):
        row = make_row('group', 'heldout', 0, 1024)
        # Use ordinary non-EOS tokens throughout the synthetic capped response.
        row['output_ids'] = [20]*1023+[99]
        row['cap_hit'] = True
        result = E._termination(row, [99], 1024)
        self.assertEqual(result['response_boundary'], 'eos')
        self.assertTrue(result['cap_hit'])
        self.assertTrue(result['eos_hit'])

    def test_anchor_seed_boundary_checkpoint_and_context_guards(self):
        original = self.plan['groups'][0]['anchors'][0]
        changes = [dict(seed_token_id=0), dict(expected_frozen_proposal_ids=[1]*8),
                   dict(canonical_next_ids=[]), dict(reaches_response_boundary=False),
                   dict(censored_suffix_positions=0), dict(eos_ids=[])]
        for change in changes:
            anchor = dict(copy.deepcopy(original), **change)
            with self.assertRaises(ValueError, msg=str(change)):
                E.validate_anchor(anchor)
        changed = copy.deepcopy(original)
        changed['checkpoints']['L8']['checkpoint_identity_sha256'] = 'bad'
        with self.assertRaises(ValueError):
            E.validate_anchor(changed)

    def test_resealed_nonmedian_control_rejected(self):
        self.plan['groups'][0]['selected_source_request_id'] = 'group_0_source_0'
        self.plan['plan_sha256'] = E.plan_digest(self.plan)
        with self.assertRaises(ValueError):
            E.validate_plan(self.plan)

    def test_resealed_selection_rule_change_rejected(self):
        self.plan['selection']['outcome_dependent'] = True
        self.plan['plan_sha256'] = E.plan_digest(self.plan)
        with self.assertRaises(ValueError):
            E.validate_plan(self.plan)

    def test_resealed_checkpoint_order_change_rejected(self):
        self.plan['arm_order'] = ['L8', 'F', 'S8']
        self.plan['plan_sha256'] = E.plan_digest(self.plan)
        with self.assertRaises(ValueError):
            E.validate_plan(self.plan)

    def test_resealed_wrong_round_identity_rejected(self):
        self.plan['groups'][0]['requests'][0]['anchors'][0]['retained_round_index'] += 1
        self.plan['plan_sha256'] = E.plan_digest(self.plan)
        with self.assertRaises(ValueError):
            E.validate_plan(self.plan)

    def test_plan_digest_tamper_is_rejected(self):
        self.plan['extra_warmups'] = 1
        with self.assertRaises(ValueError):
            E.validate_plan(self.plan)

    def test_resealed_proposal_change_requires_original_evidence(self):
        self.plan['groups'][0]['requests'][0]['anchors'][0]['expected_frozen_proposal_ids'][8] = 5
        self.plan['plan_sha256'] = E.plan_digest(self.plan)
        with self.assertRaises(ValueError):
            E.validate_plan(self.plan, self.bundle)


if __name__ == '__main__':
    unittest.main()
