"""Synthetic, CPU-only reconstruction and publication-boundary tests."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.krc_mechanism_diagnostic.build_report import (
    CANDIDATES, ROLES, WINDOWS, build_report, fp32, inspect_blocks, reconstruct_block,
    reduction_bound, validate_candidate, validate_native_pair,
)
from online_feedback.krc_mechanism_metrics import (
    fitted_objective_predictions, pooled_summary, response_metrics,
)


COMMIT = 'a' * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def make_window(event, step):
    labels = torch.tensor([[[0, 1, 2, 3]]])
    ordinary = torch.tensor([[[0, 9, 2, 9]]])
    k_margin = torch.tensor([[[1., 0., 1., -1.]]])
    predictions = dict(K_ONLY=ordinary, R_HALF=ordinary,
        R_FULL=torch.tensor([[[0, 1, 2, 9]]]), D_FULL=torch.tensor([[[0, 9, 2, 3]]]))
    record = dict(step=step, status='completed', candidate_order=list(CANDIDATES),
        restoration_passed=True, k_only_witness_passed=True, candidates={},
        gradient_alignment=dict(g_R_dot_g_D=.1, cosine=.2),
        costs=dict(pre_capture_s=.1, candidate_sequence_s=.4))
    record['plan'] = dict(step=step, event_number=event, pass_index=(step-1)//64,
        **{role: [dict(record_id=f'{role}_{index}', sampled_anchors=[
            dict(positions=[[7*index]], keep_mask=[[True]])]) for index in range(2)] for role in ROLES})
    blocks = []
    for candidate in CANDIDATES:
        margins = torch.where(predictions[candidate] == labels, 1., -1.)
        if candidate in ('K_ONLY', 'R_HALF', 'D_FULL'):
            margins[..., 1] = 0
        state_name = 'K_ONLY' if candidate == 'R_HALF' else candidate
        entry = dict(status='valid', state_identity=hashlib.sha256(state_name.encode()).hexdigest(),
            aliases=['K_ONLY'] if candidate == 'R_HALF' else ['R_HALF'] if candidate == 'K_ONLY' else [],
            branch=dict(status='unchanged_post_K' if candidate == 'K_ONLY' else 'applied',
                        coefficient_fp32=None if candidate == 'K_ONLY' else -float(torch.tensor(1/6)),
                        gradient_norm_squared_fp32=1.),
            alpha=0 if candidate == 'K_ONLY' else .5 if candidate == 'R_HALF' else 1., roles={})
        for role in ROLES:
            results = []
            for response_index in range(2):
                response = dict(labels=labels, original_proposals=labels,
                    validity_weights=torch.tensor([[[0., 1., 1., 1.]]]),
                    protected_weights=torch.tensor([[[0., 3., 2., 1.]]]),
                    anchors=torch.tensor([[7*response_index]]), keep=torch.tensor([[True]]),
                    record_id=f'{role}_{response_index}')
                result = response_metrics(response, ordinary, predictions[candidate], k_margin, margins,
                    identity=dict(event=event, step=step, **{'pass': (step-1)//64}, candidate=candidate, role=role))
                results.append(result)
                blocks.extend(result['blocks'])
            metrics = pooled_summary(results)
            summary = metrics['summary']
            entry['roles'][role] = dict(V=float(torch.tensor(summary['candidate_hinge_fp64'])),
                numerator=summary['candidate_hinge_numerator_fp64'], denominator=summary['protected_weight'],
                metrics=metrics, responses=[dict(record_id=f'{role}_{index}', protected_weight=6.,
                    violation_numerator=result['summary']['candidate_hinge_numerator_fp64'],
                    all_margins=margins.tolist(), proposals=predictions[candidate].tolist())
                    for index, result in enumerate(results)])
        if candidate != 'K_ONLY':
            entry['fitted_role'] = 'D' if candidate == 'D_FULL' else 'R'
            entry['direction'] = dict(proposed_norm=.1, realized_norm=0 if candidate == 'R_HALF' else .1,
                                      proposed_dot=-.05, realized_dot=0 if candidate == 'R_HALF' else -.05)
            entry['prediction'] = fitted_objective_predictions(float(torch.tensor(1/6)), entry['alpha'], 'applied', -.05,
                entry['direction']['realized_dot'], entry['roles'][entry['fitted_role']]['V'])
        record['candidates'][candidate] = entry
    record['original_R_post_K'] = deepcopy(record['candidates']['K_ONLY']['roles']['R'])
    record['D_post_K'] = deepcopy(record['candidates']['K_ONLY']['roles']['D'])
    return record, blocks


def create_run(root):
    source = dict(execution_git_commit=COMMIT, dirty_files=[])
    episodes = [dict(event_number=event, parent_state=f'/retained/KRC{event-1}.pt',
                     endpoint_state=f'/retained/KRC{event}.pt',
                     window_plan=f'/sealed/WINDOW_PLAN_{event:03d}.json') for event in (2, 4)]
    write(root / 'sealed/PLAN.json', dict(episodes=episodes, execution_source=source))
    files = {item[key]: dict(bytes=123, sha256=str(item['event_number'])*64)
             for item in episodes for key in ('parent_state', 'endpoint_state')}
    write(root / 'sealed/SEAL.json', dict(files=files, execution_source=source))
    write(root / 'campaign_001/STATUS.json', dict(status='completed', execution_source=source))
    write(root / 'campaign_001/COSTS.json', dict(actual_campaign_wall_s=12.5, execution_source=source,
        episodes={'event_002': {'episode_wall_s': 6., 'correction': {'responses': ['PRIVATE_CANARY'],
            'costs': {'post_k_s': 1.}, 'work': {'violation_backward_completed': 246},
            'mechanism': {'rows': [{'step': 1, 'costs': {'candidate_sequence_s': .4},
                                   'work': {'candidate_response_forwards': 24}}]}}}},
        work_ledger=dict(core_response_forwards=2136, core_response_backwards=1000,
            ordinary_Adam_steps=256, startup_response_forwards=4, startup_response_backwards=4,
            startup_Adam_steps=0, serving_generations=0, feedback_generations=0)))
    for name in ('PREFLIGHT', 'POSTFLIGHT'):
        write(root / (name + '.json'), dict(status='passed'))
    for event in (2, 4):
        folder = root / 'campaign_001' / f'event_{event:03d}'
        plans = []
        for name in ('ORIGINAL_ENDPOINT_CHECK', 'LIVE_ENDPOINT', 'LIVE_RESUME'):
            write(folder / (name + '.json'), dict(status='passed'))
        write(folder / 'ENDPOINT_EQUALITY.json', dict(status='passed', event_number=event,
            exact_decoded_scientific_state=True, scientific_mismatch_count=0, metadata_difference_count=3))
        for step in WINDOWS:
            record, blocks = make_window(event, step)
            plans.append(record['plan'])
            write(folder / 'mechanism' / f'window_{step:03d}.json', record)
            (folder / 'mechanism' / f'window_{step:03d}_BLOCKS.jsonl').write_text(
                ''.join(json.dumps(block) + '\n' for block in blocks))
        write(root / 'sealed' / f'WINDOW_PLAN_{event:03d}.json', plans)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'private_run'
        self.output = Path(self.directory.name) / 'public'
        create_run(self.root)

    def build(self):
        return build_report(self.root, self.output, execution_commit=COMMIT)

    def test_full_coverage_oracles_counted_once_and_raw_arrays_never_published(self):
        summary = self.build()
        self.assertEqual(summary['included_candidate_slots'], 80)
        self.assertEqual(summary['excluded_candidate_slots'], 0)
        self.assertEqual(len(summary['fallacy_scan']), 11)
        self.assertEqual(summary['recommendation']['status'], 'inconclusive')
        oracles = csv_rows(self.output / 'ORACLES.csv')
        self.assertEqual(len(oracles), 20*3*2)
        first = [row for row in oracles if row['oracle'] == 'FIRST_PROTECTED_ERROR']
        self.assertEqual(sum(int(row['prefix_gain_sum']) for row in first), 240)
        all_rows = [row for row in oracles if row['oracle'] == 'ALL_PROTECTED_ERRORS']
        self.assertEqual(sum(int(row['prefix_gain_sum']) for row in all_rows), 360)
        role = next(row for row in csv_rows(self.output / 'CANDIDATE_ROLE.csv')
                    if row['candidate'] == 'R_FULL' and row['role'] == 'R')
        self.assertEqual(int(role['valid_blocks']), 10)
        self.assertEqual(float(role['prefix_change_mean']), 2.)
        self.assertIn('response_max_reconstruction_error', role)
        self.assertIn('response_max_rounding_bound', role)
        window = csv_rows(self.output / 'WINDOW_SUMMARY.csv')[0]
        self.assertIn('R_response_max_reconstruction_error', window)
        self.assertIn('R_response_max_rounding_bound', window)
        public = '\n'.join(path.read_text() for path in self.output.iterdir())
        for private in ('canonical_tokens', 'K_ONLY_predictions', 'PRIVATE_CANARY'):
            self.assertNotIn(private, public)
        costs = json.loads((self.output / 'COSTS_WORK.json').read_text())
        self.assertEqual(costs['episodes']['event_002']['original_correction_costs']['post_k_s'], 1.)

    def test_failed_endpoint_excludes_episode_without_hiding_measured_work(self):
        path = self.root / 'campaign_001/event_004/ENDPOINT_EQUALITY.json'
        record = json.loads(path.read_text())
        record.update(status='failed', exact_decoded_scientific_state=False, scientific_mismatch_count=1)
        write(path, record)
        summary = self.build()
        self.assertEqual(summary['included_candidate_slots'], 40)
        self.assertEqual(summary['excluded_candidate_slots'], 40)
        self.assertEqual({row['event'] for row in csv_rows(self.output / 'CANDIDATE_ROLE.csv')}, {'2'})
        costs = json.loads((self.output / 'COSTS_WORK.json').read_text())
        self.assertEqual(costs['work_ledger']['core_response_forwards'], 2136)
        self.assertIsNone(summary['recommendation']['intervention'])

    def test_missing_original_state_identity_is_not_replaced_by_endpoint_status(self):
        path = self.root / 'sealed/SEAL.json'
        seal = json.loads(path.read_text())
        del seal['files']['/retained/KRC4.pt']
        write(path, seal)
        summary = self.build()
        self.assertEqual(summary['included_candidate_slots'], 40)
        self.assertIn('missing_sealed_endpoint_state_identity', summary['episodes'][1]['exclusions'])

    def test_wrong_block_count_excludes_affected_window_and_retains_slot_rows(self):
        path = self.root / 'campaign_001/event_002/mechanism/window_001_BLOCKS.jsonl'
        blocks = [json.loads(line) for line in path.read_text().splitlines()]
        blocks[0]['K_prefix_gain_lost'] += 1
        path.write_text(''.join(json.dumps(row) + '\n' for row in blocks))
        summary = self.build()
        self.assertEqual(summary['included_candidate_slots'], 76)
        rows = csv_rows(self.output / 'WINDOW_SUMMARY.csv')
        self.assertEqual(len(rows), 80)
        failed = [row for row in rows if row['included'] == 'False']
        self.assertEqual(len(failed), 4)
        self.assertTrue(all('block_reconstruction_failed' in row['exclusion'] for row in failed))

    def test_invalid_candidate_excluded_but_other_fixed_candidates_and_oracles_retained(self):
        path = self.root / 'campaign_001/event_002/mechanism/window_001.json'
        record = json.loads(path.read_text())
        record['candidates']['R_HALF']['status'] = 'invalid'
        write(path, record)
        summary = self.build()
        self.assertEqual(summary['included_candidate_slots'], 79)
        self.assertEqual(len(csv_rows(self.output / 'ORACLES.csv')), 120)
        self.assertIsNone(summary['recommendation']['intervention'])

    def test_commit_mismatch_stops_before_any_public_output(self):
        with self.assertRaisesRegex(ValueError, 'execution source'):
            build_report(self.root, self.output, execution_commit='b'*40)
        self.assertFalse(self.output.exists())

    def test_pooled_denominators_and_window_distribution_are_explicit(self):
        self.build()
        rows = csv_rows(self.output / 'CANDIDATE_ROLE.csv')
        full = next(row for row in rows if row['candidate'] == 'R_FULL')
        self.assertEqual(int(full['windows']), 5)
        self.assertEqual(int(full['prefix_change_sum']), 20)
        self.assertEqual(float(full['window_change_median']), 4)
        self.assertEqual(int(full['improved_windows']), 5)
        half = [row for row in csv_rows(self.output / 'WINDOW_SUMMARY.csv') if row['candidate'] == 'R_HALF']
        self.assertTrue(all(row['quantized_noop'] == 'True' for row in half))

    def test_independent_reconstruction_rejects_iterative_first_oracle(self):
        record, blocks = make_window(2, 1)
        block = deepcopy(blocks[0])
        self.assertEqual(block['oracle_first_prefix_gain'], 2)
        self.assertEqual(block['oracle_all_prefix_gain'], 3)
        block['oracle_first_prefix_gain'] = 3
        with self.assertRaisesRegex(ValueError, 'oracle_first_prefix_gain'):
            reconstruct_block(block)

    def observed(self):
        path = self.root / 'campaign_001/event_002/mechanism/window_001.json'
        record = json.loads(path.read_text())
        groups = inspect_blocks(path.with_name('window_001_BLOCKS.jsonl'), 2, 1, record['plan'])
        return record, groups

    def test_prediction_dose_role_alias_and_algebra_corruption_are_rejected(self):
        record, groups = self.observed()
        patches = (
            (('alpha',), .5), (('fitted_role',), 'D'),
            (('prediction', 'actual_V'), 5.), (('prediction', 'V_before'), 0.),
            (('prediction', 'formal_ideal'), .1), (('prediction', 'realized_linearized_V'), .3),
            (('prediction', 'actual_minus_realized_linearized'), .5),
            (('direction', 'realized_norm'), -1.), (('aliases',), ['K_ONLY']),
            (('branch', 'coefficient_fp32'), -.1),
        )
        for path, value in patches:
            with self.subTest(field=path):
                entry = deepcopy(record['candidates']['R_FULL'])
                target = entry
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    validate_candidate(entry, 'R_FULL', record, groups)

    def test_original_fitted_value_has_no_silent_baseline_fallback(self):
        record, groups = self.observed()
        del record['D_post_K']
        with self.assertRaises(KeyError):
            validate_candidate(record['candidates']['D_FULL'], 'D_FULL', record, groups)

    def test_D_forward_numerical_differences_are_descriptive_not_an_exact_gate(self):
        path = self.root / 'campaign_001/event_002/mechanism/window_001.json'
        record = json.loads(path.read_text())
        original = record['D_post_K']
        for row in original['responses']:
            row['violation_numerator'] = 2.
            row['all_margins'][0][0][3] = -2.
            row['proposals'][0][0][1] = 8
        original.update(numerator=4., V=float(torch.tensor(1/3)))
        entry = record['candidates']['D_FULL']
        entry['branch']['coefficient_fp32'] = -original['V']
        entry['prediction'] = fitted_objective_predictions(original['V'], 1., 'applied',
            entry['direction']['proposed_dot'], entry['direction']['realized_dot'], entry['roles']['D']['V'])
        write(path, record)
        summary = self.build()
        self.assertEqual(summary['included_candidate_slots'], 80)
        window = next(row for row in csv_rows(self.output / 'WINDOW_SUMMARY.csv')
                      if row['candidate'] == 'D_FULL' and row['event'] == '2' and row['local_step'] == '1')
        self.assertGreater(float(window['D_gradient_minus_no_grad_V']), 0)
        self.assertEqual(float(window['D_gradient_minus_no_grad_denominator']), 0)
        self.assertEqual(float(window['D_gradient_no_grad_margin_max_absolute_difference']), 1)
        self.assertEqual(int(window['D_gradient_no_grad_prediction_changed_positions']), 2)

    def test_D_immutable_source_bindings_remain_required(self):
        record, groups = self.observed()
        record['D_post_K']['responses'][0]['record_id'] = 'foreign_response'
        with self.assertRaisesRegex(ValueError, 'response identities'):
            validate_candidate(record['candidates']['D_FULL'], 'D_FULL', record, groups)

    def test_native_pair_arithmetic_is_fp32_exact_with_separate_response_bound(self):
        record, groups = self.observed()
        raw = deepcopy(record['candidates']['K_ONLY']['roles']['R'])
        group = groups['K_ONLY', 'R']
        validate_native_pair(raw, group)
        bad = deepcopy(raw)
        bad['V'] = 1/6  # Correct real division, wrong recorded native FP32 result.
        with self.assertRaisesRegex(ValueError, 'native pair V'):
            validate_native_pair(bad, group)
        bounded = deepcopy(raw)
        bounded['responses'][0]['violation_numerator'] = fp32(1 + 2**-23)
        bounded['numerator'] = fp32(bounded['responses'][0]['violation_numerator'] + 1.)
        bounded['V'] = fp32(bounded['numerator']/12.)
        receipt = validate_native_pair(bounded, group)
        self.assertGreater(receipt['response_max_reconstruction_error'], 0)
        self.assertLess(receipt['response_max_reconstruction_error'], receipt['response_max_rounding_bound'])
        bad = deepcopy(raw)
        bad['responses'][0]['violation_numerator'] = fp32(1.001)
        with self.assertRaisesRegex(ValueError, 'FP32 reduction bound'):
            validate_native_pair(bad, group)

    def test_triangular_magnitudes_are_checked_beyond_membership(self):
        _, blocks = make_window(2, 1)
        blocks[0]['protected_weights'][1] = 4
        with self.assertRaisesRegex(ValueError, 'triangular magnitudes'):
            reconstruct_block(blocks[0])

    def test_consistent_foreign_anchor_across_candidates_still_fails_sealed_binding(self):
        path = self.root / 'campaign_001/event_002/mechanism/window_001_BLOCKS.jsonl'
        blocks = [json.loads(line) for line in path.read_text().splitlines()]
        for block in blocks:
            block['anchor'] += 1
        path.write_text(''.join(json.dumps(row) + '\n' for row in blocks))
        summary = self.build()
        self.assertEqual(summary['included_candidate_slots'], 76)
        failures = [row for row in csv_rows(self.output / 'WINDOW_SUMMARY.csv') if row['included'] == 'False']
        self.assertTrue(all('sealed plan' in row['exclusion'] for row in failures))

    def test_scalar_binary32_reconstruction_matches_torch_cpu_operations(self):
        for a, b in ((1., 3.), (2**24, 1.), (2**-126, 2.), (1.25, .5)):
            self.assertEqual(fp32(fp32(a)+fp32(b)), float(torch.tensor(a)+torch.tensor(b)))
            self.assertEqual(fp32(fp32(a)/fp32(b)), float(torch.tensor(a)/torch.tensor(b)))
        self.assertGreater(reduction_bound(1., 100), reduction_bound(1., 10))


if __name__ == '__main__':
    unittest.main()
