"""Synthetic, CPU-only reconstruction and publication-boundary tests."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.krc_mechanism_diagnostic.build_report import (
    CANDIDATES, ROLES, WINDOWS, aggregate_roles, build_report, reconstruct_block,
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
    blocks = []
    for candidate in CANDIDATES:
        margins = torch.where(predictions[candidate] == labels, 1., -1.)
        if candidate in ('K_ONLY', 'R_HALF', 'D_FULL'):
            margins[..., 1] = 0
        entry = dict(status='valid', state_identity=candidate,
            aliases=['K_ONLY'] if candidate == 'R_HALF' else ['R_HALF'] if candidate == 'K_ONLY' else [],
            branch=dict(status='unchanged_post_K' if candidate == 'K_ONLY' else 'applied',
                        coefficient_fp32=None if candidate == 'K_ONLY' else -.1,
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
            entry['roles'][role] = dict(V=summary['candidate_hinge_fp64'],
                numerator=summary['candidate_hinge_numerator_fp64'], denominator=summary['protected_weight'],
                metrics=metrics)
        if candidate != 'K_ONLY':
            entry['fitted_role'] = 'D' if candidate == 'D_FULL' else 'R'
            entry['direction'] = dict(proposed_norm=.1, realized_norm=0 if candidate == 'R_HALF' else .1,
                                      proposed_dot=-.05, realized_dot=0 if candidate == 'R_HALF' else -.05)
            entry['prediction'] = fitted_objective_predictions(1/6, entry['alpha'], 'applied', -.05,
                entry['direction']['realized_dot'], entry['roles'][entry['fitted_role']]['V'])
        record['candidates'][candidate] = entry
    return record, blocks


def create_run(root):
    source = dict(execution_git_commit=COMMIT, dirty_files=[])
    episodes = [dict(event_number=event, parent_state=f'/retained/KRC{event-1}.pt',
                     endpoint_state=f'/retained/KRC{event}.pt') for event in (2, 4)]
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
        for name in ('ORIGINAL_ENDPOINT_CHECK', 'LIVE_ENDPOINT', 'LIVE_RESUME'):
            write(folder / (name + '.json'), dict(status='passed'))
        write(folder / 'ENDPOINT_EQUALITY.json', dict(status='passed', event_number=event,
            exact_decoded_scientific_state=True, scientific_mismatch_count=0, metadata_difference_count=3))
        for step in WINDOWS:
            record, blocks = make_window(event, step)
            write(folder / 'mechanism' / f'window_{step:03d}.json', record)
            (folder / 'mechanism' / f'window_{step:03d}_BLOCKS.jsonl').write_text(
                ''.join(json.dumps(block) + '\n' for block in blocks))


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


if __name__ == '__main__':
    unittest.main()
