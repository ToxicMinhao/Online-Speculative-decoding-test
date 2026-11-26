import json
from pathlib import Path
import tempfile
import unittest

import torch

from . import point2a_math as P
from .allocation_contract import numerical_contract
from .math import allocate_prefix


def snapshot(lengths):
    slots = torch.arange(16)
    prefix = ((slots > 0) & (slots <= torch.tensor(lengths)[:, None])).unsqueeze(0)
    repair = torch.zeros_like(prefix)
    for i, length in enumerate(lengths):
        if length < 15:
            repair[0, i, length + 1] = True
    beta = torch.exp(-slots.float() / 7).expand_as(prefix).clone()
    return dict(roles=dict(beta=beta, prefix=prefix, repair=repair,
                          denominator=(beta * (prefix | repair)).sum() + 1e-6),
                repair_coefficients=beta.clone())


class FakeLedger:
    def __init__(self):
        self.records, self.completed = [], []

    def reserve_token(self, label):
        self.records.append(label)
        return len(self.records)

    def complete_token(self, reservation):
        self.completed.append(reservation)


class FakeWork:
    def __init__(self):
        self.started, self.finished = [], []

    def start(self, label):
        self.started.append(label)

    def finish(self, label):
        self.finished.append(label)


class FakeScorer:
    """Tiny linear CPU fixture checks plumbing, not native-model fidelity."""
    def __init__(self):
        self.snapshots = [snapshot([2]), snapshot([3])]
        self.parameters = (torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)),)
        self.parameter_names = ['fixture']
        self.fast_calls = 0

    def token_fkl(self, response_index, block_index, position):
        gradient = torch.tensor([position, response_index + 1], dtype=torch.float32)
        return (self.parameters[0].float() * gradient).sum()

    def fast_scores(self, direction):
        self.fast_calls += 1
        effective = tuple(w.to(torch.bfloat16) for w in direction)
        scores, allocations, diagnostics = [], [], []
        for index, item in enumerate(self.snapshots):
            roles = item['roles']
            raw = torch.arange(16).float() * effective[0][0].float()
            raw += (index + 1) * effective[0][1].float()
            score = (-raw / (2 * roles['denominator'])).reshape(1, 1, 16)
            allocation, diagnostic = allocate_prefix(score, roles['beta'], roles['prefix'])
            scores.append(score)
            allocations.append(allocation)
            diagnostics.append(diagnostic)
        return dict(scores=scores, allocations=allocations, allocation_diagnostics=diagnostics,
                    effective_direction=effective, direction_scale=1.0,
                    direct_loss_masks=[s['roles']['prefix'] | s['roles']['repair'] for s in self.snapshots],
                    repair_coefficients=[s['repair_coefficients'] for s in self.snapshots],
                    accumulation_factor=0.5,
                    response_denominators=[s['roles']['denominator'] for s in self.snapshots],
                    attention_work={}, direction_policy='fixture BF16 cast')


class Point2AMathTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), 'Point2A CPU checks initialized CUDA')

    def test_new_contract_only_changes_reference_sample_and_caps(self):
        old, new = numerical_contract(), P.point2a_numerical_contract()
        for key in ('gate_b_relative_beta', 'gate_c_alignment_fraction', 'relative_cap',
                    'mass_atol', 'mass_rtol', 'repeatability', 'constant_reference',
                    'historical_hidden_state_gate', 'gate_a', 'gate_b', 'gate_c'):
            self.assertEqual(old[key], new[key])
        self.assertEqual((new['maximum_reference_backwards_per_audit'],
                          new['maximum_reference_backwards_including_one_correction']), (30, 60))
        self.assertEqual(json.loads(json.dumps(new, allow_nan=False)), new)

    def test_lowest_eligible_complete_pools_selected_before_scores(self):
        rows = [snapshot([1, 13, 15]), snapshot([0, 15, 15])]
        selected = P.select_reference_pools(rows, ['one', 'two'])
        self.assertEqual([(p['response_index'], p['block_index']) for p in selected['pools']], [(0, 1), (1, 1)])
        self.assertEqual(selected['reference_backward_count'], 28)
        self.assertTrue(selected['selected_before_scores'])
        self.assertEqual(selected['coverage'][0]['singleton_pools_in_response'], 1)
        self.assertEqual(selected['pools'][0]['positions'], list(range(1, 14)))

    def test_changed_or_missing_coverage_cannot_select_easier_pools(self):
        with self.assertRaisesRegex(ValueError, 'coverage changed'):
            P.select_reference_pools([snapshot([2, 13]), snapshot([15])])
        with self.assertRaisesRegex(ValueError, 'eligible complete'):
            P.select_reference_pools([snapshot([1]), snapshot([15])])

    def test_wrong_reference_wrong_gradient_and_post_c_context_rejected(self):
        args = dict(reference_name='C', parent_identity={'step': 128, 'moments': 'parent'},
                    current_identity={'step': 128, 'moments': 'parent'},
                    captured_gradient_identity={'sha256': 'native-C'},
                    supplied_gradient_identity={'sha256': 'native-C'})
        self.assertTrue(P.require_c_reference(**args)['parent_restored_before_shadow'])
        for change in (dict(reference_name='BETA'),
                       dict(supplied_gradient_identity={'sha256': 'beta'}),
                       dict(current_identity={'step': 129, 'moments': 'post-C'}),
                       dict(current_identity={'step': 128, 'moments': 'post-C'})):
            with self.assertRaises(ValueError):
                P.require_c_reference(**dict(args, **change))

    def test_adapter_reserves_immediately_before_backward_not_graph_build(self):
        ledger = FakeLedger()
        adapter = P.TokenLedgerAdapter(ledger)
        adapter.reserve({'response_index': 0, 'block_index': 1, 'position': 3})
        self.assertEqual(ledger.records, [])
        adapter.mark('backward_started')
        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(ledger.completed, [])
        adapter.mark('backward_completed')
        self.assertEqual(ledger.completed, [1])
        adapter.mark('completed')
        with self.assertRaises(ValueError):
            adapter.mark('backward_started')

    def test_failed_backward_keeps_spent_reservation(self):
        ledger = FakeLedger()
        adapter = P.TokenLedgerAdapter(ledger)
        adapter.reserve({'position': 1})
        adapter.mark('backward_started')
        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(ledger.completed, [])
        with self.assertRaises(ValueError):
            adapter.reserve({'position': 2})

    def test_oracle_reuses_native_gradients_normalizes_once_and_repeats_exact(self):
        scorer, ledger, work = FakeScorer(), FakeLedger(), FakeWork()
        selection = P.select_reference_pools(scorer.snapshots, expected_counts=(2, 3))
        direction = (torch.tensor([1.0, 0.5]),)
        with tempfile.TemporaryDirectory() as temporary:
            receipt, fast = P.qualify_direction(scorer, direction, temporary, ledger, work, selection)
            rows = json.loads((Path(temporary) / 'REFERENCE_PROGRESS.json').read_text())['rows']
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['repeat_exact'], dict(scores=True, allocations=True))
        self.assertEqual(scorer.fast_calls, 2)
        self.assertEqual(len(ledger.records), 5)
        self.assertEqual(ledger.completed, [1, 2, 3, 4, 5])
        self.assertEqual(work.started, work.finished)
        self.assertNotIn('effective_direction', fast)
        self.assertEqual(rows[0]['primary_dot_fp64'], 1.5)
        self.assertEqual(rows[0]['reference_score_fp64'], -1.5 / (2 * rows[0]['denominator']))
        self.assertEqual(receipt['selected_pool_receipts'][0]['direction_diagnostic']['primary_target'],
                         'represented_FP32_w')

    def test_qualification_rejects_changed_selection_before_any_fast_call(self):
        scorer = FakeScorer()
        selection = P.select_reference_pools(scorer.snapshots, expected_counts=(2, 3))
        selection['pools'][0]['positions'] = [1]
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, 'selection changed'):
            P.qualify_direction(scorer, (torch.ones(2),), temporary, FakeLedger(), FakeWork(), selection)
        self.assertEqual(scorer.fast_calls, 0)

    def test_allocator_keeps_beta_budget_not_native_c_coefficients(self):
        scorer = FakeScorer()
        direction = (torch.tensor([1.0, 0.5]),)
        fast = scorer.fast_scores(direction)
        for index, item in enumerate(scorer.snapshots):
            roles = item['roles']
            expected, _ = allocate_prefix(fast['scores'][index], roles['beta'], roles['prefix'])
            native_c = roles['beta'].clone()
            native_c[roles['prefix']] *= 1.2
            wrong, _ = allocate_prefix(fast['scores'][index], native_c, roles['prefix'])
            self.assertTrue(torch.equal(fast['allocations'][index], expected))
            self.assertFalse(torch.equal(wrong, expected))
            self.assertTrue(torch.allclose((expected * roles['prefix']).sum(-1),
                                           (roles['beta'] * roles['prefix']).sum(-1)))

    def test_coefficient_movement_has_pool_budget_denominator_and_exact_zero(self):
        item = snapshot([2, 1])
        roles = item['roles']
        candidate = roles['beta'].clone()
        candidate[0, 0, 1] += 0.1
        candidate[0, 0, 2] -= 0.1
        result = P.coefficient_movement(candidate, roles['beta'], roles['beta'], roles['prefix'])
        pool = result['per_pool'][0]
        delta = P.fp32_difference(candidate, roles['beta']).double()[0, 0, 1:3]
        self.assertAlmostEqual(pool['half_l1_mass_moved_over_beta_budget'],
                               float(delta.abs().sum()) / (2 * pool['beta_budget']))
        self.assertEqual(result['changed_positions'], 2)
        self.assertTrue(result['per_pool'][1]['coefficients_equal'])
        self.assertEqual(result['per_pool'][1]['half_l1_mass_moved_over_beta_budget'], 0)

    def test_score_offset_and_scaling_separate_ordering_from_raw_magnitudes(self):
        item = snapshot([3])
        roles = item['roles']
        before = torch.arange(16).float().reshape(1, 1, 16)
        shifted = P.score_changes(2 * before + 10, before, roles['beta'], roles['prefix'])
        self.assertFalse(shifted['raw_scores_equal'])
        self.assertEqual(shifted['centered_sign_changes'], 0)
        self.assertEqual(shifted['ordering_changes'], 0)
        reversed_result = P.score_changes(-before, before, roles['beta'], roles['prefix'])
        self.assertEqual(reversed_result['ordering_changes'], 3)
        self.assertEqual(reversed_result['per_pool'][0]['ordering_strict_reversals'], 3)

    def test_gradient_subtraction_promotes_bf16_operands_before_subtraction(self):
        a = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        b = torch.tensor([0.001953125, 2.0], dtype=torch.bfloat16)
        direction = torch.tensor([2.0, 3.0])
        receipt = P.gradient_contraction([a], [b], [direction], chunk_elements=1)
        expected = float(-(direction.double() * (a.float() - b.float()).double()).sum())
        wrong = float(-(direction.double() * (a - b).double()).sum())
        self.assertEqual(receipt['total'], expected)
        self.assertNotEqual(expected, wrong)
        self.assertGreater(receipt['positive'], 0)
        self.assertLess(receipt['negative'], 0)
        zero = P.gradient_contraction([a], [a], [direction])
        self.assertEqual(zero['total'], 0)
        self.assertIsNone(zero['cancellation_fraction'])

    def test_all_three_contrasts_use_c_scores_and_sum_response_normalization_once(self):
        scorer = FakeScorer()
        fast = scorer.fast_scores((torch.tensor([1.0, 0.5]),))
        base = [s['roles']['beta'] for s in scorer.snapshots]
        allocations = {'C': base, 'F_BETA': [v.clone() for v in base], 'F_C': fast['allocations']}
        gradients = {'C': [torch.tensor([1.0])], 'F_BETA': [torch.tensor([2.0])],
                     'F_C': [torch.tensor([3.0])]}
        result = P.three_contrasts(allocations, fast['scores'], scorer.snapshots, gradients, [torch.ones(1)])
        self.assertEqual(tuple(result), ('F_C-F_BETA', 'F_C-C', 'F_BETA-C'))
        receipt = result['F_C-C']
        expected = sum(row['score_contrast']['raw']['total'] for row in receipt['per_response'])
        self.assertEqual(receipt['prediction']['raw']['total'], expected)
        self.assertEqual(receipt['actual_gradient_contraction']['total'], -2.0)
        self.assertTrue(result['F_BETA-C']['coefficient_equality'])
        self.assertEqual(result['F_BETA-C']['prediction']['raw']['total'], 0)
        self.assertEqual(receipt['prediction']['reference'], 'C')


if __name__ == '__main__':
    unittest.main()
