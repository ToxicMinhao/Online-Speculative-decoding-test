"""Exact predeclared order, panel identities and gate limits."""
import copy
import itertools
import unittest

from .evaluation_protocol import (ARMS, method_order, validate_panels, validate_inputs,
                                  prepare_requests, TOTAL_GENERATIONS)
from .test_evaluate import panels, requests


class ProtocolTests(unittest.TestCase):
    def test_exact_parity_base_rotation_and_pair_precedence(self):
        positions = {arm: [0, 0, 0] for arm in ARMS}
        precedence = {pair: 0 for pair in itertools.permutations(ARMS, 2)}
        for ordinal in range(64):
            per_request = {arm: [] for arm in ARMS}
            for repeat in range(3):
                base = ARMS if ordinal % 2 == 0 else ('O', 'KW', 'K')
                shift = (ordinal + repeat) % 3
                order = method_order(ordinal, repeat)
                self.assertEqual(order, base[shift:] + base[:shift])
                for position, arm in enumerate(order):
                    positions[arm][position] += 1
                    per_request[arm].append(position)
                for first, second in precedence:
                    precedence[first, second] += order.index(first) < order.index(second)
            self.assertTrue(all(sorted(values) == [0, 1, 2] for values in per_request.values()))
        self.assertEqual(positions, {arm: [64, 64, 64] for arm in ARMS})
        self.assertTrue(all(value == 96 for value in precedence.values()))

    def test_four_panels_seven_endpoints_budget(self):
        check = validate_panels(panels())
        self.assertEqual(TOTAL_GENERATIONS, 4032)
        self.assertEqual(check['main_generations'], 7 * 64 * 3 * 3)
        self.assertEqual(check['included_gate_generations'], 72)
        self.assertFalse(check['historical_count_gates'])

    def test_wrong_seed_overlap_reference_and_panel_order_rejected(self):
        for label in ('seed', 'overlap', 'canonical', 'order'):
            fixture = copy.deepcopy(panels())
            if label == 'seed':
                fixture[-1]['selection_seed'] = 20261029
            elif label == 'overlap':
                fixture[-1]['requests'][0]['prompt_ids'] = fixture[0]['requests'][0]['prompt_ids']
            elif label == 'canonical':
                fixture[2]['requests'][0]['canonical_source_method'] = 'K1'
            else:
                fixture[0], fixture[1] = fixture[1], fixture[0]
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_panels(fixture)

    def test_no_early_final_no_extra_gate_no_missing_full_event10_gate(self):
        checkpoints = {arm: '/' + arm for arm in ARMS}
        cases = [(10, 'final', 0), (12, 'development', 8), (10, 'development', 1),
                 (10, 'development', 0), (10, 'additional', 8), (8, 'development', 0)]
        for event, panel, gate in cases:
            with self.subTest(event=event, panel=panel, gate=gate), self.assertRaises(ValueError):
                validate_inputs(prepare_requests(requests(panel, 64)), checkpoints, gate,
                    event_number=event, panel_name=panel, canonical_mode='fresh' if panel == 'final' else 'retained')


if __name__ == '__main__':
    unittest.main()
