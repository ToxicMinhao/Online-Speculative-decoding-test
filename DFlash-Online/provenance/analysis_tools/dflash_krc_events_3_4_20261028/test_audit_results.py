"""CPU-only fixtures for independent counts, rotation and source-bound gates."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import audit_results as audit


def measurements():
    requests = [dict(request_id='a', panel_ordinal=0), dict(request_id='b', panel_ordinal=1)]
    rows = []
    for request, counts in zip(requests, ((10, 9, 8), (100, 99, 104))):
        for repeat in range(3):
            for arm, count in zip(audit.ARMS, counts):
                rows.append(dict(request_id=request['request_id'], method=arm, repeat=repeat,
                    verification_calls=count, accepted_draft_tokens=20, committed_progress=30,
                    output_tokens=30, request_latency_s=1 + repeat,
                    native_generate_s=0.5 + repeat, cache_maintenance_s=0.1))
    return requests, rows


class AuditTests(unittest.TestCase):
    def test_counts_once_timing_mean_pooled_regression_and_three_comparisons(self):
        requests, rows = measurements()
        totals, comparisons, _ = audit.pool(rows, requests)
        self.assertEqual([totals[arm]['verification_calls'] for arm in audit.ARMS], [110, 108, 112])
        self.assertEqual(totals['KRC4']['summed_mean_request_latency_s'], 4)
        self.assertEqual(totals['KRC4']['all_repeats_request_latency_s'], 12)
        self.assertEqual(set(comparisons), {'KRC4_vs_O4', 'K4_vs_O4', 'KRC4_vs_K4'})
        result = comparisons['KRC4_vs_O4']
        self.assertAlmostEqual(result['call_reduction_pct'], 100 * (110 - 112) / 110)
        self.assertEqual(result['call_request_directions'], dict(improved=1, regressed=1, tied=0))
        self.assertEqual((result['call_gross_saved'], result['call_gross_extra'], result['call_net_saved']), (2, 4, -2))

    def test_missing_duplicate_unknown_and_inconsistent_repeats_fail(self):
        requests, rows = measurements()
        variants = [rows[:-1], rows + [rows[0]]]
        changed = deepcopy(rows)
        changed[0]['verification_calls'] += 1
        variants.append(changed)
        changed = deepcopy(rows)
        changed[0]['request_id'] = 'foreign'
        variants.append(changed)
        changed = deepcopy(rows)
        changed[0]['request_latency_s'] = float('nan')
        variants.append(changed)
        for variant in variants:
            with self.subTest(variant=variant[0]):
                with self.assertRaises(ValueError):
                    audit.pool(variant, requests)

    def test_exact_three_arm_budget_and_rotation(self):
        requests = [dict(request_id=str(index), panel_ordinal=index) for index in range(64)]
        slots = audit.slots(requests)
        self.assertEqual(len(slots), 576)
        self.assertEqual(len(slots) * 2, 1152)
        self.assertEqual(slots[:3], [('0', 0, 'O4'), ('0', 0, 'K4'), ('0', 0, 'KRC4')])
        starts = [Counter(arm for index, (_, rep, arm) in enumerate(slots)
                          if rep == repeat and index % 3 == 0) for repeat in range(3)]
        self.assertTrue(all(sorted(counts.values()) == [21, 21, 22] for counts in starts))
        self.assertEqual(sum(starts, Counter()), Counter(O4=64, K4=64, KRC4=64))
        for request in requests:
            selected = [arm for request_id, _, arm in slots if request_id == request['request_id']]
            for position in range(3):
                self.assertEqual(Counter(selected[position::3]), Counter(audit.ARMS))

    def test_absent_additional_O_history_and_relative_gap_not_own_improvement(self):
        requests = [dict(request_id='a')]
        values = {('a', 'O4'): dict(verification_calls=100),
                  ('a', 'K4'): dict(verification_calls=90),
                  ('a', 'KRC4'): dict(verification_calls=89)}
        historical = dict(K1={'a': 100}, KRC1={'a': 95}, K2={'a': 95}, KRC2={'a': 90})
        result = audit.count_history(values, requests, historical)
        self.assertIsNone(result['event_count_curve'][0]['O'])
        self.assertIsNone(result['event_count_curve'][1]['O'])
        self.assertEqual(result['event_count_curve'][2]['O'], 100)
        self.assertGreater(result['count_learning_curve']['KRC4_vs_KRC2']['call_reduction_pct'], 0)
        self.assertLess(result['relative_advantage_change_percentage_points']['KRC_vs_K_event2_to4'], 0)
        self.assertIsNone(result['relative_advantage_change_percentage_points']['KRC_vs_O_event2_to4'])

    def test_source_bound_development_only_gates(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            def save(relative, counts):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('\n'.join(json.dumps(dict(request_id='a', repeat=0, method=arm,
                    verification_calls=count)) for arm, count in counts.items()) + '\n')

            for name, first, first_counts in (
                    ('development', 'dflash_krc_event_20261027', {'K': 3027, 'KRC': 3005}),
                    ('additional', 'dflash_krc_confirmation_20261028', {'K1': 3150, 'KRC': 3125})):
                save(first + '/raw/campaign_001/evaluation/MEASUREMENTS.jsonl', first_counts)
                second = {'K2': 2973, 'KRC2': 2967} if name == 'development' else {'K2': 3136, 'KRC2': 3116}
                save(f'dflash_krc_continuation_20261028/raw/campaign_001/evaluation/{name}/MEASUREMENTS.jsonl', second)
            save('dflash_osd_one_event_20261025/final_review/raw/campaign_001/evaluation/MEASUREMENTS.jsonl', {'O': 3075})
            save('dflash_ok_continuation_20261027/raw/campaign_001/evaluation_002/MEASUREMENTS.jsonl', {'O': 3059})
            save('dflash_ok_continuation_20261027/raw/campaign_001/evaluation_004/MEASUREMENTS.jsonl', {'O': 2994, 'K': 2942})
            panels = [dict(name='development', expected_control_counts={'O4': {'a': 2994}, 'K4': {'a': 2942}},
                           historical_counts={arm: {'a': value} for arm, value in
                               dict(O1=3075, K1=3027, KRC1=3005, O2=3059, K2=2973, KRC2=2967).items()}),
                      dict(name='additional', expected_control_counts=None,
                           historical_counts={arm: {'a': value} for arm, value in
                               dict(K1=3150, KRC1=3125, K2=3136, KRC2=3116).items()})]
            with patch.object(audit, 'RESEARCH', root):
                for panel in panels:
                    issues = []
                    audit.audit_historical_inputs(panel, issues)
                    self.assertEqual(issues, [])
                invalid = deepcopy(panels[1])
                invalid['expected_control_counts'] = panels[0]['expected_control_counts']
                issues = []
                audit.audit_historical_inputs(invalid, issues)
                self.assertEqual(issues, ['additional: no historical O4/K4 count gate'])
                invalid = deepcopy(panels[0])
                invalid['expected_control_counts']['K4']['a'] = 2973
                issues = []
                audit.audit_historical_inputs(invalid, issues)
                self.assertEqual(issues, ['development: exact retained O4/K4 count gates'])


if __name__ == '__main__':
    unittest.main()
