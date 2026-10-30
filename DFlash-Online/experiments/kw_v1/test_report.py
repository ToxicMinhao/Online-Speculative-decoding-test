"""CPU reconstruction of all seven panels, exact paired bootstrap and outcome rules."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from . import test_evaluate as fixtures
from .test_evaluate import panels
from .evaluation_protocol import ARMS, method_order
from .report import (paired_bootstrap, outcome_label, write_campaign_report,
                     qualification_errors, aggregate_requests)


class BootstrapTests(unittest.TestCase):
    def test_pcg64_matches_independent_loop_ratio_of_sum_percentiles(self):
        import numpy as np
        reference = list(range(1, 65))
        candidate = [value - (index % 4) for index, value in enumerate(reference)]
        # Keep the first request at one call; a valid zero candidate is allowed.
        candidate[0] = 1
        before = np.random.get_state()
        result = paired_bootstrap(reference, candidate)
        after = np.random.get_state()
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(before, after)))
        rng = np.random.Generator(np.random.PCG64(20261030))
        draws = []
        for _ in range(10000):
            indices = rng.integers(0, 64, size=64)
            r = sum(reference[index] for index in indices)
            c = sum(candidate[index] for index in indices)
            draws.append(100 * (r - c) / r)
        self.assertEqual(result['percentile_95_ci_pct'], np.percentile(draws, [2.5, 97.5], method='linear').tolist())
        self.assertEqual(result['estimate_pct'], 100 * (sum(reference) - sum(candidate)) / sum(reference))
        self.assertEqual(result['request_pairs'], 64)
        self.assertEqual(result, paired_bootstrap(reference, candidate))
        self.assertNotEqual(result['estimate_pct'], sum(100 * (r-c) / r for r, c in zip(reference, candidate)) / 64)

    def test_repeats_wrong_units_noninteger_negative_and_zero_denominator_rejected(self):
        for reference, candidate in (([2] * 192, [1] * 192), ([2] * 64, [1.0] * 64),
                                     ([2] * 64, [-1] * 64), ([0] * 64, [0] * 64)):
            with self.subTest(reference=reference[:2], candidate=candidate[:2]), self.assertRaises(ValueError):
                paired_bootstrap(reference, candidate)

    def test_primary_negative_has_precedence_over_pooled_and_uncertainty(self):
        defaults = dict(valid=True, primary_reduction=2., pooled_reduction=1., interval=[.1, 3.],
                        serving_reductions=[1., 2.], costs_complete=True)
        self.assertEqual(outcome_label(**defaults), 'Promising v1')
        for primary in (-1., 0.):
            self.assertEqual(outcome_label(**dict(defaults, primary_reduction=primary, interval=None,
                                                 serving_reductions=[-3.], costs_complete=False)),
                             'Unsupported v1 in this run')
        for altered in (dict(pooled_reduction=0), dict(interval=[-1., 3.]), dict(costs_complete=False),
                        dict(serving_reductions=[-1., 2.]), dict(overhead_prevents_efficiency=True)):
            self.assertEqual(outcome_label(**dict(defaults, **altered)), 'Mixed/inconclusive')
        self.assertEqual(outcome_label(**dict(defaults, valid=False, primary_reduction=-1)), 'Implementation invalid')

    def test_bounded_qualification_false_or_missing_flags_and_budget_are_invalid(self):
        fixture = dict(status='passed', work=dict(response_forwards=8, response_backwards=8,
                       disposable_optimizer_applications=16, serving_generations=0, persistent_optimizer_updates=0),
                       native_neutral_exact=True, preview_isolation=True, full_model_memory_checked=True)
        self.assertEqual(qualification_errors(fixture), [])
        for key, value in (('response_forwards', 9), ('disposable_optimizer_applications', 17),
                           ('persistent_optimizer_updates', 1), ('preview_isolation', False),
                           ('native_neutral_exact', 1), ('response_backwards', True)):
            with self.subTest(key=key):
                changed = dict(fixture)
                if key in fixture['work']:
                    changed['work'] = dict(fixture['work'], **{key: value})
                else:
                    changed[key] = value
                self.assertTrue(qualification_errors(changed))


class CampaignReportTests(unittest.TestCase):
    setUp = fixtures.EvaluationTests.setUp
    run_panel = fixtures.EvaluationTests.run_panel
    read = fixtures.EvaluationTests.read

    def prepare_campaign(self):
        manifest = panels()
        (self.root / 'sealed').mkdir()
        schedule = [dict(event=event, panel=panel['name'], request_id=row['request_id'],
                         ordinal=index, repeat=repeat, order=list(method_order(index, repeat)))
                    for event in (10, 12) for panel in manifest
                    if event == 12 or panel['name'] != 'final'
                    for index, row in enumerate(panel['requests']) for repeat in range(3)]
        schedule_hash = hashlib.sha256(json.dumps(schedule, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        (self.root / 'data_preparation').mkdir()
        (self.root / 'data_preparation/EVALUATION_SCHEDULE.json').write_text(json.dumps(schedule))
        (self.root / 'sealed/PLAN.json').write_text(json.dumps(dict(panels=manifest,
            initial=dict(K=dict(state_identity=dict(sha256='b' * 64))), evaluation_schedule_sha256=schedule_hash,
            execution_source=dict(commit='a' * 40, dirty=False))))
        qualification_inputs = dict(rows=['sealed_fixture'])
        (self.root / 'sealed/QUALIFICATION_INPUTS.json').write_text(json.dumps(qualification_inputs))
        campaign = self.root / 'campaign_001'
        campaign.mkdir()
        status = dict(status='completed', feedback_attempted=512, feedback_returned=512,
                      optimizer_steps=1536, presentations=3072, evaluation_attempted=4032,
                      evaluation_returned=4032, evaluation_valid=4032, optimizer_previews=1024,
                      credit_opportunities=504, arm_counters={arm:1536 for arm in ARMS},
                      arm_presentations={arm:3072 for arm in ARMS}, extra_optimizer_steps=0, extra_generations=0, additional_controller_model_passes=0)
        (campaign / 'STATUS.json').write_text(json.dumps(status))
        costs = {}
        for event in range(9, 13):
            for arm in ARMS:
                directory = campaign / arm / 'events' / f'event_{event:03d}'
                directory.mkdir(parents=True)
                training = 3. if arm == 'KW' else 2.
                cost = dict(training_s=training, event_wall_s=training + 1., checkpoint_save_s=.3, export_s=.2)
                receipt = dict(export_path=self.checkpoints[arm], costs=cost, event_id=f'event_{event:03d}',
                               status='completed', start_step=(event-1)*128, end_step=event*128,
                               optimizer_steps=128, response_presentations=256)
                if arm == 'KW':
                    receipt['kw'] = dict(windows=128, previews=256, credit_opportunities=126,
                                         controller_final=dict(n=(event-8)*126), costs=dict(previews_s=1.))
                (directory / 'RESULT.json').write_text(json.dumps(receipt))
                costs[f'event_{event:03d}_{arm}'] = cost
        (campaign / 'COSTS.json').write_text(json.dumps(costs))
        qualification = dict(status='passed', work=dict(response_forwards=8, response_backwards=8,
                             disposable_optimizer_applications=16, serving_generations=0, persistent_optimizer_updates=0),
                             native_neutral_exact=True, preview_isolation=True, full_model_memory_checked=True,
                             bindings=dict(source_k8_sha256='b' * 64, execution_commit='a' * 40,
                                 qualification_inputs_sha256=hashlib.sha256(json.dumps(qualification_inputs,
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()))
        (self.root / 'IMPLEMENTATION_QUALIFICATION.json').write_text(json.dumps(qualification))
        return manifest

    def test_all_4032_raw_rows_seven_panels_primary_pooled_and_costs(self):
        manifest = self.prepare_campaign()
        for event in (10, 12):
            for panel in manifest[:3 if event == 10 else 4]:
                name = panel['name']
                self.run_panel(event=event, panel=name, rows=panel['requests'],
                    name=f'campaign_001/event_{event:03d}/evaluation/{name}',
                    gate=8 if event == 10 and name == 'development' else 0)
        report = write_campaign_report(self.root, self.root / 'analysis', execution_commit='a' * 40)
        self.assertEqual(report['reaudit_returned_measurements'], 4032)
        self.assertEqual(report['reaudit_valid_timed_generations'], 4032)
        self.assertEqual(report['complete_valid_panels'], 7)
        self.assertEqual(report['pooled_final']['valid_requests'], 256)
        self.assertEqual(report['pooled_final']['K']['verification_calls'], 512)
        self.assertEqual(report['primary_bootstrap']['percentile_95_ci_pct'], [0., 0.])
        self.assertEqual(report['outcome'], 'Unsupported v1 in this run')
        self.assertEqual(report['correctness_errors'], dict(work=[], panels=[]))
        costs = self.read('analysis/COSTS.json')
        self.assertEqual(costs['KW_vs_K_training_overhead_pct'], 50.)
        self.assertEqual(costs['arm_totals']['KW']['training_s'], 12.)
        self.assertEqual(set(report['pooled_final']['panels']), {'development', 'additional', 'fresh', 'final'})
        self.assertTrue((self.root / 'analysis/ALL_REPEATS.csv').is_file())
        self.assertEqual(len((self.root / 'analysis/EVENT_PANEL.csv').read_text().splitlines()), 22)
        # Independently alter one saved schedule and one endpoint binding. Both become invalid.
        protocol_path = self.root / 'campaign_001/event_010/evaluation/additional/PROTOCOL.json'
        protocol = json.loads(protocol_path.read_text())
        protocol['schedule'][0]['method'] = 'KW'
        protocol_path.write_text(json.dumps(protocol))
        endpoint = self.root / 'campaign_001/KW/events/event_012/RESULT.json'
        endpoint.write_text(json.dumps(dict(export_path='/substituted')))
        bad = write_campaign_report(self.root, self.root / 'bad_analysis', execution_commit='a' * 40)
        self.assertEqual(bad['outcome'], 'Implementation invalid')
        self.assertEqual(bad['complete_valid_panels'], 2)
        self.assertIn('evaluation_protocol_binding', bad['panels']['10_additional']['binding_errors'])
        self.assertIn('own_event_endpoint_binding_KW', bad['panels']['12_final']['binding_errors'])
        self.assertIsNone(bad['primary_bootstrap'])

    def test_missing_execution_keeps_all_prespecified_requests_and_no_gain_claim(self):
        self.prepare_campaign()
        result = write_campaign_report(self.root, self.root / 'analysis', execution_commit='a' * 40)
        self.assertEqual(result['outcome'], 'Implementation invalid')
        self.assertEqual(result['complete_valid_panels'], 0)
        self.assertEqual(result['reaudit_returned_measurements'], 0)
        self.assertEqual(len(result['panels']), 7)
        self.assertTrue(all(panel['invalid_or_incomplete_requests'] == 64 for panel in result['panels'].values()))
        self.assertIsNone(result['primary_bootstrap'])


if __name__ == '__main__':
    unittest.main()
