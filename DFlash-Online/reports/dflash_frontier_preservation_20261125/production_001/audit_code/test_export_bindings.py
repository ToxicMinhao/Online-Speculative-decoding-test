"""Offline adversarial binding and paired-bootstrap fixtures; no model imports."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import export_results as audit
from test_export_results import measurements


class BindingFixture:
    def __init__(self, directory):
        self.root = Path(directory) / 'raw'
        self.dependencies = Path(directory) / 'dependencies'
        self.root.mkdir()
        self.dependencies.mkdir()
        self.plan = dict(root='/sealed-run', sealed_files={})
        self.inputs = dict(
            panel=[
                dict(request_id=f'eval-{i}', request_index=i, prompt_ids=[100 + i])
                for i in range(2)
            ],
            warmup=[
                dict(request_id=f'warm-{i}', request_index=i, prompt_ids=[200 + i])
                for i in range(2)
            ],
        )
        self.selection = dict(
            evaluation_request_ids=[r['request_id'] for r in self.inputs['panel']],
            warmup_request_ids=[r['request_id'] for r in self.inputs['warmup']],
            training=[
                dict(local_ordinal=i, source_index=i + 128, request_id=f'new-{i}')
                for i in range(128)
            ],
        )
        self.seal('INPUT_SELECTION.json', self.selection)
        old_inputs = dict(
            evaluation_requests=self.inputs['panel'],
            warmup_requests=self.inputs['warmup'],
        )
        old_identity = self.write_dependency('PARENT_INPUTS.json', old_inputs)
        self.plan['sealed_files'][
            audit.PARENT_ROOT + '/sealed/INPUTS.json'
        ] = old_identity
        self.references = {}
        for group, name in [
            ('panel', 'CANONICALS.json'),
            ('warmup', 'WARMUP_CANONICALS.json'),
        ]:
            canonical = {
                row['request_id']: dict(
                    prompt_ids=row['prompt_ids'], response_ids=[4], finish_reason='eos'
                )
                for row in self.inputs[group]
            }
            self.seal(
                name, {key: dict(AR=value, F=value) for key, value in canonical.items()}
            )
            self.references[group] = canonical
        self.checkpoints = {arm: f'{arm}-export-sha' for arm in audit.ARMS}
        self.protocol = dict(
            checkpoints=deepcopy(self.checkpoints), execution_commit=audit.COMMIT
        )
        self.shared = [
            dict(
                path=f'/features/new/{i}.pt',
                identity=dict(bytes=10, sha256=f'new-{i}'),
                record_id=f'new-record-{i}',
                local_ordinal=i,
                source_index=i + 128,
                source_request_id=f'new-{i}',
            )
            for i in range(128)
        ]
        self.parent = [
            dict(
                path=f'/features/parent/{i}.pt',
                identity=dict(bytes=10, sha256=f'parent-{i}'),
            )
            for i in range(128)
        ]
        self.write('shared/FEATURE_IDENTITIES.json', self.shared)
        self.parent_identity = self.write_dependency(
            'PARENT_FEATURE_IDENTITIES.json', self.parent
        )
        self.binding = dict(
            schema='frontier_fixed_training_witness_v1',
            status='passed',
            origin_counters=[128, 256, 1],
            live_rng_before='same',
            live_rng_after='same',
            diagnostic_forwards=0,
            optimizer_steps=0,
            maximum_anchors_per_response=8,
            diagnostic_only=True,
            sets={},
        )
        for name, features in [
            ('new_buffer', self.shared),
            ('parent_training', self.parent),
        ]:
            self.binding['sets'][name] = [
                dict(
                    ordinal=i,
                    feature_path=row['path'],
                    feature_identity=row['identity'],
                    anchors=[2, 4],
                    record_id=row.get('record_id', f'parent-record-{i}'),
                )
                for i, row in enumerate(features[:8])
            ]
        self.save_binding()
        self.write('sealed/PLAN.json', self.plan)

    def write_at(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(payload)
        return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def write(self, name, value):
        return self.write_at(self.root / name, value)

    def write_dependency(self, name, value):
        return self.write_at(self.dependencies / name, value)

    def seal(self, name, value):
        self.plan['sealed_files'][self.plan['root'] + '/sealed/' + name] = self.write(
            'sealed/' + name, value
        )

    def save_binding(self):
        identity = self.write('witness/BINDINGS.json', self.binding)
        self.receipts = {
            endpoint: dict(
                schema=self.binding['schema'],
                diagnostic_only=True,
                binding_identity=identity,
                origin_digest=None if endpoint == 'parent' else 'parent-digest',
                sets={
                    name: [
                        dict(record_id=item['record_id'], anchors=item['anchors'])
                        for item in items
                    ]
                    for name, items in self.binding['sets'].items()
                },
            )
            for endpoint in ('parent', *audit.ARMS)
        }

    def reader(self):
        return audit.Inputs(self.root)

    def validate_serving(self):
        return audit.validate_serving_bindings(
            self.reader(),
            self.plan,
            self.inputs,
            self.references,
            self.protocol,
            self.checkpoints,
        )

    def validate_witness(self):
        with patch.object(audit, 'PARENT_FEATURE_IDENTITY', self.parent_identity):
            return audit.audit_witness_bindings(self.reader(), self.receipts)


def valid_costs():
    plan = dict(prior_occupied_card_s=550.701053057)
    phases = ['collect', 'train_CONTROL', 'train_FRONTIER_P', 'evaluate']
    clock = dict(
        active=None,
        completed_card_s=400.0,
        finished=[
            dict(phase=name, status='completed', elapsed_s=100.0) for name in phases
        ],
    )
    launch = dict(
        status='completed',
        clock=dict(
            active=None,
            production_occupied_card_s=400.0,
            prior_occupied_card_s=plan['prior_occupied_card_s'],
            cumulative_occupied_card_s=plan['prior_occupied_card_s'] + 400.0,
            new_phase_elapsed_s=5000.0,
        ),
    )
    work = audit.counts_receipt(
        dict(attempted={'cache': 1}, completed={'cache': 1}, pending={})
    )
    windows = [
        dict(
            window=i,
            outcome_window=i,
            status='completed',
            native_adam_steps=1,
            work=deepcopy(work),
            native=dict(
                start_step=127 + i,
                end_step=128 + i,
                presentations=2,
                native_adam_steps=1,
            ),
            score_s=17.0,
            total_window_s=22.0,
        )
        for i in range(1, 129)
    ]
    return plan, launch, clock, windows


class BindingTests(unittest.TestCase):
    def test_authorized_seals_and_exact_endpoint_bindings_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = BindingFixture(directory)
            fixture.validate_serving()
            fixture.validate_witness()

    def test_served_export_and_canonical_substitution_fail(self):
        for mutation in (
            'export',
            'canonical',
            'request_order',
            'unsealed_bytes',
            'dependency',
        ):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as directory:
                fixture = BindingFixture(directory)
                if mutation == 'export':
                    fixture.protocol['checkpoints']['FRONTIER_P'] = 'other-valid-export'
                elif mutation == 'canonical':
                    fixture.references['panel']['eval-0']['response_ids'] = [999]
                elif mutation == 'request_order':
                    fixture.inputs['panel'].reverse()
                elif mutation == 'unsealed_bytes':
                    fixture.write('sealed/CANONICALS.json', {})
                else:
                    fixture.write_dependency('PARENT_INPUTS.json', {})
                with self.assertRaises(ValueError):
                    fixture.validate_serving()

    def test_witness_actual_binding_identity_and_original_order_are_required(self):
        for mutation in (
            'unbound_bytes',
            'anchors',
            'source_order',
            'overlap',
            'parent_dependency',
        ):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as directory:
                fixture = BindingFixture(directory)
                if mutation == 'unbound_bytes':
                    fixture.binding['sets']['new_buffer'][0]['anchors'] = [3, 4]
                    fixture.write('witness/BINDINGS.json', fixture.binding)
                elif mutation == 'anchors':
                    fixture.receipts['CONTROL']['sets']['new_buffer'][0]['anchors'] = [
                        3,
                        4,
                    ]
                elif mutation == 'source_order':
                    fixture.shared[0]['source_index'] = 255
                    fixture.write('shared/FEATURE_IDENTITIES.json', fixture.shared)
                elif mutation == 'overlap':
                    fixture.binding['sets']['parent_training'][0]['record_id'] = (
                        fixture.binding['sets']['new_buffer'][0]['record_id']
                    )
                    fixture.save_binding()
                else:
                    fixture.write_dependency(
                        'PARENT_FEATURE_IDENTITIES.json', list(reversed(fixture.parent))
                    )
                with self.assertRaises(ValueError):
                    fixture.validate_witness()


class CostTests(unittest.TestCase):
    def test_complete_phase_and_window_receipts_pass(self):
        self.assertTrue(audit.validate_cost_receipts(*valid_costs())['passed'])

    def test_cost_reconciliation_rejects_incomplete_and_double_counted_work(self):
        for mutation in (
            'duplicate_window',
            'wrong_outcome_window',
            'wrong_native_step',
            'missing_work',
            'score_exceeds_window',
            'nan_time',
            'bool_time',
            'phase_sum',
            'phase_order',
            'active_phase',
            'wrong_prior',
            'wrong_cumulative',
            'elapsed_limit',
            'card_limit',
        ):
            with self.subTest(mutation=mutation):
                plan, launch, clock, windows = valid_costs()
                if mutation == 'duplicate_window':
                    windows[-1]['window'] = 127
                elif mutation == 'wrong_outcome_window':
                    windows[0]['outcome_window'] = 2
                elif mutation == 'wrong_native_step':
                    windows[0]['native']['start_step'] = 0
                elif mutation == 'missing_work':
                    windows[0]['work'] = audit.counts_receipt({})
                elif mutation == 'score_exceeds_window':
                    windows[0]['score_s'] = 23.0
                elif mutation == 'nan_time':
                    windows[0]['score_s'] = float('nan')
                elif mutation == 'bool_time':
                    windows[0]['score_s'] = True
                elif mutation == 'phase_sum':
                    clock['completed_card_s'] += 100
                elif mutation == 'phase_order':
                    clock['finished'].reverse()
                elif mutation == 'active_phase':
                    clock['active'] = dict(phase='evaluate')
                elif mutation == 'wrong_prior':
                    launch['clock']['prior_occupied_card_s'] = 0
                elif mutation == 'wrong_cumulative':
                    launch['clock']['cumulative_occupied_card_s'] += 100
                elif mutation == 'elapsed_limit':
                    launch['clock']['new_phase_elapsed_s'] = 14400.01
                elif mutation == 'card_limit':
                    plan['prior_occupied_card_s'] = 28401.0
                    launch['clock']['prior_occupied_card_s'] = 28401.0
                    launch['clock']['cumulative_occupied_card_s'] = 28801.0
                result = audit.validate_cost_receipts(plan, launch, clock, windows)
                self.assertFalse(result['passed'])
                self.assertTrue(result['failures'])

    def test_missing_or_pending_counts_are_not_recorded(self):
        for value in (
            None,
            {},
            dict(attempted={'x': 1}, completed={}, pending={}),
            dict(attempted={'x': True}, completed={'x': True}, pending={}),
        ):
            with self.subTest(value=value):
                self.assertNotEqual(audit.counts_receipt(value)['status'], 'recorded')


def correct_parent_block():
    labels = list(range(16))
    valid = [False] + [True] * 3 + [False] * 12
    roles = dict(
        labels=labels,
        valid=valid,
        prefix=list(valid),
        repair=[False] * 16,
        L=3,
        e=-1,
        H=0,
        K=3,
        m=3,
    )
    points = [
        dict(
            position=j,
            canonical=j,
            rival=20,
            origin_rival=20,
            prediction=j,
            correct=True,
            margin=1.0,
            origin_rival_margin=1.0,
            margin_tie=False,
            wrong_rival_tie=False,
            rival_changed=False,
            origin_rival_addressed=True,
        )
        for j in range(1, 4)
    ]
    value = float(
        (
            np.array([3, 2, 1], dtype=np.float32)
            * np.logaddexp(np.float32(0), np.float32(-1))
            / np.float32(math.log(2))
        ).sum(dtype=np.float32)
    )
    return dict(
        origin=deepcopy(roles),
        current=deepcopy(roles),
        proposals=labels,
        positions=points,
        metrics=dict(
            raw_repair=0,
            useful_repair=0,
            damaged_prefix_positions=0,
            damaged_prefix_blocks=0,
            surviving_followers=0,
            reachable_followers=0,
            matching_prefix_length=3,
            fixed_K_shortfall=0,
            assessment=value,
        ),
    )


class IndependentScalarTests(unittest.TestCase):
    def test_bootstrap_uses_request_pairs_and_actual_prespecified_draws(self):
        rows = measurements()
        for row in rows:
            if row['method'] == 'FRONTIER_P':
                ordinal = int(row['request_id'].rsplit('-', 1)[1])
                # Nonconstant effects make the distribution sensitive to pairing,
                # request averaging, and the actual random draws.
                row['request_latency_s'] += (ordinal % 7 - 3) * 0.08
                row['native_generate_s'] = row['request_latency_s'] * 0.9
        result = audit.serving(rows)
        ids = sorted({row['request_id'] for row in rows})
        means = {
            arm: np.array(
                [
                    sum(
                        row['request_latency_s']
                        for row in rows
                        if row['request_id'] == key and row['method'] == arm
                    )
                    / 3
                    for key in ids
                ]
            )
            for arm in audit.ARMS
        }
        draws = np.random.Generator(np.random.PCG64(20261125)).integers(
            0, 128, size=(10000, 128)
        )
        expected = 100 * (
            1 - means['FRONTIER_P'][draws].sum(1) / means['CONTROL'][draws].sum(1)
        )
        interval = result['contrasts']['FRONTIER_P_vs_CONTROL'][
            'request_latency_s_reduction_pct'
        ]
        self.assertAlmostEqual(
            interval['value'],
            100 * (1 - means['FRONTIER_P'].sum() / means['CONTROL'].sum()),
            places=12,
        )
        np.testing.assert_array_equal(
            interval['ci95'], np.quantile(expected, [0.025, 0.975], method='linear')
        )
        self.assertGreater(interval['ci95'][1] - interval['ci95'][0], 0)
        self.assertEqual(
            result['bootstrap']['draw_sha256'],
            hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
        )

    def test_origin_roles_cannot_be_replaced_by_self_consistent_fiction(self):
        good = correct_parent_block()
        audit.witness_metrics(good, good)
        bad = deepcopy(good)
        bad['origin'].update(
            prefix=[False, True] + [False] * 14,
            repair=[False, False, True] + [False] * 13,
            L=1,
            e=2,
            H=2,
        )
        # Downstream metrics are internally consistent with these invented roles.
        bad['metrics'].update(
            raw_repair=1,
            useful_repair=1,
            surviving_followers=1,
            reachable_followers=1,
        )
        with self.assertRaises(ValueError):
            audit.witness_metrics(bad, bad)


if __name__ == '__main__':
    unittest.main()
