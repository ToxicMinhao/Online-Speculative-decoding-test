"""CPU accounting and orchestration checks; these do not qualify native models."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .allocation_qualify import (
    ReferenceLedger,
    aggregate_effect,
    audit_window,
    direction_metadata,
    fp64_contractions,
    json_diagnostics,
    run_allocation_audit,
    serial_references,
)
from .math import allocate_prefix
from .native import WorkReceipt


class ScalarScorer:
    """Fresh scalar graphs let the driver prove order and normalization on CPU."""

    def __init__(self, output, *, change_rng=False, fail_token=False):
        self.output = Path(output)
        self.parameters = (torch.nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16)),)
        self.parameter_names = ('scalar',)
        self.change_rng = change_rng
        self.fail_token = fail_token
        self.events = []
        self.snapshots = []
        for _ in range(2):
            prefix = torch.zeros((1, 1, 16), dtype=torch.bool)
            prefix[..., 1:3] = True
            repair = torch.zeros_like(prefix)
            repair[..., 3] = True
            beta = (prefix | repair).float()
            roles = dict(
                prefix=prefix, repair=repair, beta=beta, denominator=beta.sum() + 1e-6
            )
            self.snapshots.append(dict(roles=roles, repair_coefficients=repair.float()))

    def freeze_current(self):
        self.events.append('freeze')
        return self.snapshots

    def reference_gradient(self, prefix_coefficients=None):
        if not (self.output / 'POOL_SELECTION.json').is_file():
            raise AssertionError('Selection must precede all gradient work')
        self.events.append('reference' if prefix_coefficients is None else 'aggregate')
        return (
            torch.tensor(
                2.0 if prefix_coefficients is None else 2.25, dtype=torch.bfloat16
            ),
        )

    def preview_and_direction(self, gradients):
        self.events.append('preview')
        return dict(w=(torch.tensor(1.0),), diagnostics={}, memory_inventory={})

    def fast_scores(self, direction):
        self.events.append('fast')
        if self.change_rng:
            torch.rand(())
        scores, allocations, diagnostics = [], [], []
        for index, snapshot in enumerate(self.snapshots):
            roles = snapshot['roles']
            value = torch.zeros_like(roles['beta'])
            for position in (1, 2):
                value[..., position] = -(index + position) / (2 * roles['denominator'])
            allocation, diagnostic = allocate_prefix(
                value, roles['beta'], roles['prefix']
            )
            scores.append(value)
            allocations.append(allocation)
            diagnostics.append(diagnostic)
        return dict(
            scores=scores,
            allocations=allocations,
            allocation_diagnostics=diagnostics,
            effective_direction=(direction[0].bfloat16(),),
            direction_scale=1.0,
            direct_loss_masks=[
                item['roles']['prefix'] | item['roles']['repair']
                for item in self.snapshots
            ],
            repair_coefficients=[
                item['repair_coefficients'] for item in self.snapshots
            ],
            response_denominators=[
                item['roles']['denominator'] for item in self.snapshots
            ],
            accumulation_factor=0.5,
            attention_work={},
        )

    def token_fkl(self, response_index, block_index, position):
        self.events.append(('token', response_index, block_index, position))
        if self.fail_token:
            raise RuntimeError('interrupted fresh native graph fixture')
        return self.parameters[0] * (response_index + position)


class AllocationAuditTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_reference_ledger_preserves_failed_allowance_and_bounds_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'ledger.json'
            first = ReferenceLedger(path, 1, root / 'first', {'parent': 'fixed'})
            first.bind_selection({'pools': [1]})
            for index in range(60):
                first.reserve({'position': index})
            with self.assertRaisesRegex(ValueError, 'exhausted'):
                first.reserve({'position': 60})
            first.finish('failed')
            with self.assertRaisesRegex(ValueError, 'restarted'):
                ReferenceLedger(path, 1, root / 'first', {'parent': 'fixed'})
            note = root / 'correction.txt'
            note.write_text('Identified implementation fix; thresholds unchanged.\n')
            second = ReferenceLedger(
                path,
                2,
                root / 'second',
                {'parent': 'fixed'},
                retry_of=root / 'first',
                correction_note=note,
            )
            with self.assertRaisesRegex(ValueError, 'selected'):
                second.bind_selection({'pools': [2]})
            second.bind_selection({'pools': [1]})
            for index in range(60):
                second.reserve({'position': index})
            with self.assertRaisesRegex(ValueError, 'exhausted'):
                second.reserve({'position': 60})
            saved = json.loads(path.read_text())
            self.assertEqual(
                sum(len(row['records']) for row in saved['attempts'].values()), 120
            )

    def test_retry_requires_existing_matching_parent_and_identified_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'new reference allowance'):
                ReferenceLedger(root / 'ledger.json', 2, root / 'two', {})
            ReferenceLedger(root / 'ledger.json', 1, root / 'one', {'parent': 'a'})
            with self.assertRaisesRegex(ValueError, 'different contract'):
                ReferenceLedger(root / 'ledger.json', 2, root / 'two', {'parent': 'b'})
            with self.assertRaisesRegex(ValueError, 'identified fix'):
                ReferenceLedger(root / 'ledger.json', 2, root / 'two', {'parent': 'a'})

    def test_chunked_fp64_dots_keep_primary_and_effective_directions_distinct(self):
        gradient = torch.tensor([0.1, -2.0, 0.003, 9.0], dtype=torch.bfloat16)
        primary = torch.tensor([0.3001, 0.7002, -0.4111, 0.00201])
        effective = primary.bfloat16()
        first, second = fp64_contractions(
            (gradient,), (primary,), (effective,), chunk_elements=2
        )
        self.assertEqual(first, float((gradient.double() * primary.double()).sum()))
        self.assertEqual(second, float((gradient.double() * effective.double()).sum()))
        metadata = direction_metadata(('w',), (primary,), (effective,), 1.0)
        self.assertFalse(metadata['effective_direction_equals_primary'])
        self.assertNotEqual(
            metadata['primary']['sha256'], metadata['effective']['sha256']
        )
        with self.assertRaisesRegex(ValueError, 'unconnected'):
            fp64_contractions((None,), (primary,), (effective,))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            fp64_contractions((gradient * float('nan'),), (primary,), (effective,))

    def test_full_audit_seals_first_runs_two_fast_calls_and_serial_unweighted_oracle(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scorer = ScalarScorer(root)
            ledger = ReferenceLedger(root / 'ledger.json', 1, root, {'parent': 'fixed'})
            work = WorkReceipt(root / 'WORK.json')
            verified = []
            result = audit_window(
                scorer,
                [SimpleNamespace(record_id=str(i)) for i in range(2)],
                ledger,
                work,
                root,
                verified.append,
            )
            self.assertEqual(result['status'], 'numerically_qualified')
            self.assertFalse(result['production_qualified'])
            self.assertEqual(
                scorer.events[:5], ['freeze', 'reference', 'preview', 'fast', 'fast']
            )
            self.assertEqual(
                scorer.events[5:9],
                [
                    ('token', 0, 0, 1),
                    ('token', 0, 0, 2),
                    ('token', 1, 0, 1),
                    ('token', 1, 0, 2),
                ],
            )
            self.assertEqual(scorer.events[-1], 'aggregate')
            self.assertEqual(
                verified,
                ['FAST_1_STATE.json', 'FAST_2_STATE.json', 'COMPLETE_AUDIT_STATE.json'],
            )
            self.assertEqual(work.value['completed']['serial_reference_token_vjp'], 4)
            self.assertEqual(work.value['native_adam_steps'], 0)
            self.assertFalse(work.value['pending'])
            self.assertIsNone(scorer.parameters[0].grad)
            rows = json.loads((root / 'REFERENCE_PROGRESS.json').read_text())['rows']
            self.assertEqual(rows[0]['primary_dot_fp64'], 1.0)
            self.assertEqual(
                rows[0]['reference_score_fp64'], -1.0 / (2 * rows[0]['denominator'])
            )
            self.assertTrue((root / 'REFERENCE_BINDING.json').is_file())

    def test_rng_change_is_rejected_before_any_reference_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scorer = ScalarScorer(root, change_rng=True)
            ledger = ReferenceLedger(root / 'ledger.json', 1, root, {})
            with self.assertRaisesRegex(ValueError, 'RNG'):
                audit_window(
                    scorer,
                    [SimpleNamespace(record_id=str(i)) for i in range(2)],
                    ledger,
                    WorkReceipt(root / 'WORK.json'),
                    root,
                    lambda _: None,
                )
            self.assertFalse(ledger.current['records'])

    def test_interrupted_token_graph_keeps_reserved_allowance_and_pending_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scorer = ScalarScorer(root, fail_token=True)
            ledger = ReferenceLedger(root / 'ledger.json', 1, root, {})
            work = WorkReceipt(root / 'WORK.json')
            selection = dict(
                pools=[dict(response_index=0, block_index=0, positions=[1, 2])],
                reference_backward_count=2,
            )
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                serial_references(
                    scorer,
                    scorer.snapshots,
                    selection,
                    (torch.tensor(1.0),),
                    (torch.tensor(1.0, dtype=torch.bfloat16),),
                    1.0,
                    ledger,
                    work,
                    root,
                )
            self.assertEqual(len(ledger.current['records']), 1)
            self.assertEqual(ledger.current['records'][0]['status'], 'graph_started')
            self.assertEqual(work.value['pending'], ['serial_reference_token_graph'])
            self.assertNotIn('serial_reference_token_vjp', work.value['attempted'])

    def test_aggregate_effect_promotes_before_difference_and_is_not_gate(self):
        scorer = ScalarScorer('.')
        fast = scorer.fast_scores((torch.tensor(1.0),))
        result = aggregate_effect(
            (torch.tensor(1.0001),),
            (torch.tensor(2.0, dtype=torch.bfloat16),),
            (torch.tensor(2.25, dtype=torch.bfloat16),),
            scorer.snapshots,
            fast,
        )
        self.assertEqual(
            result['native_effect_fp64'], -float(torch.tensor(1.0001).double()) * 0.25
        )
        self.assertEqual(result['native_gradient_difference_l2'], 0.25)
        self.assertFalse(result['admission_gate'])
        self.assertGreater(result['centered_score_prediction_fp64'], 0)
        json.dumps(json_diagnostics(fast['allocation_diagnostics']), allow_nan=False)

    def test_changed_contract_rejects_before_creating_output_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / 'contract.json'
            contract.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'sealed numerical contract'):
                run_allocation_audit(
                    'missing',
                    'missing',
                    root / 'output',
                    'commit',
                    contract,
                    root / 'ledger.json',
                    1,
                )
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
