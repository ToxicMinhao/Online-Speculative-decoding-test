"""Three-arm schedule, native failure retention, ratios and freeze decisions."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from experiments.repair_conditional_kl_pilot import evaluation as old_evaluation
from experiments.repair_conditional_kl_pilot.test_evaluation import Backend, Ledger, inputs
from experiments.repair_conditional_kl_pilot.test_analysis import fixture as pair_fixture
from experiments.frontier_preservation_pilot import point3_evaluation as evaluation
from experiments.frontier_preservation_pilot.point3_analysis import summarize, freeze_decision, reduce_requests, exported_parameters_identity, complete_generation_accounting
from experiments.frontier_preservation_pilot.point3_readout import _dual_chunk, _boundary, evaluate_responses, compare_endpoints
from experiments.frontier_preservation_pilot.native import rng_digest
from online_feedback.rng import capture_rng
from test_point1_readout import fixture as native_fixture


def measurements():
    rows = []
    for row in pair_fixture():
        row = deepcopy(row)
        row['method'] = {'BASE': 'C', 'PAIR': 'F_C_SUM'}[row['method']]
        row['checkpoint_id'] = row['method']
        rows.append(row)
        if row['method'] == 'F_C_SUM':
            maximum = deepcopy(row)
            maximum['method'] = maximum['checkpoint_id'] = 'F_C_MAX'
            maximum['request_latency_s'] *= .9
            maximum['native_generate_s'] *= .9
            rows.append(maximum)
    return rows


def identities():
    return {method: dict(parameters=method, inference='same-qualified-configuration')
            for method in evaluation.METHODS}


def run(root, backend):
    panel, warmups, canonicals, warmup_canonicals = inputs()
    ledger = Ledger()
    def activate(method, checkpoint, slot):
        backend.checkpoint = checkpoint
        return backend, {}
    status = evaluation.execute(
        dict(execution_commit='fixture', devices={'A': 'GPUa'}, canonical_provenance={'source': 'PR56'}),
        root / 'evaluation', panel, warmups, canonicals, warmup_canonicals,
        {method: method + '-export' for method in evaluation.METHODS}, activate=activate,
        boundary=lambda entry: dict(quiet=True, entry=entry), ledger=ledger,
    )
    return status, ledger


class ThreeArmExecutionTests(unittest.TestCase):
    def test_exact_schedule_and_per_request_positions(self):
        panel, *_ = inputs()
        schedule = evaluation.evaluation_schedule(panel)
        receipt = evaluation.schedule_summary(schedule)
        self.assertEqual((receipt['activations'], receipt['scored_generations'], receipt['warmup_generations']),
                         (18, 1152, 36))
        positions = Counter((request, row['method'], row['position'])
                            for row in schedule for request in row['request_ids'])
        self.assertEqual(len(positions), 1152)
        self.assertEqual(set(positions.values()), {1})
        self.assertEqual([row['method'] for row in schedule[3:6]], ['F_C_SUM', 'F_C_MAX', 'C'])
        self.assertEqual(receipt['first_position_requests_per_repeat'],
                         {'C': [64, 0, 64], 'F_C_SUM': [64, 64, 0], 'F_C_MAX': [0, 64, 64]})
        bad = deepcopy(schedule)
        bad[4]['method'] = 'C'
        with self.assertRaises(ValueError):
            evaluation.schedule_summary(bad)
        with self.assertRaises(ValueError):
            old_evaluation.evaluation_schedule(panel, methods=evaluation.METHODS)

    def test_all_1188_returned_records_are_validated_and_accounted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            status, ledger = run(root, Backend())
            self.assertEqual(status['completed_valid'], 1188)
            self.assertEqual(ledger.counts, {'scored_generation': 1152, 'warmup_generation': 36})
            self.assertEqual((status['invalid'], status['unknown'], status['unattempted']), (0, 0, 0))
            rows = [json.loads(line) for line in (root / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
            report = summarize(rows)
            self.assertEqual(len(report['contrasts']), 3)
            self.assertEqual(set(report['totals']), set(evaluation.METHODS))
            self.assertEqual(report['requests'], 128)
            with self.assertRaises(ValueError):
                summarize([row for row in rows if row['phase'] != 'warmup'])

    def test_offline_accounting_rejects_duplicate_warmup_and_cross_activation_group_changes(self):
        panel, warmups, *_ = inputs()
        rows = []
        for entry in evaluation.evaluation_schedule(panel):
            for phase, identifiers in (
                    ('warmup', [row['request_id'] for row in warmups]),
                    ('scored', entry['request_ids'])):
                rows.extend(dict(activation=entry['activation'], repeat=entry['repeat'],
                    group=entry['group'], method=entry['method'], phase=phase, request_id=request_id)
                    for request_id in identifiers)
        self.assertEqual(complete_generation_accounting(rows)['returned'], 1188)
        duplicated = deepcopy(rows)
        duplicated[1]['request_id'] = duplicated[0]['request_id']
        with self.assertRaisesRegex(ValueError, 'both warmup identities'):
            complete_generation_accounting(duplicated)
        # Swap requests across C's two groups in one repeat. Per-method request
        # coverage stays intact; only the required stable grouping is broken.
        swapped = deepcopy(rows)
        first = next(row for row in swapped if row['activation'] == 0 and row['phase'] == 'scored')
        second = next(row for row in swapped if row['activation'] == 5 and row['phase'] == 'scored')
        first['request_id'], second['request_id'] = second['request_id'], first['request_id']
        with self.assertRaisesRegex(ValueError, 'group identities changed'):
            complete_generation_accounting(swapped)
        overlapping = deepcopy(rows)
        for row in overlapping:
            if row['group'] == 1 and row['phase'] == 'scored':
                row['request_id'] = panel[int(row['request_id'][1:]) - 64]['request_id']
        with self.assertRaisesRegex(ValueError, 'disjoint scored groups'):
            complete_generation_accounting(overlapping)

    def test_real_entry_connects_three_authenticated_exports_and_closes_before_analysis(self):
        from experiments.frontier_preservation_pilot.point3_prepare import SCHEMA
        panel, warmups, references, warmup_references = inputs()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipts = {}
            for method in evaluation.METHODS:
                export = root / method / 'export'
                export.mkdir(parents=True)
                from safetensors.torch import save_file
                save_file({'weight': torch.tensor([evaluation.METHODS.index(method)])},
                          str(export / 'model.safetensors'))
                recovery = root / method / 'recovery.pt'
                recovery.write_text(method + '-state')
                receipt = dict(status='completed', global_step=256, presentations=512,
                    completed_event=2, checkpoint=str(export), recovery=str(recovery),
                    checkpoint_identity=identity(export / 'model.safetensors'),
                    recovery_identity=identity(recovery))
                status = root / 'arms' / method / 'STATUS.json'
                write_json(status, receipt)
                receipts[method] = status
            qualification = root / 'qualification.json'
            write_json(qualification, dict(status='completed', native_reference={'status': 'passed'},
                                           qualification={'status': 'passed'}))
            canonical, warmup_canonical = root / 'canonicals.json', root / 'warmups.json'
            write_json(canonical, references)
            write_json(warmup_canonical, warmup_references)
            production = root / 'production.json'
            write_json(production, {})
            plan = dict(schema=SCHEMA, production_admitted=True, root=str(root),
                execution_commit='fixture', devices={'A': 'GPUa', 'B': 'GPUb'},
                qualification_result=str(qualification), retained_C_status=str(receipts['C']),
                retained_C_status_identity=identity(receipts['C']), production_plan=str(production),
                canonicals=str(canonical), warmup_canonicals=str(warmup_canonical),
                canonical_provenance={'source': 'PR56'}, serving={'block_size': 16})
            backend = Backend()
            backend.loading_s, backend.failed, backend.closed = 1., False, False
            def publish(checkpoint):
                backend.checkpoint = checkpoint
                return None
            backend.publish = publish
            backend._call = lambda operation: {'operation': operation}
            backend.close = lambda: setattr(backend, 'closed', True)
            native_inputs = dict(evaluation_requests=panel, warmup_requests=warmups)
            with patch.dict('os.environ', CUDA_VISIBLE_DEVICES='GPUa'), \
                 patch('experiments.repair_conditional_kl_pilot.campaign.create_backend', return_value=backend), \
                 patch('experiments.repair_conditional_kl_pilot.production.quiet_evaluation', return_value={'quiet': True}), \
                 patch('experiments.frontier_preservation_pilot.allocation_campaign.selected_inputs', return_value=(native_inputs, [])), \
                 patch('experiments.frontier_preservation_pilot.point3_launch.boundary', return_value={'admitted': True}):
                result = evaluation.run_evaluation(plan)
            self.assertEqual(result['completed_valid'], 1188)
            self.assertTrue(backend.closed)
            summary = json.loads((root / 'evaluation/SUMMARY.json').read_text())
            self.assertEqual(summary['generation_accounting']['returned'], 1188)
            self.assertEqual(summary['freeze']['selected'], 'C')
            self.assertEqual(summary['serving_costs']['loading_s'], 1.)
            self.assertAlmostEqual(summary['serving_costs']['warmup_request_latency_s'], 7.2)
            self.assertEqual(set(summary['serving_identities']), set(evaluation.METHODS))

    def test_failure_and_unknown_retention_use_three_arm_budget(self):
        for unknown in (False, True):
            with self.subTest(unknown=unknown), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with self.assertRaises((RuntimeError, ValueError)):
                    run(root, Backend(transport_failure=unknown, malformed=not unknown))
                status = json.loads((root / 'evaluation/STATUS.json').read_text())
                self.assertEqual(status['unattempted'], 1187)
                self.assertEqual(status['unknown'], int(unknown))
                self.assertEqual(status['invalid'], int(not unknown))
                self.assertEqual(status['returned'], int(not unknown))
                if not unknown:
                    self.assertEqual(len(list((root / 'evaluation/raw').glob('*.json'))), 1)


class ThreeArmAnalysisTests(unittest.TestCase):
    def test_same_bootstrap_draws_and_whole_request_means(self):
        result = summarize(measurements(), expected_requests=2)
        self.assertEqual(result['totals']['C']['request_latency_s'], 6)
        self.assertEqual(result['totals']['F_C_SUM']['request_latency_s'], 4)
        self.assertAlmostEqual(result['totals']['F_C_MAX']['request_latency_s'], 3.6)
        draws = np.random.Generator(np.random.PCG64(20261125)).integers(0, 2, size=(10000, 2))
        self.assertEqual(result['bootstrap']['draw_sha256'], hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest())
        for candidate, reference in (('F_C_MAX', 'F_C_SUM'), ('F_C_SUM', 'C'), ('F_C_MAX', 'C')):
            first = np.array([row[candidate]['request_latency_s'] for row in result['paired']])
            second = np.array([row[reference]['request_latency_s'] for row in result['paired']])
            expected = np.percentile(100 * (1 - first[draws].sum(1) / second[draws].sum(1)),
                                     [2.5, 97.5], method='linear').tolist()
            self.assertEqual(result['contrasts'][candidate + '_vs_' + reference]['request_latency_s_reduction_pct']['ci95'], expected)

    def test_invalid_repeated_counts_or_incomplete_arm_never_enter_gains(self):
        for mutation in ('missing', 'count', 'invalid', 'time', 'identity'):
            rows = measurements()
            if mutation == 'missing':
                rows.pop()
            elif mutation == 'count':
                rows[-1]['verification_calls'] += 1
            elif mutation == 'invalid':
                rows[0]['errors'] = ['invalid']
            elif mutation == 'time':
                rows[0]['request_latency_s'] = float('nan')
            else:
                rows[0]['checkpoint_id'] = 'changed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reduce_requests(rows, expected_requests=2)

    def test_freeze_selection_never_uses_identical_model_timing_noise(self):
        report = summarize(measurements(), expected_requests=2)
        self.assertEqual(freeze_decision(report, identities(), authenticated_complete=True)['selected'], 'F_C_MAX')
        same = identities()
        same['F_C_MAX'] = same['F_C_SUM'].copy()
        decision = freeze_decision(report, same, authenticated_complete=True)
        self.assertEqual(decision['selected'], 'F_C_SUM')
        self.assertTrue(decision['candidates_have_identical_serving_models'])
        same['C'] = same['F_C_SUM'].copy()
        self.assertEqual(freeze_decision(report, same, authenticated_complete=True)['selected'], 'C')
        self.assertEqual(freeze_decision(None, None, authenticated_complete=False)['selected'], 'C')

    def test_parameter_identity_excludes_safetensors_metadata_and_header_order(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / 'first.safetensors', Path(folder) / 'second.safetensors'
            a, b = torch.tensor([1., 2.], dtype=torch.bfloat16), torch.tensor(3.)
            save_file({'a': a, 'b': b}, str(first), metadata={'run': 'one'})
            save_file({'b': b, 'a': a}, str(second), metadata={'run': 'other'})
            self.assertNotEqual(identity(first), identity(second))
            self.assertEqual(exported_parameters_identity(first), exported_parameters_identity(second))
            save_file({'a': a + 1, 'b': b}, str(second))
            self.assertNotEqual(exported_parameters_identity(first), exported_parameters_identity(second))

    def test_freeze_uses_unrounded_strict_latency_calls_and_ci_scope(self):
        report = summarize(measurements(), expected_requests=2)
        for method in evaluation.METHODS[1:]:
            report['totals'][method]['request_latency_s'] = 6 - 1e-10
            report['totals'][method]['verification_calls'] = 10
            report['contrasts'][method + '_vs_C']['request_latency_s_reduction_pct']['ci95'] = [-1., 1.]
        decision = freeze_decision(report, identities(), authenticated_complete=True)
        self.assertEqual(decision['selected'], 'F_C_SUM')
        self.assertEqual(decision['status'], 'provisional_development_selection')
        report['totals']['F_C_SUM']['verification_calls'] = 11
        self.assertEqual(freeze_decision(report, identities(), authenticated_complete=True)['selected'], 'F_C_MAX')
        report['totals']['F_C_MAX']['request_latency_s'] = 6
        self.assertEqual(freeze_decision(report, identities(), authenticated_complete=True)['selected'], 'C')


class DualAssessmentReadoutTests(unittest.TestCase):
    def test_single_forward_dual_readout_and_functional_endpoint_isolation(self):
        trainer, response = native_fixture(9)
        _, empty = native_fixture(1, 'empty')
        empty.tensors['loss_mask'].zero_()
        before_rng = rng_digest(capture_rng())
        model = trainer.bundle.model
        parameter = model.draft_model.weight
        native = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter)
        old_gradient = parameter.grad
        parent = evaluate_responses(trainer, [response, empty], endpoint='parent')
        self.assertEqual((parent['draft_forwards'], parent['head_calls']), (2, 3))
        self.assertEqual(model.head_shapes, [(1, 8, 16, 32), (1, 1, 16, 32), (1, 1, 16, 32)])
        for assessment in ('SUM', 'MAX'):
            self.assertEqual(parent['responses'][1]['assessments'][assessment]['fp32'], 0)
            self.assertEqual(parent['assessments'][assessment]['fp32'],
                             parent['responses'][0]['assessments'][assessment]['fp32'] / 2)
        changed = native.float().clone()
        changed[4, 31] = 3
        candidate = evaluate_responses(trainer, [response, empty], endpoint='F_C_MAX',
            origins=parent, parameters={'model.draft_model.weight': changed})
        comparison = compare_endpoints(parent, candidate)
        for assessment in ('SUM', 'MAX'):
            self.assertGreater(comparison['assessments'][assessment]['benefit_fp64'], 0)
        self.assertTrue(torch.equal(parameter, native))
        self.assertIs(parameter.grad, old_gradient)
        self.assertEqual(rng_digest(capture_rng()), before_rng)
        self.assertEqual(trainer.work, {'old': 7})
        self.assertEqual(model.scratch.item(), 7)
        self.assertEqual(model._cached_seq_len, 11)

    def test_same_margin_terms_keep_sum_and_max_distinct(self):
        blocks = []
        for margins in ([2., -1., -1., .5], []):
            prefix, repair = [False] * 16, [False] * 16
            prefix[1], repair[2] = bool(margins), bool(margins)
            blocks.append(dict(origin=dict(K=len(margins), prefix=prefix, repair=repair),
                               positions=[dict(position=i + 1, margin=m) for i, m in enumerate(margins)]))
        sum_value, max_value = _dual_chunk(blocks, torch.device('cpu'))
        phi = torch.nn.functional.softplus(-torch.tensor([2., -1., -1., .5])) / np.log(2.)
        self.assertEqual(float(sum_value), float((torch.tensor([4., 3., 2., 1.]) * torch.nn.functional.softplus(-torch.tensor([2., -1., -1., .5])) / np.log(2.)).sum()))
        self.assertEqual(float(max_value), float(phi[[0, 1, 1, 1]].sum()))
        self.assertEqual([p['bottleneck_position'] for p in blocks[0]['positions']], [1, 2, 2, 2])
        self.assertEqual([p['bottleneck_multiplicity'] for p in blocks[0]['positions']], [1, 3, 0, 0])
        self.assertEqual(blocks[0]['active_bottlenecks_by_role']['C']['multiplicity'], 0)
        self.assertEqual(blocks[1]['max_assessment_sum_fp64'], 0)

    def test_production_boundary_is_explicit_and_keeps_accumulation_guard(self):
        class Trainer:
            global_step, presentations, events = 256, 512, [{}, {}]
            class core:
                accumulation_remainder = 0
        self.assertEqual(_boundary(Trainer()), (256, 512, 2))
        trainer = Trainer()
        trainer.presentations = 513
        with self.assertRaises(ValueError):
            _boundary(trainer)


if __name__ == '__main__':
    unittest.main()
