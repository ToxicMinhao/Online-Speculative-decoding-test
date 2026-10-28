"""CPU-only tests for independent audit arithmetic and cost boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

import audit_results as audit


class AuditTests(unittest.TestCase):
    def test_native_fixture_and_report_arithmetic_are_independently_rechecked(self):
        sys.path[:0] = [str(audit.BASE / 'worktree'), str(audit.BASE / 'worktree/tests')]
        from experiments.prefix_ranking.evaluate import evaluate
        from test_osd_one_event import Backend, request

        def complete_trace(record, options):
            for block in record['metadata']['trace']:
                first, second = block['committed_tokens']
                block.update(draft_tokens=[first] + [777] * 14,
                             target_argmax=[first, second] + [0] * 14,
                             verify_size=16, raw_accepted_draft_tokens_before_stop=1,
                             positions=list(range(2 + block['output_start'], 18 + block['output_start'])))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = [request(i) for i in range(64)]
            checkpoints = {method: '/' + method for method in audit.ARMS}
            counts = {row['request_id']: {'F': 2, 'O': 2} for row in requests}
            evaluate(Backend(requests, complete_trace), requests, checkpoints, root / 'evaluation', ordinary_counts=counts)
            measurements = audit.lines(root / 'evaluation/MEASUREMENTS.jsonl')
            original = [row for row in measurements if row['method'] in ('F', 'O')]
            prior = audit.load_prior(audit.BASE / 'audit_results.py')
            issues = []
            result = audit.native_and_report(root, requests, original, prior, issues)
            self.assertEqual(issues, [])
            self.assertEqual(result['raw_measurements'], 768)
            self.assertEqual(result['native_blocks'], 1536)
            self.assertEqual(result['valid_common_requests'], 64)
            summary_path = root / 'evaluation/report/SUMMARY.json'
            summary = audit.read(summary_path)
            summary['comparisons']['P_vs_O']['call_net_saved'] = 1
            summary_path.write_text(json.dumps(summary))
            audit.native_and_report(root, requests, original, prior, issues)
            self.assertIn('Independent comparison arithmetic: P_vs_O.call_net_saved', issues)
            summary['comparisons']['P_vs_O']['call_net_saved'] = 0
            summary_path.write_text(json.dumps(summary))
            execution_path = root / 'evaluation/STATUS.json'
            execution = audit.read(execution_path)
            execution['status'] = 'failed'
            execution_path.write_text(json.dumps(execution))
            issues = []
            audit.native_and_report(root, requests, original, prior, issues)
            self.assertIn('Evaluation is not complete with all 768 valid slots and 64 paired requests', issues)

    def test_qualification_reuse_is_bound_to_sealed_dependencies(self):
        receipt = Path('/qualified/qualification/RESULT.json')
        reuse = dict(qualification_receipt=str(receipt), exact_gate_sha256={'Oa/initial_vs_pinned.json': 'gate'},
                     qualification_receipt_sha256='receipt', qualification_seal_sha256='seal',
                     qualification_runner_sha256='runner', pinned_initial_sha256='initial')
        dependencies = {str(receipt): 'receipt', '/qualified/SEAL.json': 'seal', '/qualified/runner.py': 'runner',
                        '/qualified/qualification/Oa/initial_vs_pinned.json': 'gate', '/diagnostic/trials/O2/initial.pt': 'initial'}
        issues = []
        audit.qualification_bindings(reuse, dict(dependency_sha256=dependencies), issues)
        self.assertEqual(issues, [])
        reuse['exact_gate_sha256']['Oa/initial_vs_pinned.json'] = 'changed'
        audit.qualification_bindings(reuse, dict(dependency_sha256=dependencies), issues)
        self.assertIn('Reused qualification gate differs from seal: Oa/initial_vs_pinned.json', issues)

    def test_comparisons_use_pooled_sums_and_one_request_count(self):
        rows = []
        for i, (before, after) in enumerate(zip((1., 4., 10.), (.5, 3., 12.))):
            for method in audit.ARMS:
                for repeat in range(3):
                    rows.append(dict(request_id=str(i), method=method, repeat=repeat,
                                     verification_calls=(10, 9, 11)[i] if method == 'P' else 10,
                                     request_latency_s=after if method == 'P' else before))
        result = audit.independent_comparisons(rows, ['0', '1', '2'])['P_vs_O']
        self.assertEqual(result['call_request_directions'], dict(improved=1, regressed=1, tied=1))
        self.assertEqual(result['call_gross_saved'], 1)
        self.assertEqual(result['call_gross_extra'], 1)
        self.assertEqual(result['call_reduction_pct'], 0)
        self.assertAlmostEqual(result['request_time_reduction_pct'], 100 * (1 - 15.5 / 15))
        self.assertEqual(result['latency_request_directions'], dict(improved=2, regressed=1, tied=0))
        self.assertEqual(result['latency_gross_saved'], 1.5)
        self.assertEqual(result['latency_gross_extra'], 2.)

    def test_nonfinite_numbers_never_compare_equal(self):
        self.assertFalse(audit.close(float('nan'), float('nan')))
        self.assertFalse(audit.close(float('inf'), float('inf')))
        self.assertTrue(audit.close(None, None))
        self.assertFalse(audit.close(None, 0))

    def test_costs_keep_nested_reference_out_of_phase_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            objective = dict(reference_setup_s=1., reference_forward_s=2., reference_head_s=3.)
            event_costs = dict(training_s=8., checkpoint_save_s=1., export_s=1.)
            costs, training = {}, {}
            for arm in ('U', 'P'):
                costs[arm] = dict(**event_costs, event_wall_s=12., loading_s=2., initialization_control_s=1.,
                                  objective_work=objective, peak_allocated_bytes=100, peak_reserved_bytes=150,
                                  reference_cost_is_nested_in_event_wall=True,
                                  teacher_preparation_s=0., new_feedback_generations=0)
                training[arm] = dict(costs=event_costs, objective_work=objective)
                write(root / (arm + '_COSTS.json'), costs[arm])
                initialization = dict(wall_s=1.)
                write(root / (arm + '_INITIAL.json'), dict(controlled_initialization=initialization))
                write(root / arm / 'initialization/RESULT.json', initialization)
                write(root / arm / 'events/event_001/RESULT.json', dict(records=[dict(training_s=3.), dict(training_s=5.)]))
            warmup = dict(wall_s=5.)
            costs.update(process_warmup=warmup, backend_loading_s=3., actual_campaign_wall_s=48.)
            write(root / 'process_warmup/RESULT.json', warmup)
            write(root / 'evaluation/STATUS.json', dict(evaluation_wall_s=10.))
            write(root / 'COSTS.json', costs)
            issues = []
            result = audit.cost_accounting(root, training, issues)
            self.assertEqual(issues, [])
            self.assertEqual(result['nonoverlapping_measured_phase_sum_s'], 48.)
            self.assertEqual(result['arms']['P']['reference_nested_s'], 6.)
            costs['actual_campaign_wall_s'] = 47.
            write(root / 'COSTS.json', costs)
            audit.cost_accounting(root, training, issues)
            self.assertIn('Distinct phase costs exceed campaign wall', issues)


if __name__ == '__main__':
    unittest.main()
