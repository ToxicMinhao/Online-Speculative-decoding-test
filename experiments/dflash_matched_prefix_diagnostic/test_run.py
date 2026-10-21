"""No model execution from planning/reporting or missing explicit run approval."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

from experiments.dflash_matched_prefix_diagnostic.run import main
from experiments.dflash_matched_prefix_diagnostic.protocol import digest


class CLIGuards(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_run_requires_explicit_approval_before_loading_evidence(self):
        with patch('experiments.dflash_matched_prefix_diagnostic.run.load_evidence') as load:
            with self.assertRaisesRegex(ValueError, 'explicit execution'):
                main(['run', '--baseline-root', str(self.root/'b'), '--diagnostic-root', str(self.root/'d'),
                      '--lower-lr-root', str(self.root/'l'), '--out', str(self.root/'run')])
            load.assert_not_called()

    def test_plan_cannot_take_execution_flag(self):
        with patch('experiments.dflash_matched_prefix_diagnostic.run.load_evidence') as load:
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                main(['plan', '--baseline-root', str(self.root/'b'), '--diagnostic-root', str(self.root/'d'),
                      '--lower-lr-root', str(self.root/'l'), '--out', str(self.root/'run'), '--allow-gpu-execution'])
            load.assert_not_called()

    def test_all_original_evidence_roots_required(self):
        with self.assertRaisesRegex(ValueError, 'three original'):
            main(['plan', '--baseline-root', str(self.root/'b'), '--out', str(self.root/'out')])

    def test_never_overwrite_existing_evidence(self):
        with self.assertRaisesRegex(ValueError, 'Never overwrite'):
            main(['plan', '--out', str(self.root)])

    def test_output_cannot_overlap_original_evidence(self):
        with self.assertRaisesRegex(ValueError, 'separate'):
            main(['plan', '--baseline-root', str(self.root/'b'), '--diagnostic-root', str(self.root/'d'),
                  '--lower-lr-root', str(self.root/'l'), '--out', str(self.root/'b'/'new')])

    def test_report_cannot_request_models(self):
        with self.assertRaisesRegex(ValueError, 'Reporting reads only'):
            main(['report', '--result', str(self.root/'RESULT.json'), '--out', str(self.root/'out'),
                  '--allow-gpu-execution'])

    def test_report_never_loads_evidence_or_native_backend(self):
        result = self.root/'prior'/'RESULT.json'
        result.parent.mkdir()
        result.write_text(json.dumps({'status': 'incomplete'}))
        with patch('experiments.dflash_matched_prefix_diagnostic.run.load_evidence') as load, \
             patch('experiments.dflash_matched_prefix_diagnostic.report.write_reports') as report:
            self.assertEqual(main(['report', '--result', str(result), '--out', str(self.root/'report')]), 0)
            load.assert_not_called()
            report.assert_called_once()

    def source_context(self, *, dirty=False):
        stack = ExitStack()
        pins = {'source.py': 'a'*64}
        cfg = {key: pins for key in ('learning_curve_source_sha256',
                                    'diagnostic_source_sha256', 'lower_lr_source_sha256')}
        bundle = dict(runtime_config=cfg, native_runtime={}, bindings={})
        plan = dict(plan_sha256='b'*64, groups=[], selected_location_count=0)
        prefix = 'experiments.dflash_matched_prefix_diagnostic.'
        stack.enter_context(patch(prefix+'run.load_evidence', return_value=bundle))
        stack.enter_context(patch(prefix+'run.build_plan', return_value=plan))
        stack.enter_context(patch(prefix+'run.validate_plan', return_value={'status': 'passed'}))
        stack.enter_context(patch(prefix+'run._git_identity', return_value={'commit': 'c'*40, 'dirty': dirty}))
        for package in ('dflash_matched_prefix_diagnostic', 'dflash_baseline_learning_curve',
                        'dflash_kl_native_diagnostic', 'dflash_lower_lr_control'):
            stack.enter_context(patch('experiments.'+package+'.native.source_pins', return_value=pins))
        backend = stack.enter_context(patch(prefix+'native.NativeBackend', side_effect=AssertionError('model started')))
        return stack, pins, plan, backend

    def args(self, action):
        return [action, '--baseline-root', str(self.root/'b'), '--diagnostic-root', str(self.root/'d'),
                '--lower-lr-root', str(self.root/'l'), '--out', str(self.root/'new')]

    def sealed(self, plan):
        path = self.root/'plan'/'PLAN.json'
        path.parent.mkdir()
        path.write_text(json.dumps(plan))
        return path

    def test_plan_writes_private_evidence_without_backend(self):
        stack, pins, plan, backend = self.source_context()
        with stack:
            self.assertEqual(main(self.args('plan')), 0)
            backend.assert_not_called()
            self.assertEqual(json.loads((self.root/'new'/'PLAN.json').read_text()), plan)
            receipt = json.loads((self.root/'new'/'SOURCE_MANIFEST.json').read_text())
            self.assertEqual(receipt['approved_code_sha256'], digest(pins))

    def test_changed_sealed_selection_stops_before_backend_or_writes(self):
        stack, pins, plan, backend = self.source_context()
        with stack:
            sealed = self.sealed(dict(plan, selected_location_count=1))
            with self.assertRaisesRegex(ValueError, 'never reselect'):
                main(self.args('run')+['--sealed-plan', str(sealed), '--approved-code-sha256', digest(pins),
                                       '--allow-gpu-execution'])
            backend.assert_not_called()
            self.assertFalse((self.root/'new').exists())

    def test_wrong_reviewed_hash_stops_before_model(self):
        stack, pins, plan, backend = self.source_context()
        with stack:
            with self.assertRaisesRegex(ValueError, 'clean reviewed source'):
                main(self.args('run')+['--sealed-plan', str(self.sealed(plan)), '--approved-code-sha256', 'd'*64,
                                       '--allow-gpu-execution'])
            backend.assert_not_called()

    def test_dirty_tree_stops_before_model(self):
        stack, pins, plan, backend = self.source_context(dirty=True)
        with stack:
            with self.assertRaisesRegex(ValueError, 'clean reviewed source'):
                main(self.args('run')+['--sealed-plan', str(self.sealed(plan)), '--approved-code-sha256', digest(pins),
                                       '--allow-gpu-execution'])
            backend.assert_not_called()


if __name__ == '__main__':
    unittest.main()
