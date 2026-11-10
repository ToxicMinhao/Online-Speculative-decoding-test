"""Fail before model work for unbound source, incomplete priority or wrong pins."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.survival_head import preflight as P
from online_feedback.storage import write_json
from test_seal import sealed_fixture


class PreflightTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)

    def test_only_complete_successful_priority_releases_the_host(self):
        prior=self.root/'priority'; prior.mkdir()
        plan={'priority_root':str(prior)}
        with self.assertRaisesRegex(ValueError,'still owns'): P.priority_finished(plan)
        (prior/'EXIT_CODE').write_text('1\n')
        write_json(prior/'campaign_001/STATUS.json',{'status':'completed'})
        with self.assertRaisesRegex(ValueError,'did not finish'): P.priority_finished(plan)
        (prior/'EXIT_CODE').write_text('0\n')
        write_json(prior/'campaign_001/STATUS.json',{'status':'running'})
        with self.assertRaisesRegex(ValueError,'did not finish'): P.priority_finished(plan)
        write_json(prior/'campaign_001/STATUS.json',{'status':'completed'})
        self.assertEqual(P.priority_finished(plan)['status'],'passed')

    def test_seal_must_include_plan_every_input_and_executing_source(self):
        for mutate in ('plan','source','different_execution','changed_module'):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory(dir=self.root) as folder:
                root,code,_=sealed_fixture(Path(folder))
                path=root/'sealed/SEAL.json'; seal=P.read(path)
                if mutate=='plan': seal['files'].pop(str(root/'sealed/PLAN.json'))
                if mutate=='source': seal['files'].pop(str(code/'online_feedback/survival_head.py'))
                if mutate=='different_execution': seal['execution_source']['commit']='c'*40
                if mutate=='changed_module': (code/'online_feedback/objective.py').write_text('# changed')
                write_json(path,seal)
                with patch.object(P,'__file__',str(code/'experiments/survival_head/preflight.py')):
                    with self.assertRaises(ValueError): P.verify_seal(root)

    def test_priority_failure_saves_receipt_before_tensor_load_or_runtime_probe(self):
        root,code,_=sealed_fixture(self.root)
        with patch.object(P,'__file__',str(code/'experiments/survival_head/preflight.py')), \
                patch('torch.load') as load, \
                patch('experiments.osd_components.prepare.verify_native_runtime') as runtime:
            with self.assertRaisesRegex(ValueError,'still owns'): P.verify(root)
        load.assert_not_called(); runtime.assert_not_called()
        receipt=P.read(root/'PREFLIGHT_QUALIFICATION.json')
        self.assertEqual(receipt['status'],'failed')
        self.assertEqual(receipt['model_work'],0)

    def test_runtime_replay_pin_is_additive_and_original_unchanged(self):
        original={'native_runtime_sha256':{'old.py':'a'*64},'python_version':'fixture'}
        before=deepcopy(original)
        actual=P.replay_runtime_pin(original)
        self.assertEqual(original,before)
        self.assertEqual(actual['native_runtime_sha256']['old.py'],'a'*64)
        name='managers/schedule_batch.py'
        self.assertEqual(actual['native_runtime_sha256'][name],
                         'e72161a7245e882eef40612a728fd6f72963498666a8caa4fa5b8b8d45413199')
        broken=deepcopy(original); broken['native_runtime_sha256'][name]='b'*64
        with self.assertRaisesRegex(ValueError,'disagree'): P.replay_runtime_pin(broken)

    def test_unknown_scope_fails_before_any_receipt_or_process(self):
        with patch.object(P,'verify_seal') as verify:
            with self.assertRaisesRegex(ValueError,'Unknown'): P.verify(self.root,scope='extra_run')
        verify.assert_not_called()
        self.assertEqual(list(self.root.iterdir()),[])


if __name__=='__main__': unittest.main()
