"""New-arm isolation, bounded gates, tail work and conservative capacity."""
from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.adaflash_arepo import campaign as C, contract as P, storage as S
from online_feedback.storage import write_json


class SupplementCampaignTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_all_four_method_orders_are_balanced_per_repeat(self):
        orders = Counter(P.method_order(b,r) for b in range(16) for r in range(3))
        self.assertEqual(len(orders), 24)
        self.assertEqual(set(orders.values()), {2})
        for repeat in range(3):
            for position in range(4):
                self.assertEqual(Counter(P.method_order(b,repeat)[position] for b in range(16)),
                                 dict.fromkeys(P.METHODS,4))

    def test_new_training_and_tail_budget_keeps_controls_frozen(self):
        self.assertEqual(P.ARMS, ('AREPO',))
        self.assertEqual(C.expected_work(58,49), C.ARM_LIMITS)
        self.assertEqual(P.NEW_BUDGET['optimizer_steps'], 7424)
        self.assertEqual(P.NEW_BUDGET['final_generations'], 12288)
        self.assertEqual(P.NEW_BUDGET['total_generations'], 7473+12288+4)
        self.assertEqual(P.NEW_BUDGET['disposable_optimizer_steps'], 6)
        self.assertEqual(P.NEW_BUDGET['baseline_optimizer_steps'], 0)
        before, after = C.expected_work(58), C.expected_work(58,49)
        self.assertEqual({k for k in before if before[k] != after[k]}, {'online_generations'})

    def test_capacity_covers_every_completion_and_own_states_only(self):
        budget = S.capacity_plan(2097259104)
        self.assertEqual(budget['raw_budget_generations'], 19765)
        self.assertGreater(budget['required_initial_free_bytes'], 159_000_000_000)
        self.assertEqual(budget['reserve_bytes'], 8*2**30)
        self.assertFalse(budget['offload_assumed'])
        with patch.object(S.native, 'admit', return_value='checked') as admission:
            # partial retains its callable, so inspect the bound arm directly.
            self.assertEqual(S.admit.keywords, {'arm':'AREPO'})
            self.assertEqual(S.retire_event.keywords, {'arm':'AREPO'})

    def test_qualification_receipt_binds_pass_to_execution_source(self):
        from experiments.ok_coding.storage import identity
        plan = {'execution_source':{'commit':'a'*40},'loss':{'fixed':True}}
        files=[]
        for number in range(8):
            path=self.root/f'input_{number}'
            path.write_bytes(str(number).encode())
            files.append(str(path))
        prepared=self.root/'prepared.json'
        write_json(prepared,dict(feature_paths=files[2:]))
        plan['qualification']=dict(config=files[0],initial_state=files[1],prepared=str(prepared))
        destination = self.root/'qualification/QUALIFICATION.json'
        good = dict(status='passed',execution_commit='a'*40,loss_contract=plan['loss'],
                    budget={'maximum_optimizer_steps':6},work={'adam_step_attempted':6,'adam_step_completed':6},
                    input_identities={p:identity(p) for p in [*files,str(prepared)]})
        write_json(destination, good)
        self.assertEqual(C.qualification_gate(self.root,plan)['status'], 'passed')
        for value in (dict(good,status='failed'),dict(good,execution_commit='b'*40),
                      dict(good,loss_contract={'changed':True}),
                      dict(good,work={'adam_step_attempted':7,'adam_step_completed':6}),
                      dict(good,work={'adam_step_attempted':6,'adam_step_completed':5})):
            write_json(destination,value)
            with self.assertRaisesRegex(ValueError,'qualification'):
                C.qualification_gate(self.root,plan)
        write_json(destination,good)
        Path(files[-1]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'input changed'):
            C.qualification_gate(self.root,plan)

    def test_failed_qualification_cannot_construct_native_model(self):
        plan = dict(binding='fixture', manifest='/manifest', storage={},execution_source={'commit':'a'*40})
        values = {str(self.root/'sealed/PLAN.json'):plan,'/manifest':{'train_requests':[]},
            str(self.root/'qualification/QUALIFICATION.json'):{'status':'failed','execution_commit':'a'*40}}
        with patch.object(C,'read',side_effect=lambda p:values[str(p)]), patch.object(C,'validate_plan'), \
                patch.object(C,'admit'), patch('online_feedback.sglang_backend.SGLangDFlashBackend') as backend, \
                self.assertRaisesRegex(ValueError,'qualification'):
            C.run_arm(self.root)
        backend.assert_not_called()
        status = json.loads((self.root/'campaign_001/arms/AREPO/STATUS.json').read_text())
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['actual_work']['attempts'],dict.fromkeys(C.ARM_LIMITS,0))

    def test_launcher_admits_qualification_then_full_campaign(self):
        text = (Path(C.__file__).with_name('run_campaign.sh')).read_text()
        qualification = text.index('preflight "$run_root" --qualification-only')
        numerical = text.index('qualify "$run_root"')
        main_admission = text.index('preflight "$run_root"\n',numerical)
        launch = text.index('campaign "$run_root"')
        self.assertLess(qualification,numerical)
        self.assertLess(numerical,main_admission)
        self.assertLess(main_admission,launch)
        self.assertIn('set -eu',text)
        self.assertIn('48h',text)
        self.assertIn('30m',text)
        failure=text[text.index('if [ "$qualification_exit"'):text.index('fi\n',text.index('if [ "$qualification_exit"'))]
        self.assertIn('experiments.adaflash_arepo.report',failure)
        self.assertIn('exit "$qualification_exit"',failure)


if __name__ == '__main__':
    unittest.main()
