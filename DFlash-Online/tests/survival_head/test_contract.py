from copy import deepcopy
import unittest

from experiments.survival_head import contract as C


class ContractTests(unittest.TestCase):
    def test_exact_three_arm_event_and_work(self):
        self.assertEqual(C.ARMS, ('KPLUS', 'LOCAL', 'SURVIVAL'))
        self.assertEqual(C.EVENTS, (5,))
        work = [C.expected_work(a) for a in C.ARMS]
        self.assertEqual(sum(x['optimizer_steps'] for x in work), 384)
        self.assertEqual(sum(x['response_forwards'] for x in work), 768)
        self.assertEqual(sum(x['head_optimizer_steps'] for x in work), 256)
        self.assertEqual(C.MAIN_BUDGET['endpoint_generations'], 768)

    def test_head_and_qualification_costs_are_separate(self):
        self.assertEqual(C.QUALIFICATION_BUDGET['optimizer_steps'], 4)
        self.assertEqual(C.QUALIFICATION_BUDGET['response_forwards'], 16)
        self.assertEqual(C.QUALIFICATION_BUDGET['response_backwards'], 20)
        self.assertEqual(C.QUALIFICATION_BUDGET['serving_generations'], 6)
        self.assertEqual(C.STARTUP_BUDGET['optimizer_steps'], 0)
        self.assertEqual(C.SERVING['extra_warmup_generations'], 0)

    def test_origin_and_native_contract(self):
        ordinary = C.experiment_contract('a'*64, 'KPLUS', origin_binding='b'*64)
        active = C.experiment_contract('a'*64, 'SURVIVAL', origin_binding='b'*64)
        self.assertIsNone(ordinary['head'])
        self.assertEqual(active['head']['coefficient'], 1.)
        self.assertEqual((active['start_adam_step'], active['final_adam_step']), (512, 640))
        self.assertFalse(active['seed_reset'])
        self.assertEqual(active['inherited_horizon'], 7424)
        with self.assertRaises(ValueError):
            C.experiment_contract('a'*64, 'LOCAL', 'mathqa', origin_binding='b'*64)
        with self.assertRaises(ValueError): C.expected_work('LOCAL', 6)

    def test_budget_copies_do_not_mutate_sealed_constants(self):
        copied = C.budgets()
        copied['main']['optimizer_steps'] = 1
        self.assertEqual(C.budgets()['main']['optimizer_steps'], 384)


if __name__ == '__main__': unittest.main()
