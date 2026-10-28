"""Small CPU tests for one-step orchestration, not GPU qualification."""
from collections import Counter
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

from experiments.prefix_correction import run as R
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng


REFERENCE = dict(responses=[
    dict(blocks=363, old_matching_prefix_tokens=2710, new_matching_prefix_tokens=3412,
         net_matching_prefix_change=702, old_first_errors=287,
         first_errors_repaired_with_prefix_intact=232, new_earlier_errors=36, old_prefix_tokens_lost=137),
    dict(blocks=411, old_matching_prefix_tokens=2526, new_matching_prefix_tokens=3564,
         net_matching_prefix_change=1038, old_first_errors=345,
         first_errors_repaired_with_prefix_intact=295, new_earlier_errors=11, old_prefix_tokens_lost=16),
])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_trainer():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(.05)
    model.begin_event = lambda: None
    optimizer = DirectBF16AdamW(model)
    return SimpleNamespace(
        bundle=SimpleNamespace(model=model, draft_model=model),
        backend=SimpleNamespace(optimizer=optimizer, backward=lambda loss, **kw: loss.backward()),
        core=SimpleNamespace(accumulation_remainder=0), global_step=0, presentations=0,
        close=lambda: None, failed=False,
    )


class RunnerTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_reproduction_gate_is_per_response_and_durable(self):
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'gate.json'
            R.check_reproduction(deepcopy(REFERENCE), REFERENCE, path)
            altered=deepcopy(REFERENCE)
            # Same pooled total is insufficient: each original response must match.
            altered['responses'][0]['new_earlier_errors'] += 1
            altered['responses'][1]['new_earlier_errors'] -= 1
            with self.assertRaisesRegex(AssertionError,'Exact qualification gate'):
                R.check_reproduction(altered,REFERENCE,path)
            evidence=json.loads(path.read_text())
            self.assertFalse(evidence['exact_equal'])
            self.assertTrue(evidence['metadata_differences'])

    def test_state_gate_checks_optimizer_rng_modes_and_counters(self):
        trainer=fixture_trainer()
        original=R.invariant_state(trainer)
        trainer.global_step=1
        with TemporaryDirectory() as temporary, self.assertRaises(AssertionError):
            R.persist_comparison(Path(temporary)/'state.json',R.invariant_state(trainer),original)
        self.assertIn('optimizer',original)
        self.assertIn('rng',original)
        self.assertIn('training_modes',original)
        self.assertIn('mask_cache_identity',original)

    def test_counts_actual_adam_calls_and_unmodified_backward(self):
        trainer=fixture_trainer()
        with TemporaryDirectory() as temporary:
            log=R.WorkLog(temporary)
            with R.count_k_work(trainer,log):
                for i in range(2):
                    loss=trainer.bundle.model.weight.float().square().sum()/2
                    trainer.backend.backward(loss,is_boundary=i==1)
                trainer.backend.optimizer.step()
            self.assertEqual(log.counts['k_backward_completed'],2)
            self.assertEqual(log.counts['adam_step_completed'],1)
            self.assertTrue(all(p.grad is None for p in trainer.backend.optimizer.model_params))

    def test_nonfinite_failure_values_remain_explicit(self):
        encoded=R.diagnostic_json(dict(norm=float('nan'),values=[float('inf'),float('-inf'),1.0]))
        self.assertEqual(encoded,dict(norm='nan',values=['inf','-inf',1.0]))
        json.dumps(encoded,allow_nan=False)

    def test_excess_core_work_fails_with_budget_receipt(self):
        report=dict(work=dict(original_response_forwards=2,post_k_response_forwards=2,
                             post_c_response_forwards=2,violation_backward_completed=2,direct_corrections=1))
        log=SimpleNamespace(counts=Counter(k_backward_completed=2,adam_step_completed=1))
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'budget.json'
            R.validate_core_budget(report,log,path)
            report['work']['direct_corrections']=2
            with self.assertRaisesRegex(AssertionError,'Core work'):
                R.validate_core_budget(report,log,path)
            self.assertEqual(json.loads(path.read_text())['direct_corrections'],2)

    def fixture(self, root, *, mismatch=False, mutate_optimizer=False):
        records=[dict(record_id=f'r{i}',**{'pass':i//128},sampled_anchors=[dict(positions=[[i]],keep_mask=[[True]])],
                      optimizer_stepped=bool(i%2),optimizer_step_after=(i+1)//2) for i in range(256)]
        ordinary=dict(features=[dict(path=f'feature-{i}.pt') for i in range(128)],records=records)
        plan=dict(config=str(root/'config.yaml'),ordinary_receipt=str(root/'ordinary.json'),
                  reference_probe=str(root/'reference.json'),pinned_initial=str(root/'initial.pt'))
        save(root/'sealed/PLAN.json',plan)
        save(root/'ordinary.json',ordinary)
        save(root/'reference.json',REFERENCE)
        events=[]
        trainer=fixture_trainer()
        trainer.close=lambda:events.append('trainer_closed')
        outer=self

        class Correction:
            def __init__(self, trainer, *, output_dir, record_ids):
                events.append('capture_constructed')
                outer.assertEqual(record_ids,('r0','r1'))
                self.work=dict(original_response_forwards=0,post_k_response_forwards=0,
                               post_c_response_forwards=0,violation_backward_completed=0,direct_corrections=0)
            @contextmanager
            def capture_k(self):
                yield
                self.work['original_response_forwards']=2
            def measure_post_k(self):
                events.append('measure_K')
                self.work.update(post_k_response_forwards=2,violation_backward_completed=2)
                value=deepcopy(REFERENCE)
                if mismatch:
                    value['responses'][0]['old_prefix_tokens_lost'] += 1
                if mutate_optimizer:
                    trainer.backend.optimizer.optimizer.param_groups[0]['lr']=1
                return value
            def apply_once(self):
                events.append('correct')
                self.work['direct_corrections']=1
                with torch.no_grad():
                    trainer.bundle.model.weight.add_(torch.tensor(.0001,dtype=torch.bfloat16))
                return dict(branch='applied')
            def measure_post_c(self):
                events.append('measure_C')
                self.work['post_c_response_forwards']=2
                value=deepcopy(REFERENCE)
                value['responses'][0]['old_prefix_tokens_lost']-=1
                value['responses'][0]['net_matching_prefix_change']+=1
                return value
            def report(self):
                return dict(work=dict(self.work))
            def close(self):
                events.append('correction_closed')

        def startup(config,ordinary,pin,log,path):
            events.append('startup_two_zero_Adam')
            log.counts.update(startup_response_evaluation_completed=2,startup_backward_completed=2)
            return dict(status='passed',response_evaluations=2,backwards=2,optimizer_steps=0)
        def construct(*args,**kwargs):
            events.append('fresh_K')
            return trainer
        def prepare(*args):
            events.append('initial_exact')
            return dict(exact_pinned_initial_state=True)
        def window(trainer,ordinary,log):
            for i in range(2):
                trainer.backend.backward(trainer.bundle.model.weight.float().square().sum()/2,is_boundary=i==1)
                if i==1:
                    trainer.backend.optimizer.step()
                    trainer.global_step+=1
                trainer.presentations+=1
                log.counts['response_presentations']+=1
                yield deepcopy(records[i])
        def pin(path):
            events.append('pin_false')
            return dict(effective=False)
        stack=ExitStack()
        stack.enter_context(patch.dict(sys.modules,{'online_feedback.prefix_correction':SimpleNamespace(PrefixCorrection=Correction)}))
        stack.enter_context(patch.object(R,'run_startup',side_effect=startup))
        stack.enter_context(patch.object(R,'SpecForgeEventTrainer',side_effect=construct))
        stack.enter_context(patch.object(R,'prepare_main',side_effect=prepare))
        stack.enter_context(patch.object(R,'run_window',side_effect=window))
        stack.enter_context(patch.object(R,'pin_retained_graph_runtime',side_effect=pin))
        stack.enter_context(patch.object(R,'require_runtime_pin'))
        stack.enter_context(patch.object(R,'observe_forwards',return_value=nullcontext()))
        stack.enter_context(patch.object(torch,'load',return_value={}))
        return stack,events

    def test_single_step_flow_reuses_data_and_stops_after_correction(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            stack,events=self.fixture(root)
            with stack:
                result=R.run(root)
            self.assertEqual(events[:4],['pin_false','startup_two_zero_Adam','fresh_K','initial_exact'])
            self.assertEqual(events.count('correct'),1)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['budget']['adam_steps'],1)
            self.assertEqual(result['budget']['core_response_forwards'],6)
            self.assertEqual(result['budget']['core_backward_equivalents'],4)
            self.assertEqual(result['completed_presentations'],2)
            self.assertEqual(result['complete_response_generations'],0)
            self.assertFalse(result['checkpoint_exported'])
            self.assertTrue(result['comparison']['local_advancement'])

    def test_reproduction_failure_stops_before_correction_and_preserves_work(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            stack,events=self.fixture(root,mismatch=True)
            with stack,self.assertRaises(AssertionError):
                R.run(root)
            self.assertNotIn('correct',events)
            self.assertNotIn('measure_C',events)
            result=json.loads((root/'execution_001/RESULT.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['completed_optimizer_steps'],1)
            self.assertEqual(result['core']['work']['direct_corrections'],0)
            self.assertEqual(result['core']['work']['post_k_response_forwards'],2)

    def test_post_k_optimizer_mutation_stops_before_correction(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            stack,events=self.fixture(root,mutate_optimizer=True)
            with stack,self.assertRaisesRegex(AssertionError,'POST_K_STATE_UNCHANGED'):
                R.run(root)
            self.assertNotIn('correct',events)


if __name__=='__main__':
    unittest.main()
