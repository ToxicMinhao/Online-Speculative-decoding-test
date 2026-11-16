"""CPU orchestration controls for five disposable qualification applications.

The scalar runtime is deliberately independent of DFlash. Native objective and
BF16 coefficient correctness have their own tests; these check ordering,
restoration, serialization, failure evidence and the shared work ledger.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.bounded_prefix_allocation import qualification as Q
from experiments.bounded_prefix_allocation import training as T
from experiments.bounded_prefix_allocation.runtime import initialize_work, read
from experiments.functional_step_geometry.qualify import restore_runtime
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import write_json


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        self.register_buffer('cache_marker', torch.tensor([0.]))
        self._cached_seq_len = 0
        self.costs = {}

    def begin_event(self):
        self.costs.clear()


class TinyTrainer:
    instances = []
    corrupt_cap_zero = False

    def __init__(self, config, output, **options):
        self.output_root = Path(output)
        self.events_root = self.output_root / 'events'
        self.events_root.mkdir(parents=True)
        self.bundle = SimpleNamespace(model=TinyModel())
        self.bundle.draft_model = self.bundle.model
        optimizer = torch.optim.AdamW(self.bundle.model.parameters(), lr=3e-4, foreach=False)
        self.backend = SimpleNamespace(optimizer=SimpleNamespace(
            optimizer=optimizer, zero_grad=optimizer.zero_grad,
            state_dict=optimizer.state_dict), backward=self.backward)
        self.core = SimpleNamespace(_ratio_totals={}, _sum_totals={}, _micro=0,
                                    accumulation_remainder=0)
        self.config_text = options['original_config_text']
        self.cfg = SimpleNamespace(run_id='test', model=SimpleNamespace(
            draft_model_config=str(config), target_model_path='unused', embedding_key='unused'))
        self.schedule = options['schedule']
        self.response_records, self.window_records, self._anchor_observations = [], [], []
        self.work, self.failed, self.closed = {}, False, False
        self.strategy = SimpleNamespace(forward_loss=self.forward)
        self.step_observer = None
        self.qualification_allocation(.25)
        self._load_scientific_state(torch.load(options['origin_state'], weights_only=False))
        self.instances.append(self)

    def qualification_allocation(self, cap):
        self.cap = cap

    def _load_scientific_state(self, state):
        self.bundle.model.load_state_dict(state['draft_state_dict'])
        self.backend.optimizer.optimizer.load_state_dict(deepcopy(state['optimizer']))
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.core._micro = self.presentations
        restore_rng(state['rng'])

    def state(self):
        return deepcopy(dict(draft_state_dict=self.bundle.model.state_dict(),
            optimizer=self.backend.optimizer.optimizer.state_dict(), rng=capture_rng(),
            global_step=self.global_step, presentations=self.presentations))

    def forward(self, feature):
        if feature is None:
            return self.bundle.model.weight.sum()
        draw = dict(position=int(torch.randint(100, (1,)).item()))
        self._presentation_schedule.check_anchors(draw)
        observer = getattr(self.bundle.model, 'block_observer', None)
        if observer is not None:
            values = torch.tensor([[1, 2]])
            observer(values, values, values.float(), values.bool())
        gradient = self.bundle.model.weight.new_tensor([1., -1. if self.cap == .25 else 1.])
        if self.corrupt_cap_zero and self.cap == 0:
            gradient[0] = -1
        loss = (self.bundle.model.weight * gradient).sum()
        self.bundle.model.cache_marker.add_(1)
        self.bundle.model._cached_seq_len += 1
        self.work['response_forwards'] = self.work.get('response_forwards', 0) + 1
        metrics = dict(loss=float(loss.detach()))
        allocation = dict(changed_positions=2) if self.cap == .25 else dict(changed_positions=0)
        observation = dict(loss=metrics['loss'], sum_metrics=metrics, ratio_metrics={},
                           prefix_allocation=allocation)
        row = dict(record_id=feature['record_id'], sampled_anchors=[draw],
                   response_observation=observation, metrics=metrics)
        self.response_records.append(row)
        return loss, row

    def backward(self, loss):
        loss.backward()
        observer = getattr(self.bundle.model, 'block_observer', None)
        if observer is not None:
            # Simulate checkpoint recomputation: these deliberately distinct
            # values must never enter original-forward comparison receipts.
            values = torch.tensor([[9, 9]])
            observer(values, values, values.float(), values.bool())

    def run_pair(self, pair, pass_index, ordinal):
        rows = []
        for feature in pair:
            self._presentation_schedule.begin_presentation(pass_index, feature['record_id'])
            loss, row = self.strategy.forward_loss(feature)
            self.backend.backward(loss / 2)
            self._presentation_schedule.finish_presentation()
            rows.append(row)
            self.presentations += 1
            self.core._micro += 1
        self.step_observer(self)
        torch.nn.utils.clip_grad_norm_(self.bundle.model.parameters(), 1.)
        self.backend.optimizer.optimizer.step()
        self.backend.optimizer.zero_grad()
        self.global_step += 1
        return rows

    def save_qualification(self, path):
        torch.save(dict(state=self.state(), schedule=deepcopy(vars(self._presentation_schedule))), path)
        return dict(global_step=self.global_step, presentations=self.presentations)

    def restore_qualification(self, path, *, runtime):
        payload = torch.load(path, weights_only=False)
        self._load_scientific_state(payload['state'])
        vars(self._presentation_schedule).update(deepcopy(payload['schedule']))
        restore_runtime(self.bundle.model, runtime)
        return dict(status='passed', exact_scientific_state=True)

    def close(self):
        self.closed = True


class QualificationOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.rng = capture_rng()
        self.addCleanup(lambda: restore_rng(self.rng))
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        initialize_work(self.root)
        TinyTrainer.instances.clear()
        TinyTrainer.corrupt_cap_zero = False
        self.features = [dict(record_id=f'r{i}', path=f'feature{i}') for i in range(4)]
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
        self.origin = dict(draft_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                           rng=capture_rng(), global_step=768, presentations=1536)
        torch.save(self.origin, self.root/'origin.pt')
        rows = []
        with preserve_rng():
            for pass_index in range(2):
                for index in torch.randperm(4).tolist():
                    rows.append(dict(record_id=f'r{index}', **{'pass':pass_index},
                        sampled_anchors=[dict(position=int(torch.randint(100,(1,)).item()))]))
        write_json(self.root/'schedule.json', dict(status='completed', passes=2,
            response_presentations=8, records=rows, features=self.features))
        (self.root/'config.yaml').write_text('fixture: CPU orchestration only\n')
        self.plan = dict(root=str(self.root), execution_commit='test', configs={'C':str(self.root/'config.yaml')},
            binding='fixture', origin_state=str(self.root/'origin.pt'), origin_export='unused',
            origin_config=str(self.root/'config.yaml'), event7_schedule=str(self.root/'schedule.json'),
            event7_feature_paths=['unused'] * 4)

    def execute(self, startup_override=None):
        def startup(trainer, features, output):
            for _ in range(2):
                loss = trainer.strategy.forward_loss(None)
                trainer.backend.backward(loss / 2)
            trainer.backend.optimizer.zero_grad()
            return dict(wall_s=.1, optimizer_steps=0)

        with ExitStack() as stack:
            stack.enter_context(patch.object(T, 'CandidateTrainer', TinyTrainer))
            stack.enter_context(patch.object(T, 'fork_state', lambda source, **kw: source))
            stack.enter_context(patch('online_feedback.training.SpecForgeEventTrainer._prepare_event',
                                      return_value=self.features))
            stack.enter_context(patch('experiments.recent_repair_retention.training.schedule_adapter',
                                      side_effect=lambda schedule, features:schedule))
            stack.enter_context(patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime'))
            stack.enter_context(patch('experiments.functional_step_geometry.native_pilot.discarded_startup',
                                      side_effect=startup_override or startup))
            stack.enter_context(patch('specforge.export.to_hf.export_to_hf'))
            return Q.qualify(self.plan)

    def test_five_updates_exact_native_zero_reload_and_final_rng_restoration(self):
        result = self.execute()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['optimizer_applications'], 5)
        self.assertEqual([r['case'] for r in result['cases']],
                         ['native','cap_zero','active','continued_next_pair','reloaded_next_pair'])
        self.assertTrue(result['origin_restored'])
        self.assertEqual(result['cases'][1]['native_equivalence'], 'exact')
        self.assertGreater(result['cases'][2]['changed_parameter_tensors'], 0)
        for row in result['cases']:
            self.assertGreaterEqual(row['preclip_observer_s'], 0)
            self.assertGreaterEqual(row['pair_wall_excluding_preclip_observer_s'], 0)
            self.assertGreaterEqual(row['pair_wall_excluding_observers_s'], 0)
            self.assertEqual(len(row['block_inputs']), 2)
        self.assertTrue(result['cases'][2]['unchanged_native_inputs_and_first_error_terms'])
        trainer = TinyTrainer.instances[-1]
        self.assertIsNone(trainer._presentation_schedule)
        self.assertTrue(torch.equal(capture_rng()['torch'], self.origin['rng']['torch']))
        self.assertEqual((trainer.global_step,trainer.presentations), (768,1536))
        self.assertEqual(trainer.bundle.model._cached_seq_len, 0)
        self.assertEqual(float(trainer.bundle.model.cache_marker), 0.)
        self.assertFalse(hasattr(trainer.bundle.model, 'block_observer'))
        work = read(self.root/'WORK.json')
        self.assertEqual(work['completed']['adam'], 5)
        self.assertEqual(work['completed']['response_forward'], 12)
        self.assertEqual(work['completed']['response_backward'], 12)
        self.assertEqual(work['pending'], {})
        self.assertTrue(trainer.closed)
        self.assertFalse(torch.cuda.is_initialized())

    def test_failed_cap_zero_comparison_preserves_completed_work_and_stops(self):
        TinyTrainer.corrupt_cap_zero = True
        with self.assertRaisesRegex(ValueError, 'Native/cap0'):
            self.execute()
        status = read(self.root/'qualification/training/STATUS.json')
        self.assertEqual(status['status'], 'failed')
        self.assertEqual([row['case'] for row in status['cases']], ['native', 'cap_zero'])
        self.assertIn('preclip', status['cases'][1])
        work = read(self.root/'WORK.json')
        self.assertEqual(work['completed']['adam'], 2)
        self.assertEqual(work['attempted']['adam'], 2)
        self.assertEqual(work['completed']['response_forward'], 6)
        self.assertTrue(TinyTrainer.instances[-1].failed)
        self.assertTrue(TinyTrainer.instances[-1].closed)
        self.assertFalse(torch.cuda.is_initialized())

    def test_primary_failure_is_durable_before_cleanup_and_not_replaced(self):
        TinyTrainer.corrupt_cap_zero = True
        observed = []

        def failed_close(trainer):
            observed.append(read(self.root/'qualification/training/STATUS.json'))
            raise RuntimeError('injected cleanup failure')

        with patch.object(TinyTrainer, 'close', failed_close):
            with self.assertRaisesRegex(ValueError, 'Native/cap0'):
                self.execute()
        self.assertEqual(observed[0]['status'], 'failed')
        self.assertEqual(len(observed[0]['cases']), 2)
        status = read(self.root/'qualification/training/STATUS.json')
        self.assertIn('Native/cap0', status['error'])
        self.assertIn('injected cleanup failure', status['cleanup_error'])
        self.assertEqual(read(self.root/'WORK.json')['completed']['adam'], 2)

    def test_partial_startup_counts_only_attempted_forward_and_no_adam(self):
        def fail_after_forward(trainer, features, output):
            trainer.strategy.forward_loss(None)
            raise RuntimeError('injected startup failure before backward')

        with self.assertRaisesRegex(RuntimeError, 'startup failure'):
            self.execute(startup_override=fail_after_forward)
        work = read(self.root/'WORK.json')
        self.assertEqual(work['completed']['response_forward'], 1)
        self.assertEqual(work['attempted']['response_backward'], 0)
        self.assertEqual(work['attempted']['adam'], 0)
        status = read(self.root/'qualification/training/STATUS.json')
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['cases'], [])


if __name__ == '__main__':
    unittest.main()
