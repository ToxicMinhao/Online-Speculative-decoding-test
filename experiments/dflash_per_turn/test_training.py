"""CPU fixtures verify scheduling/masks/state isolation; not native GPU qualification."""
import copy
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

try:
    from . import training
except ImportError:
    import training

HAS_TORCH = importlib.util.find_spec('torch') is not None
RELEASED, TARGET = 'a'*64, 'c'*64


def response(turn, checkpoint=RELEASED, output_length=18, prefix_length=20):
    return dict(episode_id='episode', record_id=f'episode:t{turn}', turn=turn,
                input_ids=[i % 19 for i in range(prefix_length)], input_token_count=prefix_length,
                output_ids=[(i+turn) % 19 for i in range(output_length)], output_cap=1024,
                capture_checkpoint_sha256=checkpoint, target_id='target', target_identity_sha256=TARGET)


class Metadata(unittest.TestCase):
    def test_schedule_is_one_response_twice_and_exact_turn(self):
        for event in (1, 8, 23):
            r = response(event)
            self.assertEqual(training.event_schedule(event, [r]), [(0, [r]), (1, [r])])
        for event, rows in ((1, []), (1, [response(2)]), (1, [response(1), response(2)]), (24, [response(24)])):
            with self.assertRaises(ValueError):
                training.event_schedule(event, rows)

    def test_1024_cap_full_prefix_and_latest_only_mask(self):
        r = response(10, prefix_length=27000, output_length=1024)
        audit = training.audit_response(r)
        self.assertEqual(audit['sampled_anchor_count'], 512)
        self.assertEqual(audit['real_sequence_length'], 28024)
        self.assertEqual(audit['loss_mask_sha256'], training.digest([0]*27000+[1]*1024+[0]*9))
        with self.assertRaises(ValueError):
            training.audit_response(response(1, output_length=1025))
        with self.assertRaises(ValueError):
            training.audit_response(response(1, prefix_length=31744, output_length=1024))

    def test_historical_module_unmodified(self):
        path = Path(__file__).parents[1] / 'dflash_two_update_followup/author_training.py'
        spec = importlib.util.spec_from_file_location('_historical_check', path)
        historical = importlib.util.module_from_spec(spec); spec.loader.exec_module(historical)
        self.assertEqual(historical.OUTPUT_CAP, 256)
        self.assertEqual(historical.RECIPE['responses_per_step'], 2)
        self.assertEqual(training.RECIPE['responses_per_step'], 1)
        self.assertEqual(training.RECIPE['optimizer_steps_per_event'], 2)


@unittest.skipUnless(HAS_TORCH, 'CPU torch required')
class Numerical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def trainer(self):
        import torch
        from torch import nn
        import torch.nn.functional as F
        torch.manual_seed(41)
        class Draft(nn.Module):
            def __init__(self):
                super().__init__(); self.fc = nn.Linear(8, 4); self.mix = nn.Linear(4, 4)
            def forward(self, *, position_ids, noise_embedding, target_hidden, attention_mask):
                kv = torch.cat((self.fc(target_hidden), noise_embedding), dim=1).unsqueeze(1)
                hidden = F.scaled_dot_product_attention(noise_embedding.unsqueeze(1), kv, kv,
                                                        attn_mask=attention_mask).squeeze(1)
                return self.mix(hidden+noise_embedding), None
        draft = Draft(); embed = nn.Embedding(20, 4).requires_grad_(False).eval()
        head = nn.Linear(4, 20, bias=False).requires_grad_(False).eval()
        def teacher(ids):
            base = embed(ids)
            causal = base.cumsum(1)/torch.arange(1, ids.shape[1]+1).view(1, -1, 1)
            return torch.cat((base, causal), dim=-1), causal
        return training.PersistentPerTurnTrainer(draft, embed, head, teacher, episode_id='episode',
            target_id='target', target_identity_sha256=TARGET, released_checkpoint_sha256=RELEASED,
            mask_token_id=19, attention_backend='sdpa', projection_chunk=7)

    def test_three_events_persist_adam_and_response_only_mask(self):
        trainer = self.trainer(); owner = id(trainer.optimizer)
        for event in range(1, 4):
            result = trainer.train_event(event, [response(event, trainer.active_checkpoint_sha256, prefix_length=event*20)])
            self.assertEqual(result['optimizer_steps_this_event'], 2)
            self.assertEqual(result['optimizer_steps_total'], event*2)
            self.assertEqual(id(trainer.optimizer), owner)
            self.assertEqual(result['response_presentations_this_event'], 2)
            for batch in result['batches']:
                self.assertEqual(len(batch['presentations']), 1)
                presentation = batch['presentations'][0]
                self.assertTrue(all(event*20 < p < event*20+18
                                    for p in presentation['unique_supervised_position_ids']))
            trainer.bind_checkpoint(event, str(event)*64)
        self.assertEqual(trainer.completed_events, 3)
        self.assertEqual({int(s['step']) for s in trainer.optimizer.state.values()}, {6})
        with self.assertRaises(ValueError):
            trainer.train_event(3, [response(3, trainer.active_checkpoint_sha256)])

    def test_lower_lr_restores_exact_anchors_and_main_optimizer_rng(self):
        import torch
        trainer = self.trainer()
        pre = trainer.snapshot()
        r = response(1)
        main_result = trainer.train_event(1, [r]); trainer.bind_checkpoint(1, 'b'*64)
        main = trainer.snapshot()
        main_hash = training.state_digest(main)
        trainer.restore(pre)
        trainer.learning_rate = 1e-4; trainer.optimizer.param_groups[0]['lr'] = 1e-4
        low = trainer.train_event(1, [r])
        self.assertEqual(low['anchor_exposure_sha256'], main_result['anchor_exposure_sha256'])
        self.assertEqual(low['optimizer_steps_this_event'], 2)
        self.assertTrue(any(not torch.equal(v, main['draft'][k]) for k, v in trainer.draft.state_dict().items()))
        random.random(); torch.rand(7)
        trainer.restore(main)
        self.assertEqual(training.state_digest(trainer.snapshot()), main_hash)
        # Next main update is identical to a completely separate restored run.
        result = trainer.train_event(2, [response(2, 'b'*64)]); trainer.bind_checkpoint(2, 'd'*64)
        expected = trainer.snapshot()
        other = self.trainer(); other.restore(main)
        control = other.train_event(2, [response(2, 'b'*64)]); other.bind_checkpoint(2, 'd'*64)
        self.assertEqual(control['anchor_exposure_sha256'], result['anchor_exposure_sha256'])
        self.assertEqual(training.state_digest(other.snapshot()), training.state_digest(expected))

    def test_disk_state_has_bound_weights_separate_and_roundtrips(self):
        import torch
        trainer = self.trainer()
        trainer.train_event(1, [response(1)]); trainer.bind_checkpoint(1, 'b'*64)
        weights = copy.deepcopy(trainer.draft.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            receipt = trainer.save_state(Path(directory)/'state.pt')
            state = torch.load(receipt['path'], weights_only=False)
            self.assertNotIn('draft', state)
            self.assertEqual(training.state_digest(state), receipt['state_identity_sha256'])
            with self.assertRaises(ValueError):
                trainer.restore(state)
            trainer.restore(state, draft_state=weights)
            self.assertEqual(training.state_digest(trainer.snapshot(include_draft=False)), receipt['state_identity_sha256'])
            with self.assertRaises(ValueError):
                trainer.save_state(Path(directory)/'state.pt')


if __name__ == '__main__':
    unittest.main()
