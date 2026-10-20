"""Small causal CPU models test objective/state fidelity; not GPU qualification."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import training
except ImportError:
    import training

HAS_TORCH = importlib.util.find_spec('torch') is not None
RELEASED, TARGET = 'a' * 64, 'c' * 64


def response(turn, checkpoint=RELEASED, length=18, prefix=20):
    return dict(episode_id='episode', record_id=f'episode:t{turn}', turn=turn,
                input_ids=[i % 19 for i in range(prefix)], input_token_count=prefix,
                output_ids=[(i + turn) % 19 for i in range(length)], output_cap=1024,
                capture_checkpoint_sha256=checkpoint, target_id='target', target_identity_sha256=TARGET)


def components(dtype=None, dropout=0.0):
    import torch
    from torch import nn
    import torch.nn.functional as F
    torch.manual_seed(41)
    dtype = dtype or torch.float32

    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)
            self.mix = nn.Linear(4, 4)
            self.drop = nn.Dropout(dropout)
            self.forward_calls = 0

        def forward(self, *, position_ids, noise_embedding, target_hidden, attention_mask):
            self.forward_calls += 1
            kv = torch.cat((self.fc(target_hidden), noise_embedding), dim=1).unsqueeze(1)
            hidden = F.scaled_dot_product_attention(noise_embedding.unsqueeze(1), kv, kv,
                                                    attn_mask=attention_mask).squeeze(1)
            return self.mix(self.drop(hidden + noise_embedding)), None

    draft = Draft().to(dtype)
    embed = nn.Embedding(20, 4).to(dtype).requires_grad_(False).eval()
    head = nn.Linear(4, 20, bias=False).to(dtype).requires_grad_(False).eval()

    def teacher(ids):
        base = embed(ids)
        causal = base.cumsum(1) / torch.arange(1, ids.shape[1] + 1, device=ids.device).view(1, -1, 1)
        causal = causal.to(dtype)
        return torch.cat((base, causal), dim=-1), causal

    return draft, embed, head, teacher


def trainer(arm='G', *, dtype=None, qualification=False, filter_enabled=True, ordinary=False):
    args = components(dtype)
    kwargs = dict(episode_id='episode', target_id='target', target_identity_sha256=TARGET,
                  released_checkpoint_sha256=RELEASED, mask_token_id=19,
                  attention_backend='sdpa', projection_chunk=7)
    if ordinary:
        return training.qualified.PersistentPerTurnTrainer(*args, **kwargs)
    return training.AgreementTrainer(*args, arm=arm, qualification=qualification,
                                     filter_enabled=filter_enabled, **kwargs)


def tensors(model_parts, r):
    import torch
    draft, embed, head, teacher = model_parts
    ids = torch.tensor([r['input_ids'] + r['output_ids']])
    with torch.no_grad():
        hidden, final = teacher(ids)
    pad = 9
    ids = torch.cat((ids, ids.new_full((1, pad), 19)), dim=1)
    hidden = torch.cat((hidden, hidden.new_zeros((1, pad, hidden.shape[-1]))), dim=1)
    final = torch.cat((final, final.new_zeros((1, pad, final.shape[-1]))), dim=1)
    mask = torch.zeros_like(ids, dtype=torch.float32)
    mask[:, len(r['input_ids']):len(r['input_ids']) + len(r['output_ids'])] = 1
    return draft, embed, head, ids, hidden, final, mask


class Metadata(unittest.TestCase):
    def test_midpoint_uses_potential_supervised_positions_not_anchor_examples(self):
        groups = training.response_groups(training.audit_response(response(1, length=18)))
        self.assertEqual(groups['a'], list(range(21, 29)))
        self.assertEqual(groups['b'], list(range(29, 38)))
        self.assertNotIn(20, groups['a'] + groups['b'])
        with self.assertRaisesRegex(ValueError, 'empty'):
            training.response_groups(training.audit_response(response(1, length=2)))

    def test_private_recipe_keeps_historical_sources_unmodified(self):
        path = Path(__file__).parents[1] / 'dflash_two_update_followup/author_training.py'
        spec = importlib.util.spec_from_file_location('_agreement_test_historical', path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        self.assertEqual(original.OUTPUT_CAP, 256)
        self.assertEqual(original.RECIPE['responses_per_step'], 2)
        self.assertEqual(training.author.OUTPUT_CAP, 1024)
        self.assertEqual(training.RECIPE['responses_per_step'], 1)
        self.assertEqual(training.RECIPE['optimizer_steps_per_event'], 2)


@unittest.skipUnless(HAS_TORCH, 'CPU Torch required')
class Numerical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_every_direct_trainer_pins_compiler_and_rejects_drift(self):
        import torch
        import torch._functorch.config as config
        for arm in ('H', 'L', 'G', 'N'):
            with self.subTest(arm=arm), config.patch(donated_buffer=True):
                model = trainer(arm)
                self.assertIs(config.donated_buffer, False)
                self.assertIs(model.compiler_runtime['effective_donated_buffer'], False)
                self.assertEqual(model.compiler_runtime['torch_version'], str(torch.__version__))
                self.assertEqual(model.snapshot()['agreement']['compiler_runtime'], model.compiler_runtime)
                config.donated_buffer = True
                with self.assertRaisesRegex(ValueError, 'donated_buffer=False'):
                    model.train_event(1, [response(1)])
                self.assertEqual(model.optimizer_steps, 0)
                self.assertEqual(model.draft.forward_calls, 0)

    def test_compiled_single_backward_warmup_then_shared_forward_multiple_backwards(self):
        """Exercise actual AOTAutograd cache reuse that eager toy models miss."""
        import torch
        import torch._functorch.config as config
        def forward(x, weight):
            y = torch.nn.functional.linear(x, weight)
            return y.sin().cos() * y.sigmoid()
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
        def trial():
            weight = torch.nn.Parameter(torch.eye(4))
            compiled = torch.compile(forward, backend='aot_eager', fullgraph=True)
            # The native disabled-filter reference trial warms an ordinary
            # backward before the grouped trial reuses the compiled function.
            torch.autograd.grad(compiled(x, weight).sum(), weight)
            y = compiled(x, weight)  # Exactly one forward for all three losses.
            a = torch.autograd.grad(y[:2].sum(), weight, retain_graph=True)[0]
            b = torch.autograd.grad(y[2:].sum(), weight, retain_graph=True)[0]
            full = torch.autograd.grad(y.sum(), weight)[0]
            eager = torch.autograd.grad(forward(x, weight).sum(), weight)[0]
            torch.testing.assert_close(a + b, full, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(full, eager, rtol=1e-6, atol=1e-7)
        try:
            with config.patch(donated_buffer=True):
                torch._dynamo.reset()
                with self.assertRaisesRegex(RuntimeError, 'non-empty donated buffers'):
                    trial()
                torch._dynamo.reset()
                receipt = training.configure_compiler_runtime()
                self.assertIs(receipt['effective_donated_buffer'], False)
                trial()
        finally:
            torch._dynamo.reset()

    def test_shared_forward_loss_and_gradient_sum_match_original_with_overlap(self):
        import torch
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                parts = components(dtype)
                values = tensors(parts, response(1))
                parameters = dict(parts[0].named_parameters())
                groups = training.response_groups(training.audit_response(response(1)))
                generator = torch.Generator().manual_seed(17)
                state = generator.get_state().clone()
                original, original_metadata = training.author.author_response_loss(
                    *values, mask_token_id=19, generator=generator, projection_chunk=7)
                original_grad = dict(zip(parameters, torch.autograd.grad(original, tuple(parameters.values()))))
                generator.set_state(state)
                full, a, b, metadata = training.grouped_response_loss(
                    *values, groups=groups, mask_token_id=19, generator=generator, projection_chunk=7)
                raw_a = dict(zip(parameters, [v.float() for v in torch.autograd.grad(a, tuple(parameters.values()), retain_graph=True)]))
                raw_b = dict(zip(parameters, [v.float() for v in torch.autograd.grad(b, tuple(parameters.values()), retain_graph=True)]))
                full_grad = dict(zip(parameters, torch.autograd.grad(full, tuple(parameters.values()))))
                self.assertEqual(float(full.detach()), float(original.detach()))
                self.assertEqual(metadata['sampling_and_weights_sha256'], original_metadata['sampling_and_weights_sha256'])
                self.assertEqual({k: v for k, v in metadata.items() if k != 'grouping'}, original_metadata)
                for name in parameters:
                    torch.testing.assert_close(full_grad[name], original_grad[name], rtol=0, atol=0)
                self.assertEqual(training.check_gradient_sum(raw_a, raw_b, full_grad, parameters)['status'], 'passed')
                self.assertGreater(metadata['eligible_positions'], metadata['unique_supervised_positions'])
                seen = {}
                group = metadata['grouping']
                for row, binary in zip(metadata['label_positions'], metadata['binary_loss_mask']):
                    for position, supervised in zip(row, binary):
                        if supervised:
                            assigned = 'A' if position in group['group_a_positions'] else 'B'
                            self.assertEqual(seen.setdefault(position, assigned), assigned)
                            self.assertGreater(position, 20)
                            self.assertLess(position, 38)
                self.assertEqual(group['group_a_presentations'] + group['group_b_presentations'], metadata['eligible_positions'])
                self.assertAlmostEqual(group['shared_denominator'], metadata['normalization_denominator'])

    def test_one_stochastic_forward_serves_both_groups_without_rng_consumption_from_backward(self):
        import torch
        parts = components(dropout=.35)
        values = tensors(parts, response(1))
        groups = training.response_groups(training.audit_response(response(1)))
        _, a, b, _ = training.grouped_response_loss(*values, groups=groups, mask_token_id=19,
            generator=torch.Generator().manual_seed(17), projection_chunk=7)
        before_backward = torch.get_rng_state().clone()
        params = tuple(parts[0].parameters())
        torch.autograd.grad(a, params, retain_graph=True)
        torch.autograd.grad(b, params)
        self.assertEqual(parts[0].forward_calls, 1)
        self.assertTrue(torch.equal(before_backward, torch.get_rng_state()))

    def test_sampled_empty_group_fails_once_without_resampling(self):
        import torch
        values = tensors(components(), response(1))
        groups = training.response_groups(training.audit_response(response(1)))
        with patch.object(training.author, 'sample_anchor_positions', return_value=(torch.tensor([[36]]), torch.ones(1, 1, dtype=torch.bool))) as sampler:
            with self.assertRaisesRegex(ValueError, 'empty supervision group'):
                training.grouped_response_loss(*values, groups=groups, mask_token_id=19)
            self.assertEqual(sampler.call_count, 1)

    def test_h_is_exact_qualified_updater_and_l_only_changes_lr(self):
        import torch
        for dtype in (torch.float32, torch.bfloat16):
            reference = trainer(dtype=dtype, ordinary=True)
            expected = reference.train_event(1, [response(1)])
            actual = trainer('H', dtype=dtype)
            observed = actual.train_event(1, [response(1)])
            self.assertEqual(expected['anchor_exposure_sha256'], observed['anchor_exposure_sha256'])
            for name, parameter in actual.parameters.items():
                self.assertTrue(torch.equal(parameter, reference.parameters[name]))
            self.assertEqual(training.state_digest(actual.optimizer.state_dict()), training.state_digest(reference.optimizer.state_dict()))
        low = trainer('L')
        self.assertEqual(low.learning_rate, 1e-4)
        low.train_event(1, [response(1)])
        self.assertEqual(low.optimizer_steps, 2)

    def test_grouped_two_passes_have_fixed_groups_normal_distinct_draws_and_two_steps(self):
        for arm in ('G', 'N'):
            model = trainer(arm, qualification=True)
            result = model.train_event(1, [response(1)])
            self.assertEqual(result['optimizer_steps_this_event'], 2)
            self.assertEqual(result['response_presentations_this_event'], 2)
            rows = [b['presentations'][0] for b in result['batches']]
            self.assertEqual(rows[0]['grouping']['identity_sha256'], rows[1]['grouping']['identity_sha256'])
            self.assertNotEqual(rows[0]['anchor_positions'], rows[1]['anchor_positions'])
            self.assertEqual(model.draft.forward_calls, 2)
            self.assertEqual({int(v['step']) for v in model.optimizer.state.values()}, {2})
            for b in result['batches']:
                self.assertEqual(b['agreement']['optimizer_step_calls'], 1)
                self.assertEqual(b['presentations'][0]['gradient_sum_diagnostic']['status'], 'passed')
            model.bind_checkpoint(1, 'b' * 64)
            second = model.train_event(2, [response(2, 'b' * 64, prefix=45)])
            self.assertEqual(second['optimizer_steps_total'], 4)
            self.assertEqual(second['support_turns'], [2])
            model.bind_checkpoint(2, 'd' * 64)
            with self.assertRaisesRegex(ValueError, 'exactly two'):
                model.train_event(3, [response(3, 'd' * 64)])

    def test_normal_gn_updates_execute_three_backwards_and_record_costs(self):
        import torch
        for arm in ('G', 'N'):
            model = trainer(arm, qualification=False)
            real = torch.autograd.grad
            with patch.object(torch.autograd, 'grad', wraps=real) as grad:
                result = model.train_event(1, [response(1)])
            self.assertEqual(grad.call_count, 6)
            for batch in result['batches']:
                p = batch['presentations'][0]
                self.assertEqual(p['gradient_backward_calls'], 3)
                self.assertFalse(p['sum_tolerance_is_admission_gate'])
                self.assertEqual(p['ordinary_proposal_gradient'], 'direct full-loss native autograd gradient')
                phases = p['phase_seconds']
                self.assertAlmostEqual(phases['backward_s'], sum(phases[k] for k in
                    ('backward_a_s', 'backward_b_s', 'backward_full_s')))
                self.assertGreater(p['full_gradient_storage_bytes'], 0)

    def test_disabled_filter_matches_qualified_with_pinned_native_rounding_tolerance(self):
        import torch
        for dtype in (torch.float32, torch.bfloat16):
            reference = trainer('H', dtype=dtype)
            expected = reference.train_event(1, [response(1)])
            grouped = trainer('G', dtype=dtype, qualification=True, filter_enabled=False)
            actual = grouped.train_event(1, [response(1)])
            self.assertEqual(expected['anchor_exposure_sha256'], actual['anchor_exposure_sha256'])
            for name, parameter in grouped.parameters.items():
                # Reuse the unchanged native equivalence tolerance even though
                # the direct full backward now supplies the AdamW proposal.
                ref = reference.parameters[name]
                check = training._compare_native_tensor(ref.detach(), parameter, 'parameter')
                self.assertLessEqual(check['max_error_over_tolerance'], 1.)
            self.assertEqual({int(v['step']) for v in grouped.optimizer.state.values()}, {2})
            self.assertTrue(torch.equal(grouped.generator.get_state(), reference.generator.get_state()))
        with self.assertRaisesRegex(ValueError, 'qualification-only'):
            trainer('G', filter_enabled=False)

    def test_snapshot_roundtrip_contains_arm_sampler_optimizer_rng_and_rejects_cross_arm(self):
        import torch
        source = trainer('N')
        source.train_event(1, [response(1)])
        source.bind_checkpoint(1, 'b' * 64)
        state = source.snapshot()
        expected = training.state_digest(state)
        with tempfile.TemporaryDirectory() as directory:
            receipt = source.save_state(Path(directory) / 'state.pt')
            disk = torch.load(receipt['path'], weights_only=False)
            restored = trainer('N')
            restored.restore(disk, draft_state=state['draft'])
            self.assertEqual(training.state_digest(restored.snapshot()), expected)
            with self.assertRaisesRegex(ValueError, 'arm/filter'):
                trainer('G').restore(state)
        a = source.train_event(2, [response(2, 'b' * 64, prefix=40)])
        source.bind_checkpoint(2, 'd' * 64)
        replay = trainer('N')
        replay.restore(state)
        b = replay.train_event(2, [response(2, 'b' * 64, prefix=40)])
        replay.bind_checkpoint(2, 'd' * 64)
        self.assertEqual(a['anchor_exposure_sha256'], b['anchor_exposure_sha256'])
        self.assertEqual(training.state_digest(source.snapshot()), training.state_digest(replay.snapshot()))

    def test_failed_partial_mutation_poisoned_and_no_second_attempt(self):
        model = trainer('G')
        original = model._grouped_response
        calls = []
        def fail_second(*args):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError('injected failure')
            return original(*args)
        model._grouped_response = fail_second
        with self.assertRaisesRegex(ValueError, 'injected'):
            model.train_event(1, [response(1)])
        self.assertTrue(model.failed)
        self.assertEqual(model.optimizer_steps, 1)
        with self.assertRaisesRegex(ValueError, 'failed'):
            model.train_event(1, [response(1)])
        self.assertEqual(len(calls), 2)

    def test_native_disabled_filter_qualification_restores_complete_state(self):
        import torch
        for dtype in (torch.float32, torch.bfloat16):
            model = trainer('G', dtype=dtype, qualification=True)
            before = training.state_digest(model.snapshot())
            receipt = model.qualify_disabled_filter([response(1)])
            self.assertEqual(receipt['status'], 'passed')
            self.assertEqual(receipt['temporary_optimizer_steps'], 4)
            self.assertTrue(receipt['counters']['exact_match'])
            self.assertTrue(receipt['rng']['sampler_exact'])
            self.assertTrue(receipt['rng']['stochastic_exact'])
            self.assertFalse(receipt['sum_tolerance_is_admission_gate'])
            self.assertTrue(receipt['exact_state_restoration'])
            self.assertEqual(receipt['persistent_optimizer_steps_after'], 0)
            self.assertEqual(before, training.state_digest(model.snapshot()))
            self.assertFalse(model.failed)
            # The actual qualification trajectory's update still happens once.
            model.train_event(1, [response(1)])
            self.assertEqual(model.optimizer_steps, 2)

    def test_failed_disabled_qualification_restores_then_poisoned(self):
        model = trainer('G', qualification=True)
        initial = {name: value.detach().clone() for name, value in model.parameters.items()}
        with patch.object(training, '_compare_native_tensor', side_effect=ValueError('qualification mismatch')):
            with self.assertRaisesRegex(ValueError, 'qualification mismatch'):
                model.qualify_disabled_filter([response(1)])
        self.assertTrue(model.failed)
        self.assertEqual(model.optimizer_steps, 0)
        import torch
        self.assertTrue(all(torch.equal(value, initial[name]) for name, value in model.parameters.items()))
        self.assertFalse(model.optimizer.state)
        self.assertEqual(model.last_qualification_receipt['status'], 'failed')
        self.assertTrue(model.last_qualification_receipt['exact_state_restoration'])
        self.assertEqual(model.last_qualification_receipt['temporary_optimizer_steps'], 4)

    def test_inter_trial_restoration_failure_preserves_completed_temporary_step_count(self):
        model = trainer('G', qualification=True)
        real = model.restore
        calls = []
        def restore_then_fail_once(state):
            real(state)
            calls.append(1)
            if len(calls) == 1:
                raise ValueError('injected inter-trial restoration check')
        model.restore = restore_then_fail_once
        with self.assertRaisesRegex(ValueError, 'inter-trial'):
            model.qualify_disabled_filter([response(1)])
        receipt = model.last_qualification_receipt
        self.assertEqual(receipt['temporary_optimizer_steps'], 2)
        self.assertTrue(receipt['exact_state_restoration'])
        self.assertEqual(model.optimizer_steps, 0)
        self.assertTrue(model.failed)

    def test_equivalence_rejects_a_missing_or_materially_different_update(self):
        import torch
        initial = {'p': torch.tensor([1.0, 2.0])}
        reference = {'p': torch.tensor([1.001, 1.999])}
        missing = {'p': torch.nn.Parameter(initial['p'].clone())}
        with self.assertRaisesRegex(ValueError, 'zero/nonzero mismatch'):
            training._compare_displacements(initial, reference, missing)
        half = {'p': torch.nn.Parameter(initial['p'] + (reference['p']-initial['p'])*.5)}
        with self.assertRaisesRegex(ValueError, 'relative-error ceiling'):
            training._compare_displacements(initial, reference, half)

    def test_nonfinite_feedback_is_failure_and_no_optimizer_step(self):
        import torch
        model = trainer('G')
        original = model.teacher_features
        def invalid(ids):
            a, b = original(ids)
            b = b.clone(); b[0, 0, 0] = float('nan')
            return a, b
        model.teacher_features = invalid
        with self.assertRaisesRegex(ValueError, 'teacher features'):
            model.train_event(1, [response(1)])
        self.assertTrue(model.failed)
        self.assertEqual(model.optimizer_steps, 0)
        self.assertFalse(model.optimizer.state)


if __name__ == '__main__':
    unittest.main()
