"""CPU replay contracts with fake target forwards and the native crop helper.

The five legacy FeedbackTests named below require the superseded full-sequence,
no-cache/64-row-head implementation. Their scientific boundary checks are covered
by the named native-replay replacements; the original tests remain preserved.
All other legacy tests, including ineligible-feedback checks, remain applicable.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from online_feedback.native_replay import replay_frozen_target
from online_feedback.native import NativeDFlashBackend
from online_feedback.training import validate_target_binding


SUPERSEDED_LEGACY_FEEDBACK_TESTS = {
    'test_replay_saves_complete_conditioning_and_response_only_loss':
        'test_prepare_feedback_preserves_mask_payload_and_rng',
    'test_target_argmax_alignment_spans_vocabulary_head_chunks':
        'test_accepted_bonus_then_cap_keeps_only_canonical_features',
    'test_mismatched_target_replay_saves_evidence_without_training_features':
        'test_wrong_first_label_saves_failure_before_any_verification / '
        'test_uncommitted_posterior_mismatch_is_rejected',
    'test_teacher_rng_and_preparation_time_are_separate_from_persistence':
        'test_prepare_feedback_separates_target_work_from_persistence',
    'test_nonfinite_features_are_rejected':
        'test_prepare_feedback_rejects_nonfinite_features_with_evidence',
}


class FakeCache:
    _dflash_legacy_absolute_crop = True

    def __init__(self):
        self.tokens = []
        self.crops = []

    def get_seq_length(self):
        return len(self.tokens)

    def crop(self, length):
        self.tokens = self.tokens[:length]
        self.crops.append(length)


class FakeTarget:
    config = SimpleNamespace()

    def __init__(self, posterior_rows):
        self.posterior_rows = posterior_rows
        self.calls = []

    def __call__(self, input_ids, *, position_ids, past_key_values, use_cache,
                 output_hidden_states, logits_to_keep=None):
        tokens = input_ids[0].tolist()
        begin = len(past_key_values.tokens)
        assert position_ids.tolist() == [list(range(begin, begin + len(tokens)))]
        assert use_cache and output_hidden_states
        self.calls.append({'tokens': tokens, 'start': begin, 'logits_to_keep': logits_to_keep})
        torch.rand(())  # The integration boundary must preserve training RNG.
        # Each hidden row encodes its causal prefix, exposing any retained draft
        # suffix or failure to crop the cache in the final concatenated tensor.
        hidden_rows = []
        for position, token in enumerate(tokens, begin):
            past_key_values.tokens.append(token)
            hidden_rows.append([position, sum(past_key_values.tokens)])
        hidden = torch.tensor([hidden_rows], dtype=torch.float32)
        posterior = self.posterior_rows[len(self.calls) - 1]
        logits = torch.full((1, len(posterior), 128), -1.0)
        for index, token in enumerate(posterior):
            logits[0, index, token] = 1.0
        return SimpleNamespace(logits=logits, hidden_states=(hidden - 1, hidden, hidden + 1000))


def block(start, anchor, drafts, posterior, progress):
    return {
        'output_start': start, 'anchor_token': anchor, 'draft_tokens': drafts,
        'verify_size': len(drafts) + 1, 'target_argmax': posterior,
        'committed_progress': progress, 'committed_tokens': posterior[:progress],
    }


def record(response, trace):
    return {'record_id': 'fixture', 'prompt_ids': [2, 3], 'response_ids': response,
            'metadata': {'trace': trace}}


class NativeReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.failure = Path(self.temporary.name) / 'failure.json'

    def replay(self, source, posterior_rows=None):
        self.cache = FakeCache()
        rows = posterior_rows or ([[source['response_ids'][0]]]
                                 + [b['target_argmax'] for b in source['metadata']['trace']] + [[0]])
        self.target = FakeTarget(rows)
        with patch('dflash.model._make_cache', return_value=self.cache), torch.inference_mode():
            return replay_frozen_target(self.target, [0], source, 'cpu', self.failure)

    def assert_canonical_features(self, source, features, hidden, teacher):
        tokens = source['prompt_ids'] + source['response_ids']
        expected = torch.tensor([[[index, sum(tokens[:index + 1])]
                                  for index in range(len(tokens))]], dtype=torch.float32)
        torch.testing.assert_close(features, expected, rtol=0, atol=0)
        torch.testing.assert_close(hidden, expected + 1000, rtol=0, atol=0)
        self.assertEqual(teacher, source['response_ids'])
        self.assertFalse(self.failure.exists())

    def test_accepted_bonus_then_cap_keeps_only_canonical_features(self):
        source = record([10, 11, 12, 20, 21, 22], [
            block(0, 10, [11, 12, 99], [11, 12, 20, 50], 3),
            block(3, 20, [21, 22], [21, 22, 88], 2),
        ])
        features, hidden, teacher, diagnostics = self.replay(source)
        self.assert_canonical_features(source, features, hidden, teacher)
        self.assertEqual(self.cache.crops, [2, 5, 7])
        self.assertEqual(self.cache.tokens, source['prompt_ids'] + source['response_ids'])
        self.assertEqual([c['start'] for c in self.target.calls], [0, 2, 5, 7])
        self.assertEqual(diagnostics['target_forward_count'], 4)
        self.assertEqual(diagnostics['full_posterior_rows_checked'], 7)
        self.assertEqual(diagnostics['canonical_feature_rows_before_final'], 7)
        self.assertFalse(diagnostics['final_feature_logits_used_for_labels'])

    def test_eos_clipping_discards_uncommitted_input_features(self):
        source = record([10, 11, 99], [block(0, 10, [11, 99, 77], [11, 99, 8, 55], 2)])
        features, hidden, teacher, diagnostics = self.replay(source)
        self.assert_canonical_features(source, features, hidden, teacher)
        self.assertEqual(self.cache.crops, [2, 4])
        self.assertEqual(self.target.calls[-1]['tokens'], [99])
        self.assertEqual(diagnostics['full_posterior_rows_checked'], 4)

    def test_wrong_first_label_saves_failure_before_any_verification(self):
        source = record([10, 11], [block(0, 10, [11], [11, 12], 1)])
        with self.assertRaisesRegex(RuntimeError, 'prefill_argmax_mismatch'):
            self.replay(source, [[9]])
        evidence = json.loads(self.failure.read_text())
        self.assertEqual(evidence['actual'], [9])
        self.assertEqual(evidence['expected'], [10])
        self.assertEqual(len(self.target.calls), 1)

    def test_uncommitted_posterior_mismatch_is_rejected(self):
        source = record([10, 11], [block(0, 10, [11, 70], [11, 60, 50], 1)])
        with self.assertRaisesRegex(RuntimeError, 'full_verification_posterior_mismatch'):
            self.replay(source, [[10], [11, 60, 51]])
        evidence = json.loads(self.failure.read_text())
        self.assertEqual(evidence['block_index'], 0)
        self.assertEqual(evidence['actual'], [11, 60, 51])
        self.assertEqual(evidence['diagnostics']['final_feature_forward_count'], 0)

    def test_malformed_geometry_and_noncanonical_retained_prefix_are_rejected(self):
        source = record([10, 11, 12], [block(0, 10, [11, 12], [11, 12, 50], 2)])
        changes = [
            ('target_argmax', [11, 12], 'invalid_block_geometry'),
            ('committed_progress', 4, 'invalid_block_geometry'),
            ('output_start', 1, 'invalid_block_geometry'),
            ('draft_tokens', [77, 12], 'noncanonical_committed_prefix'),
        ]
        for key, value, reason in changes:
            with self.subTest(field=key):
                changed = copy.deepcopy(source)
                changed['metadata']['trace'][0][key] = value
                with self.assertRaisesRegex(RuntimeError, reason):
                    self.replay(changed)
                self.assertEqual(json.loads(self.failure.read_text())['reason'], reason)

    def test_prepare_feedback_preserves_mask_payload_and_rng(self):
        source = record([10, 11, 12], [block(0, 10, [11, 12], [11, 12, 50], 2)])
        source['purpose'] = 'serving'
        native = NativeDFlashBackend.__new__(NativeDFlashBackend)
        native.torch, native.device, native.context_limit = torch, torch.device('cpu'), 128
        native.target = FakeTarget([[10], [11, 12, 50], [0]])
        native.draft = SimpleNamespace(target_layer_ids=[0])
        path = Path(self.temporary.name) / 'features.pt'
        rng_before = torch.get_rng_state().clone()
        with patch('dflash.model._make_cache', return_value=FakeCache()), patch('torch.cuda.synchronize'):
            receipt = native.prepare_feedback(source, path)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        payload = torch.load(path, weights_only=True)
        self.assertEqual(payload['input_ids'].tolist(), [[2, 3, 10, 11, 12]])
        self.assertEqual(payload['loss_mask'].tolist(), [[0, 0, 1, 1, 1]])
        self.assertEqual(receipt['target_argmax_ids'], source['response_ids'])
        self.assertEqual(receipt['supervised_positions'], 3)
        self.assertEqual(receipt['native_replay']['target_forward_count'], 3)
        self.assertEqual(receipt['capture_backend'], 'same_frozen_HF_target_saved_native_block_replay')
        self.assertEqual(set(payload), {
            'input_ids', 'loss_mask', 'hidden_states', 'target_last_hidden_states',
            'record_id', 'prompt_length', 'label_binding'})
        validate_target_binding(payload)
        for key, value in (('full_posterior_match', False),
                           ('final_feature_logits_used_for_labels', True)):
            with self.subTest(unqualified_field=key):
                unqualified = copy.deepcopy(payload)
                unqualified['label_binding']['native_replay'][key] = value
                with self.assertRaisesRegex(ValueError, 'verified canonical target-greedy labels'):
                    validate_target_binding(unqualified)

    def test_prepare_feedback_rejects_nonfinite_features_with_evidence(self):
        from dflash.model import extract_context_feature

        source = record([10, 11, 12], [block(0, 10, [11, 12], [11, 12, 50], 2)])
        source['purpose'] = 'serving'
        native = NativeDFlashBackend.__new__(NativeDFlashBackend)
        native.torch, native.device, native.context_limit = torch, torch.device('cpu'), 128
        native.target = FakeTarget([[10], [11, 12, 50], [0]])
        native.draft = SimpleNamespace(target_layer_ids=[0])
        path = Path(self.temporary.name) / 'features.pt'

        def nonfinite_features(states, layer_ids):
            features = extract_context_feature(states, layer_ids)
            return torch.full_like(features, float('nan'))

        with (patch('dflash.model._make_cache', return_value=FakeCache()),
              patch('dflash.model.extract_context_feature', side_effect=nonfinite_features),
              patch('torch.cuda.synchronize')):
            with self.assertRaisesRegex(RuntimeError, 'invalid_collected_features'):
                native.prepare_feedback(source, path)
        self.assertFalse(path.exists())
        self.assertFalse(path.with_suffix('.json').exists())
        failure = json.loads(path.with_suffix('.native_replay_failure.json').read_text())
        self.assertEqual(failure['reason'], 'invalid_collected_features')
        self.assertEqual(failure['record_id'], source['record_id'])
        self.assertEqual(failure['response_ids'], source['response_ids'])

    def test_prepare_feedback_separates_target_work_from_persistence(self):
        clock = [0.0]

        class TimedTarget(FakeTarget):
            def __call__(self, *args, **kwargs):
                self_inference = torch.is_inference_mode_enabled()
                assert self_inference
                clock[0] += 3
                return super().__call__(*args, **kwargs)

        def persist(path, payload):
            clock[0] += 20
            torch.save(payload, path)

        source = record([10, 11, 12], [block(0, 10, [11, 12], [11, 12, 50], 2)])
        source['purpose'] = 'serving'
        native = NativeDFlashBackend.__new__(NativeDFlashBackend)
        native.torch, native.device, native.context_limit = torch, torch.device('cpu'), 128
        native.target = TimedTarget([[10], [11, 12, 50], [0]])
        native.draft = SimpleNamespace(target_layer_ids=[0])
        path = Path(self.temporary.name) / 'features.pt'
        with (patch('dflash.model._make_cache', return_value=FakeCache()),
              patch('torch.cuda.synchronize'),
              patch('online_feedback.native.time.perf_counter', side_effect=lambda: clock[0]),
              patch('online_feedback.native.save_torch', side_effect=persist)):
            receipt = native.prepare_feedback(source, path)
        self.assertEqual(len(native.target.calls), 3)
        self.assertEqual(receipt['teacher_preparation_s'], 9)
        self.assertEqual(receipt['feature_persistence_s'], 20)
        self.assertEqual(clock[0], 29)
        self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
