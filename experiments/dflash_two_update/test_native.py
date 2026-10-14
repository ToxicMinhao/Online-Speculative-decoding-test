"""CPU contract tests; optional installed-source checks never import GPU code."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import native as N


class Tokenizer:
    """Tiny framing fixture with deliberately non-roundtrippable assistant IDs."""
    def __init__(self, trim=False):
        self.trim = trim

    def encode(self, text, add_special_tokens=False):
        values = []
        while text:
            if text.startswith(N.EOT):
                values.append(128009); text = text[len(N.EOT):]
            else:
                values.append(100 + ord(text[0])); text = text[1:]
        return values

    def apply_chat_template(self, messages, **_):
        text = ''
        for item in messages:
            content = item['content'].strip() if self.trim else item['content']
            text += '<|start_header_id|>' + item['role'] + '<|end_header_id|>\n\n' + content + N.EOT
        return [128000] + self.encode(text + N.ASSISTANT_HEADER)


def native_record(turn=17, version='U1'):
    ids, output = [1, 2, 3], [5, 99]
    block = {'block_index': 0, 'anchor': 3, 'context_length': 3,
        'positions': list(range(3, 13)), 'teacher_row_for_slot': [None] + list(range(9)),
        'teacher_prediction_positions': list(range(4, 14)),
        'noise_input_ids': [5] + [128002] * 9,
        'verification_input_token_ids': [5] + list(range(11, 20)),
        'native_top1': [99] + [20] * 9, 'accepted_before_stop': 0, 'bonus_or_correction': 99,
        'outcome': {'output_offset': 1, 'committed_token_ids': [99], 'finished': True},
        'eligible_mask': [False, True] + [False] * 8, 'eligible_positions': 1,
        'attention': 'all_original_context_and_all_same_block_noise_slots'}
    record = {'schema_version': 'd213_native_verification_response_v1', 'role': N.ROLE,
        'episode_id': 'R1-STABLE', 'turn': turn, 'record_id': 'R1-STABLE:t' + str(turn).zfill(2),
        'prompt_id': 'R1-STABLE:t' + str(turn).zfill(2), 'target_id': 'T0',
        'target_identity_sha256': 'a' * 64, 'draft_identity_sha256': 'b' * 64,
        'capture_checkpoint_sha256': 'b' * 64, 'draft_version': version, 'output_cap': 512,
        'input_ids': ids, 'output_ids': output, 'input_ids_sha256': N.object_hash(ids),
        'output_ids_sha256': N.object_hash(output), 'input_token_count': len(ids),
        'feature_layers': [1, 8, 15, 22, 29], 'feature_boundary': 'raw_target_aux_before_all_draft_transforms',
        'teacher_prehead_normalized': True, 'mask_token_id': 128002,
        'numerical_mode': {'dtype': 'bfloat16'}, 'source_sha256': {'fixture': 'c' * 64},
        'target_head_norm_binding': {'fixture': 'd' * 64}, 'blocks': [block], 'block_count': 1,
        'eligible_block_count': 1, 'eligible_position_count': 1, 'context_feature_rows': 4}
    record['cache_namespace'] = {key: record[key] for key in ('target_identity_sha256', 'draft_identity_sha256',
        'input_ids_sha256', 'numerical_mode', 'source_sha256', 'target_head_norm_binding', 'mask_token_id')}
    record['cache_namespace'].update(feature_layers=record['feature_layers'], dtype='bfloat16')
    record['cache_namespace_sha256'] = N.object_hash(record['cache_namespace'])
    N.add_response_ownership(record)
    return record


class NativeContractTests(unittest.TestCase):
    def test_history_preserves_original_output_ids_and_all_old_exchanges(self):
        tokenizer = Tokenizer()
        messages = [{'role': 'user', 'content': 'first'},
                    {'role': 'assistant', 'content': 'not used for tokens', 'output_ids': [4242, 4343]},
                    {'role': 'user', 'content': '  second  '}]
        first = N.encode_history(tokenizer, messages[:1])
        expected = first + [4242, 4343, 128009] + tokenizer.encode(N.USER_HEADER + '  second  ' + N.EOT + N.ASSISTANT_HEADER)
        self.assertEqual(N.encode_history(tokenizer, messages), expected)

    def test_history_requires_original_tokens_and_rejects_overlong_input(self):
        with self.assertRaisesRegex(ValueError, 'output_ids'):
            N.encode_history(Tokenizer(), [{'role': 'user', 'content': 'x'},
                {'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'y'}])
        with self.assertRaisesRegex(ValueError, 'truncation'):
            N.check_input([1] * (N.INPUT_LIMIT + 1))

    def test_first_and_later_prompt_trimming_fails_visibly(self):
        with self.assertRaisesRegex(ValueError, 'silent rewrite'):
            N.encode_history(Tokenizer(trim=True), [{'role': 'user', 'content': ' first '}])
        with self.assertRaisesRegex(ValueError, 'silent rewrite'):
            N.encode_history(Tokenizer(trim=True), [{'role': 'user', 'content': 'first'},
                {'role': 'assistant', 'content': '', 'output_ids': [128009]},
                {'role': 'user', 'content': ' next '}])

    def test_ownership_does_not_clip_valid_post_rejection_branch(self):
        document = native_record()
        block = document['blocks'][0]
        block['outcome']['finished'] = False
        block['eligible_mask'] = [False] + [True] * 9
        block['eligible_positions'] = 9
        N.add_response_ownership(document)
        self.assertEqual(block['owned_eligible_slots'], list(range(1, 10)))
        self.assertGreater(block['positions'][9], len(document['input_ids']) + len(document['output_ids']))
        self.assertIsNone(block['slot_owner_record_ids'][0])

    def test_training_audit_and_replay_share_persisted_metadata(self):
        from training import audit_response
        document = native_record()
        self.assertEqual(audit_response(document)['eligible_positions'], 1)
        api = SimpleNamespace(iter_blocks=lambda _: iter(copy.deepcopy(document['blocks'])))
        self.assertEqual(list(N.ReplayRecords(api).iter_blocks(document)), document['blocks'])
        api = SimpleNamespace(iter_blocks=lambda _: iter([{**document['blocks'][0], 'response_turn': 9}]))
        with self.assertRaisesRegex(ValueError, 'Replay ownership'):
            list(N.ReplayRecords(api).iter_blocks(document))

    def test_checkpoint_byte_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'model.safetensors').write_bytes(b'CPU fixture; never a model')
            (path / 'config.json').write_text('{}')
            files = {entry.name: {'bytes': entry.stat().st_size, 'sha256': N.sha(entry)} for entry in path.iterdir()}
            checkpoint = {'path': str(path), 'files': files, 'checkpoint_identity_sha256': N.object_hash(files)}
            N.verify_checkpoint(checkpoint)
            (path / 'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                N.verify_checkpoint(checkpoint)

    def test_backend_capture_rejects_frozen_second_support_before_worker_call(self):
        backend = N.NativeBackend({})
        backend.checkpoints = {'F': {}, 'U1': {}}
        with mock.patch.object(backend, '_ensure_worker', side_effect=AssertionError('must not run')):
            with self.assertRaisesRegex(ValueError, 'serving draft'):
                backend.serve([1, 2], 'F', capture=True, turn_index=17)

    def test_worker_command_uses_positional_paths_and_sources_helper_twice(self):
        config = dict(workspace_root='/local fixture/project', conda_bootstrap='/local fixture/conda.sh',
                      conda_prefix='/local fixture/env', python_executable='/local fixture/env/bin/python')
        command = N.worker_command(config)
        self.assertEqual(command[2].count('source "$1"'), 2)
        self.assertIn('conda activate "$3"', command[2])
        self.assertNotIn(config['workspace_root'], command[2])
        self.assertIn(config['python_executable'], command)

    def test_crashed_parent_still_cleans_owned_process_group(self):
        client = object.__new__(N._WorkerClient)
        client.process = mock.Mock(pid=987654)
        client.process.poll.return_value = 1
        client.log = None
        with mock.patch.object(N.os, 'killpg') as kill:
            client.close()
            kill.assert_called_once_with(987654, N.signal.SIGTERM)

    def test_trainer_service_uses_persistent_api_and_actual_export_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            service = object.__new__(N._TrainerService)
            document = native_record(turn=9, version='F')
            advertised = dict(document, record={'path': 'fixture'})
            service.records = SimpleNamespace(load_response=lambda *a, **kw: document)
            service.config = {'target': {'identity_sha256': 'a' * 64}}
            service.out = out
            service.native = SimpleNamespace(CORE=SimpleNamespace(state_fingerprints=lambda _: 'frozen'))
            service.components = None; service.frozen = 'frozen'
            service.trainer = mock.Mock()
            service.trainer.optimizer = SimpleNamespace(state={})
            service.trainer.train_event.return_value = {'optimizer_steps_total': 8, 'costs': {'training_s': 1.25}}
            service.trainer.save_checkpoint_state.return_value = {'path': 'state', 'includes_draft': False}
            checkpoint = {'checkpoint_id': 'U1', 'path': str(out / 'U1'),
                          'checkpoint_identity_sha256': 'e' * 64, 'export_s': .5}
            service._export = lambda _: checkpoint
            with mock.patch.object(N.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 2**30)):
                result = service.train({'event': 1, 'responses': [advertised]})
            self.assertEqual(result['training_s'], 1.25)
            self.assertEqual(result['optimizer_steps'], 8)
            service.trainer.bind_checkpoint.assert_called_once_with(1, 'e' * 64)
            service.trainer.save_checkpoint_state.assert_called_once_with(out / 'U1_training_state.pt', include_draft=False)

    def test_storage_bound_includes_both_exports_and_adam_states(self):
        self.assertEqual(N.episode_storage_bound(200) - N.episode_storage_bound(100), 600)
        self.assertGreater(N.episode_storage_bound(2097259104), 20 * 2**30)


@unittest.skipUnless(os.environ.get('TWO_UPDATE_WORKSPACE'), 'Optional installed-source checks need TWO_UPDATE_WORKSPACE')
class InstalledSourceTests(unittest.TestCase):
    def setUp(self):
        self.saved = {name: sys.modules.get(name) for name in ('records', 'cache')}
        self.records, self.cache = N.capture_modules(os.environ['TWO_UPDATE_WORKSPACE'])

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    def test_all_pinned_sources_match(self):
        manifest = N.check_dependencies(os.environ['TWO_UPDATE_WORKSPACE'])
        self.assertGreater(len(manifest['source_sha256']), 20)

    def test_actual_qualified_validator_accepts_u1_and_rejects_mislabeled_f(self):
        document = native_record()
        self.records.validate(document)
        document['draft_version'] = 'F'
        with self.assertRaisesRegex(ValueError, 'chronologically serving'):
            self.records.validate(document)

    def test_actual_validator_rejects_stale_capture_and_ownership(self):
        for change in ('capture_checkpoint_sha256', 'owner'):
            document = native_record()
            if change == 'owner':
                document['blocks'][0]['slot_owner_record_ids'][1] = 'earlier response'
            else:
                document[change] = 'e' * 64
            with self.assertRaises(ValueError):
                self.records.validate(document)

    def test_capture_compatibility_is_exact_and_recorded(self):
        self.assertEqual(len(self.cache.two_update_compatibility['edits']), 5)
        changes = self.cache.two_update_compatibility['edits']
        self.assertIn('Only contiguous initial prefill', changes[3]['after'])
        self.assertIn('Initial prefill is incomplete', changes[4]['after'])
        self.assertIn("draft_version=self.declaration['draft_version']",
                      [row['after'] for row in self.cache.two_update_compatibility['edits']])


if __name__ == '__main__':
    unittest.main()
