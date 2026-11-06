"""CPU fixtures for bounded native reconstruction and immutable anchor bindings."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.adaflash_clipping import prepare as module
from online_feedback.storage import write_json


def source_record(record_id, ordinal):
    prompt, response = [101 + ordinal, 102], [7, 8, 99]
    block = dict(anchor_token=7, output_start=0, verify_size=16,
        draft_tokens=[8] + list(range(12, 26)), target_argmax=[8, 99] + list(range(30, 44)),
        positions=list(range(2, 18)), committed_progress=2, committed_tokens=[8, 99],
        accepted_draft_tokens=1, native_accepted_draft_tokens=1,
        raw_accepted_draft_tokens_before_stop=1, stop_clipped=False)
    return dict(record_id=record_id, session_id='gsm8k_train', request_index=ordinal,
        checkpoint_id='/models/released', purpose='serving', prompt_ids=prompt,
        response_ids=response, finish_reason='eos', stop_token_ids=[99], max_new_tokens=2048,
        metadata=dict(block_size=16, context_limit=4096, runtime='pinned_SGLang_FA3', trace=[block],
            capture=dict(prompt_ids=prompt, response_ids=response, blocks=[block])))


def binding_for(source):
    return dict(record_id=source['record_id'], prompt_length=2, response_length=3,
        input_ids=source['prompt_ids'] + source['response_ids'], target_argmax_ids=source['response_ids'],
        target_layer_ids=[1, 2, 3], feature_shape=[1, 5, 12],
        label_binding_scope='native_prefill_and_committed_verification',
        response_only_loss=True, final_feature_logits_used_for_labels=False)


def create_campaign(root):
    arm = root / 'arms/A'
    rows = []
    for ordinal, record_id in enumerate(module.SOURCE_IDS):
        source = source_record(record_id, ordinal)
        anchors = [dict(positions=[[2, 3]], keep_mask=[[True, True]])]
        rows.append(dict(record_id=record_id, **{'pass': 0}, sampled_anchors=anchors,
            prompt_tokens=2, response_tokens=3, active_anchors=2,
            response_observation=dict(record_id=record_id, event_id='event_001', pass_index=0,
                presentation_ordinal=ordinal, sampled_anchors=deepcopy(anchors))))
        write_json(arm / f'events/event_001/loop/serving/{record_id}.json', source)
        write_json(arm / f'events/event_001/loop/features/{record_id}.json', binding_for(source))
    result = dict(status='completed', passes=2, response_presentations=256,
        records=rows + [dict(record_id='unselected', **{'pass': 0}) for _ in range(250)])
    path = arm / 'trainer/events/event_001/RESULT.json'
    write_json(path, result)
    return path


class FakeBackend:
    instances = []
    failure = None
    bad_binding = False

    def __init__(self, target_path, draft_path, **kwargs):
        self.kwargs = kwargs
        self.loading_s = 0.25
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)
        output = Path(kwargs['output_root']).parent
        assert (output / 'SELECTION.json').exists(), 'Selection must precede native startup'

    def _call(self, operation, *, name, arguments):
        assert (operation, name) == ('depth_rpc', 'depth_reconstruct')
        source = deepcopy(arguments['record'])
        self.calls.append(('reconstruct', source['record_id']))
        receipt = dict(status='passed', record_id=source['record_id'], new_generations=0,
            prefill_forwards=1, verification_forwards=1, posterior_positions_checked=16,
            active_forward=None, wall_s=0.1, record=source)
        if self.failure == source['record_id']:
            receipt.update(status='failed', verification_forwards=0, posterior_positions_checked=0,
                active_forward=dict(kind='verification', block=0), error='fixture posterior mismatch')
            write_json(arguments['outputfile'], receipt)
            raise RuntimeError('fixture posterior mismatch')
        write_json(arguments['outputfile'], receipt)
        return dict(result=receipt, rpc_wall_s=0.11)

    def prepare_feedback(self, source, path):
        self.calls.append(('terminal', source['record_id']))
        write_json(path.with_suffix('.terminal.json'), dict(target_forward_count=1))
        path.write_bytes(b'fixture prepared feature, not a tensor')
        result = dict(**binding_for(source), status='prepared', feature_path=str(path),
            teacher_preparation_s=0.2, feature_persistence_s=0.01)
        if self.bad_binding:
            result['target_layer_ids'] = [0, 1, 2]
        return result

    def generate(self, *args, **kwargs):
        raise AssertionError('Reconstruction must never generate another answer')

    def close(self):
        self.closed = True


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = self.root / 'campaign_001'
        self.result_path = create_campaign(self.campaign)
        self.output = self.root / 'diagnostic'
        self.config = dict(target_path='/models/target', draft_path='/models/released',
                           worker_python='/env/python', serving_gpu='0')
        FakeBackend.instances = []
        FakeBackend.failure = None
        FakeBackend.bad_binding = False
        self.backend = patch.object(module, 'SGLangDFlashBackend', FakeBackend)
        self.backend.start()
        self.addCleanup(self.backend.stop)

    def change_result(self, action):
        result = json.loads(self.result_path.read_text())
        action(result)
        write_json(self.result_path, result)

    def test_exact_first_six_and_retired_tensors_are_not_required(self):
        selection = module.select_sources(self.campaign)
        self.assertEqual(tuple(row['record_id'] for row in selection['records']), module.SOURCE_IDS)
        self.assertEqual(selection['pairs'], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(selection['expected_native_verification_forwards'], 6)
        self.assertEqual(selection['records'], json.loads(self.result_path.read_text())['records'][:6])
        self.assertEqual(len(selection['source_identities']), 13)
        self.assertEqual(list(self.campaign.rglob('*.pt')), [])

    def test_source_order_cannot_be_changed(self):
        self.change_result(lambda result: result['records'].reverse())
        with self.assertRaisesRegex(ValueError, 'first six'):
            module.prepare(self.config, self.campaign, self.output)
        self.assertEqual(FakeBackend.instances, [])

    def test_anchor_binding_mismatch_stops_before_gpu(self):
        self.change_result(lambda result: result['records'][0]['sampled_anchors'][0]['positions'][0].__setitem__(0, 1))
        with self.assertRaisesRegex(ValueError, 'anchor eligibility'):
            module.prepare(self.config, self.campaign, self.output)
        self.assertEqual(FakeBackend.instances, [])

    def test_anchor_observation_cannot_be_replaced(self):
        self.change_result(lambda result: result['records'][0]['response_observation'].__setitem__('presentation_ordinal', 1))
        with self.assertRaisesRegex(ValueError, 'anchor-observation'):
            module.select_sources(self.campaign)

    def test_native_proposal_and_posterior_binding_is_checked_before_replay(self):
        path = self.campaign / f'arms/A/events/event_001/loop/serving/{module.SOURCE_IDS[0]}.json'
        source = json.loads(path.read_text())
        source['metadata']['trace'][0]['draft_tokens'][0] = 9
        source['metadata']['capture']['blocks'] = deepcopy(source['metadata']['trace'])
        write_json(path, source)
        with self.assertRaisesRegex(ValueError, 'native trace'):
            module.select_sources(self.campaign)

    def test_six_reconstructions_and_terminal_preparations_without_generation(self):
        result = module.prepare(self.config, self.campaign, self.output)
        backend = FakeBackend.instances[0]
        self.assertTrue(backend.closed)
        self.assertEqual(backend.kwargs['worker_module'], 'experiments.block_depth.worker')
        self.assertEqual((backend.kwargs['context_limit'], backend.kwargs['block_size']), (4096, 16))
        self.assertEqual(backend.calls, [(kind, record_id) for record_id in module.SOURCE_IDS
                                       for kind in ('reconstruct', 'terminal')])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['feature_paths']), 6)
        self.assertEqual(result['records'], json.loads(self.result_path.read_text())['records'][:6])
        self.assertEqual(result['work']['completed'], module.LIMITS)
        self.assertEqual(result['work']['unknown'], dict(native_reconstructions=0, terminal_preparations=0))
        self.assertEqual(result['work']['new_generations'], 0)
        self.assertEqual(result['work']['native_prefill_forwards'], 6)
        self.assertEqual(result['work']['native_verification_forwards'], 6)
        self.assertEqual(result['work']['posterior_positions_checked'], 96)
        self.assertEqual(result['work']['terminal_preparation_forwards'], 6)
        self.assertEqual(result['work']['active_or_unknown_forwards'], [])

    def test_failed_native_reconstruction_retains_partial_and_unattempted_work(self):
        FakeBackend.failure = module.SOURCE_IDS[2]
        with self.assertRaisesRegex(RuntimeError, 'posterior mismatch'):
            module.prepare(self.config, self.campaign, self.output)
        result = json.loads((self.output / 'PREPARED.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['feature_paths']), 2)
        work = result['work']
        self.assertEqual(work['attempts'], dict(native_reconstructions=3, terminal_preparations=2))
        self.assertEqual(work['completed'], dict(native_reconstructions=2, terminal_preparations=2))
        self.assertEqual(work['unknown'], dict(native_reconstructions=1, terminal_preparations=0))
        self.assertEqual(work['unattempted'], dict(native_reconstructions=3, terminal_preparations=4))
        self.assertEqual(work['native_prefill_forwards'], 3)
        self.assertEqual(work['native_verification_forwards'], 2)
        self.assertEqual(len(work['active_or_unknown_forwards']), 1)
        self.assertTrue(FakeBackend.instances[0].closed)

    def test_reconstructed_feature_contract_must_match_retained_receipt(self):
        FakeBackend.bad_binding = True
        with self.assertRaisesRegex(ValueError, 'feature binding'):
            module.prepare(self.config, self.campaign, self.output)
        result = json.loads((self.output / 'PREPARED.json').read_text())
        self.assertEqual(result['work']['completed'], dict(native_reconstructions=1, terminal_preparations=1))
        self.assertEqual(result['feature_paths'], [])

    def test_existing_output_is_never_retried_or_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            module.prepare(self.config, self.campaign, self.output)
        self.assertEqual(FakeBackend.instances, [])

    def test_cleanup_failure_still_persists_completed_work(self):
        with patch.object(FakeBackend, 'close', side_effect=RuntimeError('fixture close failed')):
            with self.assertRaisesRegex(RuntimeError, 'close failed'):
                module.prepare(self.config, self.campaign, self.output)
        result = json.loads((self.output / 'PREPARED.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertIn('close failed', result['cleanup_error'])
        self.assertEqual(result['work']['completed'], module.LIMITS)
        self.assertEqual(len(result['feature_paths']), 6)


if __name__ == '__main__':
    unittest.main()
