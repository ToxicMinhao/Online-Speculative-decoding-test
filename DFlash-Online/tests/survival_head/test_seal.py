"""Source completeness, immutable historical inputs and measured storage; CPU."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.survival_head import seal as S, preflight as P, storage
from experiments.survival_head.contract import ARMS
from online_feedback.storage import write_json


def source_fixture(base):
    code = base/'code'
    names = ['experiments/survival_head/'+name+'.py' for name in
             ('seal','preflight','campaign','native_pilot','contract','data','storage')]
    names += ['online_feedback/'+name+'.py' for name in
              ('survival_head','survival_head_runtime','survival_feedback_worker','objective','training','optimizer')]
    names += ['training/specforge/training/controller.py', 'training/specforge/core/chunking.py']
    for name in names:
        path = code/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# CPU source identity fixture\n')
    source = dict(status='verified', dirty=False, commit='b'*40, repository='fixture',
                  source_path=str(code), files={name:S.identity(code/name) for name in names})
    receipt = base/'SOURCE.json'; write_json(receipt, source)
    return code, receipt, source


def prior_fixture(base):
    prior = base/'prior'; sealed = prior/'sealed'
    requests = [dict(request_id=f'train-{i}', prompt_ids=[2,3], dataset='gsm8k',
        event_id='event_005', max_new_tokens=2048, collect_feedback=True) for i in range(128)]
    panel_requests = [dict(request_id=f'dev-{i}', prompt_ids=[1,i], panel_ordinal=i,
        original_index=i, stop_token_ids=[9], max_new_tokens=2048) for i in range(256)]
    write_json(sealed/'MODEL_PIN.json', {'target':{}, 'draft':{}})
    write_json(sealed/'RUNTIME_PIN.json', {'native_runtime_sha256':{'old.py':'c'*64}})
    write_json(sealed/'TRAINING_PIN.json', {'environment':{}})
    (sealed/'ORIGINAL_CONFIG.yaml').write_text('run_id: original\noutput_dir: /original\n')
    write_json(sealed/'gsm8k/TRAINING.json', dict(dataset='gsm8k', start_ordinal=512,
                                               stop_ordinal=768, train_requests=requests))
    write_json(sealed/'gsm8k/PANEL.json', dict(dataset='gsm8k', input_only_selection=True,
        previously_used_development_panel=True, requests=panel_requests))
    write_json(sealed/'gsm8k/CANONICALS.json', {row['request_id']:dict(prompt_ids=row['prompt_ids'],
        response_ids=[8,9], finish_reason='eos', verified=True, binding_digest='c'*64) for row in panel_requests})
    rows, features = [], []
    for index, request in enumerate(requests):
        canonical = prior/'records'/f'{index}.json'
        write_json(canonical, dict(record_id=request['request_id'], prompt_ids=[2,3], response_ids=[4,9]))
        feature = prior/'retired_features'/f'{index}.pt'
        rows.append(dict(record_id=request['request_id'], canonical=dict(path=str(canonical), **S.identity(canonical)),
                         feature=dict(path=str(feature), bytes=1000+index, sha256='d'*64)))
        features.append(dict(record_id=request['request_id'], length=4, prompt_length=2,
            response_tokens=2, adjacent_response_positions=1, path=str(feature)))
    write_json(prior/'campaign_001/gsm8k/event_005/shared/POOL.json',
               dict(dataset='gsm8k', event=5, producer='KPLUS4', records=rows))
    write_json(prior/'campaign_001/gsm8k/event_005/KPLUS/RESULT.json', dict(status='completed',
        method='KPLUS', dataset='gsm8k', event_id='event_005', start_step=512, end_step=640,
        optimizer_steps=128, passes=2, response_presentations=256, features=features,
        records=[dict(record_id=r['request_id'], **{'pass':p}, sampled_anchors=[{'positions':[[1]], 'keep_mask':[[True]]}])
                 for p in (0,1) for r in requests]))
    original = dict(origin_state='/original/state.pt', origin_export='/original/export',
        origin_weights='/original/export/model.safetensors',
        origin_state_identity=dict(bytes=3000,sha256='e'*64),
        origin_weights_identity=dict(bytes=1000,sha256='f'*64),
        training_manifest=str(sealed/'gsm8k/TRAINING.json'))
    plan = dict(root=str(prior), binding='a'*64, execution_source={'commit':'a'*40},
        datasets={'gsm8k':original}, target='/target', draft='/draft', worker_python='/worker', training_python='/trainer')
    write_json(sealed/'PLAN.json', plan)
    files = {str(path):S.identity(path) for path in sealed.rglob('*') if path.is_file()}
    write_json(sealed/'SEAL.json', dict(binding=plan['binding'], execution_source=plan['execution_source'], files=files))
    return prior, plan


def sealed_fixture(base):
    code, source_path, source = source_fixture(base)
    prior, _ = prior_fixture(base)
    root = base/'pilot'
    with patch.object(S, '__file__', str(code/'experiments/survival_head/seal.py')):
        plan = S.seal(root, prior, source_path)
    return root, code, plan


class SealTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_full_seal_uses_recorded_packet_sizes_preserves_source_and_history(self):
        root, code, plan = sealed_fixture(self.root)
        self.assertEqual(plan['main_storage']['original_pool_feature_bytes'], sum(1000+i for i in range(128)))
        self.assertEqual(plan['main_storage']['largest_original_feature_bytes'], 1127)
        self.assertEqual(plan['main_storage']['reserve_bytes'], 8*2**30)
        self.assertEqual(plan['main_storage']['feature_serialization_margin_bytes'], 128*64*1024)
        self.assertEqual(len(S.read(plan['panel_manifest'])['requests']), 128)
        self.assertEqual(set(plan['configs']), set(ARMS))
        with patch.object(P, '__file__', str(code/'experiments/survival_head/preflight.py')):
            self.assertEqual(P.verify_seal(root)[0], plan)
        with patch.object(S, '__file__', str(code/'experiments/survival_head/seal.py')):
            with self.assertRaises(FileExistsError):
                S.seal(root, self.root/'prior', self.root/'SOURCE.json')

    def test_missing_new_module_dirty_source_and_absolute_manifest_are_rejected(self):
        code, _, source = source_fixture(self.root)
        for mutate in (lambda row:row['files'].pop('online_feedback/survival_head.py'),
                       lambda row:row.update(dirty=True),
                       lambda row:row['files'].update({'/outside.py':{'bytes':1,'sha256':'a'*64}})):
            broken=deepcopy(source); mutate(broken)
            with self.assertRaises(ValueError): S.authenticate_source(broken, code)
        path=code/'experiments/survival_head/new_unsealed.py'; path.write_text('# extra\n')
        with self.assertRaisesRegex(ValueError,'omits'): S.authenticate_source(source,code)

    def test_modified_prior_plan_cannot_rebind_an_origin(self):
        code, source, _=source_fixture(self.root)
        prior, plan=prior_fixture(self.root)
        plan['datasets']['gsm8k']['origin_state']='/different/state.pt'
        write_json(prior/'sealed/PLAN.json',plan)
        with patch.object(S,'__file__',str(code/'experiments/survival_head/seal.py')):
            with self.assertRaisesRegex(ValueError,'prior plan'):
                S.seal(self.root/'pilot',prior,source)
        self.assertFalse((self.root/'pilot/sealed').exists())

    def test_measured_storage_keeps_overlap_margin_and_reserve(self):
        plan=storage.capacity_plan(1000,3000,feature_bytes=2012398528,max_feature_bytes=12345678)
        expected=2012398528+3*12345678+128*64*1024
        self.assertEqual(plan['shared_feature_peak_bytes'],expected)
        self.assertEqual(plan['tensor_peak_with_margin_bytes'],(plan['tensor_peak_bytes']*105+99)//100)
        self.assertEqual(plan['required_initial_free_bytes'],
                         plan['tensor_peak_with_margin_bytes']+plan['compact_bytes']+14*2**30)
        self.assertFalse(plan['offload_assumed'])
        self.assertTrue(plan['no_future_deletion_credit'])
        for values in ({'feature_bytes':3}, {'feature_bytes':3,'max_feature_bytes':4},
                       {'feature_bytes':3.5,'max_feature_bytes':1}):
            with self.assertRaises(ValueError): storage.capacity_plan(1000,3000,**values)


if __name__=='__main__': unittest.main()
