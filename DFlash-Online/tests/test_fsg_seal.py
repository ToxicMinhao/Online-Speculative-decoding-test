"""Qualification sealing is input-only and retains exact continuation origins."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.functional_step_geometry import seal as S
from experiments.functional_step_geometry.contract import DATASETS, SCHEMA
from online_feedback.storage import write_json
from test_ok_coding_training import configuration


def status(dataset):
    export = f'/prior/{dataset}/events/event_004/export'
    state = f'/prior/{dataset}/events/event_004/training_state.pt'
    return dict(status='completed', dataset=dataset, arm='KPLUS', binding='a'*64,
        final_checkpoint=export, final_state=state,
        boundary=dict(status='passed', dataset=dataset, arm='KPLUS', event=4,
            state=state, weights=export+'/model.safetensors',
            state_identity=dict(bytes=123, sha256='1'*64),
            weights_identity=dict(bytes=42, sha256='2'*64),
            scientific_check=dict(status='passed', global_step=512, presentations=1024,
                export_exact=True, live=dict(status='passed', exact_resume=True, mismatches=[]))))


def row(index, dataset):
    key = f'{dataset}:{index}'
    return dict(dataset=dataset, source_id=key, request_id=key, prompt_ids=[index+100],
        normalized_stem=key, rendered_prompt_sha256=key, prompt_ids_sha256=key,
        event_id=f'event_{index//128+1:03d}', collect_feedback=True, max_new_tokens=2048)


def inputs(root):
    prepared = root/'prepared'; prepared.mkdir()
    prior_files = {}
    for dataset in DATASETS:
        path = root/'old_inputs'/dataset/'TRAINING.json'
        write_json(path,dict(train_requests=[row(i,dataset) for i in range(512)]))
        prior_files[str(path)] = S.identity(path)
    write_json(prepared/'PREPARATION.json', dict(schema=SCHEMA, status='sealed_inputs',
        model_executions=0, training_requests=512, events=[5,6], source_answers_used=False,
        input_identities=prior_files))
    write_json(prepared/'INPUT_AUDIT.json', dict(status='passed', model_executions=0,
        selection_uses_model_results=False, source_answers_used=False))
    write_json(prepared/'MATHQA_EXCLUSIONS.json', [])
    for dataset in DATASETS:
        folder=prepared/dataset; folder.mkdir()
        write_json(folder/'TRAINING.json', dict(dataset=dataset,
            train_requests=[row(i,dataset) for i in range(512,768)],
            source_answers_used=False, selection_uses_outputs=False,
            start_ordinal=512, stop_ordinal=768))
        panel=[row(i,dataset) for i in range(2000,2256)]
        write_json(folder/'PANEL.json', dict(requests=panel))
        write_json(folder/'CANONICALS.json', {r['request_id']:dict(response_ids=[1]) for r in panel})
    return prepared


class SealTests(unittest.TestCase):
    def test_exact_origin_receipt_and_foreign_boundaries(self):
        source=status('gsm8k'); before=deepcopy(source)
        actual=S.origin_from_status(source,dataset='gsm8k',prior_binding='a'*64)
        self.assertEqual(actual['origin_state'],source['final_state'])
        self.assertEqual(actual['origin_weights_identity'],dict(bytes=42,sha256='2'*64))
        self.assertEqual(source,before)
        for mutation in (
            lambda x:x.update(arm='ROLE'),
            lambda x:x['boundary'].update(event=3),
            lambda x:x['boundary']['scientific_check'].update(global_step=640),
            lambda x:x.update(final_state='/guessed/latest.pt'),
            lambda x:x['boundary']['state_identity'].update(bytes=0),
        ):
            changed=deepcopy(source);mutation(changed)
            with self.assertRaises(ValueError):
                S.origin_from_status(changed,dataset='gsm8k',prior_binding='a'*64)

    def test_published_commit_must_be_exact_and_clean(self):
        with patch.object(S.subprocess,'check_output',side_effect=['b'*40+'\n','']):
            self.assertEqual(S.published_source('/tmp','b'*40)['commit'],'b'*40)
        for outputs in (['c'*40+'\n',''],['b'*40+'\n',' M online_feedback/training.py\n']):
            with patch.object(S.subprocess,'check_output',side_effect=outputs),self.assertRaises(ValueError):
                S.published_source('/tmp','b'*40)

    def test_input_only_preparation_checks_overlap_and_fixed_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared=inputs(Path(temporary))
            self.assertEqual(S.check_prepared(prepared)['model_executions'],0)
            path=prepared/'gsm8k/TRAINING.json'; manifest=json.loads(path.read_text())
            manifest['train_requests'][0]['prompt_ids']=[1]*225
            write_json(path,manifest)
            with self.assertRaises(ValueError): S.check_prepared(prepared)

    def test_prior_feedback_overlap_is_rejected_before_sealing(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared=inputs(Path(temporary)); path=prepared/'gsm8k/TRAINING.json'
            manifest=json.loads(path.read_text())
            manifest['train_requests'][0]['normalized_stem']='gsm8k:1'
            write_json(path,manifest)
            with self.assertRaises(ValueError): S.check_prepared(prepared)

    def test_full_seal_contains_remote_paths_fixed_numerics_and_no_pilot_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); prepared=inputs(root)
            prior=root/'prior'; prior.mkdir()
            write_json(prior/'PLAN.json',dict(binding='a'*64,target='/target',draft='/draft',
                training_python='/training/python',worker_python='/worker/python'))
            (prior/'ORIGINAL_CONFIG.yaml').write_text(configuration(17))
            for name in S.PIN_FILES[1:]: write_json(prior/name,dict(pin=name))
            retained=root/'retained'
            for dataset in DATASETS:
                write_json(retained/'campaign_001/trajectories'/dataset/'KPLUS/STATUS.json',status(dataset))
            handoff=root/'HANDOFF.md'; handoff.write_text('Fixed handoff')
            code=root/'code'; code.mkdir(); (code/'tracked.py').write_text('pass\n')
            output=root/'qualification'
            execution=dict(commit='b'*40,dirty=False,source_path=str(S.REMOTE_CODE),repository=S.REPOSITORY)
            args=SimpleNamespace(code=code,prepared=prepared,prior_sealed=prior,retained=retained,
                output=output,handoff=handoff,commit='b'*40)
            # Git outputs contain only retained report metadata/body and the source inventory.
            with patch.object(S,'published_source',return_value=execution),patch.object(
                    S.subprocess,'check_output',side_effect=[S.PR17_BLOB+'\n',b'# retained report\n',b'tracked.py\0']):
                result=S.seal(args)
            self.assertFalse(result['main_pilot_authorized'])
            plan=json.loads((output/'sealed/PLAN.json').read_text())
            self.assertEqual(plan['root'],str(S.REMOTE_ROOT))
            self.assertEqual(plan['datasets']['mathqa']['origin_export'],status('mathqa')['final_checkpoint'])
            self.assertEqual(plan['numerical_recipe']['native_vjp_tolerances']['rtol'],.016)
            self.assertEqual(plan['numerical_recipe']['qualification_work_limits']['adam'],8)
            self.assertEqual(plan['evidence']['pr17']['blob'],S.PR17_BLOB)
            files=json.loads((output/'sealed/SEAL.json').read_text())['files']
            self.assertIn(str(S.REMOTE_CODE/'tracked.py'),files)
            self.assertIn(str(S.REMOTE_ROOT/'sealed/PLAN.json'),files)
            self.assertNotIn(str(output/'sealed/PLAN.json'),files)
            self.assertEqual(result['shared_feedback_generations'],4)


if __name__=='__main__': unittest.main()
