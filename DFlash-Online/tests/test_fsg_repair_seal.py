"""Preserve the failed attempt and count its work before qualification recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.functional_step_geometry import seal_repair as S
from experiments.functional_step_geometry.contract import QUALIFICATION_BUDGET
from online_feedback.storage import write_json
from test_ok_coding_training import configuration


def fixture(root):
    old_sealed=root/'qualification_001/sealed';old_sealed.mkdir(parents=True)
    failure=root/'failure_001';failure.mkdir()
    plan=dict(execution_source=dict(commit=S.FAILED_COMMIT),binding='a'*64,
        root=str(S.OLD_ROOT),scope='bounded_qualification_only',main_pilot_authorized=False,
        automatic_retry=False,original_config=str(S.OLD_ROOT/'sealed/ORIGINAL_CONFIG.yaml'),
        numerical_recipe=S.original.numerical_recipe(),qualification_budget=QUALIFICATION_BUDGET,
        datasets={})
    (old_sealed/'ORIGINAL_CONFIG.yaml').write_text(configuration(17))
    preparation=dict(status='completed',shared_feedback_generations=4,teacher_preparations=4,
        extra_qualification_generations=0,main_updates=0,datasets={})
    for dataset in ('gsm8k','mathqa'):
        folder=old_sealed/dataset;folder.mkdir()
        write_json(folder/'TRAINING.json',dict(train_requests=[dict(request_id='first',prompt_ids=[1]),dict(request_id='second',prompt_ids=[2])]))
        write_json(folder/'PANEL.json',dict(requests=[]));write_json(folder/'CANONICALS.json',{})
        configs={}
        for arm in ('KPLUS','FSG','SCALAR'):
            name=f'config_{arm}.yaml';(folder/name).write_text(configuration(17))
            configs[arm]=str(S.OLD_ROOT/'sealed'/dataset/name)
        plan['datasets'][dataset]=dict(configs=configs,
            training_manifest=str(S.OLD_ROOT/'sealed'/dataset/'TRAINING.json'),
            panel_manifest=str(S.OLD_ROOT/'sealed'/dataset/'PANEL.json'),
            references=str(S.OLD_ROOT/'sealed'/dataset/'CANONICALS.json'),origin_state='/retained/'+dataset,
            origin_export='/retained/'+dataset+'/export')
        rows=[]
        for ordinal,key in enumerate(('first','second'),1):
            relative=Path('qualification_feedback')/dataset/'records'/f'{key}.json'
            write_json(failure/relative,dict(record_id=key,prompt_ids=[ordinal],checkpoint_id='/retained/'+dataset+'/export'))
            rows.append(dict(request_id=key,status='completed',record_path=str(S.OLD_ROOT/relative),
                record_identity=S.identity(failure/relative),feature_path=str(S.OLD_ROOT/'qualification_feedback'/dataset/'features'/f'{key}.pt'),
                feature_identity=dict(bytes=10,sha256='e'*64)))
        preparation['datasets'][dataset]=rows
    write_json(failure/'qualification_feedback/PREPARATION.json',preparation)
    write_json(old_sealed/'PLAN.json',plan)
    files={str(S.OLD_ROOT/'sealed'/p.relative_to(old_sealed)):S.identity(p) for p in old_sealed.rglob('*') if p.is_file()}
    write_json(old_sealed/'SEAL.json',dict(files=files))
    summary=dict(status='failed',main_pilot_started=False,execution_commit=S.FAILED_COMMIT,
        error="ValueError('Native Adam counter differs: first.weight')")
    for key,relative in S.REFERENCE_FILES.items():
        if key=='presentations':value=[dict(optimizer_stepped=False),dict(optimizer_stepped=True)]
        elif key in ('decoded_state','startup_restoration'):value=dict(exact_equal=True)
        elif key=='failed_qualification_summary':value=summary
        else:value=dict(retained=True)
        write_json(failure/relative,value)
    ledger=[]
    for kind,amount in [('adam',1),('ordinary_forward',2),('ordinary_backward',2),('additional_forward',2),('additional_backward',2)]:
        for _ in range(amount):
            for status in ('attempted','completed'):ledger.append(dict(kind=kind,status=status))
    (failure/'QUALIFICATION_WORK.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    payload=dict(path=str(S.OLD_ROOT/'qualification/gsm8k/KPLUS/disposable_state.pt'),identity=dict(bytes=20,sha256='d'*64))
    write_json(failure/'DISPOSABLE_IDENTITY.json',payload)
    return old_sealed,failure,plan,payload


class RepairSealTests(unittest.TestCase):
    def test_accepts_only_exact_consumed_work_and_failure_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            _,failure,plan,_=fixture(Path(temporary))
            self.assertEqual(S.validate_failed_boundary(failure,plan)['adam_attempted'],1)
            with (failure/'QUALIFICATION_WORK.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(kind='adam',status='attempted'))+'\n')
            with self.assertRaises(ValueError):S.validate_failed_boundary(failure,plan)

    def test_canonical_packets_and_disposable_identity_are_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            sealed,failure,plan,payload=fixture(Path(temporary))
            preparation,reference,files=S.retained_dependencies(failure,plan,payload,sealed)
            self.assertEqual(len(preparation['datasets']['gsm8k']),2)
            self.assertEqual(reference['disposable_state'],payload)
            self.assertIn(payload['path'],files)
            row=preparation['datasets']['gsm8k'][0]
            local=failure/Path(row['record_path']).relative_to(S.OLD_ROOT)
            local.write_text('{}')
            with self.assertRaises(ValueError):S.retained_dependencies(failure,plan,payload,sealed)

    def test_no_new_feedback_or_live_ledger_reset_and_exact_old_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);old_sealed,failure,plan,_=fixture(root)
            code=root/'code';code.mkdir();(code/'tracked.py').write_text('pass\n')
            output=root/'qualification_002'
            args=SimpleNamespace(code=code,previous_sealed=old_sealed,failure=failure,output=output,
                commit='b'*40,disposable_identity=failure/'DISPOSABLE_IDENTITY.json')
            old_bytes={str(p):p.read_bytes() for p in failure.rglob('*') if p.is_file()}
            execution=dict(commit='b'*40,dirty=False,source_path=str(S.original.REMOTE_CODE))
            with patch.object(S.original,'published_source',return_value=execution),patch.object(
                    S.subprocess,'check_output',return_value=b'tracked.py\0'):
                result=S.seal(args)
            self.assertEqual(result['remaining_adam_steps'],7)
            self.assertEqual(result['new_feedback_generations'],0)
            self.assertFalse((output/'QUALIFICATION_WORK.jsonl').exists())
            self.assertEqual((output/'sealed/PRIOR_QUALIFICATION_WORK.jsonl').read_bytes(),(failure/'QUALIFICATION_WORK.jsonl').read_bytes())
            self.assertEqual((output/'qualification_feedback/PREPARATION.json').read_bytes(),(failure/'qualification_feedback/PREPARATION.json').read_bytes())
            current=json.loads((output/'sealed/PLAN.json').read_text())
            self.assertEqual(current['binding'],plan['binding'])
            self.assertEqual(current['numerical_recipe'],plan['numerical_recipe'])
            self.assertEqual(current['qualification_budget'],plan['qualification_budget'])
            self.assertEqual(current['execution_source']['source_path'],str(S.REMOTE_CODE))
            self.assertFalse(current['main_pilot_authorized'])
            sealed_files=json.loads((output/'sealed/SEAL.json').read_text())['files']
            self.assertIn(str(S.OLD_ROOT/'QUALIFICATION_WORK.jsonl'),sealed_files)
            self.assertNotIn(str(S.REMOTE_ROOT/'QUALIFICATION_WORK.jsonl'),sealed_files)
            for name,value in old_bytes.items():self.assertEqual(Path(name).read_bytes(),value)

    def test_altered_old_seal_cannot_be_repaired_away(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);old_sealed,failure,_,_=fixture(root)
            (old_sealed/'gsm8k/PANEL.json').write_text('changed')
            args=SimpleNamespace(code=root,previous_sealed=old_sealed,failure=failure,
                output=root/'new',commit='b'*40,disposable_identity=failure/'DISPOSABLE_IDENTITY.json')
            with patch.object(S.original,'published_source',return_value=dict(commit='b'*40)),self.assertRaises(ValueError):S.seal(args)
            self.assertFalse(args.output.exists())


if __name__=='__main__':unittest.main()
