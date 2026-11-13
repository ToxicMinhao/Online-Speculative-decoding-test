"""CPU-only evaluation continuation never repeats completed training/generation."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.verification_transition.data import identity, read
from experiments.recent_repair_retention import resume_evaluation as R


def fixture(root):
    old=root/'old';old.mkdir()
    source=root/'old_source';source.mkdir()
    repaired=root/'new_source';repaired.mkdir()
    name='experiments/recent_repair_retention/fixed_starts.py'
    for folder,content in ((source,'old'),(repaired,'fixed publication allowance')):
        path=folder/name;path.parent.mkdir(parents=True);path.write_text(content)
    for folder,commit in ((source,'old-commit'),(repaired,'repair-commit')):
        write_json(folder/'SOURCE.json',dict(commit=commit,dirty=False,symlinks={},files={name:identity(folder/name)}))
    panel=[dict(request_id='r0',prompt_ids=[1],panel_ordinal=0)]
    write_json(old/'PANEL.json',dict(requests=panel))
    canonical=dict(r0=dict(prompt_ids=[1],response_ids=[2],verified=True,finish_reason='eos'))
    write_json(old/'REUSED.json',canonical)
    write_json(old/'canonicals/CANONICAL.json',canonical)
    write_json(old/'canonicals/STATUS.json',dict(status='completed'))
    write_json(old/'STATUS.json',dict(status='failed',phase='fixed_start_diagnostic',wall_s=100.))
    write_json(old/'fixed_starts/STATUS.json',dict(status='failed',phase='pre_endpoint',parent_requests=0,
        endpoint_blocks=0,error='model_load allowance exhausted',work=dict(gpu_seconds=20.)))
    write_json(old/'fixed_starts/STARTS.json',[]);write_json(old/'fixed_starts/OUTCOMES.json',[])
    ledger=[dict(kind='model_load',status='attempted',ticket=1,metadata={}),
            dict(kind='draft_block',status='attempted',ticket=2,metadata=dict(startup=True)),
            dict(kind='draft_block',status='completed',ticket=2),
            dict(kind='model_load',status='completed',ticket=1),
            dict(kind='parent_walk',status='attempted',ticket=3,metadata=dict(request_id='r0')),
            dict(kind='parent_walk',status='failed',ticket=3)]
    (old/'fixed_starts/WORK.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    for case in ('NATIVE','DISABLED','ZERO','ACTIVE'):
        write_json(old/'qualification'/case/'RESULT.json',dict(status='passed',
            **{key:dict(exact_equal=True) for key in ('exact_native','exact_selections','active_restored','long_restored')}))
    exports={}
    for arm in R.runtime.METHODS:
        folder=old/'training'/arm;folder.mkdir(parents=True)
        export=folder/'export';export.mkdir();(export/'model.safetensors').write_bytes(arm.encode())
        state=folder/'state.pt';state.write_bytes(b'unchanged recovery')
        exports[arm]=str(export)
        write_json(folder/'STATUS.json',dict(status='completed',global_step=896,presentations=1792))
        write_json(folder/'RESULT.json',dict(status='completed',start_step=768,end_step=896,optimizer_steps=128,
            records=[{}]*256,checkpoint_path=str(export),state_path=str(state),boundary=dict(export_exact=True),
            first_firing=dict(status='not_observed'),weights_identity=identity(export/'model.safetensors'),state_identity=identity(state)))
    write_json(old/'EXPORTS.json',exports);write_json(old/'ENDPOINT_DIFFERENCES.json',{})
    write_json(old/'PRETRAINING_SEAL.json',dict(preserved=True));write_json(old/'ASSIGNMENTS.json',{})
    (old/'reference').mkdir()
    plan=dict(root=str(old),code_path=str(source),source_manifest=str(source/'SOURCE.json'),execution_commit='old-commit',
        fixed_start_seconds_limit=14400.,campaign_timeout_s=21600.,qualification_root=str(old/'qualification'),
        path_request_ids=['r0'],panel_size=1,panel_manifest=str(old/'PANEL.json'),reused_references=str(old/'REUSED.json'),
        checkpoints=dict(K6=dict(export='parent')),objective='unchanged K+',binding='sealed binding')
    write_json(old/'TRAINING_PLAN.json',plan)
    return old,repaired,plan


class ResumeTests(unittest.TestCase):
    def prepare(self,root):
        old,source,plan=fixture(root)
        with patch.object(R.runtime,'verify_execution_plan'),patch.object(R.runtime,'verify_pretraining'):
            value=R.prepare(old/'TRAINING_PLAN.json',root/'new',source,source/'SOURCE.json')
        return old,source,plan,value

    def test_preparation_links_existing_outputs_and_records_distinct_commits(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);old,source,plan,value=self.prepare(root)
            self.assertEqual(value['new_optimizer_steps'],0)
            self.assertEqual(value['pending_parent_generations'],1)
            self.assertEqual(value['training_execution_commit'],'old-commit')
            self.assertEqual(value['evaluation_execution_commit'],'repair-commit')
            self.assertEqual(value['fixed_start_remaining_s'],14380.)
            self.assertEqual(value['campaign_remaining_s'],21500.)
            self.assertTrue((root/'new/training').is_symlink())
            self.assertEqual(read(old/'TRAINING_PLAN.json'),plan)
            with patch.object(R.runtime,'verify_execution_plan'),patch.object(R.runtime,'verify_pretraining'):
                R.verify(root/'new/CONTINUATION_PLAN.json')
            self.assertEqual(read(old/'STATUS.json')['status'],'failed')

    def test_existing_parent_answer_or_timing_forbids_repeating(self):
        with TemporaryDirectory() as tmp:
            old,source,plan=fixture(Path(tmp))
            write_json(old/'fixed_starts/requests/r0/K5_CAPTURE.json',dict(response_ids=[2]))
            with self.assertRaisesRegex(ValueError,'Existing parent generation'):
                R.prior_failure(old,plan)

    def test_nonstartup_forward_is_not_assumed_unexecuted(self):
        with TemporaryDirectory() as tmp:
            old,source,plan=fixture(Path(tmp))
            file=old/'fixed_starts/WORK.jsonl'
            file.write_text(file.read_text().replace('"startup": true','"startup": false'))
            with self.assertRaisesRegex(ValueError,'Non-startup'):
                R.prior_failure(old,plan)

    def test_source_allowlist_rejects_objective_change(self):
        original=dict(files={'online_feedback/objective.py':1},symlinks={})
        with self.assertRaisesRegex(ValueError,'outside'):
            R.source_changes(original,dict(files={'online_feedback/objective.py':2},symlinks={}))

    def test_link_target_or_replacement_cannot_substitute_other_evidence(self):
        for name,replacement in (('EXPORTS.json','regular'),('canonicals','other_symlink')):
            with self.subTest(name=name),TemporaryDirectory() as tmp:
                root=Path(tmp);old,source,plan,value=self.prepare(root)
                link=root/'new'/name
                link.unlink()
                if replacement=='regular':
                    link.write_bytes((old/name).read_bytes())
                else:
                    other=root/'other_canonicals';other.mkdir()
                    link.symlink_to(other,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'evidence link changed'):
                    R.verify(root/'new/CONTINUATION_PLAN.json')

    def test_each_execution_commit_mismatch_is_rejected_independently(self):
        for changed in ('active_plan','provenance','patched_manifest'):
            with self.subTest(changed=changed),TemporaryDirectory() as tmp:
                root=Path(tmp);old,source,plan,value=self.prepare(root)
                wrapper_path=root/'new/CONTINUATION_PLAN.json'
                wrapper=read(wrapper_path)
                if changed=='patched_manifest':
                    path=source/'SOURCE.json';data=read(path);data['commit']='wrong';write_json(path,data)
                else:
                    path=Path(wrapper[changed]);data=read(path)
                    data['execution_commit' if changed=='active_plan' else 'evaluation_execution_commit']='wrong'
                    write_json(path,data);wrapper[changed+'_identity']=identity(path);write_json(wrapper_path,wrapper)
                with patch.object(R.runtime,'verify_execution_plan'),patch.object(R.runtime,'verify_pretraining'):
                    with self.assertRaisesRegex(ValueError,'source identities differ'):
                        R.verify(wrapper_path)

    def test_changed_weights_or_seals_and_nonempty_destination_fail(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);old,source,plan,value=self.prepare(root)
            with self.assertRaises(FileExistsError):
                R.prepare(old/'TRAINING_PLAN.json',root/'new',source,source/'SOURCE.json')
            with patch.object(R.runtime,'verify_execution_plan',side_effect=ValueError('old seal changed')):
                with self.assertRaisesRegex(ValueError,'old seal changed'):
                    R.verify(root/'new/CONTINUATION_PLAN.json')
            (old/'training/NATIVE/export/model.safetensors').write_bytes(b'changed')
            with patch.object(R.runtime,'verify_execution_plan'),patch.object(R.runtime,'verify_pretraining'):
                with self.assertRaisesRegex(ValueError,'weights/recovery changed'):
                    R.verify(root/'new/CONTINUATION_PLAN.json')

    def test_runner_only_launches_fixed_starts_and_serving(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);old,source,oldplan,value=self.prepare(root)
            plan=read(root/'new/TRAINING_PLAN.json')
            calls=[]
            with patch.object(R,'verify',return_value=(oldplan,plan,value)),patch.object(R,'preflight',return_value={}), \
                 patch.object(R.runtime,'launch',side_effect=lambda *args,**kwargs:calls.append(args[1])), \
                 patch.object(R.runtime,'serve',side_effect=lambda *args:calls.append('serve')), \
                 patch.object(R.runtime,'prepare_canonicals',side_effect=AssertionError('no canonical generation')):
                result=R.run(root/'new/CONTINUATION_PLAN.json')
            self.assertEqual(calls,['experiments.recent_repair_retention.fixed_starts','serve'])
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['new_optimizer_steps'],0)
            self.assertGreaterEqual(result['combined_campaign_wall_s'],100.)
            with self.assertRaises(FileExistsError),patch.object(R,'verify',return_value=(oldplan,plan,value)):
                R.run(root/'new/CONTINUATION_PLAN.json')


if __name__=='__main__':unittest.main()
