"""Exercise complete-reference assembly through the actual production loader."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from experiments.bounded_prefix_allocation.prepare import (
    _digest, _write, identity, load_inputs, reference_readiness,
    build_archive, unpack_archive, stage_complete_references, storage_plan, endpoint_from_result,
)


def fixture(root):
    remote = "/data/sources/minhao/runs/test_bounded/execution_001"
    (root / "sealed").mkdir(parents=True)
    (root / "references").mkdir()
    plan = dict(root=remote, references=remote+"/references/CANONICALS.json", input_identities={})
    def request(i, warmup=False):
        return dict(request_id=("warm" if warmup else "panel")+str(i), source_id=("warm" if warmup else "test")+str(i),
            original_index=i, panel_ordinal=i, prompt_ids=[100+i], prompt_ids_sha256=_digest([100+i]),
            max_new_tokens=2048, stop_token_ids=[151645,151643], collect_feedback=False)
    requests = [request(i) for i in range(128)]
    warmups = [request(i,True) for i in range(2)]
    def canonicals(rows):
        result={}
        for row in rows:
            entry=dict(prompt_ids=row['prompt_ids'],response_ids=[50,151645],finish_reason='eos',verified=True,
                source_id=row['source_id'],reference_kind='native_greedy_target_verifier',source_record_path='/retained/'+row['request_id'])
            entry['binding_digest']=_digest(entry)
            result[row['request_id']]=entry
        return result
    feedback=[]
    for i in range(128):
        row=request(i);row.update(request_id='train_'+str(896+i),source_id='train'+str(i),collect_feedback=True,
            request_index=897+i,native_stream_index=896+i)
        feedback.append(row)
    values=dict(data_panel=dict(requests=requests,within_block_permutations=[
        [list(range(32*b,32*(b+1))) for b in range(4)] for _ in range(3)]),
        references=canonicals(requests),warmups=dict(requests=warmups),warmup_references=canonicals(warmups),
        event8_inputs=dict(requests=feedback))
    for key,value in values.items():
        relative='references/CANONICALS.json' if key=='references' else 'sealed/'+key+'.json'
        plan[key]=remote+'/'+relative
        path=root.parent/'immutable_full_references.json' if key=='references' else root/relative
        _write(path,value);plan['input_identities'][plan[key]]=identity(path)
    stage_complete_references(plan,root.parent/'immutable_full_references.json',root)
    _write(root/'sealed/PLAN.json',plan)
    return plan


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name)
        self.package=self.base/'package'
        self.plan=fixture(self.package)

    def test_real_assembly_unpack_and_consumer_before_backend(self):
        archive=self.base/'package.tar.gz'
        build_archive(self.plan,self.package,archive)
        result=unpack_archive(archive,self.base/'unpacked')
        # No manual seeding here: the archive contains all128 actual references.
        values=load_inputs(result['plan'],package_root=self.base/'unpacked')
        backend=Mock()
        backend(values['data_panel']['requests'],values['references'])
        backend.assert_called_once()
        self.assertEqual(result['readiness']['complete_references'],128)
        self.assertEqual(result['readiness']['new_reference_generations'],0)

    def test_seed_source_and_existing_destination_are_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            stage_complete_references(self.plan,self.base/"immutable_full_references.json",self.package)
        source=self.base/"immutable_full_references.json"
        source.write_text("{}")
        with self.assertRaises(ValueError):
            stage_complete_references(self.plan,source,self.package)

    def test_missing_payload_stops_before_backend(self):
        (self.package/'references/CANONICALS.json').unlink()
        backend=Mock()
        with self.assertRaises(FileNotFoundError):
            values=load_inputs(self.plan,package_root=self.package)
            backend(values)
        backend.assert_not_called()

    def change(self,key,transform):
        path=self.package/Path(self.plan[key]).relative_to(self.plan['root'])
        value=json.loads(path.read_text());transform(value);_write(path,value)
        self.plan['input_identities'][self.plan[key]]=identity(path)
        _write(self.package/'sealed/PLAN.json',self.plan)

    def test_incomplete_or_extra_references_rejected(self):
        self.change('references',lambda refs:refs.pop('panel127'))
        with self.assertRaises(ValueError):reference_readiness(self.plan,package_root=self.package)

    def test_wrong_canonical_source_or_digest_rejected(self):
        self.change('references',lambda refs:refs['panel0'].update(source_id='other'))
        with self.assertRaises(ValueError):load_inputs(self.plan,package_root=self.package)

    def test_exact_prompt_binding_rejected(self):
        self.change('data_panel',lambda panel:panel['requests'][0].update(prompt_ids=[777]))
        with self.assertRaises(ValueError):load_inputs(self.plan,package_root=self.package)

    def test_feedback_preserves_original_global_indices(self):
        self.assertEqual(load_inputs(self.plan,package_root=self.package)['event8_inputs']['requests'][0]['request_index'],897)
        self.change('event8_inputs',lambda panel:panel['requests'][0].update(request_index=0))
        with self.assertRaises(ValueError):load_inputs(self.plan,package_root=self.package)

    def test_invalid_termination_rejected(self):
        def mutate(refs):
            entry=refs['panel0'];entry.update(response_ids=[151645,40]);entry['binding_digest']=_digest({k:v for k,v in entry.items() if k!='binding_digest'})
        self.change('references',mutate)
        with self.assertRaises(ValueError):load_inputs(self.plan,package_root=self.package)

    def test_destination_and_overwrite_fail(self):
        wrong=copy.deepcopy(self.plan);wrong['references']=wrong['root']+'/elsewhere.json'
        with self.assertRaises(ValueError):reference_readiness(wrong,package_root=self.package)
        archive=self.base/'package.tar.gz';build_archive(self.plan,self.package,archive)
        with self.assertRaises(FileExistsError):build_archive(self.plan,self.package,archive)
        with self.assertRaises(FileExistsError):unpack_archive(archive,self.package)

    def test_measured_size_budget_accounts_both_states_and_atomic_overlap(self):
        budget=storage_plan([100]*128,weight_bytes=2_097_259_104,recovery_bytes=6_300_162_779,hidden_size=4096,captured_layers=5)
        self.assertEqual(budget['checkpoint_bytes'],2*(2_097_259_104+6_300_162_779))
        self.assertEqual(budget['atomic_publication_bytes'],2_097_259_104+6_300_162_779)
        self.assertEqual(budget['qualification_payload_bytes'],budget['atomic_publication_bytes'])
        self.assertEqual(budget['prepared_feature_bytes'],128*2148*(4096*6*2+12))
        self.assertEqual(budget['allocation_cap_bytes'],80_000_000_000)
        self.assertEqual(budget['required_initial_free_bytes'],budget['peak_new_bytes']+400_000_000_000)
        self.assertEqual(budget['status'],'passed')

    def test_failed_or_partial_control_rejected(self):
        with self.assertRaises(ValueError):endpoint_from_result(dict(status='failed',event=8,arm='S'))
        result=dict(status='completed',event=8,arm='S',endpoint=dict(global_step=1024,presentations=2048),
            boundary=dict(status='passed',export_exact=True,live=dict(exact_resume=True)))
        self.assertEqual(endpoint_from_result(result)['global_step'],1024)
        result['endpoint']['global_step']=919
        with self.assertRaises(ValueError):endpoint_from_result(result)


if __name__=='__main__':unittest.main()
