"""Owned-pool routing, split canonical preparation, and direct-entry gates."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from online_feedback.storage import write_json
from experiments.c_paired_confirmation import task_runner as R
from test_full_k_gsm8k_evaluation import FullBackend
from test_evaluation import panel


class Ledger:
    def __init__(self): self.counts = Counter()
    def call(self, counts, label, function, *args, **kwargs):
        self.counts.update(counts)
        return function(*args, **kwargs)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan = dict(root=str(self.root),binding='binding',execution_commit='c'*40,
            devices={'A':'GPU-A','B':'GPU-B'},datasets={})
        for dataset, first in [('gsm8k',7),('mathqa',5)]:
            base = self.root/dataset; base.mkdir()
            sub = dict(first_event=first,second_event=first+1,origin_export='/origin/'+dataset,
                origin_weights_identity={'sha256':'origin','bytes':4},
                references=str(base/'CANONICALS.json'),warmup_references=str(base/'WARMUP_CANONICALS.json'),
                data_panel=str(base/'PANEL.json'),warmups=str(base/'WARMUPS.json'),next_inputs=str(base/'NEXT.json'))
            rows = panel(dataset)
            for i,row in enumerate(rows): row['request_index'] = i
            warm = deepcopy(rows[:2])
            for i,row in enumerate(warm): row.update(request_id=f'warm{i}',source_id=f'{dataset}:warm:{i}',prompt_ids=[1,9000+i])
            write_json(sub['data_panel'],{'requests':rows}); write_json(sub['warmups'],{'requests':warm})
            self.plan['datasets'][dataset] = sub

    def save_endpoint(self,dataset,arm,event):
        sub = self.plan['datasets'][dataset]
        suffix = 'first' if event==sub['first_event'] else 'second'
        base = R.task_path(self.plan,dataset,arm+'_'+suffix); base.mkdir(parents=True)
        state = base/'state.pt'; state.write_bytes((dataset+arm+str(event)).encode())
        export = base/'export'; export.mkdir(); (export/'model.safetensors').write_bytes(b'weights')
        saved = dict(state=str(state),export=str(export),state_identity=R.identity(state),
            weights_identity=R.identity(export/'model.safetensors'),global_step=event*128,presentations=event*256)
        write_json(base/'RESULT.json',dict(status='completed',dataset=dataset,arm=arm,event=event,
            binding=self.plan['binding'],execution_commit=self.plan['execution_commit'],endpoint=saved))
        return saved

    def test_current_own_endpoint_required_and_immutable(self):
        state = self.save_endpoint('gsm8k','N',7)
        self.assertEqual(R.endpoint(self.plan,'gsm8k','N',7),state)
        with self.assertRaises(FileNotFoundError): R.endpoint(self.plan,'gsm8k','C',7)
        Path(state['state']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'): R.endpoint(self.plan,'gsm8k','N',7)
        with self.assertRaisesRegex(ValueError,'Unknown'): R.endpoint(self.plan,'mathqa','N',7)

    def test_paired_pools_keep_own_producer_and_current_file_identity(self):
        dataset='mathqa';sub=self.plan['datasets'][dataset]
        producer=self.save_endpoint(dataset,'C',5)
        base=R.task_path(self.plan,dataset,'C_collect');(base/'loop/features').mkdir(parents=True)
        rows=[dict(request_id=f'r{i}') for i in range(128)]
        write_json(sub['next_inputs'],{'requests':rows})
        paths=[]; bindings={}
        for row in rows:
            path=base/'loop/features'/(row['request_id']+'.pt');path.write_bytes(row['request_id'].encode())
            paths.append(str(path));bindings[str(path)]={'file_identity':R.identity(path)}
        pool=dict(status='completed',dataset=dataset,arm='C',event=6,binding='binding',
            execution_commit='c'*40,own_checkpoint=producer['export'],own_weights_identity=producer['weights_identity'],
            feature_paths=paths,generated_responses=128,teacher_preparations=128,feature_bindings=bindings)
        write_json(base/'FEATURES.json',pool)
        self.assertEqual(R.feature_pool(self.plan,dataset,'C',6),paths)
        pool['arm']='N';write_json(base/'FEATURES.json',pool)
        with self.assertRaisesRegex(ValueError,'own-arm'):R.feature_pool(self.plan,dataset,'C',6)
        pool['arm']='C';write_json(base/'FEATURES.json',pool)
        Path(paths[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):R.feature_pool(self.plan,dataset,'C',6)

    def test_packet_binding_all_tensors_and_response_only_producer(self):
        path=self.root/'packet.pt'
        packet=dict(record_id='r',input_ids=torch.tensor([[1,2,3]]),loss_mask=torch.tensor([[0.,1.,1.]]),
            prompt_length=1,hidden_states=torch.ones(1,3,2,dtype=torch.bfloat16),
            target_last_hidden_states=torch.zeros(1,3,2,dtype=torch.bfloat16))
        torch.save(packet,path)
        record=dict(record_id='r',prompt_ids=[1],response_ids=[2,3],checkpoint_id='/C')
        producer=dict(export='/C',weights_identity={'sha256':'w','bytes':4})
        bound=R.packet_binding(path,record,producer)
        self.assertEqual(set(bound['tensors']),{'input_ids','loss_mask','hidden_states','target_last_hidden_states'})
        self.assertEqual(bound['producer_weights_identity'],producer['weights_identity'])
        packet['loss_mask'][0,0]=1;torch.save(packet,path)
        with self.assertRaisesRegex(ValueError,'response-only'):R.packet_binding(path,record,producer)

    def test_two_warmup_then_256_scored_references_with_immutable_seals(self):
        sub=self.plan['datasets']['gsm8k']; ledger=Ledger()
        allrows=R.read(sub['data_panel'])['requests'];backend=FullBackend(allrows)
        backend.checkpoint_id=sub['origin_export'];backend.loading_s=.1
        with patch.object(R,'create_backend',return_value=backend),patch.object(R,'close_backend'),patch.object(R,'boundary'):
            warm=R.task_path(self.plan,'gsm8k','warmup_references');warm.mkdir(parents=True)
            result=R.references_task(self.plan,'gsm8k','warmup_references',warm,ledger,'GPU-B')
            self.assertEqual(result['reference_count'],2)
            self.assertEqual((result['binding'],result['execution_commit']),('binding','c'*40))
            original=Path(sub['warmup_references']).read_bytes()
            base=R.task_path(self.plan,'gsm8k','references');base.mkdir()
            result=R.references_task(self.plan,'gsm8k','references',base,ledger,'GPU-B')
            self.assertEqual(result['reference_count'],256)
            self.assertEqual(Path(sub['warmup_references']).read_bytes(),original)
        self.assertEqual(ledger.counts,{'canonical_generation':258})
        self.assertEqual(R.verify_reference_seal(self.plan,'gsm8k')['reference_count'],258)
        Path(sub['warmup_references']).write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'changed'):R.verify_reference_seal(self.plan,'gsm8k')

    def test_production_requires_current_gates_and_finished_both_chains(self):
        write_json(self.root/'LAUNCH.json',{'status':'qualification'})
        R.require_production(self.plan,'gsm8k','warmup_references')
        with self.assertRaises(FileNotFoundError):R.require_production(self.plan,'gsm8k','N_first')
        for name in ('QUALIFICATION.json','RESOURCE_ADMISSION.json'):
            write_json(self.root/name,dict(status='passed',binding='binding',execution_commit='c'*40))
        R.require_production(self.plan,'gsm8k','N_first')
        with self.assertRaises(FileNotFoundError):R.require_production(self.plan,'gsm8k','final')
        for dataset in R.DATASETS:
            for task in ('N_first','C_first','N_collect','C_collect','N_second','C_second','references'):
                write_json(R.task_path(self.plan,dataset,task)/'STATUS.json',{'status':'completed'})
        R.require_production(self.plan,'gsm8k','final')
        with self.assertRaises(FileNotFoundError):R.require_production(self.plan,'mathqa','final')
        write_json(R.task_path(self.plan,'gsm8k','final')/'STATUS.json',{'status':'completed'})
        R.require_production(self.plan,'mathqa','final')
        write_json(self.root/'PRODUCTION_STARTED.json',{})
        with self.assertRaisesRegex(ValueError,'qualification'):R.require_production(self.plan,'gsm8k','warmup_references')

    def test_resume_reuses_old_warmup_identity_and_requires_adopted_gate(self):
        sub=self.plan['datasets']['gsm8k']
        write_json(sub['warmup_references'],{'retained':'reference'})
        old=R.reference_seal(self.plan,'gsm8k',warmups_only=True)
        write_json(Path(sub['references']).parent/'WARMUP_SEAL.json',old)
        self.plan.update(original_execution_commit='c'*40,execution_commit='d'*40,
            adopted_qualification_path=str(self.root/'resume_001/QUALIFICATION.json'))
        self.assertEqual(R.verify_reference_seal(self.plan,'gsm8k',warmups_only=True),old)
        gate=dict(status='passed',binding='binding',execution_commit='d'*40)
        write_json(self.root/'RESOURCE_ADMISSION.json',gate)
        write_json(self.root/'QUALIFICATION.json',dict(gate,execution_commit='c'*40))
        with self.assertRaises(FileNotFoundError):R.require_production(self.plan,'gsm8k','N_first')
        write_json(self.plan['adopted_qualification_path'],gate)
        R.require_production(self.plan,'gsm8k','N_first')
        write_json(sub['references'],{})
        self.assertEqual(R.reference_seal(self.plan,'gsm8k')['execution_commit'],'d'*40)
        Path(sub['warmup_references']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            R.verify_reference_seal(self.plan,'gsm8k',warmups_only=True)


if __name__=='__main__':unittest.main()
