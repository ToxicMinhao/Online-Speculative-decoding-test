"""CPU selection, immutable-stream and full-run contract checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from experiments.continuation_repair_fullrun import contract, data, configuration, prepare
from experiments.opencode_native_comparison.data import prepare_input, MATCH_FIELDS
from experiments.ok_coding.storage import identity


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '<user>'+messages[0]['content']+'</user><assistant>'
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def row(index, text=None, pool='evaluation'):
    return prepare_input(dict(pool=pool, input=text or f'input {index}', source_identity=dict(
        dataset='nvidia/OpenCodeInstruct', revision=contract.SOURCE_REVISION,
        split='train', raw_index=index, shard='train/partial-train/0000.parquet',
        shard_row_index=index)), Tokenizer())


def empty_history():
    return {field:set() for field in MATCH_FIELDS}


class PreparationTests(unittest.TestCase):
    def test_new_rank_is_literal_canonical_seed_role_source_identity(self):
        value=row(30000)
        payload=dict(seed=20261126,role='fullrun_confirm_evaluation_v1',source_identity=value['source_identity'])
        expected=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(data.rank(value,data.EVALUATION_ROLE)[0],expected)
        self.assertNotEqual(data.rank(value,data.EVALUATION_ROLE),data.rank(value,data.WITNESS_ROLE))

    def test_excludes_each_history_key_and_deduplicates_before_ranking(self):
        prepared=[row(30000+i) for i in range(10)]
        history=empty_history()
        for i,key in enumerate(MATCH_FIELDS):history[key].add(prepared[i][key])
        prepared.append(row(30100,'input 30004'))
        original={'train_requests':[], 'evaluation_requests':[prepared[5]], 'warmup_requests':[], 'qualification_requests':[]}
        panel,audit=data.choose_evaluation(prepared,history,original,count=4)
        selected={r['source_identity']['raw_index'] for r in panel['evaluation_requests']}
        self.assertTrue(selected.isdisjoint({30000,30001,30002,30003,30005}))
        self.assertFalse({30004,30100}<=selected)
        self.assertEqual(audit['exclusion_reasons']['known_project_exposure_or_reservation'],5)
        self.assertEqual(audit['exclusion_reasons']['duplicate_within_source_pool'],1)

    def test_selection_does_not_touch_training_order_and_ignores_source_iteration(self):
        original={'train_requests':[row(3,pool='train'),row(1,pool='train')],
                  'evaluation_requests':[], 'warmup_requests':[{'request_id':'warmup'}], 'qualification_requests':[]}
        unchanged=deepcopy(original)
        prepared=[row(30000+i) for i in range(6)]
        first,audit=data.choose_evaluation(prepared,empty_history(),original,count=3)
        second,_=data.choose_evaluation(list(reversed(prepared)),empty_history(),original,count=3)
        self.assertEqual(first,second)
        self.assertEqual(original,unchanged)
        self.assertFalse(audit['training_reselected'])
        self.assertEqual(first['warmup_requests'],original['warmup_requests'])

    def test_insufficient_fixed_pool_is_a_blocker(self):
        with self.assertRaisesRegex(ValueError,'requires 3'):
            data.choose_evaluation([row(30000)],empty_history(),{'warmup_requests':[]},count=3)

    def test_witness_is_hash_ranked_only_within_first_event(self):
        requests=[]
        for i in range(256):
            value=row(i,pool='train')
            value.update(request_id='r'+str(i),request_index=i)
            requests.append(value)
        witness=data.select_witness({'train_requests':requests})
        expected=sorted(requests[:128],key=lambda value:data.rank(value,data.WITNESS_ROLE))[:8]
        self.assertEqual([r['request_id'] for r in witness['responses']],[r['request_id'] for r in expected])
        self.assertEqual(witness['maximum_response_forwards'],72)
        self.assertEqual(witness['anchor_seed'],20261126)
        self.assertEqual(len({r['request_id'] for r in witness['responses']}),8)

    def test_pinned_original_bytes_cannot_be_replaced(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'INPUTS.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'identity differs'):
                data.original_inputs(path)

    def test_full_run_counts_and_own_honest_method_contract(self):
        self.assertEqual(contract.ARMS,('BASE','TAIL'))
        self.assertEqual(contract.PRODUCTION_WORK['adam'],58*128*2)
        self.assertEqual(contract.PRODUCTION_WORK['scored_generation'],1024*4*3)
        self.assertEqual(contract.PRODUCTION_WORK['warmup_generation'],48*2)
        self.assertEqual(contract.PRODUCTION_WORK['canonical_warmup_generation'],0)
        self.assertEqual(contract.OPTIONAL_WORK_CEILINGS['canonical_warmup_generation'],2)
        self.assertEqual(contract.RESOURCE_LIMITS['elapsed_s'],72*3600)
        base=contract.experiment_contract('BASE','1'*64)
        tail=contract.experiment_contract('TAIL','1'*64)
        self.assertEqual(base['initial_counters'],[0,0,0])
        self.assertEqual(base['final_counters'],[7424,14848,58])
        self.assertEqual(base['repair_pair'],tail['repair_pair'])
        self.assertNotEqual(base['continuation_repair'],tail['continuation_repair'])
        for name in ('parent_counters','parent_binding','event_id'):
            self.assertNotIn(name,tail)
        with self.assertRaises(ValueError):contract.experiment_contract('TAIL_DISABLED','1'*64)

    def test_native_recipe_allows_only_output_identity_and_fixed_context(self):
        original=yaml.safe_load((Path(__file__).parents[1]/'configs/osd.yaml').read_text())
        original['data']['max_length']=2272
        text=yaml.safe_dump(original)
        candidate=configuration.build_config(text,arm='TAIL',output_dir='/new/TAIL')
        actual=yaml.safe_load(candidate)
        self.assertEqual(actual['data']['max_length'],4096)
        self.assertEqual(actual['training'],original['training'])
        self.assertEqual(actual['online_feedback'],original['online_feedback'])
        actual['training']['learning_rate']=1e-4
        with self.assertRaises(ValueError):configuration.validate_config_text(yaml.safe_dump(actual),text)
        with self.assertRaises(ValueError):configuration.build_config(text,arm='AREPO',output_dir='/wrong')

    def test_retained_warmup_reference_identity_inputs_and_frozen_origin(self):
        requests=[];references={}
        for i in range(2):
            request=dict(request_id='w'+str(i),prompt_ids=[i+10],stop_token_ids=[151645,151643],max_new_tokens=2048)
            ref=dict(request,reference_kind='frozen_DFlash',verified=True,provenance={'runtime':'fixed'},
                source_checkpoint='/released',response_ids=[151645],finish_reason='eos')
            ref['binding_digest']=contract.digest(ref)
            requests.append(request);references[request['request_id']]=ref
        original={'warmup_requests':requests}
        self.assertEqual(prepare.validate_warmups(original,references,{'runtime':'fixed'},'/released')['new_generations'],0)
        for corruption in ('prompt_ids','binding_digest','source_checkpoint'):
            changed=deepcopy(references)
            changed['w0'][corruption]=[99] if corruption=='prompt_ids' else 'different'
            if corruption!='binding_digest':
                changed['w0']['binding_digest']=contract.digest({k:v for k,v in changed['w0'].items() if k!='binding_digest'})
            with self.assertRaises(ValueError):prepare.validate_warmups(original,changed,{'runtime':'fixed'},'/released')

    def test_panel_seal_rejects_changed_files_witness_and_training_order(self):
        original={'train_requests':[]}
        for i in range(128):
            value=row(i,pool='train');value.update(request_id='r'+str(i),request_index=i)
            original['train_requests'].append(value)
        with TemporaryDirectory() as folder:
            root=Path(folder)
            contents={'EVALUATION_INPUTS.json':{'evaluation_requests':[], 'warmup_requests':[]},
                'WITNESS_SELECTION.json':data.select_witness(original),
                'EXPOSURE.json':{'coverage':{'through_pr':65},'supplement':{'coverage':{'gaps':[]}}}}
            for name,value in contents.items():(root/name).write_text(json.dumps(value))
            seal=dict(status='sealed_before_model_work',original_inputs_identity=contract.INPUT_IDENTITY,
                original_training_order_digest=contract.digest(original['train_requests']),
                files={name:identity(root/name) for name in contents})
            (root/'DATA_SEAL.json').write_text(json.dumps(seal))
            # Request-format/count validation is independently covered by native tests;
            # this smaller fixture isolates authentication and fixed witness selection.
            with patch.object(prepare,'validate_inputs'):
                prepare.verify_panel(root,original)
                reversed_training={'train_requests':list(reversed(original['train_requests']))}
                with self.assertRaisesRegex(ValueError,'training stream'):prepare.verify_panel(root,reversed_training)
                (root/'EVALUATION_INPUTS.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'artifact changed'):prepare.verify_panel(root,original)
                (root/'EVALUATION_INPUTS.json').write_text(json.dumps(contents['EVALUATION_INPUTS.json']))
                contents['WITNESS_SELECTION.json']['anchor_seed']=18
                (root/'WITNESS_SELECTION.json').write_text(json.dumps(contents['WITNESS_SELECTION.json']))
                seal['files']['WITNESS_SELECTION.json']=identity(root/'WITNESS_SELECTION.json')
                (root/'DATA_SEAL.json').write_text(json.dumps(seal))
                with self.assertRaisesRegex(ValueError,'Witness selection'):prepare.verify_panel(root,original)

    def test_projection_rejects_extra_answer_and_wrong_reservoir(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'input.jsonl'
            value=dict(pool='evaluation',input='prompt',source_identity=row(30000)['source_identity'],answer='not allowed')
            path.write_text(json.dumps(value)+'\n')
            with self.assertRaisesRegex(ValueError,'input-only'):list(data.projected_rows(path,identity(path)))
            value.pop('answer');value['pool']='train';path.write_text(json.dumps(value)+'\n')
            with self.assertRaisesRegex(ValueError,'reservoirs'):list(data.projected_rows(path,identity(path)))

    def test_full_panel_rejects_old_tokens_even_with_a_new_identity(self):
        def requests(start,count):
            return [dict(request_id='request'+str(start+i),request_index=i,panel_ordinal=i,
                prompt_ids=[start+i],max_new_tokens=2048,stop_token_ids=[151645,151643]) for i in range(count)]
        original=dict(train_requests=requests(0,7424),evaluation_requests=requests(8000,1024),
            warmup_requests=requests(10000,2),qualification_requests=requests(11000,16))
        panel=dict(evaluation_requests=requests(12000,1024),warmup_requests=original['warmup_requests'])
        self.assertEqual(contract.validate_inputs(original,panel)['status'],'passed')
        panel['evaluation_requests'][0]['prompt_ids']=original['evaluation_requests'][0]['prompt_ids']
        with self.assertRaisesRegex(ValueError,'reused original'):contract.validate_inputs(original,panel)


if __name__=='__main__':unittest.main()
