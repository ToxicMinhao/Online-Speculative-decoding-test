from copy import deepcopy
import hashlib
import unittest

from experiments.survival_confirmation import data


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return '<user>'+messages[0]['content']+'</user><assistant>'

    def encode(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False)
        return list(text.encode())


def questions():
    return dict(test=[f'Question {i}?' for i in range(200)], train=['Training question?'])


class ConfirmationDataTests(unittest.TestCase):
    def test_opaque_request_index_is_not_official_row(self):
        mapped=data.map_study_rows([dict(request_id='dev-99', request_index=99,
            original_index=77, question='Question 4?')],split='test',questions=questions(),label='dev')
        self.assertEqual(mapped[0]['source_row'],4)
        self.assertEqual(mapped[0]['source_id'],'gsm8k:main:test:4')

    def test_explicit_wrong_identity_and_ambiguous_source_fail(self):
        for field,value in [('source_row',99),('source_id','gsm8k:main:test:99')]:
            row=dict(request_id='x',question='Question 4?');row[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                data.map_study_rows([row],split='test',questions=questions(),label='dev')
        source=questions();source['test'][5]=source['test'][4]
        with self.assertRaises(ValueError):
            data.map_study_rows([dict(request_id='x',question='Question 4?')],split='test',questions=source,label='dev')

    def test_cross_split_normalization_reservation_and_deduplication(self):
        source=questions();source['test'][0]='  Training\n question?  '
        source['test'][1]='Question ２?'
        exclusions=data.map_study_rows([dict(request_id='train',question='Training question?')],
            split='train',questions=source,label='feedback')
        selected,audit=data.select_rows(source,exclusions,count=136,reserved_test_rows=[3])
        self.assertEqual(audit['eligible_count'],197)
        self.assertFalse({0,2,3}&set(selected))
        self.assertEqual({r['source_row'] for r in audit['exclusions']},{0,2,3})

    def test_exact_hash_order_and_count_boundaries(self):
        selected,audit=data.select_rows(questions(),[],count=136)
        expected=sorted(range(200),key=lambda i:(hashlib.sha256(
            ('survival-confirm-v1|gsm8k|test|'+str(i)).encode()).hexdigest(),i))[:136]
        self.assertEqual(selected,expected)
        for count in (128,137,776,True,136.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                data.select_rows(questions(),[],count=count)
        with self.assertRaises(ValueError):data.select_rows(questions(),[],count=208)
        with self.assertRaises(ValueError):data.select_rows(questions(),[],count=136,reserved_test_rows=[200])

    def test_exposure_annotation_does_not_change_selected_ids_or_tokens(self):
        source=questions();selected,audit=data.select_rows(source,[],count=136)
        a=data.build_panel(source,selected,audit,FakeTokenizer(),{},exclusion_rows=[])
        b=data.build_panel(source,selected,audit,FakeTokenizer(),
            {selected[0]:[dict(source='prior',evidence='receipt')]},exclusion_rows=[])
        self.assertEqual([(r['source_id'],r['prompt_ids']) for r in a['requests']],
                         [(r['source_id'],r['prompt_ids']) for r in b['requests']])
        self.assertEqual(b['prior_project_execution_counts'],dict(documented=1,unknown=135))
        self.assertEqual(data.verify_panel(b)['status'],'passed')

    def test_rendered_or_token_overlap_fails_without_replacing(self):
        source=questions();selected,audit=data.select_rows(source,[],count=136)
        ids,text=data.render_question(source['test'][selected[0]],FakeTokenizer())
        for overlap in (dict(prompt_ids_sha256=data.digest(ids),rendered_prompt_sha256=None),
                dict(prompt_ids_sha256=None,rendered_prompt_sha256=hashlib.sha256(text.encode()).hexdigest())):
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                data.build_panel(source,selected,audit,FakeTokenizer(),{},exclusion_rows=[overlap])

    def test_context_failure_and_tamper_are_not_silent_replacements(self):
        source=questions();selected,audit=data.select_rows(source,[],count=136)
        panel=data.build_panel(source,selected,audit,FakeTokenizer(),{},exclusion_rows=[])
        changed=deepcopy(panel);changed['requests'][0]['prompt_ids'][0]+=1
        with self.assertRaises(ValueError):data.verify_panel(changed)
        changed=deepcopy(panel);changed['selection']['reference_availability_used']=True
        changed['binding']=data.digest({k:v for k,v in changed.items() if k!='binding'})
        with self.assertRaises(ValueError):data.verify_panel(changed)
        source['test'][selected[0]]='x'*3000
        with self.assertRaises(ValueError):
            data.build_panel(source,selected,audit,FakeTokenizer(),{},exclusion_rows=[])


if __name__=='__main__':unittest.main()
