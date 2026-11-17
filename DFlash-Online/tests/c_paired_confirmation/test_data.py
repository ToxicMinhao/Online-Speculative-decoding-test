from copy import deepcopy
from collections import Counter
import hashlib
import unittest
from experiments.c_paired_confirmation import data as D


def candidates(dataset, count=270):
    result=[]
    for i in range(count):
        question=f'Question {i}?';options='A, B' if dataset=='mathqa' else '';rendered=f'user:{question}\n{options}'
        row=dict(source_id=D.DATASETS[dataset]['source_prefix']+str(i),source_row=i,
                 question=question,options=options,rendered_prompt=rendered,prompt_ids=[1,100+i],user_message=question+'\n'+options)
        row.update(D.fingerprints(question,options,rendered,row['prompt_ids']));result.append(row)
    return result


def adaptation(dataset):
    start=D.DATASETS[dataset]['first_ordinal'];prefix='gsm8k:main:train:' if dataset=='gsm8k' else 'mathqa:train:'
    rows=[dict(source_id=prefix+str(i),request_id=f'train_{i:05d}',request_index=i+1,prompt_ids=[2,20000+i],
               prompt_ids_sha256=D.token_digest([2,20000+i]),max_new_tokens=2048,stop_token_ids=[151645,151643],collect_feedback=True)
          for i in range(start,start+256)]
    return dict(start_ordinal=start,stop_ordinal=start+256,train_requests=rows)


class DataTests(unittest.TestCase):
    def test_exact_rank_literal_includes_dataset(self):
        expected=hashlib.sha256(b'kplus_c_pair_confirm_r29_20261117_v1|gsm8k|gsm8k:main:test:7|abc').hexdigest()
        self.assertEqual(D.rank_key('gsm8k','gsm8k:main:test:7','abc'),expected)
        self.assertNotEqual(D.rank_key('mathqa','same','abc'),D.rank_key('gsm8k','same','abc'))

    def test_normalization_preserves_case_punctuation_and_option_order(self):
        self.assertEqual(D.normalize('  Ａ\n B '),'A B')
        self.assertNotEqual(D.normalize('A, B'),D.normalize('a, B'))
        self.assertNotEqual(D.normalize('A, B'),D.normalize('B, A'))

    def test_seal_first256_next2_without_mutating_inputs(self):
        rows=candidates('gsm8k');before=deepcopy(rows);roles=D.seal_requests('gsm8k',rows)
        self.assertEqual(rows,before)
        self.assertEqual([r['source_id'] for r in roles['scored']+roles['warmup']],[r['source_id'] for r in D.rank_eligible('gsm8k',rows)[:258]])
        self.assertEqual([r['panel_ordinal'] for r in roles['warmup']],[0,1])

    def test_outcomes_do_not_change_selection(self):
        rows=candidates('mathqa');first=D.seal_requests('mathqa',rows)
        for i,row in enumerate(rows):row.update(verification_calls=1000-i,response_ids=[i],latency=i)
        changed=D.seal_requests('mathqa',rows)
        self.assertEqual([r['source_id'] for r in first['scored']],[r['source_id'] for r in changed['scored']])

    def test_shortfall_has_no_smaller_panel_fallback(self):
        with self.assertRaisesRegex(ValueError,'Fewer than 258'):D.seal_requests('gsm8k',candidates('gsm8k',257))

    def test_both_datasets_validate_original_adaptation_ranges(self):
        for dataset in ('gsm8k','mathqa'):
            roles=D.seal_requests(dataset,candidates(dataset));a=adaptation(dataset)
            self.assertEqual(D.validate_manifests(dataset,{'requests':roles['scored']},{'requests':roles['warmup']},a)['status'],'passed')
            a['train_requests'][0]['request_index']-=1
            with self.assertRaisesRegex(ValueError,'input contract'):D.validate_manifests(dataset,{'requests':roles['scored']},{'requests':roles['warmup']},a)

    def test_prompt_tampering_is_rejected(self):
        roles=D.seal_requests('mathqa',candidates('mathqa'));roles['scored'][0]['prompt_ids'].append(77)
        with self.assertRaisesRegex(ValueError,'fingerprint'):D.validate_manifests('mathqa',{'requests':roles['scored']},{'requests':roles['warmup']},adaptation('mathqa'))

    def test_known_adaptation_token_identity_cannot_enter_panel(self):
        roles=D.seal_requests('gsm8k',candidates('gsm8k'));a=adaptation('gsm8k');a['train_requests'][0]['prompt_ids_sha256']=roles['scored'][0]['prompt_ids_sha256']
        with self.assertRaisesRegex(ValueError,'overlaps'):D.validate_manifests('gsm8k',{'requests':roles['scored']},{'requests':roles['warmup']},a)

    def test_schedule_balanced_and_preserves_sealed_block_order(self):
        rows=D.seal_requests('gsm8k',candidates('gsm8k'))['scored'];schedule=D.evaluation_schedule(rows)
        self.assertEqual(len(schedule),24);self.assertEqual(sum(len(r['request_ordinals']) for r in schedule),1536)
        self.assertEqual(Counter(r['method'] for r in schedule if r['position']==0),{'N':6,'C':6})
        for row in schedule:
            self.assertEqual(row['request_ordinals'],list(range(row['block_ordinal']*64,(row['block_ordinal']+1)*64)))
            expected=('N','C') if (row['repeat']+row['block_ordinal'])%2==0 else ('C','N')
            self.assertEqual(row['method'],expected[row['position']])

    def test_history_cross_split_question_does_not_read_answers(self):
        rows=candidates('gsm8k',2);source={'gsm8k:main:train:8':(rows[0]['question'],'')}
        found=D.history_matches({'source_id':'gsm8k:main:train:8','answer':rows[1]['question']},rows,source)
        self.assertEqual(set(found),{0})


if __name__=='__main__':unittest.main()
