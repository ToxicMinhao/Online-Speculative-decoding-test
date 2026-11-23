"""CPU checks of source boundaries, input isolation and prospective selection."""
import unittest
import json
from pathlib import Path
import tempfile

from experiments.opencode_native_comparison.contract import digest
from experiments.opencode_native_comparison.data import (
    DATASET, exclude_inputs, history_fingerprints, inventory_history, manifest_requests, normalize,
    prepare_input, rank, reservoir_plan, select_inputs, text_hash, training_length_bounds)
from experiments.opencode_native_comparison.seal import validate_source_rows


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        assert messages[0]['role'] == 'user'
        assert options == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return 'USER:'+messages[0]['content']+':ASSISTANT'

    def encode(self, text, **options):
        assert options == dict(add_special_tokens=False)
        return list(text.encode())


def row(index, raw=None, pool='train'):
    return dict(input=f'code {index}' if raw is None else raw, pool=pool,
        source_identity=dict(dataset=DATASET, revision='a'*40, split='train',
                             shard='train/0000.parquet', raw_index=index, shard_row_index=index))


def prepared(index, raw=None, pool='train'):
    return prepare_input(row(index, raw, pool), Tokenizer())


class DataTests(unittest.TestCase):
    def test_author_split_branch_and_bounded_ranges(self):
        self.assertEqual(reservoir_plan(['train', 'test'])['ranges'][1],
                         dict(pool='evaluation', split='test', start=0, stop=20000))
        single = reservoir_plan(['train'])
        self.assertEqual(single['ranges'][1], dict(pool='evaluation', split='train', start=30000, stop=50000))
        self.assertEqual(reservoir_plan(['test'])['ranges'][0]['split'], 'test')

    def test_raw_whitespace_preserved_and_answers_ignored(self):
        original = row(1, '  code\n\n  x ')
        original['output'] = 'DO NOT USE'
        value = prepare_input(original, Tokenizer())
        self.assertEqual(value['raw_input'], original['input'])
        self.assertEqual(value['rendered_prompt'], 'USER:  code\n\n  x :ASSISTANT')
        self.assertNotIn('output', value)

    def test_length_is_input_only_and_no_truncation(self):
        allowed = prepared(1, 'x'*(2048-len('USER::ASSISTANT')))
        denied = prepared(2, 'x'*(2049-len('USER::ASSISTANT')))
        self.assertNotIn('exclusion', allowed)
        self.assertEqual(denied['exclusion'], 'prompt_exceeds_2048_tokens')
        self.assertEqual(len(denied['prompt_ids']), 2049)
        self.assertEqual(prepared(3, '  ')['exclusion'], 'empty_or_nonstring_input')

    def test_cross_pool_duplicates_excluded_on_both_sides(self):
        rows = [prepared(1, 'same'), prepared(30001, 'same', 'evaluation'), prepared(2)]
        good, bad = exclude_inputs(rows, {})
        self.assertEqual([r['source_identity']['raw_index'] for r in good], [2])
        self.assertEqual(len(bad), 2)
        self.assertTrue(all('duplicate_across_source_pools' in r['reasons'] for r in bad))

    def test_within_pool_duplicates_use_source_identity_order(self):
        rows = [prepared(2, 'x'), prepared(1, 'x')]
        good, bad = exclude_inputs(rows, {})
        self.assertEqual(good[0]['source_identity']['raw_index'], 1)
        self.assertEqual(bad[0]['reasons'], ['duplicate_within_source_pool'])

    def test_history_checks_each_identity_form(self):
        rows = [prepared(i) for i in range(4)]
        history = {field:{rows[i][field]} for i,field in enumerate(
            ('source_identity_sha256','normalized_raw_sha256','rendered_prompt_sha256','prompt_ids_sha256'))}
        good, bad = exclude_inputs(rows, history)
        self.assertEqual(good, [])
        self.assertEqual(len(bad), 4)

    def test_history_ignores_answers_and_unwraps_user_chat(self):
        value = dict(conversations=[dict(role='user',content='Q'),dict(role='assistant',content='A')],
                     response='B', prompt='<|im_start|>user\nC<|im_end|>', prompt_ids=[4,5])
        found = history_fingerprints(value)
        self.assertIn(text_hash(normalize('Q')), found['normalized_raw_sha256'])
        self.assertIn(text_hash(normalize('C')), found['normalized_raw_sha256'])
        self.assertNotIn(text_hash(normalize('A')), found['normalized_raw_sha256'])
        self.assertNotIn(text_hash(normalize('B')), found['normalized_raw_sha256'])
        self.assertIn(digest([4,5]), found['prompt_ids_sha256'])

    def test_ranking_is_prespecified_canonical_json(self):
        value=prepared(1)
        self.assertEqual(rank(value, 'train')[0], digest(dict(seed=20261120, role='train', source_identity=value['source_identity'])))
        self.assertNotEqual(rank(value,'train'),rank(value,'evaluation'))

    def test_selection_is_disjoint_and_qualification_extremes_correct(self):
        rows=[prepared(i,'x'*(i+1)) for i in range(40)]+[prepared(30000+i,pool='evaluation') for i in range(5)]
        counts=dict(train=4,evaluation=3,warmup=2,qualification=16)
        a=select_inputs(rows,counts); b=select_inputs(list(reversed(rows)),counts)
        self.assertEqual(a,b)
        ids=[r['source_identity_sha256'] for group in a.values() for r in group]
        self.assertEqual(len(ids),len(set(ids)))
        reserved={r['source_identity_sha256'] for role in ('train','warmup') for r in a[role]}
        rest=[r for r in rows if r['pool']=='train' and r['source_identity_sha256'] not in reserved]
        self.assertEqual([r['prompt_tokens'] for r in a['qualification'][:4]],
                         sorted(r['prompt_tokens'] for r in rest)[:2]+sorted((r['prompt_tokens'] for r in rest),reverse=True)[:2])

    def test_insufficient_pool_does_not_expand(self):
        with self.assertRaisesRegex(ValueError,'Bounded reservoir insufficient'):
            select_inputs([prepared(1)])

    def test_equal_length_qualification_inputs_stay_distinct(self):
        rows=[prepared(i, f'code {i:03d}') for i in range(40)]
        rows += [prepared(30000+i, pool='evaluation') for i in range(5)]
        selected=select_inputs(rows,dict(train=4,evaluation=3,warmup=2,qualification=16))
        self.assertEqual(len({r['source_identity_sha256'] for r in selected['qualification']}),16)

    def test_history_parse_gaps_and_duplicate_copies_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('one.json','copy.json'):
                (root/name).write_text(json.dumps(dict(prompt_ids=[1,2],raw_input='code')))
            (root/'broken.json').write_text('{')
            result=inventory_history(root.glob('*.json'))
        self.assertEqual(len(result['coverage']['gaps']),1)
        self.assertEqual(sum(r.get('duplicate_content',False) for r in result['files']),1)
        self.assertEqual(result['fingerprints']['prompt_ids_sha256'],[digest([1,2])])

    def test_full_manifest_counts_and_no_gsm_tail(self):
        rows=[prepared(i) for i in range(7450)]
        rows += [prepared(30000+i,pool='evaluation') for i in range(1024)]
        manifest=manifest_requests(select_inputs(rows))
        self.assertEqual(len(manifest['train_requests']),7424)
        self.assertEqual(len(manifest['evaluation_requests']),1024)
        self.assertEqual(manifest['evaluation_requests'][-1]['panel_ordinal'],1023)
        self.assertEqual(sum(len(v) for v in manifest.values()),8466)

    def test_source_contract_rejects_extra_rows_answers_and_changed_revision(self):
        receipt=dict(split_order=['train'],split_rows={'train':2},revision='a'*40,
                     shards={'train/0000.parquet':{'sha256':'b'*64}})
        rows=[row(0),row(1)]
        validate_source_rows(rows,receipt)
        with self.assertRaises(ValueError): validate_source_rows(rows+[row(2)],receipt)
        with self.assertRaises(ValueError): validate_source_rows([rows[0],rows[0]],receipt)
        rows[0]['output']='forbidden'
        with self.assertRaises(ValueError): validate_source_rows(rows,receipt)

    def test_window_feature_bound_counts_full_output_cap(self):
        rows=[dict(prompt_ids=[1]*(i+1)) for i in range(256)]
        bounds=training_length_bounds(rows)
        self.assertEqual(bounds['windows'][0]['maximum_sequence_tokens'],sum(range(1,129))+128*2048)
        self.assertEqual(bounds['maximum_window_prompt_tokens'],sum(range(129,257)))


if __name__ == '__main__':
    unittest.main()
