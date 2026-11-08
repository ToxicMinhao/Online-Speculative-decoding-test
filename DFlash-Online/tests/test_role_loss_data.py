"""Input selection is deterministic and independent of response outcomes."""
import hashlib
import unittest

from experiments.role_loss_pilot.data import (
    REASONING_SUFFIX, overlap_index, overlap_reasons, prompt_fields,
    select_mathqa_training, select_panel, selection_key,
)


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n<|im_start|>assistant\n'

    def encode(self, text, **options):
        return list(text.encode())

    def decode(self, ids, **options):
        return bytes(ids).decode()


def panel_row(index, text):
    tokenizer = Tokenizer()
    rendered = tokenizer.apply_chat_template([dict(role='user', content=text)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return dict(request_id=f'request_{index}', dataset_id=f'original:{index}',
                panel_ordinal=index, prompt_ids=tokenizer.encode(rendered),
                max_new_tokens=2048, collect_feedback=False, stop_token_ids=[999])


class InputSelectionTest(unittest.TestCase):
    def test_hash_is_exact_and_uses_original_id(self):
        self.assertEqual(selection_key('gsm8k', 'test', 'abc'),
            hashlib.sha256(b'vfd-roleloss-pilot-v1|gsm8k|test|abc').hexdigest())
        rows = [panel_row(i, f'question {i}') for i in range(8)]
        expected = sorted(rows, key=lambda r: (selection_key('gsm8k', 'test', r['dataset_id']), r['dataset_id']))
        for row in rows:
            row['panel_ordinal'] += 100
            row['latency'] = 10000 - row['panel_ordinal']
        selected = select_panel(rows[::-1], 'gsm8k', Tokenizer(), count=4)
        self.assertEqual([r['source_id'] for r in selected], [r['dataset_id'] for r in expected[:4]])
        self.assertEqual([r['panel_ordinal'] for r in selected], list(range(4)))
        self.assertEqual([r['prompt_ids'] for r in selected], [r['prompt_ids'] for r in expected[:4]])

    def test_wrapper_is_not_part_of_normalized_stem(self):
        row = panel_row(0, '  Shared  Question? ' + REASONING_SUFFIX)
        fields = prompt_fields(row, 'gsm8k', Tokenizer())
        self.assertEqual(fields['normalized_stem'], 'shared question?')
        self.assertEqual(Tokenizer().decode(row['prompt_ids']), fields['rendered_prompt'])

    def test_cross_dataset_stem_overlap_excluded(self):
        panel = panel_row(0, 'Shared question?' + REASONING_SUFFIX)
        panel.update(prompt_fields(panel, 'gsm8k', Tokenizer()))
        sources = [dict(Problem='SHARED  QUESTION?', options='a)1'),
                   dict(Problem='Unique one?', options='a)2'),
                   dict(Problem='Unique two?', options='a)3')]
        selected, receipt = select_mathqa_training(sources, [panel], Tokenizer(), [999], count=2)
        self.assertNotIn('mathqa:train:0', [r['source_id'] for r in selected])
        self.assertIn('normalized_stem', receipt['exclusions'][0]['reason'])

    def test_source_order_dedup_precedes_hash_selection(self):
        sources = [dict(Problem='Same question?', options='a)1', correct='a'),
                   dict(Problem=' SAME QUESTION? ', options='b)2', correct='b'),
                   dict(Problem='Different?', options='c)3')]
        rows, receipt = select_mathqa_training(sources, [], Tokenizer(), [999], count=2)
        self.assertEqual({r['source_id'] for r in rows}, {'mathqa:train:0', 'mathqa:train:2'})
        self.assertEqual(receipt['exclusions'][0]['reason'], 'duplicate_earlier_source_row')
        self.assertTrue(all('correct' not in r and 'Rationale' not in r for r in rows))

    def test_cap_is_admitted_without_truncation_or_replacement(self):
        sources = [dict(Problem='x' * 250, options='a)1'), dict(Problem='Short?', options='a)2')]
        rows, receipt = select_mathqa_training(sources, [], Tokenizer(), [999], count=1)
        self.assertEqual(rows[0]['source_id'], 'mathqa:train:1')
        self.assertEqual(rows[0]['max_new_tokens'], 2048)
        self.assertEqual(receipt['exclusions'][0]['reason'], 'training_context_budget')
        with self.assertRaisesRegex(ValueError, 'shortfall'):
            select_mathqa_training(sources, [], Tokenizer(), [999], count=2)

    def test_exact_tokens_detect_overlap_even_if_stem_metadata_differs(self):
        a = panel_row(0, 'An input'); a.update(prompt_fields(a, 'gsm8k', Tokenizer()))
        b = dict(a, dataset='mathqa', source_id='different', normalized_stem='altered')
        self.assertIn('prompt_ids_sha256', overlap_reasons(b, overlap_index([a])))

    def test_duplicate_panel_source_ids_fail(self):
        rows = [panel_row(0, 'one'), panel_row(0, 'two')]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            select_panel(rows, 'gsm8k', Tokenizer(), count=2)


if __name__ == '__main__':
    unittest.main()
