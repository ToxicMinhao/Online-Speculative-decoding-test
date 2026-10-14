"""Synthetic selection-integrity checks; no datasets or network required."""
import argparse
import contextlib
import copy
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import select_prompts as selector
from build_streams import read_selection


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for label, source in selector.SOURCES.items():
            indices = list(range(source['rows']))
            random.Random(selector.SEED).shuffle(indices)
            for rank, index in enumerate(indices[:130], 1):
                prompt = f'Synthetic {label} question {rank}.\u2028No outside input is needed.'
                self.rows.append({'label': label, 'source_dataset': source['repo'], 'source_revision': source['revision'],
                    'source_split': 'train', 'source_row_index': index, 'source_id': f'{label}_{rank}',
                    'question_id': f'{label}:{label}_{rank}', 'prompt': prompt,
                    'prompt_sha256': selector.digest(prompt.encode()), 'sampling_rank': rank,
                    'seed': selector.SEED, 'extraction_valid': True})
        self.review = {'decisions': [{'question_id': r['question_id'], 'prompt_sha256': r['prompt_sha256'],
            'decision': 'accept', 'reason': 'Synthetic valid fixture'} for r in self.rows]}

    def run_selection(self, rows=None, review=None):
        (self.root/'candidates.jsonl').write_bytes(selector.jsonl_bytes(self.rows if rows is None else rows))
        (self.root/'review.json').write_bytes(selector.json_bytes(self.review if review is None else review))
        args = argparse.Namespace(candidates=str(self.root/'candidates.jsonl'), reviews=[str(self.root/'review.json')],
                                  output_dir=str(self.root/'output'))
        with contextlib.redirect_stdout(io.StringIO()):
            selector.select(args)
        return read_selection(self.root/'output/selected_prompts.jsonl')[0]

    def test_replacement_uses_next_rank_and_preserves_unicode(self):
        review = copy.deepcopy(self.review)
        review['decisions'][0].update(decision='reject', reason='Synthetic invalid fixture')
        rows = self.run_selection(review=review)
        code = [r for r in rows if r['label'] == 'code']
        self.assertEqual([r['sampling_rank'] for r in code], list(range(2, 130)))
        self.assertIn('\u2028', code[0]['prompt'])
        self.assertEqual(len({r['question_id'] for r in rows}), 256)

    def test_modified_prompt_does_not_inherit_review(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['prompt'] += ' Different content.'
        with self.assertRaises(ValueError):
            self.run_selection(rows=rows)
        self.assertFalse((self.root/'output').exists())

    def test_answer_contamination_rejected_before_write(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['assistant_answer'] = 'Must never be exported.'
        with self.assertRaises(ValueError):
            self.run_selection(rows=rows)
        self.assertFalse((self.root/'output').exists())

    def test_seeded_index_binding_and_missing_review(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['source_row_index'] = (rows[0]['source_row_index'] + 1) % selector.SOURCES['code']['rows']
        with self.assertRaises(ValueError):
            self.run_selection(rows=rows)
        review = copy.deepcopy(self.review)
        review['decisions'].pop(0)
        with self.assertRaises(ValueError):
            self.run_selection(review=review)


if __name__ == '__main__':
    unittest.main()
