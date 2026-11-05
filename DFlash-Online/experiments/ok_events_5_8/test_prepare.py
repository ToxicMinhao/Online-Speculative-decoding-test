"""Question-only preparation contracts; no model/runtime or remote work."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from .prepare import (SEED, bind_source_rows, build_panels, digest, normalized, retained_counts, save,
                      select_panel, select_training)


class PreparationTests(unittest.TestCase):
    def test_development_original_index_is_not_the_source_row(self):
        request = dict(request_id='dev', original_index=5, prompt_ids=[42])
        source = dict(request_id='dev', source_row=599, prompt_ids_sha256=digest([42]))
        self.assertEqual(bind_source_rows([request], [source]), {'dev': 599})
        self.assertEqual(request['original_index'], 5)
        source['prompt_ids_sha256'] = digest([43])
        with self.assertRaisesRegex(ValueError, 'prompt binding'):
            bind_source_rows([request], [source])

    def test_training_order_excludes_normalized_duplicates_and_prior_evaluation(self):
        questions = ['Ａ  one', 'old two', 'A one', 'panel\tquestion', 'new five', 'new six']
        selected, skipped = select_training(questions, previous_count=2,
            excluded_questions=['panel question'], count=2)
        self.assertEqual(selected, [4, 5])
        self.assertEqual([r['source_row'] for r in skipped], [2, 3])
        self.assertEqual(normalized('Ａ\t one'), 'A one')

    def test_training_exhaustion_does_not_shrink(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_training(['old', 'old', 'test'], previous_count=1,
                            excluded_questions=['test'], count=1)

    def test_panel_source_content_dedup_and_isolated_rng(self):
        questions = ['zero', 'one', 'ＯＮＥ', 'train duplicate', 'four', 'four', 'six', 'seven']
        state = random.getstate()
        selected, skipped, eligible = select_panel(questions, excluded_test_ids={0},
            excluded_normalized={normalized('train duplicate'), normalized('ＯＮＥ')}, count=3)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(eligible, [1, 4, 6, 7])
        expected = eligible.copy()
        random.Random(SEED).shuffle(expected)
        self.assertEqual(selected, expected[:3])
        self.assertEqual([row['source_row'] for row in skipped], [0, 2, 3, 5])

    def test_panel_normalization_and_exhaustion(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_panel(['A  B', 'Ａ\tB', 'training'], excluded_test_ids=set(),
                         excluded_normalized={'training'}, count=2)

    def test_write_once_and_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'plan.json'
            save(path, {'fixed': 1})
            save(path, {'fixed': 1})
            with self.assertRaisesRegex(ValueError, 'Refusing'):
                save(path, {'fixed': 2})
            self.assertEqual(json.loads(path.read_text()), {'fixed': 1})

    def test_historical_counts_require_exact_three_valid_repeats(self):
        rows = [dict(request_id='x', method='O4', repeat=i, verification_calls=7,
                     valid_measurement=True, errors=[]) for i in range(3)]
        self.assertEqual(retained_counts(rows, 'O4', [{'request_id': 'x'}]), {'x': 7})
        for mutation in ('repeat', 'calls', 'invalid'):
            bad = copy.deepcopy(rows)
            if mutation == 'repeat':
                bad[-1]['repeat'] = 1
            elif mutation == 'calls':
                bad[-1]['verification_calls'] = 8
            else:
                bad[-1]['valid_measurement'] = False
            with self.assertRaises(ValueError):
                retained_counts(bad, 'O4', [{'request_id': 'x'}])

    def test_panel_builder_preserves_retained_inputs_and_fresh_has_no_labels(self):
        totals = {'development': {'O4': 2994, 'K4': 2942}, 'additional': {'O4': 3130, 'K4': 3058}}
        panels, measurements = [], {}
        for name, methods in totals.items():
            requests = [dict(request_id=f'{name}_{i}', panel_ordinal=i, original_index=i,
                             prompt_ids=[i + 1], response_ids=[99], finish_reason='eos') for i in range(64)]
            panels.append(dict(name=name, requests=requests, provenance={'fixed': True}))
            measurements[name] = [dict(request_id=r['request_id'], method=m, repeat=repeat,
                verification_calls=total - 63 * 40 if i == 0 else 40, valid_measurement=True, errors=[])
                for m, total in methods.items() for repeat in range(3) for i, r in enumerate(requests)]
        frozen = copy.deepcopy(panels)
        fresh = [dict(request_id='fresh', prompt_ids=[777])]
        built = build_panels(panels, measurements, fresh)
        self.assertEqual(panels, frozen)
        self.assertEqual(built[0]['requests'], frozen[0]['requests'])
        self.assertEqual([row['canonical_mode'] for row in built], ['retained', 'retained', 'fresh'])
        self.assertEqual(built[2]['selection_seed'], SEED)
        self.assertNotIn('response_ids', built[2]['requests'][0])
        self.assertEqual({key: sum(value.values()) for key, value in built[1]['historical_counts'].items()}, totals['additional'])


if __name__ == '__main__':
    unittest.main()
