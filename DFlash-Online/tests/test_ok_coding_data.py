"""CPU fixtures for source-only selection, historical exclusions and sealed slots."""
from collections import Counter
from pathlib import Path
import random
import tempfile
import unittest

from experiments.ok_coding.prepare import (
    assemble, canonical_digest, evaluation_schedule, history_candidates,
    history_matches, normalize, prepare_population, select_requests,
)


class CodingDataTests(unittest.TestCase):
    def test_source_solution_never_changes_prompt_or_selection(self):
        first = [{'instruction': 'Write code', 'input': '  x = 1\n    y = 2', 'output': 'secret solution'},
                 {'instruction': 'Second request', 'input': '', 'output': 'second secret'}]
        changed = [{**row, 'output': 'different hidden solution'} for row in first]
        self.assertEqual(assemble(first[0]), 'Write code\n\n  x = 1\n    y = 2')
        render = lambda text: list(text.encode())
        self.assertEqual(prepare_population(first, {}, render, evaluation_reservoir=1),
                         prepare_population(changed, {}, render, evaluation_reservoir=1))

    def test_partition_before_filter_and_lowest_index_deduplication(self):
        source = [{'instruction': 'a', 'input': ''}, {'instruction': ' a ', 'input': ''},
                  {'instruction': '', 'input': ''}, {'instruction': 'eval', 'input': ''}]
        eligible, excluded, audit = prepare_population(source, {0: [{'path': 'old allocation'}]},
                                                       lambda text: [1], evaluation_reservoir=1)
        self.assertEqual(audit['partition_boundary'], 3)
        self.assertEqual(eligible['training'], [])
        self.assertEqual(eligible['evaluation'][0]['source_index'], 3)
        self.assertIn('duplicate_within_reservoir', excluded[1]['reasons'])
        self.assertEqual(excluded[1]['lowest_source_index'], 0)
        self.assertIn('historical_normalized_prompt', excluded[1]['reasons'])

    def test_cross_partition_duplicate_removed_from_both_even_with_length_difference(self):
        source = [{'instruction': 'a b', 'input': ''}, {'instruction': 'unique', 'input': ''},
                  {'instruction': 'a   b', 'input': ''}]
        eligible, excluded, audit = prepare_population(source, {}, lambda text: [1], evaluation_reservoir=1)
        self.assertEqual([row['source_index'] for row in eligible['training']], [1])
        self.assertEqual(eligible['evaluation'], [])
        self.assertEqual(audit['cross_reservoir_duplicate_keys'], 1)
        self.assertTrue(all('duplicate_across_reservoirs' in row['reasons'] for row in excluded))

    def test_normalization_does_not_rewrite_actual_prompt(self):
        source = [{'instruction': 'cafe\u0301\n  indent', 'input': ''},
                  {'instruction': 'café indent', 'input': ''}, {'instruction': 'eval', 'input': ''}]
        seen = []
        def render(text):
            seen.append(text)
            return [1]
        eligible, _, _ = prepare_population(source, {}, render, evaluation_reservoir=1)
        self.assertEqual(eligible['training'][0]['text'], 'cafe\u0301\n  indent')
        self.assertEqual(seen, ['cafe\u0301\n  indent', 'eval'])
        self.assertEqual(normalize(source[0]['instruction']), normalize(source[1]['instruction']))

    def test_exact_rendered_cap_and_invalid_fields(self):
        source = [{'instruction': 'limit', 'input': ''}, {'instruction': 'long', 'input': ''},
                  {'instruction': 'bad', 'input': None}, {'instruction': 'eval', 'input': ''}]
        eligible, excluded, _ = prepare_population(source, {}, lambda text: [1] * (225 if text == 'long' else 224),
                                                   evaluation_reservoir=1)
        self.assertEqual([row['source_index'] for row in eligible['training']], [0])
        self.assertEqual([row['source_index'] for row in eligible['evaluation']], [3])
        self.assertEqual(excluded[0]['reasons'], ['rendered_prompt_outside_1_224'])
        self.assertEqual(excluded[1]['reasons'], ['invalid_instruction_or_input'])

    def test_history_matches_source_ids_user_text_and_not_answers_or_candidates(self):
        lookup = {'allocated prompt': {2}, 'solution only': {3}, 'discarded candidate': {4}}
        value = {'requests': [{'source_id': 'codealpaca:1'}, {'role': 'user', 'content': 'allocated\n prompt'},
                              {'source_key': 'codealpaca_train', 'source_index': 5}],
                 'output': 'solution only', 'messages': [{'role': 'assistant', 'content': 'solution only'}],
                 'rejections': [{'text': 'discarded candidate'}]}
        self.assertEqual(set(history_matches(value, lookup)), {1, 2, 5})

    def test_source_catalog_is_lookup_only_but_reserved_pool_is_selection_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ['research/old/source_cache/codealpaca_prompt_only.jsonl',
                     'research/old/inputs/codealpaca_candidate_pool.jsonl',
                     'research/old/inputs/manifest.json',
                     'research/new/PLAN.json', 'research/old/source_catalog.json']
            for name in paths:
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('{}')
            selected = {str(path.relative_to(root)) for path in history_candidates(root)}
            self.assertEqual(selected, {'research/old/inputs/manifest.json', 'research/new/PLAN.json'})
            # Reserved IDs are in the retained allocation manifest, not inferred
            # from every row of its raw source catalog.
            self.assertEqual(set(history_matches({'files': {'reserved': {'ids': ['codealpaca:9']}}}, {})), {9})

    def test_selection_uses_isolated_fixed_rng_and_retains_order(self):
        def candidates(count):
            return [{'source_index': i, 'source_id': f'codealpaca_train:{i}', 'text': f'q{i}',
                     'prompt_ids': [i], 'prompt_tokens': 1} for i in range(count)]
        eligible = {'training': candidates(600), 'evaluation': candidates(200)}
        global_state = random.getstate()
        selected, missing = select_requests(eligible, [151645, 151643])
        self.assertEqual(random.getstate(), global_state)
        self.assertEqual(missing, [])
        for role, seed, length in [('training', 20261030, 512), ('evaluation', 20261031, 128)]:
            expected = list(range(len(eligible[role])))
            random.Random(seed).shuffle(expected)
            self.assertEqual([row['source_index'] for row in selected[role]], expected[:length])
            self.assertEqual([row['request_index'] for row in selected[role]], list(range(length)))
        self.assertEqual(canonical_digest(selected), canonical_digest(select_requests(eligible, [151645, 151643])[0]))

    def test_shortfall_is_explicit(self):
        selected, missing = select_requests({'training': [], 'evaluation': []}, [1])
        self.assertEqual(selected, {'training': [], 'evaluation': []})
        self.assertEqual(missing, [{'reservoir': 'training', 'required': 512, 'available': 0, 'shortfall': 512},
                                  {'reservoir': 'evaluation', 'required': 128, 'available': 0, 'shortfall': 128}])

    def test_schedule_has_fixed_cells_and_exact_position_and_precedence_balance(self):
        requests = [{'request_id': str(i)} for i in range(128)]
        schedule = evaluation_schedule(requests)
        self.assertEqual(len(schedule), 4608)
        self.assertEqual(list(dict.fromkeys((x['seed'], x['event']) for x in schedule)),
                         [(17, 2), (17, 4), (29, 2), (29, 4)])
        for seed, event in [(17, 2), (17, 4), (29, 2), (29, 4)]:
            cell = [x for x in schedule if (x['seed'], x['event']) == (seed, event)]
            for ordinal in range(128):
                position_counts = Counter((x['method'], x['position']) for x in cell if x['panel_ordinal'] == ordinal)
                self.assertEqual(position_counts, Counter({(arm, pos): 1 for arm in 'FOK' for pos in range(3)}))
            for left, right in [('F', 'O'), ('F', 'K'), ('O', 'K')]:
                before = 0
                for start in range(0, len(cell), 3):
                    order = [x['method'] for x in cell[start:start+3]]
                    before += order.index(left) < order.index(right)
                self.assertEqual(before, 192)


if __name__ == '__main__':
    unittest.main()
