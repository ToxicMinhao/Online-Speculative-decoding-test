import json
from pathlib import Path
import random

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.k_prefix_protection_v1 import prepare
from experiments.ok_coding import prepare as original


def candidates(count):
    return [{'source_index': index, 'source_id': f'codealpaca_train:{index}',
             'text': f'Write code {index}\n    preserve indentation',
             'prompt_ids': [1, index + 3], 'prompt_tokens': 2}
            for index in range(count)]

class TestPrepare(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_selection_matches_fixed_source_order_shuffle_and_preserves_text(self):
        eligible = candidates(140)
        state = random.getstate()
        selected = prepare.select_evaluation(list(reversed(eligible)), [2])
        expected = list(range(140))
        random.Random(20261101).shuffle(expected)
        assert [row['source_index'] for row in selected] == expected[:128]
        assert random.getstate() == state
        assert selected[0]['messages'][0]['content'] == eligible[expected[0]]['text']
        assert selected[0]['prompt_ids'] == eligible[expected[0]]['prompt_ids']
        assert selected[-1]['panel_ordinal'] == 127

    def test_selection_shortfall_never_shrinks_panel(self):
        with self.assertRaisesRegex(ValueError, '127 < 128'):
            prepare.select_evaluation(candidates(127), [2])

    def test_four_method_schedule_exact_balances_and_included_gate(self):
        requests = prepare.select_evaluation(candidates(128), [2])
        schedule = prepare.evaluation_schedule(requests)
        assert len(schedule) == 3072
        assert set(schedule[0]) == {'seed', 'event', 'request_id', 'panel_ordinal', 'repeat', 'method', 'position'}
        assert [row['method'] for row in schedule[:4]] == ['F', 'K', 'K_HALF', 'KP']
        for seed in (17, 29):
            cell = [row for row in schedule if row['seed'] == seed]
            balance = prepare.schedule_balance(cell)
            assert balance['generations'] == 1536
            assert all(values == [96] * 4 for values in balance['method_positions'].values())
            assert set(balance['pairwise_precedence'].values()) == {192}
            for begin in range(0, 128, 8):
                block = prepare.schedule_balance([row for row in cell if begin <= row['panel_ordinal'] < begin + 8])
                assert block['generations'] == 96
                assert all(values == [6] * 4 for values in block['method_positions'].values())
                assert set(block['pairwise_precedence'].values()) == {12}
            assert cell == prepare.evaluation_schedule(requests, seed=seed)
        assert prepare.evaluation_schedule(requests)[0]['event'] == 4
        with self.assertRaises(ValueError):
            prepare.evaluation_schedule(requests, event=2)

    def test_explicit_completed_allocations_match_id_and_normalized_text(self):
        tmp_path = self.directory
        source = [{'instruction': 'same\n  code', 'input': ''},
                  {'instruction': 'same code', 'input': ''},
                  {'instruction': 'reserved later', 'input': ''},
                  {'instruction': 'only appears in answer', 'input': ''}]
        first = tmp_path / 'TRAINING_REQUESTS.json'
        first.write_text(json.dumps([{'source_id': 'codealpaca_train:0', 'text': 'same\n  code'}]))
        later = tmp_path / 'LATER_PANEL.json'
        later.write_text(json.dumps({'requests': [{'source_key': 'codealpaca_train', 'source_index': 2}],
                                    'output': 'only appears in answer',
                                    'messages': [{'role': 'assistant', 'content': 'only appears in answer'}]}))
        history, receipts = prepare.add_allocation_evidence({}, source, [first, later, first])
        assert set(history) == {0, 1, 2}
        assert len(receipts) == 2
        assert 'normalized_user_prompt' in history[1][0]['reason']

    def test_coding_inventory_is_explicitly_added_despite_old_folder_skip(self):
        tmp_path = self.directory
        folder = tmp_path / 'research/dflash_ok_coding_20261030/production_002/data_preparation'
        folder.mkdir(parents=True)
        selected = folder / 'EVALUATION_REQUESTS.json'
        selected.write_text('[]')
        (folder / 'SOURCE.json').write_text('[]')
        (folder / 'METRICS.json').write_text('[]')
        assert selected not in list(original.history_candidates(tmp_path))
        assert list(prepare.coding_allocation_paths(tmp_path)) == [selected]

    def test_upstream_binding_preserves_original_training_inventory(self):
        tmp_path = self.directory
        rows = [{'instruction': 'prompt ' + str(index), 'input': ''} for index in range(640)]
        selected = {'training': candidates(512), 'evaluation': candidates(128)}
        for role, indices in (('training', range(512)), ('evaluation', range(512, 640))):
            selected[role] = [{'source_index': i, 'source_id': f'codealpaca_train:{i}',
                               'text': 'prompt ' + str(i)} for i in indices]
            (tmp_path / (role.upper() + '_REQUESTS.json')).write_text(json.dumps(selected[role]))
        binding = original.canonical_digest(selected)
        context = patch.object(prepare, 'UPSTREAM_BINDING', binding)
        context.start()
        self.addCleanup(context.stop)
        (tmp_path / 'DATA_CONTRACT.json').write_text(json.dumps({'data_binding': binding}))
        assert prepare.upstream_requests(tmp_path, rows) == selected
        selected['training'][0]['text'] = 'changed'
        (tmp_path / 'TRAINING_REQUESTS.json').write_text(json.dumps(selected['training']))
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            prepare.upstream_requests(tmp_path, rows)


if __name__ == "__main__":
    unittest.main()
