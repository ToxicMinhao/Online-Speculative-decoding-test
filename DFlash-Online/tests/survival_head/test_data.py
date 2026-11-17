from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.survival_head import data


def source():
    return dict(dataset='gsm8k', previously_used_development_panel=True, input_only_selection=True,
        requests=[dict(request_id=f'dev-{i}', panel_ordinal=i, prompt_ids=[1, i]) for i in range(256)])


def feedback():
    return [dict(request_id=f'train-{i}') for i in range(128)]


class DataTests(unittest.TestCase):
    def test_sha256_exact128_and_original_index_preserved(self):
        panel = source()
        result = data.select_panel(panel, feedback())
        expected = sorted(panel['requests'], key=lambda r: hashlib.sha256(
            ('survival-v1|' + r['request_id']).encode('utf-8')).digest())[:128]
        self.assertEqual([r['request_id'] for r in result['requests']], [r['request_id'] for r in expected])
        self.assertEqual([r['source_panel_ordinal'] for r in result['requests']], [r['panel_ordinal'] for r in expected])
        self.assertEqual([r['panel_ordinal'] for r in result['requests']], list(range(128)))
        self.assertEqual(data.verify_panel(result, panel, feedback())['status'], 'passed')
        self.assertEqual(panel, source())

    def test_unicode_id_uses_utf8_and_no_lexicographic_shortcut(self):
        panel = source(); panel['requests'][0]['request_id'] = '題目é'
        result = data.select_panel(panel, feedback())
        for row in result['requests']:
            self.assertEqual(row['selection_sha256'], hashlib.sha256(
                ('survival-v1|' + row['request_id']).encode('utf-8')).hexdigest())

    def test_reject_overlap_duplicates_outputs_and_provenance_change(self):
        for change in ('overlap', 'duplicate', 'output', 'history'):
            panel, train = source(), feedback()
            if change == 'overlap': train[0]['request_id'] = 'dev-255'
            if change == 'duplicate': panel['requests'][1]['request_id'] = 'dev-0'
            if change == 'output': panel['requests'][0]['response_ids'] = [1]
            if change == 'history': panel['previously_used_development_panel'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                data.select_panel(panel, train)

    def test_panel_immutable_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'PANEL.json'
            result = data.select_panel(source(), feedback(), output=path)
            with self.assertRaises(FileExistsError): data.select_panel(source(), feedback(), output=path)
            result['requests'][0]['prompt_ids'][0] += 1
            with self.assertRaises(ValueError): data.verify_panel(result, source(), feedback())

    def test_packet_identity_content_and_order_without_tensor_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); features=[]; records=[]; ids=[]
            for index in range(128):
                rid=f'r{index}'; ids.append(rid)
                feature=root/f'{index}.pt'; feature.write_bytes(b'not-a-tensor')
                record=root/f'{index}.json'; record.write_text(json.dumps(dict(record_id=rid)))
                features.append(feature); records.append(record)
            pool=data.seal_shared_packets(features, records, output=root/'POOL.json', expected_request_ids=ids)
            self.assertEqual(data.verify_shared_packets(pool)['packets'], 128)
            reordered=deepcopy(pool); reordered['records'].reverse()
            with self.assertRaises(ValueError): data.verify_shared_packets(reordered)
            features[0].write_bytes(b'changed')
            with self.assertRaises(ValueError): data.verify_shared_packets(pool)


if __name__ == '__main__': unittest.main()
