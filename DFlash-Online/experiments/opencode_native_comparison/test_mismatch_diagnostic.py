"""CPU checks for diagnostic row mapping and historical-reproduction gates."""
import unittest
from .mismatch_diagnostic import logit_positions, replay_check


class DiagnosticTest(unittest.TestCase):
    def test_prefill_final_row(self):
        self.assertEqual(logit_positions(list(range(65)), 1), [64])

    def test_decode_and_verify(self):
        self.assertEqual(logit_positions([343], 1), [343])
        self.assertEqual(logit_positions(list(range(343, 359)), 16), list(range(343, 359)))

    def test_unknown_row_layout_rejected(self):
        with self.assertRaises(ValueError):
            logit_positions([1, 2, 3], 2)

    def test_token_mismatch_remains_failed(self):
        old = dict(response_ids=[11, 151645], finish_reason='eos', stop_token_ids=[151645],
                   metadata={'native_meta_info': {'finish_reason': {'type': 'stop', 'matched': 151645}}})
        out = dict(output_ids=[323, 151645], meta_info=old['metadata']['native_meta_info'])
        self.assertFalse(replay_check(out, old)['passed'])
        out['output_ids'] = old['response_ids']
        self.assertTrue(replay_check(out, old)['passed'])

    def test_trace_and_stop_must_match(self):
        old = dict(response_ids=[11, 151645], finish_reason='eos', stop_token_ids=[151645],
                   metadata={'trace': [{'target_argmax': [11]}]})
        out = dict(output_ids=old['response_ids'], meta_info={'finish_reason': {'type': 'stop'}},
                   trace={'blocks': old['metadata']['trace']})
        self.assertTrue(replay_check(out, old)['passed'])
        out['trace']['blocks'] = []
        self.assertFalse(replay_check(out, old)['passed'])
        out['meta_info']['finish_reason'] = {'type': 'length'}
        self.assertFalse(replay_check(out, old)['exact_termination'])
