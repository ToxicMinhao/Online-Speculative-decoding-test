import unittest
from types import SimpleNamespace
from experiments.successor_shuffle.parent_worker import stop_after_first_round


class ParentStopTests(unittest.TestCase):
    def test_stops_only_after_complete_first_round(self):
        request = SimpleNamespace(output_ids=[1, 2, 3], finished_reason=None)
        request.finished = lambda: request.finished_reason is not None
        stop_after_first_round(
            SimpleNamespace(trace=[{}]), request, lambda n: ("diagnostic", n)
        )
        self.assertEqual(request.finished_reason, ("diagnostic", 3))
        self.assertEqual(request.output_ids, [1, 2, 3])

    def test_preserves_native_eos(self):
        request = SimpleNamespace(
            output_ids=[1, 9], finished_reason="eos", finished=lambda: True
        )
        stop_after_first_round(
            SimpleNamespace(trace=[{}]), request, lambda n: ("diagnostic", n)
        )
        self.assertEqual(request.finished_reason, "eos")

    def test_missing_or_second_block_rejected(self):
        for blocks in ([], [{}, {}]):
            with self.assertRaises(ValueError):
                stop_after_first_round(SimpleNamespace(trace=blocks), None, None)
