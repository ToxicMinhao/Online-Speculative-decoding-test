from copy import deepcopy
import unittest
from experiments.successor_shuffle import data as D
from experiments.survival_confirmation import data as P


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return list(text.encode())


def prior():
    questions = dict(test=[f"Question {i}?" for i in range(1319)])
    audit = dict(
        eligible_count=1063,
        eligible_rows_in_hash_order=list(range(1063)),
        outcomes_used=False,
        reference_availability_used=False,
        prior_exposure_used=False,
    )
    return P.build_panel(
        questions, list(range(768)), audit, Tokenizer(), {}, exclusion_rows=[]
    )


class DataTests(unittest.TestCase):
    def test_remaining295_exact_order_and256(self):
        selected, audit = D.select_rows(prior())
        self.assertEqual(selected, sorted(range(768, 1063), key=D.row_key)[:256])
        self.assertEqual(audit["eligible_count"], 295)
        self.assertFalse(set(range(768)) & set(selected))

    def test_additional_reservation_and_complete_eight_blocks(self):
        selected, audit = D.select_rows(
            prior(), additional_excluded_rows=range(768, 816), reserved_test_rows=[816]
        )
        self.assertEqual(len(selected), 240)
        self.assertFalse(set(range(768, 817)) & set(selected))
        self.assertEqual(audit["eligible_count"], 246)

    def test_parent_tamper_fails_before_selection(self):
        p = prior()
        p["requests"][0]["source_row"] = 1299
        with self.assertRaises(ValueError):
            D.select_rows(p)

    def test_matched64_deterministic_independent_of_endpoint_order(self):
        selected, _ = D.select_rows(prior())
        requests = [dict(source_row=i, request_id=f"r{i}") for i in selected]
        a = D.matched_subset(requests)
        b = D.matched_subset(list(reversed(requests)))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        self.assertEqual(
            [r["source_row"] for r in a],
            sorted(selected, key=lambda i: D.row_key(i, D.MATCHED_PREFIX))[:64],
        )


if __name__ == "__main__":
    unittest.main()
