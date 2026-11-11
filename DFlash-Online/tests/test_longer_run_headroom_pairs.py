import copy
import unittest
from experiments.longer_run_headroom.census import classify, digest
from experiments.longer_run_headroom.pairs import (
    score,
    neutral_parity,
    retained_same_state,
)


def block(proposals, accepted, *, start=0):
    return dict(
        output_start=start,
        anchor_token=0,
        draft_tokens=proposals,
        accepted_draft_tokens=accepted,
        native_accepted_draft_tokens=accepted,
        committed_tokens=list(range(1, accepted + 2)),
        committed_progress=accepted + 1,
        positions=list(range(start, start + 16)),
        target_argmax=list(range(1, 17)),
        finished=False,
        stop_clipped=False,
    )


class PairTests(unittest.TestCase):
    def fixture(self):
        parent = block([1, 9, 3, 4, 8], 1)
        s = dict(
            parent_block_sha256=digest(parent),
            **classify(parent["draft_tokens"], [1, 2, 3, 4, 5], 0, [99])
        )
        return parent, s, [0, 1, 2, 3, 4, 5, 6]

    def test_blocked_opportunity_requires_intact_prefix_and_two_followers(self):
        p, s, y = self.fixture()
        r = score(s, p, p, y, provenance="fixture")
        self.assertTrue(r["blocked_with_both"])
        self.assertFalse(r["repair"])
        child = block([9, 2, 3, 4, 5], 0)
        r = score(s, p, child, y, provenance="fixture")
        self.assertFalse(r["blocked_with_both"])
        self.assertTrue(r["earlier_prefix_break"])
        self.assertEqual(r["native_prefix_change"], -1)

    def test_repair_must_be_natively_accepted(self):
        p, s, y = self.fixture()
        child = block([1, 2, 3, 4, 5], 1)
        r = score(s, p, child, y, provenance="fixture")
        self.assertTrue(r["canonical_repair"])
        self.assertFalse(r["repair"])

    def test_realized_two_followers_and_signed_gain(self):
        p, s, y = self.fixture()
        r = score(s, p, block([1, 2, 3, 4, 5], 4), y, provenance="fixture")
        self.assertTrue(r["repair_with_two_accepted"])
        self.assertEqual(r["native_prefix_change"], 3)

    def test_repair_without_followers_is_not_payoff(self):
        p, s, y = self.fixture()
        r = score(s, p, block([1, 2, 8, 4, 5], 2), y, provenance="fixture")
        self.assertTrue(r["repair"])
        self.assertFalse(r["repair_with_two_accepted"])

    def test_wrong_committed_ids_are_invalid(self):
        p, s, y = self.fixture()
        c = block([1, 2, 3, 4, 5], 4)
        c["committed_tokens"] = [999]
        with self.assertRaisesRegex(ValueError, "emitted"):
            score(s, p, c, y, provenance="fixture")

    def test_neutral_all_proposals_gate_including_unaccepted_tail(self):
        p, s, y = self.fixture()
        c = copy.deepcopy(p)
        c["draft_tokens"][-1] = 99
        self.assertEqual(neutral_parity(p, c)["status"], "failed")

    def test_offset_only_is_not_a_reusable_state(self):
        first = block([1, 2, 3], 2)
        second = block([4, 5, 6], 2, start=3)
        p = dict(
            prompt_ids=[7],
            response_ids=[0, 1, 2, 3, 4, 5, 6],
            stop_token_ids=[99],
            max_new_tokens=2048,
            finish_reason="length",
            metadata=dict(trace=[first, second]),
        )
        c = copy.deepcopy(p)
        self.assertTrue(retained_same_state(p, c, 2))
        c["metadata"]["trace"][0]["draft_tokens"][-1] = 99
        self.assertFalse(retained_same_state(p, c, 2))


if __name__ == "__main__":
    unittest.main()
