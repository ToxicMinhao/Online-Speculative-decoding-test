from copy import deepcopy
import unittest
from experiments.successor_shuffle import matched_start as M
from experiments.successor_shuffle.data import digest


def fixture():
    requests = []
    refs = {}
    parents = {}
    children = {}
    for i in range(64):
        rid = f"r{i}"
        q = dict(request_id=rid, prompt_ids=[1, i], stop_token_ids=[151645])
        requests.append(q)
        tokens = [10, 20, 21, 151645] if i % 5 != 4 else [151645]
        refs[rid] = dict(
            prompt_ids=q["prompt_ids"], response_ids=tokens, binding_digest="a" * 64
        )
        proposals = [20, 21, 151645] + [9] * 12 if i % 5 != 4 else []
        if i % 5 == 0:
            proposals[1] = 99  # first error followed by valid EOS: high
        if i % 5 == 1:
            proposals[0] = proposals[1] = 99  # eligible successor: low
        if i % 5 == 2:
            proposals[2] = 99  # EOS error without successor: ineligible
        p = dict(
            request_id=rid,
            prompt_ids=q["prompt_ids"],
            seed_token=tokens[0],
            draft_tokens=proposals,
            start_digest=digest(q["prompt_ids"] + [tokens[0]]),
            checkpoint_id="/parent",
            canonical_binding_digest="a" * 64,
        )
        parents[rid] = p
        children[rid] = {m: dict(p, checkpoint_id="/" + m) for m in M.METHODS}
        if proposals:
            children[rid]["TRUE"]["draft_tokens"] = [20, 21, 151645] + [9] * 12
    return parents, children, requests, refs


class MatchedTests(unittest.TestCase):
    def test_eos_high_ineligible_all_correct_and_ties_have_denominators(self):
        p, c, q, r = fixture()
        a = M.analyze(p, c, q, r)
        self.assertEqual(
            a["parent_strata"],
            dict(
                successor_high=13,
                successor_low=13,
                successor_ineligible=13,
                all_correct=13,
                seed_eos=12,
            ),
        )
        rows = {(x["parent_stratum"], x["method"]): x for x in a["groups"]}
        self.assertEqual(rows["successor_high", "TRUE"]["repair_count"], 13)
        self.assertEqual(rows["successor_ineligible", "TRUE"]["repair_count"], 13)
        self.assertEqual(rows["all_correct", "TRUE"]["old_error_denominator"], 0)
        self.assertEqual(rows["successor_high", "SHUFFLED"]["ties"], 13)

    def test_prefix_break_prevents_counting_repair(self):
        p, c, q, r = fixture()
        c["r0"]["TRUE"]["draft_tokens"] = [99, 21, 151645] + [9] * 12
        a = M.analyze(p, c, q, r)
        row = next(
            v for v in a["rows"] if v["request_id"] == "r0" and v["method"] == "TRUE"
        )
        self.assertTrue(row["earlier_correct_prefix_broken"])
        self.assertFalse(row["old_first_error_repaired_with_prefix_intact"])

    def test_missing_start_arm_or_wrong_seed_rejected(self):
        for change in ("missing", "arm", "seed"):
            p, c, q, r = fixture()
            if change == "missing":
                p.pop("r0")
            elif change == "arm":
                c["r0"].pop("K")
            else:
                c["r0"]["TRUE"]["seed_token"] = 999
            with self.subTest(change=change), self.assertRaises(ValueError):
                M.analyze(p, c, q, r)

    def test_native_observation_uses_proposals_and_canonical_not_target_argmax(self):
        p, c, q, r = fixture()
        request = q[0]
        canonical = r["r0"]
        record = dict(
            checkpoint_id="/parent",
            prompt_ids=request["prompt_ids"],
            response_ids=canonical["response_ids"],
            metadata=dict(
                trace=[
                    dict(
                        output_start=0,
                        anchor_token=10,
                        draft_tokens=p["r0"]["draft_tokens"],
                        target_argmax=[999] * 16,
                    )
                ]
            ),
        )
        observed = M.observation(record, request, canonical, checkpoint="/parent")
        self.assertEqual(observed, p["r0"])
        with self.assertRaises(ValueError):
            M.observation(record, request, canonical, checkpoint="/wrong")


if __name__ == "__main__":
    unittest.main()
