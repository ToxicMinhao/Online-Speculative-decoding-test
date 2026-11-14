"""Selection tests use no model, optimizer, probe outcome or training RNG."""
from copy import deepcopy
import hashlib
import math
import unittest
from unittest import mock

from experiments.threatened_repair_replay import core


def tag(rid, slot=3, start=0, pid=0, ordinal=0):
    return dict(request_id=rid, canonical_start=start, proposal_slot=slot,
        reference_presentation_id=pid, anchor_ordinal=ordinal,
        layout_id=f"{rid}:{pid}", canonical_token=2, weight=math.exp(-(slot-1)/7))


def observation(t, prefix, *, correct=None, margin=None, valid=15):
    correct = prefix >= t["proposal_slot"] if correct is None else correct
    return dict(t, accepted_prefix=prefix, valid_proposals=valid, valid=True,
        native_correct=correct, margin=(1. if correct else -1.) if margin is None else margin)


class SelectionTests(unittest.TestCase):
    def select(self, tags, pre, native, excluded=()):
        return core.select_replay(tags, pre, native, window=8, current_request_ids=excluded)

    def test_threat_requires_first_rejection_and_current_acceptance(self):
        tags = [tag(rid) for rid in ("threat", "earlier_wrong", "earlier_correct", "already", "current")]
        pre = [observation(t, 5 if t["request_id"] != "already" else 1) for t in tags]
        native = [observation(tags[0], 2), observation(tags[1], 1),
                  observation(tags[2], 1, correct=True), observation(tags[3], 2), observation(tags[4], 2)]
        result = self.select(tags, pre, native, excluded=("current",))
        self.assertEqual([t["request_id"] for t in result["threats"]], ["threat"])
        self.assertEqual({t["request_id"] for t in result["eligible"]},
                         {"threat", "earlier_wrong", "earlier_correct"})
        earlier = [row for row in result["rows"] if row["earlier_break"]]
        self.assertEqual(sum(row["N_direct_correct"] for row in earlier), 1)

    def test_native_wrong_tie_is_threat_and_eos_last_slot_is_eligible(self):
        t = tag("eos", slot=2)
        result = self.select([t], [observation(t, 2, valid=2)],
            [observation(t, 1, margin=0., valid=2)])
        self.assertTrue(result["rows"][0]["N_wrong_tie"])
        self.assertEqual(result["selected_U"], result["selected_T"])
        self.assertTrue(result["identical_sets"])

    def test_hashes_match_explicit_contract_and_do_not_use_input_order(self):
        tags = [tag(str(i), slot=1+i % 3) for i in range(12)]
        pre = [observation(t, 5) for t in tags]
        native = [observation(t, t["proposal_slot"]-1) for t in tags]
        result = self.select(tags, pre, native)
        self.assertEqual(result, self.select(tags[::-1], pre[::-1], native[::-1]))
        for row in result["rows"]:
            rid, start, j = core.tag_key(row)
            self.assertEqual(row["T_hash"], hashlib.sha256(f"trr-v1-T|8|{rid}|{start}|{j}".encode()).hexdigest())
            self.assertEqual(row["U_hash"], hashlib.sha256(f"trr-v1-U|8|{j}|{rid}|{start}".encode()).hexdigest())
        self.assertEqual(len(result["selected_T"]), 4)
        self.assertEqual(sorted(t["proposal_slot"] for t in result["selected_T"]),
                         sorted(t["proposal_slot"] for t in result["selected_U"]))

    def test_hash_collision_uses_explicit_identity_tiebreak(self):
        tags = [tag(str(i)) for i in reversed(range(6))]
        with mock.patch.object(core, "selection_hash", return_value="collision"):
            result = self.select(tags, [observation(t, 4) for t in tags],
                                  [observation(t, 2) for t in tags])
        self.assertEqual([t["request_id"] for t in result["selected_T"]], ["0", "1", "2", "3"])

    def test_empty_threats_return_direct_native_and_no_selection(self):
        t = tag("r")
        result = self.select([t], [observation(t, 4)], [observation(t, 5)])
        self.assertFalse(result["active"])
        self.assertEqual(result["selected_U"], [])

    def test_missing_and_inconsistent_native_observations_fail(self):
        t = tag("r")
        with self.assertRaises(ValueError):
            self.select([t], [observation(t, 4)], [])
        with self.assertRaises(ValueError):
            self.select([t], [observation(t, 4)], [observation(t, 2, correct=True)])


class LayoutTests(unittest.TestCase):
    def test_union_order_and_zero_layouts_are_shared_and_immutable(self):
        a, b = tag("z"), tag("a")
        layouts = [dict(layout_id=t["layout_id"], request_id=t["request_id"],
            reference_presentation_id=0, binding={"anchors": [1, 2]}) for t in (a, b)]
        original = deepcopy(layouts)
        result = core.padded_layouts(layouts, [a], [b])
        self.assertEqual([row["request_id"] for row in result], ["a", "z"])
        self.assertEqual(result[0]["selected_U"], [])
        self.assertEqual(result[1]["selected_T"], [])
        result[0]["binding"]["anchors"][0] = 99
        self.assertEqual(layouts, original)

    def test_overlap_is_one_layout_and_nonmatched_positions_fail(self):
        t = tag("r")
        layouts = [dict(layout_id="r:0", request_id="r", reference_presentation_id=0)]
        self.assertEqual(len(core.padded_layouts(layouts, [t], [t])), 1)
        with self.assertRaises(ValueError):
            core.padded_layouts(layouts, [t], [tag("r", slot=2)])


if __name__ == "__main__":
    unittest.main()
