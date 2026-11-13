from copy import deepcopy
import csv
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
import random
import unittest

from experiments.recent_repair_retention.preparation import (
    canonical_start, candidate_slots, join_parent_occurrences, recent_slots,
    verify_pr42_tag_sanity,
)
from experiments.recent_repair_retention.matching import (
    _hash_cost, match_control, validate_assignment_equality,
)


def start(offset=0, a5=1, a6=4, **changes):
    row = dict(request_id="r",canonical_start=offset,a5=a5,a6=a6,
               valid_proposals=15,parity_valid=True,genuine_k5_rejection=True)
    row.update(changes)
    return row


def occurrence(offset, margin, *, pid=0,j=2,request="r",weight=0.875,a6=4):
    return dict(request_id=request,canonical_start=offset,proposal_slot=j,j=j,
        presentation_id=pid,pass_index=pid,anchor_ordinal=offset,canonical_token=123,
        margin=margin,weight=weight,floor=.5*min(margin,1),a5=1,a6=a6,
        correct_prefix=4,valid=True,native_correct=True,label_match=True)


class TagTests(unittest.TestCase):
    def test_selects_only_first_old_error_not_public_H_or_j(self):
        tags=recent_slots([start(a5=6,a6=10,H=False,j=11)])
        self.assertEqual([r["proposal_slot"] for r in tags],[7])

    def test_seed_cap_eos_and_parity(self):
        rows=[start(0,0,1),start(1,2,3,genuine_k5_rejection=False),
              start(2,2,3,parity_valid=False),start(3,2,2),start(4,2,1)]
        self.assertEqual([(r["canonical_start"],r["proposal_slot"]) for r in recent_slots(rows)],[(0,1)])

    def test_csv_false_is_not_true(self):
        self.assertEqual(recent_slots([start(parity_valid="False")]),[])

    def test_offset_and_overlapping_identity(self):
        self.assertEqual(canonical_start(110,100),10)
        self.assertNotEqual(canonical_start(111,100),10)
        with self.assertRaises(ValueError): canonical_start(99,100)

    def test_established_requires_same_depth_native_acceptance(self):
        rows=[start(0,1,4),start(1,4,4),start(2,4,5),start(3,1,4)]
        c=candidate_slots(rows)
        self.assertEqual([(r["canonical_start"],r["proposal_slot"]) for r in c["established"]],[(1,2)])

    def test_retained_828_sanity(self):
        path=Path(__file__).resolve().parents[1]/"reports/dflash_lost_repair_20261114/ALL_STARTS.csv"
        with path.open() as handle: rows=list(csv.DictReader(handle))
        self.assertEqual(verify_pr42_tag_sanity(rows)["recent_slots"],828)

    def test_eligibility_is_per_presentation(self):
        weights=[dict(presentation_id=0,pass_index=0,request_id="r",canonical_start=0,
                      anchor_ordinal=0,correct_prefix=4,valid_proposals=15),
                 dict(presentation_id=1,pass_index=1,request_id="r",canonical_start=0,
                      anchor_ordinal=0,correct_prefix=1,valid_proposals=15)]
        result=join_parent_occurrences([start()],weights,[occurrence(0,2.4)])
        self.assertEqual(len(result["recent"]),1)
        self.assertEqual(result["recent"][0]["floor"],.5)
        self.assertEqual(result["exclusion_counts"],{"frozen_training_prefix_disagreement":1})

    def test_missing_or_conflicting_measurement_is_failure_not_exclusion(self):
        w=[dict(presentation_id=0,pass_index=0,request_id="r",canonical_start=0,
                anchor_ordinal=0,correct_prefix=4,valid_proposals=15)]
        with self.assertRaisesRegex(ValueError,"Missing required"):
            join_parent_occurrences([start()],w,[])
        bad=occurrence(0,.5);bad["label_match"]=False
        with self.assertRaisesRegex(ValueError,"canonical token"):
            join_parent_occurrences([start()],w,[bad])


class MatchingTests(unittest.TestCase):
    def test_maximum_cardinality_over_greedy_distance(self):
        # Source .25 can use either; source0 can only use control.125.
        recent=[occurrence(0,.25),occurrence(1,0.0625)]
        controls=[occurrence(2,.125),occurrence(3,.5)]
        result=match_control(recent,controls)
        self.assertEqual(result["summary"]["matched_distinct_slots"],2)
        self.assertEqual({r["source_start"]:r["destination_start"] for r in result["matches"]},{0:3,1:2})

    def test_minimum_total_distance_and_no_replacement(self):
        recent=[occurrence(0,.5),occurrence(1,.625)]
        controls=[occurrence(2,.5),occurrence(3,.75)]
        result=match_control(recent,controls)
        self.assertEqual({r["source_start"]:r["destination_start"] for r in result["matches"]},{0:2,1:3})
        self.assertEqual(result["summary"]["maximum_destination_reuse"],1)

    def test_two_pass_caliper_and_eligibility_must_both_match(self):
        recent=[occurrence(0,.5,pid=0),occurrence(0,.75,pid=1)]
        controls=[occurrence(1,.5,pid=0),occurrence(1,.4,pid=1),
                  occurrence(2,.5,pid=0)]
        result=match_control(recent,controls)
        self.assertTrue(result["summary"]["control_identical"])
        self.assertEqual([r["canonical_start"] for r in result["arms"]["CONTROL"]],[0,0])

    def test_zero_matches_only_zero_and_constant_smd_zero(self):
        result=match_control([occurrence(0,0)], [occurrence(1,.01),occurrence(2,0)])
        self.assertEqual(result["matches"][0]["destination_start"],2)
        self.assertEqual(result["summary"]["absolute_standardized_margin_difference"],0)

    def test_recent_destinations_excluded_and_residual_preserved(self):
        recent=[occurrence(0,.5),occurrence(1,2)]
        result=match_control(recent,[occurrence(0,.5),occurrence(2,.5)])
        self.assertEqual(len(result["arms"]["CONTROL"]),2)
        self.assertEqual(result["summary"]["shared_residual_slots"],1)
        self.assertEqual([r["canonical_start"] for r in result["arms"]["CONTROL"]],[1,2])

    def test_different_slots_can_use_same_block(self):
        recent=[occurrence(0,.5,j=1),occurrence(1,.5,j=2)]
        controls=[occurrence(2,.5,j=1),occurrence(2,.5,j=2)]
        self.assertEqual(match_control(recent,controls)["summary"]["matched_distinct_slots"],2)

    def test_ties_stable_under_input_reorder(self):
        recent=[occurrence(i,.5) for i in range(4)]
        controls=[occurrence(i,.5) for i in range(4,8)]
        first=match_control(recent,controls)["matches"]
        random.Random(17).shuffle(recent);random.Random(19).shuffle(controls)
        self.assertEqual(match_control(recent,controls)["matches"],first)

    def test_exact_per_presentation_depth_weights(self):
        recent=[occurrence(0,.5,j=1),occurrence(1,.5,j=2)]
        bad=deepcopy(recent);bad[0]["weight"]+=.001
        with self.assertRaisesRegex(ValueError,"count or raw weight"):
            validate_assignment_equality(recent,bad)

    def test_no_tags_is_explicit_infeasibility(self):
        result=match_control([],[])
        self.assertFalse(result["summary"]["training_feasible"])
        self.assertTrue(result["summary"]["control_identical"])

    def test_matching_agrees_with_independent_exhaustive_small_graphs(self):
        rng=random.Random(1707)
        for case in range(24):
            recent=[occurrence(i,rng.randrange(1,10)/16) for i in range(3)]
            controls=[occurrence(i+3,rng.randrange(1,10)/16) for i in range(3)]
            options=[]
            for n in range(4):
                for left in combinations(range(3),n):
                    for right in permutations(range(3),n):
                        distances=[abs(Fraction(recent[a]["margin"])-Fraction(controls[b]["margin"]))
                                   for a,b in zip(left,right)]
                        if any(d>Fraction(1,4) for d in distances):continue
                        tie=sum(_hash_cost(("r",a,2),("r",b+3,2)) for a,b in zip(left,right))
                        options.append((-n,sum(distances,Fraction(0)),tie))
            result=match_control(recent,controls)
            objective=(-len(result["matches"]),
                sum((Fraction(r["total_margin_distance"]) for r in result["matches"]),Fraction(0)),
                sum(int(r["tie_sha256"],16) for r in result["matches"]))
            self.assertEqual(objective,min(options),f"graph {case}")


if __name__=="__main__":unittest.main()
