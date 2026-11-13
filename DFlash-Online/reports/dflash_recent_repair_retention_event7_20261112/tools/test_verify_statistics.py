import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import verify_statistics as V


def table(path,rows,fields=()):
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def native_rows():
    rows=[]
    for rid in ("z","a"):
        for arm in V.ARMS:
            good=arm=="RECENT"
            rows.append(dict(request_id=rid,method=arm,verification_calls=8 if good else 10,
                accepted_draft_tokens=22 if good else 20,output_tokens=30,committed_progress=29,
                finish_reason="eos",request_latency_s=4. if good else 5.,
                native_generate_s=3. if good else 4.,cache_maintenance_s=1.))
    return rows


class StatisticsVerifierTests(unittest.TestCase):
    def test_degenerate_bootstrap_and_ratio_of_sums(self):
        result=V.primary(native_rows(),"NATIVE",draws=100)
        self.assertEqual(result["recent_calls"],16)
        self.assertEqual(result["recent_accepted"],44)
        self.assertEqual(result["accepted_length_relative_gain_pct"],37.5)
        self.assertEqual(result["accepted_length_interval"]["lower"],37.5)
        self.assertEqual(result["accepted_length_interval"]["upper"],37.5)
        self.assertAlmostEqual(result["time_reduction_interval"]["lower"],20)
        changed=native_rows()
        next(row for row in changed if row["request_id"]=="z" and row["method"]=="RECENT")["verification_calls"]=80
        result=V.primary(changed,"NATIVE",draws=20)
        self.assertEqual(result["recent_accepted_per_call"],44/88)

    def test_linear_quantile_and_nonfinite_draw_accounting(self):
        result=V.interval([0.,10.,float("nan")])
        self.assertEqual(result,dict(lower=.25,upper=9.75,valid_draws=2,undefined_draws=1))
        self.assertEqual(V.interval([float("nan")]*3)["lower"],None)
        zero=native_rows()
        for row in zero:row["accepted_draft_tokens"]=0
        result=V.primary(zero,"CONTROL",draws=17)
        self.assertEqual(result["accepted_length_interval"]["undefined_draws"],17)

    def test_seeded_multiplicity_matches_prespecified_draws(self):
        indices=np.random.Generator(np.random.PCG64(1707)).integers(0,3,size=(9,3))
        for actual,expected in zip(V.multiplicities(3,9),indices):
            self.assertEqual(actual.tolist(),np.bincount(expected,minlength=3).tolist())

    def test_repeat_validation_preserves_panel_order_without_treating_repeats_as_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            table(root/"panel.csv",[dict(request_id="z",panel_ordinal=0),dict(request_id="a",panel_ordinal=1)])
            rows=native_rows();table(root/"per_request.csv",rows)
            repeats=[]
            for row in rows:
                for repeat in (0,1):
                    repeats.append(dict(row,repeat=repeat,panel_ordinal=0 if row["request_id"]=="z" else 1,
                        valid_measurement=True,recorded_error_count=0,raw_audit_passed=True))
            table(root/"all_repeats.csv",repeats)
            check=V.Check();actual,order=V.paired_table(root,check)
            self.assertEqual(order,["z","a"])
            self.assertEqual(check.failures,[])
            self.assertEqual(V.primary(actual,"NATIVE",draws=10)["recent_calls"],16)
            repeats[0]["verification_calls"]=999
            table(root/"all_repeats.csv",repeats)
            check=V.Check();V.paired_table(root,check)
            self.assertTrue(check.failures)

    def test_fixed_request_clusters_include_no_start_requests_and_earlier_breaks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            table(root/"fixed_start_origins.csv",[dict(request_id="with",canonical_start=0,repair_slot=2,genuine_k6_rejection=True)])
            outputs=[]
            for arm in V.ARMS:
                r=dict(request_id="with",canonical_start=0,method=arm,valid=True,
                    exact_slot_relapse=False,earlier_break=arm=="RECENT",accepts_through=arm!="RECENT",
                    direct_canonical_agreement=arm!="RECENT",passes_parent_first_blocker=False,
                    acquired_prefix_occurrences=0,lost_prefix_occurrences=2 if arm=="RECENT" else 0,
                    net_prefix_occurrences=-2 if arm=="RECENT" else 0)
                outputs.append(r)
            table(root/"fixed_start_outcomes.csv",outputs)
            check=V.Check();r=V.fixed_starts(root,["empty","with"],check,draws=100)
            self.assertEqual(check.failures,[])
            self.assertEqual(r["totals"]["RECENT"]["repairs"],1)
            self.assertEqual(r["contrasts"]["NATIVE"]["earlier_break"]["difference_per_eligible_start"],1.)
            self.assertGreater(r["contrasts"]["NATIVE"]["earlier_break"]["interval"]["undefined_draws"],0)
            self.assertEqual(r["contrasts"]["NATIVE"]["earlier_break"]["denominator"],1)

    def test_incomplete_report_never_invents_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"SUMMARY.json").write_text(json.dumps(dict(complete_valid_comparison=False,execution_commit="a"*40)))
            (root/"PLAN.json").write_text(json.dumps(dict(path_request_ids=[])))
            result=V.verify(root,root/"PLAN.json",root/"AUDIT.json")
            self.assertEqual(result["status"],"not_ready")
            self.assertNotIn("primary",result)


if __name__=="__main__":unittest.main()
