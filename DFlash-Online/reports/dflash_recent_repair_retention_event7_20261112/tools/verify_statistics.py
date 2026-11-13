#!/usr/bin/env python3
"""Independent CPU verification of the already specified report statistics.

No experiment modules are imported. Bootstrap draws are reduced as request
multiplicities, rather than indexing and summing the production arrays. This
checks published arithmetic; native validity remains a separate raw-record audit.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS=("NATIVE","RECENT","CONTROL")
DRAW_COUNT=10000
SEED=1707
COUNT_FIELDS=("verification_calls","accepted_draft_tokens","output_tokens","committed_progress")
TIME_FIELDS=("request_latency_s","native_generate_s","cache_maintenance_s")
LOCAL_FIELDS=("exact_slot_relapse","earlier_break","accepts_through","direct_canonical_agreement",
              "passes_parent_first_blocker","acquired_prefix_occurrences","lost_prefix_occurrences",
              "net_prefix_occurrences")


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def scalar(text):
    if text in (None,""):return None
    if text in (True,"True","true"):return True
    if text in (False,"False","false"):return False
    try:return int(text)
    except (ValueError,TypeError):return float(text)


def ratio(numerator,denominator):
    return numerator/denominator if denominator else math.nan


def interval(values):
    finite=np.sort(np.asarray([value for value in values if math.isfinite(value)],dtype=np.float64))
    # Explicit linear quantiles are independent of numpy.quantile's reduction.
    def percentile(p):
        if not len(finite):return None
        position=(len(finite)-1)*p
        low=int(math.floor(position));high=int(math.ceil(position))
        return float(finite[low]+(finite[high]-finite[low])*(position-low))
    return dict(lower=percentile(.025),upper=percentile(.975),valid_draws=len(finite),
                undefined_draws=len(values)-len(finite))


def multiplicities(count,draws=DRAW_COUNT):
    rng=np.random.Generator(np.random.PCG64(SEED))
    indices=rng.integers(0,count,size=(draws,count)) if count else np.empty((draws,0),dtype=int)
    return [np.bincount(row,minlength=count) for row in indices]


class Check:
    def __init__(self):self.count=0;self.failures=[]
    def equal(self,observed,expected,label):
        self.count+=1
        if isinstance(expected,(float,np.floating)):
            good=observed is not None and math.isfinite(float(observed)) and math.isclose(
                float(observed),float(expected),rel_tol=1e-12,abs_tol=1e-12)
        else:good=observed==expected
        if not good:self.failures.append(dict(field=label,observed=observed,expected=expected))
    def nested(self,observed,expected,label):
        if isinstance(expected,dict):
            if not isinstance(observed,dict):
                self.equal(observed,"required mapping",label)
                return
            for key,value in expected.items():self.nested(observed.get(key),value,label+"."+key)
        else:self.equal(observed,expected,label)


def paired_table(report,check):
    panel=csv_rows(report/"panel.csv")
    ids=[row["request_id"] for row in panel]
    check.equal(len(set(ids)),len(ids),"panel.unique")
    check.equal([int(row["panel_ordinal"]) for row in panel],list(range(len(ids))),"panel.sealed_order")
    saved=csv_rows(report/"per_request.csv")
    repeated=csv_rows(report/"all_repeats.csv")
    index={(row["request_id"],row["method"],int(row["repeat"])):row for row in repeated}
    check.equal(len(index),len(repeated),"repeats.unique")
    check.equal(len(index),6*len(ids),"repeats.full_population")
    result=[]
    for ordinal,rid in enumerate(ids):
        for arm in ARMS:
            values=[index[rid,arm,repeated_index] for repeated_index in (0,1)]
            for row in values:
                check.equal(scalar(row["valid_measurement"]),True,f"{rid}.{arm}.valid")
                check.equal(int(row["panel_ordinal"]),ordinal,f"{rid}.{arm}.ordinal")
                if "recorded_error_count" in row:check.equal(int(row["recorded_error_count"]),0,f"{rid}.{arm}.errors")
                if "raw_audit_passed" in row:check.equal(scalar(row["raw_audit_passed"]),True,f"{rid}.{arm}.raw_audit")
            for key in (*COUNT_FIELDS,"finish_reason"):
                check.equal(values[0][key],values[1][key],f"{rid}.{arm}.repeat.{key}")
            result.append(dict(request_id=rid,method=arm,**{key:int(values[0][key]) for key in COUNT_FIELDS},
                finish_reason=values[0]["finish_reason"],
                **{key:math.fsum(float(row[key]) for row in values)/2 for key in TIME_FIELDS}))
    index={(row["request_id"],row["method"]):row for row in saved}
    check.equal(len(index),len(saved),"paired.unique")
    check.equal(len(index),3*len(ids),"paired.full_population")
    for row in result:
        observed=index[row["request_id"],row["method"]]
        for key,value in row.items():
            check.equal(observed.get(key) if isinstance(value,str) else scalar(observed.get(key)),
                        value,f"paired.{row['request_id']}.{row['method']}.{key}")
    return result,ids


def primary(rows,other,draws=DRAW_COUNT):
    own={row["request_id"]:row for row in rows if row["method"]=="RECENT"}
    ref={row["request_id"]:row for row in rows if row["method"]==other}
    if not own or own.keys()!=ref.keys():raise ValueError("Unpaired primary population")
    # This deterministic index is the pre-existing production policy. The sealed
    # generation order is separately checked above and is never changed.
    ids=sorted(own)
    matrix=np.asarray([[own[r]["verification_calls"],ref[r]["verification_calls"],
        own[r]["accepted_draft_tokens"],ref[r]["accepted_draft_tokens"],
        own[r]["request_latency_s"],ref[r]["request_latency_s"]] for r in ids],dtype=np.float64)
    ca,cb,aa,ab,ta,tb=[math.fsum(column) for column in matrix.T]
    boot={"accepted_length_interval":[],"call_reduction_interval":[],"time_reduction_interval":[]}
    for weights in multiplicities(len(ids),draws):
        c,d,a,b,t,u=np.matmul(weights,matrix)
        boot["accepted_length_interval"].append(100*(ratio(ratio(a,c),ratio(b,d))-1))
        boot["call_reduction_interval"].append(100*(1-ratio(c,d)))
        boot["time_reduction_interval"].append(100*(1-ratio(t,u)))
    def finite(value):return value if math.isfinite(value) else None
    return dict(reference=other,requests=len(ids),recent_calls=int(ca),reference_calls=int(cb),
        recent_accepted=int(aa),reference_accepted=int(ab),recent_accepted_per_call=finite(ratio(aa,ca)),
        reference_accepted_per_call=finite(ratio(ab,cb)),
        accepted_length_relative_gain_pct=finite(100*(ratio(ratio(aa,ca),ratio(ab,cb))-1)),
        call_reduction_pct=finite(100*(1-ratio(ca,cb))),
        gross_calls_saved=sum(max(0,ref[r]["verification_calls"]-own[r]["verification_calls"]) for r in ids),
        gross_calls_added=sum(max(0,own[r]["verification_calls"]-ref[r]["verification_calls"]) for r in ids),
        requests_improved=sum(own[r]["verification_calls"]<ref[r]["verification_calls"] for r in ids),
        requests_regressed=sum(own[r]["verification_calls"]>ref[r]["verification_calls"] for r in ids),
        requests_tied=sum(own[r]["verification_calls"]==ref[r]["verification_calls"] for r in ids),
        recent_request_s=ta,reference_request_s=tb,speedup=finite(ratio(tb,ta)),
        time_reduction_pct=finite(100*(1-ratio(ta,tb))),**{key:interval(value) for key,value in boot.items()})


def fixed_starts(report,request_ids,check,draws=DRAW_COUNT):
    starts=csv_rows(report/"fixed_start_origins.csv")
    outcomes=csv_rows(report/"fixed_start_outcomes.csv")
    ids=sorted(request_ids)
    check.equal(len(ids),len(set(ids)),"fixed.sealed_unique_requests")
    origin={(row["request_id"],int(row["canonical_start"])):row for row in starts}
    index={(row["request_id"],int(row["canonical_start"]),row["method"]):row for row in outcomes}
    check.equal(len(origin),len(starts),"fixed.unique_starts")
    check.equal(len(index),len(outcomes),"fixed.unique_outcomes")
    check.equal(len(index),3*len(origin),"fixed.full_outcome_population")
    check.equal(set(rid for rid,_ in origin)<=set(ids),True,"fixed.sealed_membership")
    aggregates={(rid,arm):Counter() for rid in ids for arm in ARMS}
    for (rid,start),source in origin.items():
        repair=scalar(source["repair_slot"]) is not None
        blocker=bool(scalar(source["genuine_k6_rejection"]))
        for arm in ARMS:
            outcome=index[rid,start,arm]
            check.equal(scalar(outcome["valid"]),True,f"fixed.{rid}.{start}.{arm}.valid")
            values=aggregates[rid,arm]
            values.update(starts=1,repairs=int(repair),blockers=int(blocker))
            for key in LOCAL_FIELDS:
                value=scalar(outcome[key])
                if value is not None:values[key]+=int(value)
            # An earlier break is a counted repair outcome, not a dropped case.
            if repair:
                check.equal(sum(bool(scalar(outcome[k])) for k in
                    ("earlier_break","exact_slot_relapse","accepts_through")),1,
                    f"fixed.{rid}.{start}.{arm}.repair_partition")
    totals={arm:{key:sum(aggregates[rid,arm][key] for rid in ids)
                     for key in ("starts","repairs","blockers",*LOCAL_FIELDS)} for arm in ARMS}
    weights=multiplicities(len(ids),draws)
    contrasts={}
    for other in ("NATIVE","CONTROL"):
        contrast={}
        for field in LOCAL_FIELDS:
            denominator="repairs" if field in LOCAL_FIELDS[:4] else "blockers" if field==LOCAL_FIELDS[4] else "starts"
            data=np.asarray([[aggregates[rid,"RECENT"][field]-aggregates[rid,other][field],
                              aggregates[rid,"RECENT"][denominator]] for rid in ids],dtype=float).reshape(len(ids),2)
            point=ratio(math.fsum(data[:,0]),math.fsum(data[:,1]))
            estimates=[ratio(*np.matmul(w,data)) for w in weights]
            contrast[field]=dict(difference_per_eligible_start=point if math.isfinite(point) else None,
                                 interval=interval(estimates),denominator=int(math.fsum(data[:,1])))
        contrasts[other]=contrast
    return dict(totals=totals,contrasts=contrasts)


def verify(report,plan_path,output):
    report=Path(report);plan=read(plan_path);summary=read(report/"SUMMARY.json")
    result=dict(schema="retention-independent-statistics-v1",status="not_ready",model_operations=0,
        bootstrap_draws=DRAW_COUNT,seed=SEED,bit_generator="PCG64",percentile="linear",
        bootstrap_index="Lexically sorted request IDs, matching the pre-existing fixed implementation; sealed generation order checked separately.",
        uncertainty_scope="Whole-request conditional uncertainty for the fixed trained checkpoints, not training-seed uncertainty.",
        authorship="Independent arithmetic implementation; same-team author also contributed objective/report code; no independent native replication.",
        execution_commit=summary.get("evaluation_execution_commit",summary["execution_commit"]),
        training_execution_commit=summary.get("training_execution_commit",summary["execution_commit"]))
    if not summary.get("complete_valid_comparison"):
        result["reason"]="No complete valid primary comparison is available; no intervals were invented."
    else:
        check=Check()
        try:
            paired,panel_ids=paired_table(report,check)
            check.equal(len(panel_ids),summary["panel_requests"],"panel.summary_count")
            contrasts={other:primary(paired,other) for other in ("NATIVE","CONTROL")}
            for other,value in contrasts.items():check.nested(summary["comparisons"][other],value,"primary."+other)
            paths=plan["path_request_ids"]
            check.equal(set(paths)<=set(panel_ids),True,"fixed.subset_of_sealed_panel")
            local=fixed_starts(report,paths,check)
            for key in ("totals","contrasts"):check.nested(summary["fixed_starts"][key],local[key],"fixed."+key)
            result.update(status="passed" if not check.failures else "failed",checks=check.count,
                failures=check.failures,primary=contrasts,fixed_starts=local,intervals_checked=22)
        except (ValueError,KeyError,TypeError,IndexError) as error:
            result.update(status="failed",checks=check.count,failures=check.failures,
                          malformed_evidence=type(error).__name__)
    sources=[report/name for name in ("SUMMARY.json","per_request.csv","all_repeats.csv","panel.csv",
        "fixed_start_origins.csv","fixed_start_outcomes.csv")]+[Path(plan_path)]
    result["source_identities"]={path.name:dict(bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources if path.exists()}
    Path(output).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("--plan",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=verify(args.report,args.plan,args.output)
    print(json.dumps({key:result[key] for key in ("status","execution_commit","model_operations")}))
    raise SystemExit(1 if result["status"]=="failed" else 0)
