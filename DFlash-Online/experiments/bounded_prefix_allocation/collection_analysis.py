"""Secondary C7/N7/S7 collection counts; never a collection-latency comparison.

Inputs are the already necessary C7 feedback generations and authenticated old
N7/S7 records. This reducer makes no model call and never changes the endpoint
screen's decision. Callers supply authenticated source/checkpoint provenance.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np

from experiments.pr49_collection_cpu.reducer import Checks, interval, validate_record

ARMS = ("N", "C", "S")
CONTRASTS = (("C", "N"), ("C", "S"), ("S", "N"))
SESSIONS = dict(N="persistent_N_event8", C="bounded_prefix_allocation_C_event8", S="persistent_S_event8")


def _statistics(rows):
    ordered = {a: sorted((r for r in rows if r["arm"] == a), key=lambda r: r["source_id"]) for a in ARMS}
    draws = np.random.Generator(np.random.PCG64(1708)).integers(0, 128, size=(10000, 128), dtype=np.int64)
    totals, samples = {}, {}
    for arm in ARMS:
        values = ordered[arm]
        counts = np.array([[r[k] for k in ("V", "A", "P", "M")] for r in values], dtype=np.int64)
        samples[arm] = counts[draws].sum(axis=1)
        v, a, p, m = map(int, counts.sum(axis=0))
        totals[arm] = dict(requests=128, V=v, A=a, P=p, M=m,
            output_tokens=sum(r["output_tokens"] for r in values),
            eos=sum(r["finish_reason"] == "eos" for r in values),
            cap_hits=sum(r["cap_hit"] for r in values), zero_calls=sum(r["zero_calls"] for r in values),
            native_acceptance=a/v if v else None, progress_per_call=p/v if v else None)
    comparisons = {}
    for candidate, reference in CONTRASTS:
        c, r = totals[candidate], totals[reference]
        saved = np.array([y["V"]-x["V"] for x, y in zip(ordered[candidate], ordered[reference])])
        with np.errstate(divide="ignore", invalid="ignore"):
            ca = samples[candidate][:, 1] / samples[candidate][:, 0]
            ra = samples[reference][:, 1] / samples[reference][:, 0]
            distribution = dict(call_reduction=1-samples[candidate][:, 0]/samples[reference][:, 0],
                acceptance_improvement=ca/ra-1, acceptance_difference=ca-ra)
        c_accept, r_accept = c["native_acceptance"], r["native_acceptance"]
        points = dict(call_reduction=1-c["V"]/r["V"] if r["V"] else None,
            acceptance_improvement=c_accept/r_accept-1 if c_accept is not None and r_accept else None,
            acceptance_difference=c_accept-r_accept if c_accept is not None and r_accept is not None else None)
        comparisons[candidate+"_vs_"+reference] = dict(
            **{key:dict(point=points[key], **interval(value)) for key,value in distribution.items()},
            requests_fewer_calls=int((saved>0).sum()),requests_more_calls=int((saved<0).sum()),
            request_call_ties=int((saved==0).sum()),gross_calls_saved=int(saved.clip(min=0).sum()),
            gross_calls_added=int((-saved).clip(min=0).sum()),net_calls_saved=int(saved.sum()))
    return dict(totals=totals, comparisons=comparisons,
        bootstrap=dict(seed=1708,generator="PCG64",draws=10000,unit="whole paired request",
            order="lexicographic source_id",quantile="linear",shared_draws=True,
            draw_matrix_sha256=hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()))


def analyze(requests, records, captures, checkpoints, provenance):
    """Validate all128 existing trajectories; retain every invalid/missing row.

    records maps each arm to a list, so duplicates remain detectable. captures
    maps each arm/request_id to the independently persisted native capture JSON.
    checkpoints maps each arm to its authenticated export and weights_identity.
    The supplied provenance is a source audit, not merely candidate agreement.
    """
    checks, rows, pairs = Checks(), [], []
    ids = [r["request_id"] for r in requests]
    checks.require(len(requests)==128 and len(set(ids))==128, "fixed128_request_ids")
    checks.require(len({r["source_id"] for r in requests})==128, "fixed128_source_ids")
    checks.require(len({tuple(r["prompt_ids"]) for r in requests})==128, "fixed128_distinct_prompts")
    checks.require(set(records)==set(ARMS) and set(captures)==set(ARMS) and set(checkpoints)==set(ARMS), "three_declared_arms")
    checks.require(provenance.get("status")=="passed", "source_provenance_passed")
    checks.require(provenance.get("checkpoints")==checkpoints, "checkpoint_provenance_binding")
    checks.require(bool(provenance.get("consumed_inputs")), "source_provenance_inputs_present")
    for arm in ARMS:
        endpoint=checkpoints.get(arm,{})
        weight=endpoint.get("weights_identity",{})
        checks.require(bool(endpoint.get("export")) and type(weight.get("bytes")) is int and weight["bytes"]>0
            and isinstance(weight.get("sha256"),str) and len(weight["sha256"])==64
            and all(c in "0123456789abcdef" for c in weight["sha256"]), arm+":checkpoint_identity")
    by_arm={}
    for arm in ARMS:
        values=records.get(arm,[])
        actual=[row.get("record_id") for row in values]
        checks.require(len(actual)==128 and Counter(actual)==Counter(ids),arm+":complete_population")
        by_arm[arm]={}
        for record in values:
            by_arm[arm].setdefault(record.get("record_id"),[]).append(record)
    for request in requests:
        pair=dict(request_id=request["request_id"],source_id=request["source_id"],valid=True)
        current={}
        for arm in ARMS:
            before=len(checks.errors)
            found=by_arm[arm].get(request["request_id"],[])
            if len(found)!=1:
                rows.append(dict(request_id=request["request_id"],source_id=request["source_id"],arm=arm,
                    valid=False,unavailable=not found,duplicate=len(found)>1))
                pair["valid"]=False
                continue
            record=found[0]
            try:
                checks.require(record["session_id"]==SESSIONS[arm],arm+":"+request["request_id"]+":actual_session")
                # The old reader's one hardcoded metadata expectation is checked
                # above under the actual new protocol before this explicit view.
                view=dict(record,session_id=f"persistent_{arm}_event8")
                row=validate_record(view,request,arm,checkpoints[arm]["export"],
                    captures.get(arm,{}).get(request["request_id"]),checks)
                if row is None:raise ValueError("No usable native count row")
                row["valid"]=row["valid"] and len(checks.errors)==before
                rows.append(row);current[arm]=record
                pair["valid"]=pair["valid"] and row["valid"]
                for key in ("V","A","P","M","output_tokens","finish_reason"):
                    pair[arm+"_"+key]=row[key]
            except (KeyError,TypeError,ValueError,IndexError) as error:
                checks.require(False,arm+":"+request["request_id"]+":malformed:"+type(error).__name__)
                rows.append(dict(request_id=request["request_id"],source_id=request["source_id"],arm=arm,valid=False))
                pair["valid"]=False
        if set(current)==set(ARMS):
            for arm in ("C","S"):
                for key in ("prompt_ids","response_ids","finish_reason"):
                    good=current[arm][key]==current["N"][key]
                    checks.require(good,request["request_id"]+":"+arm+":paired_"+key)
                    pair["valid"]=pair["valid"] and good
        else:
            pair["valid"]=False
        pairs.append(pair)
    for arm in ARMS:
        for record in records.get(arm,[]):
            if record.get("record_id") not in ids:
                rows.append(dict(request_id=record.get("record_id"),source_id=None,arm=arm,valid=False,unbound=True))
    checks.require(len(rows)==384 and len(pairs)==128 and all(x["valid"] for x in pairs), "complete_valid_secondary")
    result=dict(status="completed" if not checks.errors else "INVALID_OR_INCOMPLETE_SECONDARY",
        scope="SECONDARY_CAPTURE_ENABLED_C7_COLLECTION_COUNTS",latency_excluded=True,
        primary_endpoint_decision_affected=False,new_model_work=0,request_count=len(requests),
        validation=dict(checks=checks.count,errors=checks.errors,valid_pairs=sum(p["valid"] for p in pairs),
            expected_pairs=128,record_counts={a:len(records.get(a,[])) for a in ARMS}),
        per_request=rows,paired_requests=pairs,checkpoints=checkpoints,
        interpretation=["One validated native trajectory per request; no collection latency comparison.",
            "This population differs from the final panel; differences between panels are not a learning curve.",
            "Invalid secondary evidence cannot alter the independent final endpoint screen.",
            "Call reduction and native acceptance are linked views, not independent evidence."])
    if not checks.errors:result.update(_statistics(rows))
    return result
