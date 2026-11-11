"""PR40 parent starts, freshly reconstructed for both step-640 children.

CPU preparation authenticates only the parent census/history. ``run`` is an
explicit native execution entry point: the caller must close its other model
workers and pass the remaining portion of the campaign's 4,500-second budget.
The supervisor includes loading/publication/reconstruction/cleanup in that clock.
No historical step-768 child result is an input to this diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.verification_transition.data import identity, read
from experiments.verification_transition.budget import (
    WorkBudget, BudgetService, BudgetExceeded, connect_budget,
)
from experiments.verification_transition.run import active_owned_processes, stop_owned_groups
from experiments.longer_run_headroom.census import digest, parent_rows, validate_walk
from experiments.longer_run_headroom.pairs import neutral_parity, score
from experiments.longer_run_headroom.run import counted

ARMS = ("NATIVE", "SECOND")
GROUPS = ("SHORT", "LONG")
CENSUS_WEIGHTS = {"SHORT": 263 / 2697, "LONG": 403 / 2697}
NATIVE_SHA256 = "9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795"
PARENT_SHA256 = "4f3010e961bd7dcf35fa100486c355ffbc7629d816dda6e2b8fa001878255701"
INPUT_SHA256 = {
    "census_seal": "8e587d47d21ef0829354f54eb67496405cf194b41c2e87f6a2c1f838320553aa",
    "bound_seal": "2bec0d70819605e89cc7e3dfd0427452d4d15fd8106f50d50bd51e3f71c99644",
    "retained_manifest": "2d6e3275ed367bb4f74bbf9b6a328f2821b12f3c02dd0a416b66980c5ff27221",
}
LIMITS = dict(original_completion=57, neutral_block=128, child_block=256,
              draft_block=3000, model_load=172)
STARTUP_BLOCK_LIMIT = 64
CLEANUP_RESERVE_S = 30
SCHEMA = "second_follower_matched_start_v1"


def _check_file(path, expected):
    actual = identity(path)
    if (actual != expected if isinstance(expected, dict) else actual["sha256"] != expected):
        raise ValueError("Authenticated input changed: " + str(path))
    return dict(path=str(path), **actual)


def prepare_inputs(census_dir, bound_dir, retained_dir):
    """Build the exact GSM subset; all outcomes begin pending, never PR40 CHILD.

    Paths may name a verified private mirror or the original remote artifacts.
    The fixed seal identities authenticate transport; the returned jobs embed
    parent records so no path guessing or later historical-child access is needed.
    """
    census, bound, retained = map(Path, (census_dir, bound_dir, retained_dir))
    checked = []
    for path, key in ((census / "SEAL.json", "census_seal"),
                      (bound / "SEAL.json", "bound_seal"),
                      (retained / "DELIVERY_MANIFEST.json", "retained_manifest")):
        checked.append(_check_file(path, INPUT_SHA256[key]))
    census_seal, bound_seal = read(census / "SEAL.json"), read(bound / "SEAL.json")
    for name, expected in census_seal["files"].items():
        checked.append(_check_file(census / name, expected))
    checked.append(_check_file(bound / "PLAN.json", bound_seal["plan"]))
    if bound_seal["parent_selection"] != identity(census / "SELECTED_STARTS.json"):
        raise ValueError("Bound plan names another parent selection")
    original = read(bound / "PLAN.json")
    selected = [r for r in read(census / "SELECTED_STARTS.json")["starts"]
                if r["dataset"] == "gsm8k"]
    inventory = read(retained / "DELIVERY_MANIFEST.json")["files"]
    parent_checkpoint = original["checkpoints"]["gsm8k"]["KPLUS4"]
    if (parent_checkpoint["cumulative_adam_steps"] != 512 or
            parent_checkpoint["weights_identity"]["sha256"] != PARENT_SHA256):
        raise ValueError("PR40 original KPLUS4 identity changed")
    jobs = []
    requests = [i for i in original["original_requests"] if i["dataset"] == "gsm8k"]
    for item in requests:
        starts = [s for s in selected if s["request_id"] == item["request_id"]]
        if not starts:
            continue
        prefix = f'replays/gsm8k/{item["selection_rank"]:03d}'
        names = [prefix + "/PARENT.json", prefix + "/PARENT_PARITY.json"]
        for name in names:
            checked.append(_check_file(retained / name, inventory[name]))
        parent, parity = (read(retained / name) for name in names)
        validate_walk(parent, item)
        blocks = parent["metadata"]["trace"]
        if (parent["checkpoint_id"] != parent_checkpoint["export"] or
                parity["status"] != "passed" or len(parity["blocks"]) != len(blocks)):
            raise ValueError("Retained parent provenance is incomplete")
        for block, proof in zip(blocks, parity["blocks"]):
            if (proof != neutral_parity(block, block)):
                raise ValueError("PR40 parent differs from authenticated PR34 original")
        recensus = {r["parent_round_ordinal"]: r for r in parent_rows(parent, item)}
        for start in starts:
            if {k: v for k, v in start.items() if k != "inclusion_fraction"} != recensus[start["parent_round_ordinal"]]:
                raise ValueError("Sealed start differs from original parent/canonical history")
        jobs.append(dict(item=item, starts=sorted(starts, key=lambda s: s["parent_round_ordinal"]),
                         parent=parent, expected_parent_rounds=len(blocks)))
    result = dict(schema=SCHEMA, parent_checkpoint=parent_checkpoint,
                  shared=original["shared"], selected=selected, jobs=jobs,
                  request_clusters=[i["request_id"] for i in requests],
                  development_requests=[dict(request_id=i["request_id"], prompt_ids=i["request"]["prompt_ids"], prompt_ids_sha256=digest(i["request"]["prompt_ids"])) for i in requests],
                  inputs=checked, limits=LIMITS, startup_block_limit=STARTUP_BLOCK_LIMIT,
                  census_weights=CENSUS_WEIGHTS, historical_child_outcomes_consumed=False,
                  work_plan=dict(parent_walks=len(jobs), parent_rounds=sum(j["expected_parent_rounds"] for j in jobs),
                                 neutral_blocks=128, child_blocks=256, expected_startup_blocks=51,
                                 engine_initializations=1, checkpoint_publications=171))
    result["binding_sha256"] = digest(result)
    validate_inputs(result)
    return result


def validate_inputs(inputs):
    content = {k: v for k, v in inputs.items() if k != "binding_sha256"}
    selected, jobs = inputs["selected"], inputs["jobs"]
    keys = lambda rows: [(r["request_id"], r["parent_round_ordinal"]) for r in rows]
    joined = [s for job in jobs for s in job["starts"]]
    if (inputs.get("schema") != SCHEMA or inputs.get("binding_sha256") != digest(content)
            or len(selected) != 128 or len(set(keys(selected))) != 128
            or Counter(r["group"] for r in selected) != {"SHORT": 64, "LONG": 64}
            or any(s["dataset"] != "gsm8k" or s["opening"] for s in selected)
            or len(jobs) != 57 or len(set(inputs["request_clusters"])) != 64
            or sorted(keys(joined)) != sorted(keys(selected))
            or any(s not in selected for s in joined)
            or inputs["limits"] != LIMITS or inputs["startup_block_limit"] != 64
            or inputs["census_weights"] != CENSUS_WEIGHTS
            or inputs["historical_child_outcomes_consumed"] is not False
            or sum(j["expected_parent_rounds"] for j in jobs) != 2534):
        raise ValueError("Require the unchanged 128-start PR40 GSM diagnostic")
    parent = inputs["parent_checkpoint"]
    if parent["cumulative_adam_steps"] != 512 or parent["weights_identity"]["sha256"] != PARENT_SHA256:
        raise ValueError("Require authenticated original step-512 parent")
    if not {s["request_id"] for s in selected} <= set(inputs["request_clusters"]):
        raise ValueError("Missing selected request cluster")
    return inputs


def verify_panel_disjoint(inputs, requests):
    """Check all64 development prompts, including7 without selected starts.

    Caller supplies the authenticated PR39 panel. Different ID namespaces alone
    are insufficient; the rendered token sequence identity is also compared.
    """
    development = inputs["development_requests"]
    if len(development) != 64 or len(requests) != 256:
        raise ValueError("Require complete64-development and256-endpoint panels")
    dev_hashes = {digest(r["prompt_ids"]) for r in development}
    panel_hashes = {digest(r["prompt_ids"]) if "prompt_ids" in r else r["prompt_ids_sha256"] for r in requests}
    ids = {r["request_id"] for r in development} & {r["request_id"] for r in requests}
    overlap = dev_hashes & panel_hashes
    if ids or overlap:
        raise ValueError("Endpoint overlaps PR40 development requests")
    return dict(status="passed",development_requests=64,endpoint_requests=256,
                request_id_overlap=0,rendered_prompt_token_identity_overlap=0,
                scope="Exact token-list identity under the inherited wrapper; ID namespaces are not source row numbers",
                normalized_raw_text_checked=False,model_work=0)


def checkpoint_contract(checkpoints):
    if set(checkpoints) != set(ARMS):
        raise ValueError("Require the two real NATIVE and SECOND children")
    normalized = {}
    for arm in ARMS:
        item = dict(checkpoints[arm])
        item["export"] = str(item.get("export", item.get("checkpoint_path", "")))
        item["weights"] = str(item.get("weights", Path(item["export"]) / "model.safetensors"))
        if not item["export"] or item.get("cumulative_adam_steps") != 640:
            raise ValueError("Both children must be authenticated step-640 exports")
        if len(item["weights_identity"]["sha256"]) != 64:
            raise ValueError("Missing child weight identity")
        normalized[arm] = item
    if normalized["NATIVE"]["weights_identity"]["sha256"] != NATIVE_SHA256:
        raise ValueError("Native step-640 control differs from PR39 handoff")
    if (normalized["NATIVE"]["export"] == normalized["SECOND"]["export"] or
            normalized["NATIVE"]["weights_identity"] == normalized["SECOND"]["weights_identity"]):
        raise ValueError("Inactive or aliased treatment: omit duplicate serving and diagnostics")
    return normalized


def score_child(selected, parent_block, observed, canonical, *, arm, checkpoint):
    """Original follower preservation is separate from realized acceptance."""
    if arm not in ARMS or observed["status"] != "completed" or len(observed["blocks"]) != 1:
        raise ValueError("Missing one native child block")
    reconstruction = observed["reconstruction"]
    if (reconstruction["status"] != "passed" or reconstruction["target_prefix_cache_exact"] is not True
            or reconstruction["destination_draft_rebuilt"] is not True
            or reconstruction["source_draft_cache_transplanted"] is not False
            or observed["original_cap"] != 2048 or observed["cap_unchanged"] is not True
            or observed["original_remaining_budget"] != selected["remaining_budget"]
            or reconstruction["original_cap"] != 2048
            or reconstruction["remaining_budget"] != selected["remaining_budget"]
            or observed["output_matches_canonical_prefix"] is not True):
        raise ValueError("Child constructor/cap/target KV contract failed")
    child = observed["blocks"][0]
    response = observed["response_ids"]
    if (response != canonical[:len(response)] or not response
            or observed["native_finished"] != (len(response) == len(canonical))
            or len(child["draft_tokens"]) != 15
            or len(response) != selected["output_start"] + 1 + child["committed_progress"]
            or child["finished"] != observed["native_finished"]
            or child["committed_progress"] != len(child["committed_tokens"])):
        raise ValueError("Child canonical output or boundary contract failed")
    accepted = child["accepted_draft_tokens"]
    raw = child["raw_accepted_draft_tokens_before_stop"]
    if (type(accepted) is not int or not 0 <= accepted <= 15
            or type(raw) is not int or not 0 <= raw <= 15
            or type(child["committed_progress"]) is not int or not 1 <= child["committed_progress"] <= 16
            or accepted != min(raw, child["committed_progress"])
            or child["native_accepted_draft_tokens"] != max(child["committed_progress"] - 1, 0)):
        raise ValueError("Child native acceptance counters disagree")
    scored = score(selected, parent_block, child, canonical,
                   provenance="PR40_exact_parent_history_neutral_parity_step640")
    j = selected["j"] - 1
    followers = 2 if selected["group"] == "LONG" else 1
    labels = canonical[selected["output_start"] + 1:][:selected["valid_slots"]]
    preserved = child["draft_tokens"][j + 1:j + 1 + followers] == labels[j + 1:j + 1 + followers]
    return dict(scored, arm=arm, checkpoint=checkpoint, child_adam_step=640,
                original_correct_followers_measured=followers,
                follower_scope="first_two_original_followers" if followers == 2 else "first_original_follower",
                first_two_original_followers_accepted=(accepted >= j + 3) if followers == 2 else None,
                earlier_prefix_eligible=j > 0,
                first_follower_intact=child["draft_tokens"][j + 1] == labels[j + 1],
                original_followers_intact=preserved,
                first_follower_accepted=accepted >= j + 2,
                original_followers_accepted=accepted >= j + 1 + followers,
                repair_with_original_followers=scored["repair"] and accepted >= j + 1 + followers,
                child_native_accepted=child["native_accepted_draft_tokens"],
                child_committed_progress=child["committed_progress"])


def initial_rows(inputs):
    return [dict(selected=s, status="pending", children={}) for s in inputs["selected"]]


def row_counts(rows):
    valid = sum(r["status"] == "valid" for r in rows)
    children = sum(len(r["children"]) for r in rows)
    return dict(planned_starts=len(rows), valid=valid,
                invalid=sum(r["status"] == "invalid" for r in rows),
                pending=sum(r["status"] == "pending" for r in rows),
                validated_child_blocks=children, unvalidated_or_unattempted_child_blocks=2 * len(rows) - children)


METRICS = ("repair", "original_followers_intact", "first_follower_accepted",
           "original_followers_accepted", "repair_with_original_followers",
           "earlier_prefix_break", "native_prefix_change")


def summarize(rows, request_clusters, *, resamples=10000, seed=20261113):
    """Paired cluster draws over all64 original requests; fixed census weights.

    Incomplete diagnostics expose coverage and raw pairs, never a valid-only
    estimate of the full selected sample. Undefined cluster draws stay visible.
    """
    import numpy as np

    counts = row_counts(rows)
    base = dict(schema=SCHEMA, **counts, model_work=0, census_weights=CENSUS_WEIGHTS,
                analysis=dict(resamples=resamples, seed=seed, generator="PCG64",
                              percentile_method="linear", unit="paired_whole_request_cluster",
                              census_weights_fixed=True),
                limitation="Other parent starts unmeasured; local prefix changes are not request call savings.")
    if (len(rows) != 128 or counts["valid"] != 128 or counts["validated_child_blocks"] != 256):
        return dict(base, status="incomplete", reason="No complete-sample inference from a valid-only subset")
    keys = [(r["selected"]["request_id"], r["selected"]["parent_round_ordinal"]) for r in rows]
    if (len(set(keys)) != 128 or Counter(r["selected"]["group"] for r in rows) != {"SHORT":64,"LONG":64}
            or any(set(r["children"]) != set(ARMS) for r in rows)
            or len(request_clusters) != len(set(request_clusters))
            or not {k[0] for k in keys} <= set(request_clusters)):
        raise ValueError("Invalid paired matched-start analysis population")
    # One shared cluster draw carries all starts, both classes, and both arms.
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = rng.integers(0, len(request_clusters), size=(resamples, len(request_clusters)))
    draws = np.stack([(sampled == i).sum(axis=1) for i in range(len(request_clusters))], axis=1)
    lookup = {r:i for i,r in enumerate(request_clusters)}
    groups, paired_arrays = {}, {}
    def interval(values):
        valid = values[np.isfinite(values)]
        return dict(ci95=np.percentile(valid, [2.5,97.5], method="linear").tolist() if len(valid) else None,
                    defined_draws=int(len(valid)), undefined_draws=int(len(values)-len(valid)))
    for group in GROUPS:
        own = [r for r in rows if r["selected"]["group"] == group]
        n = np.zeros(len(request_clusters))
        arrays = {a:np.zeros((len(request_clusters),len(METRICS))) for a in ARMS}
        for row in own:
            i = lookup[row["selected"]["request_id"]]
            n[i] += 1
            for arm in ARMS:
                arrays[arm][i] += [row["children"][arm][m] for m in METRICS]
        den = draws @ n
        arm_report = {}
        for arm in ARMS:
            values = arrays[arm].sum(axis=0)
            sampled_values = np.divide(draws @ arrays[arm], den[:,None],
                                       out=np.full((resamples,len(METRICS)),np.nan), where=den[:,None]!=0)
            arm_report[arm] = dict(starts=64, contributing_requests=int((n>0).sum()),
                metrics={m:dict(total=float(values[k]), per_start=float(values[k]/64),
                                contributing_requests=len({r["selected"]["request_id"] for r in own if r["children"][arm][m] != 0}), **interval(sampled_values[:,k]))
                         for k,m in enumerate(METRICS)},
                gross_tokens_gained=sum(r["children"][arm]["prefix_gained"] for r in own),
                gross_tokens_lost=sum(r["children"][arm]["prefix_lost"] for r in own))
        delta = arrays["SECOND"] - arrays["NATIVE"]
        bootstrap = np.divide(draws @ delta, den[:,None], out=np.full((resamples,len(METRICS)),np.nan), where=den[:,None]!=0)
        paired_arrays[group] = bootstrap[:, METRICS.index("native_prefix_change")]
        paired = {m:dict(total=float(delta[:,k].sum()), per_start=float(delta[:,k].sum()/64),
                         contributing_requests=len({r["selected"]["request_id"] for r in own if r["children"]["SECOND"][m] != r["children"]["NATIVE"][m]}), **interval(bootstrap[:,k]))
                  for k,m in enumerate(METRICS)}
        # Starts with opposing effects in one request remain visible separately.
        pair_delta = [r["children"]["SECOND"]["child_accepted"]-r["children"]["NATIVE"]["child_accepted"] for r in own]
        groups[group] = dict(arms=arm_report, paired_SECOND_minus_NATIVE=paired,
                            earlier_prefix_eligible=sum(r["selected"]["j"]>1 for r in own),
                            follower_scope="first_two_original_followers" if group == "LONG" else "first_original_follower",
                            paired_gross_gained=sum(max(d,0) for d in pair_delta),
                            paired_gross_lost=sum(max(-d,0) for d in pair_delta),
                            improved=sum(d>0 for d in pair_delta), worse=sum(d<0 for d in pair_delta), tied=sum(d==0 for d in pair_delta),
                            requests_with_changed_starts=len({r["selected"]["request_id"] for r,d in zip(own,pair_delta) if d}))
    weighted_draws = sum(CENSUS_WEIGHTS[g]*paired_arrays[g] for g in GROUPS)
    terms = {g:CENSUS_WEIGHTS[g]*groups[g]["paired_SECOND_minus_NATIVE"]["native_prefix_change"]["per_start"] for g in GROUPS}
    return dict(base, status="complete", groups=groups,
                request_clusters=len(request_clusters),
                paired_draws_sha256=hashlib.sha256(sampled.astype("<i8").tobytes()).hexdigest(),
                census_weighted_SECOND_minus_NATIVE=dict(per_later_parent_start=sum(terms.values()),
                    contributions=terms, **interval(weighted_draws)))


class DiagnosticBudget(WorkBudget):
    """PR34 accounting plus a separate startup-only ceiling."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_blocks = 0

    def attempt(self, kind, **metadata):
        startup = kind == "draft_block" and metadata.get("startup") is True
        if startup and self.startup_blocks >= STARTUP_BLOCK_LIMIT:
            self._write(dict(kind="budget_stop", reason="startup_block_limit", startup_blocks=self.startup_blocks))
            raise BudgetExceeded("64 startup draft/standalone verifier evaluations exhausted")
        ticket = super().attempt(kind, **metadata)
        if startup:
            self.startup_blocks += 1
        return ticket

    def summary(self):
        return dict(super().summary(), startup_blocks=self.startup_blocks, startup_block_limit=STARTUP_BLOCK_LIMIT)


def worker(plan_path, output, connection, *, backend_factory=None):
    """Native worker loop; dependency injection is only for CPU unit fixtures."""
    if backend_factory is None:
        from experiments.longer_run_headroom.native_worker import HeadroomBackend
        backend_factory = HeadroomBackend
    jobplan = read(plan_path)
    inputs, checkpoints = jobplan["matched_start"], jobplan["checkpoints"]
    budget = connect_budget(read(connection))
    rows = initial_rows(inputs)
    index = {(r["selected"]["request_id"],r["selected"]["parent_round_ordinal"]):r for r in rows}
    backend, active, load = None, [], None
    started = time.monotonic()
    def persist(status, **extras):
        write_json(output / "ROWS.json", rows)
        write_json(output / "STATUS.json", dict(status=status, **row_counts(rows), work=budget.summary(), **extras))
    persist("loading")
    try:
        load = budget.attempt("model_load")
        backend = backend_factory(inputs["shared"]["target_path"], inputs["parent_checkpoint"]["export"],
            worker_python=inputs["shared"]["runtime"]["python"], output_root=output / "native",
            budget_connection=read(connection))
        budget.complete(load, loading_s=backend.loading_s)
        for number, job in enumerate(inputs["jobs"]):
            item, parent = job["item"], job["parent"]
            request_id = item["request_id"]
            folder = output / "replays" / f'{item["selection_rank"]:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            active = [(request_id,s["parent_round_ordinal"]) for s in job["starts"]]
            parent_path = inputs["parent_checkpoint"]["export"]
            boundaries = sorted(s["parent_round_ordinal"]-1 for s in job["starts"])
            result = counted(budget,"original_completion",lambda: backend.native_walk(
                dict(item["request"],canonical=item["canonical"]),parent_path,boundaries), request_id=request_id)
            fresh = result["record"]
            write_json(folder / "PARENT.json",fresh)
            validate_walk(fresh,item)
            a,b = parent["metadata"]["trace"],fresh["metadata"]["trace"]
            parity = [neutral_parity(x,y) for x,y in zip(a,b)]
            valid = len(a)==len(b) and fresh["checkpoint_id"]==parent_path and all(r["status"]=="passed" for r in parity)
            write_json(folder / "PARENT_PARITY.json",dict(status="passed" if valid else "failed",blocks=parity))
            if not valid:
                raise ValueError("Fresh parent walk differs from archived original")
            for s in job["starts"]:
                ordinal = s["parent_round_ordinal"]
                active = [(request_id,ordinal)]
                handle = result["successors"][str(ordinal-1)]
                observed = counted(budget,"neutral_block",lambda:backend.one_block(handle,parent_path),request_id=request_id,ordinal=ordinal)
                write_json(folder / f"{ordinal:04d}_NEUTRAL.json",observed)
                check = neutral_parity(a[ordinal-1],observed["blocks"][0])
                write_json(folder / f"{ordinal:04d}_PARITY.json",check)
                # Apply the same full constructor contract, then require parent draft KV.
                score_child(s,a[ordinal-1],observed,item["canonical"]["response_ids"],arm="NATIVE",checkpoint=parent_path)
                if check["status"] != "passed" or observed["reconstruction"]["neutral_draft_prefix_cache_exact"] is not True:
                    raise ValueError("Neutral one-block exact proposal/KV parity failed")
            for arm in ARMS:
                checkpoint = checkpoints[arm]["export"]
                for s in job["starts"]:
                    ordinal = s["parent_round_ordinal"]
                    active = [(request_id,ordinal)]
                    handle = result["successors"][str(ordinal-1)]
                    observed = counted(budget,"child_block",lambda:backend.one_block(handle,checkpoint),arm=arm,request_id=request_id,ordinal=ordinal)
                    write_json(folder / f"{ordinal:04d}_{arm}.json",observed)
                    row = index[request_id,ordinal]
                    row["children"][arm] = score_child(s,a[ordinal-1],observed,item["canonical"]["response_ids"],arm=arm,checkpoint=checkpoint)
                    row["status"] = "valid" if len(row["children"])==2 else "pending"
                    persist("running",completed_parent_requests=number,active_request=request_id)
            backend.release_request()
            persist("running",completed_parent_requests=number+1)
        persist("completed",worker_wall_s=time.monotonic()-started)
        return 0
    except BaseException as error:
        if backend is None and load is not None:
            budget.fail(load,repr(error))
        for key in active:
            if isinstance(error,ValueError):
                index[key]["status"]="invalid"
            index[key]["reason"]=repr(error)
        persist("incomplete",reason=repr(error),traceback=traceback.format_exc(),worker_wall_s=time.monotonic()-started)
        return 2
    finally:
        if backend is not None:
            backend.close()


def preflight(inputs, checkpoints):
    validate_inputs(inputs)
    checks = []
    for item in [inputs["parent_checkpoint"], *checkpoints.values()]:
        checks.append(_check_file(item["weights"],item["weights_identity"]))
    shared = inputs["shared"]
    for name, expected in shared["target_files"].items():
        checks.append(_check_file(Path(shared["target_path"])/name,expected))
    runtime = shared["runtime"]
    version = ".".join(runtime["python_version"].split(".")[:2])
    package = Path(runtime["python"]).parent.parent/"lib"/f"python{version}"/"site-packages/sglang/srt"
    for name, expected in runtime["native_runtime_sha256"].items():
        checks.append(_check_file(package/name,expected))
    # Parent metadata is embedded under a digest; original path existence is not
    # required in another authenticated mirror. Verify all read artifacts at preparation.
    return dict(status="passed",artifact_checks=checks,input_binding=inputs["binding_sha256"])


def run(root, plan, checkpoints, *, remaining_s, deadline=None):
    """Execute at most the remaining campaign allowance; returns an explicit receipt.

    ``root/matched_start`` must be fresh. ``plan['matched_start']`` is the sealed
    output of prepare_inputs. ``checkpoints`` maps NATIVE/SECOND to export (or
    checkpoint_path), weights_identity and cumulative_adam_steps=640. An optional
    monotonic ``deadline`` can only shorten remaining_s. Both physical cards must
    be quiet and this inherited backend uses physical GPU0.
    """
    started = time.monotonic()
    if not isinstance(remaining_s,(int,float)) or not math.isfinite(remaining_s) or not 0 <= remaining_s <= 4500:
        raise ValueError("Require the remaining portion of the 4500-second campaign cap")
    stop_at = min(started+remaining_s, float(deadline)) if deadline is not None else started+remaining_s
    if not math.isfinite(stop_at):
        raise ValueError("Require a finite monotonic deadline")
    inputs = validate_inputs(read(plan["matched_start"]) if isinstance(plan["matched_start"], (str,Path)) else plan["matched_start"])
    checkpoints = checkpoint_contract(checkpoints)
    if str(plan.get("serving_gpu","0")) != "0":
        raise ValueError("Inherited PR40 backend requires physical GPU0")
    root = Path(root).absolute()
    output = root/"matched_start"
    output.mkdir(parents=True,exist_ok=False)
    jobplan = dict(matched_start=inputs,checkpoints=checkpoints)
    write_json(output/"PLAN.json",jobplan)
    write_json(output/"ROWS.json",initial_rows(inputs))
    write_json(output/"STATUS.json",dict(status="pending",**row_counts(initial_rows(inputs))))
    write_json(output/"BUDGET.json",dict(remaining_s=remaining_s,cleanup_reserve_s=CLEANUP_RESERVE_S,
              limits=LIMITS,startup_block_limit=STARTUP_BLOCK_LIMIT,overall_campaign_cap_s=4500))
    budget = service = proxy = child = None
    timed_out, code = False, 2
    try:
        if stop_at-time.monotonic() <= CLEANUP_RESERVE_S:
            raise BudgetExceeded("No remaining time after cleanup reserve; diagnostic unattempted")
        admission = preflight(inputs,checkpoints)
        write_json(output/"PREFLIGHT.json",admission)
        import shutil
        if shutil.disk_usage(output).free < 8*1024**3 + 128*1024**2:
            raise RuntimeError("Require 8GiB reserve plus128MiB diagnostic evidence allowance")
        jobs = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
        if jobs.strip():
            raise RuntimeError("Matched-start phase requires both physical GPUs quiet")
        left = stop_at-time.monotonic()
        if left <= CLEANUP_RESERVE_S:
            raise BudgetExceeded("No remaining time after admission; diagnostic unattempted")
        budget = DiagnosticBudget(output/"WORK.jsonl",limits=LIMITS,gpu_seconds=left)
        service = BudgetService(budget)
        proxy = connect_budget(service.connection)
        write_json(output/"budget_connection.json",service.connection)
        (output/"budget_connection.json").chmod(0o600)
        proxy.start_gpu(1)
        source = Path(__file__).resolve().parents[2]
        with (output/"worker.log").open("w") as log:
            child = subprocess.Popen([inputs["shared"]["runtime"]["python"],"-u","-m",
                "experiments.second_follower.matched_start","--worker","--root",str(root),
                "--plan",str(output/"PLAN.json"),"--connection",str(output/"budget_connection.json")],
                cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(output/"PROCESS.json",dict(pid=child.pid,occupied_gpus=1,remaining_s=left))
            try:
                code = child.wait(timeout=max(0.01,stop_at-time.monotonic()-CLEANUP_RESERVE_S))
            except subprocess.TimeoutExpired:
                timed_out=True
                write_json(output/"TIMEOUT_CLEANUP.json",stop_owned_groups(proxy,child))
        remaining = active_owned_processes({p["pgid"] for p in proxy.processes()})
        if remaining:
            write_json(output/"ORPHAN_CLEANUP.json",stop_owned_groups(proxy,child))
            code=2
        proxy.end_gpu()
        # A read-only postflight preserves the authenticated exports and inputs.
        postflight = preflight(inputs,checkpoints)
        write_json(output/"POSTFLIGHT.json",postflight)
    except BaseException as error:
        write_json(output/"FAILURE.json",dict(error=repr(error),traceback=traceback.format_exc()))
        code=2
    finally:
        if proxy is not None:
            if ((child is not None and child.poll() is None) or
                    active_owned_processes({p["pgid"] for p in proxy.processes()})):
                write_json(output/"FINAL_CLEANUP.json",stop_owned_groups(proxy,child))
            proxy.end_gpu()
        rows=read(output/"ROWS.json")
        try:
            summary=summarize(rows,inputs["request_clusters"])
        except BaseException as error:
            code=2
            summary=dict(status="incomplete",reason="Matched-start analysis failed",error=repr(error))
            write_json(output/"ANALYSIS_FAILURE.json",dict(error=repr(error),traceback=traceback.format_exc()))
        status=read(output/"STATUS.json")
        complete=code==0 and not timed_out and summary["status"]=="complete" and status["status"]=="completed"
        result=dict(status="completed" if complete else "incomplete",exit_code=0 if complete else 2,
                    timed_out=timed_out,**row_counts(rows),wall_s=time.monotonic()-started,
                    work=budget.summary() if budget else dict(attempted={},completed={},failed={},unknown={},gpu_seconds=0),
                    remaining_s_at_entry=remaining_s,model_work_scope="parent walks, neutral blocks, two fresh640 child blocks only")
        if not complete:
            summary=dict(summary,status="incomplete",reason="Native phase or postflight incomplete; no full-sample inference")
            summary.pop("groups",None)
            summary.pop("census_weighted_SECOND_minus_NATIVE",None)
        write_json(output/"SUMMARY.json",summary)
        write_json(output/"RESULT.json",result)
        if not complete:
            write_json(output/"STATUS.json",dict(status,status="incomplete",reason="phase incomplete or budget stop",**row_counts(rows)))
        if service is not None:
            service.close()
    return result


if __name__ == "__main__":
    # Same operational-stop semantics as role_loss_pilot.campaign, kept local
    # so the CPU CLI does not import the training/model stack.
    import signal
    def interrupted(signum, _frame):
        raise InterruptedError(f"Operational stop signal {signum}; no automatic retry")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_positional", nargs="?", type=Path)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--checkpoints",type=Path)
    parser.add_argument("--remaining-s",type=float)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--connection",type=Path)
    args=parser.parse_args()
    root = args.root or args.root_positional
    if root is None:
        parser.error("A positional root or --root is required")
    if args.worker:
        if args.plan is None or args.connection is None:
            parser.error("Worker requires --plan and --connection")
        raise SystemExit(worker(args.plan,root/"matched_start",args.connection))
    if args.remaining_s is None:
        parser.error("--remaining-s is required for supervisor")
    receipt=run(root,read(args.plan or root/"sealed/PLAN.json"),
                read(args.checkpoints or root/"CHECKPOINTS.json"),remaining_s=args.remaining_s)
    raise SystemExit(receipt["exit_code"])
