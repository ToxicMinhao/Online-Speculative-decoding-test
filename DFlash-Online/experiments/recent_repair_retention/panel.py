"""Input-only retained-repair panel, sealed before event-seven optimization."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path

from experiments.lost_repair.prepare import identity, read, write
from experiments.mathqa_transfer.data import normalize, TOKENIZER_HASHES
from experiments.repair_history_margin.prepare import match_keys
from experiments.survival_confirmation.data import REVISION, SOURCE_HASHES, STOP_IDS, digest, render_question

PANEL_PREFIX = "retention-e7-v1|"
PATH_PREFIX = "retention-e7-path|"


def order_hash(source_id, prefix=PANEL_PREFIX):
    return hashlib.sha256((prefix + source_id).encode("utf-8")).hexdigest()


def select_panel(candidates, exclusions, prior_exposure, *, count=256):
    """Prefer no recorded endpoint exposure, then hash stable source identities.

    All prompt eligibility/duplicate checks happen before ranking. Selection
    never consults canonical answers, output lengths, metrics or availability.
    """
    excluded = defaultdict(list)
    for row in exclusions:
        for key in match_keys(row):
            excluded[key].append(row)
    audit, eligible, seen = [], [], {}
    for candidate in sorted(candidates, key=lambda r:(r["source_row"], r["source_id"])):
        row = deepcopy(candidate)
        keys = match_keys(row)
        hits = {(entry["reason"], entry["request_id"]):entry
                for key in keys for entry in excluded.get(key, ())}
        duplicates = sorted({seen[k] for k in keys if k in seen})
        history = deepcopy(prior_exposure.get(row["source_id"], []))
        reason = "excluded" if hits else "duplicate" if duplicates else "eligible"
        if not row["prompt_ids"] or len(row["prompt_ids"]) + 2048 + 16 > 4096:
            reason = "context_ineligible"
        if not hits and not duplicates:
            for key in keys:
                seen[key] = row["source_id"]
        audit.append(dict(source_id=row["source_id"], source_row=row["source_row"],
            status=reason, duplicate_of=duplicates,
            exclusion_matches=[dict(reason=e["reason"],request_id=e["request_id"],
                shared_keys=[list(k) for k in sorted(keys & match_keys(e))])
                for _,e in sorted(hits.items())],
            prior_endpoint_exposure=bool(history)))
        if reason == "eligible":
            row.update(prior_endpoint_exposure=bool(history),
                prior_endpoint_evidence=history, exposure_tier=int(bool(history)),
                selection_sha256=order_hash(row["source_id"]))
            eligible.append(row)
    eligible.sort(key=lambda r:(r["exposure_tier"],r["selection_sha256"],r["source_id"]))
    rows = eligible[:count]
    for ordinal,row in enumerate(rows):
        row.update(panel_ordinal=ordinal,request_index=ordinal,selection_rank=ordinal)
    path_rows = sorted(rows,key=lambda r:(order_hash(r["source_id"],PATH_PREFIX),r["source_id"]))[:64]
    panel = dict(schema="recent-repair-retention-panel-v1", requests=rows,
        count=len(rows), requested_count=count, shortfall=max(0,count-len(rows)),
        source_revision=REVISION, input_only_selection=True,
        outcomes_used=False, generated_lengths_used=False, reference_availability_used=False,
        selection_prefix=PANEL_PREFIX, path_prefix=PATH_PREFIX,
        eligibility_count=len(eligible), candidate_count=len(candidates),
        eligible_order=[r["source_id"] for r in eligible],
        eligible_tier_counts=dict(Counter(str(r["exposure_tier"]) for r in eligible)),
        selected_tier_counts=dict(Counter(str(r["exposure_tier"]) for r in rows)),
        path_request_ids=[r["request_id"] for r in path_rows],
        exclusion_audit=audit,
        exclusion_status_counts=dict(Counter(r["status"] for r in audit)),
        exposure_scope="Recorded endpoint inventories and completed panels; no-record does not assert globally unused",
        normalization="NFKC, casefold and collapsed whitespace; source/rendered/token identity checks")
    panel["binding"] = digest(panel)
    return panel


def verify_panel(panel):
    if digest({k:v for k,v in panel.items() if k != "binding"}) != panel.get("binding"):
        raise ValueError("Retention panel seal changed")
    rows = panel["requests"]
    if (len(rows) != panel["count"] or panel["source_revision"] != REVISION
            or len(rows) > 256 or not panel["input_only_selection"]
            or any(panel[k] for k in ("outcomes_used","generated_lengths_used","reference_availability_used"))):
        raise ValueError("Changed retention panel contract")
    if [r["source_id"] for r in rows] != panel["eligible_order"][:len(rows)]:
        raise ValueError("Changed retained input order")
    for ordinal,row in enumerate(rows):
        if (row["panel_ordinal"] != ordinal or row["source_id"] != f"gsm8k:main:test:{row['source_row']}"
                or row["prompt_ids_sha256"] != digest(row["prompt_ids"])
                or row["selection_sha256"] != order_hash(row["source_id"])
                or row["stop_token_ids"] != STOP_IDS or row["max_new_tokens"] != 2048
                or row["collect_feedback"]):
            raise ValueError("Changed retention request identity or serving configuration")
    if rows != sorted(rows,key=lambda r:(r["exposure_tier"],r["selection_sha256"],r["source_id"])):
        raise ValueError("Changed exposure tier/hash order")
    expected = sorted(rows,key=lambda r:(order_hash(r["source_id"],PATH_PREFIX),r["source_id"]))[:64]
    if panel["path_request_ids"] != [r["request_id"] for r in expected]:
        raise ValueError("Changed fixed-start subset")
    for name in ("source_id","prompt_ids_sha256","normalized_question_sha256"):
        if len({r[name] for r in rows}) != len(rows):
            raise ValueError("Duplicate selected input")
    return dict(status="passed",count=len(rows),binding=panel["binding"])


def public_panel(panel):
    return dict(panel, requests=[{k:v for k,v in row.items() if k != "prompt_ids"}
                                for row in panel["requests"]])


def prepare_panel(research_root, tokenizer_path, output):
    """Reuse PR43's authenticated exclusions; add its complete outcome panel."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    root, tokenizer_path, output = map(Path,(research_root,tokenizer_path,output))
    if output.exists():
        raise FileExistsError("Preserve the previous input seal")
    source_path = root / "projected_target_feasibility_20261022/data/approved/manifest.json"
    source = read(source_path)["sources"]["gsm8k_test"]
    if source["revision"] != REVISION or identity(source["local_path"])["sha256"] != SOURCE_HASHES["test"]:
        raise ValueError("Pinned GSM8K source differs")
    questions = pq.read_table(source["local_path"],columns=["question"])["question"].to_pylist()
    inputs = {str(source_path):identity(source_path), source["local_path"]:identity(source["local_path"])}
    for filename,expected in TOKENIZER_HASHES.items():
        path = tokenizer_path / filename
        if identity(path)["sha256"] != expected:
            raise ValueError("Pinned serving tokenizer differs")
        inputs[str(path)] = identity(path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)

    pr43 = root / "dflash_repair_history_margin_20261115/input_bundle_001"
    seal = read(pr43 / "INPUT_SEAL.json")
    for name in ("EXCLUSIONS.json","PANEL.json"):
        if identity(pr43/name) != seal["files"][name]:
            raise ValueError("PR43 input exclusion/panel seal differs")
        inputs[str(pr43/name)] = identity(pr43/name)
    exclusions = read(pr43/"EXCLUSIONS.json")["rows"]
    counts = Counter(r["reason"] for r in exclusions)
    if counts != {"events_1_6_training":768,"PR42_discovery_batch":128,"PR34_PR40_discovery_panel":64}:
        raise ValueError("Expected complete event1–7 and PR34/40 exclusion population")
    prior_panel = read(pr43/"PANEL.json")
    if prior_panel["count"] != 192:
        raise ValueError("Expected every PR43 outcome input")
    exclusions += [dict(r,reason="PR43_outcome_input") for r in prior_panel["requests"]]

    exposure = defaultdict(list)
    history_path = root / "dflash_first_error_fresh_confirmation_20261027/inventory_audit/INVENTORY_AUDIT.json"
    extra_path = root / "dflash_survival_confirmation_20261111/preparation/input_evidence/ADDITIONAL_EXPOSURE.json"
    history = read(history_path)
    inputs[str(history_path)] = identity(history_path)
    inputs[str(extra_path)] = identity(extra_path)
    for item in history["primary_completed_execution_sets"]:
        for index in item["source_rows"]:
            exposure[f"gsm8k:main:test:{index}"].append(dict(source="verified_actual_use_inventory",evidence=item["directory"]))
    for key in ("oct22_additional_completed_source_rows","async_final_additional_completed_source_rows"):
        for index in history[key]:
            exposure[f"gsm8k:main:test:{index}"].append(dict(source=key,evidence=str(history_path)))
    for item in read(extra_path)["sets"]:
        for index in item["source_rows"]:
            exposure[f"gsm8k:main:test:{index}"].append(dict(source=item["name"],evidence=item["evidence"]))
    # These input panels have completed, preserved endpoint reports. They add
    # exposure only; no outcome number enters eligibility or ranking.
    for relative,label in (
        ("dflash_survival_confirmation_20261111/preparation/data_001/PANEL.json","completed_PR38"),
        ("dflash_successor_shuffle_20261111/delivery/retained/sealed/PANEL.json","completed_PR39"),
    ):
        path = root / relative
        inputs[str(path)] = identity(path)
        for row in read(path)["requests"]:
            exposure[row["source_id"]].append(dict(source=label,evidence=str(path)))

    candidates = []
    for index,question in enumerate(questions):
        ids,rendered = render_question(question,tokenizer)
        source_id = f"gsm8k:main:test:{index}"
        candidates.append(dict(request_id=f"retention-e7-gsm8k-{index:04d}",dataset="gsm8k",
            source_id=source_id,dataset_id=source_id,source_row=index,source_split="test",
            source_revision=REVISION,original_index=index,prompt_ids=ids,
            prompt_ids_sha256=digest(ids),
            rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            normalized_question_sha256=hashlib.sha256(normalize(question).encode()).hexdigest(),
            stop_token_ids=list(STOP_IDS),max_new_tokens=2048,collect_feedback=False))
    panel = select_panel(candidates,exclusions,exposure)
    verify_panel(panel)
    output.mkdir(parents=True)
    write(output/"PANEL.json",panel)
    write(output/"PANEL_MANIFEST.json",public_panel(panel))
    write(output/"EXCLUSIONS.json",dict(rows=exclusions,counts=dict(Counter(r["reason"] for r in exclusions))))
    write(output/"PANEL_PROVENANCE.json",dict(inputs=inputs,source_columns_read=["question"],
        source_answers_used=False,model_work=0,selection_precedes_outcomes=True))
    write(output/"PANEL_SEAL.json",dict(status="sealed_before_optimization",binding=panel["binding"],
        files={p.name:identity(p) for p in output.iterdir() if p.is_file()}))
    return dict(count=panel["count"],shortfall=panel["shortfall"],
        eligible=panel["eligibility_count"],tiers=panel["selected_tier_counts"],binding=panel["binding"])


def extend_block_depth_references(requests, references, reuse_audit, research_root):
    """Authenticate original native FIVE records for exact already-selected inputs."""
    from experiments.block_depth.endpoint import validate_execution

    root = Path(research_root)/"dflash_block_depth_20261105"
    archive = root/"full_retrieved_002/full_production_002"
    inventory_path = root/"full_retrieved_002/full_production_002_inventory.json"
    inventory = read(inventory_path)["files"]
    checked = {str(inventory_path):identity(inventory_path)}

    def checked_read(relative):
        path = archive/relative
        actual = identity(path)
        expected = inventory[relative]
        if any(actual[k] != expected[k] for k in ("bytes","sha256")):
            raise ValueError("Changed original block-depth native reference evidence")
        checked[str(path)] = actual
        return read(path)

    plan, status = checked_read("PLAN.json"), checked_read("STATUS.json")
    if status["status"] != "completed" or plan["code_commit"] != "0f704167b268672d980fabe704898abf9b134c39":
        raise ValueError("Wrong block-depth execution source or completion")

    def pinned_input(name):
        original = plan[name]
        path = root/original.split("/dflash_block_depth_20261105/",1)[1]
        actual, expected = identity(path), plan["input_identities"][original]
        if any(actual[k] != expected[k] for k in ("bytes","sha256")):
            raise ValueError("Changed block-depth model/runtime/input pin")
        checked[str(path)] = actual
        return read(path)

    manifests, models, runtime, prior = [pinned_input(k) for k in
        ("manifest_file","model_pin","runtime_pin","canonical_audit_file")]
    binding = reuse_audit["target_runtime_binding"]
    if models["target"]["files"] != binding["target_files"] or prior["binding_digest"] != reuse_audit["binding_digest"]:
        raise ValueError("Target or inherited canonical binding differs")
    for field in ("python_version","torch","transformers"):
        if runtime[field] != binding[field]:
            raise ValueError("Block-depth native software differs")
    if any(runtime["native_runtime_sha256"].get(k) != v for k,v in binding["native_runtime_sha256"].items()):
        raise ValueError("Block-depth native runtime source differs")
    if any(plan["serving"].get(k) != v for k,v in binding["serving"].items() if k != "reasoning"):
        raise ValueError("Block-depth serving settings differ")
    by_prompt = {digest(r["prompt_ids"]):r for r in manifests["manifests"]["FINAL"]["requests"]}
    canonical = checked_read("CANONICALS.json")
    added = []
    for request in requests:
        if request["request_id"] in references:
            continue
        old = by_prompt.get(digest(request["prompt_ids"]))
        if old is None:
            continue
        if any(request[k] != old[k] for k in ("prompt_ids","stop_token_ids","max_new_tokens")):
            raise ValueError("Block-depth canonical input configuration differs")
        record = canonical[old["request_id"]]
        if "metadata" not in record:
            raise ValueError("Earlier canonical unexpectedly absent from PR25/VFD audit")
        relative = "canonicals/"+old["request_id"]+".json"
        raw, native = checked_read(relative), checked_read(relative.replace(".json",".native.json"))
        errors = validate_execution(raw,old,record)
        if raw != record or raw["arm"] != "FIVE" or raw["repeat"] != 0:
            errors.append("original_five_reference")
        if any(raw["metadata"].get(k) != v for k,v in native.items()):
            errors.append("native_sidecar_mismatch")
        if native["output_ids"] != raw["response_ids"] or native["behavior"] != "FIVE" or native["status"] != "passed":
            errors.append("native_five_output")
        if native["feedback_capture"] or native["trace_instrumentation"]:
            errors.append("capture_in_canonical")
        if errors:
            raise ValueError("Invalid retained native FIVE canonical: "+repr(errors))
        references[request["request_id"]] = dict(prompt_ids=request["prompt_ids"],
            response_ids=raw["response_ids"],finish_reason=raw["finish_reason"],verified=True,
            binding_digest=reuse_audit["binding_digest"],source="audited_block_depth_FINAL_native_FIVE_exact_prompt",
            source_request_id=old["request_id"],source_record=dict(path=str(archive/relative),**checked[str(archive/relative)]),
            source_native=dict(path=str(archive/relative.replace(".json",".native.json")),
                               **checked[str(archive/relative.replace(".json",".native.json"))]))
        added.append(request["request_id"])
    return dict(status="passed",added=len(added),request_ids=added,source_files=checked,
        additional_native_runtime_sha256={k:v for k,v in runtime["native_runtime_sha256"].items()
                                         if k not in binding["native_runtime_sha256"]},model_work=0)


def prepare_canonical_inputs(input_directory, research_root, retrieval_directory):
    """Revalidate retained native outputs after selection; never replace inputs."""
    from experiments.block_depth.references import prepare_references
    from experiments.o_half_matched_lr.archive import validate_record

    folder, root, retrieval = map(Path,(input_directory,research_root,retrieval_directory))
    if any((folder/name).exists() for name in ("INPUTS.json","REUSED_CANONICAL.json","CANONICAL_REUSE_AUDIT.json")):
        raise FileExistsError("Keep the previously authenticated canonical input bundle")
    panel = read(folder/"PANEL.json")
    verify_panel(panel)
    refs, audit = prepare_references(panel["requests"],
        pr25_root=root/"dflash_kplus_20261103/retrieved_001/production_001",
        pr25_inventory=root/"dflash_kplus_20261103/retrieved_001/REMOTE_INVENTORY.json",
        vfd_root=root/"dflash_vfd_second_pass_20261104/retrieved_001/evidence/production_001",
        vfd_inventory=root/"dflash_vfd_second_pass_20261104/retrieved_001/PRODUCTION_INVENTORY.json")
    block_audit = extend_block_depth_references(panel["requests"],refs,audit,root)
    source_files, recent_added = {}, []
    for name in ("dflash_survival_confirmation_20261111","dflash_successor_shuffle_20261111"):
        archive = root/name/"delivery/retained"
        inventory_path = archive/"COMPACT_DELIVERY_MANIFEST.json"
        inventory = read(inventory_path)["files"]
        source_files[str(inventory_path)] = identity(inventory_path)

        def authenticated(relative):
            path = archive/relative
            actual = identity(path)
            if actual != inventory[relative]:
                raise ValueError("Retained canonical catalog/plan changed")
            source_files[str(path)] = actual
            return read(path)

        prior_plan = authenticated("sealed/PLAN.json")
        if prior_plan["canonical_binding"] != audit["binding_digest"]:
            raise ValueError("Recent endpoint canonical runtime binding differs")
        catalog = authenticated("references/CANONICAL.json")
        old_panel = authenticated("sealed/PANEL.json")
        old_requests = {r["request_id"]:r for r in old_panel["requests"]}
        by_prompt = {digest(c["prompt_ids"]):(key,c) for key,c in catalog.items()}
        for request in panel["requests"]:
            if request["request_id"] in refs:
                continue
            match = by_prompt.get(digest(request["prompt_ids"]))
            if match is None:
                continue
            old_key, canonical = match
            if canonical["source"] != "fresh_pinned_native_verifier_before_timing":
                raise ValueError("An expected earlier canonical failed prior source validation")
            remote = Path(canonical["source_record_path"])
            path = retrieval/remote.relative_to("/data/sources/minhao")
            expected = canonical["source_record_identity"]
            actual = identity(path)
            if any(actual[k] != expected[k] for k in ("bytes","sha256")):
                raise ValueError("Retrieved canonical raw record differs")
            raw = read(path)
            old = old_requests[old_key]
            if any(request[k] != old[k] for k in ("prompt_ids","max_new_tokens","stop_token_ids")):
                raise ValueError("Retained canonical serving input differs")
            errors = validate_record(raw,old,raw["checkpoint_id"],canonical,
                                     label=raw["record_id"],session=raw["session_id"])
            native_remote = Path(raw["metadata"]["native_trace_path"])
            native_path = retrieval/native_remote.relative_to("/data/sources/minhao")
            native = read(native_path)
            if native != raw["metadata"]["capture"]:
                errors.append("native_sidecar_mismatch")
            if (canonical["binding_digest"] != audit["binding_digest"]
                    or canonical["prompt_ids"] != request["prompt_ids"] or not canonical["verified"]):
                errors.append("canonical_binding")
            if errors:
                raise ValueError("Invalid retained recent canonical: "+repr(errors))
            source_files[str(path)] = actual
            source_files[str(native_path)] = identity(native_path)
            refs[request["request_id"]] = dict(canonical,source="audited_"+name+"_exact_prompt",
                source_request_id=old_key,source_record=dict(path=str(path),**actual),
                source_native=dict(path=str(native_path),**identity(native_path)))
            recent_added.append(request["request_id"])
    missing = [r["request_id"] for r in panel["requests"] if r["request_id"] not in refs]
    full_audit = dict(status="passed",requested=len(panel["requests"]),reused=len(refs),
        missing_reference_ids=missing,maximum_new_generations=len(missing),
        pr25_vfd_audit=audit,block_depth_audit=block_audit,recent_endpoint_added=recent_added,
        source_files=source_files,model_work=0,selection_precedes_reference_lookup=True)
    inputs = dict(schema="recent-repair-retention-inputs-v1",panel=panel,references=refs,
        missing_reference_ids=missing,canonical_binding=audit["binding_digest"],
        target_runtime_binding=audit["target_runtime_binding"],canonical_reuse_audit=full_audit,
        source_files={**read(folder/"PANEL_PROVENANCE.json")["inputs"],**source_files},
        panel_identity=identity(folder/"PANEL.json"),model_work=0)
    write(folder/"INPUTS.json",inputs)
    write(folder/"REUSED_CANONICAL.json",refs)
    write(folder/"CANONICAL_REUSE_AUDIT.json",full_audit)
    return inputs


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("research-root","tokenizer-path","output"):
        parser.add_argument("--"+name,required=True)
    print(json.dumps(prepare_panel(**vars(parser.parse_args())),sort_keys=True))
