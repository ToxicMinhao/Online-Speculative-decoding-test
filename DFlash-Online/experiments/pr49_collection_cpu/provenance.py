"""Authenticate retained PR49 event-seven collection evidence, CPU/stdlib only.

This validates provenance and input/capture identity. It deliberately performs
no native-count reduction, statistics, model imports, tensor reads or remote IO.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time

COLLECTION_COMMIT = "db6e1221d8d1b234d9d07e246025d1afd5dc30d5"
RESTART_COMMIT = "af708012e729437068fbef93cb11c9996a9109a9"
REPORT_COMMIT = "27e307cbd29efb08106cacb16e7cfe2bbdd54738"
BASE = "/data/sources/minhao/runs/dflash_persistent_prefix_withholding_20261115/"
ARMS = ("N", "E", "S")


def identity(path):
    path=Path(path)
    with path.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    return dict(bytes=path.stat().st_size,sha256=digest)


class Audit:
    def __init__(self,delivery,qualification,restart):
        self.roots={"production_continuation_001":Path(delivery),
                    "qualification_retry_002":Path(qualification),
                    "production_restart_001":Path(restart)}
        self.errors=[]
        self.checks=0
        self.inputs={}
        self.maps=[]
        self.manifests={}

    def check(self,condition,label):
        self.checks+=1
        if not condition:self.errors.append(label)
        return bool(condition)

    def consume(self,path,expected=None,remote=None,basis=None):
        path=Path(path)
        if not path.is_file():
            self.errors.append("Missing evidence: "+str(path));return None
        actual=identity(path)
        self.check(expected is None or actual==expected,"Identity differs: "+str(path))
        self.inputs[str(path)]=dict(local_path=str(path),remote_path=remote,
            identity=actual,expected_identity=expected,basis=basis)
        return json.loads(path.read_text())

    def local(self,remote):
        historical="/data/sources/minhao/runs/dflash_lost_repair_20261114/execution_001/campaign_001/training/SCHEDULE.json"
        if str(remote)==historical:
            research=self.roots["production_continuation_001"].parents[2]
            return research/"dflash_lost_repair_20261114/delivery/retained/campaign_001/training/SCHEDULE.json","lost_repair","SCHEDULE.json"
        relative=str(remote).removeprefix(BASE)
        run,separator,name=relative.partition("/")
        if not str(remote).startswith(BASE) or not separator or run not in self.roots:
            raise ValueError("No authenticated local path mapping: "+str(remote))
        return self.roots[run]/name,run,name

    def remote(self,path,expected=None):
        local,run,name=self.local(path)
        declared=self.manifests.get(run,{}).get(name)
        self.check(expected is not None or declared is not None,"No manifest binding: "+path)
        if expected is not None and declared is not None:
            self.check(expected==declared,"Conflicting immutable/delivery bindings: "+path)
        return self.consume(local,expected or declared,path,"sealed identity and/or delivery manifest")

    def original(self,name):
        return self.remote(BASE+"production_continuation_001/"+name)

    def chain(self):
        original=self.roots["production_continuation_001"]
        retained=self.consume(original.parent.parent/"PRODUCTION_DELIVERY_VERIFIED.json",basis="existing verified transfer receipt")
        archive=original/"COMPACT_EVIDENCE.tar.gz"
        self.check(identity(archive)==retained["archive"],"Original compact archive differs from retained transfer receipt")
        self.inputs[str(archive)]=dict(local_path=str(archive),remote_path=BASE+"production_continuation_001/COMPACT_EVIDENCE.tar.gz",
            identity=identity(archive),expected_identity=retained["archive"],basis="existing transfer receipt")
        with tarfile.open(archive) as stream:
            payload=stream.extractfile("DELIVERY_MANIFEST.json").read()
        manifest_path=original/"DELIVERY_MANIFEST.json"
        self.check(payload==manifest_path.read_bytes(),"Original delivery manifest differs from authenticated archive")
        self.manifests["production_continuation_001"]=self.consume(manifest_path,
            dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()),basis="manifest inside authenticated compact archive")
        final=self.roots["production_restart_001"]
        transfer=self.consume(final/"DELIVERY.json",basis="existing completed restart delivery receipt")
        self.manifests["production_restart_001"]=self.consume(final/transfer["manifest"],transfer["manifest_identity"],basis="completed restart delivery identity")
        proof=self.remote(BASE+"production_restart_001/RESTART_IMPORT.json")
        self.check(proof["status"]=="passed" and proof["execution_commit"]==RESTART_COMMIT and
            proof["failed_execution_commit"]==COLLECTION_COMMIT,"Restart proof must reuse the original collection execution")
        restart_manifest=self.remote(proof["manifest"]["path"],proof["manifest"]["identity"])
        for key in ("failed_plan","failed_source","failed_campaign","failed_qualification_reuse"):
            node=restart_manifest[key]
            self.remote(node["path"],node["identity"])
        for name in ("N_collect","E_collect","S_collect","E7","S7"):
            node=proof["completed_tasks"][name]
            self.check(node==restart_manifest["completed_tasks"][name],"Restart imported task declaration differs: "+name)
            for row in node["files"]:
                if row["relative_path"].startswith(("records/","native/")) or row["relative_path"] in ("STATUS.json","RESULT.json","FEATURES.json","BOUNDARY.json"):
                    full="tasks/"+name+"/"+row["relative_path"]
                    self.check(self.manifests["production_continuation_001"].get(full)==row["identity"],"Imported compact identity differs: "+full)
        return proof


def pinned_sources(audit,source_manifest,checkout,remote_source):
    names=("experiments/persistent_prefix_withholding/task_runner.py",
        "experiments/persistent_prefix_withholding/worker.py","online_feedback/sglang_backend.py",
        "online_feedback/sglang_worker.py","online_feedback/sglang_hooks.py","online_feedback/native.py",
        "experiments/o_half_matched_lr/archive.py")
    result=[]
    texts={}
    for name in names:
        payload=subprocess.check_output(["git","show",COLLECTION_COMMIT+":DFlash-Online/"+name],cwd=checkout)
        observed=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        audit.check(observed==source_manifest["files"][name],"Pinned source blob differs: "+name)
        candidates=[audit.roots["production_continuation_001"]/"source_delta"/name,Path(checkout)/"DFlash-Online"/name]
        local=next((p for p in candidates if p.is_file() and identity(p)==observed),None)
        audit.check(local is not None,"No exact local pinned source mapping: "+name)
        if local is not None:
            audit.inputs[str(local)]=dict(local_path=str(local),remote_path=remote_source+"/"+name,
                identity=observed,expected_identity=source_manifest["files"][name],basis="original source manifest and Git blob "+COLLECTION_COMMIT)
        text=payload.decode();texts[name]=text
        result.append(dict(source_name=name,execution_commit=COLLECTION_COMMIT,git_object=COLLECTION_COMMIT+":DFlash-Online/"+name,
            local_path=str(local) if local else None,remote_path=remote_source+"/"+name,identity=observed))
    tree=ast.parse(texts[names[0]])
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="collect_task")
    calls=[node.func.attr for node in ast.walk(function) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    audit.check(not any(name in calls for name in ("train_event","step","publish","update_weights_from_disk")),"Collector invokes training or publication")
    worker=texts["online_feedback/sglang_worker.py"]
    audit.check("'temperature': 0" in worker and "'top_k': 1" in worker and "'ignore_eos': False" in worker,"Pinned worker lacks fixed greedy/EOS contract")
    return result


def run(delivery,qualification,restart,output):
    began=time.perf_counter()
    audit=Audit(delivery,qualification,restart)
    result=dict(status="incomplete",source_commit=COLLECTION_COMMIT,restart_commit=RESTART_COMMIT,report_commit=REPORT_COMMIT,
        model_work=0,remote_operations=0,tensors_read=0,threads=1,counts_reduced=False,latencies_compared=False)
    try:
        proof=audit.chain()
        plan=audit.original("sealed/PLAN.json")
        audit.check(plan["execution_commit"]==COLLECTION_COMMIT and proof["binding"]==plan["binding"],"Scientific/source identity differs")
        source=audit.original("SOURCE.json")
        audit.check(source["commit"]==COLLECTION_COMMIT and source["dirty"] is False,"Collection source is not the clean pinned execution")
        checkout=Path(delivery).parent.parent/"implementation_checkout"
        result["source_anchors"]=pinned_sources(audit,source,checkout,plan["source_root"])
        consumed={key:audit.remote(plan[key],plan["input_identities"][plan[key]]) for key in
            ("event8_inputs","expected_ancestry","n7_parity_receipt","event7_schedule","installed_runtime_identities")}
        requests=consumed["event8_inputs"]["requests"]
        audit.check(len(requests)==128 and len({r["request_id"] for r in requests})==128 and len({r["source_id"] for r in requests})==128,"Sealed population is not exactly128 distinct paired requests")
        ancestry=consumed["expected_ancestry"]["requests"]
        # This plan describes the complete prospective event-eight ancestry.
        # The independently inspected K6/N7 receipts below authenticate only
        # its first768/896 rows as already learned at collection time.
        audit.check(len(ancestry)==1024 and [r["original_index"] for r in ancestry]==list(range(1024)),"Complete prospective ancestry order differs")
        inherited=ancestry[:896]
        audit.check(not {r["request_id"] for r in requests}&{r["request_id"] for r in inherited},"Collection requests overlap already learned K6/event7 ancestry")
        audit.check([(r["request_id"],r["source_id"],r["prompt_ids"]) for r in requests]==
            [(r["request_id"],r["source_id"],r["prompt_ids"]) for r in ancestry[896:]],"Prospective event-eight ancestry differs from exact collection inputs")
        parity=consumed["n7_parity_receipt"]
        audit.check(parity["status"]=="passed" and parity["weights_optimizer_counters_rng_exact"] is True and parity["global_step"]==896 and not parity["mismatches"],"N7 parity receipt fails")
        preflight=audit.original("CPU_PREFLIGHT.json")
        audit.check(preflight["status"]=="passed" and preflight["execution_commit"]==COLLECTION_COMMIT,"Original CPU provenance preflight fails")
        for arm,event,records in (("K6",6,768),("N7",7,896)):
            state=preflight["states"][arm]
            audit.check(state["status"]=="passed" and state["completed_event"]==event and state["global_step"]==128*event and state["export_exact"] is True and state["actual_ancestry_records"]==records,"Complete own-state ancestry/export receipt fails: "+arm)
        for key,expected in plan["runtime_contract"]["native_runtime_sha256"].items():
            found=[v for path,v in preflight["runtime"]["installed_sources"].items() if path.endswith("/sglang/srt/"+key)]
            audit.check(len(found)==1 and found[0]["sha256"]==expected,"Installed runtime source pin differs: "+key)
        campaign=audit.original("CAMPAIGN.json")
        schedule={row["task"]:row for row in campaign["schedule"]}
        checkpoints={"N":dict(export=plan["native_n7_export"],state=plan["native_n7_state"],
            weights_identity=plan["input_identities"][plan["native_n7_export"]+"/model.safetensors"],
            state_identity=plan["input_identities"][plan["native_n7_state"]],global_step=896,parity_receipt=plan["n7_parity_receipt"])}
        for arm in ("E","S"):
            endpoint=audit.original("tasks/"+arm+"7/RESULT.json")
            boundary=audit.original("tasks/"+arm+"7/BOUNDARY.json")
            resume=audit.original("tasks/"+arm+"7/trainer/RESUME_CHECK.json")
            audit.check(endpoint["status"]=="completed" and endpoint["event"]==7 and endpoint["arm"]==arm,"Wrong event-seven endpoint: "+arm)
            audit.check(boundary["status"]=="passed" and boundary["export_exact"] is True and boundary["global_step"]==896,"Event-seven boundary failed: "+arm)
            audit.check(resume["status"]=="passed" and resume["exact_resume"] is True and resume["origin_state"]==plan["origin_state"] and resume["origin_export"]==plan["origin_export"] and resume["global_step"]==768 and not resume["mismatches"],"K6 resume origin differs: "+arm)
            audit.check(boundary["actual_learning_rate"]==(3e-4 if arm=="E" else 3e-4*.96),"Arm learning-rate receipt differs: "+arm)
            checkpoints[arm]=endpoint["endpoint"]
        result["checkpoints"]=checkpoints
        chronology={}
        mappings=[]
        expected_runtime={"model_path":plan["target"],"speculative_algorithm":"DFLASH","speculative_num_draft_tokens":16,
            "dtype":"bfloat16","attention_backend":"fa3","tp_size":1,"max_running_requests":1,"context_length":4112,"random_seed":17,"skip_server_warmup":True}
        for arm in ARMS:
            prefix="tasks/"+arm+"_collect/"
            status=audit.original(prefix+"STATUS.json");collection=audit.original(prefix+"RESULT.json");features=audit.original(prefix+"FEATURES.json")
            engine=audit.original(prefix+"native/engine_settings.json");ready=audit.original(prefix+"native/ready.json")
            audit.check(status["status"]=="completed" and status["execution_commit"]==COLLECTION_COMMIT,"Collection task did not complete at pinned source: "+arm)
            audit.check(collection==features and collection["status"]=="completed" and collection["generated_responses"]==128 and collection["teacher_preparations"]==128 and collection["own_checkpoint"]==checkpoints[arm]["export"],"Collection endpoint/feature manifest differs: "+arm)
            audit.check(all(engine.get(k)==v for k,v in expected_runtime.items()) and engine["speculative_draft_model_path"]==checkpoints[arm]["export"],"Native engine pins differ: "+arm)
            audit.check(ready["status"]["status"]=="passed" and ready["status"]["publication"] is None,"Initial native worker state is not unmodified: "+arm)
            next_status=audit.original("tasks/"+arm+"8/STATUS.json")
            timing=schedule[arm+"_collect"]
            audit.check(timing["status"]=="completed" and timing["finished_epoch"]<=next_status["started_epoch"],"Own event-eight training preceded collection completion: "+arm)
            if arm!="N":audit.check(schedule[arm+"7"]["finished_epoch"]<=timing["started_epoch"],"Collection predates its endpoint: "+arm)
            chronology[arm]=dict(collection_start=timing["started_epoch"],collection_finish=timing["finished_epoch"],own_event8_training_start=next_status["started_epoch"],own_collection_precedes_training=True)
            actual={p.stem for p in (Path(delivery)/prefix/"records").glob("*.json")}
            audit.check(actual=={r["request_id"] for r in requests},"Missing/extra/duplicate collection records: "+arm)
            for ordinal,request in enumerate(requests):
                rid=request["request_id"];remote=BASE+"production_continuation_001/"+prefix+"records/"+rid+".json"
                record=audit.remote(remote)
                if record is None:continue
                audit.check(record["record_id"]==rid and record["request_index"]==request["request_index"] and record["session_id"]==f"persistent_{arm}_event8" and record["purpose"]=="serving" and record["checkpoint_id"]==checkpoints[arm]["export"],"Request/arm/checkpoint binding differs: "+arm+"/"+rid)
                audit.check(record["prompt_ids"]==request["prompt_ids"] and record["stop_token_ids"]==request["stop_token_ids"]==[151645,151643] and record["max_new_tokens"]==request["max_new_tokens"]==2048,"Exact prompt/stops/cap differ: "+arm+"/"+rid)
                meta=record["metadata"];capture=meta["capture"]
                native_path=meta["native_trace_path"]
                audit.check(native_path==BASE+"production_continuation_001/"+prefix+f"native/request_{ordinal+1:06d}.json","Native serial/order differs: "+arm+"/"+rid)
                native=audit.remote(native_path)
                audit.check(native==capture,"Independent persisted capture differs: "+arm+"/"+rid)
                audit.check(meta["feedback_capture_in_timed_request"] is True and meta["block_size"]==16 and meta["context_limit"]==4096 and meta["runtime"]=="pinned_SGLang_FA3" and meta["cache_policy"]=="idle_full_flush_before_every_request_and_publication","Record native configuration differs: "+arm+"/"+rid)
                audit.check(capture["status"]=="passed" and capture["collect_features"] is True and capture["native_capture_validated"] is True and capture["label_binding_scope"]=="native_prefill_and_committed_verification","Native capture not qualified: "+arm+"/"+rid)
                audit.check(capture["prompt_ids"]==record["prompt_ids"] and capture["response_ids"]==record["response_ids"]==capture["target_argmax_ids"] and capture["blocks"]==meta["trace"] and capture["covered_token_ids"]==(record["prompt_ids"]+record["response_ids"])[:-1] and capture["covered_rows"]==len(capture["covered_token_ids"]),"Native capture token/trace binding differs: "+arm+"/"+rid)
                mappings.append(dict(arm=arm,ordinal=ordinal,request_id=rid,source_id=request["source_id"],request_index=request["request_index"],
                    record_path=str(audit.local(remote)[0]),record_remote_path=remote,native_path=str(audit.local(native_path)[0]),native_remote_path=native_path,
                    checkpoint_id=record["checkpoint_id"]))
        result["chronology"]=chronology
        e8=schedule["E8"];sc=schedule["S_collect"]
        result["cross_arm_overlap"]=dict(E8_training_overlapped_S_collection=max(e8["started_epoch"],sc["started_epoch"])<min(e8["finished_epoch"],sc["finished_epoch"]),
            interpretation="Own-arm chronological validity passes; no global collection-before-all-training claim and no collection latency comparison.")
        result["runtime"]=dict(target=plan["target"],engine=expected_runtime,stop_token_ids=[151645,151643],max_new_tokens=2048,application_context_limit=4096,
            sampler=dict(temperature=0,top_p=1,top_k=1,ignore_eos=False),
            inherited_tokenizer_identities={k:v for k,v in preflight["inputs"].items() if "/target/" in k and ("tokenizer" in k or "config" in k)},
            runtime_contract=plan["runtime_contract"],model_files_reread=False)
        result["request_mappings"]=mappings
        result["expected_records"]=384;result["authenticated_records"]=len(mappings)
        audit.check(len(mappings)==384,"Not all384 generation/native pairs authenticated")
        result["status"]="passed" if not audit.errors else "incomplete"
    except (KeyError,ValueError,TypeError,FileNotFoundError) as error:
        audit.errors.append(type(error).__name__+": "+str(error))
    process_status={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in Path("/proc/self/status").read_text().splitlines() if ":" in line}
    imported_model_modules=[name for name in ("torch","transformers","sglang","safetensors") if name in sys.modules]
    audit.check(int(process_status["Threads"])==1 and not imported_model_modules,"CPU-only one-thread process contract differs")
    if audit.errors:result["status"]="incomplete"
    result.update(checks=audit.checks,errors=audit.errors,consumed_inputs=list(audit.inputs.values()),
        wall_s=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        proc_vm_peak_rss_bytes=int(process_status["VmHWM"].split()[0])*1024,observed_threads=int(process_status["Threads"]),imported_model_modules=imported_model_modules,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        limits="Provenance only; native-count/termination/cross-arm response validation and statistics belong to the separate reducer.")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"PROVENANCE.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result.get(k) for k in ("status","checks","errors","authenticated_records","wall_s","peak_rss_bytes")}))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery",type=Path,required=True)
    parser.add_argument("--qualification-delivery",type=Path,required=True)
    parser.add_argument("--restart-delivery",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise SystemExit("Set CUDA_VISIBLE_DEVICES empty for this CPU analysis")
    result=run(args.delivery,args.qualification_delivery,args.restart_delivery,args.output)
    raise SystemExit(result["status"]!="passed")
