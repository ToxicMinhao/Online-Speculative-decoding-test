"""Read-only, native matched-prefix DFlash blocks; never generate a response.

The scheduler RPC cold-prefills the exact canonical prefix (excluding the known
seed token), uses the native DFlash materializer and one native draft forward,
then stops immediately after the unchanged greedy vocabulary head. It does not
run target verification, append outputs, sample training anchors, or train.

Independent prefill can differ numerically from the original incremental cache.
Every frozen proposal is therefore checked against retained native evidence;
failed reconstruction is evidence, not a reason to replace or retry an anchor.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.dflash_correction_bank import native
from experiments.dflash_correction_bank.artifacts import atomic_json, binding, checked

base = native.base
require, digest, write_json = native.require, native.digest, native.write_json
STREAMS = ("wildchat_00", "wildchat_01", "wildchat_02", "wildchat_19", "personamem_persona42")
FINAL_TURNS = dict(zip(STREAMS, (20, 16, 17, 23, 18)))
MAX_ANCHORS = 160
MAX_BLOCKS = 320
BLOCK_SIZE = 10
PREFILL_CHUNK = 8192


def source_pins():
    return {name: native.sha(HERE / name) for name in ("native_probe.py", "bound.py")}


def validate_anchor(anchor):
    ids, output = anchor["input_ids"], anchor["output_ids"]
    require(isinstance(ids, list) and ids and isinstance(output, list) and output,
            "Exact saved request/input token IDs are required")
    require(all(type(t) is int and t >= 0 for t in ids + output), "Invalid canonical token IDs")
    start = anchor["block_output_start"]
    require(type(start) is int and 1 <= start < len(output), "Block starts after its already-known seed")
    prefix = ids + output[:start]
    require(anchor["canonical_prefix_ids"] == prefix and anchor["seed_token_id"] == output[start-1],
            "Canonical prefix/seed geometry differs from retained native block")
    require(anchor["canonical_next_ids"] == output[start:start+9], "Canonical next positions changed")
    expected_positions = list(range(len(ids)+start, len(ids)+start+9))
    require(anchor.get("expected_absolute_positions", expected_positions) == expected_positions,
            "Retained absolute proposal positions disagree with the canonical prefix")
    expected = anchor["expected_frozen_proposal_ids"]
    require(len(expected) == 9 and all(type(t) is int and t >= 0 for t in expected),
            "Retained native block must contain nine draft proposals, excluding its seed")
    require(anchor["stream"] in STREAMS and type(anchor["turn"]) is int and
            anchor["turn"] == FINAL_TURNS.get(anchor["stream"]),
            "Only the actual final saved request of each completed main stream is admitted")
    require(0 < len(prefix)-1 and len(prefix)-1+BLOCK_SIZE <= base.CONTEXT_LIMIT,
            "Native complete context plus block exceeds the qualified limit")
    return prefix[:-1]


def validate_plan(config):
    plan = config.get("attribution_probe_plan", [])
    require(isinstance(plan, list) and 0 < len(plan) <= MAX_ANCHORS, "One bounded predeclared plan is required")
    seen, per_stream, turns = set(), {}, {}
    seen_starts = set()
    verified_checkpoints, verified_projections = set(), set()
    for anchor in plan:
        validate_anchor(anchor)
        key = anchor["anchor_id"]
        require(isinstance(key, str) and key and key not in seen, "Duplicate/empty anchor identity")
        seen.add(key)
        position_key = (anchor["stream"], anchor["turn"], anchor["block_output_start"])
        require(position_key not in seen_starts, "Duplicate block start under a different anchor identity")
        seen_starts.add(position_key)
        stream = anchor["stream"]
        per_stream[stream] = per_stream.get(stream, 0)+1
        require(per_stream[stream] <= 32, "At most 32 anchors per final request")
        require(stream not in turns or turns[stream] == anchor["turn"], "One final saved request per stream")
        turns[stream] = anchor["turn"]
        require(set(anchor["checkpoints"]) == {"F", "OSD"}, "Only F and actual serving OSD checkpoints")
        require(anchor["checkpoints"]["F"] == config["released"], "Released checkpoint changed")
        for checkpoint in anchor["checkpoints"].values():
            identity = digest(checkpoint)
            if identity not in verified_checkpoints:
                base.verify_checkpoint(checkpoint)
                verified_checkpoints.add(identity)
        projection_identity = digest(anchor["projections"])
        if projection_identity not in verified_projections:
            checked(anchor["projections"])
            verified_projections.add(projection_identity)
    require(config.get("output_cap", 1024) == 1024 and config.get("block_size", 10) == 10 and
            base.OUTPUT_CAP == 1024 and base.BLOCK_SIZE == 10, "Pinned proposal geometry changed")
    return {anchor["anchor_id"]: anchor for anchor in plan}


def prefix_score(proposals, canonical, eos_ids):
    """Nine proposed positions only. EOS is a stopping boundary, never padding."""
    eos = set(eos_ids)
    matched = 0
    matched_eos = False
    boundary = "canonical_end"
    for proposal, wanted in zip(proposals, canonical):
        if proposal != wanted:
            boundary = "mismatch"
            break
        if proposal in eos:
            matched_eos = True
            boundary = "eos"
            break
        matched += 1
    else:
        if len(canonical) >= len(proposals):
            boundary = "block_end"
    return {"correct_proposed_prefix_length": matched, "boundary": boundary,
            "matched_eos": matched_eos, "prefix_including_eos": matched+int(matched_eos),
            "compared_positions": min(len(proposals), len(canonical))}


class ProbeComplete(BaseException):
    """Internal control flow after the one native draft forward/head, no verify."""
    def __init__(self, result):
        self.result = result


def inspect_head(hidden_states, lm_head, proposals, anchor, arm, out):
    """Capture native BF16 margins separately from the float64 fixed-space bound."""
    import numpy as np
    import torch
    from experiments.dflash_correction_bank.projections import FixedProjections, array_sha256
    from experiments.dflash_osd_attribution.bound import pairwise_capacity
    require(hidden_states.shape[0] == 9 and hidden_states.dtype == torch.bfloat16,
            "Exactly nine native post-normalization BF16 features required")
    shard = lm_head.shard_indices
    require(int(shard.num_added_elements) == 0 and int(shard.org_vocab_start_index) == 0,
            "Only the pinned single-rank base-vocabulary head is qualified")
    E = lm_head.weight
    vocab = int(shard.num_org_elements)
    require(E.dtype == torch.bfloat16 and E.shape[0] >= vocab, "Native fixed BF16 vocabulary head changed")
    # Same shape, dtype and matmul as the native one-chunk TP1 greedy path.
    logits = hidden_states @ E[:vocab].T
    require(torch.equal(logits.argmax(-1).to(torch.long), proposals.to(torch.long)),
            "Recomputed native BF16 head argmax differs from the unchanged native sampler")
    h = hidden_states.detach().float().cpu().numpy()
    frozen = arm == "F"
    projections = FixedProjections.load(checked(anchor["projections"])) if frozen else None
    canonical = anchor["canonical_next_ids"]
    rows, vocabulary_rows = [], []
    for index, y in enumerate(canonical):
        require(0 <= y < vocab, "Canonical token lies outside the fixed vocabulary")
        j = int(proposals[index])
        target_logit = logits[index, y].float()
        competitors = logits[index].clone()
        competitors[y] = -torch.inf
        best_other = int(competitors.argmax())
        other_logit = competitors[best_other].float()
        native_margin = float(target_logit-other_logit)
        row = dict(block_position=index, native_block_slot=index+1, absolute_output_position=anchor["block_output_start"]+index,
                   canonical_token_id=y, proposal_token_id=j, correct=j == y,
                   native_best_other_id=best_other, native_canonical_logit=float(target_logit),
                   native_best_other_logit=float(other_logit), native_bf16_margin=native_margin,
                   native_bf16_canonical_margin=native_margin)
        if frozen:
            # Stored h/E rows are exact float32 representations of BF16 source
            # tensors. The bound helper promotes these and fixed projections to
            # float64 before both deficit and bound calculations.
            E_y = E[y].detach().float().cpu().numpy()
            E_j = E[j].detach().float().cpu().numpy()
            capacity = pairwise_capacity(h[index], E_y, E_j, projections.U,
                projections.V_in, projections.kappa, frozen_error=j != y,
                osd_repaired=False, native_bf16_margin=native_margin)
            capacity["eligibility_note"] = "Computed before OSD pairing; classify only valid OSD-repaired frozen errors"
            row["capacity"] = capacity
            vocabulary_rows.append((E_y, E_j))
        rows.append(row)
    artifact = Path(out) / (digest([anchor["anchor_id"], arm])+".features.npz")
    require(not artifact.exists(), "Never overwrite sampled feature evidence")
    arrays = {"h": h, "canonical_ids": np.array(canonical, dtype=np.int64),
              "proposal_ids": proposals.detach().cpu().numpy()}
    if frozen:
        arrays.update(E_y=np.stack([p[0] for p in vocabulary_rows]),
                      E_j=np.stack([p[1] for p in vocabulary_rows]))
    np.savez(artifact, **arrays)
    return dict(proposal_ids=proposals.detach().cpu().tolist(), proposed_token_ids=proposals.detach().cpu().tolist(), positions=rows,
                feature_artifact=binding(artifact), native_feature_sha256=array_sha256(h),
                feature_location="Native DFlash normalized hidden_states passed unchanged to vocabulary head",
                native_precision="BF16 features, fixed BF16 E, BF16 matmul logits; margins subtracted in FP32",
                mathematical_precision="FP64 h/E pairwise dot and original fixed FP64 U/V_in with clipping",
                **prefix_score(proposals.detach().cpu().tolist(), canonical, anchor.get("eos_ids", [])))


def run_native_block(scheduler, anchor, arm, out):
    """Use native chunked prefill and stop before target verification/output."""
    import types
    import torch
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    prefix_ids = validate_anchor(anchor)
    native_entry = base._serving_entry(base._worker_config())
    native_entry.inherited.strict_idle(scheduler)
    require(scheduler.flush_cache(), "Cold cache flush failed before diagnostic")
    worker = scheduler.draft_worker
    require(worker.block_size == 10 and worker.tp_size == 1 and
            not worker.dynamic_verify_len and not worker.use_thresh_head_two_model,
            "Preserve pinned fixed-block single-rank DFlash without adaptive head")
    require(not worker.use_compact_draft_cache, "Qualified full draft-context cache is required")
    model = worker.draft_model
    versions = tuple((name, id(p), p._version) for name, p in model.named_parameters())
    original = worker._greedy_sample_from_vocab_parallel_head
    original_forward = worker.draft_model_runner.forward
    head_calls, forwards = 0, 0
    # Count native draft forward invocations. Prefill only materializes context
    # through the native fused-KV helper and must not execute another draft block.
    def counted_forward(*args, **kwargs):
        nonlocal forwards
        forwards += 1
        require(forwards <= 1, "More than one native draft block attempted")
        return original_forward(*args, **kwargs)
    def capture_head(_self, *, hidden_states, lm_head, chunk_size=256):
        nonlocal head_calls
        head_calls += 1
        require(head_calls == 1, "More than one native vocabulary head attempted")
        positions = worker._draft_block_positions_buf[0, 1:10].detach().cpu().tolist()
        expected_positions = list(range(len(anchor["input_ids"])+anchor["block_output_start"],
                                        len(anchor["input_ids"])+anchor["block_output_start"]+9))
        require(positions == expected_positions, "Native absolute block positions changed")
        proposals = original(hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size)
        result = inspect_head(hidden_states, lm_head, proposals, anchor, arm, out)
        result["native_absolute_proposal_positions"] = positions
        raise ProbeComplete(result)
    worker._greedy_sample_from_vocab_parallel_head = types.MethodType(capture_head, worker)
    worker.draft_model_runner.forward = counted_forward
    batch = None
    chunks = []
    try:
        sampling = SamplingParams(temperature=0, max_new_tokens=1024, ignore_eos=False)
        sampling.normalize(scheduler.tokenizer)
        req = Req(rid="attribution-"+digest([anchor["anchor_id"], arm]), origin_input_text="",
                  origin_input_ids=prefix_ids, sampling_params=sampling,
                  eos_token_ids=set(anchor.get("eos_ids", [])), vocab_size=scheduler.model_config.vocab_size)
        req.tokenizer = scheduler.tokenizer
        with torch.inference_mode():
            for start in range(0, len(prefix_ids), PREFILL_CHUNK):
                stop = min(start+PREFILL_CHUNK, len(prefix_ids))
                req.fill_ids = prefix_ids[:stop]
                if start:
                    # Native pool indices are int32, but the prefix-copy kernel
                    # consumes int64 pointers. Preserve values/device and own
                    # the copy rather than passing the pool's storage/view.
                    req.prefix_indices = scheduler.req_to_token_pool.req_to_token[
                        req.req_pool_idx, :start].to(dtype=torch.int64, copy=True)
                req.set_extend_input_len(stop-start)
                req.is_chunked = int(stop < len(prefix_ids))
                batch = ScheduleBatch.init_new([req], scheduler.req_to_token_pool,
                    scheduler.token_to_kv_pool_allocator, scheduler.tree_cache,
                    scheduler.model_config, scheduler.enable_overlap, scheduler.spec_algorithm,
                    chunked_req=req if stop < len(prefix_ids) else None)
                batch.prepare_for_extend()
                worker.forward_batch_generation(batch)
                chunks.append(stop-start)
            require(batch is not None and int(batch.seq_lens[0]) == len(prefix_ids),
                    "Native prefix length changed")
            reconstructed_seed = int(batch.spec_info.verified_id[0])
            # Anchor is recorded canonical input, not a newly generated answer.
            # A different target prefill argmax is retained as invalid evidence.
            batch.spec_info.verified_id.fill_(anchor["seed_token_id"])
            batch.forward_mode = ForwardMode.DECODE
            worker._prepare_for_speculative_decoding(batch, batch.spec_info)
        raise RuntimeError("Diagnostic returned without the expected one-block stop")
    except ProbeComplete as complete:
        require(forwards == head_calls == 1, "Diagnostic block accounting differs")
        value = complete.result
        frozen_reproduced = (value["proposal_ids"] == anchor["expected_frozen_proposal_ids"]) if arm == "F" else None
        value.update(status="passed", anchor_id=anchor["anchor_id"], arm=arm,
                     canonical_prefix_ids=anchor["canonical_prefix_ids"],
                     checkpoint_identity_sha256=anchor["checkpoints"][arm]["checkpoint_identity_sha256"],
                     prefix_input_ids_sha256=digest(prefix_ids), canonical_prefix_ids_sha256=digest(anchor["canonical_prefix_ids"]),
                     seed_token_id=anchor["seed_token_id"], reconstructed_target_seed_id=reconstructed_seed,
                     target_seed_matches=reconstructed_seed == anchor["seed_token_id"],
                     frozen_proposals_match_retained=frozen_reproduced,
                     valid_reconstruction=(reconstructed_seed == anchor["seed_token_id"] and frozen_reproduced is not False),
                     native_draft_block_evaluations=forwards, draft_block_evaluations=forwards, native_head_calls=head_calls,
                     cache_valid=True, no_generated_history=True, no_training=True,
                     native_hook_passed=True,
                     target_prefill_chunks=chunks, target_verifications=0, regenerated_response_tokens=0,
                     training_steps=0, cache_policy="Cold target/draft pools before and after each anchor/checkpoint",
                     timing_scope="Local diagnostic forwards only; not deployment latency or saved-time attribution")
        return value
    finally:
        worker._greedy_sample_from_vocab_parallel_head = original
        worker.draft_model_runner.forward = original_forward
        require(versions == tuple((name, id(p), p._version) for name, p in model.named_parameters()),
                "Diagnostic changed draft parameters")
        require(scheduler.flush_cache(), "Cold cache flush failed after diagnostic")


def _scheduler_rpc(self, anchor_id, arm):
    config = base._worker_config()
    plan = {a["anchor_id"]: a for a in config["attribution_probe_plan"]}
    require(anchor_id in plan and arm in ("F", "OSD"), "Undeclared diagnostic anchor/arm")
    anchor = plan[anchor_id]
    expected = anchor["checkpoints"][arm]
    require(self._d181_current_draft == expected["path"], "Native active checkpoint differs from planned serving state")
    out = Path(config["worker_out"]) / "blocks"
    out.mkdir(exist_ok=True)
    value = run_native_block(self, anchor, arm, out)
    value["checkpoint"] = expected
    atomic_json(out / (digest([anchor_id, arm])+".json"), value)


def _probe_scheduler(*args, **kwargs):
    config = base._worker_config()
    entry = base._serving_entry(config)
    from sglang.srt.managers.scheduler import Scheduler
    Scheduler.attribution_probe = _scheduler_rpc
    # No bank routing/correction hooks, capture hooks or learned training path.
    return entry.serving_scheduler(*args, **kwargs)


class ProbeService(base._ServingService):
    def __init__(self, config):
        self.plan = validate_plan(config)
        self.used = set()
        self.claims = Path(config["worker_out"]) / "claims"
        self.claims.mkdir(exist_ok=False)
        base._sd_scheduler = _probe_scheduler
        super().__init__(config)

    def _publish_exact(self, checkpoint):
        if self.active == checkpoint:
            return {"status": "passed", "activation_s": 0., "already_active": True}
        base.verify_checkpoint(checkpoint)
        self.bridge._two_update_admitted.add(Path(checkpoint["path"]))
        started = time.perf_counter()
        receipt = self.bridge.publish({"model_path": checkpoint["path"], "reset": True})
        require(receipt["status"] == "passed" and receipt["success"] and
                receipt["helper_matches_current_layers"] and receipt["helper_norm_matches_current_layers"] and
                receipt["cache_after"]["tree_total_size"] == 0, "Publication/cache helper refresh failed")
        require(self.bridge.state()["draft_model_path"] == checkpoint["path"], "Wrong native checkpoint")
        self.active = checkpoint
        write_json(self.out / "ACTIVE_CHECKPOINT.json", checkpoint)
        return {"status": "passed", "activation_s": time.perf_counter()-started, "native_receipt": receipt}

    def probe(self, body):
        anchor_id, arm = body["anchor_id"], body["arm"]
        require(anchor_id in self.plan and arm in ("F", "OSD"), "Only planned anchor/arm pairs")
        anchor = self.plan[anchor_id]
        require(body["checkpoint"] == anchor["checkpoints"][arm], "Checkpoint identity changed")
        key = (anchor_id, arm)
        require(key not in self.used and len(self.used) < MAX_BLOCKS, "No repeated or extra block evaluations")
        claim = self.claims / (digest(list(key))+".json")
        require(not claim.exists(), "Previously claimed block may not be retried")
        atomic_json(claim, {"anchor_id": anchor_id, "arm": arm, "checkpoint": body["checkpoint"],
                           "claimed_before_model_work": True, "training": False})
        self.used.add(key)
        publication = self._publish_exact(body["checkpoint"])
        self.flush({})
        self.engine.collective_rpc("attribution_probe", anchor_id=anchor_id, arm=arm)
        value = json.loads((self.out / "blocks" / (digest(list(key))+".json")).read_text())
        value["publication"] = publication
        self.flush({})
        return value

    def generate(self, _):
        raise RuntimeError("Response generation is forbidden in this diagnostic")


class NativeProbeClient:
    """Starts no training worker; each evaluate consumes one declared block."""
    def __init__(self, config, out):
        self.config = dict(config)
        self.config["attribution_probe_source_sha256"] = source_pins()
        validate_plan(self.config)
        previous = base.worker_command
        def command(cfg):
            value = native._original_worker_command(cfg)
            require(value[-1] == str(base.HERE / "native.py"), "Native worker command changed")
            value[-1] = str(HERE / "native_probe.py")
            return value
        base.worker_command = command
        try:
            self.worker = base._WorkerClient(self.config, "sd", Path(out))
        finally:
            base.worker_command = previous

    def evaluate(self, anchor, checkpoint, arm, projections=None):
        planned = next(a for a in self.config["attribution_probe_plan"] if a["anchor_id"] == anchor["anchor_id"])
        require(anchor == planned and checkpoint == planned["checkpoints"][arm], "Declared anchor/checkpoint changed")
        if projections is not None:
            require(projections == planned["projections"], "Fixed projections changed")
        return self.worker.call("probe", {"anchor_id": anchor["anchor_id"], "arm": arm, "checkpoint": checkpoint})

    def close(self):
        self.worker.close()


def worker_main():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    config = base._worker_config()
    require(config.get("attribution_probe_source_sha256") == source_pins(), "Probe source changed")
    native.check_sources(config)
    native.check_numerical_environment()
    base.check_dependencies(config["workspace_root"])
    allocation = os.environ.pop("CUDA_VISIBLE_DEVICES")
    try:
        base.check_environment(config)
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = allocation
    require(config["kind"] == "sd", "This diagnostic has no trainer worker")
    service = ProbeService(config)
    poisoned = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        def log_message(self, *_):
            pass
        def do_POST(self):
            try:
                require(self.headers.get("X-Two-Update-Token") == config["rpc_token"], "Foreign client")
                length = int(self.headers.get("Content-Length", "0"))
                require(0 < length <= 64*2**20 and not poisoned, "Invalid payload or poisoned worker")
                body = json.loads(self.rfile.read(length))
                action = self.path.lstrip("/")
                require(action in ("ready", "probe"), "Action outside bounded diagnostic")
                result = {"status": "passed", "pid": os.getpid()} if action == "ready" else service.probe(body)
                status = 200
            except Exception as error:
                poisoned.append(repr(error))
                result = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
                atomic_json(Path(config["worker_out"])/f"FAILURE_{len(poisoned)}.json", result)
                status = 500
            payload = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = HTTPServer(("127.0.0.1", config["port"]), Handler)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    require(sys.argv[1:] == ["--worker"], "Use reviewed attribution controller")
    worker_main()
