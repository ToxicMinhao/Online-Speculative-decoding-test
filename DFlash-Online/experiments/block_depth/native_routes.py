"""Isolated 3/5-layer serving routes for the frozen K+ depth feasibility screen.

The original worker still owns committed-prefix KV, temporary allocations,
verification and its single vocabulary projection. FIVE delegates to its original
runner. Other routes use separate CUDA graphs over the native C1 graph buffers;
no original graph or author model method is replaced.
"""
from __future__ import annotations

import copy
import math
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


MODES = {"FIVE", "THREE", "FORCED3", "FORCED5", "ADAPTIVE3", "ADAPTIVE5", "ALTERNATE", "ADAPTIVE"}


class ShallowReadout(nn.Module):
    """Native sidecar norm and rank-32 residual readout; no vocabulary weights."""

    def __init__(self, final_norm, hidden_size, seed=17, rank=32):
        super().__init__()
        self.norm = copy.deepcopy(final_norm)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.empty(rank, hidden_size, dtype=torch.float32)
        initial.uniform_(-1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size), generator=generator)
        self.A = nn.Parameter(initial.to(final_norm.weight))
        self.B = nn.Parameter(torch.zeros(hidden_size, rank, device=final_norm.weight.device,
                                         dtype=final_norm.weight.dtype))

    def features(self, hidden, residual):
        # Native fused RMSNorm mutates both arguments. The original residual pair
        # must remain available, unchanged, if this round continues to layer four.
        normalized, _ = self.norm(hidden.clone(), residual.clone())
        return normalized, F.linear(normalized, self.A)

    def finish(self, normalized, features):
        return normalized + F.linear(features, self.B)


class DepthGate(nn.Module):
    """97 features: three pooled rank-32 groups and log(1 + prefix length)."""

    def __init__(self, device):
        super().__init__()
        self.register_buffer("weight", torch.zeros(97, device=device, dtype=torch.float32))
        self.register_buffer("intercept", torch.zeros((), device=device, dtype=torch.float32))
        self.register_buffer("kappa_ms_per_token", torch.ones((), device=device, dtype=torch.float32))
        self.register_buffer("continue_cost_ms", torch.zeros((), device=device, dtype=torch.float32))

    def forward(self, rank_features, prefix_length):
        proposals = rank_features[1:16].float()
        features = torch.cat((proposals[:3].mean(0), proposals[3:7].mean(0),
                              proposals[7:15].mean(0), torch.log1p(prefix_length.float()).reshape(1)))
        predicted_progress = features @ self.weight + self.intercept
        return self.kappa_ms_per_token * predicted_progress - self.continue_cost_ms


def run_layers(model, batch, start, end, hidden, residual=None):
    """Keep the original fused-residual layer calling convention intact."""
    for layer in model.layers[start:end]:
        hidden, residual = layer(batch.positions, hidden, batch, residual)
    return hidden, residual


class EagerSegments:
    """CPU fixture implementation of the same graph boundaries, not a fallback."""

    def __init__(self, controller):
        self.controller = controller
        self.setup_records = []
        self.launches = {"static_three": 0, "prefix": 0, "shallow": 0, "deep": 0}

    def prepare(self, batch):
        self.batch = batch
        self.controller.runner.attn_backend.init_forward_metadata(batch)

    def static_three(self):
        self.launches["static_three"] += 1
        model, readout = self.controller.model, self.controller.readout
        hidden, residual = run_layers(model, self.batch, 0, 3, self.batch.input_embeds.clone())
        return readout.finish(*readout.features(hidden, residual))

    def prefix(self):
        self.launches["prefix"] += 1
        model, readout = self.controller.model, self.controller.readout
        self.hidden, self.residual = run_layers(model, self.batch, 0, 3, self.batch.input_embeds.clone())
        self.normalized, self.rank_features = readout.features(self.hidden, self.residual)
        return self.controller.gate(self.rank_features, self.batch.seq_lens[0])

    def shallow(self):
        self.launches["shallow"] += 1
        return self.controller.readout.finish(self.normalized, self.rank_features)

    def deep(self):
        self.launches["deep"] += 1
        hidden, residual = run_layers(self.controller.model, self.batch, 3, 5,
                                      self.hidden.clone(), self.residual.clone())
        return self.controller.model.norm(hidden, residual)[0]


class CudaSegments(EagerSegments):
    """Four graphs sharing the native runner's stable C1 inputs and FA3 metadata.

    Each graph has its own allocation pool. The original full graph is neither
    recaptured nor redirected. Native replay_prepare refreshes the same persistent
    attention metadata and input addresses before every route, including shadows.
    """

    def __init__(self, controller):
        super().__init__(controller)
        self.native_graph = controller.runner.graph_runner
        if self.native_graph is None:
            raise RuntimeError("Depth pricing requires the qualified native CUDA graph runner")
        self.graphs = {}
        self.outputs = {}

    def prepare(self, batch):
        if batch.batch_size != 1 or batch.input_ids.numel() != 16:
            raise ValueError("Depth routes require C1 and the original 16-slot block")
        if not self.native_graph.can_run(batch):
            raise RuntimeError("Native graph does not support this matched round; no eager substitution")
        self.native_graph.replay_prepare(batch)
        if self.native_graph.bs != 1 or self.native_graph.raw_num_token != 16:
            raise RuntimeError("Native graph padding changed the qualified C1 block")
        if not self.graphs:
            self._capture(batch)

    def _capture(self, source_batch):
        buffers = self.native_graph.buffers
        self.batch = copy.copy(source_batch)
        for name, length in (("input_ids", 16), ("positions", 16), ("out_cache_loc", 16),
                             ("input_embeds", 16), ("req_pool_indices", 1), ("seq_lens", 1),
                             ("seq_lens_cpu", 1)):
            setattr(self.batch, name, getattr(buffers, name)[:length])
        self.batch.orig_seq_lens = self.batch.seq_lens
        self.batch.num_token_non_padded = buffers.num_token_non_padded
        self.batch.capture_hidden_mode = self.native_graph.capture_hidden_mode
        self.batch.global_forward_mode = self.batch.forward_mode
        # C1 Qwen3 has no tensor-parallel collectives or MoE/DP state to synthesize.
        stream = torch.cuda.Stream(device=buffers.input_embeds.device)
        stream.wait_stream(torch.cuda.current_stream())
        self._capture_one("static_three", self._static_three_body, stream, source_batch)
        self._capture_one("prefix", self._prefix_body, stream, source_batch)
        self._capture_one("shallow", self._shallow_body, stream, source_batch)
        self._capture_one("deep", self._deep_body, stream, source_batch)
        torch.cuda.current_stream().wait_stream(stream)
        self.native_graph.replay_prepare(source_batch)

    def _capture_one(self, name, function, stream, source_batch):
        started = time.perf_counter()
        # Warm-up and capture are explicitly outside pricing. Refreshing source
        # inputs also undoes native fused-normalization mutations of those inputs.
        for _ in range(2):
            self.native_graph.replay_prepare(source_batch)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                function()
            stream.synchronize()
        self.native_graph.replay_prepare(source_batch)
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        layer_calls = [0] * 5
        hooks = []
        for index, layer in enumerate(self.controller.model.layers):
            def observed(module, args, index=index):
                layer_calls[index] += 1
            hooks.append(layer.register_forward_pre_hook(observed))
        try:
            with torch.cuda.graph(graph, stream=stream):
                output = function()
        finally:
            for hook in hooks:
                hook.remove()
        stream.synchronize()
        self.graphs[name], self.outputs[name] = graph, output
        self.setup_records.append({"graph": name, "warmup_executions": 2, "capture_executions": 1,
                                   "captured_layer_calls": layer_calls,
                                   "wall_s": time.perf_counter() - started})

    def _static_three_body(self):
        hidden, residual = run_layers(self.controller.model, self.batch, 0, 3,
                                      self.batch.input_embeds)
        return self.controller.readout.finish(*self.controller.readout.features(hidden, residual))

    def _prefix_body(self):
        self.hidden, self.residual = run_layers(self.controller.model, self.batch, 0, 3,
                                              self.batch.input_embeds)
        self.normalized, self.rank_features = self.controller.readout.features(self.hidden, self.residual)
        return self.controller.gate(self.rank_features, self.batch.seq_lens[0])

    def _shallow_body(self):
        return self.controller.readout.finish(self.normalized, self.rank_features)

    def _deep_body(self):
        hidden, residual = run_layers(self.controller.model, self.batch, 3, 5,
                                      self.hidden.clone(), self.residual.clone())
        return self.controller.model.norm(hidden, residual)[0]

    def _replay(self, name):
        self.launches[name] += 1
        self.graphs[name].replay()
        return self.outputs[name]

    def static_three(self):
        return self._replay("static_three")

    def prefix(self):
        return self._replay("prefix")

    def shallow(self):
        return self._replay("shallow")

    def deep(self):
        return self._replay("deep")


class DepthRoutes:
    """Instance-local adapter, installed only in the dedicated experiment worker.

    set_mode chooses a whole-block route. before_round may collect matched-prefix
    shadow evidence; proposal() bypasses that callback and never advances history.
    close() removes interception, restoring exactly the original bound method.
    """

    def __init__(self, worker, *, seed=17, cpu_fixture=False):
        self.worker = worker
        self.runner = worker.draft_model_runner
        self.model = worker.draft_model
        if len(self.model.layers) != 5 or int(worker.block_size) != 16:
            raise ValueError("This experiment is fixed to the five-layer b16 K+ export")
        self.original_forward = self.runner.forward
        self.cpu_fixture = cpu_fixture
        self.readout = ShallowReadout(self.model.norm, int(self.model.config.hidden_size), seed)
        self.gate = DepthGate(self.model.norm.weight.device)
        self.segments = EagerSegments(self) if cpu_fixture else CudaSegments(self)
        self.mode = "FIVE"
        self.round_index = 0
        self.before_round = None
        self.routes = []
        self.audit = False
        self.audit_records = []
        self.audit_output_path = None
        self.head_invocations = 0
        self.original_head = worker._greedy_sample_from_vocab_parallel_head

    def set_mode(self, mode, audit=False):
        if mode not in MODES:
            raise ValueError("Unknown depth route: " + str(mode))
        self.mode, self.round_index = mode, 0
        self.routes.clear()
        self.audit_records.clear()
        self.head_invocations = 0
        self.audit = audit
        self.runner.forward = self.original_forward if mode == "FIVE" else self.forward
        self.worker._greedy_sample_from_vocab_parallel_head = self._counted_head if audit else self.original_head

    def _counted_head(self, *args, **kwargs):
        self.head_invocations += 1
        return self.original_head(*args, **kwargs)

    @torch.no_grad()
    def forward(self, batch, *args, **kwargs):
        if self.before_round is not None:
            self.before_round(self, batch)
        if self.audit:
            original, _ = self.proposal(batch, "FIVE")
            routed, _ = self.proposal(batch, "FORCED5")
            original_ids, routed_ids = original.cpu().tolist(), routed.cpu().tolist()
            valid = original_ids == routed_ids
            self.audit_records.append({"round_index": self.round_index,
                                       "canonical_start": int(batch.positions[0].item()),
                                       "original_five_proposals": original_ids,
                                       "forced_five_proposals": routed_ids, "all_15_equal": valid})
            if self.audit_output_path is not None:
                from online_feedback.storage import write_json
                write_json(self.audit_output_path, self.receipt())
            if not valid:
                raise RuntimeError("Forced-five differs from native FIVE at identical live prefix")
        output, route = self._forward(batch, self.mode, *args, **kwargs)
        self.routes.append(route)
        self.round_index += 1
        return output

    def _forward(self, batch, mode, *args, **kwargs):
        mode = {"ADAPTIVE3": "FORCED3", "ADAPTIVE5": "FORCED5"}.get(mode, mode)
        if mode == "FIVE":
            return self.original_forward(batch, *args, **kwargs), 5
        self.segments.prepare(batch)
        if mode == "THREE":
            hidden, depth = self.segments.static_three(), 3
        else:
            # Even forced adaptive modes compute the real gate and incur the same
            # device-to-host synchronization before the explicit route override.
            continue_score = float(self.segments.prefix().item())
            depth = 5 if continue_score >= 0 else 3
            if mode in {"FORCED3", "FORCED5"}:
                depth = int(mode[-1])
            elif mode == "ALTERNATE":
                depth = (3, 5)[self.round_index % 2]
            hidden = self.segments.deep() if depth == 5 else self.segments.shallow()
        if self.cpu_fixture:
            return SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden), can_run_graph=False), depth
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput
        return ModelRunnerOutput(logits_output=LogitsProcessorOutput(next_token_logits=None,
                                 hidden_states=hidden), can_run_graph=True), depth

    @torch.no_grad()
    def proposal(self, batch, mode):
        """Replay one route at the live canonical start, without verifier/history work.

        Caller times this method through return with CUDA synchronization. Shared
        cache append/allocation that precedes runner entry is charged separately.
        No output is committed; the live route must run again after all shadows.
        """
        output, depth = self._forward(batch, mode)
        hidden = output.logits_output.hidden_states.view(1, 16, -1)[:, 1:].reshape(15, -1)
        target = self.worker.target_worker.model_runner.model
        tokens = self.worker._greedy_sample_from_vocab_parallel_head(hidden_states=hidden,
                                                                    lm_head=target.lm_head)
        return tokens.clone(), depth

    def receipt(self):
        return {"status": "passed" if all(row["all_15_equal"] for row in self.audit_records) else "failed",
                "mode": self.mode, "live_rounds": len(self.routes), "routes": list(self.routes),
                "graph_launches": dict(self.segments.launches),
                "graph_setup": list(self.segments.setup_records),
                "original_five_unchanged": True, "feature_width": 97, "readout_rank": 32,
                "audit_enabled": self.audit, "audit_records": list(self.audit_records),
                "head_invocations": self.head_invocations,
                "shadow_proposal_forwards": 2 * len(self.audit_records),
                "shadow_vocab_projections": 2 * len(self.audit_records)}

    def status(self):
        return self.receipt()

    def profile_completed_request(self, record, *, prompt_ordinal, output_path):
        return profile_completed_request(self, record["prompt_ids"], record["response_ids"],
            record["metadata"]["trace"], record["record_id"], prompt_ordinal, output_path)

    def close(self):
        self.runner.forward = self.original_forward
        self.worker._greedy_sample_from_vocab_parallel_head = self.original_head


def selected_round_indices(round_count, maximum=8):
    """Distinct evenly spaced native FIVE starts, including both endpoints."""
    count = min(round_count, maximum)
    if count < 2:
        return list(range(count))
    return [(index * (round_count - 1)) // (count - 1) for index in range(count)]


def bitwise_equal(first, second):
    """Compare storage bits, including NaNs in legitimately unused cache slots."""
    return (first.shape == second.shape and first.dtype == second.dtype and
            torch.equal(first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8)))


def _save_tensor_fields(value, names):
    return {name: getattr(value, name).clone() for name in names
            if isinstance(getattr(value, name, None), torch.Tensor)}


def _restore_tensor_fields(value, saved):
    for name, tensor in saved.items():
        getattr(value, name).copy_(tensor)


@torch.inference_mode()
def profile_completed_request(controller, prompt_ids, response_ids, blocks, request_id,
                              prompt_ordinal, output_path=None):
    """Replay recorded FIVE starts while its completed request's KV still exists.

    Call at idle immediately after online_finish and before the next cache flush.
    The freed request row and KV bytes are still retained by the pinned allocator.
    Clone before allocating any new slots, since freed slots can overlap them.
    Both original and scratch cache bytes, mapping and allocator state are restored.
    """
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from online_feedback.storage import write_json

    worker, runner = controller.worker, controller.runner
    if worker.page_size != 1 or worker.use_compact_draft_cache:
        raise RuntimeError("Idle prefix replay requires the qualified full-context page-size-one cache")
    pool, allocator = runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    if not pool.free_slots:
        raise RuntimeError("Completed request did not release a request-pool row")
    row = pool.free_slots[-1]
    if not blocks:
        raise RuntimeError("No native round exists for bounded depth pricing")
    # Only prefixes consumed by historical draft rounds are needed. The final
    # verifier suffix may not have been appended once the request finished.
    committed_length = max(int(block["positions"][0]) for block in blocks)
    saved_row = pool.req_to_token[row].clone()
    old_locations = saved_row[:committed_length].to(torch.int64)
    if old_locations.unique().numel() != committed_length or committed_length <= 0:
        raise RuntimeError("Completed request mapping is missing or overlaps; replay is unavailable")
    cache_pairs = [runner.token_to_kv_pool.get_kv_buffer(layer.self_attn.attn.layer_id)
                   for layer in controller.model.layers]
    prefix_cache = [(key[old_locations].clone(), value[old_locations].clone())
                    for key, value in cache_pairs]
    saved_allocator = allocator.backup_state()
    saved_free_rows = list(pool.free_slots)
    graph = runner.graph_runner
    input_names = ("input_ids", "positions", "out_cache_loc", "input_embeds", "req_pool_indices",
                   "seq_lens", "seq_lens_cpu", "num_token_non_padded", "custom_mask")
    saved_inputs = _save_tensor_fields(graph.buffers, input_names)
    metadata = runner.attn_backend.target_verify_metadata[1]
    saved_metadata = _save_tensor_fields(metadata, vars(metadata))
    metadata_scalars = {name: value for name, value in vars(metadata).items()
                        if isinstance(value, (int, float, bool, type(None)))}
    saved_forward_metadata = runner.attn_backend.forward_metadata
    saved_graph_scalars = {name: getattr(graph, name, None)
                           for name in ("raw_bs", "raw_num_token", "bs", "_replay_ntpb")}
    started = time.perf_counter()
    slots, overwritten = None, None
    receipt = {"request_id": request_id, "prompt_ordinal": prompt_ordinal,
               "native_rounds": len(blocks), "sampled_round_indices": selected_round_indices(len(blocks)),
               "timing_scope": "Native graph preparation, route dispatch and one vocabulary projection; "
                               "shared committed-KV append and allocator setup excluded",
               "source": "Exact retained post-completion native FIVE committed KV, before next flush",
               "warmups": [], "measurements": [], "parity": [], "status": "running",
               "active_operation": None, "restoration_status": "pending"}
    try:
        slots = allocator.alloc(committed_length + 16)
        if slots is None:
            raise RuntimeError("Insufficient scratch KV slots for bounded idle replay")
        slots = slots.to(torch.int64)
        overwritten = [(key[slots].clone(), value[slots].clone()) for key, value in cache_pairs]
        prefix_slots, speculative_slots = slots[:committed_length], slots[committed_length:]
        for (key, value), (original_key, original_value) in zip(cache_pairs, prefix_cache):
            key[prefix_slots] = original_key
            value[prefix_slots] = original_value
        device = slots.device
        target = worker.target_worker.model_runner.model
        embed = target.get_input_embeddings()
        worker._ensure_draft_block_buffers(1)
        first_setup = time.perf_counter()
        for ordinal, block_index in enumerate(receipt["sampled_round_indices"]):
            block = blocks[block_index]
            anchor = len(prompt_ids) + int(block["output_start"])
            pool.req_to_token[row, :committed_length] = prefix_slots.to(pool.req_to_token.dtype)
            pool.req_to_token[row, anchor:anchor + 16] = speculative_slots.to(pool.req_to_token.dtype)
            ids = torch.full((16,), worker._mask_token_id, device=device, dtype=torch.long)
            ids[0] = int(response_ids[block["output_start"]])
            positions = torch.arange(anchor, anchor + 16, device=device, dtype=torch.int64)
            batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY, batch_size=1, input_ids=ids,
                req_pool_indices=torch.tensor([row], device=device, dtype=torch.int64),
                seq_lens=torch.tensor([anchor], device=device, dtype=torch.int32),
                seq_lens_cpu=torch.tensor([anchor], dtype=torch.int32), seq_lens_sum=anchor,
                out_cache_loc=speculative_slots, positions=positions,
                req_to_token_pool=pool, token_to_kv_pool=runner.token_to_kv_pool,
                attn_backend=runner.attn_backend, input_embeds=embed(ids),
                spec_algorithm=SpeculativeAlgorithm.DFLASH, spec_info=worker._draft_block_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL)
            expected = list(block["draft_tokens"])
            if len(expected) != 15 or block["positions"] != list(range(anchor, anchor + 16)):
                raise RuntimeError("Retained round geometry is not the original 15-proposal block")
            # Explicit graph build precedes all timing. It executes no verifier.
            receipt["active_operation"] = {"kind": "graph_preparation", "block_index": block_index}
            if output_path is not None:
                write_json(output_path, receipt)
            controller.segments.prepare(batch)
            torch.cuda.synchronize()
            receipt["active_operation"] = None
            if ordinal == 0:
                receipt["scratch_and_graph_setup_s"] = time.perf_counter() - first_setup
            routes = ("FIVE", "THREE", "ADAPTIVE3", "ADAPTIVE5")
            for mode in routes:
                receipt["active_operation"] = {"kind": "warmup", "block_index": block_index, "mode": mode}
                if output_path is not None:
                    write_json(output_path, receipt)
                begin = time.perf_counter()
                tokens, _ = controller.proposal(batch, mode)
                torch.cuda.synchronize()
                values = tokens.cpu().tolist()
                receipt["warmups"].append({"block_index": block_index, "mode": mode,
                                           "wall_s": time.perf_counter() - begin})
                receipt["active_operation"] = None
                if mode in {"FIVE", "ADAPTIVE5"}:
                    valid = values == expected
                    receipt["parity"].append({"block_index": block_index, "mode": mode,
                                               "all_15_equal": valid, "observed": values,
                                               "expected": expected})
                    if not valid:
                        raise RuntimeError("Historical native FIVE proposal reconstruction failed")
            for repeat in range(4):
                offset = (prompt_ordinal + ordinal + repeat) % 4
                order = routes[offset:] + routes[:offset]
                for order_index, mode in enumerate(order):
                    receipt["active_operation"] = {"kind": "timed_route", "round_index": block_index,
                                                   "repeat": repeat, "route": mode}
                    if output_path is not None:
                        write_json(output_path, receipt)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    tokens, depth = controller.proposal(batch, mode)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    values = tokens.cpu().tolist()
                    valid = mode not in {"FIVE", "ADAPTIVE5"} or values == expected
                    receipt["measurements"].append({
                        "request_id": request_id, "prompt_ordinal": prompt_ordinal,
                        "round_index": block_index, "anchor_ordinal": ordinal,
                        "prefix_length": anchor, "route": mode, "depth": depth,
                        "repeat": repeat, "route_position": order_index, "elapsed_ms": 1000 * elapsed,
                        "native_rounds": len(blocks), "sampled_rounds": len(receipt["sampled_round_indices"]),
                        "valid": valid, "proposal_ids": values})
                    receipt["active_operation"] = None
                    if output_path is not None:
                        write_json(output_path, receipt)
                    if not valid:
                        raise RuntimeError("Native FIVE parity changed during route pricing")
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        restore_started = time.perf_counter()
        try:
            if overwritten is not None:
                for (key, value), (previous_key, previous_value) in zip(cache_pairs, overwritten):
                    key[slots] = previous_key
                    value[slots] = previous_value
            pool.req_to_token[row].copy_(saved_row)
            allocator.restore_state(saved_allocator)
            _restore_tensor_fields(graph.buffers, saved_inputs)
            _restore_tensor_fields(metadata, saved_metadata)
            for name, value in metadata_scalars.items():
                setattr(metadata, name, value)
            runner.attn_backend.forward_metadata = saved_forward_metadata
            for name, value in saved_graph_scalars.items():
                setattr(graph, name, value)
            torch.cuda.synchronize()
            receipt["restoration_checks"] = {
                "request_row_ids": pool.free_slots == saved_free_rows,
                "request_mapping": torch.equal(pool.req_to_token[row], saved_row),
                "allocator": all(torch.equal(actual, saved) for actual, saved in
                                 zip(allocator.backup_state(), saved_allocator)),
                "native_graph_inputs": all(bitwise_equal(getattr(graph.buffers, name), value)
                                           for name, value in saved_inputs.items()),
                "attention_metadata": all(bitwise_equal(getattr(metadata, name), value)
                                          for name, value in saved_metadata.items()),
                "cache_bytes": overwritten is None or all(
                    bitwise_equal(key[slots], previous_key) and bitwise_equal(value[slots], previous_value)
                    for (key, value), (previous_key, previous_value) in zip(cache_pairs, overwritten)),
            }
            receipt["restored_request_rows"] = receipt["restoration_checks"]["request_row_ids"]
            if not all(receipt["restoration_checks"].values()):
                raise RuntimeError("Idle prefix replay did not restore exact native state")
            receipt["restoration_status"] = "passed"
        except BaseException as error:
            receipt.update(status="failed", restoration_status="failed", restoration_error=repr(error))
            raise
        finally:
            receipt["restoration_s"] = time.perf_counter() - restore_started
            receipt["total_wall_s"] = time.perf_counter() - started
            receipt["route_state"] = controller.receipt()
            active = receipt["active_operation"]
            receipt["accounting"] = {
                "planned_timed_measurements": 16 * len(receipt["sampled_round_indices"]),
                "returned_timed_measurements": len(receipt["measurements"]),
                "unknown_timed_measurements": int(active is not None and active["kind"] == "timed_route"),
                "returned_warmup_measurements": len(receipt["warmups"]),
                "unknown_warmup_measurements": int(active is not None and active["kind"] == "warmup"),
                "interrupted_operation": active,
            }
            receipt["accounting"]["unattempted_timed_measurements"] = (
                receipt["accounting"]["planned_timed_measurements"]
                - receipt["accounting"]["returned_timed_measurements"]
                - receipt["accounting"]["unknown_timed_measurements"])
            if output_path is not None:
                write_json(output_path, receipt)
    return receipt
