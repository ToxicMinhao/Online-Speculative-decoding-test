"""Native cold-prefill + exactly one draft block, stopping before verification.

Adapted from the historical OSD attribution probe. No correction projections,
feature exports, training forward, response generation or response-level KL.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from experiments.dflash_matched_prefix_diagnostic import native

base = native.base
require, digest, write_json = native.require, native.digest, native.write_json
BLOCK_SIZE, PREFILL_CHUNK, MAX_ANCHORS, MAX_BLOCKS = 10, 8192, 240, 720


def preserve_invalid_numbers(value):
    """Keep nonfinite failure evidence JSON-safe without replacing it with zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float':repr(value)}
    if isinstance(value, dict):
        return {key:preserve_invalid_numbers(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [preserve_invalid_numbers(item) for item in value]
    return value


def validate_anchor(anchor):
    ids, output = anchor['input_ids'], anchor['output_ids']
    require(isinstance(ids, list) and ids and isinstance(output, list) and output and
            all(type(t) is int and 0 <= t < 128256 for t in ids + output), 'Exact canonical token IDs required')
    start = anchor['block_output_start']
    require(type(start) is int and 1 <= start < len(output), 'Block starts after its known seed')
    prefix = ids + output[:start]
    require(anchor['canonical_prefix_ids'] == prefix and anchor['seed_token_id'] == output[start-1],
            'Canonical prefix/seed geometry changed')
    require(anchor['canonical_next_ids'] == output[start:start+9], 'Canonical next positions changed')
    require(anchor['expected_absolute_positions'] == list(range(len(ids)+start, len(ids)+start+9)),
            'Absolute proposal positions changed')
    expected = anchor['expected_frozen_proposal_ids']
    require(isinstance(expected, list) and len(expected) == 9 and
            all(type(t) is int and 0 <= t < 128256 for t in expected), 'All nine retained F proposals required')
    require(anchor['population'] in ('source', 'heldout') and
            all(isinstance(anchor[k], str) and anchor[k] for k in ('anchor_id','request_id','group_id','domain')),
            'Original request/group/population identities required')
    require(anchor['eos_ids'] == list(native.kl.STOP_IDS), 'Original target EOS definitions changed')
    require(len(ids) <= base.INPUT_LIMIT and len(output) <= 1024 and
            0 < len(prefix)-1 and len(prefix)-1+BLOCK_SIZE <= base.CONTEXT_LIMIT,
            'Canonical context plus complete proposal block exceeds the qualified limit')
    return prefix[:-1]


def validate_plan(config):
    plan = config.get('matched_prefix_probe_plan')
    require(isinstance(plan, list) and len(plan) <= MAX_ANCHORS, 'One bounded predeclared probe plan required')
    seen, starts, per_request = {}, set(), {}
    for anchor in plan:
        validate_anchor(anchor)
        key = anchor['anchor_id']
        require(key not in seen, 'Duplicate anchor identity')
        require(anchor['group_id'] == config['episode_id'] and
                anchor['checkpoints'] == config['matched_prefix_checkpoints'], 'Anchor group/checkpoint binding changed')
        request = (anchor['population'], anchor['request_id'])
        position = (*request, anchor['block_output_start'])
        require(position not in starts, 'Duplicate canonical block start')
        starts.add(position)
        per_request[request] = per_request.get(request, 0) + 1
        require(per_request[request] <= 8, 'At most eight locations per request')
        seen[key] = anchor
    require(sum(k[0] == 'source' for k in per_request) <= 1 and
            sum(k[0] == 'heldout' for k in per_request) <= 4, 'One source/four held-out requests per group')
    require(set(config['matched_prefix_checkpoints']) == set(native.ARMS), 'Only exact F/S8/L8 arms')
    require(config['released'] == config['matched_prefix_checkpoints']['F'], 'Released checkpoint changed')
    return seen


def state_signature(model):
    """Mutation guard over identities and PyTorch version counters, not value hashes.

    Checkpoint file hashes bind values at load. These signatures then detect
    parameter/buffer replacement, ordinary in-place mutation and mode changes;
    they are not a bytewise proof against unsafe writes bypassing PyTorch.
    """
    rows = []
    for kind, iterator in (('parameter',model.named_parameters()), ('buffer',model.named_buffers())):
        for name, tensor in iterator:
            rows.append([kind, name, id(tensor), tensor.data_ptr(), tensor._version,
                         str(tensor.dtype), list(tensor.shape), tensor.requires_grad])
    return digest(dict(tensors=rows, modes=[(name,m.training) for name,m in model.named_modules()]))


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _memory(torch):
    if not torch.cuda.is_available():
        return dict(peak_allocated_gpu_bytes=0, peak_reserved_gpu_bytes=0)
    return dict(peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())


class ProbeComplete(BaseException):
    def __init__(self, result):
        self.result = result


def inspect_head(hidden_states, lm_head, proposals, anchor, arm, out=None):
    """Repeat the native BF16 vocabulary matmul; inspect its logits in FP32."""
    import torch
    require(hidden_states.shape[0] == 9 and hidden_states.dtype == torch.bfloat16,
            'Exactly nine native BF16 post-normalization features required')
    shard = lm_head.shard_indices
    require(int(shard.num_added_elements) == 0 and int(shard.org_vocab_start_index) == 0,
            'Only the original TP1 base-vocabulary head is qualified')
    E, vocab = lm_head.weight, int(shard.num_org_elements)
    require(E.dtype == torch.bfloat16 and E.shape[0] >= vocab and vocab > 1,
            'Unchanged native BF16 vocabulary head required')
    logits = hidden_states @ E[:vocab].T
    proposal_ids = proposals.detach().reshape(-1).cpu().tolist()
    argmax_ids = logits.argmax(-1).detach().cpu().tolist()
    finite = torch.isfinite(logits)
    value = dict(proposal_ids=proposal_ids, proposed_token_ids=proposal_ids, native_argmax_ids=argmax_ids,
        positions=[], native_logits_finite=bool(finite.all()),
        nonfinite_logit_count=int((~finite).sum()), native_feature_dtype=str(hidden_states.dtype),
        native_logits_dtype=str(logits.dtype), inspection_reduction_dtype='torch.float32',
        native_precision='Unchanged BF16 features/head/matmul; logsumexp, subtraction and reductions in FP32')
    if not value['native_logits_finite']:
        value['nonfinite_logit_examples'] = [dict(block_position=int(i),vocabulary_token_id=int(j),
            value=repr(float(logits[i,j]))) for i,j in (~finite).nonzero()[:32].detach().cpu().tolist()]
    try:
        require(value['native_logits_finite'], 'Nonfinite native BF16 logits')
        require(len(proposal_ids) == 9 and proposal_ids == argmax_ids, 'Inspected argmax differs from native sampler')
        fp32 = logits.float()
        log_normalizer = torch.logsumexp(fp32, dim=-1)
        for index, wanted in enumerate(anchor['canonical_next_ids']):
            require(0 <= wanted < vocab, 'Canonical token outside fixed vocabulary')
            row = fp32[index]
            canonical_logit = row[wanted]
            competitors = row.clone()
            competitors[wanted] = -torch.inf
            best_other = int(competitors.argmax())
            maximum = row.max()
            value['positions'].append(dict(block_position=index, native_block_slot=index+1,
                absolute_output_position=anchor['block_output_start']+index,
                absolute_position=anchor['expected_absolute_positions'][index], canonical_token_id=wanted,
                proposal_token_id=proposal_ids[index], native_argmax_id=argmax_ids[index],
                correct=proposal_ids[index] == wanted, native_best_other_id=best_other,
                native_canonical_logit=float(canonical_logit), native_best_other_logit=float(row[best_other]),
                native_canonical_log_probability=float(canonical_logit-log_normalizer[index]),
                native_bf16_canonical_margin=float(canonical_logit-row[best_other]),
                native_argmax_tie_count=int((row == maximum).sum()),
                canonical_tied_for_max=bool(canonical_logit == maximum and int((row == maximum).sum()) > 1)))
        require(all(torch.isfinite(torch.tensor([p['native_canonical_log_probability'],
            p['native_bf16_canonical_margin']])).all() for p in value['positions']), 'Nonfinite FP32 inspection')
        return value
    except BaseException as error:
        error.diagnostic_partial = value
        raise


def run_native_block(scheduler, anchor, arm, out):
    import torch
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    prefix_ids = validate_anchor(anchor)
    require(arm in native.ARMS, 'Undeclared arm')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out / (digest([anchor['anchor_id'], arm])+'.json')
    require(not path.exists(), 'Never overwrite a native block result')
    value = {k:anchor[k] for k in ('anchor_id','group_id','domain','population','request_id','seed_token_id')}
    identity = anchor['checkpoints'][arm]['checkpoint_identity_sha256']
    value.update(status='running', arm=arm, checkpoint=anchor['checkpoints'][arm],
        checkpoint_identity_sha256=identity, canonical_prefix_ids=list(anchor['canonical_prefix_ids']),
        canonical_prefix_sha256=digest(anchor['canonical_prefix_ids']), prefix_input_ids_sha256=digest(prefix_ids),
        cache_checkpoint_identity_sha256=identity, helper_checkpoint_identity_sha256=identity,
        target_prefill_chunks=[], target_prefill_chunk_attempts=[], positions=[], target_verifications=0,
        regenerated_response_tokens=0, training_steps=0, optimizer_creations=0, optimizer_loads=0,
        native_draft_block_evaluations=0, draft_block_evaluations=0, native_head_calls=0,
        attempted_probes=1, target_seed_matches=False, valid_reconstruction=False,
        state_hash_kind='parameter_and_buffer_identity_version_signature',
        no_generated_history=True, no_training=True, cache_valid=False, helper_refresh_valid=True,
        cache_policy='Cold target/draft caches before and after every checkpoint/location',
        costs=dict(target_prefill_s=0., native_draft_forward_s=0., native_head_s=0., head_inspection_s=0.,
                   cache_maintenance_s=0.), timing_scope='Diagnostic only; no saved-call or speedup interpretation')
    started = time.perf_counter()
    worker = scheduler.draft_worker
    target = worker.target_worker.model_runner.model
    draft = worker.draft_model
    originals = {}
    failure = None
    phase = 'idle'
    before = {}
    try:
        entry = base._serving_entry(base._worker_config())
        entry.inherited.strict_idle(scheduler)
        require(worker.block_size == 10 and worker.tp_size == 1 and not worker.dynamic_verify_len and
                not worker.use_thresh_head_two_model and not worker.use_compact_draft_cache,
                'Preserve the original fixed-block single-rank full-context DFlash path')
        for name, model in (('target',target), ('draft',draft)):
            model.eval().requires_grad_(False)
            before[name] = state_signature(model)
            value[name+'_state_before_sha256'] = before[name]
        value['eval_mode'] = all(not m.training for model in (target,draft) for m in model.modules())
        require(value['eval_mode'], 'Target and draft must stay in frozen eval mode')
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cache_started = time.perf_counter()
        try:
            value['cache_before_cold'] = bool(scheduler.flush_cache())
            require(value['cache_before_cold'], 'Cold cache flush failed before diagnostic')
        finally:
            value['costs']['cache_maintenance_s'] += time.perf_counter()-cache_started
        originals['head'] = worker._greedy_sample_from_vocab_parallel_head
        originals['draft'] = worker.draft_model_runner.forward
        originals['target'] = worker.target_worker.forward_batch_generation

        def counted_forward(*args, **kwargs):
            require(value['native_draft_block_evaluations'] == 0 and phase == 'block',
                    'More than one native draft block attempted or draft called during prefill')
            value['native_draft_block_evaluations'] += 1
            value['draft_block_evaluations'] += 1
            tick = time.perf_counter()
            try:
                return originals['draft'](*args, **kwargs)
            finally:
                _sync(torch)
                value['costs']['native_draft_forward_s'] += time.perf_counter()-tick

        def guarded_target(*args, **kwargs):
            require(phase == 'prefill', 'Target verification/output generation is forbidden')
            return originals['target'](*args, **kwargs)

        def capture_head(_self, *, hidden_states, lm_head, chunk_size=256):
            value['native_head_calls'] += 1
            require(value['native_head_calls'] == 1 and phase == 'block', 'More than one native head attempted')
            positions = worker._draft_block_positions_buf[0, 1:10].detach().cpu().tolist()
            value['native_absolute_proposal_positions'] = positions
            require(positions == anchor['expected_absolute_positions'], 'Native absolute block positions changed')
            block_ids = worker._draft_block_ids_buf[0].detach().cpu().tolist()
            value['native_materialized_block_ids'] = block_ids
            require(block_ids == [anchor['seed_token_id']] + [128002]*9, 'Native seed/mask materialization changed')
            tick = time.perf_counter()
            proposals = originals['head'](hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size)
            _sync(torch)
            value['costs']['native_head_s'] += time.perf_counter()-tick
            # Preserve the native IDs before any inspection validation can fail.
            value.update(proposal_ids=proposals.detach().cpu().tolist(), proposed_token_ids=proposals.detach().cpu().tolist())
            tick = time.perf_counter()
            try:
                inspected = inspect_head(hidden_states, lm_head, proposals, anchor, arm, out)
            except BaseException as error:
                value.update(getattr(error, 'diagnostic_partial', {}))
                raise
            finally:
                _sync(torch)
                value['costs']['head_inspection_s'] += time.perf_counter()-tick
            raise ProbeComplete(inspected)

        worker._greedy_sample_from_vocab_parallel_head = types.MethodType(capture_head, worker)
        worker.draft_model_runner.forward = counted_forward
        worker.target_worker.forward_batch_generation = guarded_target
        sampling = SamplingParams(temperature=0, max_new_tokens=1024, ignore_eos=False)
        sampling.normalize(scheduler.tokenizer)
        req = Req(rid='matched-prefix-'+digest([anchor['anchor_id'],arm]), origin_input_text='',
            origin_input_ids=prefix_ids, sampling_params=sampling, eos_token_ids=set(anchor['eos_ids']),
            vocab_size=scheduler.model_config.vocab_size)
        req.tokenizer = scheduler.tokenizer
        batch = None
        with torch.inference_mode():
            value['inference_mode'] = torch.is_inference_mode_enabled()
            phase = 'prefill'
            for start in range(0, len(prefix_ids), PREFILL_CHUNK):
                stop = min(start+PREFILL_CHUNK, len(prefix_ids))
                req.fill_ids = prefix_ids[:stop]
                if start:
                    # Pool stores int32; the native prefix-copy kernel needs an
                    # owned int64 tensor. Conversion must preserve device/values.
                    req.prefix_indices = scheduler.req_to_token_pool.req_to_token[
                        req.req_pool_idx, :start].to(dtype=torch.int64, copy=True)
                req.set_extend_input_len(stop-start)
                req.is_chunked = int(stop < len(prefix_ids))
                batch = ScheduleBatch.init_new([req], scheduler.req_to_token_pool,
                    scheduler.token_to_kv_pool_allocator, scheduler.tree_cache, scheduler.model_config,
                    scheduler.enable_overlap, scheduler.spec_algorithm,
                    chunked_req=req if stop < len(prefix_ids) else None)
                batch.prepare_for_extend()
                value['target_prefill_chunk_attempts'].append(stop-start)
                tick = time.perf_counter()
                try:
                    worker.forward_batch_generation(batch)
                finally:
                    _sync(torch)
                    value['costs']['target_prefill_s'] += time.perf_counter()-tick
                value['target_prefill_chunks'].append(stop-start)
            require(batch is not None and int(batch.seq_lens[0]) == len(prefix_ids), 'Native prefix length changed')
            reconstructed = int(batch.spec_info.verified_id[0])
            value.update(reconstructed_target_seed_id=reconstructed,
                         target_seed_matches=reconstructed == anchor['seed_token_id'])
            # This gate MUST precede filling a saved seed and materializing any
            # draft block. Failed seed reconstruction consumes zero draft calls.
            require(value['target_seed_matches'], 'Reconstructed target seed differs from saved canonical seed')
            batch.spec_info.verified_id.fill_(anchor['seed_token_id'])
            phase = 'block'
            batch.forward_mode = ForwardMode.DECODE
            worker._prepare_for_speculative_decoding(batch, batch.spec_info)
        raise ValueError('Native diagnostic returned without stopping at one draft head')
    except ProbeComplete as complete:
        value.update(complete.result)
        try:
            require(value['native_draft_block_evaluations'] == value['native_head_calls'] == 1,
                    'Exactly one native block/head required')
            matched = value['proposal_ids'] == anchor['expected_frozen_proposal_ids'] if arm == 'F' else None
            value['frozen_proposals_match_retained'] = matched
            require(matched is not False, 'Frozen reconstruction differs from all nine retained proposals')
            value.update(valid_reconstruction=True, native_hook_passed=True)
        except BaseException as error:
            failure = error
    except BaseException as error:
        failure = error
    finally:
        if 'head' in originals:
            worker._greedy_sample_from_vocab_parallel_head = originals['head']
            worker.draft_model_runner.forward = originals['draft']
            worker.target_worker.forward_batch_generation = originals['target']
        cleanup_errors = []
        for name, model in (('target',target), ('draft',draft)):
            try:
                after = state_signature(model)
                value[name+'_state_after_sha256'] = after
                value[name+'_weights_unchanged'] = name in before and after == before[name]
                require(value[name+'_weights_unchanged'], 'Diagnostic changed '+name+' parameters/buffers/mode')
            except BaseException as error:
                cleanup_errors.append(str(error))
        value['immutable_weights'] = value.get('target_weights_unchanged',False) and value.get('draft_weights_unchanged',False)
        cache_started = time.perf_counter()
        try:
            require(scheduler.flush_cache(), 'Cold cache flush failed after diagnostic')
            value['cache_after_cold'] = True
            value['cache_valid'] = value.get('cache_before_cold',False)
        except BaseException as error:
            cleanup_errors.append(str(error))
        value['costs']['cache_maintenance_s'] += time.perf_counter()-cache_started
        if cleanup_errors:
            value['cleanup_errors'] = cleanup_errors
            failure = failure or ValueError('; '.join(cleanup_errors))
        value['costs'].update(block_wall_s=time.perf_counter()-started, **_memory(torch))
        value['status'] = 'failed' if failure else 'passed'
        if failure:
            value.update(error_type=type(failure).__name__, error=str(failure))
        value = preserve_invalid_numbers(value)
        write_json(path, value)
    if failure:
        failure.diagnostic_partial = value
        raise failure
    return value


def _scheduler_rpc(self, anchor_id, arm):
    config = base._worker_config()
    plan = validate_plan(config)
    require(anchor_id in plan and arm in native.ARMS, 'Undeclared scheduler anchor/arm')
    anchor = plan[anchor_id]
    require(self._d181_current_draft == anchor['checkpoints'][arm]['path'], 'Active checkpoint differs from plan')
    run_native_block(self, anchor, arm, Path(config['worker_out']) / 'blocks')


def _probe_scheduler(*args, **kwargs):
    entry = base._serving_entry(base._worker_config())
    from sglang.srt.managers.scheduler import Scheduler
    Scheduler.matched_prefix_probe = _scheduler_rpc
    return entry.serving_scheduler(*args, **kwargs)


class ProbeService(base._ServingService):
    def __init__(self, config):
        self.plan = validate_plan(config)
        self.used, self.refreshed = set(), set()
        self.claims = Path(config['worker_out']) / 'claims'
        self.claims.mkdir(exist_ok=False)
        base._sd_scheduler = _probe_scheduler
        super().__init__(config)

    def _publish_exact(self, checkpoint):
        identity = checkpoint['checkpoint_identity_sha256']
        if self.active == checkpoint and identity in self.refreshed:
            return dict(status='passed', activation_s=0., already_active=True, helper_refresh_valid=True)
        started = time.perf_counter()
        native.admit_checkpoint(self.config, checkpoint)
        self.bridge._two_update_admitted.add(Path(checkpoint['path']))
        receipt = self.bridge.publish({'model_path':checkpoint['path'], 'reset':True})
        require(receipt['status'] == 'passed' and receipt['success'] and
                receipt['helper_matches_current_layers'] and receipt['helper_norm_matches_current_layers'] and
                receipt['cache_after']['tree_total_size'] == 0 and
                self.bridge.state()['draft_model_path'] == checkpoint['path'], 'Checkpoint/helper/cache refresh failed')
        self.active = checkpoint
        self.refreshed.add(identity)
        write_json(self.out / 'ACTIVE_CHECKPOINT.json', checkpoint)
        return dict(status='passed', activation_s=time.perf_counter()-started,
                    native_receipt=receipt, helper_refresh_valid=True)

    def probe(self, body):
        anchor_id, arm = body['anchor_id'], body['arm']
        require(anchor_id in self.plan and arm in native.ARMS, 'Only planned anchor/arm pairs')
        anchor = self.plan[anchor_id]
        require(body['checkpoint'] == anchor['checkpoints'][arm], 'Checkpoint identity changed')
        key = (anchor_id, arm)
        require(key not in self.used and len(self.used) < MAX_BLOCKS, 'No repeated or extra block evaluations')
        claim = self.claims / (digest(list(key))+'.json')
        require(not claim.exists(), 'Claimed native block may not be retried')
        write_json(claim, dict(anchor_id=anchor_id, arm=arm, checkpoint=body['checkpoint'],
            claimed_before_model_work=True, training=False))
        self.used.add(key)
        path = self.out / 'blocks' / (digest(list(key))+'.json')
        publication = None
        started = time.perf_counter()
        publication_elapsed = 0.
        rpc_started = False
        try:
            try:
                publication = self._publish_exact(body['checkpoint'])
            finally:
                publication_elapsed = time.perf_counter()-started
            rpc_started = True
            self.engine.collective_rpc('matched_prefix_probe', anchor_id=anchor_id, arm=arm)
            require(path.exists(), 'Scheduler omitted durable raw block evidence')
            value = json.loads(path.read_text())
            require(value['status'] == 'passed', 'Native scheduler retained a failing block')
        except BaseException as error:
            if path.exists():
                value = json.loads(path.read_text())
            else:
                path.parent.mkdir(exist_ok=True)
                value = dict(status='failed', anchor_id=anchor_id, arm=arm, checkpoint=body['checkpoint'],
                    attempted_probes=1, native_draft_block_evaluations=None if rpc_started else 0,
                    native_head_calls=None if rpc_started else 0, counters_known=not rpc_started,
                    target_prefill_chunks=[], error=str(error),
                    failure_stage='scheduler_transport' if rpc_started else 'activation')
            value.update(status='failed', publication=publication, service_error=str(error))
            value.setdefault('costs', {}).update(activation_s=publication_elapsed,
                service_wall_s=time.perf_counter()-started)
            if rpc_started and not value.get('counters_known', True):
                value['costs']['unresolved_scheduler_transport_s'] = time.perf_counter()-started-publication_elapsed
            write_json(path, value)
            error.diagnostic_partial = value
            raise
        value['publication'] = publication
        value.setdefault('costs', {})['activation_s'] = publication['activation_s']
        write_json(path, value)
        return value

    def generate(self, _):
        raise ValueError('Response generation is forbidden in the matched-prefix diagnostic')


def worker_main():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    config = base._worker_config()
    native.check_sources(config)
    base.check_dependencies(config['workspace_root'])
    allocation = os.environ.pop('CUDA_VISIBLE_DEVICES')
    try:
        base.check_environment(config)
    finally:
        os.environ['CUDA_VISIBLE_DEVICES'] = allocation
    require(config['kind'] == 'sd', 'This diagnostic has no training worker')
    service = ProbeService(config)
    poisoned = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        def log_message(self, *_):
            pass
        def do_POST(self):
            try:
                require(self.headers.get('X-Two-Update-Token') == config['rpc_token'], 'Foreign client')
                length = int(self.headers.get('Content-Length', '0'))
                require(0 < length <= 64*2**20 and not poisoned, 'Invalid payload or poisoned worker')
                body, action = json.loads(self.rfile.read(length)), self.path.lstrip('/')
                require(action in ('ready','probe'), 'Operation outside read-only one-block diagnostic')
                result = dict(status='passed',pid=os.getpid()) if action == 'ready' else service.probe(body)
                status = 200
            except Exception as error:
                poisoned.append(str(error))
                result = dict(status='failed',error=str(error),traceback=traceback.format_exc())
                if hasattr(error,'diagnostic_partial'):
                    result['diagnostic_partial'] = error.diagnostic_partial
                write_json(Path(config['worker_out']) / ('FAILURE_%d.json'%len(poisoned)), result)
                status = 500
            payload = json.dumps(result,allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
    server = HTTPServer(('127.0.0.1',config['port']),Handler)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close(); service.close()


if __name__ == '__main__':
    require(sys.argv[1:] == ['--worker'], 'Use the matched-prefix diagnostic controller')
    worker_main()
