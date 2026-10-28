"""Request-local observation and cold publication for the pinned SGLang runtime.

The native proposal, attention and verification implementations remain unchanged.
Only committed input rows are retained. The last emitted token is explicitly
absent from serving capture; an idle full-context target forward supplies its
real feature separately. This module does not import SGLang or CUDA on import.
"""
from __future__ import annotations

import functools
from pathlib import Path
import time

from .storage import save_torch, write_json


_ACTIVE = None
_POISONED = None
_INSTALLED = False


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _healthy():
    require(_POISONED is None, 'SGLang worker is blocked: ' + str(_POISONED))


def _poison(error):
    global _POISONED
    _POISONED = repr(error)


def _ids(value):
    if hasattr(value, 'detach'):
        value = value.detach().reshape(-1).cpu().tolist()
    return [int(item) for item in value]


def _token_ids(value):
    require(isinstance(value, list) and value and
            all(type(item) is int and item >= 0 for item in value),
            'Require a nonempty exact token-ID list')
    return list(value)


def _feature_pair(output, rows):
    import torch

    aux = output.hidden_states
    final = getattr(output, 'teacher_hidden_states', None)
    require(aux is not None and final is not None,
            'Pinned runtime must expose auxiliary and normalized teacher states')
    require(aux.ndim == final.ndim == 2 and aux.shape[0] == final.shape[0] == rows,
            'Target feature rows do not match the actual forward inputs')
    require(aux.dtype == final.dtype == torch.bfloat16 and
            aux.shape[1] >= final.shape[1], 'Target feature dtype/width contract changed')
    return aux, final


class RequestCapture:
    """Reconcile contiguous prefill and native verification with exact outputs."""

    def __init__(self, prompt_ids, outputfile, collect_features):
        require(type(collect_features) is bool, 'collect_features must be boolean')
        self.prompt_ids = _token_ids(prompt_ids)
        self.outputfile = Path(outputfile)
        require(self.outputfile.is_absolute(), 'Capture output path must be absolute')
        require(not self.outputfile.exists() and not self.outputfile.with_suffix('.pt').exists(),
                'Capture artifacts are immutable')
        self.collect_features = collect_features
        self.covered_ids = []
        self.aux, self.final, self.trace, self.teacher_ids = [], [], [], []
        self.prefill_rows = 0
        self.prefill_chunks = 0
        self.block_size = None
        self.feature_bytes = 0
        self.started = time.perf_counter()
        self.time_to_first_token_s = None
        self.capture_host_s = 0.0
        self.cache_maintenance_s = 0.0

    def _append(self, start, tokens, aux=None, final=None):
        require(start == len(self.covered_ids), 'Feature positions have a gap or overlap')
        if self.collect_features:
            import torch

            require(aux is not None and final is not None and
                    aux.shape[0] == final.shape[0] == len(tokens), 'Missing committed feature rows')
            # CPU copies own their storage before the next CUDA graph replay.
            a = aux.detach().to('cpu', copy=True).contiguous()
            h = final.detach().to('cpu', copy=True).contiguous()
            require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(h).all()),
                    'Nonfinite target features')
            self.aux.append(a)
            self.final.append(h)
            self.feature_bytes += a.numel() * a.element_size() + h.numel() * h.element_size()
        self.covered_ids.extend(tokens)

    def prefill(self, start, tokens, first_token, *, aux=None, final=None, block_size):
        require(not self.trace and not self.teacher_ids, 'Prefill arrived after generation began')
        require(start == self.prefill_rows and tokens and
                self.prompt_ids[start:start + len(tokens)] == tokens and
                start + len(tokens) <= len(self.prompt_ids), 'Prefill does not cover the canonical prompt')
        if self.block_size is None:
            self.block_size = int(block_size)
        require(self.block_size == block_size, 'Native block size changed within a request')
        self._append(start, tokens, aux, final)
        self.prefill_rows += len(tokens)
        self.prefill_chunks += 1
        if self.prefill_rows == len(self.prompt_ids):
            require(type(first_token) is int and first_token >= 0, 'Final prefill lacks first output token')
            self.teacher_ids.append(first_token)
            self.time_to_first_token_s = time.perf_counter() - self.started
        else:
            require(first_token is None, 'Partial prefill cannot commit an output token')

    def verification(self, *, before, after, candidates, positions, posterior,
                     commit_rows, native_accepted, finished, aux=None, final=None):
        require(self.prefill_rows == len(self.prompt_ids) and before == self.teacher_ids,
                'Verification does not extend the captured response')
        width = self.block_size
        anchor = len(self.prompt_ids) + len(before) - 1
        require(len(candidates) == len(posterior) == width and
                positions == list(range(anchor, anchor + width)) and
                candidates[0] == before[-1], 'Native verification input geometry changed')
        require(type(commit_rows) is int and 1 <= commit_rows <= width and
                after[:len(before)] == before and len(after) == len(before) + commit_rows,
                'Native committed count differs from actual outputs')
        accepted_end = next((slot for slot in range(1, width)
                             if candidates[slot] != posterior[slot - 1]), width)
        expected = candidates[1:accepted_end] + [posterior[accepted_end - 1]]
        committed = after[len(before):]
        require(commit_rows <= accepted_end and committed == expected[:commit_rows] and
                committed == posterior[:commit_rows], 'Native posterior does not explain committed outputs')
        require(native_accepted == max(commit_rows - 1, 0), 'Native stop-clipped acceptance count changed')
        canonical = self.prompt_ids + after
        require(candidates[:commit_rows] == canonical[anchor:anchor + commit_rows],
                'Attempted to retain rejected-history features')
        self._append(anchor, candidates[:commit_rows],
                     None if aux is None else aux[:commit_rows],
                     None if final is None else final[:commit_rows])
        self.trace.append({
            'output_start': len(before) - 1,
            'anchor_token': candidates[0],
            'draft_tokens': candidates[1:],
            'target_argmax': posterior,
            'verify_size': width,
            'accepted_draft_tokens': min(accepted_end - 1, commit_rows),
            'native_accepted_draft_tokens': native_accepted,
            'raw_accepted_draft_tokens_before_stop': accepted_end - 1,
            'committed_progress': commit_rows,
            'committed_tokens': committed,
            'positions': positions,
            'stop_clipped': commit_rows < accepted_end,
            'finished': bool(finished),
        })
        self.teacher_ids.extend(posterior[:commit_rows])

    def finish(self, output_ids):
        started = time.perf_counter()
        output_ids = _token_ids(output_ids)
        require(self.prefill_rows == len(self.prompt_ids) and self.teacher_ids == output_ids,
                'Returned outputs differ from native first-token/verification observations')
        expected = self.prompt_ids + output_ids[:-1]
        require(self.covered_ids == expected, 'Canonical captured feature coverage is incomplete')
        document = {
            'status': 'passed', 'prompt_ids': self.prompt_ids, 'response_ids': output_ids,
            'target_argmax_ids': self.teacher_ids, 'blocks': self.trace,
            'covered_token_ids': self.covered_ids, 'covered_rows': len(self.covered_ids),
            'prefill_chunks': self.prefill_chunks, 'block_size': self.block_size,
            'native_capture_validated': True,
            'label_binding_scope': 'native_prefill_and_committed_verification',
            'final_feature_logits_used_for_labels': False,
            'final_emitted_token_feature_present': False,
            'collect_features': self.collect_features,
            'time_to_first_token_s': self.time_to_first_token_s,
            'feature_bytes': self.feature_bytes,
            'cache_maintenance_s': self.cache_maintenance_s,
            'timing_scope': 'Host wall time including transfers/synchronization; observer work excludes original native calls',
        }
        persistence_s = 0.0
        if self.collect_features:
            import torch

            path = self.outputfile.with_suffix('.pt')
            payload = {
                'input_ids': list(self.covered_ids),
                'hidden_states': torch.cat(self.aux, dim=0),
                'target_last_hidden_states': torch.cat(self.final, dim=0),
                **document,
            }
            self.capture_host_s += time.perf_counter() - started
            save_started = time.perf_counter()
            save_torch(path, payload)
            persistence_s = time.perf_counter() - save_started
            document['feature_path'] = str(path)
        else:
            self.capture_host_s += time.perf_counter() - started
        document.update(capture_host_s=self.capture_host_s, feature_persistence_s=persistence_s)
        write_json(self.outputfile, document)
        return document


def strict_idle(scheduler):
    require(scheduler.enable_overlap is False, 'Native hooks require nonoverlapping scheduling')
    require(scheduler._is_no_request() and not scheduler.waiting_queue and
            not scheduler.running_batch.reqs and scheduler.chunked_req is None,
            'Scheduler has active, queued or chunked requests')
    queue = getattr(scheduler.grammar_manager, 'grammar_queue', None)
    require(queue is not None and not queue, 'Grammar queue is absent or nonempty')


def flush_idle(scheduler):
    started = time.perf_counter()
    strict_idle(scheduler)
    require(scheduler.flush_cache() is True, 'Native cache flush failed')
    require(scheduler.tree_cache.total_size() == 0, 'Native cache remains populated')
    return time.perf_counter() - started


def _rpc(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        _healthy()
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            _poison(error)
            raise
    return guarded


@_rpc
def online_begin(self, outputfile, collect_features, prompt_ids):
    global _ACTIVE
    require(_ACTIVE is None, 'A prior request capture is unfinished')
    maintenance_s = flush_idle(self)
    _ACTIVE = RequestCapture(prompt_ids, outputfile, collect_features)
    _ACTIVE.cache_maintenance_s = maintenance_s


@_rpc
def online_finish(self, output_ids):
    global _ACTIVE
    strict_idle(self)
    require(_ACTIVE is not None, 'No request is armed for completion')
    result = _ACTIVE.finish(output_ids)
    _ACTIVE = None
    return result


def check_fused_helper(worker):
    import torch

    helper = worker._fused_kv_helper
    require(worker._use_fused_kv_materialize and helper is not None,
            'Qualified fused KV helper is unavailable')
    layers = worker.draft_model.layers
    require(len(helper.k_norm_weights) == len(layers), 'Fused helper layer count differs')
    expected = torch.stack([
        layer.self_attn.qkv_proj.weight[
            layer.self_attn.q_size:layer.self_attn.q_size + 2 * layer.self_attn.kv_size]
        for layer in layers])
    weights = bool(torch.equal(helper.batched_kv_weight, expected))
    norms = all(bool(torch.equal(weight, layer.self_attn.k_norm.weight))
                for weight, layer in zip(helper.k_norm_weights, layers))
    require(weights and norms, 'Fused KV helper contains stale draft weights')
    return {'helper_matches_current_layers': weights,
            'helper_norm_matches_current_layers': norms, 'fused_enabled': True}


def publish_update(scheduler, request, original):
    """Refresh copied draft weights before native publication becomes visible."""
    from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqOutput
    import torch

    _healthy()
    started = time.perf_counter()
    try:
        require(_ACTIVE is None, 'Cannot publish while a request capture is pending')
        require(request.update_speculative_draft is True and request.flush_cache is True,
                'Only a cold native draft publication is supported')
        maintenance_s = flush_idle(scheduler)
        torch.cuda.synchronize()
        load_started = time.perf_counter()
        result = original(scheduler, request)
        torch.cuda.synchronize()
        native_load_s = time.perf_counter() - load_started
        require(result.success, 'Native draft weight loading failed: ' + str(result.message))
        helper_started = time.perf_counter()
        scheduler.draft_worker._init_fused_kv_helper()
        validation = check_fused_helper(scheduler.draft_worker)
        torch.cuda.synchronize()
        helper_s = time.perf_counter() - helper_started
        require(scheduler.tree_cache.total_size() == 0, 'Publication left cached tokens')
        scheduler._online_publication = {
            'status': 'passed', 'checkpoint_path': str(request.model_path),
            'activation_s': time.perf_counter() - started,
            'native_load_s': native_load_s, 'helper_refresh_s': helper_s,
            'cache_maintenance_s': maintenance_s,
            'timing_scope': 'Host wall time including synchronization; native load includes its requested internal flush',
            'cache_policy': 'full_flush_before_publication_and_every_request',
            'cache_after_tokens': 0, **validation,
        }
        return result
    except BaseException as error:
        _poison(error)
        scheduler._online_publication = {'status': 'failed', 'error': repr(error),
                                        'checkpoint_path': str(request.model_path)}
        # No rollback is claimed. The worker remains poisoned after cleanup.
        try:
            scheduler.flush_cache()
        except Exception:
            pass
        return UpdateWeightFromDiskReqOutput(False, str(error), 0)


def online_status(self, outputfile):
    """Status remains readable after failure; no further model operation is allowed."""
    value = {
        'status': 'passed' if _POISONED is None else 'failed', 'poisoned': _POISONED,
        'capture_pending': _ACTIVE is not None,
        'publication': getattr(self, '_online_publication', None),
        'cache_tokens': self.tree_cache.total_size(),
        'attention_backend': self.server_args.attention_backend,
        'block_size': self.draft_worker.block_size,
    }
    write_json(outputfile, value)
    return value


@_rpc
def online_teacher(self, input_ids, outputfile):
    """Compute the last input token with its full causal prefix, on the same target."""
    import torch
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils import require_mlp_sync

    require(_ACTIVE is None, 'Teacher forward cannot overlap capture')
    input_ids = _token_ids(input_ids)
    destination = Path(outputfile)
    require(destination.is_absolute() and not destination.exists() and
            not destination.with_suffix('.pt').exists(), 'Teacher artifacts must be new absolute paths')
    require(self.server_args.tp_size == self.server_args.pp_size == 1 and
            not require_mlp_sync(self.server_args), 'Teacher RPC requires the qualified single-rank target')
    flush_idle(self)
    runner = self.tp_worker.model_runner
    require(len(input_ids) <= runner.model_config.context_len, 'Teacher input exceeds target context')
    started = time.perf_counter()
    try:
        req = Req(rid='online_final_feature', origin_input_text='', origin_input_ids=input_ids,
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        tree_cache = RadixCache(CacheInitParams(
            disable=False, req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            page_size=runner.server_args.page_size))
        batch = ScheduleBatch.init_new(
            reqs=[req], req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache, model_config=runner.model_config,
            enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
        batch.prepare_for_extend()
        forward = ForwardBatch.init_new(batch.get_model_worker_batch(), runner)
        forward.capture_hidden_mode = CaptureHiddenMode.FULL
        with torch.inference_mode():
            output = runner.forward(forward)
            output = getattr(output, 'logits_output', output)
            aux, final = _feature_pair(output, len(input_ids))
            last_aux = aux[-1:].detach().to('cpu', copy=True)
            last_hidden = final[-1:].detach().to('cpu', copy=True)
            require(bool(torch.isfinite(last_aux).all()) and bool(torch.isfinite(last_hidden).all()),
                    'Nonfinite final-token feature')
        torch.cuda.synchronize()
        document = {
            'status': 'passed', 'input_ids': input_ids,
            'positions': [len(input_ids) - 1], 'target_forward_count': 1,
            'forward_input_rows': len(input_ids), 'retained_feature_rows': 1,
            'final_feature_logits_used_for_labels': False,
            'capture_backend': 'same_frozen_SGLang_full_context_final_row',
            'teacher_preparation_s': time.perf_counter() - started,
            'tensor_path': str(destination.with_suffix('.pt')),
        }
        save_torch(destination.with_suffix('.pt'), {
            **document, 'hidden_states': last_aux, 'target_last_hidden_states': last_hidden})
        write_json(destination, document)
    finally:
        flush_idle(self)
    return document


def checkpoint_tensor_comparison(model, checkpoint):
    """Qualification-only TP1 comparison against every exported draft tensor."""
    import torch
    from safetensors import safe_open

    root = Path(checkpoint)
    paths = sorted(root.glob('*.safetensors'))
    require(paths, 'Checkpoint has no safetensors weights')
    source = {}
    for path in paths:
        with safe_open(path, framework='pt', device='cpu') as tensors:
            for name in tensors.keys():
                require(name not in source, 'Duplicate checkpoint parameter: ' + name)
                source[name] = tensors.get_tensor(name)
    params = dict(model.named_parameters())
    consumed, compared = set(), []

    def source_name(name):
        candidates = [name, name[6:] if name.startswith('model.') else 'model.' + name]
        matches = [candidate for candidate in candidates if candidate in source]
        require(len(matches) == 1, 'Missing or ambiguous checkpoint tensor: ' + name)
        consumed.add(matches[0])
        return source[matches[0]]

    for name, actual in params.items():
        if '.qkv_proj.' in name:
            pieces = [source_name(name.replace('qkv_proj', part))
                      for part in ('q_proj', 'k_proj', 'v_proj')]
            expected = torch.cat(pieces, dim=0)
        elif '.gate_up_proj.' in name:
            expected = torch.cat([source_name(name.replace('gate_up_proj', part))
                                  for part in ('gate_proj', 'up_proj')], dim=0)
        else:
            expected = source_name(name)
        require(actual.shape == expected.shape and actual.dtype == expected.dtype and
                torch.equal(actual.detach().cpu(), expected), 'Published native tensor differs: ' + name)
        compared.append(name)
    require(consumed == set(source), 'Checkpoint contains tensors absent from native draft')
    return {'all_equal': True, 'tensors': len(source), 'source_tensor_count': len(source),
            'native_tensor_count': len(compared), 'native_tensor_names': compared}


@_rpc
def online_check_weights(self, checkpoint, outputfile):
    require(_ACTIVE is None, 'Weight qualification cannot overlap capture')
    strict_idle(self)
    require(self.server_args.tp_size == 1, 'Exact native checkpoint comparison requires TP1')
    result = checkpoint_tensor_comparison(self.draft_worker.draft_model, checkpoint)
    result.update(status='passed', checkpoint_path=str(checkpoint),
                  **check_fused_helper(self.draft_worker))
    write_json(outputfile, result)
    return result


def install():
    """Install only in the isolated scheduler process before engine initialization."""
    global _INSTALLED
    require(not _INSTALLED, 'Native online hooks already installed')
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.speculative.dflash_worker import DFlashWorker
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    original_forward = DFlashWorker.forward_batch_generation
    original_verify = DFlashVerifyInput.verify
    original_update = Scheduler.update_weights_from_disk

    @functools.wraps(original_forward)
    def observed_forward(worker, batch, **kwargs):
        _healthy()
        recorder = _ACTIVE
        if recorder is None:
            return original_forward(worker, batch, **kwargs)
        observed_started = time.perf_counter()
        try:
            require(batch.batch_size() == 1 and len(batch.reqs) == 1,
                    'Online capture requires one request at a time')
            req = batch.reqs[0]
            require(_ids(req.origin_input_ids) == recorder.prompt_ids,
                    'Armed and actual prompt IDs differ')
            require(not worker.use_compact_draft_cache, 'Online capture requires full causal draft context')
            prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
            if not prefill:
                recorder.capture_host_s += time.perf_counter() - observed_started
                return original_forward(worker, batch, **kwargs)
            # Capture lengths before the native call mutates batch state.
            model_batch = batch.get_model_worker_batch()
            starts = _ids(model_batch.extend_prefix_lens)
            lengths = _ids(model_batch.extend_seq_lens)
            require(len(starts) == len(lengths) == 1, 'Unexpected prefill batch geometry')
            start, length = starts[0], lengths[0]
            require(_ids(model_batch.input_ids) == recorder.prompt_ids[start:start + length],
                    'Actual prefill input rows differ from canonical prompt IDs')
            require(not req.output_ids, 'Prefill arrived after output tokens were committed')
            recorder.capture_host_s += time.perf_counter() - observed_started
            result = original_forward(worker, batch, **kwargs)
            observed_started = time.perf_counter()
            aux = final = None
            if recorder.collect_features:
                aux, final = _feature_pair(result.logits_output, length)
            complete = start + length == len(recorder.prompt_ids)
            first = _ids(result.next_token_ids)
            require(len(first) == 1, 'Prefill must expose one sampling result')
            recorder.prefill(start, recorder.prompt_ids[start:start + length],
                             first[0] if complete else None, aux=aux, final=final,
                             block_size=worker.block_size)
            recorder.capture_host_s += time.perf_counter() - observed_started
            return result
        except BaseException as error:
            _poison(error)
            raise

    @functools.wraps(original_verify)
    def observed_verify(info, *, batch, logits_output, page_size):
        _healthy()
        recorder = _ACTIVE
        if recorder is None or batch.forward_mode.is_idle():
            return original_verify(info, batch=batch, logits_output=logits_output, page_size=page_size)
        observed_started = time.perf_counter()
        try:
            require(batch.batch_size() == 1 and batch.sampling_info.is_all_greedy,
                    'Online native capture requires C1 greedy verification')
            require(info.draft_token_num == recorder.block_size and
                    not info._packed and info.candidate_lens is None,
                    'Online native capture requires fixed rectangular verification')
            req = batch.reqs[0]
            before = list(req.output_ids)
            candidates, positions = _ids(info.draft_token), _ids(info.positions)
            posterior = _ids(logits_output.next_token_logits.argmax(dim=-1))
            aux = final = None
            if recorder.collect_features:
                aux, final = _feature_pair(logits_output, info.draft_token_num)
                aux, final = aux.detach().clone(), final.detach().clone()
            recorder.capture_host_s += time.perf_counter() - observed_started
            result = original_verify(info, batch=batch, logits_output=logits_output, page_size=page_size)
            observed_started = time.perf_counter()
            next_ids, commits, next_aux, accepted = result
            count = _ids(commits)[0]
            after = list(req.output_ids)
            require(_ids(next_ids) == after[-1:] and len(accepted) == 1,
                    'Native next-token result differs from the committed answer')
            if recorder.collect_features:
                import torch
                require(torch.equal(aux[:count], next_aux),
                        'Captured auxiliary rows differ from native committed feature rows')
            recorder.verification(before=before, after=after, candidates=candidates,
                                  positions=positions, posterior=posterior, commit_rows=count,
                                  native_accepted=int(accepted[0]), finished=req.finished(),
                                  aux=aux, final=final)
            recorder.capture_host_s += time.perf_counter() - observed_started
            return result
        except BaseException as error:
            _poison(error)
            raise

    DFlashWorker.forward_batch_generation = observed_forward
    DFlashVerifyInput.verify = observed_verify
    Scheduler.update_weights_from_disk = lambda self, request: publish_update(self, request, original_update)
    for function in (online_begin, online_finish, online_teacher, online_status, online_check_weights):
        setattr(Scheduler, function.__name__, function)
    _INSTALLED = True


def scheduler_process(*args, **kwargs):
    install()
    from sglang.srt.managers.scheduler import run_scheduler_process
    return run_scheduler_process(*args, **kwargs)
