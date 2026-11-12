"""Observe native K+5 scores and retain exact parent states across pair sealing.

The head multiplication and greedy choice are the installed runtime's original
operations. A narrow observer retains the multiplication's output; it neither
recomputes logits nor changes their layout. Completed request caches are compacted
into bitwise-checked prefix views, so all pre-K+6 states can stay in one worker.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

from experiments.lost_repair.native_worker import CensusSuccessors
from experiments.longer_run_headroom.native_worker import transition_one_block
from experiments.verification_transition import native_worker as base
from experiments.verification_transition.native import tensor_bits_equal
from online_feedback import sglang_hooks
from online_feedback.storage import save_torch, write_json

_QA_ROWS = {}
_QA_LIMIT = 32


def qa_key(source_id, start_offset, slot):
    data = json.dumps([20261115, str(source_id), int(start_offset), int(slot)],
                      ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def native_margin_rows(scores, chosen, canonical, accepted, *, source_id, start_offset):
    """Reduce native scores, with FP32 subtraction only after scalar selection."""
    if scores.ndim != 2 or scores.shape[0] != 15 or len(chosen) != 15:
        raise ValueError('Require the actual fifteen native proposal score rows')
    if not 0 <= accepted <= min(15, len(canonical)):
        raise ValueError('Accepted slots exceed the canonical suffix')
    if not accepted:
        return [], []
    values = scores[:accepted]
    labels = torch.tensor(canonical[:accepted], device=values.device, dtype=torch.long)
    selected = torch.tensor(chosen[:accepted], device=values.device, dtype=torch.long)
    if not torch.equal(labels, selected):
        raise ValueError('An accepted native proposal is not canonical')
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite native vocabulary scores')
    correct_scores = values.gather(1, labels[:, None]).squeeze(1)
    rivals = values.clone()
    rivals.scatter_(1, labels[:, None], -torch.inf)
    competing_scores, competing_ids = rivals.max(dim=1)
    margins = correct_scores.float() - competing_scores.float()
    if not torch.isfinite(margins).all() or (margins < 0).any():
        raise ValueError('Accepted native token has a negative/nonfinite margin')
    rows, samples = [], []
    for offset, (right, other, competitor, margin) in enumerate(zip(
            correct_scores.tolist(), competing_scores.tolist(), competing_ids.tolist(), margins.tolist())):
        slot = offset + 1
        row = dict(source_id=str(source_id), start_offset=int(start_offset), slot=slot,
                   canonical_id=int(canonical[offset]), chosen_id=int(chosen[offset]),
                   canonical_score=float(right), competitor_score=float(other),
                   competitor_id=int(competitor), margin=float(margin),
                   native_dtype=str(scores.dtype),
                   qa_hash=qa_key(source_id, start_offset, slot))
        rows.append(row)
        # Only candidates for the global fixed bottom-32 hash sample cross to CPU.
        key = row['qa_hash']
        if key in _QA_ROWS:
            raise ValueError('Duplicate accepted row in the pre-K6 margin census')
        if len(_QA_ROWS) < _QA_LIMIT or key < max(_QA_ROWS):
            sample = dict(row, native_scores=values[offset].detach().to('cpu', copy=True))
            _QA_ROWS[key] = sample
            if len(_QA_ROWS) > _QA_LIMIT:
                del _QA_ROWS[max(_QA_ROWS)]
            samples.append(key)
    return rows, samples


def audit_native_margin_samples(samples):
    """Independent full-vocabulary Python reduction of the retained native rows."""
    results = []
    for sample in sorted(samples, key=lambda value: value['qa_hash']):
        scores = sample['native_scores'].tolist()
        canonical = sample['canonical_id']
        rival = max((token for token in range(len(scores)) if token != canonical),
                    key=lambda token: (scores[token], -token))
        chosen = max(range(len(scores)), key=lambda token: (scores[token], -token))
        # BF16 values are exactly representable in FP32; do the requested scalar
        # subtraction independently of the production gathered tensor operation.
        margin = float(torch.tensor(scores[canonical], dtype=torch.float32)
                       - torch.tensor(scores[rival], dtype=torch.float32))
        passed = (chosen == sample['chosen_id'] == canonical
                  and rival == sample['competitor_id']
                  and scores[canonical] == sample['canonical_score']
                  and scores[rival] == sample['competitor_score']
                  and margin == sample['margin'])
        results.append(dict(qa_hash=sample['qa_hash'], source_id=sample['source_id'],
                            start_offset=sample['start_offset'], slot=sample['slot'],
                            passed=passed, recomputed_margin=margin,
                            vocabulary_size=len(scores)))
    return dict(status='passed' if all(row['passed'] for row in results) else 'failed',
                sample_count=len(results), rows=results)


@contextmanager
def observe_original_matmul(observer):
    """Observe, without replacing, torch.matmul inside one native head call."""
    original = torch.matmul

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        observer(result)
        return result

    torch.matmul = observed
    try:
        yield
    finally:
        torch.matmul = original


def compact_prefix_caches(session):
    """Share immutable cache prefixes after exact equality checks.

    Scheduler objects and their mapping/allocator state stay live. Only redundant
    CPU KV tensor storage changes, never tensor values or native reconstruction.
    """
    states = list(session.successors.values())
    if not states:
        return dict(states=0, before_bytes=0, after_bytes=0, prefixes_checked=0)
    longest = max(states, key=lambda state: len(state.locations))
    fields = ('target_cache', 'draft_reference_cache')
    before = sum(t.numel() * t.element_size() for state in states for field in fields
                 for layer in getattr(state, field) for t in layer)
    checks = 0
    for state in states:
        length = len(state.locations)
        for field in fields:
            saved, complete = getattr(state, field), getattr(longest, field)
            if len(saved) != len(complete):
                raise ValueError('Cache compaction layer count differs')
            shared = []
            for original, full in zip(saved, complete):
                pair = []
                for original_tensor, full_tensor in zip(original, full):
                    prefix = full_tensor[:length]
                    if not tensor_bits_equal(original_tensor, prefix):
                        raise ValueError('Earlier native cache rows changed; cannot compact')
                    pair.append(prefix)
                    checks += 1
                shared.append(tuple(pair))
            setattr(state, field, shared)
    after = sum(t.numel() * t.element_size() for field in fields
                for layer in getattr(longest, field) for t in layer)
    return dict(states=len(states), before_bytes=before, after_bytes=after,
                prefixes_checked=checks, cache_prefixes_exact=True)


def _map_tensors(value, path, transform):
    if isinstance(value, torch.Tensor):
        return transform(value, path)
    if isinstance(value, list):
        return [_map_tensors(item, path + (index,), transform)
                for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, path + (index,), transform)
                     for index, item in enumerate(value))
    if isinstance(value, dict):
        return {key: _map_tensors(item, path + (key,), transform)
                for key, item in value.items()}
    # Model runners, pools, runtime services and other opaque objects stay shared.
    return value


def _map_fields(value, path, transform):
    result = copy.copy(value)
    for name, field in vars(value).items():
        setattr(result, name, _map_tensors(field, path + (name,), transform))
    return result


def _map_batch(batch, transform):
    # Mirror precisely the private field ownership used by PR34 clone_batch.
    result = _map_fields(batch, ('batch',), transform)
    result.reqs = []
    for index, request in enumerate(batch.reqs):
        path = ('batch', 'reqs', index)
        cloned = _map_fields(request, path, transform)
        cloned.sampling_params = _map_fields(request.sampling_params,
                                             path + ('sampling_params',), transform)
        result.reqs.append(cloned)
    result.spec_info = _map_fields(batch.spec_info, ('batch', 'spec_info'), transform)
    result.sampling_info = _map_fields(batch.sampling_info, ('batch', 'sampling_info'), transform)
    return result


def offload_successor_metadata(state):
    """Keep allocator and mutable batch tensors on host between the two phases.

    Each tensor keeps its own original device. In particular CPU sequence-length
    fields remain CPU after restoration; runtime services are never serialized.
    """
    if hasattr(state, '_tensor_devices'):
        raise ValueError('A native successor was already offloaded')
    devices = {}
    sizes = {'tensor_bytes': 0, 'device_bytes_released': 0}
    def store(tensor, path):
        devices[path] = str(tensor.device)
        size = tensor.numel() * tensor.element_size()
        sizes['tensor_bytes'] += size
        sizes['device_bytes_released'] += size if tensor.device.type != 'cpu' else 0
        return tensor.detach().to('cpu', copy=True)
    state.batch = _map_batch(state.batch, store)
    for field in ('mapping', 'locations', 'allocator'):
        setattr(state, field, _map_tensors(getattr(state, field), (field,), store))
    state._tensor_devices = devices
    return dict(**sizes, tensors=len(devices))


def materialize_successor_metadata(state):
    """Restore only the chosen state into temporary native-device metadata."""
    if not hasattr(state, '_tensor_devices'):
        return state
    def restore(tensor, path):
        return tensor.to(state._tensor_devices[path], copy=True)
    result = copy.copy(state)
    result.batch = _map_batch(state.batch, restore)
    for field in ('mapping', 'locations', 'allocator'):
        setattr(result, field, _map_tensors(getattr(state, field), (field,), restore))
    return result


class MarginSuccessors(CensusSuccessors):
    """Opening-inclusive PR42 capture, observing only the native draft head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin_rows = []
        self.observed_head_calls = 0
        self._pending_scores = None
        self._pending_chosen = None
        self._observe_head = False

    def install(self):
        super().install()
        worker = self.worker
        ordinary_head = worker._greedy_sample_from_vocab_parallel_head
        self._ordinary_head = ordinary_head
        ordinary_forward = worker.forward_batch_generation

        def head(*, hidden_states, lm_head, chunk_size=256):
            if not self._observe_head:
                return ordinary_head(hidden_states=hidden_states, lm_head=lm_head,
                                     chunk_size=chunk_size)
            shard = lm_head.shard_indices
            if (worker.tp_size != 1 or int(shard.num_added_elements) != 0
                    or int(shard.org_vocab_start_index) != 0
                    or hidden_states.shape[0] != 15 or int(chunk_size) != 256):
                raise ValueError('Native margin observer requires the pinned TP1 15-row head')
            captured = []
            with observe_original_matmul(captured.append):
                chosen = ordinary_head(hidden_states=hidden_states, lm_head=lm_head,
                                       chunk_size=chunk_size)
            if len(captured) != 1 or captured[0].shape != (15, int(shard.num_org_elements)):
                raise ValueError('Native greedy head multiplication layout changed')
            if captured[0].dtype != torch.bfloat16:
                raise ValueError('Native margin scores must retain BF16 precision')
            self._pending_scores = captured[0].detach()
            self._pending_chosen = chosen.detach().reshape(-1).tolist()
            self.observed_head_calls += 1
            return chosen

        def forward(batch, **kwargs):
            prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
            observe = self.capture_enabled and not prefill
            self._observe_head = observe
            try:
                result = ordinary_forward(batch, **kwargs)
                if observe:
                    recorder = sglang_hooks._ACTIVE
                    block = recorder.trace[-1]
                    start = block['output_start']
                    canonical = self.request['canonical_response_ids'][start + 1:]
                    accepted = block['accepted_draft_tokens']
                    if self._pending_scores is None:
                        raise ValueError('Native draft head scores were not observed')
                    rows, _ = native_margin_rows(
                        self._pending_scores, self._pending_chosen, canonical, accepted,
                        source_id=self.request['source_id'], start_offset=start)
                    self.margin_rows.extend(rows)
                return result
            finally:
                self._observe_head = False
                self._pending_scores = self._pending_chosen = None

        worker._greedy_sample_from_vocab_parallel_head = head
        worker.forward_batch_generation = forward

    def rebuild(self, successor, destination_checkpoint):
        return super().rebuild(materialize_successor_metadata(successor), destination_checkpoint)

    def close(self):
        if self.installed:
            self.worker._greedy_sample_from_vocab_parallel_head = self._ordinary_head
        super().close()


def transition_margin_finished(self, outputfile):
    session = base._SESSION
    receipt = base.transition_walk_finished(self, outputfile)
    receipt.update(margins=session.margin_rows, observed_head_calls=session.observed_head_calls)
    write_json(outputfile, receipt)
    return receipt


def transition_compact_request(self, request_id, outputfile):
    sglang_hooks.strict_idle(self)
    sessions = [session for session in base._SESSIONS
                if session.request['request_id'] == request_id]
    if len(sessions) != 1 or base._SESSION is not None:
        raise ValueError('Require one completed native request for compaction')
    receipt = dict(status='passed', request_id=request_id,
                   **compact_prefix_caches(sessions[0]))
    metadata = [offload_successor_metadata(state) for state in sessions[0].successors.values()]
    receipt['metadata_offload'] = {name: sum(row[name] for row in metadata)
                                   for name in ('tensor_bytes', 'device_bytes_released', 'tensors')}
    write_json(outputfile, receipt)
    return receipt


def transition_margin_qa(self, tensor_path, outputfile):
    sglang_hooks.strict_idle(self)
    samples = sorted(_QA_ROWS.values(), key=lambda row: row['qa_hash'])
    save_torch(tensor_path, dict(rows=samples))
    result = audit_native_margin_samples(samples)
    result['tensor_path'] = str(tensor_path)
    write_json(outputfile, result)
    if result['status'] != 'passed':
        raise ValueError('Independent full-vocabulary native margin audit failed')
    return result


def transition_drop_request(self, request_id, outputfile):
    sglang_hooks.strict_idle(self)
    if base._SESSION is not None:
        raise ValueError('Cannot release a live capture')
    matches = [session for session in base._SESSIONS
               if session.request['request_id'] == request_id]
    if len(matches) != 1:
        raise ValueError('Require exactly one retained request to release')
    base._SESSIONS.remove(matches[0])
    receipt = dict(status='released', request_id=request_id,
                   requests_retained=len(base._SESSIONS))
    write_json(outputfile, receipt)
    return receipt


def scheduler_process(*args, **kwargs):
    base.NativeSuccessors = MarginSuccessors
    base.install_accounting()
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
    for function in (base.transition_begin, transition_margin_finished,
                     transition_one_block, transition_compact_request,
                     transition_margin_qa, transition_drop_request, base.transition_release):
        setattr(Scheduler, function.__name__, sglang_hooks._rpc(function))
    return run_scheduler_process(*args, **kwargs)


class MarginBackend(base.NativeTransitionBackend):
    def __init__(self, target_path, draft_path, *, worker_python, output_root,
                 budget_connection):
        from online_feedback.sglang_backend import SGLangDFlashBackend
        variable = 'DFLASH_TRANSITION_BUDGET'
        previous = os.environ.get(variable)
        os.environ[variable] = json.dumps(budget_connection)
        try:
            self.backend = SGLangDFlashBackend(
                target_path, draft_path, worker_python=worker_python,
                output_root=output_root, serving_gpu='0', context_limit=4096,
                block_size=16, worker_module='experiments.repair_history_margin.native_worker')
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
        self.loading_s = self.backend.loading_s
        self.serial = 0
        self._request_margins = {}

    def capture(self, request, checkpoint, folder):
        if request['stop_token_ids'] != self.backend.stop_tokens:
            raise ValueError('Sealed EOS IDs differ from the native target')
        self.backend.publish(checkpoint)
        canonical = request.get('canonical')
        if canonical is None:
            canonical = json.loads(Path(request['canonical_path']).read_text())
        native = dict(request, canonical_response_ids=canonical['response_ids'],
                      canonical_finish_reason=canonical['finish_reason'])
        self._rpc('transition_begin', request=native, checkpoint=str(checkpoint),
                  capture_rounds=list(range(1, request['max_new_tokens'] + 1)))
        record = self.backend.generate(
            request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
            request_id=request['request_id'], session_id='repair_history_margin',
            request_index=request['panel_ordinal'], purpose='evaluation',
            save_to=Path(folder) / 'K5_CAPTURE.json')
        receipt = self._rpc('transition_margin_finished')
        if (receipt['native_rounds'] != record['metadata']['verification_calls']
                or record['response_ids'] != canonical['response_ids']
                or record['finish_reason'] != canonical['finish_reason']):
            raise ValueError('Instrumented K5 differs from canonical outputs or native rounds')
        grouped = {str(row['output_start']): [] for row in record['metadata']['trace']}
        for row in receipt['margins']:
            grouped.setdefault(str(row['start_offset']), []).append(row)
        self._request_margins[request['source_id']] = dict(starts=grouped,
            observed_head_calls=receipt['observed_head_calls'])
        return record, receipt['successors']

    def one_block(self, state, checkpoint):
        if self.backend.checkpoint_id != str(checkpoint):
            self.backend.publish(checkpoint)
        return self._rpc('transition_one_block', handle=state['handle'],
                         checkpoint=str(checkpoint))

    def request_margins(self, source_id):
        return self._request_margins[source_id]

    def margin_qa(self):
        return self.audit_margins(self.backend.root / 'NATIVE_MARGIN_QA.pt')

    def compact_request(self, request_id):
        return self._rpc('transition_compact_request', request_id=request_id)

    def audit_margins(self, tensor_path):
        return self._rpc('transition_margin_qa', tensor_path=str(tensor_path))

    def drop_request(self, request_id):
        return self._rpc('transition_drop_request', request_id=request_id)


if __name__ == '__main__':
    base.scheduler_process = scheduler_process
    base.main()
