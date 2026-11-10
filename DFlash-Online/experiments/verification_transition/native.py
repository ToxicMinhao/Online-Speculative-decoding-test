"""Live native successors for the isolated, frozen five-layer diagnostic.

Target cache rows and committed target-feature ingestion are retained separately
from draft cache rows. A checkpoint switch always rebuilds the destination draft
cache through the ordinary native ingestion method. No training path is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import time
from types import SimpleNamespace

import torch


def tensor_bits_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8).cpu(),
                            right.contiguous().view(torch.uint8).cpu()))


def clone_values(value):
    """Own mutable batch values while leaving runtime services shared."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [clone_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_values(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_values(item) for key, item in value.items()}
    return value


def clone_fields(value):
    result = copy.copy(value)
    for name, field in vars(value).items():
        setattr(result, name, clone_values(field))
    return result


def clone_batch(batch):
    result = clone_fields(batch)
    result.reqs = [clone_fields(req) for req in batch.reqs]
    for req in result.reqs:
        req.sampling_params = clone_fields(req.sampling_params)
    result.spec_info = clone_fields(batch.spec_info)
    result.sampling_info = clone_fields(batch.sampling_info)
    return result


def run_counted(budget, kind, function, *, metadata=None):
    """Admission is durable before a potentially failing native operation."""
    ticket = budget.attempt(kind, **(metadata or {}))
    try:
        result = function()
    except BaseException as error:
        budget.fail(ticket, repr(error))
        raise
    budget.complete(ticket)
    return result


def acceptance_totals(blocks):
    """Keep the ordinary record convention and the native counter distinct.

    At a terminal boundary the runtime's committed-minus-one counter may be
    smaller than the number of retained draft proposals. Both remain visible.
    """
    return dict(accepted_draft_tokens=sum(row['accepted_draft_tokens'] for row in blocks),
                native_accepted_draft_tokens=sum(row['native_accepted_draft_tokens'] for row in blocks))


def cache_buffers(runner):
    """The qualified Qwen3 and DFlash models expose ordered attention layers."""
    model = runner.model
    layers = model.layers if hasattr(model, 'layers') else model.model.layers
    return [runner.token_to_kv_pool.get_kv_buffer(layer.self_attn.attn.layer_id)
            for layer in layers]


def copy_prefix_cache(runner, locations):
    return [(key[locations].detach().to('cpu', copy=True),
             value[locations].detach().to('cpu', copy=True))
            for key, value in cache_buffers(runner)]


def restore_prefix_cache(runner, locations, saved):
    buffers = cache_buffers(runner)
    if len(buffers) != len(saved):
        raise ValueError('Native cache layer count changed')
    for (key, value), (saved_key, saved_value) in zip(buffers, saved):
        key[locations] = saved_key.to(key.device)
        value[locations] = saved_value.to(value.device)


def compare_prefix_cache(runner, locations, saved):
    buffers = cache_buffers(runner)
    if len(buffers) != len(saved):
        return False
    return all(tensor_bits_equal(key[locations], original_key)
               and tensor_bits_equal(value[locations], original_value)
               for (key, value), (original_key, original_value) in zip(buffers, saved))


@dataclass
class Ingestion:
    start: int
    end: int
    hidden: torch.Tensor
    forward_mode: object
    is_extend_in_batch: bool
    draft_prefix_length: int


@dataclass
class Successor:
    handle: str
    checkpoint: str
    round_index: int
    original_max_new_tokens: int
    batch: object
    mapping: torch.Tensor
    locations: torch.Tensor
    allocator: object
    free_rows: list
    target_cache: list
    draft_reference_cache: list
    ingestions: tuple

    @property
    def output_ids(self):
        return list(self.batch.reqs[0].output_ids)

    @property
    def terminal(self):
        return bool(self.batch.reqs[0].finished())

    def metadata(self):
        return dict(handle=self.handle, checkpoint=self.checkpoint,
                    round_index=self.round_index, output_tokens=len(self.output_ids),
                    original_max_new_tokens=self.original_max_new_tokens,
                    remaining_output_budget=self.original_max_new_tokens-len(self.output_ids),
                    target_prefix_tokens=int(self.locations.numel()),
                    ingestion_segments=len(self.ingestions), terminal=self.terminal)


class NativeSuccessors:
    """One request's source states, captured before the scheduler releases KV.

    The adapter observes native calls, then removes itself at request completion.
    Destination publication can flush caches: saved source target rows are owned
    CPU copies, and no destination draft cache is copied from the source.
    """

    def __init__(self, scheduler, request, checkpoint, capture_rounds, budget):
        self.scheduler = scheduler
        self.worker = scheduler.draft_worker
        self.request = request
        self.checkpoint = str(checkpoint)
        self.capture_rounds = set(capture_rounds)
        self.budget = budget
        self.ingestions = []
        self.successors = {}
        self.round_index = 0
        self.capture_enabled = True
        self.installed = False
        self.reconstruction = None
        worker = self.worker
        if (worker.page_size != 1 or worker.use_compact_draft_cache
                or worker.block_size != 16 or len(worker.draft_model.layers) != 5):
            raise ValueError('Require the qualified five-layer, block-16, full-context native cache')
        if worker.draft_model_runner.req_to_token_pool is not worker.model_runner.req_to_token_pool:
            raise ValueError('Qualified target and draft request mappings must alias')
        if any(type(index) is not int or index < 1 for index in self.capture_rounds):
            raise ValueError('Capture boundaries are one-based native round indices')

    def install(self):
        if self.installed:
            raise RuntimeError('Native successor capture is already installed')
        worker = self.worker
        self.original_append = worker._append_target_hidden_to_draft_kv
        self.original_forward = worker.forward_batch_generation

        def append(batch, draft_input):
            length = int(draft_input.target_hidden.shape[0])
            if length:
                end = int(batch.seq_lens[0].item())
                if self.capture_enabled and any(index > self.round_index for index in self.capture_rounds):
                    start = end-length
                    expected = self.ingestions[-1].end if self.ingestions else 0
                    if start != expected or int(draft_input.ctx_lens[0]) != length:
                        raise ValueError('Canonical target ingestion has a gap or overlapping rows')
                    self.ingestions.append(Ingestion(start, end,
                        draft_input.target_hidden.detach().to('cpu', copy=True),
                        batch.forward_mode, bool(batch.is_extend_in_batch),
                        int(draft_input.draft_seq_lens[0].item())))
                return run_counted(self.budget, 'context_forward',
                    lambda: self.original_append(batch, draft_input),
                    metadata=dict(operation='native_draft_context_ingestion', tokens=length,
                                  rebuild=not self.capture_enabled))
            return self.original_append(batch, draft_input)

        def forward(batch, **kwargs):
            prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
            result = self.original_forward(batch, **kwargs)
            if not prefill:
                self.round_index += 1
                if self.capture_enabled and self.round_index in self.capture_rounds:
                    self.capture(batch)
            return result

        worker._append_target_hidden_to_draft_kv = append
        worker.forward_batch_generation = forward
        self.installed = True

    def close(self):
        if self.installed:
            self.worker._append_target_hidden_to_draft_kv = self.original_append
            self.worker.forward_batch_generation = self.original_forward
            self.installed = False

    @torch.inference_mode()
    def capture(self, batch):
        req = batch.reqs[0]
        if len(batch.reqs) != 1 or req.grammar is not None:
            raise ValueError('Successor capture requires C1 without grammar')
        limit = int(req.sampling_params.max_new_tokens)
        if limit != self.request['max_new_tokens']:
            raise ValueError('Native request output cap differs from its sealed original cap')
        if list(req.origin_input_ids) != self.request['prompt_ids']:
            raise ValueError('Native successor belongs to another prompt')
        prefix_length = int(batch.seq_lens[0].item())
        if prefix_length != len(req.origin_input_ids)+len(req.output_ids)-1:
            raise ValueError('Native cache includes a missing or uncommitted seed position')
        if not self.ingestions or self.ingestions[-1].end != prefix_length:
            raise ValueError('Committed target-feature history is incomplete')
        runner = self.worker.model_runner
        pool = runner.req_to_token_pool
        row = int(batch.req_pool_indices[0].item())
        mapping = pool.req_to_token[row].clone()
        locations = mapping[:prefix_length].to(torch.int64)
        if locations.unique().numel() != prefix_length:
            raise ValueError('Native valid-prefix slots overlap')
        handle = f'{self.request["request_id"]}:{self.checkpoint}:{self.round_index}'
        self.successors[self.round_index] = Successor(handle, self.checkpoint,
            self.round_index, limit, clone_batch(batch), mapping, locations,
            clone_values(runner.token_to_kv_pool_allocator.backup_state()),
            list(pool.free_slots), copy_prefix_cache(runner, locations),
            copy_prefix_cache(self.worker.draft_model_runner, locations), tuple(self.ingestions))

    @torch.inference_mode()
    def rebuild(self, successor, destination_checkpoint):
        """Restore source target context, then ingest it with destination weights."""
        worker = self.worker
        runner = worker.model_runner
        batch = clone_batch(successor.batch)
        pool = runner.req_to_token_pool
        row = int(batch.req_pool_indices[0].item())
        runner.token_to_kv_pool_allocator.restore_state(clone_values(successor.allocator))
        pool.free_slots[:] = successor.free_rows
        pool.req_to_token[row].copy_(successor.mapping)
        restore_prefix_cache(runner, successor.locations, successor.target_cache)
        self.capture_enabled = False
        self.round_index = successor.round_index
        for segment in successor.ingestions:
            self.budget.check_time()
            # Replay each original native ingestion shape, including prefill
            # chunks, instead of concatenating all features into a new kernel.
            context_batch = copy.copy(batch)
            context_batch.forward_mode = segment.forward_mode
            context_batch.is_extend_in_batch = segment.is_extend_in_batch
            context_batch.seq_lens = torch.full_like(batch.seq_lens, segment.end)
            info = SimpleNamespace(target_hidden=segment.hidden.to(worker.device),
                ctx_lens=torch.full_like(batch.seq_lens, segment.end-segment.start),
                draft_seq_lens=torch.full_like(batch.seq_lens, segment.draft_prefix_length))
            worker._append_target_hidden_to_draft_kv(context_batch, info)
            if int(info.draft_seq_lens[0]) != segment.end or info.target_hidden.shape[0] != 0:
                raise ValueError('Native context ingestion did not consume its exact segment')
        if int(batch.spec_info.draft_seq_lens[0]) != len(successor.locations):
            raise ValueError('Restored native draft position metadata changed')
        target_equal = compare_prefix_cache(runner, successor.locations, successor.target_cache)
        neutral = str(destination_checkpoint) == successor.checkpoint
        draft_equal = (compare_prefix_cache(worker.draft_model_runner, successor.locations,
                                            successor.draft_reference_cache) if neutral else None)
        self.reconstruction = dict(status='passed' if target_equal and (not neutral or draft_equal) else 'failed',
            target_prefix_cache_exact=target_equal,
            neutral_draft_prefix_cache_exact=draft_equal, destination_draft_rebuilt=True,
            source_draft_cache_transplanted=False, original_cap=successor.original_max_new_tokens,
            remaining_budget=successor.original_max_new_tokens-len(successor.output_ids),
            ingestion_segments=len(successor.ingestions),
            ingestion_tokens=sum(r.end-r.start for r in successor.ingestions))
        if not target_equal or (neutral and not draft_equal):
            raise ValueError('Native successor cache reconstruction differs bitwise')
        return batch


def assert_neutral_suffix(original, successor, replay):
    """No proposal, progress, output, termination or call-count tolerance."""
    expected = original['metadata']['trace'][successor['round_index']:]
    actual = replay['metadata']['trace']
    fields = ('draft_tokens', 'committed_progress', 'committed_tokens', 'positions',
              'target_argmax', 'finished', 'output_start')
    if len(expected) != len(actual):
        raise ValueError('Neutral continuation verification count differs')
    for index, (left, right) in enumerate(zip(expected, actual)):
        if any(left[key] != right[key] for key in fields):
            raise ValueError(f'Neutral native suffix differs at subsequent round {index+1}')
    if (original['response_ids'] != replay['response_ids']
            or original['finish_reason'] != replay['finish_reason']):
        raise ValueError('Neutral canonical answer or termination differs')
    return dict(status='passed', verification_calls=len(actual),
                all_proposals_exact=True, all_commits_exact=True,
                canonical_output_exact=True, termination_exact=True)
