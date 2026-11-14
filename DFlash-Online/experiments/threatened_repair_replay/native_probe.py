"""Reconstruct saved K6 starts, then use the qualified native one-block replay.

No engine.generate call or new parent trajectory is exposed. A saved path is
replayed with strict proposal gates before target verification. CPU-owned target
cache snapshots retain the original ingestion shapes; each destination rebuilds
its own draft KV through NativeSuccessors.rebuild.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
import traceback

import torch

from online_feedback import sglang_hooks
from online_feedback.native import validate_generation
from online_feedback.storage import write_json
from experiments.block_depth.reconstruct import replay_request
from experiments.lost_repair.native_worker import CensusSuccessors
from experiments.longer_run_headroom.native_worker import transition_one_block
from experiments.longer_run_headroom.pairs import neutral_parity
from experiments.repair_history_margin.native_worker import (
    compact_prefix_caches, materialize_successor_metadata, offload_successor_metadata,
)
from experiments.survival_head.feedback import BLOCK_FIELDS
from experiments.survival_head.feedback_replay import check_cache_prefix
from experiments.verification_transition import native_worker as base


class ProbeSuccessors(CensusSuccessors):
    """Capture only the four sealed pre-block boundaries, including zero."""
    def __init__(self, scheduler, request, checkpoint, ordinals, budget):
        self.selected_ordinals = set(ordinals)
        super().__init__(scheduler, request, checkpoint,
                         sorted(i for i in self.selected_ordinals if i > 0), budget)

    def capture_opening(self, batch):
        if 0 in self.selected_ordinals:
            super().capture_opening(batch)

    def rebuild(self, successor, destination_checkpoint):
        return super().rebuild(materialize_successor_metadata(successor), destination_checkpoint)


def validate_parent(record, keys):
    prompt, response, blocks = record['prompt_ids'], record['response_ids'], record['metadata']['trace']
    _, finish = validate_generation(prompt, prompt + response, blocks,
        max_new_tokens=record['max_new_tokens'], stop_tokens=record['stop_token_ids'])
    metadata = record['metadata']
    if (finish != record['finish_reason'] or metadata['verification_calls'] != len(blocks)
            or metadata['block_size'] != 16 or metadata['runtime'] != 'pinned_SGLang_FA3'
            or metadata['cache_policy'] != 'idle_full_flush_before_every_request_and_publication'
            or not 0 < len(prompt) <= 4096 or len(prompt) + len(response) > 4096
            or metadata['capture']['prefill_chunks'] != 1):
        raise ValueError('Retained parent runtime, context or exact prefill geometry differs')
    identities = {(row['request_id'], row['canonical_start'], row['round_ordinal']) for row in keys}
    if len(keys) != 4 or len(identities) != 4:
        raise ValueError('Require the four sealed distinct starts for this request')
    for row in keys:
        ordinal = row['round_ordinal']
        if (row['request_id'] != record['record_id'] or not 0 <= ordinal < len(blocks)
                or blocks[ordinal]['output_start'] != row['canonical_start']):
            raise ValueError('Sealed probe key differs from retained parent')
    for block in blocks:
        if (len(block['draft_tokens']) != 15 or block['verify_size'] != 16
                or len(block['target_argmax']) != 16 or not set(BLOCK_FIELDS).issubset(block)):
            raise ValueError('Retained parent does not contain all fifteen proposals and native verdicts')
    return prompt, response, blocks


@contextmanager
def require_saved_proposals(block, verify_type=None):
    """Reject all-15 proposal mismatch before allocating/verifying a new block."""
    if verify_type is None:
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput
        verify_type = DFlashVerifyInput
    ordinary = verify_type.prepare_for_verify
    receipt = dict(calls=0, expected_tokens=[block['anchor_token']] + block['draft_tokens'])

    def checked(info, *args, **kwargs):
        receipt['calls'] += 1
        actual = info.draft_token.reshape(-1).tolist()
        positions = info.positions.reshape(-1).tolist()
        if actual != receipt['expected_tokens'] or positions != block['positions'] or info.draft_token_num != 16:
            receipt.update(actual_tokens=actual, actual_positions=positions)
            raise ValueError('Reconstructed K6 proposals/positions differ before target verification')
        return ordinary(info, *args, **kwargs)

    verify_type.prepare_for_verify = checked
    try:
        yield receipt
        if receipt['calls'] != 1:
            raise ValueError('Expected exactly one native saved-block verification preparation')
    finally:
        verify_type.prepare_for_verify = ordinary


def validate_replayed_block(record, ordinal, observed, output_ids, finished):
    expected = record['metadata']['trace'][ordinal]
    parity = neutral_parity(expected, observed)
    end = expected['output_start'] + expected['committed_progress'] + 1
    if (parity['status'] != 'passed' or list(output_ids) != record['response_ids'][:end]
            or bool(finished) != expected['finished']):
        raise ValueError(f'Reconstructed native posterior/commit/output/termination differs at round {ordinal}')
    return parity


def host_memory_available():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def cache_storage_bytes(sessions):
    """Count shared CPU tensor storages once, including immutable ingestion rows."""
    seen = {}
    for session in sessions:
        for state in session.successors.values():
            for cache in (state.target_cache, state.draft_reference_cache):
                for pair in cache:
                    for tensor in pair:
                        storage = tensor.untyped_storage()
                        seen[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
            for segment in state.ingestions:
                storage = segment.hidden.untyped_storage()
                seen[(str(segment.hidden.device), storage.data_ptr())] = storage.nbytes()
    return sum(seen.values())


def make_batch(scheduler, record):
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    runner = scheduler.tp_worker.model_runner
    req = replay_request(Req, SamplingParams, record, runner.model_config)
    req.rid = 'threat_probe_reconstruction:' + record['record_id']
    req.fill_ids = list(record['prompt_ids'])
    req.extend_input_len = len(req.fill_ids)
    cache = RadixCache(CacheInitParams(disable=False, req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator, page_size=runner.server_args.page_size))
    batch = ScheduleBatch.init_new(reqs=[req], req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator, tree_cache=cache,
        model_config=runner.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.DFLASH)
    batch.prepare_for_extend()
    actual = batch.get_model_worker_batch()
    if (list(actual.extend_prefix_lens) != [0] or list(actual.extend_seq_lens) != [len(req.fill_ids)]
            or actual.input_ids.reshape(-1).tolist() != req.fill_ids):
        raise ValueError('Reconstruction prefill geometry differs from retained one-chunk input')
    return batch


def transition_reconstruct_saved(self, record, keys, checkpoint, outputfile,
                                 host_reserve_bytes=32 * 1024**3):
    """Re-evaluate only an authenticated saved path; publish snapshots after parity."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    sglang_hooks.strict_idle(self)
    if base._SESSION is not None or sglang_hooks._ACTIVE is not None:
        raise ValueError('Probe reconstruction requires an idle isolated worker')
    prompt, response, blocks = validate_parent(record, keys)
    if str(checkpoint) != str(self._online_publication['checkpoint_path']) or record['checkpoint_id'] != str(checkpoint):
        raise ValueError('Original K6 checkpoint must be published for saved-path reconstruction')
    if any(s.request['request_id'] == record['record_id'] for s in base._SESSIONS):
        raise ValueError('Parent has already been reconstructed; no automatic retries')
    destination = Path(outputfile)
    if not destination.is_absolute() or destination.exists():
        raise ValueError('Reconstruction requires a new absolute private receipt path')
    request = dict(request_id=record['record_id'], prompt_ids=prompt, max_new_tokens=record['max_new_tokens'],
        stop_token_ids=record['stop_token_ids'], canonical_response_ids=response,
        canonical_finish_reason=record['finish_reason'])
    session = ProbeSuccessors(self, request, checkpoint, [row['round_ordinal'] for row in keys], base._BUDGET)
    recorder = sglang_hooks.RequestCapture(prompt, str(destination.with_suffix('.trace.json')), False)
    receipt = dict(status='running', request_id=record['record_id'], new_generations=0,
        saved_rounds=len(blocks), rounds_replayed=0, selected_starts=len(keys),
        host_reserve_bytes=host_reserve_bytes,
        historical_kv_byte_equality='unavailable_original_KV_not_retained',
        source='retained_PR44_PARENT_K6_exact_tokens_and_proposals')
    began = time.perf_counter()
    failure = None
    write_json(destination, receipt)
    try:
        recorder.cache_maintenance_s = sglang_hooks.flush_idle(self)
        batch = make_batch(self, record)
        req = batch.reqs[0]
        base._SESSION = session
        sglang_hooks._ACTIVE = recorder
        session.install()
        with torch.inference_mode():
            result = session.worker.forward_batch_generation(batch)
            first = result.next_token_ids.reshape(-1).tolist()
            if first != response[:1]:
                raise ValueError('Reconstructed prefill seed differs from retained K6')
            req.output_ids.extend(first)
            req.check_finished()
            batch.forward_mode = ForwardMode.DECODE
            batch.is_extend_in_batch = False
            locations = check_cache_prefix(batch, len(prompt))
            for ordinal, block in enumerate(blocks):
                base._BUDGET.check_time()
                if req.finished() or req.output_ids != response[:block['output_start'] + 1]:
                    raise ValueError('Saved-path reconstruction left its exact canonical prefix')
                receipt['active_round'] = ordinal
                write_json(destination, receipt)
                with require_saved_proposals(block) as gate:
                    session.worker.forward_batch_generation(batch)
                observed = recorder.trace[-1]
                validate_replayed_block(record, ordinal, observed, req.output_ids, req.finished())
                locations = check_cache_prefix(batch, len(prompt) + len(req.output_ids) - 1, locations)
                receipt.update(rounds_replayed=ordinal + 1, active_round=None,
                               last_proposal_gate_calls=gate['calls'])
            if list(req.output_ids) != response or not req.finished():
                raise ValueError('Reconstructed saved path did not reach its original terminal output')
            torch.cuda.synchronize()
        recorder.finish(response)
        if set(session.successors) != {row['round_ordinal'] for row in keys}:
            raise ValueError('Reconstruction did not retain exactly the sealed starts')
        compaction = compact_prefix_caches(session)
        metadata = [offload_successor_metadata(state) for state in session.successors.values()]
        projected_bytes = cache_storage_bytes(base._SESSIONS + [session])
        if host_memory_available() < host_reserve_bytes:
            raise MemoryError('Probe snapshots reached the sealed host-memory reserve')
        # No snapshot is admitted before every retained round passes exact parity.
        base._SESSIONS.append(session)
        receipt.update(status='passed', all_15_proposals_exact=True, all_posteriors_exact=True,
            all_commits_exact=True, canonical_output_exact=True, termination_exact=True,
            snapshots={str(k): state.metadata() for k, state in session.successors.items()},
            compaction=compaction, metadata_offload=metadata, retained_cache_bytes=projected_bytes,
            host_available_bytes=host_memory_available())
        return receipt
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                       completed_trace=recorder.trace, emitted_prefix=recorder.teacher_ids)
        raise
    finally:
        session.close()
        base._SESSION = None
        sglang_hooks._ACTIVE = None
        try:
            receipt['cleanup_s'] = sglang_hooks.flush_idle(self)
        except BaseException as error:
            receipt.update(status='failed', cleanup_error=repr(error))
            if failure is None:
                receipt['wall_s'] = time.perf_counter() - began
                write_json(destination, receipt)
                raise
        receipt['wall_s'] = time.perf_counter() - began
        write_json(destination, receipt)


def install_context_accounting():
    """Real context ingestion is not a startup proposal block during replay."""
    from sglang.srt.speculative.dflash_worker import DFlashWorker
    original = DFlashWorker._append_target_hidden_to_draft_kv

    def append(worker, batch, info):
        if base._SESSION is None:
            return original(worker, batch, info)
        base._DRAFT_ADMITTED += 1
        try:
            return original(worker, batch, info)
        finally:
            base._DRAFT_ADMITTED -= 1
    DFlashWorker._append_target_hidden_to_draft_kv = append


def scheduler_process(*args, **kwargs):
    base.install_accounting()
    install_context_accounting()
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
    for function in (transition_reconstruct_saved, transition_one_block, base.transition_release):
        setattr(Scheduler, function.__name__, sglang_hooks._rpc(function))
    return run_scheduler_process(*args, **kwargs)


if __name__ == '__main__':
    base.scheduler_process = scheduler_process
    base.main()
