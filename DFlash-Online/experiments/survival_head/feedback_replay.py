"""Native frozen-target walk adapted from block_depth/reconstruct.py.

No author or ordinary serving implementation is modified. Only saved proposals
are used; the draft model is not evaluated. Original target KV bytes were retired,
so exact historical cache equality is not claimed. Native cache transitions and
the original full posterior/commit sequence are checked on every saved block.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from .feedback import (BLOCK_FIELDS, expected_reconstruction_work,
                       source_contract_sha256, validate_saved_record)


def check_cache_prefix(batch, expected_length, previous_locations=None):
    """Validate native retained prefix geometry and stable committed mappings."""
    import torch

    if len(batch.reqs) != 1 or int(batch.seq_lens[0]) != expected_length:
        raise RuntimeError('Native cache sequence length differs from canonical conditioning')
    if batch.seq_lens_cpu.tolist() != [expected_length] or int(batch.seq_lens_sum) != expected_length:
        raise RuntimeError('Native cache CPU/device lengths disagree')
    req = batch.reqs[0]
    if req.kv_committed_len != expected_length or req.kv_allocated_len != expected_length:
        raise RuntimeError('Native request KV accounting includes uncommitted rows')
    row = int(batch.req_pool_indices[0])
    locations = batch.req_to_token_pool.req_to_token[row, :expected_length].detach().clone()
    if (locations.numel() != expected_length or bool((locations < 0).any())
            or locations.unique().numel() != expected_length):
        raise RuntimeError('Native committed KV mapping is missing or aliases rows')
    if previous_locations is not None and not torch.equal(
            locations[:len(previous_locations)], previous_locations):
        raise RuntimeError('Native verify changed a previously committed cache mapping')
    return locations


def reconstruct(self, record, outputfile):
    import torch
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
    from sglang.srt.utils import require_mlp_sync
    from online_feedback import sglang_hooks as hooks
    from experiments.block_depth.reconstruct import replay_request

    prompt, response, blocks = validate_saved_record(record)
    destination = Path(outputfile)
    capture_path = destination.with_suffix('.capture.json')
    if (not destination.is_absolute() or any(p.exists() for p in (
            destination, capture_path, capture_path.with_suffix('.pt')))):
        raise FileExistsError('Require new absolute reconstruction artifacts')
    if (hooks._ACTIVE is not None or self.server_args.tp_size != 1
            or self.server_args.pp_size != 1 or require_mlp_sync(self.server_args)
            or self.server_args.page_size != 1):
        raise RuntimeError('Reconstruction requires idle qualified single-rank page-size-one native target')
    runner = self.tp_worker.model_runner
    if (self.draft_worker.block_size != 16 or self.draft_worker.use_compact_draft_cache
            or len(prompt) + len(response) > runner.model_config.context_len):
        raise RuntimeError('Native reconstruction context/block configuration differs')
    receipt = dict(status='running', record_id=record['record_id'],
        source_contract_sha256=source_contract_sha256(record),
        expected_work=expected_reconstruction_work(record),
        new_generations=0, draft_forwards=0, prefill_attempts=0, prefill_forwards=0,
        verification_attempts=0, verification_forwards=0, terminal_forwards=0,
        posterior_positions_checked=0, cache_prefix_checks=0, active_forward=None,
        historical_kv_byte_equality='unavailable_original_KV_not_retained',
        cache_check_scope='native_canonical_prefix_lengths_unique_slots_and_stable_committed_mapping',
        automatic_retry=False)
    started = time.perf_counter()
    capture = hooks.RequestCapture(prompt, capture_path, True)
    failure = None
    write_json(destination, receipt)
    try:
        capture.cache_maintenance_s = hooks.flush_idle(self)
        req = replay_request(Req, SamplingParams, record, runner.model_config)
        req.rid = 'survival_feedback_reconstruction:' + record['record_id']
        req.fill_ids = list(prompt)
        req.extend_input_len = len(prompt)
        cache = RadixCache(CacheInitParams(disable=False,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            page_size=runner.server_args.page_size))
        batch = ScheduleBatch.init_new(reqs=[req],
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=cache, model_config=runner.model_config,
            enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.DFLASH)
        batch.prepare_for_extend()
        model_batch = batch.get_model_worker_batch()
        if (list(model_batch.extend_prefix_lens) != [0]
                or list(model_batch.extend_seq_lens) != [len(prompt)]
                or model_batch.input_ids.reshape(-1).tolist() != prompt):
            raise RuntimeError('Native prefill input geometry differs from the retained capture')
        model_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        receipt.update(prefill_attempts=1, active_forward='prefill')
        write_json(destination, receipt)
        with torch.inference_mode():
            result = self.tp_worker.forward_batch_generation(model_batch)
        receipt.update(prefill_forwards=1, active_forward=None)
        first = result.next_token_ids.reshape(-1).cpu().tolist()
        if first != response[:1]:
            receipt['mismatch'] = dict(kind='prefill', expected=response[:1], actual=first)
            raise RuntimeError('Retained native prefill prediction differs')
        aux, final = hooks._feature_pair(result.logits_output, len(prompt))
        capture.prefill(0, prompt, first[0], aux=aux, final=final, block_size=16)
        req.output_ids.extend(first)
        req.check_finished()
        locations = check_cache_prefix(batch, len(prompt))
        receipt['cache_prefix_checks'] += 1
        _, custom_mask = resolve_dflash_verify_mask_policy(runner.attn_backend)
        for index, block in enumerate(blocks):
            before = list(req.output_ids)
            anchor = len(prompt) + len(before) - 1
            if req.finished() or before != response[:block['output_start'] + 1]:
                raise RuntimeError('Native saved start is terminal or differs from its canonical prefix')
            locations = check_cache_prefix(batch, anchor, locations)
            receipt['cache_prefix_checks'] += 1
            candidates = [block['anchor_token']] + block['draft_tokens']
            verify = DFlashVerifyInput(
                draft_token=torch.tensor(candidates, device=runner.device, dtype=torch.int64),
                positions=torch.tensor(block['positions'], device=runner.device, dtype=torch.int64),
                draft_token_num=16)
            verify.prepare_for_verify(batch, runner.server_args.page_size,
                                      build_custom_mask=custom_mask)
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            batch.spec_info = verify
            batch.return_hidden_states = False
            receipt['active_forward'] = dict(kind='verification', block=index)
            receipt['verification_attempts'] += 1
            write_json(destination, receipt)
            with torch.inference_mode():
                result = self.tp_worker.forward_batch_generation(batch.get_model_worker_batch(), is_verify=True)
            receipt['verification_forwards'] += 1
            receipt['active_forward'] = None
            posterior = result.logits_output.next_token_logits.argmax(-1).cpu().tolist()
            if posterior != block['target_argmax']:
                receipt['mismatch'] = dict(kind='verification', block=index,
                    expected=block['target_argmax'], actual=posterior)
                raise RuntimeError('Retained native full verification posterior differs')
            receipt['posterior_positions_checked'] += 16
            aux, final = hooks._feature_pair(result.logits_output, 16)
            with torch.inference_mode():
                next_ids, lengths, next_aux, accepted = verify.verify(batch=batch,
                    logits_output=result.logits_output, page_size=runner.server_args.page_size)
            count = int(lengths[0])
            if (count != block['committed_progress']
                    or req.output_ids != response[:len(before) + count]
                    or next_ids.reshape(-1).tolist() != req.output_ids[-1:]
                    or not torch.equal(aux[:count], next_aux)
                    or bool(req.finished()) != block['finished']):
                raise RuntimeError('Native reconstruction committed a different continuation or terminal state')
            capture.verification(before=before, after=list(req.output_ids),
                candidates=candidates, positions=block['positions'], posterior=posterior,
                commit_rows=count, native_accepted=int(accepted[0]), finished=req.finished(),
                aux=aux, final=final)
            if any(capture.trace[-1][key] != block[key] for key in BLOCK_FIELDS):
                raise RuntimeError('Native reconstructed trace differs from saved proposal/commit evidence')
            locations = check_cache_prefix(batch, anchor + count, locations)
            if not torch.equal(locations[-count:], batch.out_cache_loc.reshape(-1)):
                raise RuntimeError('Native retained cache slots differ from committed verify rows')
            receipt['cache_prefix_checks'] += 1
            batch.forward_mode = ForwardMode.DECODE
        if list(req.output_ids) != response or not req.finished():
            raise RuntimeError('Native replay did not terminate on the saved complete answer')
        torch.cuda.synchronize()
        observation = capture.finish(response)
        restored = deepcopy(record)
        restored['metadata']['capture'] = observation
        restored['metadata']['reconstructed_feedback'] = True
        restored['metadata']['reconstruction_receipt'] = str(destination)
        restored['metadata']['reconstruction_source_contract_sha256'] = receipt['source_contract_sha256']
        receipt.update(status='passed', record=restored,
            full_posterior_match=True, all_commits_exact=True,
            retained_native_rows=observation['covered_rows'],
            target_forwards=receipt['prefill_forwards'] + receipt['verification_forwards'])
        return receipt
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                       completed_trace=capture.trace, emitted_prefix=capture.teacher_ids)
        raise
    finally:
        try:
            receipt['cache_cleanup_s'] = hooks.flush_idle(self)
        except BaseException as error:
            receipt.update(status='failed', cleanup_error=repr(error))
            if failure is None:
                receipt['wall_s'] = time.perf_counter() - started
                write_json(destination, receipt)
                raise
        receipt['wall_s'] = time.perf_counter() - started
        write_json(destination, receipt)
