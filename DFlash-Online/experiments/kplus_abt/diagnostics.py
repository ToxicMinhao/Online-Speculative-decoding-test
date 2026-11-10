"""Four fixed post-update clean replays per event; no learning from probes."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import time

import torch

from online_feedback.first_error import first_error_mask
from online_feedback.adversarial_block import FrozenKPlusChunk
from online_feedback.prefix_guard import prefix_changes
from online_feedback.ranking import fixed_anchors, matching_prefix_mask
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from experiments.ok_continuation.resume import differences
from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from .contract import DIAGNOSTIC_WINDOWS


class BlockCapture:
    """Observe already-computed proposals once, despite checkpoint recomputation."""
    def __init__(self):
        self.chunks = {}

    def __call__(self, proposals, labels, weights, selection):
        key = labels.storage_offset()
        if key not in self.chunks:
            self.chunks[key] = tuple(x.detach().cpu().clone() for x in
                                    (proposals, labels, weights, selection))

    def joined(self):
        return tuple(torch.cat([self.chunks[key][i] for key in sorted(self.chunks)], dim=1)
                     for i in range(4))


class FixedObjective:
    """Score the original clean support/groups, with new post-update logits."""
    def __init__(self, snapshot, device, eos):
        self.snapshot = {k:v.to(device) for k,v in snapshot.items() if isinstance(v, torch.Tensor)}
        self.eos, self.chunks = eos, {}

    def freeze_selection(self, proposals, labels, weights, selection):
        key = labels.storage_offset()
        block_start = key // labels.shape[-1]
        length = labels.shape[1]
        sl = (slice(None), slice(block_start, block_start+length))
        old = self.snapshot
        if not torch.equal(labels, old['labels'][sl]) or not torch.equal(weights, old['weights'][sl]):
            raise ValueError('Fixed diagnostic block identity changed')
        if key not in self.chunks:
            self.chunks[key] = FrozenKPlusChunk.capture(old['clean_proposals'][sl], labels,
                weights, self.eos, old['selection'][sl])
        return self.chunks[key].selection

    def group_terms(self, token_loss, labels):
        return self.chunks[labels.storage_offset()].terms(token_loss)[2:]

    def freeze_response(self, terms, denominator):
        if not torch.equal(denominator, self.snapshot['denominator']):
            raise ValueError('Fixed diagnostic response denominator changed')
        return self.snapshot['denominator']


@contextmanager
def capture_blocks(model, observer):
    prior = getattr(model, 'block_observer', None)
    model.block_observer = observer
    try:
        yield
    finally:
        if prior is None:
            del model.block_observer
        else:
            model.block_observer = prior


def install(trainer, feature_paths, output, event):
    """Attach bounded probes after the normal optimizer step, for every arm."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    results, pending = [], []
    model = trainer.bundle.model
    raw_forward = trainer._native_forward
    before_step = trainer._native_step
    ledger = WorkLedger(root/'WORK.jsonl', dict(response_forwards=8))

    def window():
        return trainer.global_step-(event-1)*128+1

    def remember(batch, output, snapshot, record):
        if window() in DIAGNOSTIC_WINDOWS:
            pending.append((batch, snapshot, record['clean_loss'],
                            str(batch.sample_ids[0])))

    if getattr(trainer, 'two_view_installed', False):
        trainer.response_observer = remember
    else:
        def observed(batch, ctx=None):
            if window() not in DIAGNOSTIC_WINDOWS:
                return raw_forward(batch, ctx)
            observer = BlockCapture()
            old_noise = model._create_noise_embed
            measured = {}
            def noise(*args, **kwargs):
                measured['rng'] = capture_rng()
                return old_noise(*args, **kwargs)
            model._create_noise_embed = noise
            try:
                with capture_blocks(model, observer):
                    result = raw_forward(batch, ctx)
            finally:
                model._create_noise_embed = old_noise
            proposals, labels, weights, selection = observer.joined()
            draws = trainer._anchor_observations
            if len(draws) != 1:
                raise ValueError('A response diagnostic requires one native anchor draw')
            draw = draws[0]
            snapshot = dict(model_rng=measured['rng'], clean_proposals=proposals, labels=labels, weights=weights,
                selection=selection, denominator=result.loss_terms[1].detach().cpu(),
                anchors=torch.as_tensor(draw['positions']),
                keep=torch.as_tensor(draw['keep_mask'], dtype=torch.bool))
            remember(batch, result, snapshot, dict(clean_loss=float(result.loss.detach())))
            return result
        trainer._native_forward = observed

    def score_after_step():
        from specforge.training.strategies.base import StepContext
        if len(pending) != 2:
            raise ValueError('Fixed diagnostic window must contain exactly two current responses')
        started = time.perf_counter()
        before_rng, runtime = capture_rng(), runtime_state(model)
        receipt = dict(window=window(), status='running', rows=[], response_forwards=0,
                       response_backwards=0, optimizer_steps=0)
        failure = None
        try:
            with preserve_rng(), torch.no_grad():
                # Keep training conditioning/mode; only autograd is disabled.
                for batch, snapshot, pre_loss, record_id in pending:
                    device = next(model.draft_model.parameters()).device
                    observer = BlockCapture()
                    restore_rng(snapshot['model_rng'])
                    model.adversarial_block = FixedObjective(snapshot, device, model.eos_token_ids)
                    try:
                        with fixed_anchors(model, snapshot['anchors'].to(device), snapshot['keep'].to(device)), capture_blocks(model, observer):
                            result = ledger.call('response_forwards', raw_forward, batch,
                                StepContext(global_step=trainer.global_step, total_steps=trainer.horizon,
                                            collect_detailed_metrics=True))
                    finally:
                        del model.adversarial_block
                    proposals, labels, weights, _ = observer.joined()
                    changes = prefix_changes(snapshot['clean_proposals'], proposals, labels,
                                             weights, model.eos_token_ids)
                    receipt['rows'].append(dict(record_id=record_id, clean_fixed_support_loss_before=pre_loss,
                        clean_fixed_support_loss_after=float(result.loss.detach()), **changes))
                    receipt['response_forwards'] += 1
                    save_torch(root/f"window_{window():03d}_{len(receipt['rows'])}.pt",
                               dict(before=snapshot, after_proposals=proposals))
            receipt['status'] = 'completed'
        except BaseException as error:
            failure = error
            receipt.update(status='failed', error=repr(error))
            raise
        finally:
            restore_runtime(model, runtime)
            receipt['rng_restored_exact'] = not differences(capture_rng(), before_rng)
            receipt['wall_s'] = time.perf_counter()-started
            write_json(root/f'window_{window():03d}.json', receipt)
            pending.clear()
            if failure is None and not receipt['rng_restored_exact']:
                raise ValueError('Diagnostic changed training RNG')
        results.append(receipt)
        write_json(root/'RESULTS.json', results)

    def observed_step():
        result = before_step()
        if window() in DIAGNOSTIC_WINDOWS:
            score_after_step()
        return result
    trainer._native_step = observed_step
    return results
