"""Small matched native-block diagnostic, isolated from training RNG/state."""
from copy import deepcopy
from pathlib import Path
import time

import torch

from online_feedback.first_error import first_error_mask, valid_prefix_mask
from online_feedback.ranking import fixed_anchors, matching_prefix_mask
from online_feedback.rng import preserve_rng
from online_feedback.storage import save_torch, write_json

SCHEMA = 'repair_pair_fixed_witness_v1'


def freeze_witness(trainer, feature_paths, destination):
    """Seal native draws from exactly the first eight BASE event-one responses.

    Feature packets stay in their retained private event paths. Their identities
    and all selected anchors are stored here; no response is replaced if short.
    """
    from experiments.ok_coding.storage import identity
    from specforge.training.strategies.base import _cpu_max_valid_anchors
    if trainer.arm != 'BASE' or trainer.global_step or trainer.presentations or trainer.events:
        raise ValueError('Freeze the shared witness at BASE origin, before its first update')
    if len(feature_paths) < 8 or Path(destination).exists():
        raise ValueError('Need the first eight ordered responses and a new witness path')
    model = trainer.bundle.model
    device = next(model.draft_model.parameters()).device
    rows = []
    with preserve_rng(), torch.no_grad():
        for ordinal, path in enumerate(feature_paths[:8]):
            raw = torch.load(path, map_location='cpu', weights_only=True)
            batch = trainer._batch(raw)
            mask = batch.tensors['loss_mask']
            available = _cpu_max_valid_anchors(mask)
            if available:
                anchors, keep = trainer._original_sampler(
                    batch.tensors['input_ids'].shape[-1], mask.to(device), device,
                    max_valid_anchors=available)
                if anchors.shape[0] != 1:
                    raise ValueError('Witness requires the native single-response block sampler')
                chosen = anchors[0, keep[0]][:8].detach().cpu()
            else:
                chosen = torch.empty(0, dtype=torch.long)
            rows.append(dict(ordinal=ordinal, record_id=str(raw['record_id']),
                feature_path=str(Path(path).absolute()), feature_identity=identity(path),
                anchors=chosen.tolist(), keep=[True] * len(chosen)))
        receipt = dict(schema=SCHEMA, frozen_at='BASE released origin before any production update',
            feature_source='first8_BASE_event1_own_feedback', maximum_anchors=8, rows=rows,
            optimizer_steps=0, diagnostic_forwards=0, rng_preserved=True)
        save_torch(destination, receipt)
    return receipt


def block_summary(proposals, labels, weights, eos_token_ids=()):
    """Keep exact predictions/validity and the native consecutive-match count."""
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    prefix = matching_prefix_mask(proposals, labels, weights, eos_token_ids)
    selected = first_error_mask(proposals, labels, weights, eos_token_ids)
    error = selected & valid & (proposals != labels)
    return dict(proposals=proposals.cpu().tolist(), labels=labels.cpu().tolist(),
        valid=valid.cpu().tolist(), prefix=prefix.cpu().tolist(),
        first_error=error.cpu().tolist(), prefix_length=prefix.sum(-1).cpu().tolist())


def compare_blocks(origin, endpoint):
    """A repaired old boundary is reported alongside any earlier prefix damage."""
    if origin['labels'] != endpoint['labels'] or origin['valid'] != endpoint['valid']:
        raise ValueError('Matched witness labels or native validity changed')
    y = torch.tensor(origin['labels'])
    q = torch.tensor(endpoint['proposals'])
    old_prefix = torch.tensor(origin['prefix'], dtype=torch.bool)
    old_error = torch.tensor(origin['first_error'], dtype=torch.bool)
    damage = old_prefix & (q != y)
    repaired = old_error & (q == y)
    old_lengths = torch.tensor(origin['prefix_length'])
    new_lengths = torch.tensor(endpoint['prefix_length'])
    return dict(original_boundaries=int(old_error.sum()), repaired_boundaries=int(repaired.sum()),
        original_prefix_positions=int(old_prefix.sum()), damaged_prefix_positions=int(damage.sum()),
        damaged_prefix_blocks=int(damage.any(-1).sum()),
        repaired_without_prefix_damage=int((repaired.any(-1) & ~damage.any(-1)).sum()),
        net_consecutive_match_length_change=int((new_lengths - old_lengths).sum()),
        improved_blocks=int((new_lengths > old_lengths).sum()),
        regressed_blocks=int((new_lengths < old_lengths).sum()))


def evaluate_witness(trainer, witness_path, output_path, *, origin_path=None, endpoint=None):
    """One draft forward per nonempty response, no backward or teacher forward."""
    from experiments.ok_coding.storage import identity
    witness = torch.load(witness_path, map_location='cpu', weights_only=False)
    if witness.get('schema') != SCHEMA or len(witness['rows']) != 8:
        raise ValueError('Malformed sealed matched witness')
    if Path(output_path).exists():
        raise FileExistsError('Do not overwrite a witness readout')
    if origin_path is None and (trainer.global_step or trainer.presentations):
        raise ValueError('The common witness origin must precede all production updates')
    model = trainer.bundle.model
    device = next(model.draft_model.parameters()).device
    before = (trainer.global_step, trainer.presentations, len(trainer.events))
    work, costs, anchors_seen = deepcopy(trainer.work), deepcopy(model.costs), deepcopy(trainer._anchor_observations)
    modes = [(module, module.training) for module in model.modules()]
    started = time.perf_counter()
    rows = []
    result = dict(schema=SCHEMA, status='running', arm=trainer.arm, endpoint=endpoint,
        global_step=trainer.global_step, witness_identity=identity(witness_path), rows=rows,
        diagnostic_draft_forwards_attempted=0, diagnostic_draft_forwards=0,
        diagnostic_head_calls_attempted=0, diagnostic_head_calls=0,
        teacher_forwards=0, backward_calls=0, optimizer_steps=0, automatic_retry=False)
    write_json(output_path, result)
    failure = None
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            for item in witness['rows']:
                if identity(item['feature_path']) != item['feature_identity']:
                    raise ValueError('A retained witness feature packet changed')
                if not item['anchors']:
                    rows.append(dict(record_id=item['record_id'], anchors=[], blocks=None))
                    continue
                raw = torch.load(item['feature_path'], map_location='cpu', weights_only=True)
                batch = trainer._batch(raw)
                tensors = {name: batch.tensors[name].to(device) for name in
                           ('input_ids', 'hidden_states', 'loss_mask')}
                anchors = torch.tensor([item['anchors']], device=device, dtype=torch.long)
                keep = torch.ones_like(anchors, dtype=torch.bool)
                result['diagnostic_draft_forwards_attempted'] += 1
                write_json(output_path, result)
                with fixed_anchors(model, anchors, keep):
                    actual, actual_keep, hidden = model._forward_draft_blocks(**tensors)
                result['diagnostic_draft_forwards'] += 1
                if not torch.equal(actual, anchors) or not torch.equal(actual_keep, keep):
                    raise ValueError('Native witness block anchors changed')
                labels, weights, _ = model._block_labels_and_weights(
                    tensors['input_ids'], tensors['loss_mask'], anchors, keep)
                hidden = hidden.reshape(1, len(item['anchors']), model.block_size, -1)
                predictions = []
                for begin in range(0, hidden.shape[1], model.objective_chunk_blocks):
                    result['diagnostic_head_calls_attempted'] += 1
                    write_json(output_path, result)
                    predictions.append(model.lm_head(hidden[:, begin:begin + model.objective_chunk_blocks])
                        .to(model.loss_dtype).argmax(-1))
                    result['diagnostic_head_calls'] += 1
                blocks = block_summary(torch.cat(predictions, dim=1), labels, weights, model.eos_token_ids)
                rows.append(dict(record_id=item['record_id'], anchors=item['anchors'], blocks=blocks))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
    except BaseException as error:
        failure = error
        result.update(status='failed', error=repr(error), rng_preserved=True)
        raise
    finally:
        # Capture attempted physical work before restoring production counters.
        # This receipt survives a failed forward or head projection; a process
        # interruption leaves a running receipt with its last admitted attempt.
        result.update(physical_work={key: value - work.get(key, 0)
            for key, value in trainer.work.items()}, wall_s=time.perf_counter() - started)
        try:
            write_json(output_path, result)
        except BaseException as receipt_error:
            if failure is None:
                raise
            failure.add_note('Witness failure receipt also failed: ' + repr(receipt_error))
        finally:
            for module, mode in modes:
                module.training = mode
            trainer.work, model.costs = work, costs
            trainer._anchor_observations[:] = anchors_seen
    try:
        if before != (trainer.global_step, trainer.presentations, len(trainer.events)):
            raise ValueError('Diagnostic changed training counters')
        if origin_path is not None:
            import json
            origin = json.loads(Path(origin_path).read_text())
            if (origin['global_step'] != 0 or origin['witness_identity'] != result['witness_identity']
                    or len(origin['rows']) != len(rows)):
                raise ValueError('Endpoint diagnostic has a different witness or trained origin')
            comparisons = []
            for first, current in zip(origin['rows'], rows):
                if first['record_id'] != current['record_id'] or first['anchors'] != current['anchors']:
                    raise ValueError('Endpoint witness response or anchors changed')
                if first['blocks'] is not None:
                    comparisons.append(compare_blocks(first['blocks'], current['blocks']))
            result['comparison'] = {key: sum(row[key] for row in comparisons)
                for key in comparisons[0]} if comparisons else {}
        result.update(status='passed', rng_preserved=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error), rng_preserved=True)
        raise
    finally:
        write_json(output_path, result)
    return result
