"""Small matched native-block diagnostic, isolated from training RNG/state."""

from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from online_feedback.first_error import first_error_mask, valid_prefix_mask
from online_feedback.ranking import fixed_anchors, matching_prefix_mask
from online_feedback.rng import preserve_rng
from online_feedback.storage import save_torch, write_json
from .contract import digest

SCHEMA = 'repair_pair_fixed_witness_v1'
MARGIN_SCHEMA = 'repair_pair_origin_margin_v1'


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
                    batch.tensors['input_ids'].shape[-1],
                    mask.to(device),
                    device,
                    max_valid_anchors=available,
                )
                if anchors.shape[0] != 1:
                    raise ValueError(
                        'Witness requires the native single-response block sampler'
                    )
                chosen = anchors[0, keep[0]][:8].detach().cpu()
            else:
                chosen = torch.empty(0, dtype=torch.long)
            rows.append(
                dict(
                    ordinal=ordinal,
                    record_id=str(raw['record_id']),
                    feature_path=str(Path(path).absolute()),
                    feature_identity=identity(path),
                    anchors=chosen.tolist(),
                    keep=[True] * len(chosen),
                )
            )
        receipt = dict(
            schema=SCHEMA,
            frozen_at='BASE released origin before any production update',
            feature_source='first8_BASE_event1_own_feedback',
            maximum_anchors=8,
            rows=rows,
            optimizer_steps=0,
            diagnostic_forwards=0,
            rng_preserved=True,
        )
        save_torch(destination, receipt)
    return receipt


def block_summary(proposals, labels, weights, eos_token_ids=()):
    """Keep exact predictions/validity and the native consecutive-match count."""
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    prefix = matching_prefix_mask(proposals, labels, weights, eos_token_ids)
    selected = first_error_mask(proposals, labels, weights, eos_token_ids)
    error = selected & valid & (proposals != labels)
    return dict(
        proposals=proposals.cpu().tolist(),
        labels=labels.cpu().tolist(),
        valid=valid.cpu().tolist(),
        prefix=prefix.cpu().tolist(),
        first_error=error.cpu().tolist(),
        prefix_length=prefix.sum(-1).cpu().tolist(),
    )


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
    return dict(
        original_boundaries=int(old_error.sum()),
        repaired_boundaries=int(repaired.sum()),
        original_prefix_positions=int(old_prefix.sum()),
        damaged_prefix_positions=int(damage.sum()),
        damaged_prefix_blocks=int(damage.any(-1).sum()),
        repaired_without_prefix_damage=int((repaired.any(-1) & ~damage.any(-1)).sum()),
        net_consecutive_match_length_change=int((new_lengths - old_lengths).sum()),
        improved_blocks=int((new_lengths > old_lengths).sum()),
        regressed_blocks=int((new_lengths < old_lengths).sum()),
    )


@torch.no_grad()
def origin_margin_observations(logits, labels, weights, eos_token_ids=(), *, origin=None):
    """Read fixed origin errors from an existing native head chunk.

    The full-vocabulary wrong-token argmax uses a private BF16 copy of only
    observed rows. Gather BF16 values first, then subtract in FP32. Endpoint
    role changes never select or remove an origin observation.
    """
    if (
        logits.dtype != torch.bfloat16
        or logits.shape[:-1] != labels.shape
        or labels.shape[0] != 1
    ):
        raise ValueError('Witness margins require native BF16 single-response block logits')
    predictions = logits.detach().argmax(-1)
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    if origin is None:
        selected = first_error_mask(predictions, labels, weights, eos_token_ids)
        errors = selected & (predictions != labels)
        origin = []
        for block in range(labels.shape[1]):
            positions = errors[0, block].nonzero().flatten()
            if not positions.numel():
                origin.append(None)
                continue
            position = int(positions[0])
            origin.append(
                dict(
                    position=position,
                    canonical_token_id=int(labels[0, block, position]),
                    original_rival_token_id=int(predictions[0, block, position]),
                )
            )
    if len(origin) != labels.shape[1]:
        raise ValueError('Origin margin observations lost their block binding')
    observations = []
    for block, frozen in enumerate(origin):
        if frozen is None:
            observations.append(None)
            continue
        position = frozen['position']
        y, rival = frozen['canonical_token_id'], frozen['original_rival_token_id']
        if (
            not 0 <= position < labels.shape[-1]
            or not bool(valid[0, block, position])
            or int(labels[0, block, position]) != y
            or rival == y
            or not 0 <= rival < logits.shape[-1]
        ):
            raise ValueError('Origin repair position, canonical token or rival changed')
        native = logits[0, block, position].detach()
        if not bool(torch.isfinite(native).all()):
            raise ValueError('Nonfinite native witness logits')
        wrong_logits = native.clone()
        wrong_logits[y] = -torch.inf
        wrong = int(wrong_logits.argmax(-1))
        gathered = native[torch.tensor([y, rival, wrong], device=native.device)].float()
        original_margin = float(gathered[0] - gathered[1])
        current_margin = float(gathered[0] - gathered[2])
        predicted = int(predictions[0, block, position])
        observations.append(
            dict(
                position=position,
                canonical_token_id=y,
                original_rival_token_id=rival,
                native_predicted_token_id=predicted,
                current_wrong_token_id=wrong,
                correct=predicted == y,
                M_original=original_margin,
                M_current=current_margin,
                original_margin_positive=original_margin > 0,
                original_margin_tie=original_margin == 0,
                current_margin_tie=current_margin == 0,
                original_rival_addressed=(
                    original_margin > 0 or (original_margin == 0 and y < rival)
                ),
                rival_changed=wrong != rival,
            )
        )
    return observations


def _matched_rows(origin, endpoint):
    if (
        origin.get('margin_schema') != MARGIN_SCHEMA
        or endpoint.get('margin_schema') != MARGIN_SCHEMA
        or origin.get('status') != 'passed'
        or endpoint.get('status') != 'passed'
        or origin['global_step'] != 0
        or origin['witness_identity'] != endpoint['witness_identity']
        or endpoint.get('origin_readout_digest') != digest(origin)
        or len(origin['rows']) != len(endpoint['rows'])
    ):
        raise ValueError(
            'Matched readouts require a passed common origin and identical witness binding'
        )
    for first, current in zip(origin['rows'], endpoint['rows']):
        if (
            first['record_id'] != current['record_id']
            or first['anchors'] != current['anchors']
            or (first['blocks'] is None) != (current['blocks'] is None)
        ):
            raise ValueError('Endpoint witness response or anchors changed')
        if first['blocks'] is not None:
            compare_blocks(first['blocks'], current['blocks'])
            if len(first['margin_observations']) != len(first['anchors']):
                raise ValueError('Origin margin occurrence count differs from frozen anchors')
            if len(current['margin_observations']) != len(first['anchors']):
                raise ValueError('Endpoint margin occurrence count differs from frozen anchors')
        yield first, current


def compare_matched_witnesses(origin, base, pair):
    """Pure occurrence-level BASE/PAIR comparison on the frozen origin support.

    The returned per_block table is private: it retains response/anchor/token
    identities. Summary counts include every block, including origin no-error
    blocks, while margin statistics include only origin repair occurrences.
    """
    if (
        origin.get('arm') != 'BASE'
        or base.get('arm') != 'BASE'
        or pair.get('arm') != 'PAIR'
        or base.get('global_step') != 128
        or pair.get('global_step') != 128
    ):
        raise ValueError(
            'Matched endpoint comparison requires completed BASE1 and PAIR1 readouts'
        )
    base_rows = list(_matched_rows(origin, base))
    pair_rows = list(_matched_rows(origin, pair))
    per_block = []
    for response_index, ((first, left), (_, right)) in enumerate(zip(base_rows, pair_rows)):
        if first['blocks'] is None:
            continue
        for block, anchor in enumerate(first['anchors']):
            original = first['blocks']
            row = dict(
                witness_identity=origin['witness_identity'],
                record_id=first['record_id'],
                response_index=response_index,
                block_index=block,
                anchor=anchor,
                origin_prefix_length=original['prefix_length'][0][block],
                origin_margin=first['margin_observations'][block],
            )
            for arm, current in (('BASE', left), ('PAIR', right)):
                predictions = current['blocks']['proposals'][0][block]
                labels = original['labels'][0][block]
                prefix = original['prefix'][0][block]
                damage = sum(
                    keep and prediction != label
                    for keep, prediction, label in zip(prefix, predictions, labels)
                )
                frozen = row['origin_margin']
                margin = current['margin_observations'][block]
                if frozen is None:
                    if margin is not None:
                        raise ValueError(
                            'No-error origin block acquired an invented repair margin'
                        )
                elif margin is None or any(
                    margin[key] != frozen[key]
                    for key in ('position', 'canonical_token_id', 'original_rival_token_id')
                ):
                    raise ValueError('Endpoint changed its frozen origin repair identity')
                repaired = margin is not None and margin['correct']
                length = current['blocks']['prefix_length'][0][block]
                row[arm] = dict(
                    margin=margin,
                    boundary_repaired=repaired,
                    useful_repair=repaired and damage == 0,
                    damaged_prefix_positions=damage,
                    damaged_prefix_block=damage > 0,
                    prefix_length=length,
                    prefix_length_change=length - row['origin_prefix_length'],
                )
                if margin is not None:
                    row[arm]['M_original_change'] = margin['M_original'] - frozen['M_original']
                    row[arm]['M_current_change'] = margin['M_current'] - frozen['M_current']
            row['PAIR_minus_BASE_prefix_length'] = (
                row['PAIR']['prefix_length'] - row['BASE']['prefix_length']
            )
            if row['origin_margin'] is not None:
                for name in ('M_original', 'M_current'):
                    row['PAIR_minus_BASE_' + name] = (
                        row['PAIR']['margin'][name] - row['BASE']['margin'][name]
                    )
            per_block.append(row)
    repair_rows = [row for row in per_block if row['origin_margin'] is not None]
    summary = dict(
        block_occurrences=len(per_block),
        response_count=len(origin['rows']),
        origin_repair_occurrences=len(repair_rows),
        original_prefix_positions=sum(row['origin_prefix_length'] for row in per_block),
        useful_repair_PAIR_only=sum(
            row['PAIR']['useful_repair'] and not row['BASE']['useful_repair']
            for row in repair_rows
        ),
        useful_repair_BASE_only=sum(
            row['BASE']['useful_repair'] and not row['PAIR']['useful_repair']
            for row in repair_rows
        ),
        useful_repair_both=sum(
            row['BASE']['useful_repair'] and row['PAIR']['useful_repair'] for row in repair_rows
        ),
        useful_repair_neither=sum(
            not row['BASE']['useful_repair'] and not row['PAIR']['useful_repair']
            for row in repair_rows
        ),
        PAIR_minus_BASE_prefix_length=sum(
            row['PAIR_minus_BASE_prefix_length'] for row in per_block
        ),
        PAIR_improved_blocks=sum(row['PAIR_minus_BASE_prefix_length'] > 0 for row in per_block),
        PAIR_regressed_blocks=sum(
            row['PAIR_minus_BASE_prefix_length'] < 0 for row in per_block
        ),
        PAIR_tied_blocks=sum(row['PAIR_minus_BASE_prefix_length'] == 0 for row in per_block),
    )
    for arm in ('BASE', 'PAIR'):
        values = [row[arm] for row in per_block]
        margins = [row[arm]['margin'] for row in repair_rows]
        summary[arm] = dict(
            boundaries_repaired=sum(row['boundary_repaired'] for row in values),
            useful_repairs=sum(row['useful_repair'] for row in values),
            damaged_prefix_positions=sum(row['damaged_prefix_positions'] for row in values),
            damaged_prefix_blocks=sum(row['damaged_prefix_block'] for row in values),
            full_valid_block_prefix_length=sum(row['prefix_length'] for row in values),
            net_prefix_length_change=sum(row['prefix_length_change'] for row in values),
            old_rival_addressed_but_still_wrong=sum(
                row['original_rival_addressed'] and not row['correct'] for row in margins
            ),
            positive_original_margin_but_still_wrong=sum(
                row['original_margin_positive'] and not row['correct'] for row in margins
            ),
            addressed_still_wrong_with_rival_change=sum(
                row['original_rival_addressed'] and not row['correct'] and row['rival_changed']
                for row in margins
            ),
            rival_changed=sum(row['rival_changed'] for row in margins),
            original_margin_ties=sum(row['original_margin_tie'] for row in margins),
            current_margin_ties=sum(row['current_margin_tie'] for row in margins),
            improved_blocks=sum(row['prefix_length_change'] > 0 for row in values),
            regressed_blocks=sum(row['prefix_length_change'] < 0 for row in values),
            tied_blocks=sum(row['prefix_length_change'] == 0 for row in values),
        )
        for name in ('M_original', 'M_current'):
            changes = [row[arm][name + '_change'] for row in repair_rows]
            summary[arm][name + '_change_mean'] = (
                sum(changes) / len(changes) if changes else None
            )
    for name in ('M_original', 'M_current'):
        differences = [row['PAIR_minus_BASE_' + name] for row in repair_rows]
        summary['PAIR_minus_BASE_' + name + '_mean'] = (
            sum(differences) / len(differences) if differences else None
        )
    return dict(
        schema=MARGIN_SCHEMA,
        summary=summary,
        per_block=per_block,
        units='overlapping sampled block occurrences, not independent examples',
        old_rival_addressed_definition='canonical wins native pair ordering, including lowest-token-ID ties',
        correctness_definition='full native vocabulary argmax equals canonical token',
    )


def evaluate_witness(trainer, witness_path, output_path, *, origin_path=None, endpoint=None):
    """One draft forward per nonempty response, no backward or teacher forward."""
    from experiments.ok_coding.storage import identity

    witness = torch.load(witness_path, map_location='cpu', weights_only=False)
    if witness.get('schema') != SCHEMA or len(witness['rows']) != 8:
        raise ValueError('Malformed sealed matched witness')
    if Path(output_path).exists():
        raise FileExistsError('Do not overwrite a witness readout')
    if origin_path is None and (trainer.global_step or trainer.presentations or trainer.events):
        raise ValueError('The common witness origin must precede all production updates')
    if origin_path is not None and (
        trainer.global_step,
        trainer.presentations,
        len(trainer.events),
    ) != (128, 256, 1):
        raise ValueError('Endpoint witness requires the completed one-event checkpoint')
    witness_identity = identity(witness_path)
    origin = None if origin_path is None else json.loads(Path(origin_path).read_text())
    if origin is not None and (
        origin.get('margin_schema') != MARGIN_SCHEMA
        or origin.get('status') != 'passed'
        or origin['global_step'] != 0
        or origin['witness_identity'] != witness_identity
        or len(origin['rows']) != len(witness['rows'])
    ):
        raise ValueError('Endpoint diagnostic requires the passed frozen origin margin readout')
    model = trainer.bundle.model
    device = next(model.draft_model.parameters()).device
    before = (trainer.global_step, trainer.presentations, len(trainer.events))
    work, costs, anchors_seen = (
        deepcopy(trainer.work),
        deepcopy(model.costs),
        deepcopy(trainer._anchor_observations),
    )
    modes = [(module, module.training) for module in model.modules()]
    started = time.perf_counter()
    rows = []
    result = dict(
        schema=SCHEMA,
        margin_schema=MARGIN_SCHEMA,
        status='running',
        arm=trainer.arm,
        endpoint=endpoint,
        global_step=trainer.global_step,
        presentations=trainer.presentations,
        completed_events=len(trainer.events),
        witness_identity=witness_identity,
        rows=rows,
        origin_readout_digest=None if origin is None else digest(origin),
        diagnostic_draft_forwards_attempted=0,
        diagnostic_draft_forwards=0,
        diagnostic_head_calls_attempted=0,
        diagnostic_head_calls=0,
        teacher_forwards=0,
        backward_calls=0,
        optimizer_steps=0,
        automatic_retry=False,
    )
    write_json(output_path, result)
    failure = None
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            for response_index, item in enumerate(witness['rows']):
                original = None if origin is None else origin['rows'][response_index]
                if original is not None and (
                    original['record_id'] != item['record_id']
                    or original['anchors'] != item['anchors']
                ):
                    raise ValueError('Endpoint witness response or anchors changed')
                if identity(item['feature_path']) != item['feature_identity']:
                    raise ValueError('A retained witness feature packet changed')
                if not item['anchors']:
                    rows.append(
                        dict(
                            record_id=item['record_id'],
                            anchors=[],
                            blocks=None,
                            margin_observations=[],
                        )
                    )
                    continue
                raw = torch.load(item['feature_path'], map_location='cpu', weights_only=True)
                batch = trainer._batch(raw)
                tensors = {
                    name: batch.tensors[name].to(device)
                    for name in ('input_ids', 'hidden_states', 'loss_mask')
                }
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
                    tensors['input_ids'], tensors['loss_mask'], anchors, keep
                )
                hidden = hidden.reshape(1, len(item['anchors']), model.block_size, -1)
                predictions, margins = [], []
                for begin in range(0, hidden.shape[1], model.objective_chunk_blocks):
                    result['diagnostic_head_calls_attempted'] += 1
                    write_json(output_path, result)
                    stop = begin + model.objective_chunk_blocks
                    logits = model.lm_head(hidden[:, begin:stop]).to(model.loss_dtype)
                    result['diagnostic_head_calls'] += 1
                    predictions.append(logits.argmax(-1))
                    margins.extend(
                        origin_margin_observations(
                            logits,
                            labels[:, begin:stop],
                            weights[:, begin:stop],
                            model.eos_token_ids,
                            origin=(
                                None
                                if original is None
                                else original['margin_observations'][begin:stop]
                            ),
                        )
                    )
                    del logits
                blocks = block_summary(
                    torch.cat(predictions, dim=1), labels, weights, model.eos_token_ids
                )
                rows.append(
                    dict(
                        record_id=item['record_id'],
                        anchors=item['anchors'],
                        blocks=blocks,
                        margin_observations=margins,
                    )
                )
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
        result.update(
            physical_work={
                key: value - work.get(key, 0) for key, value in trainer.work.items()
            },
            wall_s=time.perf_counter() - started,
        )
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
            comparisons = []
            for first, current in zip(origin['rows'], rows):
                if (
                    first['record_id'] != current['record_id']
                    or first['anchors'] != current['anchors']
                ):
                    raise ValueError('Endpoint witness response or anchors changed')
                if first['blocks'] is not None:
                    comparisons.append(compare_blocks(first['blocks'], current['blocks']))
            result['comparison'] = (
                {key: sum(row[key] for row in comparisons) for key in comparisons[0]}
                if comparisons
                else {}
            )
        result.update(status='passed', rng_preserved=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error), rng_preserved=True)
        raise
    finally:
        write_json(output_path, result)
    return result
