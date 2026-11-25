"""Private, fixed training witnesses for the warm-parent continuation.

These observations never enter training or selection. Each retained response is
read once at the parent and once at each endpoint, with at most eight native
blocks and one native head chunk. Origin roles, rather than endpoint successes,
define every matched repair, follower and fixed-K assessment occurrence.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.ok_coding.storage import identity
from experiments.repair_conditional_kl_pilot.contract import digest
from online_feedback.ranking import fixed_anchors
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.storage import write_json
from .math import current_roles
from .native import rng_digest

SCHEMA = 'frontier_fixed_training_witness_v1'
SETS = ('new_buffer', 'parent_training')
ENDPOINTS = ('parent', 'CONTROL', 'FRONTIER_P')
ROLE_KEYS = ('labels', 'valid', 'prefix', 'repair', 'L', 'e', 'H', 'K', 'm')
LOWER_BETTER = {
    'damaged_prefix_positions',
    'damaged_prefix_blocks',
    'fixed_K_shortfall',
    'assessment',
}
METRICS = (
    'raw_repair',
    'useful_repair',
    'damaged_prefix_positions',
    'damaged_prefix_blocks',
    'surviving_followers',
    'reachable_followers',
    'matching_prefix_length',
    'fixed_K_shortfall',
    'assessment',
)


def _counters(trainer):
    return (trainer.global_step, trainer.presentations, len(trainer.events))


def _boundary(trainer, endpoint):
    expected = (128, 256, 1) if endpoint == 'parent' else (256, 512, 2)
    if endpoint not in ENDPOINTS or _counters(trainer) != expected:
        raise ValueError(
            'Witness requires the exact parent or committed second-event endpoint'
        )
    if trainer.core.accumulation_remainder:
        raise ValueError('Witness cannot interrupt an accumulation window')
    if trainer.arm != 'PAIR' or (endpoint != 'parent' and trainer.fork_arm != endpoint):
        raise ValueError('Witness endpoint differs from the restored PAIR fork arm')
    if any(p.grad is not None for p in trainer.bundle.model.parameters()):
        raise ValueError('Witness requires a clean gradient boundary')


def freeze_sets(trainer, new_feature_paths, parent_feature_paths, root):
    """Freeze two ordered, disjoint eight-response sets without advancing RNG.

    The original native sampler receives a diagnostic width of at most eight.
    It uses the same valid-candidate random ranking and sorted output; no extra
    sampled training anchors, model forwards, tensor copies or optimizer steps
    are requested. The caller supplies authenticated chronological packet lists.
    """
    from specforge.training.strategies.base import _cpu_max_valid_anchors

    _boundary(trainer, 'parent')
    root = Path(root)
    path = root / 'BINDINGS.json'
    if path.exists() or len(new_feature_paths) < 8 or len(parent_feature_paths) < 8:
        raise ValueError(
            'Need a new witness binding and eight ordered responses in each set'
        )
    model = trainer.bundle.model
    device = next(model.draft_model.parameters()).device
    before = rng_digest(capture_rng())
    rows, identifiers, packet_paths = {}, set(), set()
    with preserve_rng(), torch.no_grad():
        for name, paths in zip(SETS, (new_feature_paths, parent_feature_paths)):
            rows[name] = []
            for ordinal, value in enumerate(paths[:8]):
                packet = Path(value).resolve()
                raw = torch.load(packet, map_location='cpu', weights_only=True)
                record_id = str(raw['record_id'])
                if record_id in identifiers or str(packet) in packet_paths:
                    raise ValueError(
                        'Witness sets must contain disjoint response identities and paths'
                    )
                identifiers.add(record_id)
                packet_paths.add(str(packet))
                batch = trainer._batch(raw)
                mask = batch.tensors['loss_mask']
                available = _cpu_max_valid_anchors(mask)
                sampling_before = rng_digest(capture_rng())
                if available:
                    anchors, keep = trainer._original_sampler(
                        batch.tensors['input_ids'].shape[-1],
                        mask.to(device),
                        device,
                        max_valid_anchors=min(8, available),
                    )
                    if anchors.shape[0] != 1 or anchors.shape != keep.shape:
                        raise ValueError(
                            'Witness requires the native single-response sampler'
                        )
                    chosen = anchors[0, keep[0]].detach().cpu().tolist()
                    if len(chosen) > 8 or len(chosen) != len(set(chosen)):
                        raise ValueError(
                            'Native witness selection exceeded its distinct-anchor cap'
                        )
                else:
                    chosen = []
                rows[name].append(
                    dict(
                        ordinal=ordinal,
                        record_id=record_id,
                        feature_path=str(packet),
                        feature_identity=identity(packet),
                        anchors=chosen,
                        sampling_rng_before=sampling_before,
                        sampling_rng_after=rng_digest(capture_rng()),
                    )
                )
        result = dict(
            schema=SCHEMA,
            status='passed',
            sets=rows,
            maximum_anchors_per_response=8,
            source='first eight ordered responses in each authenticated training list',
            selection='original native sampler with max_valid_anchors=min(8, available)',
            origin_counters=list(_counters(trainer)),
            live_rng_before=before,
            private_sampling_rng_after=rng_digest(capture_rng()),
            diagnostic_forwards=0,
            optimizer_steps=0,
            diagnostic_only=True,
        )
    result['live_rng_after'] = rng_digest(capture_rng())
    if result['live_rng_after'] != before:
        raise ValueError('Freezing witnesses changed the production RNG')
    write_json(path, result)
    return result


@torch.no_grad()
def block_readouts(logits, labels, weights, eos_token_ids=(), *, origin=None):
    """Read native BF16 logits once, retaining frozen parent support and roles."""
    if (
        logits.dtype != torch.bfloat16
        or logits.shape[:-1] != labels.shape
        or labels.ndim != 3
        or labels.shape[0] != 1
        or labels.shape[-1] != 16
    ):
        raise ValueError(
            'Witness requires native BF16 single-response sixteen-slot logits'
        )
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite native witness logits')
    predictions = logits.argmax(-1)
    roles = current_roles(predictions, labels, weights, eos_token_ids)
    if origin is not None and len(origin) != labels.shape[1]:
        raise ValueError('Endpoint witness lost a frozen block')
    output = []
    for b in range(labels.shape[1]):
        current = {key: roles[key][0, b].cpu().tolist() for key in ROLE_KEYS}
        first = current if origin is None else origin[b]['origin']
        if first['labels'] != current['labels'] or first['valid'] != current['valid']:
            raise ValueError('Endpoint canonical labels or native validity changed')
        q = predictions[0, b].cpu().tolist()
        positions = []
        for j in range(1, first['K'] + 1):
            if not first['valid'][j]:
                raise ValueError(
                    'Frozen reachable endpoint includes an invalid position'
                )
            native = logits[0, b, j]
            y = first['labels'][j]
            wrong = native.clone()
            wrong[y] = -torch.inf
            rival = int(wrong.argmax())
            old_rival = (
                rival if origin is None else origin[b]['positions'][j - 1]['rival']
            )
            gathered = native[
                torch.tensor([y, rival, old_rival], device=native.device)
            ].float()
            margin, old_margin = float(gathered[0] - gathered[1]), float(
                gathered[0] - gathered[2]
            )
            positions.append(
                dict(
                    position=j,
                    canonical=y,
                    rival=rival,
                    origin_rival=old_rival,
                    prediction=q[j],
                    correct=q[j] == y,
                    margin=margin,
                    origin_rival_margin=old_margin,
                    margin_tie=margin == 0,
                    wrong_rival_tie=int((wrong == wrong[rival]).sum()) > 1,
                    rival_changed=rival != old_rival,
                    origin_rival_addressed=old_margin > 0
                    or (old_margin == 0 and y < old_rival),
                )
            )
        # FP32 margin / softplus / reduction, exactly the declared assessment.
        margins = torch.tensor(
            [p['margin'] for p in positions], device=logits.device, dtype=torch.float32
        )
        assessment_weights = torch.arange(
            first['K'], 0, -1, device=logits.device, dtype=torch.float32
        )
        value = float((assessment_weights * F.softplus(-margins) / math.log(2.0)).sum())
        damage = sum(
            flag and q[j] != first['labels'][j]
            for j, flag in enumerate(first['prefix'])
        )
        error = first['e']
        repaired = error >= 1 and q[error] == first['labels'][error]
        followers = list(range(error + 1, first['K'] + 1)) if error >= 1 else []
        metrics = dict(
            raw_repair=int(repaired),
            useful_repair=int(repaired and not damage),
            damaged_prefix_positions=damage,
            damaged_prefix_blocks=int(damage > 0),
            surviving_followers=sum(q[j] == first['labels'][j] for j in followers),
            reachable_followers=sum(current['L'] >= j for j in followers),
            matching_prefix_length=current['L'],
            fixed_K_shortfall=max(0, first['K'] - current['L']),
            assessment=value,
        )
        output.append(
            dict(
                origin=first,
                current=current,
                proposals=q,
                positions=positions,
                metrics=metrics,
            )
        )
    return output


def evaluate_sets(trainer, root, endpoint):
    """Read both frozen sets at one committed boundary; no backward or teacher."""
    _boundary(trainer, endpoint)
    root = Path(root)
    binding_path, output_path = root / 'BINDINGS.json', root / f'{endpoint}.json'
    if output_path.exists():
        raise FileExistsError('Do not overwrite a fixed witness readout')
    binding = json.loads(binding_path.read_text())
    if (
        binding.get('schema') != SCHEMA
        or binding.get('status') != 'passed'
        or any(len(binding['sets'][name]) != 8 for name in SETS)
    ):
        raise ValueError('Malformed frozen witness sets')
    origin = (
        None if endpoint == 'parent' else json.loads((root / 'parent.json').read_text())
    )
    binding_identity = identity(binding_path)
    if origin is not None and (
        origin.get('status') != 'passed'
        or origin['binding_identity'] != binding_identity
        or origin['endpoint'] != 'parent'
    ):
        raise ValueError(
            'Endpoint needs the passed, identically bound common-parent readout'
        )
    model = trainer.bundle.model
    if model.block_size != 16 or model.objective_chunk_blocks != 8:
        raise ValueError(
            'Witness head must retain the native sixteen-slot/eight-block contract'
        )
    device = next(model.draft_model.parameters()).device
    before, history = _counters(trainer), digest(trainer.events)
    rng_before = rng_digest(capture_rng())
    versions = [(p, p._version) for p in model.parameters()]
    work, costs, observations = (
        deepcopy(trainer.work),
        deepcopy(model.costs),
        deepcopy(trainer._anchor_observations),
    )
    modes = [(module, module.training) for module in model.modules()]
    result = dict(
        schema=SCHEMA,
        status='running',
        endpoint=endpoint,
        counters=list(before),
        binding_identity=binding_identity,
        origin_digest=None if origin is None else digest(origin),
        sets={name: [] for name in SETS},
        diagnostic_only=True,
        draft_forwards_attempted=0,
        draft_forwards_completed=0,
        head_calls_attempted=0,
        head_calls_completed=0,
        teacher_forwards=0,
        backwards=0,
        optimizer_steps=0,
    )
    started = time.perf_counter()
    write_json(output_path, result)
    failure = None
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            for name in SETS:
                for index, item in enumerate(binding['sets'][name]):
                    if identity(item['feature_path']) != item['feature_identity']:
                        raise ValueError('Retained witness feature packet changed')
                    first = None if origin is None else origin['sets'][name][index]
                    if first is not None and (
                        first['record_id'] != item['record_id']
                        or first['anchors'] != item['anchors']
                    ):
                        raise ValueError('Endpoint witness response or anchors changed')
                    blocks = []
                    if item['anchors']:
                        raw = torch.load(
                            item['feature_path'], map_location='cpu', weights_only=True
                        )
                        if str(raw['record_id']) != item['record_id']:
                            raise ValueError('Witness packet response identity changed')
                        batch = trainer._batch(raw)
                        tensors = {
                            key: batch.tensors[key].to(device)
                            for key in ('input_ids', 'hidden_states', 'loss_mask')
                        }
                        anchors = torch.tensor(
                            [item['anchors']], device=device, dtype=torch.long
                        )
                        keep = torch.ones_like(anchors, dtype=torch.bool)
                        result['draft_forwards_attempted'] += 1
                        write_json(output_path, result)
                        with fixed_anchors(model, anchors, keep):
                            actual, actual_keep, hidden = model._forward_draft_blocks(
                                **tensors
                            )
                        result['draft_forwards_completed'] += 1
                        if not torch.equal(actual, anchors) or not torch.equal(
                            actual_keep, keep
                        ):
                            raise ValueError(
                                'Native witness anchors changed during forward'
                            )
                        labels, weights, _ = model._block_labels_and_weights(
                            tensors['input_ids'], tensors['loss_mask'], anchors, keep
                        )
                        hidden = hidden.reshape(1, len(item['anchors']), 16, -1)
                        result['head_calls_attempted'] += 1
                        write_json(output_path, result)
                        logits = model.lm_head(hidden).to(model.loss_dtype)
                        result['head_calls_completed'] += 1
                        blocks = block_readouts(
                            logits,
                            labels,
                            weights,
                            model.eos_token_ids,
                            origin=None if first is None else first['blocks'],
                        )
                        del logits, hidden, tensors, batch, raw
                    elif first is not None and first['blocks']:
                        raise ValueError('An empty response gained witness blocks')
                    nonempty = sum(b['origin']['m'] > 0 for b in blocks)
                    result['sets'][name].append(
                        dict(
                            record_id=item['record_id'],
                            anchors=item['anchors'],
                            blocks=blocks,
                            nonempty_blocks=nonempty,
                            response_assessment=(
                                sum(b['metrics']['assessment'] for b in blocks)
                                / nonempty
                                if nonempty
                                else 0.0
                            ),
                        )
                    )
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        if (
            _counters(trainer) != before
            or digest(trainer.events) != history
            or any(p._version != v for p, v in versions)
        ):
            raise ValueError('Witness changed training parameters, history or counters')
        _boundary(trainer, endpoint)
        if rng_digest(capture_rng()) != rng_before:
            raise ValueError('Witness changed the production RNG')
        result.update(
            status='passed', rng_preserved=True, parameters_and_counters_unchanged=True
        )
    except BaseException as error:
        failure = error
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result.update(
            wall_s=time.perf_counter() - started,
            physical_work={
                key: value - work.get(key, 0) for key, value in trainer.work.items()
            },
        )
        for module, mode in modes:
            module.training = mode
        trainer.work, model.costs = work, costs
        trainer._anchor_observations[:] = observations
        try:
            write_json(output_path, result)
        except BaseException as receipt_error:
            if failure is None:
                raise
            failure.add_note('Witness receipt failed: ' + repr(receipt_error))
    return result


def _comparison(first, second, *, label):
    """Per-block paired changes; no independence assumption or hypothesis test."""
    result = {}
    for metric in METRICS:
        changes = [
            b['metrics'][metric] - a['metrics'][metric] for a, b in zip(first, second)
        ]
        sign = -1 if metric in LOWER_BETTER else 1
        result[metric] = dict(
            comparison=label,
            first_sum=sum(b['metrics'][metric] for b in first),
            second_sum=sum(b['metrics'][metric] for b in second),
            delta=sum(changes),
            gains=sum(sign * d > 0 for d in changes),
            losses=sum(sign * d < 0 for d in changes),
            ties=sum(d == 0 for d in changes),
            lower_is_better=sign < 0,
        )
    return result


def summarize_sets(root):
    """Match all three private readouts; publish only deliberately selected scalars."""
    root = Path(root)
    receipts = {
        endpoint: json.loads((root / f'{endpoint}.json').read_text())
        for endpoint in ENDPOINTS
    }
    parent = receipts['parent']
    for endpoint, receipt in receipts.items():
        if (
            receipt.get('schema') != SCHEMA
            or receipt.get('status') != 'passed'
            or receipt['endpoint'] != endpoint
            or receipt['binding_identity'] != parent['binding_identity']
            or (endpoint != 'parent' and receipt['origin_digest'] != digest(parent))
        ):
            raise ValueError(
                'Witness summary requires three authenticated, passed readouts'
            )
    result = dict(
        schema=SCHEMA,
        diagnostic_only=True,
        units='overlapping fixed block occurrences in training responses; not held-out evidence',
        definitions=dict(
            raw_repair='original first error now equals canonical',
            useful_repair='raw repair with every original prefix position still canonical',
            damaged_prefix_positions='original prefix positions now wrong; not the entire lost prefix',
            surviving_followers='original positions e+1..K still canonical, regardless of earlier damage',
            reachable_followers='original positions e+1..K inside the endpoint consecutive correct prefix',
            fixed_K_shortfall='max(0, origin K - endpoint consecutive matching prefix length)',
            assessment='origin-K weighted FP32 softplus margin sum; fresh strongest wrong rivals',
        ),
        sets={},
    )
    for name in SETS:
        flat = {}
        for endpoint, receipt in receipts.items():
            rows = receipt['sets'][name]
            if len(rows) != 8:
                raise ValueError('Witness set lost an ordered response')
            for old, row in zip(parent['sets'][name], rows):
                if (
                    old['record_id'] != row['record_id']
                    or old['anchors'] != row['anchors']
                    or len(old['blocks']) != len(row['blocks'])
                ):
                    raise ValueError(
                        'Witness summary response, anchors or blocks differ'
                    )
                if any(
                    a['origin'] != b['origin']
                    for a, b in zip(old['blocks'], row['blocks'])
                ):
                    raise ValueError('Witness endpoint changed frozen parent roles')
            flat[endpoint] = [block for row in rows for block in row['blocks']]
        original = flat['parent']
        summary = dict(
            responses=8,
            block_occurrences=len(original),
            nonempty_blocks=sum(b['origin']['m'] > 0 for b in original),
            original_errors=sum(b['origin']['e'] >= 1 for b in original),
            original_prefix_positions=sum(b['origin']['L'] for b in original),
            usable_followers=sum(max(0, b['origin']['H'] - 1) for b in original),
            parent_to_CONTROL=_comparison(
                original, flat['CONTROL'], label='CONTROL minus parent'
            ),
            parent_to_FRONTIER_P=_comparison(
                original, flat['FRONTIER_P'], label='FRONTIER_P minus parent'
            ),
            CONTROL_to_FRONTIER_P=_comparison(
                flat['CONTROL'], flat['FRONTIER_P'], label='FRONTIER_P minus CONTROL'
            ),
            endpoints={},
        )
        for endpoint, blocks in flat.items():
            positions = [p for b in blocks for p in b['positions']]
            summary['endpoints'][endpoint] = dict(
                response_mean_assessment=sum(
                    r['response_assessment'] for r in receipts[endpoint]['sets'][name]
                )
                / 8,
                assessed_positions=len(positions),
                margin_mean=(
                    sum(p['margin'] for p in positions) / len(positions)
                    if positions
                    else None
                ),
                origin_rival_margin_mean=(
                    sum(p['origin_rival_margin'] for p in positions) / len(positions)
                    if positions
                    else None
                ),
                margin_ties=sum(p['margin_tie'] for p in positions),
                wrong_rival_ties=sum(p['wrong_rival_tie'] for p in positions),
                rival_changes=sum(p['rival_changed'] for p in positions),
                old_rival_addressed_but_wrong=sum(
                    p['origin_rival_addressed'] and not p['correct'] for p in positions
                ),
            )
        result['sets'][name] = summary
    path = root / 'SUMMARY.json'
    if path.exists():
        raise FileExistsError('Do not overwrite a completed witness summary')
    write_json(path, result)
    return result
