"""Read-only native endpoint replay; no trainer, optimizer or target forward.

The author block construction and fixed-anchor helper are reused directly. Only
compact predictions and native gathered margins survive a vocabulary chunk.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch

from online_feedback.ranking import fixed_anchors
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.full_k_gsm8k.costs import compiler_snapshot, compiler_delta

from .metrics import compact_blocks


CACHE_FIELDS = ('_cached_block_mask', '_cached_seq_len', '_cached_bsz',
                'max_seq_len_cached', 'original_max_seq_len')
ROLE_KEYS = ('labels', 'valid', 'prefix', 'repair', 'L', 'e', 'H', 'K', 'm')
ZERO_WORK = dict(backwards=0, optimizer_steps=0, teacher_forwards=0,
                 feedback_generations=0, serving_generations=0)


def rng_identity():
    state = capture_rng()
    numpy = state['numpy']
    represented = dict(python=state['python'],
        numpy=[numpy[0], numpy[1].tolist(), *numpy[2:]],
        torch=state['torch'].tolist(),
        cuda=[value.cpu().tolist() for value in state['cuda']])
    return hashlib.sha256(json.dumps(represented, sort_keys=True).encode()).hexdigest()


def load_export(draft, path, expected_identity):
    """Validate the complete native mapping and stream the export into the draft."""
    from safetensors import safe_open

    path = Path(path)
    if path.is_dir():
        path = path / 'model.safetensors'
    actual_identity = identity(path)
    if actual_identity != expected_identity:
        raise ValueError('Endpoint export differs from its authenticated binding')
    destination = draft.state_dict()
    mapping = []
    with safe_open(path, framework='pt', device='cpu') as source, torch.no_grad():
        if set(source.keys()) != set(destination):
            raise ValueError('Serving export and native draft tensor names differ')
        for name, target in destination.items():
            value = source.get_tensor(name)
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('Serving export tensor shape or dtype differs: ' + name)
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise ValueError('Endpoint tensor is not finite native BF16: ' + name)
            target.copy_(value)
            if not torch.equal(target.detach().cpu(), value):
                raise ValueError('Native draft did not receive the exact exported tensor: ' + name)
            mapping.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype)))
    return dict(status='passed', export_identity=actual_identity,
                tensors=len(mapping), mapping=mapping, exact_copy=True)


@contextmanager
def preserved_runtime(model):
    """Restore eval flags, small rotary/mask state and RNG after each readout."""
    modules = dict(model.named_modules())
    modes = {name: module.training for name, module in modules.items()}
    buffers = {name: dict(module._buffers) for name, module in modules.items()}
    buffer_values = {name: value.detach().clone() for name, value in model.named_buffers()}
    nonpersistent = {name: set(module._non_persistent_buffers_set)
                     for name, module in modules.items()}
    caches = {name: {key: getattr(module, key) for key in CACHE_FIELDS if hasattr(module, key)}
              for name, module in modules.items()}
    costs = deepcopy(model.costs)
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            yield
    finally:
        with torch.no_grad():
            for name, module in modules.items():
                module._buffers.clear()
                module._buffers.update(buffers[name])
                module._non_persistent_buffers_set = nonpersistent[name]
                module.training = modes[name]
                for key in CACHE_FIELDS:
                    if key in caches[name]:
                        setattr(module, key, caches[name][key])
                    elif hasattr(module, key):
                        delattr(module, key)
            for name, value in buffer_values.items():
                owner, _, key = name.rpartition('.')
                modules[owner]._buffers[key].copy_(value)
        model.costs = costs


def tensor_batch(raw, maximum_length, device):
    """Use the exact retained training packet normalization, without a teacher."""
    from specforge.algorithms.common.hidden_states_data import normalize_offline_sample

    sample = {**raw, 'input_ids': raw['input_ids'][0], 'loss_mask': raw['loss_mask'][0]}
    normalized = normalize_offline_sample(sample, maximum_length)
    return {name: normalized[name].to(device)
            for name in ('input_ids', 'hidden_states', 'loss_mask')}


class ReadOnlyEndpoint:
    """One directly loaded draft and its unchanged frozen target head/embedding."""

    def __init__(self, model, *, endpoint, maximum_length, loading=None,
                 allowed_endpoints=('parent', 'child')):
        if endpoint not in allowed_endpoints:
            raise ValueError("Endpoint label is outside this caller's explicit read-only contract")
        if (model.block_size, model.objective_chunk_blocks, model.loss_dtype) != (16, 8, torch.bfloat16):
            raise ValueError('Readout requires native BF16 block16/head-chunk8 geometry')
        self.model = model
        self.endpoint = endpoint
        self.maximum_length = maximum_length
        self.loading = loading
        self.initial_parameters = [(name, parameter, parameter._version, parameter.requires_grad)
                                   for name, parameter in model.named_parameters()]
        self.initial_modes = {name: module.training for name, module in model.named_modules()}
        self.work = dict(response_reads_attempted=0, response_reads_completed=0,
            draft_forwards_attempted=0, draft_forwards_completed=0,
            head_calls_attempted=0, head_calls_completed=0, blocks_completed=0, **ZERO_WORK)

    @classmethod
    def load(cls, config_path, export_path, expected_export_identity, *, endpoint, device='cuda:0',
             allowed_endpoints=('parent', 'child')):
        """Assemble author inference geometry only; never construct TrainerCore."""
        from specforge.application import resolve_run
        from specforge.training.assembly import build_model_bundle
        from online_feedback.objective import TargetObjectiveModel
        from online_feedback.recipe import load_event_config
        from online_feedback.training import ranking_eos_token_ids
        from accelerate.utils import set_seed

        started = time.perf_counter()
        device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.init()
            torch.cuda.set_device(device)
        device_initialization_s = time.perf_counter() - started
        compiler_before = compiler_snapshot()
        before_rng = rng_identity()
        config, recipe = load_event_config(config_path)
        if (config.model.torch_dtype != 'bfloat16' or recipe.kl_precision != 'bfloat16'
                or config.training.attention_backend != 'flex_attention'
                or config.training.objective_chunk_blocks != 8):
            raise ValueError('Diagnostic model configuration differs from the qualified native path')
        with preserve_rng(), torch.no_grad():
            set_seed(config.training.seed)
            resolved = resolve_run(config)
            bundle = build_model_bundle(config, algorithm=resolved.algorithm)
            model = TargetObjectiveModel(bundle.model, 'osd_first_error',
                loss_precision=recipe.kl_precision,
                eos_token_ids=ranking_eos_token_ids(config.model.target_model_path))
            if next(model.draft_model.parameters()).device != device:
                raise ValueError('Native model construction selected a different device')
            frozen = tuple(model.lm_head.parameters()) + tuple(model.embed_tokens.parameters())
            if any(parameter.requires_grad for parameter in frozen):
                raise ValueError('Target vocabulary head and embedding must remain frozen')
            frozen_versions = [parameter._version for parameter in frozen]
            mapping = load_export(model.draft_model, export_path, expected_export_identity)
            if frozen_versions != [parameter._version for parameter in frozen]:
                raise ValueError('Loading the draft changed its frozen target components')
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError('Read-only construction unexpectedly created gradients')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        loading = dict(status='passed', wall_s=time.perf_counter() - started,
            device_initialization_s_nested=device_initialization_s,
            construction_seed=config.training.seed,
            rng_scope='Diagnostic caller RNG preserved; no historical training RNG claimed or loaded',
            mapping=mapping, frozen_components_unchanged=True,
            compiler_timers=compiler_delta(compiler_before, compiler_snapshot()),
            compiler_timer_scope='Passive native timers, nested inside model loading wall time',
            rng_before=before_rng, rng_after=rng_identity(),
            startup_forwards=0, startup_backwards=0, optimizer_created=False, **ZERO_WORK)
        if loading['rng_before'] != loading['rng_after']:
            raise ValueError('Read-only model loading changed the caller RNG')
        return cls(model, endpoint=endpoint, maximum_length=config.data.max_length, loading=loading,
                   allowed_endpoints=allowed_endpoints)

    def read_response(self, binding, receipt_path):
        """Forward one sealed geometry and durably account for every operation."""
        path = Path(receipt_path)
        if path.exists():
            raise FileExistsError('Preserve an existing native response receipt')
        model = self.model
        device = next(model.draft_model.parameters()).device
        chosen = binding['anchors']
        if chosen != sorted(set(chosen)) or len(chosen) > 64:
            raise ValueError('Expected at most64 distinct ascending sealed anchors')
        versions = [(p, p._version, p.grad) for p in model.parameters()]
        if any(gradient is not None for _, _, gradient in versions):
            raise ValueError('A read-only endpoint must not carry training gradients')
        compiler_before = compiler_snapshot()
        before_rng = rng_identity()
        work = dict(response_reads_attempted=1, response_reads_completed=0,
            draft_forwards_attempted=0, draft_forwards_completed=0,
            head_calls_attempted=0, head_calls_completed=0, blocks_completed=0, **ZERO_WORK)
        receipt = dict(schema='continuation_value_native_readout_v1', status='running',
            endpoint=self.endpoint, record_id=str(binding['record_id']),
            feature_identity=binding['feature_identity'],
            anchors=chosen, blocks=[], work=work, rng_before=before_rng,
            geometry=dict(block_size=16, projection_chunk_blocks=8, anchor_count=len(chosen)),
            projection_shapes=[],
            evaluation_mode='eval/no_grad', parameter_state_check='tensor versions and gradient identity')
        started = time.perf_counter()
        write_json(path, receipt)
        try:
            with preserved_runtime(model):
                if chosen:
                    raw = torch.load(binding['feature_path'], map_location='cpu', weights_only=True)
                    if str(raw['record_id']) != receipt['record_id']:
                        raise ValueError('Feature packet and sealed response identity differ')
                    tensors = tensor_batch(raw, self.maximum_length, device)
                    receipt['input_shape'] = list(tensors['input_ids'].shape)
                    receipt['target_features_shape'] = list(tensors['hidden_states'].shape)
                    receipt['target_features_dtype'] = str(tensors['hidden_states'].dtype)
                    anchors = torch.tensor([chosen], dtype=torch.long, device=device)
                    keep = torch.ones_like(anchors, dtype=torch.bool)
                    if chosen[-1] >= tensors['input_ids'].shape[-1] or chosen[0] < 0:
                        raise ValueError('Sealed anchors are outside the retained normalized sequence')
                    work['draft_forwards_attempted'] += 1
                    write_json(path, receipt)
                    with fixed_anchors(model, anchors, keep):
                        actual, actual_keep, hidden = model._forward_draft_blocks(**tensors)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    work['draft_forwards_completed'] += 1
                    if not torch.equal(actual, anchors) or not torch.equal(actual_keep, keep):
                        raise ValueError('Native forward changed sealed anchors or keep mask')
                    labels, weights, _ = model._block_labels_and_weights(
                        tensors['input_ids'], tensors['loss_mask'], anchors, keep)
                    hidden = hidden.reshape(1, len(chosen), 16, -1)
                    for start in range(0, len(chosen), 8):
                        part = slice(start, start + 8)
                        receipt['projection_shapes'].append(list(hidden[:, part].shape))
                        work['head_calls_attempted'] += 1
                        write_json(path, receipt)
                        logits = model.lm_head(hidden[:, part]).to(model.loss_dtype)
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                        work['head_calls_completed'] += 1
                        blocks = compact_blocks(logits, labels[:, part], weights[:, part], model.eos_token_ids)
                        work['blocks_completed'] += len(blocks)
                        receipt['blocks'].extend(blocks)
                        del logits
                    del hidden, tensors, raw
            if any(p._version != version or p.grad is not gradient for p, version, gradient in versions):
                raise ValueError('Forward-only readout changed parameters or gradients')
            receipt['rng_after'] = rng_identity()
            if receipt['rng_after'] != before_rng:
                raise ValueError('Forward-only readout changed RNG')
            work['response_reads_completed'] = 1
            receipt.update(status='passed', parameters_unchanged=True,
                gradients_unchanged=True, runtime_restored=True, rng_preserved=True)
            return receipt
        except BaseException as error:
            receipt.update(status='failed', error=repr(error), rng_after=rng_identity())
            raise
        finally:
            receipt['wall_s'] = time.perf_counter() - started
            receipt['compiler_timers'] = compiler_delta(compiler_before, compiler_snapshot())
            receipt['compiler_timer_scope'] = 'Passive native timers, nested in readout wall time; no extra compilation warmups'
            for key, value in work.items():
                self.work[key] += value
            write_json(path, receipt)

    def close(self):
        """Finish the endpoint state receipt, then release its read-only model."""
        versions_unchanged = all(parameter._version == version and
            parameter.requires_grad == requires_grad and parameter.grad is None
            for _, parameter, version, requires_grad in self.initial_parameters)
        modes_unchanged = self.initial_modes == {
            name: module.training for name, module in self.model.named_modules()}
        represented = [dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                            version=parameter._version, requires_grad=parameter.requires_grad)
                       for name, parameter, _, _ in self.initial_parameters]
        device = next(self.model.draft_model.parameters()).device
        result = dict(status='passed' if versions_unchanged and modes_unchanged else 'failed',
            endpoint=self.endpoint, parameters_unchanged=versions_unchanged,
            modes_unchanged=modes_unchanged, optimizer_created=False,
            state_evidence='Exact export copies at load; parameter versions/flags and absence of gradients at close',
            parameter_structure_digest=hashlib.sha256(json.dumps(represented, sort_keys=True).encode()).hexdigest(),
            work=dict(self.work), loading=self.loading,
            process_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            process_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
        self.initial_parameters.clear()
        self.model = None
        if result['status'] != 'passed':
            raise ValueError('Read-only endpoint state changed before close: ' + json.dumps(result))
        return result


def compare_legacy(actual, retained):
    """Compare old geometry exactly; the legacy record retains margins only to K."""
    discrepancies = []
    if actual['record_id'] != str(retained['record_id']) or actual['anchors'] != retained['anchors']:
        discrepancies.append('response identity or anchors')
    if len(actual['blocks']) != len(retained['blocks']):
        discrepancies.append('block count')
    for index, (new, old) in enumerate(zip(actual['blocks'], retained['blocks'])):
        expected_roles = old['current']
        for key in ROLE_KEYS:
            if new['roles'][key] != expected_roles[key]:
                discrepancies.append(f'block{index}.roles.{key}')
        if new['proposals'] != old['proposals']:
            discrepancies.append(f'block{index}.proposals')
        positions = {item['position']: item for item in new['positions']}
        for first in old['positions']:
            second = positions[first['position']]
            for key in ('canonical', 'rival', 'prediction', 'margin', 'margin_tie', 'wrong_rival_tie'):
                if second[key] != first[key]:
                    discrepancies.append(f'block{index}.position{first["position"]}.{key}')
    return dict(status='passed' if not discrepancies else 'failed',
        discrepancies=discrepancies, exact_old_geometry=True,
        margin_scope='All positions retained by the original parent-K witness')


def compare_repeated(first, second):
    fields = ('record_id', 'anchors', 'geometry', 'blocks')
    discrepancies = [key for key in fields if first[key] != second[key]]
    return dict(status='passed' if not discrepancies else 'failed', discrepancies=discrepancies)


def qualify_endpoint(adapter, legacy_bindings, legacy_readouts, primary_binding, output_root):
    """Two old geometries plus two identical primary reads, all inside budget."""
    if len(legacy_bindings) != 2 or len(legacy_readouts) != 2:
        raise ValueError('Qualification requires exactly two retained witness responses')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'QUALIFICATION.json').exists():
        raise FileExistsError('Preserve previous qualification evidence in its original directory')
    result = dict(status='running', endpoint=adapter.endpoint,
        loading=adapter.loading, comparisons=[], response_reads_completed=0,
        planned_response_forwards=4, **ZERO_WORK)
    write_json(root / 'QUALIFICATION.json', result)
    try:
        for index, (binding, expected) in enumerate(zip(legacy_bindings, legacy_readouts, strict=True)):
            actual = adapter.read_response(binding, root / f'legacy_{index}.json')
            result['response_reads_completed'] += 1
            comparison = compare_legacy(actual, expected)
            result['comparisons'].append(comparison)
            write_json(root / 'QUALIFICATION.json', result)
            if comparison['status'] != 'passed':
                raise ValueError('Legacy native witness parity failed; see retained discrepancies')
        first = adapter.read_response(primary_binding, root / 'primary_0.json')
        result['response_reads_completed'] += 1
        second = adapter.read_response(primary_binding, root / 'primary_1.json')
        result['response_reads_completed'] += 1
        comparison = compare_repeated(first, second)
        result['comparisons'].append(comparison)
        if comparison['status'] != 'passed':
            raise ValueError('Primary diagnostic geometry is not deterministic')
        result['status'] = 'passed'
        return result
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(root / 'QUALIFICATION.json', result)
