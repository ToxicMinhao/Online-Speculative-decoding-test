"""Request-local correction at the pinned native DFlash pre-vocabulary boundary.

No model, CUDA context, or worker is created on import. The hooks leave target
verification, draft attention, normalization and the vocabulary GEMM unchanged.
They are deliberately specific to the already pinned C1/TP1 native runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def corrected_features(hidden, matrix, output_basis, input_basis, *, known_nonzero=False):
    """h + U C phi(h), in row convention; return exact input for a zero C.

    The zero path is bit-preserving. FP32 computes the small correction and its
    addition; one cast restores the unchanged native vocabulary GEMM dtype.
    """
    import torch
    if not known_nonzero and not bool(torch.count_nonzero(matrix)):
        return hidden
    values = hidden.float()
    projected = values @ input_basis.float()
    phi = projected / projected.norm(dim=-1, keepdim=True).clamp_min(1.0)
    delta = (phi @ matrix.float().T) @ output_basis.float().T
    return (values + delta).to(hidden.dtype)


def _jsonable(value):
    import dataclasses
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


class RequestCorrection:
    """One immutable pre-read bank state, one prompt descriptor, one fusion."""
    def __init__(self, declaration, projections, bank_factory, *, active_bank=None):
        self.declaration = declaration
        self.enabled = declaration['arm'] == 'BANK'
        self.projections = projections
        self.bank_factory = bank_factory
        self.read_result = None
        self.forecast = None
        self.effective = None
        self.prompt_seen = 0
        self.costs = dict(snapshot_restore_s=0., descriptor_s=0., routing_s=0., fusion_s=0.)
        self.bank = None
        if self.enabled:
            require(active_bank is not None, 'Activate pre-read bank state before request timing')
            require(active_bank['state_sha256'] == declaration['state_sha256'], 'Active bank snapshot differs')
            self.bank = active_bank['bank']
            self.state_identity = _light_identity(self.bank)

    def prefill(self, features, *, start, prompt_length, forecast_path):
        """Consume real target-prefill chunks; only the last prompt row routes."""
        if not self.enabled:
            return
        require(self.read_result is None, 'Prompt arrived after read was fixed')
        require(start == self.prompt_seen, 'Prompt chunks are not contiguous')
        require(features.ndim == 2 and features.shape[0] > 0, 'Missing prompt teacher feature')
        self.prompt_seen += int(features.shape[0])
        require(self.prompt_seen <= prompt_length, 'Descriptor contains generated response positions')
        if self.prompt_seen != prompt_length:
            return
        import numpy as np
        import torch
        started = time.perf_counter()
        basis = torch.as_tensor(self.projections['V_descriptor'], dtype=torch.float32, device=features.device)
        descriptor = features[-1].float() @ basis
        norm = descriptor.norm()
        require(bool(torch.isfinite(descriptor).all()) and float(norm) > 0., 'Invalid prompt descriptor')
        descriptor = (descriptor / norm).detach().cpu().double().numpy()
        self.costs['descriptor_s'] = time.perf_counter() - started
        started = time.perf_counter()
        self.read_result = self.bank.read(descriptor)
        self.costs['routing_s'] = time.perf_counter() - started
        started = time.perf_counter()
        effective = np.asarray(self.read_result.effective, dtype=np.float64)
        self.effective = torch.as_tensor(effective, dtype=torch.float32, device=features.device)
        self.zero_correction = not bool(np.count_nonzero(effective))
        self.output_basis = torch.as_tensor(self.projections['U'], dtype=torch.float32, device=features.device)
        self.input_basis = torch.as_tensor(self.projections['V_in'], dtype=torch.float32, device=features.device)
        if features.is_cuda:
            torch.cuda.synchronize(features.device)
        self.costs['fusion_s'] = time.perf_counter() - started
        require(_light_identity(self.bank) == self.state_identity, 'Read mutated bank state')
        self.forecast = {
            'forecast_saved_before_response': True,
            'boundary': 'Final target prompt-prefill result; before first corrected draft proposal',
            'snapshot_sha256': self.declaration['state_sha256'],
            'descriptor_extracted': True, 'routing_recomputed': True, 'fusion_recomputed': True,
            'cost_in_request_s': True,
            'descriptor': descriptor.tolist(), 'coefficients': _jsonable(self.read_result.alpha),
            'alpha': _jsonable(self.read_result.alpha),
            'forecast': _jsonable(self.read_result.forecast), 'effective': effective.tolist(),
            'diagnostics': _jsonable(self.read_result.diagnostics), 'costs': dict(self.costs),
            'prompt_positions': prompt_length,
        }
        path = Path(forecast_path)
        require(not path.exists(), 'Forecast must be saved once before response')
        path.write_text(json.dumps(self.forecast, sort_keys=True, allow_nan=False) + '\n')

    def apply(self, hidden):
        if not self.enabled:
            return hidden
        require(self.read_result is not None and self.forecast is not None,
                'Drafting started before a causal prompt read')
        if self.zero_correction:
            return hidden
        return corrected_features(hidden, self.effective, self.output_basis, self.input_basis, known_nonzero=True)

    def finish(self):
        if not self.enabled:
            return None
        require(self.read_result is not None, 'Complete request has no prompt descriptor/read')
        require(_light_identity(self.bank) == self.state_identity, 'Serving mutated bank/memory')
        return self.forecast


def _light_identity(bank):
    """Read-only arrays plus small mutable content; no Fisher-tensor recopies."""
    matrices = hashlib.sha256(bank.matrices.tobytes()).hexdigest()
    allocated = hashlib.sha256(bank.allocated.tobytes()).hexdigest()
    return (matrices, allocated, tuple(id(x) for x in bank.feedback),
            tuple(id(x) for x in bank.protection), len(bank.write_log))


def install(entry, config, *, projections=None, bank_factory=None):
    """Install before the qualified scheduler starts (and before CUDA graphs)."""
    from sglang.srt.speculative.dflash_worker import DFlashWorker
    from sglang.srt.managers.scheduler import Scheduler
    require(not getattr(DFlashWorker, '_correction_bank_installed', False), 'Bank hooks already installed')
    native_trace = entry.RequestTrace
    pending = Path(config['worker_out']) / 'BANK_PENDING.json'
    cached_projections = {}
    active_bank = {}
    if bank_factory is None:
        from experiments.dflash_correction_bank.core import CorrectionBank
        bank_factory = CorrectionBank.from_snapshot

    def activate(scheduler):
        entry.inherited.strict_idle(scheduler)
        started = time.perf_counter()
        from experiments.dflash_correction_bank.artifacts import load_snapshot
        import numpy as np
        import torch
        declaration = json.loads((Path(config['worker_out']) / 'BANK_ACTIVATE.json').read_text())
        state = declaration['state']
        require(state['state_sha256'] == state['snapshot']['sha256'], 'Activation state hash differs')
        bound = declaration['projections']
        if bound['sha256'] not in cached_projections:
            raw = Path(bound['path']).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == bound['sha256'], 'Fixed projections changed')
            with np.load(bound['path'], allow_pickle=False) as data:
                cached_projections[bound['sha256']] = {
                    k: torch.as_tensor(data[k], dtype=torch.float32, device=torch.cuda.current_device())
                    for k in ('U', 'V_in', 'V_descriptor')}
            torch.cuda.synchronize()
        reused = active_bank.get('state_sha256') == state['state_sha256']
        if not reused:
            bank = bank_factory(load_snapshot(state['snapshot']))
            bank.matrices.setflags(write=False)
            bank.allocated.setflags(write=False)
            active_bank.clear()
            active_bank.update(state_sha256=state['state_sha256'], bank=bank)
        active_bank['projections_sha256'] = bound['sha256']
        receipt = dict(status='passed', state_sha256=state['state_sha256'],
                       reused=reused, loading_s=time.perf_counter()-started)
        (Path(config['worker_out']) / 'BANK_ACTIVE.json').write_text(json.dumps(receipt, allow_nan=False)+'\n')
        return receipt

    Scheduler.correction_bank_activate = activate

    class BankTrace(native_trace):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            declaration = json.loads(pending.read_text())
            require(declaration['input_ids_sha256'] == digest(self.input_ids), 'Bank trace input differs')
            active_projections = projections
            if declaration['arm'] == 'BANK' and active_projections is None:
                bound = declaration['projections']
                require(bound['sha256'] == active_bank.get('projections_sha256'), 'Projection activation missing')
                active_projections = cached_projections[bound['sha256']]
            self.bank_correction = RequestCorrection(declaration, active_projections, bank_factory,
                                                     active_bank=active_bank)
            self.bank_forecast_path = declaration['forecast_path']

        def finish(self, output_ids, meta):
            value = super().finish(output_ids, meta)
            value['correction_bank_read'] = self.bank_correction.finish()
            return value

    entry.RequestTrace = BankTrace
    original_forward = DFlashWorker.forward_batch_generation
    original_head = DFlashWorker._greedy_sample_from_vocab_parallel_head

    def forward(worker, batch, **kwargs):
        trace = entry._active_trace
        correction = getattr(trace, 'bank_correction', None)
        is_prompt = (batch.forward_mode.is_extend() or getattr(batch, 'is_extend_in_batch', False))
        capture_prompt = correction is not None and correction.enabled and is_prompt
        if capture_prompt:
            require(batch.batch_size() == 1 and int(worker.tp_size) == 1 and not worker.use_compact_draft_cache,
                    'Bank requires qualified C1 full-context runtime')
            # Pinned ScheduleBatch.get_model_worker_batch() passes these lists
            # as extend_seq_lens/extend_prefix_lens. Snapshot scalar geometry
            # before native execution: _append_target_hidden_to_draft_kv()
            # consumes and zeros draft_input.ctx_lens before forward returns.
            # The first prefill may have spec_info=None; never read it here.
            lengths = getattr(batch, 'extend_lens', None)
            prefixes = getattr(batch, 'prefix_lens', None)
            require(isinstance(lengths, (tuple, list)) and isinstance(prefixes, (tuple, list)) and
                    len(lengths) == len(prefixes) == 1,
                    'Missing qualified ScheduleBatch prefill geometry')
            chunk_length, start = lengths[0], prefixes[0]
            require(type(chunk_length) is int and chunk_length > 0 and type(start) is int and start >= 0,
                    'Invalid native prefill length or prefix offset')
        result = original_forward(worker, batch, **kwargs)
        if capture_prompt:
            features = getattr(result.logits_output, 'teacher_hidden_states', None)
            require(features is not None, 'Pinned target final-normalized prehead features unavailable')
            require(features.ndim == 2 and chunk_length == features.shape[0],
                    'Target prompt features differ from actual native prefill length')
            correction.prefill(features, start=start, prompt_length=len(trace.input_ids),
                               forecast_path=trace.bank_forecast_path)
        return result

    def head(worker, *, hidden_states, lm_head, chunk_size=256):
        trace = entry._active_trace
        correction = getattr(trace, 'bank_correction', None)
        if correction is not None:
            hidden_states = correction.apply(hidden_states)
        return original_head(worker, hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size)

    DFlashWorker.forward_batch_generation = forward
    DFlashWorker._greedy_sample_from_vocab_parallel_head = head
    DFlashWorker._correction_bank_installed = True
    return {
        'draft_hook': 'DFlashWorker._greedy_sample_from_vocab_parallel_head(hidden_states), after DFlashModel.norm',
        'descriptor_hook': 'DFlashWorker.forward_batch_generation extend result.logits_output.teacher_hidden_states[-1]',
        'descriptor_scope': 'Last prompt token, final frozen target norm; complete chunked prefill only',
        'prefill_geometry_hook': 'ScheduleBatch.extend_lens/prefix_lens copied before native forward; consumed draft_input.ctx_lens is never read',
        'extra_target_calls': 0, 'compiled_vocabulary_projections_per_draft_position': 1,
        'zero_path': 'Exact original hidden tensor; no addition or cast',
        'nonzero_precision': 'FP32 small correction/addition then cast to original BF16 vocabulary input',
    }
