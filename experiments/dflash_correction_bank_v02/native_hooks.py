"""Frozen DFlash post-normalization residual and request-local native evidence.

No runtime/CUDA import occurs until installation.  The original verifier,
backbone, norm, vocabulary projection and cache policy are not replaced.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def corrected_features(hidden, B, A, *, known_nonzero=False):
    """Return h + B A h: FP32 thin products/addition, then the native cast.

    Native tensors use row convention.  Empty/zero factors bypass arithmetic
    and return the exact input tensor.  Neither input clipping nor a fixed
    output basis is used.  Concatenated rank-64 pairs are not truncated.
    """
    import torch
    require(hidden.ndim == 2 and B.ndim == A.ndim == 2, 'Residual requires matrix inputs')
    require(B.shape[0] == A.shape[1] == hidden.shape[1] and B.shape[1] == A.shape[0],
            'Residual factor/hidden dimensions differ')
    require(B.shape[1] <= 64, 'Compiled correction exceeds two rank32 modules')
    if B.shape[1] == 0 or (not known_nonzero and
                           (not bool(torch.count_nonzero(B)) or not bool(torch.count_nonzero(A)))):
        return hidden
    values = hidden.float()
    return (values + (values @ A.float().T) @ B.float().T).to(hidden.dtype)


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _immutable_json(path, value):
    from experiments.dflash_correction_bank.artifacts import atomic_json
    return atomic_json(path, value)


class RequestCorrection:
    """A restored pre-read state routes exactly once and stays fixed to EOS."""
    def __init__(self, declaration, projections, bank_factory=None, *, active_bank=None):
        self.declaration = declaration
        self.enabled = declaration['arm'] == 'BANK'
        self.projections = projections
        self.read_result = self.forecast = self.bank = None
        self.prompt_seen = 0
        self.costs = dict(descriptor_s=0., routing_s=0., fusion_s=0., forecast_persistence_s=0.)
        if self.enabled:
            require(active_bank is not None, 'Activate pre-read bank state before request timing')
            require(active_bank['state_sha256'] == declaration['state_sha256'], 'Active bank snapshot differs')
            self.bank = active_bank['bank']
            self.state_identity = self.bank.identity()

    def prefill(self, features, *, start, prompt_length, forecast_path):
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
        require(tuple(basis.shape) == (features.shape[-1], 64), 'Require pinned 64-dimensional descriptor projection')
        descriptor = features[-1].float() @ basis
        norm = descriptor.norm()
        require(bool(torch.isfinite(descriptor).all()) and float(norm) > 0., 'Invalid prompt descriptor')
        descriptor = (descriptor / norm).detach().cpu().double().numpy()
        self.costs['descriptor_s'] = time.perf_counter() - started
        started = time.perf_counter()
        self.read_result = self.bank.read(descriptor, context_length=prompt_length)
        self.costs['routing_s'] = time.perf_counter() - started
        started = time.perf_counter()
        factors = self.read_result.effective
        b, a = np.asarray(factors.B), np.asarray(factors.A)
        require(b.ndim == a.ndim == 2 and b.shape == (features.shape[-1], a.shape[0]) and
                a.shape[1] == features.shape[-1] and a.shape[0] <= 64, 'Invalid fused native factors')
        require(np.isfinite(b).all() and np.isfinite(a).all(), 'Nonfinite native correction')
        self.B = torch.tensor(b, dtype=torch.float32, device=features.device)
        self.A = torch.tensor(a, dtype=torch.float32, device=features.device)
        self.zero_correction = a.shape[0] == 0 or not np.count_nonzero(b) or not np.count_nonzero(a)
        if features.is_cuda:
            torch.cuda.synchronize(features.device)
        self.costs['fusion_s'] = time.perf_counter() - started
        require(self.bank.identity() == self.state_identity, 'Read mutated bank state')
        self.forecast = dict(forecast_saved_before_response=True,
            boundary='Final target prompt-prefill result, before corrected draft proposals',
            snapshot_sha256=self.declaration['state_sha256'], descriptor_extracted=True,
            routing_recomputed=True, fusion_recomputed=True, cost_in_request_s=True,
            descriptor=descriptor.tolist(), coefficients=_jsonable(self.read_result.alpha),
            alpha=_jsonable(self.read_result.alpha), forecast=_jsonable(self.read_result.forecast),
            diagnostics=_jsonable(self.read_result.diagnostics), compiled_rank=int(a.shape[0]),
            prompt_positions=prompt_length, costs=dict(self.costs))
        started = time.perf_counter()
        _immutable_json(forecast_path, self.forecast)
        self.costs['forecast_persistence_s'] = time.perf_counter() - started

    def apply(self, hidden):
        if not self.enabled:
            return hidden
        require(self.read_result is not None and self.forecast is not None,
                'Drafting started before a causal prompt read')
        if self.zero_correction:
            return hidden
        return corrected_features(hidden, self.B, self.A, known_nonzero=True)

    def finish(self):
        if not self.enabled:
            return None
        require(self.read_result is not None, 'Complete request has no prompt descriptor/read')
        require(self.bank.identity() == self.state_identity, 'Serving mutated bank/memory')
        # The immutable pre-response forecast stays intact; the final trace may
        # additionally disclose the cost of persisting that forecast.
        return {**self.forecast, 'costs': dict(self.costs)}


class FrozenHeadCapture:
    """Capture canonical-serving frozen h; join exact block identities at finish.

    The ordinary token observer remains authoritative for native starts and
    candidates.  No feature capture or transfers occur on timing-only replays.
    Teacher moments are intentionally absent: they require canonical teacher
    replay, rather than treating post-rejection branch labels as canonical.
    """
    def __init__(self, *, enabled, max_bytes=512 * 2**20):
        require(type(max_bytes) is int and max_bytes > 0, 'Positive head capture budget required')
        self.enabled, self.max_bytes = bool(enabled), max_bytes
        self.hidden, self.proposals = [], []
        self.bytes = 0
        self.costs = dict(device_capture_host_s=0., transfer_s=0., serialization_s=0.)

    def append(self, hidden, proposals):
        if not self.enabled:
            return
        import torch
        require(hidden.ndim == 2 and hidden.shape[0] == 9 and hidden.dtype == torch.bfloat16,
                'Require nine native BF16 normalized frozen draft positions')
        require(proposals.numel() == 9, 'Native head proposal count changed')
        # Includes future FP32 serialization and stack overlap, not just BF16.
        required = hidden.numel() * 10 + proposals.numel() * 24
        require(self.bytes + required <= self.max_bytes, 'Bank head capture budget exceeded; no truncation')
        started = time.perf_counter()
        self.hidden.append(hidden.detach().clone())
        self.proposals.append(proposals.detach().reshape(-1).clone())
        self.bytes += required
        self.costs['device_capture_host_s'] += time.perf_counter() - started

    def finish(self, trajectory, destination):
        if not self.enabled:
            return None
        import numpy as np
        import torch
        from experiments.dflash_correction_bank.artifacts import binding
        events = [e for e in trajectory['events'] if e['kind'] == 'verification']
        require(len(events) == len(self.hidden), 'Frozen-head/native verification block counts differ')
        started = time.perf_counter()
        h = torch.stack(self.hidden).float().cpu().numpy() if self.hidden else np.empty((0, 9, 0), np.float32)
        proposed = torch.stack(self.proposals).cpu().numpy() if self.proposals else np.empty((0, 9), np.int64)
        self.costs['transfer_s'] = time.perf_counter() - started
        inputs, outputs = trajectory['input_token_ids'], trajectory['final_output_token_ids']
        blocks = []
        for index, event in enumerate(events):
            require(proposed[index].tolist() == event['draft_proposed_token_ids'],
                    'Frozen-head proposals differ from native trajectory')
            start = int(event['context']['verification_input_start_position'])
            positions = list(event['draft_proposed_positions'])
            require(positions == list(range(start + 1, start + 10)), 'Native block positions changed')
            require(start + 1 == len(inputs) + event['output_offset'], 'Block prefix/output alignment differs')
            eligible = [p < len(inputs) + len(outputs) for p in positions]
            labels = [outputs[p - len(inputs)] if ok else None for p, ok in zip(positions, eligible)]
            blocks.append(dict(block_index=index, block_start=event['output_offset'] - 1,
                absolute_block_start=start, output_start=event['output_offset'],
                output_positions=[p - len(inputs) for p in positions],
                positions=positions, eligible_mask=eligible, canonical_token_ids=labels,
                proposal_token_ids=proposed[index].tolist(),
                native_accepted_prefix=event['native_acceptance_length_before_stop'],
                committed_tokens=len(event['committed_token_ids']), finished=event['finished'],
                stop_clipped=event['stop_clipped'], hidden_key='frozen_h', hidden_index=index))
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), 'Immutable head-capture artifact exists')
        started = time.perf_counter()
        fd, temporary = tempfile.mkstemp(prefix='.' + destination.name, dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, frozen_h=h, native_proposals=proposed.astype(np.int64))
                stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.unlink(temporary)
        self.costs['serialization_s'] = time.perf_counter() - started
        self.hidden.clear(); self.proposals.clear()
        document = dict(schema='correction_bank_v02_native_head_capture_v1',
            input_ids_sha256=digest(inputs), output_ids_sha256=digest(outputs),
            tensors=binding(destination), blocks=blocks, costs=dict(self.costs),
            precision='Frozen native BF16 features stored losslessly as FP32',
            conditioning='Exact native block start and unchanged frozen backbone inputs; teacher moments require canonical replay',
            timing_scope='Canonical serving capture only; costs already included in that request, never add twice')
        return _immutable_json(destination.with_suffix('.json'), document)


def install(entry, config, *, projections=None, bank_factory=None, snapshot_loader=None):
    """Install before scheduler initialization and CUDA graph construction."""
    from sglang.srt.speculative.dflash_worker import DFlashWorker
    from sglang.srt.managers.scheduler import Scheduler
    require(not getattr(DFlashWorker, '_correction_bank_v02_installed', False), 'V02 hooks already installed')
    require(not getattr(DFlashWorker, '_correction_bank_installed', False), 'V01 and V02 hooks cannot be stacked')
    if bank_factory is None:
        from .state import BankState
        bank_factory = BankState.from_snapshot
    if snapshot_loader is None:
        from experiments.dflash_correction_bank.artifacts import load_snapshot
        snapshot_loader = load_snapshot
    native_trace = entry.RequestTrace
    pending = Path(config['worker_out']) / 'BANK_PENDING.json'
    cached_projections, active_bank = {}, {}
    profile_context = {}

    def profile(scheduler):
        """Bounded qualification head/correction profile; never an answer replay."""
        entry.inherited.strict_idle(scheduler)
        require(config.get('bank_qualification') is True, 'Cost profiling is qualification-only')
        require(profile_context, 'Cost profile requires a prior canonical BANK head')
        import torch
        hidden = profile_context['hidden']
        worker, lm_head = profile_context['worker'], profile_context['lm_head']
        require(hidden.is_cuda, 'Native cost profile requires the serving CUDA device')
        d, rank = hidden.shape[1], min(32, hidden.shape[1])
        generator = torch.Generator(device='cpu').manual_seed(17)
        # Fixed synthetic factors exercise arithmetic, not a data-fitted action.
        B = (torch.randn(d, 2 * rank, generator=generator, dtype=torch.float32) * 1e-3).to(hidden.device)
        A = (torch.randn(2 * rank, d, generator=generator, dtype=torch.float32) * 1e-3).to(hidden.device)
        result = []
        for modules, width in ((0, 0), (1, rank), (2, 2 * rank)):
            def invoke():
                value = hidden if modules == 0 else corrected_features(hidden, B[:, :width], A[:width], known_nonzero=True)
                return original_head(worker, hidden_states=value, lm_head=lm_head, chunk_size=256)
            for _ in range(5):
                invoke()
            measurements = []
            for _ in range(20):
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record(); invoke(); after.record(); after.synchronize()
                measurements.append(before.elapsed_time(after) / 1000.)
            result.append(dict(modules=modules, compiled_rank=width, repeat_s=measurements,
                               mean_s=sum(measurements) / len(measurements)))
        baseline = result[0]['mean_s']
        for row in result:
            row['increment_over_zero_s'] = max(0., row['mean_s'] - baseline)
        receipt = dict(schema='correction_bank_v02_cost_profile_v1', paths=result,
            warmups_per_path=5, repeats_per_path=20, seed=17, positions=9,
            arithmetic='Original native head with fixed small FP32 rank32/64 residual factors',
            context_bins=[8192, 16384, 32768],
            scope='Qualification native-head increment only; add to observed full draft/verification cycle cost',
            no_generated_answers=True, no_feedback_or_bank_mutation=True,
            last_canonical_context_length=profile_context['context_length'])
        (Path(config['worker_out']) / 'BANK_PROFILE.json').write_text(json.dumps(receipt, allow_nan=False) + '\n')
        return receipt

    def activate(scheduler):
        entry.inherited.strict_idle(scheduler)
        started = time.perf_counter()
        import numpy as np
        declaration = json.loads((Path(config['worker_out']) / 'BANK_ACTIVATE.json').read_text())
        state, bound = declaration['state'], declaration['projections']
        require(state['state_sha256'] == state['snapshot']['sha256'], 'Activation state hash differs')
        if bound['sha256'] not in cached_projections:
            raw = Path(bound['path']).read_bytes()
            require(len(raw) == bound['bytes'] and hashlib.sha256(raw).hexdigest() == bound['sha256'],
                    'Fixed descriptor projection changed')
            with np.load(bound['path'], allow_pickle=False) as data:
                require(set(data.files) == {'V_descriptor'}, 'V02 must not inherit fixed correction projections')
                cached_projections[bound['sha256']] = {'V_descriptor': data['V_descriptor'].copy()}
        reused = active_bank.get('state_sha256') == state['state_sha256']
        if not reused:
            active_bank.clear()
            active_bank.update(state_sha256=state['state_sha256'], bank=bank_factory(snapshot_loader(state['snapshot'])))
        active_bank['projections_sha256'] = bound['sha256']
        receipt = dict(status='passed', state_sha256=state['state_sha256'], reused=reused,
                       loading_s=time.perf_counter() - started)
        (Path(config['worker_out']) / 'BANK_ACTIVE.json').write_text(json.dumps(receipt, allow_nan=False) + '\n')
        return receipt

    Scheduler.correction_bank_activate = activate
    Scheduler.correction_bank_profile = profile

    class BankTrace(native_trace):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            declaration = json.loads(pending.read_text())
            require(declaration['input_ids_sha256'] == digest(self.input_ids), 'Bank trace input differs')
            selected = projections
            if declaration['arm'] == 'BANK' and selected is None:
                bound = declaration['projections']
                require(bound['sha256'] == active_bank.get('projections_sha256'), 'Descriptor projection activation missing')
                selected = cached_projections[bound['sha256']]
            self.bank_correction = RequestCorrection(declaration, selected, bank_factory, active_bank=active_bank)
            self.bank_forecast_path = declaration['forecast_path']
            capture = bool(declaration.get('capture', False))
            require(not capture or (declaration['arm'] == 'BANK' and declaration.get('canonical') is True),
                    'Only actual canonical BANK serving may collect frozen head features')
            self.bank_head_capture = FrozenHeadCapture(enabled=capture,
                max_bytes=config.get('bank_head_capture_max_bytes', 512 * 2**20))
            self.bank_head_capture_path = declaration.get('head_capture_path',
                str(Path(self.bank_forecast_path).with_suffix('.heads.npz')))
            self.bank_cycle_observations = []

        def finish(self, output_ids, meta):
            value = super().finish(output_ids, meta)
            value['correction_bank_read'] = self.bank_correction.finish()
            value['bank_head_capture'] = self.bank_head_capture.finish(value, self.bank_head_capture_path)
            cycles = []
            for item in self.bank_cycle_observations:
                record = {k: v for k, v in item.items() if k not in ('cuda_start', 'cuda_end')}
                if 'cuda_end' in item:
                    item['cuda_end'].synchronize()
                    record['device_elapsed_s'] = item['cuda_start'].elapsed_time(item['cuda_end']) / 1000.
                cycles.append(record)
            value['correction_bank_cycle_observations'] = cycles
            return value

    entry.RequestTrace = BankTrace
    original_forward = DFlashWorker.forward_batch_generation
    original_head = DFlashWorker._greedy_sample_from_vocab_parallel_head

    def forward(worker, batch, **kwargs):
        trace = entry._active_trace
        correction = getattr(trace, 'bank_correction', None)
        is_prompt = batch.forward_mode.is_extend() or getattr(batch, 'is_extend_in_batch', False)
        capture_prompt = correction is not None and correction.enabled and is_prompt
        recorder = getattr(trace, 'bank_head_capture', None)
        capture_cycle = recorder is not None and recorder.enabled and not is_prompt
        if capture_cycle:
            import torch
            context_length = len(trace.input_ids) + len(batch.reqs[0].output_ids)
            trace.bank_current_context_length = context_length
            cycle = dict(context_length=context_length, scope='Native draft-plus-target-verification forward; prefill excluded')
            if torch.cuda.is_available():
                cycle.update(cuda_start=torch.cuda.Event(enable_timing=True), cuda_end=torch.cuda.Event(enable_timing=True))
                cycle['cuda_start'].record()
            cycle_started = time.perf_counter()
        if capture_prompt:
            require(batch.batch_size() == 1 and int(worker.tp_size) == 1 and not worker.use_compact_draft_cache,
                    'Bank requires qualified C1 full-context runtime')
            # Native forward consumes/zeros spec_info.ctx_lens; even spec_info
            # can initially be None. Copy stable scalar geometry BEFORE it runs.
            lengths, prefixes = getattr(batch, 'extend_lens', None), getattr(batch, 'prefix_lens', None)
            require(isinstance(lengths, (tuple, list)) and isinstance(prefixes, (tuple, list)) and
                    len(lengths) == len(prefixes) == 1, 'Missing qualified prefill geometry')
            chunk_length, start = lengths[0], prefixes[0]
            require(type(chunk_length) is int and chunk_length > 0 and type(start) is int and start >= 0,
                    'Invalid native prefill geometry')
        result = original_forward(worker, batch, **kwargs)
        if capture_cycle:
            cycle['host_forward_s'] = time.perf_counter() - cycle_started
            if 'cuda_end' in cycle:
                cycle['cuda_end'].record()
            trace.bank_cycle_observations.append(cycle)
        if capture_prompt:
            features = getattr(result.logits_output, 'teacher_hidden_states', None)
            require(features is not None, 'Pinned final-normalized target features unavailable')
            require(features.ndim == 2 and chunk_length == features.shape[0],
                    'Target features differ from actual native prefill length')
            correction.prefill(features, start=start, prompt_length=len(trace.input_ids),
                               forecast_path=trace.bank_forecast_path)
        return result

    def head(worker, *, hidden_states, lm_head, chunk_size=256):
        trace = entry._active_trace
        correction = getattr(trace, 'bank_correction', None)
        original_features = hidden_states
        if correction is not None:
            hidden_states = correction.apply(hidden_states)
        result = original_head(worker, hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size)
        recorder = getattr(trace, 'bank_head_capture', None)
        if recorder is not None:
            recorder.append(original_features, result)
            if recorder.enabled and config.get('bank_qualification') is True:
                require(hasattr(trace, 'bank_current_context_length'),
                        'Native profile lacks pre-forward canonical context length')
                profile_context.clear()
                profile_context.update(hidden=original_features.detach().clone(), worker=worker, lm_head=lm_head,
                    context_length=trace.bank_current_context_length)
        return result

    DFlashWorker.forward_batch_generation = forward
    DFlashWorker._greedy_sample_from_vocab_parallel_head = head
    DFlashWorker._correction_bank_v02_installed = True
    return dict(draft_hook='DFlashWorker._greedy_sample_from_vocab_parallel_head after frozen DFlashModel.norm',
        descriptor_hook='Final target prompt-prefill logits_output.teacher_hidden_states[-1]',
        descriptor_scope='64-dimensional fixed seed17 projection; last prompt token only',
        prefill_geometry_hook='ScheduleBatch.extend_lens/prefix_lens copied before native mutation',
        zero_path='Original tensor without addition/cast',
        nonzero_precision='FP32 B,A thin residual products/addition, then original native BF16 head cast',
        fusion='Concatenated factors, at most rank64; no serving dense matrix or SVD',
        capture='Canonical BANK only; exact native block starts joined through the retained token observer',
        extra_target_calls=0, compiled_vocabulary_projections_per_draft_position=1)
