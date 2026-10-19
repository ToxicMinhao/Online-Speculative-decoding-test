"""Fixed singleton at the original native head boundary; no read/write routing."""
from pathlib import Path
import hashlib
import json
import time

from .artifacts import checked
from .native_hooks import corrected_features, digest, require


def selected_matrix(case):
    import numpy as np
    document = json.loads(checked(case['after_snapshot']).read_text())
    require(document['schema'] == 'correction_bank_snapshot_v1', 'Unknown retained snapshot')
    encoded = document['value']['matrices']
    matrices = np.load(checked(encoded['__array__']), allow_pickle=False)
    require(matrices.dtype.str == encoded['dtype'] and list(matrices.shape) == encoded['shape'],
            'Retained matrix schema changed')
    require(matrices.shape == (4, 64, 64), 'Retained bank geometry differs')
    matrix = matrices[case['module']].copy()
    require(np.isfinite(matrix).all(), 'Nonfinite retained correction')
    if 'matrix_sha256' in case:
        require(hashlib.sha256(matrix.tobytes()).hexdigest() == case['matrix_sha256'], 'Selected matrix changed')
    return matrix


def install(entry, config):
    """Reuse exact head arithmetic; do not install the descriptor/read hook."""
    import numpy as np
    import torch
    from sglang.srt.speculative.dflash_worker import DFlashWorker
    from sglang.srt.managers.scheduler import Scheduler
    require(not getattr(DFlashWorker, '_retained_correction_installed', False), 'Replay hooks already installed')
    require(not getattr(DFlashWorker, '_correction_bank_installed', False), 'Campaign hooks must not be active')
    cases = {case['after_snapshot']['sha256']: case for case in config['retained_cases']}
    active, projection_cache = {}, {}
    out = Path(config['worker_out'])

    def activate(scheduler):
        entry.inherited.strict_idle(scheduler)
        started = time.perf_counter()
        body = json.loads((out/'BANK_ACTIVATE.json').read_text())
        state = body['state']; key = state['state_sha256']
        require(key in cases, 'Unapproved retained state')
        case = cases[key]
        require(state['snapshot'] == case['after_snapshot'] and body['projections'] == case['projections'],
                'Snapshot/projections differ from fixed case')
        matrix = selected_matrix(case)
        bound = case['projections']
        if bound['sha256'] not in projection_cache:
            with np.load(checked(bound), allow_pickle=False) as values:
                projection_cache[bound['sha256']] = tuple(torch.as_tensor(values[k], dtype=torch.float32,
                    device=torch.cuda.current_device()) for k in ('U', 'V_in'))
        U, V = projection_cache[bound['sha256']]
        active.clear(); active.update(state_sha256=key, module=case['module'],
            matrix_sha256=case['matrix_sha256'],
            C=torch.as_tensor(matrix, dtype=torch.float32, device=torch.cuda.current_device()), U=U, V=V,
            nonzero=bool(np.count_nonzero(matrix)))
        torch.cuda.synchronize()
        receipt = dict(status='passed', state_sha256=key, module=case['module'], coefficient=1.,
            matrix_sha256=case['matrix_sha256'], routing_bypassed=True, loading_s=time.perf_counter()-started)
        (out/'BANK_ACTIVE.json').write_text(json.dumps(receipt, allow_nan=False)+'\n')
        return receipt

    Scheduler.correction_bank_activate = activate
    original_trace = entry.RequestTrace
    original_head = DFlashWorker._greedy_sample_from_vocab_parallel_head

    class RetainedTrace(original_trace):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            declaration = json.loads((out/'BANK_PENDING.json').read_text())
            require(declaration['input_ids_sha256'] == digest(self.input_ids), 'Replay input changed')
            self.retained_correction = dict(active) if declaration['arm'] == 'BANK' else None
            self.retained_head_calls = 0
            if self.retained_correction is not None:
                require(active.get('state_sha256') == declaration['state_sha256'], 'Wrong fixed singleton activated')

        def finish(self, output_ids, meta):
            value = super().finish(output_ids, meta)
            correction = self.retained_correction
            if correction is None:
                value['correction_bank_read'] = None
            else:
                require(active['state_sha256'] == correction['state_sha256'], 'Correction changed during request')
                alpha = [0.]*4; alpha[correction['module']] = 1.
                value['correction_bank_read'] = dict(routing_bypassed=True, descriptor_extracted=False,
                    fixed_for_entire_request=True, coefficients=alpha, module=correction['module'],
                    snapshot_sha256=correction['state_sha256'], matrix_sha256=correction['matrix_sha256'],
                    corrected_head_calls=self.retained_head_calls,
                    scope='Same-response retained singleton; not a causal online read or deployment timing.')
            return value

    def head(worker, *, hidden_states, lm_head, chunk_size=256):
        trace = entry._active_trace
        correction = getattr(trace, 'retained_correction', None)
        if correction is not None:
            trace.retained_head_calls += 1
            hidden_states = corrected_features(hidden_states, correction['C'], correction['U'], correction['V'],
                                                known_nonzero=correction['nonzero'])
        return original_head(worker, hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size)

    entry.RequestTrace = RetainedTrace
    DFlashWorker._greedy_sample_from_vocab_parallel_head = head
    DFlashWorker._retained_correction_installed = True
    return dict(status='passed', routing_bypassed=True, descriptor_hook_installed=False,
        head_hook='DFlashWorker._greedy_sample_from_vocab_parallel_head after native norm',
        native_correction_arithmetic='Existing FP32 correction/addition then native BF16 cast',
        verifier_modified=False, training_enabled=False)
