"""Review-gated correction-bank integration with the qualified native runtime.

Import is CPU/metadata only. Launch is exclusively through run.py after review.
Historical author sampler, OSD optimizer and native verifier sources are reused
unchanged and source-pinned. Three methods share this serving process.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
_spec = importlib.util.spec_from_file_location('_bank_per_turn_native', HERE.parent / 'dflash_per_turn/native.py')
per_turn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = per_turn
_spec.loader.exec_module(per_turn)
base = per_turn.base
require, sha, digest, write_json = base.require, base.sha, base.object_hash, base.write_json
_original_worker_command = per_turn._original_worker_command
_original_sd_scheduler = base._sd_scheduler
_original_storage_bound = per_turn.storage_bound


def source_pins():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}


def check_sources(config):
    require(config.get('correction_bank_source_sha256') == source_pins(), 'Reviewed bank code changed')
    per_turn.check_sources(config)


def check_numerical_environment():
    """Numerical solver/projection pins; no package installation or CUDA load."""
    import numpy
    import scipy
    actual = {'numpy': numpy.__version__, 'scipy': scipy.__version__}
    expected = {'numpy': '2.2.6', 'scipy': '1.15.3'}
    require(actual == expected, f'Bank numerical environment differs: {actual}; require {expected}')
    return actual


def worker_command(config):
    command = _original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Qualified worker command changed')
    command[-1] = str(HERE / 'native.py')
    return command


base.worker_command = worker_command


def _sd_scheduler(*args, **kwargs):
    config = base._worker_config()
    entry = base._serving_entry(config)
    from experiments.dflash_correction_bank import native_hooks
    receipt = native_hooks.install(entry, config)
    write_json(Path(config['worker_out']) / 'BANK_NATIVE_HOOKS.json', receipt)
    return _original_sd_scheduler(*args, **kwargs)


base._sd_scheduler = _sd_scheduler


def campaign_storage_bound(model_bytes, *, main_bank_bytes=8*2**30,
                           qualification_bank_bytes=12*2**30, reserve_bytes=8*2**30):
    """Conservative whole-campaign disk peak, with no assumed offload/deletion.

    Seven episodes = one qualification plus six fresh streams. Inherited
    retirement keeps BOTH pre/post bundles for the final OSD update. Each is
    one BF16 checkpoint and two BF16 Adam moments (3 model-sized units), plus
    64MiB metadata. Thus 6 model units remain per episode. The qualified rolling
    bound is 11 units; subtract the 6 already retained, leaving 5 transient
    units and the full 512MiB rolling overhead. No final-state deletion is
    assumed. The extra per-bundle metadata allowance is not subtracted from
    that rolling overhead.

    Each main bank retains at most 24 response Fisher/feature summaries in an
    8GiB envelope. Qualification has a 12GiB envelope: at most 20 responses,
    512*9 eligible positions each, plus FP64 h+teacher replay features. No
    shrinking of these envelopes or silent disposal is an admission strategy.
    """
    require(type(model_bytes) is int and model_bytes > 0, 'Positive model byte size required')
    require(type(main_bank_bytes) is int and main_bank_bytes >= 8*2**30, 'Main bank envelope below 8GiB')
    require(type(qualification_bank_bytes) is int and qualification_bank_bytes >= 12*2**30,
            'Qualification bank/replay envelope below 12GiB')
    require(type(reserve_bytes) is int and reserve_bytes >= 8*2**30, 'Reserve below 8GiB')
    bundles_per_episode = 2
    retained_model_units = bundles_per_episode*3
    final_osd = 7*bundles_per_episode*(3*model_bytes+64*2**20)
    banks = 6*main_bank_bytes+qualification_bank_bytes
    rolling = _original_storage_bound(model_bytes, remaining_diagnostic_cases=0)
    transient = rolling-retained_model_units*model_bytes
    require(transient >= 0, 'Rolling bound cannot be smaller than retained OSD state')
    return dict(schema='bank_campaign_storage_v2', episodes=7, main_streams=6,
        model_bytes=model_bytes, retained_osd_bundles_bytes=final_osd,
        retained_bank_and_qualification_bytes=banks, transient_rolling_bytes=transient,
        active_episode_osd_peak_bytes=rolling,
        reserve_bytes=reserve_bytes, planned_writes_bytes=final_osd+banks+transient,
        required_free_bytes=final_osd+banks+transient+reserve_bytes,
        assumptions={'final_osd_bundle_model_units': 3, 'final_osd_bundles_per_episode': bundles_per_episode,
            'retained_osd_model_units_per_episode': retained_model_units,
            'metadata_bytes_per_osd_bundle': 64*2**20,
            'rolling_osd_model_units': 11, 'transient_model_units': 5,
            'main_bank_envelope_bytes': main_bank_bytes, 'qualification_bank_envelope_bytes': qualification_bank_bytes,
            'author_anchors_max': 512, 'supervised_positions_per_anchor_max': 9,
            'first_qualification_requests': 20, 'main_stream_requests_max': 24,
            'offload_assumed': False, 'final_osd_retirement_assumed': False,
            'shared_array_content_addressing_required': True})


def admit_campaign_storage(config):
    """Read-only budget admission before any qualification GPU process starts."""
    models = json.loads(base.checked_binding(config['model_bindings']).read_text())
    model_bytes = sum(v['bytes'] for k, v in models['released_draft']['checkpoint_files'].items()
                      if k.endswith('.safetensors'))
    main = config.get('bank_artifact_budget_bytes', 8*2**30)
    qual = config.get('qualification_bank_artifact_budget_bytes', 12*2**30)
    plan = campaign_storage_bound(model_bytes, main_bank_bytes=main,
                                  qualification_bank_bytes=qual,
                                  reserve_bytes=config.get('reserve_bytes', 8*2**30))
    declared = config.get('campaign_storage_budget_bytes', plan['planned_writes_bytes'])
    require(type(declared) is int and declared >= plan['planned_writes_bytes'],
            'Declared cumulative campaign storage budget is below the retention/temporary bound')
    location = Path(config['output_root']).resolve().parent
    while not location.exists():
        location = location.parent
    free = shutil.disk_usage(location).free
    plan.update(location=str(location), free_before_bytes=free,
                configured_campaign_storage_budget_bytes=declared,
                required_free_bytes=declared+plan['reserve_bytes'],
                status='passed' if free >= declared+plan['reserve_bytes'] else 'blocked')
    require(plan['status'] == 'passed',
            f'Complete qualification + six-stream retention requires {plan["required_free_bytes"]} free bytes; '
            f'only {free} are available. No GPU started. Review an explicit offload/retention plan before launch.')
    return plan


class NativeBackend(per_turn.NativeBackend):
    def start_episode(self, episode_id, *, seed=17, qualification=False):
        require(seed == 17, 'Each stream starts with seed17')
        actual = source_pins()
        require(self.config.get('correction_bank_source_sha256', actual) == actual, 'Bank source pin changed')
        self.config.update(correction_bank_source_sha256=actual, remaining_diagnostic_cases=0,
                           bank_qualification=bool(qualification))
        if qualification:
            admission = admit_campaign_storage(self.config)
            minimum = 12*2**30
            bank_budget = self.config.get('qualification_bank_artifact_budget_bytes', minimum)
        else:
            admission = None
            minimum = 8*2**30
            bank_budget = self.config.get('bank_artifact_budget_bytes', minimum)
        require(type(bank_budget) is int and bank_budget >= minimum,
                'Bank factors, projection and qualification replay exceed the declared envelope')
        self.config['effective_bank_artifact_budget_bytes'] = bank_budget
        per_turn.storage_bound = lambda size, remaining=0: _original_storage_bound(size, remaining)+bank_budget
        self.qualification = bool(qualification)
        self.bank_updates = []
        self.bank_after_turn = self.osd_after_turn = 0
        result = super().start_episode(episode_id)
        if admission is not None:
            write_json(self.out / 'CAMPAIGN_STORAGE_ADMISSION.json', admission)
        self.trainer = base._WorkerClient(self.config, 'train', self.out / 'training_worker')
        self.totals['startup_s'] += self.trainer.startup_s
        started = time.perf_counter()
        prepared = self.trainer.call('bank_prepare', {'qualification': self.qualification})
        self.projections = prepared['projections']
        self.bank_snapshot = prepared['bank_snapshot']
        base.checked_binding(self.projections)
        base.checked_binding(self.bank_snapshot)
        self.qualification_evidence = []
        result.update(training_startup_s=self.trainer.startup_s,
                      bank_preparation_s=time.perf_counter()-started, bank_preparation=prepared)
        write_json(self.out / 'BANK_PREPARATION.json', result)
        return result

    def snapshot_state(self, arm, turn):
        require(arm in ('F', 'OSD', 'BANK') and 1 <= turn <= 24, 'Invalid pre-request state')
        if arm == 'BANK':
            require(self.bank_after_turn == turn-1, 'Bank learning chronology differs')
            bound = copy.deepcopy(self.bank_snapshot)
            base.checked_binding(bound)
            return dict(arm=arm, turn=turn, after_turn=self.bank_after_turn,
                        before_read=True, snapshot=bound, state_sha256=bound['sha256'])
        after = 0 if arm == 'F' else self.osd_after_turn
        require(arm == 'F' or after == turn-1, 'OSD learning chronology differs')
        label = 'F' if after == 0 else 'D'+str(after)
        checkpoint = copy.deepcopy(self.checkpoints[label])
        return dict(arm=arm, turn=turn, after_turn=after, checkpoint=checkpoint,
                    state_sha256=checkpoint['checkpoint_identity_sha256'])

    def serve(self, input_ids, arm, state, capture=False, *, turn_index, repeat=None, canonical=False):
        base.check_input(input_ids)
        require(capture is False, 'All paired serving disables training capture')
        require(arm == state['arm'] and turn_index == state['turn'], 'Wrong request state')
        require(canonical == (repeat is None), 'Canonical and benchmark provenance differs')
        label = 'F' if arm == 'BANK' else state['checkpoint']['checkpoint_id']
        startup = self._ensure_worker('sd')
        publication = self.publish(label)
        bank_activation_s = 0.
        bank_loading = None
        if arm == 'BANK':
            activation_start = time.perf_counter()
            bank_loading = self.worker.call('bank_activate', {'state': state, 'projections': self.projections})
            bank_activation_s = time.perf_counter()-activation_start
        reset = self.worker.call('flush', {})
        started = time.perf_counter()
        result = self.worker.call('generate', {'input_ids': list(input_ids), 'checkpoint_id': label,
            'capture': False, 'turn': turn_index, 'repeat': repeat, 'arm': arm,
            'state': state, 'projections': self.projections, 'canonical': bool(canonical)})
        result['request_s'] = time.perf_counter()-started
        require(result['input_ids'] == input_ids and result['input_ids_sha256'] == digest(input_ids),
                'Native replay differs from canonical input tokens')
        result.update(arm=arm, state_sha256=state['state_sha256'], checkpoint_id=label,
            startup_s=startup, activation_s=publication['activation_s']+bank_activation_s,
            native_activation_s=publication['native_activation_s'], cache_maintenance_s=reset['seconds'],
            bank_activation_s=bank_activation_s, bank_loading=bank_loading,
            deployment_activation_s=(bank_activation_s if arm == 'BANK' else publication['activation_s']) if canonical else 0.,
            benchmark_method_restore_s=publication['activation_s'] if arm == 'BANK' else 0.,
            cap_hit=len(result['output_ids']) == 1024, capture=False, canonical=bool(canonical),
            timing_scope='Available checkpoint: complete request RPC including prompt prefill, BANK descriptor/read/fusion, drafting and target verification; excludes external state/weight activation, cache reset and learning.')
        if hasattr(self, 'target'):
            result.update(target_id='T0', target_identity_sha256=self.target['identity_sha256'])
        return result

    def learn(self, arm, item, pre_state, *, turn_index, final=False, qualification=False):
        require(item.get('canonical') is True and item['arm'] == arm and
                item['state_sha256'] == pre_state['state_sha256'], 'Only actual canonical response is feedback')
        require(qualification == self.qualification, 'Qualification mode differs')
        started = time.perf_counter()
        if arm == 'OSD':
            require(not final and turn_index == self.osd_after_turn+1, 'Wrong OSD update event')
            support = self.prepare_support(item)
            result = self.train_event(turn_index, [support])
            self.osd_after_turn = turn_index
            result['costs']['capture_preparation_s'] = support['collection_s']
            # Comparisons for the preceding state have completed. Preserve the
            # newest restart state and checkpoint; retire only old bulk through
            # the qualified manifest-writing helper.
            retirement_started = time.perf_counter()
            result['retention'] = self.retire({'F', 'D'+str(turn_index)}, set())
            result['retirement_s'] = time.perf_counter()-retirement_started
            result['costs']['retention_maintenance_s'] = result['retirement_s']
            result['retention_cost_scope'] = 'Recurring rolling-checkpoint validation/retirement; included once in adaptation outer wall, not added to training time.'
            result['recurring_adaptation_s'] = time.perf_counter()-started
            return result
        require(arm == 'BANK' and turn_index == self.bank_after_turn+1, 'Wrong bank feedback event')
        response = {**item, 'target_id': 'T0', 'target_identity_sha256': self.target['identity_sha256'],
                    'input_token_count': len(item['input_ids']), 'output_cap': 1024,
                    'capture_checkpoint_sha256': pre_state['state_sha256'],
                    'capture_checkpoint': 'BANK'}
        result = self.trainer.call('bank_feedback', {
            'response': response, 'before_snapshot': pre_state['snapshot'], 'served_read': item['read'],
            'turn': turn_index, 'final': bool(final), 'qualification': bool(qualification)})
        result['feedback_rpc_wall_s'] = time.perf_counter()-started
        result['recurring_adaptation_s'] = max(0., result['feedback_rpc_wall_s']-
                                               result.get('costs', {}).get('qualification_only_s', 0.))
        result['state_before_sha256'] = pre_state['state_sha256']
        result['input_ids_sha256'] = digest(item['input_ids'])
        result['output_ids_sha256'] = digest(item['output_ids'])
        self.bank_updates.append(result)
        if result.get('qualification_checks'):
            self.qualification_evidence.append(result['qualification_checks'])
        if not final:
            self.bank_snapshot = result['bank_snapshot']
            base.checked_binding(self.bank_snapshot)
            self.bank_after_turn = turn_index
        else:
            result['analysis_only_final_feedback_s'] = result['recurring_adaptation_s']
            result['recurring_adaptation_s'] = 0.
        return result

    def qualify(self, result):
        cold = [v for v in result['canonical'] if v['arm'] == 'BANK' and v['turn'] <= 2]
        cold_start = len(cold) == 2 and all(not any(v['read']['coefficients']) for v in cold)
        zero_rows = [v for v in result['scores'] if v['arm'] == 'BANK' and not any(v['read']['coefficients'])]
        zero_keys = {(v['turn'], v['repeat']) for v in zero_rows}
        paired_frozen = [v for v in result['scores'] if v['arm'] == 'F' and (v['turn'], v['repeat']) in zero_keys]
        zero_preserves = (bool(zero_rows) and len(paired_frozen) == len(zero_rows) and
                          all(v.get('same_generation_valid') is True for v in zero_rows+paired_frozen))
        evidence = self.qualification_evidence
        nonempty = [v for v in evidence if v.get('sampled_positions',0)>0]
        checks = dict(
            compiled_fusion=bool(nonempty) and all(v.get('compiled_fusion', {}).get('passed') is True for v in evidence),
            masks_and_conditioning=bool(evidence) and all(v.get('current_response_only') and v.get('fixed_sampled_objective') for v in evidence),
            compressed_feedback=bool(nonempty) and all(v.get('compressed_direct', {}).get('passed') is True for v in evidence),
            direct_retention_bounds=bool(evidence) and all(v.get('direct_historical', {}).get('passed') is True for v in evidence),
            current_response_fit_reported=bool(nonempty) and all('source_fit' in v for v in evidence),
            native_precision_reported=bool(nonempty) and all(
                v.get('native_precision_reported') is True and v.get('native_precision',{}).get('status')=='measured'
                for v in evidence))
        checks.update(cold_start=cold_start, zero_preserves_outputs=zero_preserves,
                      direct_evidence=self.qualification_evidence, nonempty_feedback_responses=len(nonempty),
                      performance_win_required=False)
        write_json(self.out / 'QUALIFICATION.json', checks)
        return checks

    def close_episode(self, result):
        # Admission can fail before the inherited backend assigns self.out.
        # Still close any initialized workers, but never create a directory or
        # reuse a previous episode's output path just to record cleanup.
        initialized = self.episode_id is not None
        episode = self.episode_id if initialized else result.get('episode_id')
        out = self.out if initialized else None
        finished = self.finish_episode()
        receipt = dict(status='passed', episode_id=episode, native=finished,
            episode_initialized=initialized,
            retention='Bank snapshots/projections/statistics/protection/anchors/solver records retained; final OSD restart preserved. Old completed OSD comparisons retired with exact manifests.',
            qualification_direct_features_retained=bool(initialized and getattr(self, 'qualification', False)))
        if out is not None:
            write_json(out / 'BANK_CLOSE.json', receipt)
        return receipt


class ServingService(per_turn.ServingService):
    def __init__(self, config):
        check_sources(config)
        super().__init__(config)

    def bank_activate(self, body):
        started = time.perf_counter()
        require(body['state']['arm'] == 'BANK' and body['state']['before_read'], 'Require pre-read bank state')
        write_json(self.out / 'BANK_ACTIVATE.json', body)
        self.engine.collective_rpc('correction_bank_activate')
        receipt = json.loads((self.out / 'BANK_ACTIVE.json').read_text())
        require(receipt['status'] == 'passed' and receipt['state_sha256'] == body['state']['state_sha256'],
                'Scheduler bank activation failed or used a stale snapshot')
        return dict(status='passed', receipt=receipt, activation_s=time.perf_counter()-started)

    def generate(self, body):
        arm, state = body['arm'], body['state']
        require(arm in ('F', 'OSD', 'BANK') and state['arm'] == arm, 'Unknown bank comparison arm')
        require(body['checkpoint_id'] == self.logical_checkpoint, 'Stale full-weight checkpoint')
        require(arm != 'BANK' or self.logical_checkpoint == 'F', 'Bank backbone must be released DFlash')
        serial = self.serial+1
        directory = self.out / 'bank_forecasts'; directory.mkdir(exist_ok=True)
        declaration = dict(arm=arm, input_ids_sha256=digest(body['input_ids']),
            state_sha256=state['state_sha256'], projections=body['projections'],
            snapshot=state.get('snapshot'), forecast_path=str(directory / f'{serial:06d}.json'),
            canonical=body['canonical'], turn=body['turn'])
        write_json(self.out / 'BANK_PENDING.json', declaration)
        result = super().generate(body)
        document = json.loads(base.checked_binding(result['trajectory']).read_text())
        result['read'] = document.get('correction_bank_read')
        require((result['read'] is not None) == (arm == 'BANK'), 'Missing or leaked bank correction')
        return result


def worker_main():
    """Authenticated local RPC; failures poison a worker rather than retrying."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import traceback
    config = base._worker_config()
    check_sources(config)
    numerical = check_numerical_environment()
    write_json(Path(config['worker_out']) / 'BANK_NUMERICAL_ENVIRONMENT.json', numerical)
    base.check_dependencies(config['workspace_root'])
    allocation = os.environ.pop('CUDA_VISIBLE_DEVICES')
    try:
        base.check_environment(config)
    finally:
        os.environ['CUDA_VISIBLE_DEVICES'] = allocation
    if config['kind'] == 'train':
        from experiments.dflash_correction_bank.training import BankTrainingService
        service = BankTrainingService(config)
    else:
        service = ServingService(config)
    poisoned = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        def log_message(self, *_):
            pass
        def do_POST(self):
            try:
                require(self.headers.get('X-Two-Update-Token') == config['rpc_token'], 'Foreign worker client')
                length = int(self.headers.get('Content-Length', '0'))
                require(0 < length <= 64*2**20, 'Invalid RPC payload')
                body = json.loads(self.rfile.read(length))
                action = self.path.lstrip('/')
                require(not poisoned, 'Worker poisoned after prior failure')
                if action == 'ready':
                    value = dict(status='passed', pid=os.getpid())
                else:
                    allowed = {'train', 'bank_prepare', 'bank_feedback'} if config['kind'] == 'train' else {'generate', 'publish', 'flush', 'bank_activate'}
                    require(action in allowed, 'Unknown worker action')
                    method = {'bank_prepare': 'prepare', 'bank_feedback': 'feedback'}.get(action, action)
                    value = getattr(service, method)(body)
                status = 200
            except Exception as error:
                poisoned.append(repr(error))
                value = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
                write_json(Path(config['worker_out']) / f'FAILURE_{len(poisoned)}.json', value)
                status = 500
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

    server = HTTPServer(('127.0.0.1', config['port']), Handler)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close(); service.close()


if __name__ == '__main__':
    require(sys.argv[1:] == ['--worker'], 'Use the reviewed controlled campaign CLI')
    worker_main()
