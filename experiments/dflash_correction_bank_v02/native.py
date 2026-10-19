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
_spec = importlib.util.spec_from_file_location('_bank_v02_per_turn_native', HERE.parent / 'dflash_per_turn/native.py')
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
    from experiments.dflash_correction_bank_v02 import native_hooks
    receipt = native_hooks.install(entry, config)
    write_json(Path(config['worker_out']) / 'BANK_NATIVE_HOOKS.json', receipt)
    return _original_sd_scheduler(*args, **kwargs)


base._sd_scheduler = _sd_scheduler


def campaign_storage_bound(model_bytes, *, main_bank_bytes=512*2**20,
                           qualification_bank_bytes=2*2**30, reserve_bytes=8*2**30):
    """Peak includes rolling OSD recovery, all compact banks, staging and reserve.

    Successful episode result/validity is persisted before final OSD retirement.
    Failed episodes retain their rolling state and stop the campaign. Thus two
    rolling episodes never coexist. No offload or compression is assumed.
    """
    require(type(model_bytes) is int and model_bytes > 0, 'Positive model size required')
    require(main_bank_bytes >= 512*2**20 and qualification_bank_bytes >= 2*2**30,
            'Compact-artifact envelopes below pinned budget')
    require(reserve_bytes >= 8*2**30, 'Reserve below 8GiB')
    rolling = _original_storage_bound(model_bytes, remaining_diagnostic_cases=0)
    compact = 6*main_bank_bytes+qualification_bank_bytes
    staging = 2*2**30
    return dict(schema='bank_v02_storage_v1', episodes=7, main_streams=6,
        model_bytes=model_bytes, active_episode_osd_peak_bytes=rolling,
        compact_artifacts_bytes=compact, staging_bytes=staging, reserve_bytes=reserve_bytes,
        planned_writes_bytes=rolling+compact+staging,
        required_free_bytes=rolling+compact+staging+reserve_bytes,
        assumptions={'qualification_requests':3,'main_requests':116,
            'retire_completed_osd_after_validated_result':True,
            'retain_failed_episode_osd':True,'rolling_model_units':11,
            'max_memory_blocks_per_stream':48,'fp32_features_and_moments':True,
            'content_addressed_array_sharing':True,'offload_assumed':False})


def admit_campaign_storage(config):
    """Read-only budget admission before any qualification GPU process starts."""
    models = json.loads(base.checked_binding(config['model_bindings']).read_text())
    model_bytes = sum(v['bytes'] for k, v in models['released_draft']['checkpoint_files'].items()
                      if k.endswith('.safetensors'))
    main = config.get('bank_artifact_budget_bytes', 512*2**20)
    qual = config.get('qualification_bank_artifact_budget_bytes', 2*2**30)
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
    def retire_native_capture(self, item, update):
        """Retire only owned transient head rows after compact evidence verifies.

        The capture JSON/token trajectory remain.  Selected full-hidden memory,
        anchor/RNG metadata, solver receipt and bank snapshot are independently
        loaded (including every array hash) before the original NPZ is removed.
        This is future execution policy; importing/testing does not clean disks.
        """
        if self.qualification:
            return dict(status='retained', reason='Qualification keeps original native head reconstruction evidence')
        from experiments.dflash_correction_bank.artifacts import checked, load_snapshot
        require(update.get('status') == 'passed' and update.get('event') == item['turn'],
                'Capture retirement requires the successful matching feedback event')
        require(item.get('canonical') is True and item.get('arm') == 'BANK',
                'Only own canonical BANK captures may be retired')
        capture_binding = item['bank_head_capture']
        capture_path = checked(capture_binding)
        capture = json.loads(capture_path.read_text())
        require(capture['input_ids_sha256'] == digest(item['input_ids']) and
                capture['output_ids_sha256'] == digest(item['output_ids']), 'Capture retirement identity mismatch')
        trajectory = json.loads(checked(item['trajectory']).read_text())
        require(trajectory['input_token_ids'] == item['input_ids'] and
                trajectory['final_output_token_ids'] == item['output_ids'],
                'Retained native trajectory differs from capture being retired')
        feedback = load_snapshot(update['feedback'])
        load_snapshot(update['solver_diagnostics'])
        load_snapshot(update['bank_snapshot'])
        require(feedback['response']['input_ids'] == item['input_ids'] and
                feedback['response']['output_ids'] == item['output_ids'], 'Retained feedback trajectory differs')
        require({'memory', 'metadata', 'reservation', 'rng_before', 'rng_after', 'canonical_checks'} <= set(feedback),
                'Compact feedback lacks required memory/anchor/RNG evidence')
        for block in feedback['memory']:
            require({'h', 'mu', 'omega', 'token_ids', 'output_positions'} <= set(block),
                    'Selected memory evidence is incomplete')
        tensor_binding = capture['tensors']
        tensor_path = checked(tensor_binding)
        require(tensor_path.is_relative_to(self.out.resolve()) and not Path(tensor_binding['path']).is_symlink()
                and tensor_path.parent.name == 'bank_forecasts' and tensor_path.name.endswith('.heads.npz'),
                'Refuse capture retirement outside owned native forecast directory')
        directory = self.out / 'bank_capture_retirement'
        directory.mkdir(exist_ok=True)
        plan_path = directory / f'turn_{item["turn"]:02d}_PLAN.json'
        receipt_path = directory / f'turn_{item["turn"]:02d}.json'
        require(not plan_path.exists() and not receipt_path.exists(), 'Capture retirement already recorded')
        plan = dict(status='planned', event=item['turn'], capture_metadata=capture_binding,
            removed_tensor=tensor_binding, retained_feedback=update['feedback'],
            retained_solver_diagnostics=update['solver_diagnostics'], retained_bank_snapshot=update['bank_snapshot'],
            retained_native_trajectory=item['trajectory'],
            reason='Selected full-hidden memory and metadata verified; remaining per-request head rows are transient',
            lost_capability='Re-evaluate unselected native block features without exact reconstruction; tensors were retired, not archived',
            free_before_bytes=shutil.disk_usage(self.out).free)
        write_json(plan_path, plan)
        tensor_path.unlink()
        fd = os.open(tensor_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        receipt = {**plan, 'status': 'retired', 'free_after_bytes': shutil.disk_usage(self.out).free,
                   'removed_bytes': tensor_binding['bytes'], 'compact_evidence_verified': True}
        write_json(receipt_path, receipt)
        return {**receipt, 'receipt': base.binding(receipt_path)}

    def start_episode(self, episode_id, *, seed=17, qualification=False):
        require(seed == 17, 'Each stream starts with seed17')
        actual = source_pins()
        require(self.config.get('correction_bank_source_sha256', actual) == actual, 'Bank source pin changed')
        self.config.update(correction_bank_source_sha256=actual, remaining_diagnostic_cases=0,
                           bank_qualification=bool(qualification))
        if qualification:
            admission = admit_campaign_storage(self.config)
            minimum = 2*2**30
            bank_budget = self.config.get('qualification_bank_artifact_budget_bytes', minimum)
        else:
            admission = None
            minimum = 512*2**20
            bank_budget = self.config.get('bank_artifact_budget_bytes', minimum)
        require(type(bank_budget) is int and bank_budget >= minimum,
                'Bank factors, projection and qualification replay exceed the declared envelope')
        self.config['effective_bank_artifact_budget_bytes'] = bank_budget
        per_turn.storage_bound = lambda size, remaining=0: _original_storage_bound(size, remaining)+bank_budget
        self.qualification = bool(qualification)
        self.bank_updates = []
        self.cost_profile = None
        self.cost_profile_binding = None
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
        if canonical and arm == 'BANK' and self.qualification and self.cost_profile is None:
            if result.get('correction_bank_cycle_observations'):
                profile_started = time.perf_counter()
                raw = self.worker.call('bank_profile', {})
                from experiments.dflash_correction_bank_v02.costs import profile_from_native
                self.cost_profile = profile_from_native(raw, result['correction_bank_cycle_observations'])
                path = self.out / 'QUALIFIED_COST_PROFILE.json'
                write_json(path, self.cost_profile)
                self.cost_profile_binding = base.binding(path)
                result['qualification_cost_profile_s'] = time.perf_counter()-profile_started
                result['qualification_cost_profile_status'] = 'measured'
            else:
                result['qualification_cost_profile_status'] = 'deferred_no_native_draft_blocks'
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
            'turn': turn_index, 'final': bool(final), 'qualification': bool(qualification),
            'cost_profile': self.cost_profile, 'native_head_capture': item.get('bank_head_capture'),
            'cycle_observations': item.get('correction_bank_cycle_observations', [])})
        result['feedback_rpc_wall_s'] = time.perf_counter()-started
        diagnostic_s = (result.get('costs', {}).get('qualification_only_s', 0.) +
                        result.get('costs', {}).get('source_fit_diagnostic_only_s', 0.))
        result['recurring_adaptation_s'] = max(0., result['feedback_rpc_wall_s']-diagnostic_s)
        result['nondeployment_diagnostic_s'] = diagnostic_s
        result['state_before_sha256'] = pre_state['state_sha256']
        result['input_ids_sha256'] = digest(item['input_ids'])
        result['output_ids_sha256'] = digest(item['output_ids'])
        retirement_started = time.perf_counter()
        result['native_capture_retention'] = self.retire_native_capture(item, result)
        retirement_s = time.perf_counter()-retirement_started
        result.setdefault('costs', {})['native_capture_retirement_s'] = retirement_s
        result['recurring_adaptation_s'] += retirement_s
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
        cold = [v for v in result['canonical'] if v['arm'] == 'BANK' and v['turn'] == 1]
        cold_start = len(cold) == 1 and all(not any(v['read']['coefficients']) for v in cold)
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
                      performance_win_required=False,
                      rank_preserving_updates=bool(evidence) and all(v.get('rank_preserving_updates') for v in evidence),
                      cumulative_protection_fixtures=False, versioned_action_cache=False,
                      cost_profile_binding=getattr(self, 'cost_profile_binding', None),
                      native_cost_profile_available=bool(getattr(self, 'cost_profile_binding', None)))
        from experiments.dflash_correction_bank_v02.qualification import run_fixtures
        fixtures=run_fixtures()
        checks.update(cumulative_protection_fixtures=fixtures['passed'],
                      versioned_action_cache=fixtures['passed'], deterministic_fixtures=fixtures)
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
            retention='Bank snapshots/projections/statistics/protection/anchors/solver records retained; OSD recovery retained until successful episode result/paired checks, then retired explicitly. Old completed OSD comparisons retired with exact manifests.',
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

    def bank_profile(self, body):
        self.engine.collective_rpc('correction_bank_profile')
        # Scheduler diagnostics are not themselves worker-RPC envelopes.  The
        # shared client requires a successful status on every completed action.
        return dict(json.loads((self.out / 'BANK_PROFILE.json').read_text()), status='passed')

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
            canonical=body['canonical'], capture=bool(body['canonical'] and arm == 'BANK'), turn=body['turn'])
        write_json(self.out / 'BANK_PENDING.json', declaration)
        result = super().generate(body)
        document = json.loads(base.checked_binding(result['trajectory']).read_text())
        result['read'] = document.get('correction_bank_read')
        result['bank_head_capture'] = document.get('bank_head_capture')
        result['correction_bank_cycle_observations'] = document.get('correction_bank_cycle_observations', [])
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
        from experiments.dflash_correction_bank_v02.training import BankTrainingService
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
                    allowed = {'train', 'bank_prepare', 'bank_feedback'} if config['kind'] == 'train' else {'generate', 'publish', 'flush', 'bank_activate', 'bank_profile'}
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
