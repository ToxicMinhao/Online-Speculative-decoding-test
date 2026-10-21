"""CPU wrapper qualification; no model/native GPU execution is performed."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from . import evidence as ev, native as n
from .test_evidence import groups_fixture, pools_fixture


def checkpoint(tmp_path):
    pool = pools_fixture()[0]
    out = tmp_path / pool['pool_id']
    path = out / 'checkpoints/P8'
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    (path / 'model.safetensors').write_bytes(b'P8')
    files = {p.name: dict(bytes=p.stat().st_size, sha256=n.sha(p)) for p in path.iterdir()}
    cp = dict(path=str(path), files=files, checkpoint_id='P8', checkpoint_identity_sha256=n.digest(files),
        pool_id=pool['pool_id'], domain=pool['domain'], group_ids=pool['group_ids'],
        optimizer_steps=8, learning_rate=3e-4)
    config = dict(episode_out=str(out), episode_id=pool['pool_id'], original_pool=pool)
    return config, cp


def materialized_pool(tmp_path, *, colliding_request_ids=False):
    groups = groups_fixture(colliding_request_ids=colliding_request_ids)
    for group in groups:
        for index, record in enumerate(group['support']):
            path = tmp_path / f"{group['group_id']}_support_{index}.json"
            saved = {key: value for key, value in record.items() if key != 'metadata_binding'}
            n.write_json(path, saved)
            record['metadata_binding'] = n.base.binding(path)
    return ev.build_pools(groups)[0]


@pytest.mark.parametrize('field,value', [
    ('checkpoint_id', 'F'), ('checkpoint_id', 'S8'), ('checkpoint_id', 'L8'), ('checkpoint_id', 'U2'),
    ('pool_id', 'p8_other'), ('domain', 'other'), ('group_ids', ['foreign', 'group']),
    ('optimizer_steps', 2), ('learning_rate', 3e-5)])
def test_checkpoint_requires_own_p8_pool_domain_group_membership_steps_lr(tmp_path, field, value):
    config, cp = checkpoint(tmp_path)
    assert n.admit_checkpoint(config, cp) == 'P8'
    with pytest.raises(ValueError): n.admit_checkpoint(config, {**cp, field: value})


def test_checkpoint_requires_exact_path_files_and_group_order(tmp_path):
    config, cp = checkpoint(tmp_path)
    for key, value in [('path', str(tmp_path / 'checkpoints/P8')),
                       ('group_ids', list(reversed(cp['group_ids']))),
                       ('checkpoint_identity_sha256', 'f' * 64)]:
        with pytest.raises(ValueError): n.admit_checkpoint(config, {**cp, key: value})
    (Path(cp['path']) / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'): n.admit_checkpoint(config, cp)


def test_storage_includes_three_weights_artifact_evidence_both_destinations_and_reserve(tmp_path):
    from .storage import build_storage_plan
    bound = n.campaign_storage_bound()  # Historical floor, not execution admission.
    budget = build_storage_plan(pools_fixture())
    required = budget['same_filesystem_required_bytes']
    config = dict(output_root=str(tmp_path/'nested/native'), controller_root=str(tmp_path/'new'),
                  storage_budget=budget)
    with patch.object(n.shutil, 'disk_usage', return_value=SimpleNamespace(free=required)):
        receipt = n.storage_preflight(config)
    assert receipt['total_required_bytes'] == required
    assert receipt['same_filesystem'] and receipt['status'] == 'passed'
    assert budget['final_checkpoints'] == 3
    assert budget['final_checkpoint_bytes'] == 3*(1048626432*2 + 64*2**20)
    assert budget['export_staging_overlap_bytes'] == 1048626432*2 + 64*2**20
    assert budget['optimizer_state_exports'] == budget['intermediate_checkpoints'] == 0
    assert budget['controller_retrieval_staging_bytes'] > 0
    assert bound['planned_write_bytes'] + n.MIN_RESERVE_BYTES < required
    assert not Path(config['output_root']).exists()
    with patch.object(n.shutil, 'disk_usage', return_value=SimpleNamespace(free=required-1)):
        with pytest.raises(ValueError, match='exceed free'): n.storage_preflight(config)
    for reserve in (1, 8.0*2**30, True):
        with pytest.raises(ValueError, match='8 GiB'):n.storage_preflight({**config,'reserve_bytes':reserve})
    with pytest.raises(ValueError, match='native and controller'):
        n.storage_preflight({'output_root':str(tmp_path)})


def start_fixture(tmp_path, monkeypatch):
    pool = materialized_pool(tmp_path)
    from .storage import build_storage_plan
    storage = build_storage_plan([pool,*pools_fixture()[1:]])
    frozen = pool['checkpoints']['F']
    target = dict(target_id='T0', model_path='/synthetic/target', identity_sha256='a' * 64, checkpoint_files={})
    registry = tmp_path / 'models.json'
    n.write_json(registry, dict(targets=[target], tokenizer_path='/synthetic/tokenizer',
        released_draft=dict(path=frozen['path'], checkpoint_files=frozen['files'])))
    config = dict(output_root=str(tmp_path / 'outputs'), workspace_root=str(tmp_path),
        storage_budget=storage, serving_contract=dict(n.SERVING_CONTRACT), model_bindings=n.base.binding(registry),
        runtime_source_sha256={'runtime.py': 'a' * 64}, output_cap=1024, block_size=10)
    dependencies = dict(runtime_source_sha256=config['runtime_source_sha256'],
        qualified_model_bindings={'sha256': n.sha(registry)})
    monkeypatch.setattr(n, 'check_sources', lambda _: None)
    monkeypatch.setattr(n.base, 'check_environment', lambda _: None)
    monkeypatch.setattr(n.base, 'check_dependencies', lambda _: dependencies)
    monkeypatch.setattr(n.base, 'verify_checkpoint', lambda _: None)
    monkeypatch.setattr(n.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 2**30))
    return n.NativeBackend(config), pool


def test_start_pool_pins_complete_contract_and_original_support_without_starting_worker(tmp_path, monkeypatch):
    backend, pool = start_fixture(tmp_path, monkeypatch)
    with patch.object(n.base, '_WorkerClient') as worker:
        receipt = backend.start_pool(pool)
    assert worker.call_count == 0 and receipt['model_worker_started'] is False
    assert receipt['seed'] == 17 and receipt['learning_rate'] == 3e-4 and receipt['training_steps'] == 8
    assert backend.episode_id == pool['pool_id'] and backend.pool == pool
    assert backend.pool is not pool and backend.config['original_pool'] == pool
    assert backend.config['tokenizer_path'] == '/synthetic/tokenizer'
    assert backend.config['serving_contract'] == n.SERVING_CONTRACT
    assert backend.config['effective_limits']['input_tokens'] == n.base.INPUT_LIMIT
    assert backend.config['effective_limits']['total_context_tokens'] == 32768
    assert (backend.out / 'NATIVE_CONFIG.json').is_file()
    with pytest.raises(ValueError, match='Close prior'): backend.start_pool(pool)


@pytest.mark.parametrize('key,value', [('concurrency', 2), ('greedy', False), ('temperature', .5),
    ('block_size', 5), ('output_cap', 256), ('context_limit', 16384), ('chunked_prefill_size', 4096),
    ('feedback_capture', True), ('cache_policy', 'retain cache'),
    ('tokenizer_binding', 'unbound tokenizer'), ('runtime_binding', 'unbound runtime')])
def test_start_rejects_each_material_serving_contract_change(tmp_path, monkeypatch, key, value):
    backend, pool = start_fixture(tmp_path, monkeypatch)
    backend.config['serving_contract'][key] = value
    with pytest.raises(ValueError, match='serving contract'): backend.start_pool(pool)


def test_full_prompt_context_envelope_still_prevents_proposal_overflow():
    assert n.SERVING_CONTRACT['input_limit'] == n.base.INPUT_LIMIT
    assert n.SERVING_CONTRACT['context_limit'] == n.base.CONTEXT_LIMIT == 32768
    maximum = 32768 - 1024 - 9
    n.base.check_input([1] * maximum)
    with pytest.raises(ValueError, match='context'): n.base.check_input([1] * (maximum + 1))


def measured_trace(tmp_path, canonical, index):
    output = canonical['output_ids']
    trace = dict(input_token_ids=canonical['input_ids'], final_output_token_ids=output,
        reconciliation={'status': 'passed'}, events=[
            dict(kind='target_prefill', committed_token_ids=[output[0]]),
            dict(kind='verification', output_offset=1, draft_proposed_token_ids=[output[1], 50],
                rejection_boundary_proposal_index=1, committed_token_ids=output[1:],
                committed_provenance=['accepted_draft', 'target_correction'], stop_clipped=False)])
    path = tmp_path / f'trace_{index}.json'
    n.write_json(path, trace)
    return dict(status='passed', output_ids=list(output), verification_calls=1, accepted_draft_tokens=1,
        verification_committed_tokens=2, trajectory=n.base.binding(path),
        finish_reason={'type': 'stop', 'matched': 128009})


def test_native_generates_all_eight_source_and_heldout_with_original_owners_no_capture(tmp_path):
    pool = pools_fixture(colliding_request_ids=True)[0]
    backend = n.NativeBackend({})
    backend.pool, backend.episode_id, backend.training_closed = pool, pool['pool_id'], True
    calls, results = [], []
    def serve(_self, ids, checkpoint_id, capture, *, turn_index):
        calls.append((list(ids), checkpoint_id, capture, turn_index))
        population = 'source' if turn_index <= 8 else 'heldout'
        canonical = pool['reference_pairs'][population][(turn_index - 1) % 8]['F']['native']
        return measured_trace(tmp_path, canonical, turn_index)
    with patch.object(n.base.NativeBackend, 'serve', serve):
        for population in ('source', 'heldout'):
            for pair in pool['reference_pairs'][population]:
                result = backend.serve_request(pair['F']['native'], population)
                results.append(result)
                assert result['group_id'] == pair['group_id']
                assert result['pool_id'] == pool['pool_id'] and result['pool_request_id'] == pair['pool_request_id']
                assert result['request_id'] == pair['request_id'] and result['population'] == population
        with pytest.raises(ValueError, match='no repetitions'):
            backend.serve_request(pool['reference_pairs']['source'][0]['F']['native'], 'source')
    assert len(calls) == len(results) == 16 and len(backend.served) == 16
    assert [call[3] for call in calls] == list(range(1, 17))
    assert all(call[1:3] == ('P8', False) for call in calls)
    assert len({row['request_id'] for row in results}) == 8
    assert len({row['pool_request_id'] for row in results}) == 16
    assert all(row['feedback_capture_enabled'] is False and row['capture'] is False for row in results)
    assert all(row['initial_output_tokens'] == 1 and row['committed_progress_per_call'] == 2 for row in results)
    assert all(row['rounds'][0]['first_rejection_offset'] == 1 and row['eos_hit'] and not row['cap_hit'] for row in results)
    assert all(row['serving_contract'] == n.SERVING_CONTRACT for row in results)


@pytest.mark.parametrize('field,value', [('group_id', 'foreign'), ('role', 'evaluation'),
    ('request_id', 'foreign'), ('input_ids', [1]), ('output_ids', [1])])
def test_serving_rejects_changed_original_identity_or_tokens(field, value):
    pool = pools_fixture()[0]
    backend = n.NativeBackend({})
    backend.pool, backend.episode_id, backend.training_closed = pool, pool['pool_id'], True
    canonical = pool['reference_pairs']['source'][0]['F']['native']
    with pytest.raises(ValueError, match='differs from bound'):
        backend.serve_request({**canonical, field: value}, 'source')
    assert backend.served == set()


def test_native_generation_waits_for_trainer_and_children_to_exit():
    backend = n.NativeBackend({})
    with pytest.raises(ValueError, match='Stop training'): backend.serve_request({}, 'source')
    backend.training_closed, backend.retired_trainer_pid = True, 123
    with patch.object(n.curve, 'process_group_alive', return_value=True):
        with pytest.raises(ValueError, match='child still alive'): backend.serve_request({}, 'source')


def test_serving_only_accepts_real_p8_capture_disabled_and_rejects_capture_results():
    service = n.ServingService.__new__(n.ServingService)
    for body in ({'capture': True, 'checkpoint_id': 'P8'}, {'capture': False, 'checkpoint_id': 'U2'},
                 {'capture': False, 'checkpoint_id': 'S8'}, {'capture': False, 'checkpoint_id': 'L8'}):
        with pytest.raises(ValueError, match='natural P8'): service.generate(body)
    with patch.object(n.ServingService, 'native_generate', return_value={'feedback': None, 'capture_costs': None}) as generate:
        service.generate({'capture': False, 'checkpoint_id': 'P8'})
        assert generate.call_count == 1
    with patch.object(n.ServingService, 'native_generate', return_value={'feedback': {'saved': True}, 'capture_costs': None}):
        with pytest.raises(ValueError, match='capture was active'):
            service.generate({'capture': False, 'checkpoint_id': 'P8'})


def test_real_p8_trace_registration_leaves_historical_method_and_four_response_guards_unchanged():
    original = n.base._ServingService.generate
    assert "{'F': 0, 'U1': 1, 'U2': 2}" in inspect.getsource(original)
    generated = n._p8_generate_method()
    assert ('F', 'P8') in generated.__code__.co_consts and 8 in generated.__code__.co_consts
    assert generated is not original and n.base._ServingService.generate is original
    from experiments.dflash_baseline_learning_curve.training import step_schedule
    with pytest.raises(ValueError, match='four'):
        step_schedule(0, 2, pools_fixture()[0]['support'])
    config = dict(workspace_root='/tmp/root', conda_bootstrap='/tmp/conda.sh',
                  conda_prefix='/tmp/env', python_executable='/tmp/env/bin/python')
    assert n.worker_command(config)[-1] == str(n.HERE / 'native.py')


def test_publication_resets_real_p8_and_checks_helpers_cache_and_outer_timer(tmp_path):
    config, cp = checkpoint(tmp_path)
    calls = []
    receipt = dict(status='passed', success=True, helper_matches_current_layers=True,
        helper_norm_matches_current_layers=True, cache_after={'tree_total_size': 0})
    bridge = SimpleNamespace(_two_update_admitted=set(), publish=lambda body: (calls.append(body) or receipt),
        state=lambda: {'draft_model_path': cp['path']})
    service = n.ServingService.__new__(n.ServingService)
    service.config, service.out, service.bridge = config, tmp_path, bridge
    assert service.publish(cp)['checkpoint_id'] == 'P8'
    assert calls == [{'model_path': cp['path'], 'reset': True}]
    assert service.active == cp and Path(cp['path']) in bridge._two_update_admitted
    receipt['cache_after']['tree_total_size'] = 2
    with pytest.raises(ValueError, match='cache invalidation'): service.publish(cp)
    clock = [0.]
    backend = n.NativeBackend(config)
    backend.checkpoints, backend.active_checkpoint = {'P8': cp}, 'F'
    backend._ensure_worker = lambda _: 0
    def publish(action, body):
        assert action == 'publish' and body == cp
        clock[0] += 1.25
        return dict(status='passed', checkpoint_id='P8', activation_s=.5)
    backend.worker = SimpleNamespace(call=publish)
    with patch.object(n.base.time, 'perf_counter', side_effect=lambda: clock[0]):
        result, again = backend.publish('P8'), backend.publish('P8')
    assert result['activation_s'] == 1.25 and result['native_activation_s'] == .5
    assert backend.totals['activation_s'] == 1.25 and again['activation_s'] == 0


def test_stop_training_checks_process_group_and_cleanup_closes_owned_workers(tmp_path):
    backend = n.NativeBackend({})
    backend.training_started, backend.out, backend.checkpoints = True, tmp_path, {'P8': {}}
    backend.episode_id = 'p8_gsm8k'
    worker = SimpleNamespace(process=SimpleNamespace(pid=123))
    backend.trainer = worker
    with patch.object(n, 'close_owned_worker', return_value=dict(status='passed', shutdown_s=.4)) as close, \
         patch.object(n.curve, 'process_group_alive', return_value=False):
        receipt = backend.stop_training()
    assert close.call_args.args == (worker,) and backend.trainer is None and backend.training_closed
    assert receipt['training_quiescent'] and backend.totals['shutdown_s'] == .4
    assert (tmp_path / 'TRAINING_QUIESCENT.json').is_file()
    backend.worker = SimpleNamespace(name='serving')
    with patch.object(n, 'close_owned_worker', return_value=dict(status='passed', shutdown_s=.2)) as close:
        result = backend.finish_pool()
    assert close.call_count == 1 and backend.worker is None and backend.episode_id is None
    assert result['status'] == 'passed' and result['shutdown_s'] == pytest.approx(.6)


def test_cleanup_failure_is_visible_and_recorded(tmp_path):
    backend = n.NativeBackend({})
    backend.out, backend.episode_id, backend.worker = tmp_path, 'p8_gsm8k', SimpleNamespace()
    with patch.object(n, 'close_owned_worker', side_effect=RuntimeError('child did not exit')):
        with pytest.raises(ValueError, match='failed shutdown'): backend.finish_pool()
    result = json.loads((tmp_path / 'NATIVE_FINISH.json').read_text())
    assert result['status'] == 'failed' and 'child did not exit' in result['errors'][0]


def test_worker_training_failure_preserves_partial_without_registering_p8(tmp_path):
    directory = tmp_path / 'training_worker'
    directory.mkdir()
    n.write_json(directory / 'PARTIAL.json', dict(status='failed', accounting={'optimizer_steps_total': 2},
        losses={'source': [], 'heldout': []}))
    class Worker:
        startup_s = .1
        def __init__(self, *args): pass
        def call(self, action, body):
            assert action == 'train_pool' and body == {}
            raise RuntimeError('worker failed')
    backend = n.NativeBackend({})
    backend.episode_id, backend.out, backend.checkpoints = 'p8_gsm8k', tmp_path, {'F': {}}
    with patch.object(n.base, '_WorkerClient', Worker):
        with pytest.raises(RuntimeError, match='worker failed') as failure: backend.train_pool()
    assert failure.value.diagnostic_partial['accounting']['optimizer_steps_total'] == 2
    assert failure.value.partial_binding['sha256'] == n.sha(directory / 'PARTIAL.json')
    assert 'P8' not in backend.checkpoints


def service_fixture(tmp_path):
    import torch
    pool = materialized_pool(tmp_path)
    service = n.TrainerService.__new__(n.TrainerService)
    service.config = dict(episode_id=pool['pool_id'], original_pool=pool)
    service.out, service.completed = tmp_path, False
    service.draft = torch.nn.Linear(1, 1)
    service.components = SimpleNamespace(embed_tokens=torch.nn.Linear(1, 1), lm_head=torch.nn.Linear(1, 1))
    service.teacher_features = lambda ids: None
    service.check_frozen = lambda: None
    return service, pool


@pytest.fixture
def cuda_accounting(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 100)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 200)


def test_trainer_service_trains_eight_exports_p8_and_evaluates_exact_pool_populations(tmp_path, cuda_accounting):
    service, pool = service_fixture(tmp_path)
    order = []
    cp = dict(checkpoint_id='P8', checkpoint_identity_sha256='a' * 64, export_s=.4)
    def train(rows):
        assert rows == pool['support'] and len(rows) == 8
        assert [r['episode_id'] for r in rows] == pool['group_ids'] * 4
        order.append('train8')
        return dict(optimizer_steps_total=8, response_presentations_total=16, costs={'training_s': 1.})
    service.trainer = SimpleNamespace(train_eight=train,
        bind_p8_checkpoint=lambda value: order.append(('bind', value)), _check_owner=lambda: None)
    service._export_p8 = lambda: (order.append('export_P8') or cp)
    def evaluate(draft, embed, head, teacher, rows, references, **kwargs):
        population = kwargs['population']
        order.append(('loss', population))
        assert kwargs['checkpoint_id'] == 'P8' and kwargs['original_pool'] is pool
        assert kwargs['checkpoint_identity_sha256'] == cp['checkpoint_identity_sha256']
        expected = pool['support'] if population == 'source' else [p['F']['native'] for p in pool['reference_pairs']['heldout']]
        assert rows == expected and references == [p['F']['kl'] for p in pool['reference_pairs'][population]]
        assert len(rows) == len(references) == 8
        return dict(status='passed', response_losses=[dict(phase_seconds={'teacher_preparation_s': .01}) for _ in rows],
            diagnostic_wall_s=.2, optimizer_steps=0)
    with patch('experiments.dflash_feedback_allocation_control.loss.evaluate_responses', side_effect=evaluate):
        result = service.train_pool({})
        with pytest.raises(ValueError, match='One bounded'): service.train_pool({})
    assert order == ['train8', 'export_P8', ('bind', 'a' * 64), ('loss', 'source'), ('loss', 'heldout')]
    assert len(result['losses']['source']) == len(result['losses']['heldout']) == 8
    assert result['checkpoint'] == cp and result['costs']['optimizer_state_save_s'] == 0
    assert result['costs']['source_diagnostic_teacher_preparation_s'] == .08
    assert result['costs']['heldout_diagnostic_teacher_preparation_s'] == .08


def test_metadata_failure_is_durable_before_any_optimizer_work(tmp_path, cuda_accounting):
    service, pool = service_fixture(tmp_path)
    Path(pool['support'][0]['metadata_binding']['path']).write_text('{"tampered":true}')
    calls = []
    service.trainer = SimpleNamespace(optimizer_steps=0, response_presentations=0,
        train_eight=lambda rows: calls.append('training'))
    with pytest.raises(ValueError): service.train_pool({})
    saved = json.loads((tmp_path / 'PARTIAL.json').read_text())
    assert calls == [] and saved['checkpoint'] is None
    assert saved['accounting']['optimizer_steps_total'] == saved['accounting']['response_presentations_total'] == 0
    assert saved['accounting']['status'] == 'failed' and (tmp_path / 'TRAINING_BEGIN.json').is_file()


def test_partial_training_preserves_applied_and_uncommitted_exposure(tmp_path, cuda_accounting):
    service, _ = service_fixture(tmp_path)
    failure = ValueError('training failed')
    failure.training_partial = dict(optimizer_steps=2, response_presentations=4, completed_batches=[{}, {}],
        attempted_response_calls=6, completed_backward_presentations=6, uncommitted_backward_presentations=2)
    def train(rows): raise failure
    service.trainer = SimpleNamespace(train_eight=train)
    with pytest.raises(ValueError, match='training failed'): service.train_pool({})
    saved = json.loads((tmp_path / 'PARTIAL.json').read_text())
    assert saved['accounting']['optimizer_steps_total'] == 2 and saved['accounting']['response_presentations_total'] == 4
    assert saved['accounting']['attempted_response_calls'] == saved['accounting']['completed_backward_presentations'] == 6
    assert saved['accounting']['uncommitted_backward_presentations'] == 2 and saved['checkpoint'] is None


def test_partial_diagnostic_preserves_completed_rows_and_no_heldout_retry(tmp_path, cuda_accounting):
    service, _ = service_fixture(tmp_path)
    service.trainer = SimpleNamespace(train_eight=lambda _: dict(optimizer_steps_total=8),
        bind_p8_checkpoint=lambda _: None, _check_owner=lambda: None)
    service._export_p8 = lambda: dict(checkpoint_id='P8', checkpoint_identity_sha256='a' * 64)
    failure = ValueError('nonfinite loss')
    failure.diagnostic_partial = dict(response_losses=[dict(loss=.1)], bad_value=float('nan'))
    with patch('experiments.dflash_feedback_allocation_control.loss.evaluate_responses', side_effect=failure) as evaluate:
        with pytest.raises(ValueError, match='nonfinite'): service.train_pool({})
    saved = json.loads((tmp_path / 'PARTIAL.json').read_text())
    assert evaluate.call_count == 1 and saved['losses']['source'] == [{'loss': .1}]
    assert saved['losses']['heldout'] == [] and saved['checkpoint']['checkpoint_id'] == 'P8'
    assert saved['failure_detail']['bad_value']['nonfinite_float'] == 'nan'


def test_old_training_paths_and_optimizer_resume_are_unavailable():
    with pytest.raises(ValueError, match='explicit P8'): n.NativeBackend({}).train_event(1, [])
    assert not issubclass(n.TrainerService, n.followup.TrainerService)
    assert not hasattr(n.TrainerService, 'train') and not hasattr(n.TrainerService, '_export')
    text = (n.HERE / 'native.py').read_text()
    assert 'save_checkpoint_state(' not in text and 'load_checkpoint_state(' not in text
