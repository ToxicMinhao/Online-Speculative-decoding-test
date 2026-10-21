"""CPU fake-runtime qualification only; never initializes a real model/GPU."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from . import native as n, storage
from .test_storage import groups_fixture


def checkpoints(tmp_path):
    released = tmp_path / 'released'
    retained = tmp_path / 'original/g/checkpoints/U2'
    released.mkdir()
    retained.mkdir(parents=True)
    result = {}
    for label, path, physical in [('F', released, 'F'), ('S8', retained, 'U2')]:
        (path / 'config.json').write_text('{}')
        (path / 'model.safetensors').write_bytes(label.encode())
        files = {p.name: dict(bytes=p.stat().st_size, sha256=n.sha(p)) for p in path.iterdir()}
        result[label] = dict(path=str(path), files=files, checkpoint_id=physical,
            checkpoint_identity_sha256=n.digest(files))
    config = dict(related_instance_checkpoints=result, released=result['F'], episode_id='g')
    return config, result


def children_fixture():
    return [dict(request_id='new%d' % i, parent_request_id='parent%d' % i,
        group_id='g', domain='gsm8k', role='related_instance', text='New math %d' % i,
        input_ids=[10, 11 + i], input_ids_sha256=n.digest([10, 11 + i])) for i in range(2)]


def group_fixture(cps=None):
    group = dict(group_id='g', domain='gsm8k', status='complete',
        feedback=[dict(request_id='parent%d' % i) for i in range(4)],
        evaluation_requests=[dict(request_id='heldout%d' % i) for i in range(4)],
        checkpoints=cps or {'F': {'checkpoint_identity_sha256': 'f' * 64},
            'S8': {'checkpoint_identity_sha256': '8' * 64}}, children=children_fixture())
    group['group_payload_sha256'] = n.digest(group)
    return group


def reseal(group):
    group['group_payload_sha256'] = n.digest({k: v for k, v in group.items() if k != 'group_payload_sha256'})


def backend_fixture(tmp_path):
    backend = n.NativeBackend({})
    backend.episode_id, backend.out = 'g', tmp_path
    backend.group = group_fixture()
    backend.config = {'storage_budget': {'request_artifact_bound_bytes': {c['request_id']: 2**20
        for c in backend.group['children']}}}
    backend._check_artifact_capacity = lambda: {'used_bytes': 0, 'bound_bytes': 2**30}
    return backend


def generated_fixture(tmp_path, child, label='F', *, output=None):
    output = output or [20, 21, 128009]
    trace = dict(input_token_ids=child['input_ids'], final_output_token_ids=output,
        reconciliation={'status': 'passed'}, events=[dict(kind='target_prefill', committed_token_ids=[output[0]]),
        dict(kind='verification', output_offset=1, draft_proposed_token_ids=[output[1], 99],
            rejection_boundary_proposal_index=1, committed_token_ids=output[1:],
            committed_provenance=['accepted_draft', 'target_correction'], stop_clipped=False)])
    path = tmp_path / ('trace_' + child['request_id'] + '_' + label + '.json')
    n.write_json(path, trace)
    return dict(status='passed', input_ids_sha256=child['input_ids_sha256'], output_ids=output,
        text='synthetic output', accepted_draft_tokens=1, verification_calls=1, verification_committed_tokens=2,
        finish_reason={'type': 'stop', 'matched': 128009}, feedback=None, capture_costs=None,
        trajectory=n.base.binding(path), checkpoint_identity_sha256=('f' if label == 'F' else '8') * 64,
        native_generation_s=.1, request_s=.2, startup_s=.3, activation_s=0., native_activation_s=0., cache_maintenance_s=.01)


def test_exact_own_original_checkpoint_and_no_relabel_or_substitution(tmp_path, monkeypatch):
    config, cps = checkpoints(tmp_path)
    monkeypatch.setattr(n, 'ORIGINAL_NATIVE_ROOT', tmp_path / 'original')
    assert n.admit_checkpoint(config, cps['F']) == 'F'
    assert n.admit_checkpoint(config, cps['S8']) == 'S8'
    other = copy.deepcopy(cps['S8'])
    other['checkpoint_id'] = 'S8'
    with pytest.raises(ValueError, match='exact bound'): n.admit_checkpoint(config, other)
    config['episode_id'] = 'another'
    with pytest.raises(ValueError, match='different original group'): n.admit_checkpoint(config, cps['S8'])


def test_checkpoint_file_mutation_is_rejected(tmp_path, monkeypatch):
    config, cps = checkpoints(tmp_path)
    monkeypatch.setattr(n, 'ORIGINAL_NATIVE_ROOT', tmp_path / 'original')
    (Path(cps['S8']['path']) / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError): n.admit_checkpoint(config, cps['S8'])


@pytest.mark.parametrize('change', ['old_id', 'role', 'parent', 'history', 'tokens', 'domain', 'context'])
def test_child_isolation_cannot_be_relaxed_by_resealing(change):
    group = group_fixture()
    child = group['children'][0]
    if change == 'old_id': child['request_id'] = 'parent0'
    if change == 'role': child['role'] = 'feedback'
    if change == 'parent': child['parent_request_id'] = 'foreign'
    if change == 'history': child['messages'] = [{'role': 'assistant', 'content': 'parent answer'}]
    if change == 'tokens': child['input_ids'] = [999]
    if change == 'domain': child['domain'] = 'sharegpt'
    if change == 'context':
        child['input_ids'] = [1] * 31736
        child['input_ids_sha256'] = n.digest(child['input_ids'])
    reseal(group)
    with pytest.raises(ValueError): n._validate_children(group)


def test_exact_four_new_request_order_and_native_metrics(tmp_path):
    backend = backend_fixture(tmp_path)
    children = backend.group['children']
    calls = []
    def serve(owner, ids, checkpoint, capture, *, turn_index):
        calls.append((checkpoint, ids, capture, turn_index))
        child = children[(turn_index - 1) % 2]
        return generated_fixture(tmp_path, child, 'F' if checkpoint == 'F' else 'S8')
    with patch.object(n.base.NativeBackend, 'serve', serve):
        rows = [backend.serve_request(child, label) for label in ('F', 'S8') for child in children]
        with pytest.raises(ValueError, match='no repeats'): backend.serve_request(children[0], 'S8')
    assert calls == [(label, children[i % 2]['input_ids'], False, i + 1)
        for i, label in enumerate(('F', 'F', 'U2', 'U2'))]
    assert len(backend.attempted) == len(backend.returned) == 4
    for row in rows:
        assert row['role'] == row['population'] == 'related_instance'
        assert row['parent_request_id'].startswith('parent') and row['capture'] is False
        assert row['feedback_capture_enabled'] is False and row['feedback'] is None
        assert row['rounds'][0]['accepted_draft_tokens'] == 1
        assert row['rounds'][0]['absolute_token_offset'] == 3
        assert row['initial_output_tokens'] == 1 and row['verification_committed_tokens'] == 2
        assert row['eos_hit'] and not row['cap_hit'] and row['serving_contract'] == n.SERVING_CONTRACT
        assert n.base.checked_binding(row['raw_native_receipt']).is_file()


def test_out_of_order_and_modified_child_never_enter_generation(tmp_path):
    backend = backend_fixture(tmp_path)
    with patch.object(n.base.NativeBackend, 'serve') as generate:
        with pytest.raises(ValueError, match='sealed F'): backend.serve_request(backend.group['children'][1], 'F')
        changed = dict(backend.group['children'][0], text='different')
        with pytest.raises(ValueError, match='sealed F'): backend.serve_request(changed, 'F')
        with pytest.raises(ValueError, match='sealed F'): backend.serve_request(backend.group['children'][0], 'S8')
        assert generate.call_count == 0 and not backend.attempted


@pytest.mark.parametrize('failure', ['nonfinite', 'checkpoint', 'trace', 'capture'])
def test_returned_invalid_receipt_is_durable_and_cannot_retry(tmp_path, failure):
    backend = backend_fixture(tmp_path)
    child = backend.group['children'][0]
    value = generated_fixture(tmp_path, child)
    if failure == 'nonfinite': value['native_generation_s'] = float('nan')
    if failure == 'checkpoint': value['checkpoint_identity_sha256'] = 'wrong'
    if failure == 'trace': value['verification_calls'] = 2
    if failure == 'capture': value['feedback'] = {'forbidden': True}
    with patch.object(n.base.NativeBackend, 'serve', return_value=value) as generate:
        with pytest.raises(ValueError) as error: backend.serve_request(child, 'F')
        assert error.value.native_partial['generation_returned'] is True
        assert error.value.native_partial['raw_binding']['path'] == str(tmp_path / 'RAW_NATIVE_01.json')
        raw = json.loads((tmp_path / 'RAW_NATIVE_01.json').read_text())
        assert raw['output_ids'] == value['output_ids']
        if failure == 'nonfinite': assert raw['native_generation_s']['nonfinite_float'] == 'nan'
        with pytest.raises(ValueError, match='unpoisoned'): backend.serve_request(child, 'F')
        assert generate.call_count == 1
    assert len(backend.attempted) == len(backend.returned) == 1


def test_unknown_rpc_failure_remains_unknown_and_no_retry(tmp_path):
    backend = backend_fixture(tmp_path)
    child = backend.group['children'][0]
    with patch.object(n.base.NativeBackend, 'serve', side_effect=RuntimeError('worker lost')):
        with pytest.raises(RuntimeError) as error: backend.serve_request(child, 'F')
    assert not hasattr(error.value, 'native_partial')
    assert backend.poisoned and len(backend.attempted) == 1 and not backend.returned


def test_no_historical_serve_training_episode_or_other_worker_paths():
    backend = n.NativeBackend({})
    for name in ('serve', 'train_event', 'start_episode'):
        with pytest.raises(ValueError): getattr(backend, name)()
    for kind in ('train', 'ar'):
        with pytest.raises(ValueError): backend._ensure_worker(kind)
    assert not hasattr(n, 'TrainerService') and not hasattr(n, 'ReadOnlyLossService')


def service_fixture():
    service = n.ServingService.__new__(n.ServingService)
    group = group_fixture()
    cps = {'F': {'checkpoint_id': 'F'}, 'S8': {'checkpoint_id': 'U2'}}
    service.config = dict(original_group=group, related_instance_checkpoints=cps)
    service.active = cps['F']
    service.attempted_children, service.poisoned = 0, False
    return service


def test_worker_admits_exact_child_only_and_no_extra_warmup():
    service = service_fixture()
    seen = []
    service.native_generate = lambda body: (seen.append(copy.deepcopy(body)) or dict(feedback=None, capture_costs=None))
    for ordinal, label in enumerate(('F', 'F', 'U2', 'U2')):
        child = service.config['original_group']['children'][ordinal % 2]
        service.active = service.config['related_instance_checkpoints']['F' if label == 'F' else 'S8']
        service.generate(dict(input_ids=child['input_ids'], checkpoint_id=label, capture=False,
            turn=ordinal + 1, repeat=None))
        assert service.current_child == child
    assert len(seen) == 4
    with pytest.raises(ValueError, match='No extra'): service.generate(dict(capture=False, repeat=None))


@pytest.mark.parametrize('change', ['capture', 'repeat', 'wrong_tokens', 'wrong_checkpoint', 'wrong_turn', 'wrong_active'])
def test_worker_rejects_bad_request_before_native_generation(change):
    service = service_fixture()
    body = dict(input_ids=service.config['original_group']['children'][0]['input_ids'],
        checkpoint_id='F', capture=False, repeat=None, turn=1)
    if change == 'capture': body['capture'] = True
    if change == 'repeat': body['repeat'] = 1
    if change == 'wrong_tokens': body['input_ids'] = [999]
    if change == 'wrong_checkpoint': body['checkpoint_id'] = 'U2'
    if change == 'wrong_turn': body['turn'] = 2
    if change == 'wrong_active': service.active = {'checkpoint_id': 'F', 'wrong': True}
    with patch.object(service, 'native_generate') as generate:
        with pytest.raises(ValueError): service.generate(body)
        assert generate.call_count == 0


def test_worker_failure_poisoning_preserves_attempt_count():
    service = service_fixture()
    body = dict(input_ids=service.config['original_group']['children'][0]['input_ids'],
        checkpoint_id='F', capture=False, repeat=None, turn=1)
    with patch.object(service, 'native_generate', side_effect=RuntimeError('native fail')) as generate:
        with pytest.raises(RuntimeError): service.generate(body)
        with pytest.raises(ValueError): service.generate(body)
        assert generate.call_count == 1 and service.poisoned and service.attempted_children == 1


def test_publish_preserves_cold_cache_and_both_helper_checks(tmp_path, monkeypatch):
    config, cps = checkpoints(tmp_path)
    monkeypatch.setattr(n, 'ORIGINAL_NATIVE_ROOT', tmp_path / 'original')
    service = n.ServingService.__new__(n.ServingService)
    service.config, service.out = config, tmp_path
    receipt = dict(status='passed', success=True, helper_matches_current_layers=True,
        helper_norm_matches_current_layers=True, cache_after={'tree_total_size': 0})
    publications = []
    service.bridge = SimpleNamespace(_two_update_admitted=set(),
        publish=lambda body: (publications.append(body) or receipt), state=lambda: {'draft_model_path': cps['S8']['path']})
    assert service.publish(cps['S8'])['logical_checkpoint_id'] == 'S8'
    assert publications == [{'model_path': cps['S8']['path'], 'reset': True}]
    for key in ('helper_matches_current_layers', 'helper_norm_matches_current_layers'):
        changed = dict(receipt, **{key: False})
        service.bridge.publish = lambda body: changed
        with pytest.raises(ValueError, match='helper refresh'): service.publish(cps['S8'])


def test_finish_checks_checkpoint_and_cleanup_and_preserves_partial_counts(tmp_path, monkeypatch):
    config, cps = checkpoints(tmp_path)
    monkeypatch.setattr(n, 'ORIGINAL_NATIVE_ROOT', tmp_path / 'original')
    registry = tmp_path / 'registry.json'
    registry.write_text('{}')
    tokenizer = tmp_path / 'tokenizer'
    tokenizer.mkdir()
    (tokenizer / 'tokenizer.json').write_text('synthetic')
    monkeypatch.setattr(n.framing, 'TOKENIZER_SHA256', {'tokenizer.json': n.sha(tokenizer / 'tokenizer.json')})
    config.update(registry_path=str(registry), registry_sha256=n.sha(registry), tokenizer_path=str(tokenizer))
    backend = n.NativeBackend(config)
    backend.config, backend.episode_id, backend.out = config, 'g', tmp_path
    backend.target = {'checkpoint_files': {}, 'model_path': str(tmp_path)}
    backend.attempted, backend.returned = [1, 2], [1]
    backend.worker = object()
    with patch.object(n, 'close_owned_worker', return_value={'status': 'passed', 'shutdown_s': .1}):
        receipt = backend.finish_group()
    assert receipt['status'] == 'passed' and receipt['checkpoint_weights_unchanged']
    assert receipt['training_quiescent'] and receipt['optimizer_steps'] == receipt['response_kl_evaluations'] == 0
    assert (receipt['native_attempted'], receipt['native_returned'], receipt['native_unknown'], receipt['native_unattempted']) == (2, 1, 1, 2)
    assert json.loads((tmp_path / 'NATIVE_FINISH.json').read_text()) == receipt


@pytest.mark.parametrize('failure', ['weights', 'cleanup'])
def test_finish_failure_is_durable_and_attached_to_exception(tmp_path, monkeypatch, failure):
    config, cps = checkpoints(tmp_path)
    monkeypatch.setattr(n, 'ORIGINAL_NATIVE_ROOT', tmp_path / 'original')
    registry = tmp_path / 'registry.json'
    registry.write_text('{}')
    config.update(registry_path=str(registry), registry_sha256=n.sha(registry), tokenizer_path=str(tmp_path))
    monkeypatch.setattr(n.framing, 'TOKENIZER_SHA256', {})
    backend = n.NativeBackend(config)
    backend.config, backend.episode_id, backend.out = config, 'g', tmp_path
    backend.target = {'checkpoint_files': {}, 'model_path': str(tmp_path)}
    if failure == 'weights': (Path(cps['S8']['path']) / 'model.safetensors').write_bytes(b'mutated')
    if failure == 'cleanup': backend.worker = object()
    with patch.object(n, 'close_owned_worker', side_effect=ValueError('surviving process group')):
        with pytest.raises(ValueError, match='mutation or owned-worker') as error: backend.finish_group()
    receipt = error.value.native_finish
    assert receipt['status'] == 'failed' and receipt['errors']
    assert json.loads((tmp_path / 'NATIVE_FINISH.json').read_text()) == receipt
    if failure == 'weights': assert receipt['checkpoint_weights_unchanged'] is False


def test_storage_preflight_runtime_binding_before_any_mutation(tmp_path, monkeypatch):
    config = dict(runtime_source_sha256={'test': 'hash'}, serving_contract=n.SERVING_CONTRACT,
        output_root=str(tmp_path / 'native'), controller_root=str(tmp_path / 'controller'))
    config['storage_budget'] = storage.build_storage_plan(groups_fixture(), config)
    monkeypatch.setattr(n, 'preflight_paths', lambda *args: {'status': 'passed'})
    assert n.storage_preflight(config)['status'] == 'passed'
    config['runtime_source_sha256'] = {'test': 'different'}
    with pytest.raises(ValueError, match='different runtime'): n.storage_preflight(config)


def test_cpu_import_is_private_and_worker_entrypoint_is_new_request_only():
    from experiments.dflash_two_update import native as historical
    cfg = dict(workspace_root='/tmp/root', conda_bootstrap='/tmp/conda.sh',
        conda_prefix='/tmp/env', python_executable='/tmp/env/bin/python')
    assert n.worker_command(cfg)[-1] == str(n.HERE / 'native.py')
    assert historical.OUTPUT_CAP == 256 and n.base.OUTPUT_CAP == 1024
    assert historical.NativeBackend.serve is not n.base.NativeBackend.serve
    assert 'generate' in n.ServingService.native_generate.__code__.co_names
    assert 'current_child' in n.ServingService.native_generate.__code__.co_names


def test_source_check_preserves_verified_derived_pin_guards():
    config = dict(related_instance_source_sha256=n.source_pins())
    with patch.object(n.kl, 'check_sources') as qualified:
        n.check_sources(config)
        assert qualified.call_count == 1 and config['diagnostic_source_sha256'] == n.kl.source_pins()
    config['diagnostic_source_sha256'] = {}
    with pytest.raises(ValueError, match='diagnostic source changed'): n.check_sources(config)


def test_qualified_base_path_flushes_every_request_and_publishes_once(tmp_path):
    backend = backend_fixture(tmp_path)
    backend.checkpoints = {'F': {'checkpoint_id': 'F'}, 'U2': {'checkpoint_id': 'U2'}}
    backend.active_checkpoint = 'F'
    backend._ensure_worker = lambda kind: 0.
    seen = []
    def call(action, body):
        seen.append((action, copy.deepcopy(body)))
        if action == 'flush': return {'seconds': .01}
        if action == 'publish': return {'status': 'passed', 'activation_s': .02}
        child = backend.group['children'][(body['turn'] - 1) % 2]
        return generated_fixture(tmp_path, child, 'F' if body['checkpoint_id'] == 'F' else 'S8')
    backend.worker = SimpleNamespace(call=call)
    for label in ('F', 'S8'):
        for child in backend.group['children']:
            backend.serve_request(child, label)
    assert [action for action, _ in seen] == ['flush', 'generate', 'flush', 'generate',
        'publish', 'flush', 'generate', 'flush', 'generate']
    assert seen[4][1] == {'checkpoint_id': 'U2'}
    for action, body in seen:
        if action == 'generate':
            assert body['capture'] is False and body['repeat'] is None
            assert set(body) == {'input_ids', 'checkpoint_id', 'capture', 'turn', 'repeat'}
            assert not any(k in body for k in ('parent_answer', 'output_ids', 'history'))


def test_private_generate_metadata_changes_only_identity_and_uses_normal_bridge(tmp_path):
    service = n.ServingService.__new__(n.ServingService)
    child = children_fixture()[0]
    service.current_child, service.out, service.serial = child, tmp_path, 0
    service.active = dict(checkpoint_id='F', checkpoint_identity_sha256='f' * 64)
    service.config = dict(kind='sd', episode_id='g', target=dict(identity_sha256='t' * 64, checkpoint_files={}),
        public_source_sha256={})
    generated = generated_fixture(tmp_path, child)
    trace_path = Path(generated['trajectory']['path'])
    trace = json.loads(trace_path.read_text())
    trace['reconciliation'].update(input_ids_sha256=n.digest(child['input_ids']),
        output_ids_sha256=n.digest(generated['output_ids']), verification_rounds=1, actual_accepted_draft_tokens=1)
    n.write_json(trace_path, trace)
    value = dict(output_ids=generated['output_ids'], text='natural native output',
        meta_info=dict(cached_tokens=0, spec_verify_ct=1, finish_reason=generated['finish_reason']),
        _trajectory=dict(relative_path=trace_path.name, sha256=n.sha(trace_path), native_generation_s=.1))
    calls = []
    service.bridge = SimpleNamespace(generate=lambda body: (calls.append(body) or value))
    result = service.native_generate(dict(input_ids=child['input_ids'], capture=False, checkpoint_id='F', turn=1, repeat=None))
    declaration = json.loads((tmp_path / 'FEATURE_PENDING.json').read_text())
    assert declaration['record_id'] == declaration['prompt_id'] == child['request_id']
    assert declaration['role'] == 'related_instance' and declaration['domain'] == child['domain']
    assert declaration['capture'] is False
    assert result['output_ids'] == generated['output_ids'] and result['feedback'] is None
    assert calls[0]['input_ids'] == child['input_ids']
    assert calls[0]['sampling_params'] == dict(temperature=0, top_p=1, top_k=1, max_new_tokens=1024, ignore_eos=False)
    assert calls[0]['trace_context']['conversation_id'] == child['request_id']
    assert not any(k in calls[0] for k in ('output_ids', 'parent_answer', 'history'))
    assert (tmp_path / 'RAW_RESPONSE_000001.json').is_file()


def test_start_group_validates_exact_standalone_tokens_without_worker_or_warmup(tmp_path, monkeypatch):
    config, cps = checkpoints(tmp_path)
    monkeypatch.setattr(n, 'ORIGINAL_NATIVE_ROOT', tmp_path / 'original')
    group = group_fixture(cps)
    registry = tmp_path / 'registry.json'
    target = dict(target_id='T0', model_path=str(tmp_path / 'target'), checkpoint_files={}, identity_sha256='t' * 64)
    n.write_json(registry, dict(targets=[target], released_draft=dict(path=cps['F']['path'],
        checkpoint_files=cps['F']['files']), tokenizer_path=str(tmp_path / 'tokenizer')))
    config = dict(output_root=str(tmp_path / 'new_native'), controller_root=str(tmp_path / 'controller'),
        workspace_root=str(tmp_path), serving_contract=n.SERVING_CONTRACT, model_bindings=n.base.binding(registry),
        runtime_source_sha256={'runtime': 'pinned'}, storage_budget={
            'group_payload_sha256': {'g': group['group_payload_sha256']}})
    monkeypatch.setattr(n, 'storage_preflight', lambda cfg: {'status': 'passed'})
    monkeypatch.setattr(n, 'check_sources', lambda cfg: None)
    monkeypatch.setattr(n.base, 'check_environment', lambda cfg: None)
    monkeypatch.setattr(n.base, 'check_dependencies', lambda root: dict(runtime_source_sha256=config['runtime_source_sha256'],
        qualified_model_bindings={'sha256': n.sha(registry)}))
    monkeypatch.setattr(n.framing, 'load_pinned_tokenizer', lambda path: 'pinned_tokenizer')
    encoded = []
    def encode(tokenizer, messages):
        encoded.append((tokenizer, copy.deepcopy(messages)))
        return next(c['input_ids'] for c in group['children'] if c['text'] == messages[0]['content'])
    monkeypatch.setattr(n.framing, 'encode_history', encode)
    backend = n.NativeBackend(config)
    with patch.object(n.base, '_WorkerClient', side_effect=AssertionError('No model init at start_group')):
        receipt = backend.start_group(group)
    assert receipt['status'] == 'passed' and receipt['warmup_generations'] == 0 and not receipt['model_worker_started']
    assert encoded == [('pinned_tokenizer', [{'role': 'user', 'content': child['text']}]) for child in group['children']]
    assert backend.worker is backend.trainer is None and not backend.attempted
    assert backend.checkpoints == {'F': cps['F'], 'U2': cps['S8']}
    assert backend.config['effective_limits']['effective_prompt_tokens'] == 31735
    assert backend.config['original_group'] == group
    assert (Path(config['output_root']) / 'g/NATIVE_CONFIG.json').is_file()
    with pytest.raises(ValueError, match='Close previous'): backend.start_group(group)


def test_worker_reply_saved_before_rpc_validation_failure_is_returned_invalid(tmp_path):
    backend = backend_fixture(tmp_path)
    child = backend.group['children'][0]
    worker_out = tmp_path / 'worker'
    worker_out.mkdir()
    backend.worker = SimpleNamespace(out=worker_out)
    raw = {'output_ids': [20, 128009], 'text': 'returned but fails reconciliation'}
    n.write_json(worker_out / 'RAW_RESPONSE_000001.json', raw)
    n.write_json(worker_out / 'FEATURE_PENDING.json', dict(record_id=child['request_id'], draft_version='F', capture=False))
    with patch.object(n.base.NativeBackend, 'serve', side_effect=RuntimeError('RPC validation failed')):
        with pytest.raises(RuntimeError) as error: backend.serve_request(child, 'F')
    assert error.value.native_partial['generation_returned'] is True
    assert error.value.native_partial['stage'] == 'worker_native_validation'
    assert error.value.native_partial['raw_receipt'] == raw
    assert len(backend.attempted) == len(backend.returned) == 1


def test_unbound_worker_partial_is_not_mistaken_for_current_reply(tmp_path):
    backend = backend_fixture(tmp_path)
    child = backend.group['children'][0]
    worker_out = tmp_path / 'worker'
    worker_out.mkdir()
    backend.worker = SimpleNamespace(out=worker_out)
    n.write_json(worker_out / 'RAW_RESPONSE_000001.json', {'output_ids': [99]})
    n.write_json(worker_out / 'FEATURE_PENDING.json', dict(record_id='another_child', draft_version='F', capture=False))
    with patch.object(n.base.NativeBackend, 'serve', side_effect=RuntimeError('RPC failure')):
        with pytest.raises(RuntimeError) as error: backend.serve_request(child, 'F')
    assert not hasattr(error.value, 'native_partial') and not backend.returned
    assert 'identity differs' in error.value.native_partial_recovery_error


def test_unicode_evidence_group_and_runtime_payload_use_canonical_seal_digest(monkeypatch):
    from . import evidence
    from .test_evidence import make_bundle, make_construction, make_renderer
    bundle = make_bundle()
    bundle['groups'][0]['feedback'][0]['retained_note'] = '原始資料 한국어 café'
    construction = make_construction(bundle)
    construction['children'][0]['text'] += ' 新問題 한국어'
    sealed = evidence.seal_children(bundle, construction, render=make_renderer(bundle, construction))
    groups = evidence.bind_children(bundle, sealed)
    assert n._validate_children(groups[0]) == groups[0]['children']
    assert n.digest({k: v for k, v in groups[0].items() if k != 'group_payload_sha256'}) != groups[0]['group_payload_sha256']
    config = dict(workspace_root='/研究/실험', output_root='/new-native', controller_root='/new-controller',
        serving_contract=n.SERVING_CONTRACT)
    config['storage_budget'] = storage.build_storage_plan(groups, config)
    monkeypatch.setattr(n, 'preflight_paths', lambda *args: {'status': 'passed'})
    assert n.storage_preflight(config)['status'] == 'passed'


def test_real_native_body_saves_nonfinite_bridge_reply_then_poisoning_is_visible(tmp_path):
    service = service_fixture()
    child = service.config['original_group']['children'][0]
    service.out, service.serial = tmp_path, 0
    service.config.update(kind='sd', episode_id='g',
        target=dict(identity_sha256='t' * 64, checkpoint_files={}), public_source_sha256={})
    service.config['related_instance_checkpoints']['F']['checkpoint_identity_sha256'] = 'f' * 64
    value = generated_fixture(tmp_path, child)
    path = Path(value['trajectory']['path'])
    trace = json.loads(path.read_text())
    trace['reconciliation'].update(input_ids_sha256=n.digest(child['input_ids']),
        output_ids_sha256=n.digest(value['output_ids']), verification_rounds=1, actual_accepted_draft_tokens=1)
    n.write_json(path, trace)
    raw = dict(output_ids=value['output_ids'], text='known returned output',
        meta_info=dict(cached_tokens=0, spec_verify_ct=1, finish_reason=value['finish_reason']),
        _trajectory=dict(relative_path=path.name, sha256=n.sha(path), native_generation_s=float('nan')))
    service.bridge = SimpleNamespace(generate=lambda body: raw)
    body = dict(input_ids=child['input_ids'], capture=False, checkpoint_id='F', turn=1, repeat=None)
    with pytest.raises(ValueError, match='Nonfinite'): service.generate(body)
    saved = json.loads((tmp_path / 'RAW_RESPONSE_000001.json').read_text())
    assert saved['output_ids'] == raw['output_ids']
    assert saved['_trajectory']['native_generation_s']['nonfinite_float'] == 'nan'
    assert not (tmp_path / 'RAW_RESPONSE_000001.json.partial').exists()
    assert service.poisoned and service.attempted_children == 1
    backend = backend_fixture(tmp_path)
    backend.worker = SimpleNamespace(out=tmp_path)
    with patch.object(n.base.NativeBackend, 'serve', side_effect=RuntimeError('HTTP nonfinite failure')):
        with pytest.raises(RuntimeError) as error: backend.serve_request(child, 'F')
    assert error.value.native_partial['generation_returned'] is True
    assert error.value.native_partial['raw_receipt'] == saved
