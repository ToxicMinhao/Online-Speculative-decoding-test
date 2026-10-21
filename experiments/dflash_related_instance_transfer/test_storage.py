"""CPU capacity tests using synthetic sealed inputs, never actual model paths."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import storage as s


def groups_fixture():
    groups = []
    for index in range(6):
        gid = 'group_%d' % index
        group = dict(group_id=gid, retained_evidence='synthetic', children=[dict(
            group_id=gid, request_id=gid + '_child_%d' % j, role='related_instance',
            input_ids=[index + 1, j + 1], input_ids_sha256=s.digest([index + 1, j + 1]),
            text='New synthetic question 雪%d' % j) for j in range(2)])
        group['group_payload_sha256'] = s.digest(group)
        groups.append(group)
    return groups


def test_exact_serialized_size_and_no_input_mutation(tmp_path):
    from .native import write_json
    groups = groups_fixture()
    before = copy.deepcopy(groups)
    config = {'nested': {'unicode': '雪😀', 'tokens': list(range(80))}}
    plan = s.build_storage_plan(groups, config)
    assert groups == before
    for group in groups:
        path = tmp_path / (group['group_id'] + '.json')
        write_json(path, dict(config, original_group=group))
        assert plan['serialized_group_config_bytes'][group['group_id']] == path.stat().st_size
    assert plan['runtime_payload_sha256'] == s.digest(config)
    with pytest.raises(ValueError):
        s.serialized_bytes({'nonfinite': float('nan')})


def test_budget_has_actual_requests_configs_full_retrieval_and_no_weights():
    groups = groups_fixture()
    plan = s.build_storage_plan(groups, {'worker_env': {'X': 'a' * 10000}})
    assert plan['maximum_generations'] == 24 and plan['maximum_rounds_per_request'] == 1024
    assert len(plan['request_artifact_bound_bytes']) == 12
    assert plan['native_configuration_bytes'] == 3 * sum(plan['serialized_group_config_bytes'].values())
    assert plan['controller_retrieval_staging_bytes'] == plan['native_write_bytes'] + plan['controller_primary_write_bytes']
    assert plan['total_write_bytes'] == plan['native_write_bytes'] + plan['controller_write_bytes']
    assert plan['same_filesystem_required_bytes'] == plan['total_write_bytes'] + 8 * 2**30
    assert all(plan[k] == 0 for k in ('checkpoint_copies', 'checkpoint_exports', 'optimizer_state_exports',
        'feature_tensor_exports', 'model_downloads'))


def test_larger_real_payloads_increase_both_filesystem_bounds():
    groups = groups_fixture()
    original = s.build_storage_plan(groups)
    groups[0]['retained_evidence'] += 'x' * 100000
    groups[0]['group_payload_sha256'] = s.digest({k: v for k, v in groups[0].items() if k != 'group_payload_sha256'})
    larger = s.build_storage_plan(groups)
    assert larger['native_write_bytes'] == original['native_write_bytes'] + 300000
    assert larger['controller_write_bytes'] > original['controller_write_bytes'] + 1600000


@pytest.mark.parametrize('change', ['group_count', 'duplicate_group', 'seal', 'child_count', 'child_role', 'child_hash', 'duplicate_child'])
def test_storage_rejects_changed_or_incomplete_inputs(change):
    groups = groups_fixture()
    if change == 'group_count': groups.pop()
    if change == 'duplicate_group': groups[1] = copy.deepcopy(groups[0])
    if change == 'seal': groups[0]['retained_evidence'] += 'changed'
    if change == 'child_count': groups[0]['children'].pop()
    if change == 'child_role': groups[0]['children'][0]['role'] = 'feedback'
    if change == 'child_hash': groups[0]['children'][0]['input_ids'][0] += 1
    if change == 'duplicate_child': groups[1]['children'][0]['request_id'] = groups[0]['children'][0]['request_id']
    if change not in ('seal', 'group_count', 'duplicate_group'):
        for group in groups:
            group['group_payload_sha256'] = s.digest({k: v for k, v in group.items() if k != 'group_payload_sha256'})
    with pytest.raises(ValueError): s.build_storage_plan(groups)


def mock_probes(monkeypatch, plan, *, separate=False, shortfall=0):
    same_required = plan['total_write_bytes'] + s.MIN_RESERVE_BYTES
    probes = {
        '/native': dict(destination='/native', existing_ancestor='/', filesystem_device=1,
            free_bytes=(plan['native_write_bytes'] + s.MIN_RESERVE_BYTES if separate else same_required) - shortfall),
        '/controller': dict(destination='/controller', existing_ancestor='/', filesystem_device=2 if separate else 1,
            free_bytes=(plan['controller_write_bytes'] + s.MIN_RESERVE_BYTES if separate else same_required))}
    monkeypatch.setattr(s, '_probe', lambda path: copy.deepcopy(probes[str(path)]))


@pytest.mark.parametrize('separate', [True, False])
def test_capacity_exact_boundary_sums_shared_writes_and_retains_per_device_reserve(monkeypatch, separate):
    plan = s.build_storage_plan(groups_fixture())
    mock_probes(monkeypatch, plan, separate=separate)
    receipt = s.preflight_paths(plan, '/native', '/controller')
    assert receipt['status'] == 'passed' and receipt['same_filesystem'] is not separate
    assert len(receipt['filesystems']) == (2 if separate else 1)
    assert receipt['total_required_bytes'] == plan['total_write_bytes'] + (2 if separate else 1) * s.MIN_RESERVE_BYTES
    assert all(r['free_bytes'] == r['required_bytes'] for r in receipt['filesystems'])


@pytest.mark.parametrize('separate', [True, False])
def test_one_byte_shortfall_returns_complete_failure_receipt(monkeypatch, separate):
    plan = s.build_storage_plan(groups_fixture())
    mock_probes(monkeypatch, plan, separate=separate, shortfall=1)
    with pytest.raises(ValueError, match='shortfall') as error:
        s.preflight_paths(plan, '/native', '/controller')
    receipt = error.value.storage_preflight
    assert receipt['status'] == 'failed' and receipt['gpu_started'] is False
    assert sum(r['shortfall_bytes'] for r in receipt['filesystems']) == 1


@pytest.mark.parametrize('reserve', [0, True, 8.0 * 2**30, 8 * 2**30 - 1])
def test_cannot_reduce_or_coerce_reserve(reserve):
    with pytest.raises(ValueError, match='8 GiB'):
        s.preflight_paths(s.build_storage_plan(groups_fixture()), '/native', '/controller', reserve)


def test_changed_plan_rejected_before_probing(monkeypatch):
    plan = s.build_storage_plan(groups_fixture())
    plan['native_write_bytes'] = 1
    monkeypatch.setattr(s, '_probe', lambda _: (_ for _ in ()).throw(AssertionError('must not probe')))
    with pytest.raises(ValueError, match='changed'):
        s.preflight_paths(plan, '/native', '/controller')


def test_actual_paths_probe_nearest_existing_directory_without_writing(tmp_path, monkeypatch):
    plan = s.build_storage_plan(groups_fixture())
    monkeypatch.setattr(s.shutil, 'disk_usage', lambda _: SimpleNamespace(free=plan['same_filesystem_required_bytes']))
    a, b = tmp_path / 'native/deep', tmp_path / 'controller/deep'
    receipt = s.preflight_paths(plan, a, b)
    assert receipt['destinations']['native']['destination'] == str(a)
    assert receipt['destinations']['native']['existing_ancestor'] == str(tmp_path)
    assert receipt['filesystems'][0]['filesystem_device'] == tmp_path.stat().st_dev
    assert not a.exists() and not b.exists() and not list(tmp_path.iterdir())


def test_file_ancestor_rejected_and_preserved(tmp_path):
    path = tmp_path / 'file'
    path.write_text('preserve')
    with pytest.raises(ValueError, match='not a directory'): s._probe(path / 'child')
    assert path.read_text() == 'preserve'


def test_runtime_payload_omits_only_derived_recursive_paths_and_source_inventory():
    config = dict(storage_budget={'large': 'bound'}, output_root='/native', controller_root='/controller',
        related_instance_source_sha256={'code': 'hash'}, serving_contract={'concurrency': 1}, runtime_source_sha256={'runtime': 'hash'})
    assert s.runtime_payload(config) == {'serving_contract': {'concurrency': 1}, 'runtime_source_sha256': {'runtime': 'hash'}}


def bound_attempt(tmp_path, *, nested=True):
    from .native import write_json
    controller = tmp_path / 'controller'
    native = controller / 'native' if nested else tmp_path / 'native'
    controller.mkdir()
    groups = groups_fixture()
    plan = s.build_storage_plan(groups)
    inventory = {'fixture.py': 'a' * 64}
    manifest = {'synthetic': 'new requests only'}
    write_json(controller / 'PLAN.json', dict(storage_budget=plan, group_order=plan['group_order'],
        manifest_sha256=s.digest(manifest)))
    write_json(controller / 'REQUEST_MANIFEST.json', manifest)
    write_json(controller / 'SOURCE_MANIFEST.json', dict(related_instance_source_sha256=inventory,
        approved_code_sha256=s.digest(inventory)))
    config = dict(storage_budget=plan, output_root=str(native), controller_root=str(controller),
        related_instance_source_sha256=inventory)
    return plan, groups, config, controller, native


def test_initial_and_sequential_capacity_credit_at_exact_peak_boundary(tmp_path, monkeypatch):
    from .native import write_json
    plan, groups, config, controller, native = bound_attempt(tmp_path)
    initial_required = plan['same_filesystem_required_bytes']
    consumed = lambda: sum(p.stat().st_size for p in controller.rglob('*') if p.is_file())
    monkeypatch.setattr(s.shutil, 'disk_usage', lambda _: SimpleNamespace(free=initial_required - consumed()))
    # CLI already wrote these three controller artifacts before first admission.
    first = s.preflight_paths(plan, native, controller)
    assert first['existing_artifact_credit']['controller'] == consumed() > 0
    assert first['filesystems'][0]['free_bytes'] == first['filesystems'][0]['required_bytes']
    write_json(controller / 'RUNTIME_CONFIG.json', config)
    write_json(controller / 'STORAGE_PREFLIGHT.json', first)
    write_json(controller / 'PROGRESS.json', dict(selected_groups=[dict(group_id=g['group_id'],
        group_payload_sha256=g['group_payload_sha256']) for g in groups]))
    (controller / 'EVENTS.jsonl').write_text('{"event":"completed"}\n')
    for group in groups:
        directory = native / group['group_id']
        directory.mkdir(parents=True)
        write_json(directory / 'NATIVE_CONFIG.json', dict(config, episode_id=group['group_id'], original_group=group))
        write_json(directory / 'NATIVE_FINISH.json', dict(status='passed', group_id=group['group_id'],
            checkpoint_weights_unchanged=True, training_quiescent=True))
        (directory / 'RAW_NATIVE_01.json').write_text('{"raw":"' + 'x' * 1000 + '"}')
        next_admission = s.preflight_paths(plan, native, controller)
        credit = next_admission['existing_artifact_credit']
        assert credit['native'] + credit['controller'] == consumed()
        assert next_admission['filesystems'][0]['free_bytes'] == next_admission['filesystems'][0]['required_bytes']
        assert next_admission['remaining_controller_write_bytes'] >= plan['controller_retrieval_staging_bytes']
        assert len({r['path'] for r in credit['files']}) == len(credit['files'])
    assert next_admission['remaining_total_write_bytes'] == plan['total_write_bytes'] - consumed()


def test_foreign_files_are_not_credited_or_native_nested_paths_counted_twice(tmp_path):
    from .native import write_json
    plan, groups, config, controller, native = bound_attempt(tmp_path)
    foreign = controller / 'unrelated.bin'
    foreign.write_bytes(b'x' * 10000)
    credit = s._existing_artifact_credit(plan, native, controller)
    assert str(foreign) not in {row['path'] for row in credit['files']}
    assert credit['controller'] == sum((controller / name).stat().st_size
        for name in ('PLAN.json', 'REQUEST_MANIFEST.json', 'SOURCE_MANIFEST.json'))
    native.mkdir()
    directory = native / groups[0]['group_id']
    directory.mkdir()
    write_json(directory / 'NATIVE_CONFIG.json', dict(config, episode_id=groups[0]['group_id'], original_group=groups[0]))
    write_json(directory / 'NATIVE_FINISH.json', dict(status='passed', group_id=groups[0]['group_id'],
        checkpoint_weights_unchanged=True, training_quiescent=True))
    credited = s._existing_artifact_credit(plan, native, controller)
    assert credited['controller'] == credit['controller']
    assert credited['native'] == sum(p.stat().st_size for p in directory.iterdir())


@pytest.mark.parametrize('change', ['manifest', 'source', 'runtime', 'progress', 'unfinished_native', 'foreign_group'])
def test_existing_attempt_credit_rejects_unbound_or_unfinished_artifacts(tmp_path, change):
    from .native import write_json
    plan, groups, config, controller, native = bound_attempt(tmp_path)
    if change == 'manifest': write_json(controller / 'REQUEST_MANIFEST.json', {'foreign': True})
    if change == 'source': write_json(controller / 'SOURCE_MANIFEST.json', {'approved_code_sha256': 'changed'})
    if change == 'runtime': write_json(controller / 'RUNTIME_CONFIG.json', dict(config, output_root='/foreign'))
    if change == 'progress': write_json(controller / 'PROGRESS.json', dict(selected_groups=[]))
    if change in ('unfinished_native', 'foreign_group'):
        directory = native / groups[0]['group_id']
        directory.mkdir(parents=True)
        write_json(directory / 'NATIVE_CONFIG.json', dict(config, episode_id='foreign', original_group=groups[0]))
        if change == 'foreign_group':
            write_json(directory / 'NATIVE_FINISH.json', dict(status='passed', group_id=groups[0]['group_id'],
                checkpoint_weights_unchanged=True, training_quiescent=True))
    with pytest.raises(ValueError): s._existing_artifact_credit(plan, native, controller)
