"""Read-only CPU storage accounting with synthetic sealed pool payloads."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import evidence, storage as s
from .test_evidence import pools_fixture


def plan_fixture():
    return s.build_storage_plan(pools_fixture())


def test_json_size_matches_native_writer_for_multibyte_and_nested_content(tmp_path):
    from .native import write_json
    value = {'original_pool': {'unicode': '雪😀', 'tokens': [1, 123456], 'value': 0.1, 'keep': True}}
    path = tmp_path / 'exact.json'
    write_json(path, value)
    assert s.serialized_bytes(value) == path.stat().st_size
    assert s.serialized_bytes(value) > len(json.dumps(value, indent=2, ensure_ascii=False))
    with pytest.raises(ValueError): s.serialized_bytes({'invalid': float('nan')})


def test_budget_counts_actual_nested_configs_weights_dynamic_rows_and_retrieval_staging():
    pools = pools_fixture()
    plan = s.build_storage_plan(pools)
    assert plan['gpu_started'] is False and plan['final_checkpoints'] == 3
    checkpoint = 1048626432 * 2 + 64 * 2**20
    assert plan['final_checkpoint_bytes'] == 3 * checkpoint
    assert plan['export_staging_overlap_bytes'] == checkpoint
    for pool in pools:
        owner = pool['pool_id']
        actual = len(json.dumps({'original_pool': pool}, indent=2, ensure_ascii=False, allow_nan=False).encode()) + 1
        assert plan['serialized_pool_config_bytes'][owner] == actual
        assert plan['pool_evidence_bytes'][owner] == 3 * actual + 512 * 2**20
        assert plan['pool_payload_sha256'][owner] == pool['pool_payload_sha256']
    expected_native = 4 * checkpoint + sum(plan['pool_evidence_bytes'].values())
    primary = 512 * 2**20 + 12 * sum(plan['serialized_pool_config_bytes'].values())
    assert plan['native_write_bytes'] == expected_native
    assert plan['controller_primary_write_bytes'] == plan['controller_retrieval_staging_bytes'] == primary
    assert plan['controller_write_bytes'] == 2 * primary
    assert plan['total_write_bytes'] == expected_native + 2 * primary
    assert plan['same_filesystem_required_bytes'] == plan['total_write_bytes'] + 8 * 2**30
    assert plan['separate_filesystem_required_bytes'] == {
        'native': expected_native + 8 * 2**30, 'controller': 2 * primary + 8 * 2**30}
    assert plan['optimizer_state_exports'] == plan['intermediate_checkpoints'] == 0


def test_sizes_depend_on_real_pool_content_without_changing_originals():
    pools = pools_fixture()
    before = copy.deepcopy(pools)
    baseline = s.build_storage_plan(pools)
    assert pools == before
    changed = copy.deepcopy(pools)
    changed[0]['extra_retained_evidence'] = 'x' * 10000
    changed[0]['pool_payload_sha256'] = evidence.digest({k: v for k, v in changed[0].items() if k != 'pool_payload_sha256'})
    larger = s.build_storage_plan(changed)
    owner = pools[0]['pool_id']
    delta = larger['serialized_pool_config_bytes'][owner] - baseline['serialized_pool_config_bytes'][owner]
    assert delta > 10000
    assert larger['native_write_bytes'] - baseline['native_write_bytes'] == 3 * delta
    assert larger['controller_write_bytes'] - baseline['controller_write_bytes'] == 24 * delta


@pytest.mark.parametrize('mutation', ['missing_pool', 'duplicate_pool', 'unsealed', 'changed_seal', 'changed_payload'])
def test_budget_rejects_incomplete_or_changed_pool_identity(mutation):
    pools = pools_fixture()
    if mutation == 'missing_pool': pools.pop()
    if mutation == 'duplicate_pool': pools[1] = copy.deepcopy(pools[0])
    if mutation == 'unsealed': pools[0]['status'] = 'running'
    if mutation == 'changed_seal': pools[0]['pool_payload_sha256'] = 'a' * 64
    if mutation == 'changed_payload': pools[0]['domain'] = 'other'
    with pytest.raises(ValueError): s.build_storage_plan(pools)


def set_probes(monkeypatch, *, native_device, controller_device, native_free, controller_free):
    values = {
        '/native/new': dict(destination='/native/new', existing_ancestor='/native',
            filesystem_device=native_device, free_bytes=native_free),
        '/controller/new': dict(destination='/controller/new', existing_ancestor='/controller',
            filesystem_device=controller_device, free_bytes=controller_free)}
    calls = []
    def probe(path):
        calls.append(str(path))
        return copy.deepcopy(values[str(path)])
    monkeypatch.setattr(s, '_probe', probe)
    return calls


def test_same_filesystem_sums_both_destinations_and_preserves_reserve_once(monkeypatch):
    plan = plan_fixture()
    required = plan['total_write_bytes'] + s.MIN_RESERVE_BYTES
    calls = set_probes(monkeypatch, native_device=1, controller_device=1,
        native_free=required + 1000, controller_free=required)
    receipt = s.preflight_paths(plan, '/native/new', '/controller/new')
    assert calls == ['/native/new', '/controller/new']
    assert receipt['status'] == 'passed' and receipt['same_filesystem'] and receipt['gpu_started'] is False
    assert len(receipt['filesystems']) == 1
    filesystem = receipt['filesystems'][0]
    assert filesystem['free_bytes'] == filesystem['required_bytes'] == receipt['total_required_bytes'] == required
    assert filesystem['write_bytes'] == plan['total_write_bytes'] and filesystem['reserve_bytes'] == s.MIN_RESERVE_BYTES
    assert filesystem['destinations'] == ['native', 'controller']


def test_same_filesystem_does_not_admit_each_destination_individually(monkeypatch):
    plan = plan_fixture()
    free = max(plan['native_write_bytes'], plan['controller_write_bytes']) + s.MIN_RESERVE_BYTES
    set_probes(monkeypatch, native_device=1, controller_device=1, native_free=free, controller_free=free)
    with pytest.raises(ValueError, match='shortfall') as error:
        s.preflight_paths(plan, '/native/new', '/controller/new')
    receipt = error.value.storage_preflight
    assert receipt['status'] == 'failed' and receipt['gpu_started'] is False
    assert receipt['filesystems'][0]['shortfall_bytes'] == min(plan['native_write_bytes'], plan['controller_write_bytes'])


def test_distinct_filesystems_each_have_their_own_budget_and_reserve(monkeypatch):
    plan = plan_fixture()
    native = plan['native_write_bytes'] + s.MIN_RESERVE_BYTES
    controller = plan['controller_write_bytes'] + s.MIN_RESERVE_BYTES
    set_probes(monkeypatch, native_device=1, controller_device=2, native_free=native, controller_free=controller)
    receipt = s.preflight_paths(plan, '/native/new', '/controller/new')
    assert receipt['status'] == 'passed' and receipt['same_filesystem'] is False
    assert receipt['total_required_bytes'] == native + controller
    assert [row['required_bytes'] for row in receipt['filesystems']] == [native, controller]


@pytest.mark.parametrize('failed_destination', ['native', 'controller', 'both'])
def test_any_insufficient_filesystem_fails_with_all_capacity_receipts(monkeypatch, failed_destination):
    plan = plan_fixture()
    native = plan['native_write_bytes'] + s.MIN_RESERVE_BYTES - int(failed_destination in ('native', 'both'))
    controller = plan['controller_write_bytes'] + s.MIN_RESERVE_BYTES - int(failed_destination in ('controller', 'both'))
    set_probes(monkeypatch, native_device=1, controller_device=2, native_free=native, controller_free=controller)
    with pytest.raises(ValueError, match='exceed free space') as error:
        s.preflight_paths(plan, '/native/new', '/controller/new')
    receipt = error.value.storage_preflight
    assert len(receipt['filesystems']) == 2
    assert sum(row['shortfall_bytes'] for row in receipt['filesystems']) == (2 if failed_destination == 'both' else 1)


@pytest.mark.parametrize('reserve', [0, True, 8.0 * 2**30, 8 * 2**30 - 1])
def test_reserve_cannot_be_omitted_coerced_or_reduced(reserve):
    with pytest.raises(ValueError, match='8 GiB'):
        s.preflight_paths(plan_fixture(), '/not-created-native', '/not-created-controller', reserve)


def test_custom_larger_reserve_applies_per_filesystem(monkeypatch):
    plan, reserve = plan_fixture(), 10 * 2**30
    set_probes(monkeypatch, native_device=1, controller_device=2,
        native_free=plan['native_write_bytes'] + reserve, controller_free=plan['controller_write_bytes'] + reserve)
    receipt = s.preflight_paths(plan, '/native/new', '/controller/new', reserve)
    assert receipt['total_required_bytes'] == plan['total_write_bytes'] + 2 * reserve


def test_mutating_plan_fails_before_filesystem_probe(monkeypatch):
    plan = plan_fixture()
    plan['native_write_bytes'] = 0
    monkeypatch.setattr(s, '_probe', lambda _: (_ for _ in ()).throw(AssertionError('must not probe')))
    with pytest.raises(ValueError, match='changed'):
        s.preflight_paths(plan, '/native/new', '/controller/new')


def test_real_nearest_existing_ancestor_uses_st_dev_and_creates_nothing(tmp_path, monkeypatch):
    plan = plan_fixture()
    required = plan['total_write_bytes'] + s.MIN_RESERVE_BYTES
    calls = []
    monkeypatch.setattr(s.shutil, 'disk_usage', lambda path: (calls.append(Path(path)) or SimpleNamespace(free=required)))
    native, controller = tmp_path / 'native/deep/new', tmp_path / 'controller/deep/new'
    receipt = s.preflight_paths(plan, native, controller)
    assert calls == [tmp_path, tmp_path]
    assert receipt['same_filesystem'] and receipt['filesystems'][0]['filesystem_device'] == tmp_path.stat().st_dev
    assert not native.exists() and not controller.exists() and list(tmp_path.iterdir()) == []
    assert receipt['destinations']['native']['existing_ancestor'] == str(tmp_path)


def test_non_directory_existing_ancestor_is_rejected_without_writes(tmp_path):
    existing = tmp_path / 'file'
    existing.write_text('retain')
    with pytest.raises(ValueError, match='not a directory'):
        s._probe(existing / 'child')
    assert existing.read_text() == 'retain' and list(tmp_path.iterdir()) == [existing]
