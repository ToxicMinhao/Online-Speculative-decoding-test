"""Synthetic CPU-only admission checks; no private evidence or model execution."""
import copy
import json

import pytest

from . import evidence as ev
from experiments.dflash_kl_native_diagnostic.test_experiment import native_row, loss_row


def groups_fixture(*, colliding_request_ids=False):
    """Six complete synthetic evidence groups with repeated local turn numbers."""
    groups = []
    frozen_files = {'model.safetensors': {'bytes': 1, 'sha256': 'f' * 64}}
    frozen = dict(checkpoint_id='F', path='/synthetic/released', files=frozen_files,
                  checkpoint_identity_sha256=ev.digest(frozen_files))
    for group_index, domain in enumerate(('gsm8k', 'gsm8k', 'codealpaca', 'codealpaca', 'sharegpt', 'sharegpt')):
        group_id = f'{domain}_{group_index % 2:02d}'
        adapted_files = {'model.safetensors': {'bytes': 1, 'sha256': str(group_index) * 64}}
        group = dict(group_id=group_id, domain=domain, status='complete', feedback=[], support=[],
            scores=[], evaluation_requests=[], reference_pairs={'source': [], 'heldout': []},
            checkpoints={'F': copy.deepcopy(frozen), 'S8': dict(checkpoint_id='U2',
                path='/synthetic/' + group_id + '/checkpoints/U2', files=adapted_files,
                checkpoint_identity_sha256=ev.digest(adapted_files))})
        for population, role in ev.POPULATION_ROLE.items():
            for index in range(4):
                native = {}
                for label in ('F', 'S8'):
                    row = native_row(group_id, role, index, label)
                    row['domain'] = domain
                    row['input_ids'] = [10, 100 + group_index * 8 + (4 if population == 'heldout' else 0) + index]
                    row['input_ids_sha256'] = ev.digest(row['input_ids'])
                    row['checkpoint_identity_sha256'] = group['checkpoints'][label]['checkpoint_identity_sha256']
                    if colliding_request_ids:
                        row['request_id'] = f'{role}_{index}'
                    native[label] = row
                pair = dict(group_id=group_id, domain=domain, population=population,
                    request_id=native['F']['request_id'], checkpoint_bindings={
                        label: group['checkpoints'][label]['checkpoint_identity_sha256'] for label in ('F', 'S8')},
                    validation={'status': 'passed', 'errors': []},
                    **{label: dict(native=native[label], kl=loss_row(native[label])) for label in ('F', 'S8')})
                group['reference_pairs'][population].append(pair)
                if population == 'source':
                    group['feedback'].append(copy.deepcopy(native['F']))
                    support = dict(episode_id=group_id, record_id=native['F']['request_id'], turn=index + 1,
                        input_ids=list(native['F']['input_ids']), output_ids=list(native['F']['output_ids']),
                        input_token_count=len(native['F']['input_ids']), output_cap=1024,
                        input_ids_sha256=ev.digest(native['F']['input_ids']),
                        output_ids_sha256=ev.digest(native['F']['output_ids']), target_id='T0',
                        target_identity_sha256='a' * 64, capture_checkpoint='F',
                        capture_checkpoint_sha256=frozen['checkpoint_identity_sha256'],
                        metadata_binding=dict(path=f'/synthetic/{group_id}/support/{index}.json', bytes=123,
                                              sha256=ev.digest([group_id, index])))
                    group['support'].append(support)
                else:
                    group['scores'].extend(copy.deepcopy(list(native.values())))
                    group['evaluation_requests'].append(dict(request_id=native['F']['request_id'],
                                                            input_ids=list(native['F']['input_ids'])))
        groups.append(group)
    return groups


def pools_fixture(**kwargs):
    return ev.build_pools(groups_fixture(**kwargs))


def _reseal(pool):
    """A seal alone must not excuse inconsistent lineage, ownership or ordering."""
    pool['binding_inventory'] = ev.pool_binding_inventory(pool)
    pool['pool_payload_sha256'] = ev._payload_sha256(pool)


def test_three_pools_preserve_manifest_order_and_every_original_record():
    originals = groups_fixture()
    before = copy.deepcopy(originals)
    pools = ev.build_pools(originals)
    assert originals == before
    assert [p['domain'] for p in pools] == ['gsm8k', 'codealpaca', 'sharegpt']
    for index, pool in enumerate(pools):
        first, second = originals[2 * index:2 * index + 2]
        assert pool['groups'] == [first, second]
        assert pool['group_ids'] == [first['group_id'], second['group_id']]
        assert pool['pool_id'] == 'p8_' + first['domain'] and 'group_id' not in pool
        assert pool['checkpoints'] == {'F': first['checkpoints']['F']}
        assert ev.validate_pool(pool) == {'status': 'passed', 'errors': []}
        for field in ('feedback', 'support'):
            assert [r['group_id'] for r in pool[field]] == pool['group_ids'] * 4
        assert [r['turn'] for r in pool['support']] == [1, 1, 2, 2, 3, 3, 4, 4]
        assert [r['group_id'] for r in pool['reference_pairs']['source']] == pool['group_ids'] * 4
        assert [r['group_id'] for r in pool['reference_pairs']['heldout']] == [first['group_id']] * 4 + [second['group_id']] * 4
        for offset, row in enumerate(pool['support']):
            original = (first, second)[offset % 2]['support'][offset // 2]
            assert all(row[key] == value for key, value in original.items())
        for population in ev.POPULATION_ROLE:
            for pair in pool['reference_pairs'][population]:
                owner = next(g for g in pool['groups'] if g['group_id'] == pair['group_id'])
                original = next(r for r in owner['reference_pairs'][population] if r['request_id'] == pair['request_id'])
                assert {k: v for k, v in pair.items() if k not in ev._ADDED_FIELDS} == original
                assert pair['checkpoint_bindings']['S8'] == owner['checkpoints']['S8']['checkpoint_identity_sha256']


def test_dataset_and_original_group_order_follow_manifest_encounters_not_sorting():
    groups = groups_fixture()
    ordered = [groups[i] for i in (5, 3, 1, 4, 0, 2)]
    pools = ev.build_pools(ordered)
    assert [pool['domain'] for pool in pools] == ['sharegpt', 'codealpaca', 'gsm8k']
    assert [pool['group_ids'] for pool in pools] == [
        [group['group_id'] for group in ordered if group['domain'] == domain]
        for domain in ('sharegpt', 'codealpaca', 'gsm8k')]


def test_composite_ids_distinguish_local_request_and_turn_collisions():
    pool = pools_fixture(colliding_request_ids=True)[0]
    assert len({r['record_id'] for r in pool['support']}) == 4
    assert len({r['turn'] for r in pool['support']}) == 4
    assert len({r['pool_request_id'] for r in pool['support']}) == 8
    for support, native, pair in zip(pool['support'], pool['feedback'], pool['reference_pairs']['source']):
        assert support['pool_request_id'] == native['pool_request_id'] == pair['pool_request_id']
        assert json.loads(support['pool_request_id']) == list(ev.composite_identity(support))
        assert support['episode_id'] == native['group_id']


def test_composite_encoding_does_not_confuse_delimiters_or_unicode():
    values = [dict(domain='a|b', group_id='c', role='feedback', request_id='d'),
              dict(domain='a', group_id='b|c', role='feedback', request_id='d'),
              dict(domain='a', group_id='b', role='feedback', request_id='c|d'),
              dict(domain='a', group_id='b', role='evaluation', request_id='c|d'),
              dict(domain='a', group_id='b', role='feedback', request_id='c\u2603d')]
    assert len({ev._pool_request_id(row) for row in values}) == len(values)


@pytest.mark.parametrize('change', ['episode', 'record', 'population', 'missing', 'unknown_role'])
def test_composite_identity_rejects_ambiguous_aliases(change):
    row = dict(domain='d', group_id='g', episode_id='g', role='feedback', population='source',
               request_id='r', record_id='r')
    if change == 'episode': row['episode_id'] = 'other'
    if change == 'record': row['record_id'] = 'other'
    if change == 'population': row['population'] = 'heldout'
    if change == 'missing': del row['domain']
    if change == 'unknown_role': row['role'] = 'query'
    with pytest.raises(ValueError): ev.composite_identity(row)


@pytest.mark.parametrize('change', ['count', 'group_duplicate', 'domain_count', 'domain_leak',
    'support_role', 'support_owner', 'support_request', 'support_tokens', 'support_turn', 'support_hash',
    'support_metadata', 'support_cap', 'support_target', 'frozen_origin', 'frozen_capture', 'different_F',
    'source_pair_owner', 'source_pair_role', 'source_pair_checkpoint', 'native_owner', 'heldout_native_role',
    'pair_validation', 'diagnostic_mask', 'source_heldout_duplicate', 'crossgroup_prompt', 'missing_pair',
    'original_pool_owner'])
def test_build_rejects_changed_or_missing_evidence(change):
    groups = groups_fixture()
    group = groups[0]
    support = group['support'][0]
    pair = group['reference_pairs']['source'][0]
    if change == 'count': groups.pop()
    if change == 'group_duplicate': groups[1]['group_id'] = group['group_id']
    if change == 'domain_count': groups[1]['domain'] = groups[2]['domain']
    if change == 'domain_leak': support['domain'] = 'codealpaca'
    if change == 'support_role': support['role'] = 'evaluation'
    if change == 'support_owner': support['episode_id'] = groups[1]['group_id']
    if change == 'support_request': support['record_id'] = 'other'
    if change == 'support_tokens': support['output_ids'][0] += 1
    if change == 'support_turn': support['turn'] = None
    if change == 'support_hash': support['input_ids_sha256'] = 'b' * 64
    if change == 'support_metadata': del support['metadata_binding']
    if change == 'support_cap': support['output_cap'] = 256
    if change == 'support_target': support['target_identity_sha256'] = 'b' * 64
    if change == 'frozen_origin': support['capture_checkpoint_sha256'] = 'b' * 64
    if change == 'frozen_capture': support['capture_checkpoint'] = 'S8'
    if change == 'different_F': group['checkpoints']['F']['path'] = '/synthetic/other'
    if change == 'source_pair_owner': pair['group_id'] = groups[1]['group_id']
    if change == 'source_pair_role': pair['population'] = 'heldout'
    if change == 'source_pair_checkpoint': pair['checkpoint_bindings']['S8'] = groups[1]['checkpoints']['S8']['checkpoint_identity_sha256']
    if change == 'native_owner': pair['S8']['native']['group_id'] = groups[1]['group_id']
    if change == 'heldout_native_role': group['reference_pairs']['heldout'][0]['S8']['native']['role'] = 'feedback'
    if change == 'pair_validation': pair['validation']['status'] = 'failed'
    if change == 'diagnostic_mask': pair['S8']['kl']['binary_loss_mask'][0][1] = 0.
    if change == 'source_heldout_duplicate': group['reference_pairs']['heldout'][0] = copy.deepcopy(pair)
    if change == 'crossgroup_prompt':
        other = groups[1]
        own = group['feedback'][0]['input_ids']
        other['support'][0].update(input_ids=list(own), input_ids_sha256=ev.digest(own))
        other['feedback'][0].update(input_ids=list(own), input_ids_sha256=ev.digest(own))
        for label in ('F', 'S8'):
            other['reference_pairs']['source'][0][label]['native'].update(input_ids=list(own), input_ids_sha256=ev.digest(own))
            other['reference_pairs']['source'][0][label]['kl']['input_ids_sha256'] = ev.digest(own)
    if change == 'missing_pair': group['reference_pairs']['source'].pop()
    if change == 'original_pool_owner': pair['pool_id'] = 'p8_other'
    with pytest.raises((ValueError, KeyError)): ev.build_pools(groups)


@pytest.mark.parametrize('change', ['support_order', 'feedback_order', 'source_order', 'heldout_order',
    'pool_owner', 'request_owner', 'original_request', 'checkpoint_owner', 'inventory', 'seal', 'extra'])
def test_validate_rejects_mutation_even_when_a_new_seal_is_supplied(change):
    pool = pools_fixture()[0]
    if change == 'support_order': pool['support'].reverse()
    if change == 'feedback_order': pool['feedback'].reverse()
    if change == 'source_order': pool['reference_pairs']['source'].reverse()
    if change == 'heldout_order': pool['reference_pairs']['heldout'].reverse()
    if change == 'pool_owner': pool['support'][0]['pool_id'] = 'p8_other'
    if change == 'request_owner': pool['feedback'][0]['pool_request_id'] = pool['feedback'][1]['pool_request_id']
    if change == 'original_request': pool['reference_pairs']['heldout'][0]['request_id'] = 'foreign'
    if change == 'checkpoint_owner': pool['checkpoints']['S8'] = copy.deepcopy(pool['groups'][0]['checkpoints']['S8'])
    if change == 'inventory': pool['binding_inventory']['support'][0]['payload_sha256'] = 'a' * 64
    if change == 'seal': pool['pool_payload_sha256'] = 'a' * 64
    if change == 'extra': pool['support'][0]['substituted'] = True
    if change not in ('inventory', 'seal'): _reseal(pool)
    with pytest.raises(ValueError): ev.validate_pool(pool)


def test_seal_binds_original_training_or_other_unprojected_fields():
    pool = pools_fixture()[0]
    pool['groups'][0]['unprojected_original_metadata'] = {'changed': True}
    with pytest.raises(ValueError, match='inventory'): ev.validate_pool(pool)


def test_inventory_contains_exact_original_diagnostics_and_token_bindings():
    pool = pools_fixture()[0]
    inventory = ev.pool_binding_inventory(pool)
    assert inventory == pool['binding_inventory']
    assert ev._payload_sha256(pool) == pool['pool_payload_sha256']
    for population in ev.POPULATION_ROLE:
        for reference, binding in zip(pool['reference_pairs'][population], inventory['reference_pairs'][population]):
            assert binding['reference_pair_sha256'] == ev.digest(reference)
            assert binding['diagnostic_bindings'] == {label: ev.digest(reference[label]['kl']) for label in ('F', 'S8')}
    # Round-tripping a sealed plan preserves hashes and all identity aliases.
    assert ev.validate_pool(json.loads(json.dumps(pool)))['status'] == 'passed'


def test_loader_reuses_immutable_reader_and_keeps_original_bundle(monkeypatch):
    groups = groups_fixture()
    original = dict(groups=groups, manifest={'original': True}, runtime_config={'seed': 17},
                    bindings={'diagnostic': {'retained': True}})
    calls = []
    def reader(*args):
        calls.append(args)
        return original
    monkeypatch.setattr(ev, 'load_retained_evidence', reader)
    bundle = ev.load_evidence('/synthetic/pr9', '/synthetic/pr10')
    assert calls == [('/synthetic/pr9', '/synthetic/pr10')]
    assert bundle is original and bundle['groups'] is groups
    assert bundle['manifest'] == {'original': True} and bundle['runtime_config'] == {'seed': 17}
    assert bundle['bindings']['diagnostic'] == {'retained': True}
    assert len(bundle['pools']) == len(bundle['bindings']['pools']) == 3
    assert [binding['pool_payload_sha256'] for binding in bundle['bindings']['pools']] == [
        pool['pool_payload_sha256'] for pool in bundle['pools']]


def test_missing_evidence_propagates_without_regeneration(monkeypatch):
    def missing(*args):
        raise FileNotFoundError('Original retained result missing')
    monkeypatch.setattr(ev, 'load_retained_evidence', missing)
    with pytest.raises(FileNotFoundError, match='retained result missing'):
        ev.load_evidence('/missing/pr9', '/missing/pr10')
