"""Seal P8 allocations over immutable PR9/PR10 evidence; never generate records.

Original groups and nested F/S8 evidence are copied without rewriting identities.
Only the new pool views acquire P8 ownership and a composite request identifier.
"""
from __future__ import annotations

import copy
import json

from experiments.dflash_baseline_learning_curve.experiment import digest, require
from experiments.dflash_kl_native_diagnostic.evidence import heldout_records
from experiments.dflash_kl_native_diagnostic.experiment import validate_pair
from experiments.dflash_lower_lr_control.evidence import load_evidence as load_retained_evidence


POPULATION_ROLE = {'source': 'feedback', 'heldout': 'evaluation'}
_ADDED_FIELDS = ('pool_id', 'pool_request_id')


def composite_identity(record):
    """Return original dataset/group/role/request identity, never a local turn.

Support rows must first be aligned to their own original native record. Aliases
are checked rather than silently preferring one inconsistent identity field.
"""
    domain = record.get('domain')
    group = record.get('group_id', record.get('episode_id'))
    request = record.get('request_id', record.get('record_id'))
    population = record.get('population')
    role = record.get('role', POPULATION_ROLE.get(population))
    require(all(isinstance(value, str) and value for value in (domain, group, role, request)),
            'Missing original composite identity')
    require(role in POPULATION_ROLE.values(), 'Unknown original response role')
    if population is not None:
        require(POPULATION_ROLE.get(population) == role, 'Population/role identity differs')
    if 'episode_id' in record:
        require(record['episode_id'] == group, 'Original group/episode identity differs')
    if 'record_id' in record:
        require(record['record_id'] == request, 'Original request/record identity differs')
    return domain, group, role, request


def _pool_request_id(record):
    # JSON arrays retain component boundaries even when identifiers contain
    # punctuation; delimiter concatenation would not be collision-free.
    return json.dumps(composite_identity(record), ensure_ascii=True, separators=(',', ':'))


def _owned(record, pool_id):
    require(not any(key in record for key in _ADDED_FIELDS), 'Original evidence already has pool ownership')
    result = copy.deepcopy(record)
    result.update(pool_id=pool_id, pool_request_id=_pool_request_id(record))
    return result


def _aligned_support(support, native):
    require(support.get('episode_id') == native['group_id'] and
            support.get('record_id') == native['request_id'], 'Support original owner/request differs')
    result = copy.deepcopy(support)
    for key in ('domain', 'group_id', 'request_id', 'role'):
        require(key not in result or result[key] == native[key], 'Support/native identity differs: ' + key)
        result.setdefault(key, native[key])
    require(result['role'] == 'feedback', 'Held-out response entered training support')
    require(all(result.get(key) == native[key] for key in ('input_ids', 'output_ids')),
            'Support differs from exact frozen-generated tokens')
    require(result.get('capture_checkpoint') == 'F' and
            result.get('capture_checkpoint_sha256') == native['checkpoint_identity_sha256'],
            'Support must originate from its retained frozen F response')
    for key in ('input_ids', 'output_ids'):
        require(result.get(key + '_sha256') == digest(result[key]), 'Support token hash differs: ' + key)
    require(type(result.get('turn')) is int and result['turn'] > 0, 'Missing original support turn')
    require(result.get('input_token_count') == len(native['input_ids']) and result.get('output_cap') == 1024,
            'Support token count/output cap differs')
    binding = result.get('metadata_binding')
    require(isinstance(binding, dict) and isinstance(binding.get('path'), str) and
            type(binding.get('bytes')) is int and binding['bytes'] > 0 and
            isinstance(binding.get('sha256'), str) and len(binding['sha256']) == 64,
            'Missing original support metadata binding')
    composite_identity(result)
    return result


def _interleave(first, second):
    return [row for pair in zip(first, second) for row in pair]


def _validate_original_group(group):
    require(group.get('status') == 'complete', 'Original group must be complete')
    require(len(group['feedback']) == len(group['support']) == 4,
            'Keep four original feedback/support responses per group')
    require(set(group['reference_pairs']) == set(POPULATION_ROLE) and
            all(len(group['reference_pairs'][p]) == 4 for p in POPULATION_ROLE),
            'All four original source and held-out reference pairs are required')
    frozen = group['checkpoints']['F']
    require(frozen.get('checkpoint_id') == 'F' and
            frozen['checkpoint_identity_sha256'] == digest(frozen['files']),
            'Released F checkpoint binding differs')
    require(group['checkpoints']['S8']['checkpoint_identity_sha256'] ==
            digest(group['checkpoints']['S8']['files']), 'Original S8 checkpoint binding differs')
    seen_requests, seen_prompts = set(), set()
    for population, role in POPULATION_ROLE.items():
        originals = group['feedback'] if population == 'source' else heldout_records(group)
        require(len(originals) == 4, 'Original population has changed')
        for reference, original in zip(group['reference_pairs'][population], originals):
            expected = (group['domain'], group['group_id'], role, original['request_id'])
            require(composite_identity(reference) == composite_identity(original) == expected,
                    'Reference dataset/group/role/request differs')
            require(reference.get('validation', {}).get('status') == 'passed' and
                    validate_pair(reference)['status'] == 'passed', 'Invalid retained F/S8 reference pair')
            require(reference['F']['native'] == original, 'Reference F differs from own retained record')
            for label in ('F', 'S8'):
                require(composite_identity(reference[label]['native']) == expected,
                        'Reference native belongs to another original dataset/group/role/request')
                require(reference['checkpoint_bindings'][label] ==
                        group['checkpoints'][label]['checkpoint_identity_sha256'],
                        'Reference uses another original group checkpoint')
            require(original['request_id'] not in seen_requests, 'Duplicate or cross-role original request')
            prompt_hash = digest(original['input_ids'])
            require(prompt_hash not in seen_prompts, 'Duplicate or cross-role original prompt')
            seen_requests.add(original['request_id'])
            seen_prompts.add(prompt_hash)
    for support, native in zip(group['support'], group['feedback']):
        _aligned_support(support, native)


def _expected_views(groups, pool_id):
    first, second = groups
    support_by_group = [[_aligned_support(s, n) for s, n in zip(g['support'], g['feedback'])] for g in groups]
    return dict(
        support=[_owned(r, pool_id) for r in _interleave(*support_by_group)],
        feedback=[_owned(r, pool_id) for r in _interleave(first['feedback'], second['feedback'])],
        reference_pairs={
            'source': [_owned(r, pool_id) for r in _interleave(
                first['reference_pairs']['source'], second['reference_pairs']['source'])],
            'heldout': [_owned(r, pool_id) for g in groups for r in g['reference_pairs']['heldout']]})


def pool_binding_inventory(pool):
    """Compact provenance inventory; metadata paths are bound, not re-created."""
    def record_binding(row):
        return dict(pool_request_id=row['pool_request_id'], original_identity=list(composite_identity(row)),
                    payload_sha256=digest(row), input_ids_sha256=digest(row['input_ids']),
                    output_ids_sha256=digest(row['output_ids']))

    references = {}
    for population, rows in pool['reference_pairs'].items():
        references[population] = [dict(
            pool_request_id=r['pool_request_id'], original_identity=list(composite_identity(r)),
            checkpoint_bindings=copy.deepcopy(r['checkpoint_bindings']),
            reference_pair_sha256=digest(r),
            diagnostic_bindings={label: digest(r[label]['kl']) for label in ('F', 'S8')}) for r in rows]
    return dict(pool_id=pool['pool_id'], domain=pool['domain'], group_ids=list(pool['group_ids']),
        original_group_payloads=[dict(group_id=g['group_id'], sha256=digest(g)) for g in pool['groups']],
        released_checkpoint_identity_sha256=pool['checkpoints']['F']['checkpoint_identity_sha256'],
        support=[dict(**record_binding(r), metadata_binding=copy.deepcopy(r['metadata_binding'])) for r in pool['support']],
        feedback=[record_binding(r) for r in pool['feedback']], reference_pairs=references)


def _payload_sha256(pool):
    return digest({key: value for key, value in pool.items() if key != 'pool_payload_sha256'})


def validate_pool(pool):
    """Fail closed on pool mutation, role leakage, lineage or order changes."""
    require(isinstance(pool, dict), 'Missing P8 pool')
    groups = pool.get('groups')
    require(isinstance(groups, list) and len(groups) == 2, 'P8 requires two original groups')
    require(pool.get('status') == 'prepared', 'P8 allocation must be prepared before execution')
    require(pool.get('pool_id') == 'p8_' + pool['domain'] and
            all(g['domain'] == pool['domain'] for g in groups), 'Cross-dataset pool or owner differs')
    require(pool.get('group_ids') == [g['group_id'] for g in groups] and
            len(set(pool['group_ids'])) == 2, 'Original group membership/order differs')
    for group in groups:
        _validate_original_group(group)
    require(set(pool['checkpoints']) == {'F'} and
            all(g['checkpoints']['F'] == pool['checkpoints']['F'] for g in groups),
            'Each P8 pool starts from the shared released F checkpoint')
    expected = _expected_views(groups, pool['pool_id'])
    for key, value in expected.items():
        require(pool.get(key) == value, 'P8 ordered original identity/payload binding differs: ' + key)
    references = [r for rows in pool['reference_pairs'].values() for r in rows]
    identities = [composite_identity(row) for row in references]
    require(len(identities) == len(set(identities)) == 16, 'Duplicate composite evaluation identity')
    require(len({row['pool_request_id'] for row in references}) == 16, 'Composite pool request collision')
    require(len({digest(row['F']['native']['input_ids']) for row in references}) == 16,
            'Duplicate or cross-role pooled prompts')
    require(len({(r.get('target_id'), r.get('target_identity_sha256')) for r in pool['support']}) == 1 and
            all(r.get('target_id') and r.get('target_identity_sha256') for r in pool['support']),
            'Training support target identities differ')
    require(pool.get('binding_inventory') == pool_binding_inventory(pool), 'P8 binding inventory changed')
    require(pool.get('pool_payload_sha256') == _payload_sha256(pool), 'P8 sealed payload changed')
    return dict(status='passed', errors=[])


def build_pools(groups):
    """Build exactly three dataset pools in original manifest encounter order."""
    require(isinstance(groups, list) and len(groups) == 6, 'Exactly six original groups are required')
    require(len({g['group_id'] for g in groups}) == 6, 'Duplicate original group')
    by_domain = {}
    for group in groups:
        by_domain.setdefault(group['domain'], []).append(group)
    require(len(by_domain) == 3 and all(len(rows) == 2 for rows in by_domain.values()),
            'Exactly two original groups per each of three datasets are required')
    require(all(g['checkpoints']['F'] == groups[0]['checkpoints']['F'] for g in groups),
            'All dataset pools require the same released F checkpoint')
    pools = []
    for domain, originals in by_domain.items():
        require(isinstance(domain, str) and domain and '/' not in domain and '\\' not in domain and
                domain not in ('.', '..'), 'Invalid dataset pool owner')
        for group in originals:
            _validate_original_group(group)
        pool_id = 'p8_' + domain
        pool = dict(pool_id=pool_id, domain=domain, group_ids=[g['group_id'] for g in originals],
                    groups=copy.deepcopy(originals), status='prepared',
                    checkpoints={'F': copy.deepcopy(originals[0]['checkpoints']['F'])})
        pool.update(_expected_views(originals, pool_id))
        pool['binding_inventory'] = pool_binding_inventory(pool)
        pool['pool_payload_sha256'] = _payload_sha256(pool)
        validate_pool(pool)
        pools.append(pool)
    return pools


def load_evidence(baseline_root, diagnostic_root):
    """Use the unchanged, hash-pinned PR9/PR10 readers; missing evidence fails."""
    bundle = load_retained_evidence(baseline_root, diagnostic_root)
    bundle['pools'] = build_pools(bundle['groups'])
    bundle['bindings']['pools'] = [dict(**copy.deepcopy(pool['binding_inventory']),
        pool_payload_sha256=pool['pool_payload_sha256']) for pool in bundle['pools']]
    return bundle
