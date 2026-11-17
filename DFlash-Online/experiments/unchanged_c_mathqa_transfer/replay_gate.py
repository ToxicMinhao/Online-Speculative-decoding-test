"""Typed common-E5 replay prerequisite; never claims historical tensor equality.

A current pool may enter only the two original control replays. Main C work
requires both complete, exact replay comparisons bound to that same pool.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.bounded_prefix_allocation.prepare import identity, _digest, _write

PREREQUISITE = 'CURRENT_E5_REPLAY_PREREQUISITE'
CURRENT_FEATURES = 'CURRENT_E5_FEATURE_MANIFEST'
CURRENT_AUTHENTICATION = 'CURRENT_E5_REPLAY_ASSET_AUTHENTICATION'
QUALIFIED = 'FUNCTIONAL_E5_REPLAY_QUALIFIED'
LIMITATION = ('Functional reproducibility of both E5 control boundaries is established; '
              'historical feature byte identity remains unproved. This is not C efficacy evidence.')
TENSOR_FIELDS = {'input_ids','loss_mask','hidden_states','target_last_hidden_states'}
from .replay_comparison import EXCLUSIONS as COMPARISON_EXCLUSIONS


def read(path):
    return json.loads(Path(path).read_text())


def pinned(node):
    if identity(node['path']) != node['identity']:
        raise ValueError('Changed replay evidence: '+node['path'])
    return read(node['path'])


def make_current_manifest(assets, audit):
    """Select current common-E5 fingerprints from a read-only CPU packet audit."""
    if (audit.get('status') != 'current_files_validated_historical_content_unproven'
            or audit.get('historical_feature_identity_verified') is not False
            or audit.get('model_work') != 0 or audit.get('cuda_initialized') is not False):
        raise ValueError('Expected current-only CPU packet audit')
    pool=deepcopy(audit['pools']['common5']);expected=assets['common_e5']
    if (pool['manifest_path'] != expected['manifest'] or pool['manifest_identity'] != expected['manifest_identity']
            or pool['producer'] != expected['own_checkpoint']
            or pool['producer_identity'] != expected['own_weights_identity']
            or [row['path'] for row in pool['features']] != expected['feature_paths']):
        raise ValueError('Current packet audit uses a different common-E5 producer or order')
    for row in pool['features']:
        if (row['matches_original_token_mask_shape_metadata'] is not True
                or row['old_content_hash_available'] is not False or set(row['tensors']) != TENSOR_FIELDS):
            raise ValueError('Current packet lacks complete tensor/metadata evidence')
        for value in row['tensors'].values():
            if (not value['shape'] or not isinstance(value['dtype'],str)
                    or len(value['sha256']) != 64):
                raise ValueError('Malformed current tensor fingerprint')
    return dict(kind=CURRENT_FEATURES,status='current_pool_sealed',
        historical_feature_identity_verified=False,model_work=0,**pool)


def validate_current_manifest(plan, manifest):
    expected=plan['common_e5']
    if (manifest.get('kind') != CURRENT_FEATURES or manifest.get('status') != 'current_pool_sealed'
            or manifest.get('historical_feature_identity_verified') is not False
            or manifest.get('model_work') != 0
            or manifest['manifest_path'] != expected['manifest']
            or manifest['manifest_identity'] != expected['manifest_identity']
            or manifest['producer'] != expected['own_checkpoint']
            or manifest['producer_identity'] != expected['own_weights_identity']
            or [r['path'] for r in manifest['features']] != expected['feature_paths']
            or len(manifest['features']) != 128):
        raise ValueError('Current feature manifest is not the original common-E5 pool')
    for row in manifest['features']:
        if (row['identity'] != expected['feature_identities'].get(row['path'])
                or set(row['tensors']) != TENSOR_FIELDS
                or row['matches_original_token_mask_shape_metadata'] is not True
                or row['old_content_hash_available'] is not False):
            raise ValueError('Incomplete current feature/tensor identity: '+row['path'])
    return manifest


def validate_prerequisite(plan, receipt):
    """Current content pins authorize only N5/S5 validation, never C work."""
    prerequisite=plan.get('prerequisite',{})
    if (prerequisite.get('kind') != PREREQUISITE or prerequisite.get('authorized_arms') != ['N','S']
            or prerequisite.get('event') != 5 or prerequisite.get('historical_feature_identity_verified') is not False
            or prerequisite.get('qualified_receipt') != plan['root']+'/replay/FUNCTIONAL_E5_REPLAY_QUALIFIED.json'):
        raise ValueError('No exact N5/S5 replay-only prerequisite authorization')
    if (receipt.get('kind') != CURRENT_AUTHENTICATION or receipt.get('status') != 'passed'
            or receipt.get('historical_feature_identity_verified') is not False
            or receipt.get('authorized_work') != ['N5_REPLAY','S5_REPLAY']):
        raise ValueError('Current-only authentication cannot authorize main C work')
    expected_files=plan['retained_asset_identities']
    if any(receipt['files'].get(path) != expected for path,expected in expected_files.items()):
        raise ValueError('Original retained checkpoint, metadata or configuration identity differs')
    manifest=validate_current_manifest(plan,pinned(prerequisite['feature_manifest']))
    if receipt.get('feature_manifest_identity') != prerequisite['feature_manifest']['identity']:
        raise ValueError('Prerequisite authentication uses a different feature manifest')
    if any(receipt['files'].get(r['path']) != r['identity'] for r in manifest['features']):
        raise ValueError('Current authentication lacks full-file feature identity')
    return dict(status='passed',kind=PREREQUISITE,historical_feature_identity_verified=False,
        authorized_work=['N5_REPLAY','S5_REPLAY'],feature_paths=list(plan['common_e5']['feature_paths']))


def require_current_features(plan):
    """Recheck each sealed file and its tensor bytes on CPU before stage entry."""
    import torch
    from online_feedback.training import validate_feature,validate_target_binding
    manifest=validate_current_manifest(plan,pinned(plan['prerequisite']['feature_manifest']))
    for row in manifest['features']:
        path=Path(row['path']);before=path.stat()
        if identity(path) != row['identity']:
            raise ValueError('Current common-E5 file changed: '+str(path))
        packet=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        metadata=validate_feature(packet,max_length=2272,hidden_width=20480,vocab_size=151936)
        validate_target_binding(packet)
        if metadata != row['metadata']:
            raise ValueError('Current common-E5 metadata changed: '+str(path))
        actual={}
        for name,value in packet.items():
            if torch.is_tensor(value):
                raw=value.detach().cpu().contiguous().view(torch.uint8).numpy()
                actual[name]=dict(shape=list(value.shape),dtype=str(value.dtype),
                                 sha256=hashlib.sha256(memoryview(raw)).hexdigest())
        if actual != row['tensors']:
            raise ValueError('Current common-E5 tensor changed: '+str(path))
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Feature changed during CPU authentication: '+str(path))
        del packet
    return dict(status='passed',features=128,full_file_and_tensor_identities=True,
                feature_manifest_identity=plan['prerequisite']['feature_manifest']['identity'])


def validate_replay_prerequisite(plan, arm):
    if arm not in ('N','S'):
        raise ValueError('Prerequisite authorizes only original N5 and S5 replays')
    result=validate_prerequisite(plan,pinned(plan['asset_authentication']))
    result['current_features']=require_current_features(plan)
    return result


def context(plan):
    """The exact scientific/source/runtime context shared by both stages."""
    return dict(binding=plan['binding'],execution_commit=plan['execution_commit'],
        root=plan['root'],work_ledger=plan['root']+'/WORK.json',clock=plan['root']+'/CLOCK.json',
        source_root=plan['source_root'],runtime_contract=plan['runtime_contract'],
        origin_state_identity=plan['origin_state_identity'],origin_weights_identity=plan['origin_weights_identity'],
        historical_boundaries={arm:plan['retained_endpoints'][arm+'5']['endpoint'] for arm in ('N','S')},
        feature_manifest=plan['prerequisite']['feature_manifest'],
        current_asset_authentication=plan['asset_authentication'])


def validate_main_receipt(plan, receipt):
    """Only the typed complete replay gate can replace unavailable old hashes."""
    if (receipt.get('kind') != QUALIFIED or receipt.get('status') != 'passed'
            or receipt.get('historical_feature_identity_verified') is not False
            or receipt.get('context') != context(plan) or receipt.get('limitation') != LIMITATION
            or receipt.get('comparison_exclusions') != deepcopy(COMPARISON_EXCLUSIONS)
            or set(receipt.get('comparisons',{})) != {'N','S'}):
        raise ValueError('Both complete exact N5/S5 replay receipts are required for C')
    # Comparison schema is shared with the replay worker; every required group
    # must pass, including all per-presentation and per-window observations.
    for arm,nodes in receipt['comparisons'].items():
        validate_comparison(plan,arm,pinned(nodes['result']),pinned(nodes['comparison']))
    validate_prerequisite(plan,pinned(plan['asset_authentication']))
    manifest=pinned(plan['prerequisite']['feature_manifest'])
    for row in manifest['features']:
        if identity(row['path']) != row['identity']:
            raise ValueError('Current feature changed after replay: '+row['path'])
    return dict(status='passed',kind=QUALIFIED,historical_feature_identity_verified=False,
                functional_e5_replay_verified=True,comparisons=receipt['comparisons'])


def validate_comparison(plan, arm, result, comparison):
    """Require the complete worker result and its independent exact comparator."""
    counts=dict(optimizer_steps=128,response_presentations=256,start_step=512,
                end_step=640,end_presentations=1280)
    if (result.get('schema') != 'pr52_e5_control_replay_v1' or result.get('status') != 'passed'
            or result.get('arm') != arm or result.get('event') != 5
            or result.get('binding') != plan['binding'] or result.get('source_binding') != plan['pr51_binding']
            or result.get('execution_commit') != plan['execution_commit']
            or result.get('historical_endpoint') != plan['retained_endpoints'][arm+'5']['endpoint']
            or result.get('counts') != counts or result.get('historical_feature_identity_verified') is not False
            or result.get('checkpoint_files_written') != 0 or result.get('export_files_written') != 0):
        raise ValueError('Incomplete or inexact '+arm+'5 replay result')
    prerequisite=result.get('prerequisite',{})
    if (prerequisite.get('status') != 'passed' or prerequisite.get('kind') != PREREQUISITE
            or prerequisite.get('feature_paths') != plan['common_e5']['feature_paths']
            or prerequisite.get('current_features',{}).get('feature_manifest_identity')
                != plan['prerequisite']['feature_manifest']['identity']):
        raise ValueError('Replay result does not authenticate the sealed current pool')
    groups={'draft_tensors','optimizer_moments_counters_options','rng','scientific_configuration',
            'original_event_ancestry','sampler_accumulation_boundary','event_scientific_receipts'}
    if (comparison.get('schema') != 'mathqa_e5_exact_replay_comparison_v1'
            or comparison.get('status') != 'passed' or comparison.get('arm') != arm
            or comparison.get('responses_compared') != 256 or comparison.get('windows_compared') != 128
            or comparison.get('presentations_compared') != 256 or comparison.get('event_compared') is not True
            or comparison.get('endpoint_compared') is not True
            or comparison.get('categories') != {name:'exact' for name in groups}
            or comparison.get('exclusions') != COMPARISON_EXCLUSIONS
            or 'first_divergence' not in comparison or comparison['first_divergence'] is not None
            or comparison.get('historical_feature_identity_verified') is not False):
        raise ValueError('Missing required exact scientific replay comparison')
    return result


def make_functional_receipt(plan, native_result, scaled_result):
    rows={}
    for arm,path in (('N',native_result),('S',scaled_result)):
        path=Path(path);comparison=path.parent/'comparison/COMPARISON.json'
        rows[arm]=dict(result=dict(path=str(path),identity=identity(path)),
                       comparison=dict(path=str(comparison),identity=identity(comparison)))
        validate_comparison(plan,arm,read(path),read(comparison))
    receipt=dict(kind=QUALIFIED,status='passed',context=context(plan),comparisons=rows,
        historical_feature_identity_verified=False,limitation=LIMITATION,
        comparison_exclusions=deepcopy(COMPARISON_EXCLUSIONS))
    validate_main_receipt(plan,receipt)
    return receipt
