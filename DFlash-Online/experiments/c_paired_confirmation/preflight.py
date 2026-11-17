"""CPU-only origin, common-pool and staged-package admission before model work."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.preflight import verify_source
from experiments.unchanged_c_mathqa_transfer.preflight import runtime_inventory
from experiments.unchanged_c_mathqa_transfer.replay_gate import pinned, validate_main_receipt, require_current_features
from .prepare import identity, load_inputs, plan_binding
from .resources import storage_plan
from .runtime import LIMITS, ALLOCATION_BYTES, RESERVE_BYTES


def read(path):
    return json.loads(Path(path).read_text())


def same_runtime_contract(left, right):
    """PR50 used python_version; PR52 renamed that same pinned value python."""
    def canonical(value):
        result=deepcopy(value)
        for role in ('training','serving'):
            fields=result.get(role,{})
            if 'python_version' in fields:
                old=fields.pop('python_version')
                if 'python' in fields and fields['python']!=old:
                    raise ValueError('Contradictory runtime Python version aliases')
                fields['python']=old
        return result
    return canonical(left)==canonical(right)


def verify_origin_paths(sub):
    for path, expected in ((sub['origin_state'], sub['origin_state_identity']),
            (Path(sub['origin_export'])/'model.safetensors', sub['origin_weights_identity'])):
        if identity(path) != expected:
            raise ValueError('Complete origin or export content changed: '+str(path))
    if identity(sub['origin_config']) != sub['original_config']['identity']:
        raise ValueError('Packaged original configuration differs from retained original')


def inspect_states(plan):
    import torch
    from experiments.ok_coding.training import _export_equal
    from .state import validate_origin
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU origin checks must not initialize CUDA')
    result={}
    for dataset, sub in plan['datasets'].items():
        verify_origin_paths(sub)
        state=torch.load(sub['origin_state'],map_location='cpu',weights_only=False,mmap=True)
        original=Path(sub['origin_config']).read_text()
        receipt=validate_origin(state,original,dataset=dataset,expected_cuda_rng_states=1)
        expected=read(sub['expected_ancestry'])['requests']
        count=768 if dataset=='gsm8k' else 512
        actual=[r['record_id'] for event in state['events'] for r in event['features']]
        if len(expected)!=count or actual!=[r['request_id'] for r in expected]:
            raise ValueError('Complete original adaptation ancestry differs: '+dataset)
        _export_equal(state,sub['origin_export'])
        result[dataset]=dict(status='passed',receipt=receipt,ancestry_records=count,export_exact=True,
            step=state['global_step'],presentations=state['presentations'],mmap=True,read_only=True)
        del state
        gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU origin validation initialized CUDA')
    return result


def packet_input_binding(packet, request, preparation, capture, producer):
    """Bind the saved canonical teacher rows to the original native producer."""
    if (capture['checkpoint_id']!=producer or capture['record_id']!=request['request_id']
            or capture['prompt_ids']!=request['prompt_ids']
            or packet['record_id']!=request['request_id']
            or packet['prompt_length']!=len(request['prompt_ids'])
            or preparation['record_id']!=request['request_id']
            or preparation['input_ids']!=capture['prompt_ids']+capture['response_ids']
            or packet['input_ids'][0].tolist()!=preparation['input_ids']
            or preparation['target_argmax_ids']!=capture['response_ids']
            or preparation['native_capture_validated'] is not True
            or preparation['label_mismatches']!=[]):
        raise ValueError('Original common-pool input/producer/target binding differs')


def inspect_common_gsm8k(plan, inputs):
    import torch
    import yaml
    from online_feedback.training import validate_feature,validate_target_binding
    sub=plan['datasets']['gsm8k'];rows=sub['common_features'];requests=inputs['gsm8k']['first_inputs']['requests']
    if (len(rows)!=128 or len(requests)!=128 or [r['path'] for r in rows]!=sub['common_feature_paths']
            or [r['record_id'] for r in rows]!=[r['request_id'] for r in requests]):
        raise ValueError('Common GSM E7 pool order/cardinality differs')
    historical=pinned(sub['historical_plan'])
    if (historical['origin_state']!=sub['origin_state'] or historical['origin_export']!=sub['origin_export']
            or historical['event7_feature_paths']!=sub['common_feature_paths']
            or not same_runtime_contract(historical['runtime_contract'],plan['runtime_contract'])):
        raise ValueError('Common GSM pool does not bind the retained PR50 origin/runtime')
    geometry=sub['geometry'];max_length=yaml.safe_load(Path(sub['origin_config']).read_text())['data']['max_length']
    checked=[]
    for row,request in zip(rows,requests):
        path=Path(row['path']);before=path.stat()
        if row['identity']!=historical['input_identities'][str(path)] or identity(path)!=row['identity']:
            raise ValueError('Common GSM feature differs from historical content: '+str(path))
        preparation=pinned(row['preparation']);capture=pinned(row['capture'])
        packet=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        metadata=validate_feature(packet,max_length=max_length,
            hidden_width=geometry['hidden_size']*geometry['captured_layers'],vocab_size=151936)
        validate_target_binding(packet)
        packet_input_binding(packet,request,preparation,capture,sub['common_producer'])
        if (packet['hidden_states'].dtype!=torch.bfloat16
                or packet['target_last_hidden_states'].dtype!=torch.bfloat16
                or packet['target_last_hidden_states'].shape[-1]!=geometry['hidden_size']
                or packet['loss_mask'].dtype!=torch.float32):
            raise ValueError('Qualified BF16 feature geometry differs')
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Common feature changed during CPU admission')
        checked.append(dict(record_id=row['record_id'],identity=row['identity'],metadata=metadata))
        del packet
    return dict(status='passed',features=128,historical_feature_identity_verified=True,
                native_producer_and_input_binding=True,records=checked)


def inspect_common_mathqa(plan, inputs):
    sub=plan['datasets']['mathqa'];old=pinned(sub['historical_plan'])
    if (old['origin_state']!=sub['origin_state'] or old['origin_export']!=sub['origin_export']
            or old['common_e5']!=sub['common_e5'] or not same_runtime_contract(old['runtime_contract'],plan['runtime_contract'])
            or old['prerequisite']['feature_manifest']!=sub['current_feature_manifest']
            or sub['common_feature_paths']!=old['common_e5']['feature_paths']):
        raise ValueError('MathQA E5 is not the exact functionally qualified PR52 pool')
    receipt=validate_main_receipt(old,pinned(sub['functional_replay']))
    features=require_current_features(old)
    original_inputs=read(old['event5_inputs'])['requests']
    if original_inputs!=inputs['mathqa']['first_inputs']['requests']:
        raise ValueError('MathQA E5 original input order or tokenization changed')
    return dict(status='passed',historical_feature_identity_verified=False,
        functional_replay=receipt,current_features=features,original_input_order_preserved=True)


def allocation_roots(plan):
    expected=[plan['root'],plan['source_root']]
    if plan['allocation_roots']!=expected or len(set(expected))!=2:
        raise ValueError('New run/source roots must be charged exactly once')
    root,source=map(Path,expected)
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('New charged roots overlap')
    for path in (root,source):
        if not path.is_absolute() or not path.is_dir() or path.resolve()!=path:
            raise ValueError('New charged root is missing or linked')
    if root.parent.name!='dflash_c_paired_confirmation_20261117' or not root.name.startswith('execution_'):
        raise ValueError('Charged run root belongs to another experiment')
    return dict(status='passed',allocation_roots=expected,historical_assets_charged=False)


def inspect(plan_path):
    plan=read(plan_path);root=Path(plan['root']);began=time.perf_counter()
    result=dict(status='running',execution_commit=plan['execution_commit'],model_work=0,inputs={})
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
            raise ValueError('Preflight requires CUDA_VISIBLE_DEVICES empty')
        if plan['binding']!=plan_binding(plan) or read(root/'sealed/SEAL.json')!=dict(
                binding=plan['binding'],execution_commit=plan['execution_commit'],input_identities=plan['input_identities']):
            raise ValueError('Staged plan or input seal differs')
        if (plan['limits']!=LIMITS or plan['methods']!=['N','C'] or plan['continuation_rng_seed']!=29
                or plan['reference_reuse'] is not False or plan['native_control_reuse'] is not False):
            raise ValueError('Fixed paired campaign contract differs')
        for path,expected in plan['input_identities'].items():
            if identity(path)!=expected:raise ValueError('Immutable staged input differs: '+path)
            result['inputs'][path]=expected
        inputs=load_inputs(plan)
        result['input_readiness']={k:dict(scored=256,warmup=2,adaptation=256) for k in inputs}
        result['source']=verify_source(plan)
        required={'online_feedback/prefix_allocation.py','online_feedback/objective.py'}
        if set(plan['scientific_source_identities'])!=required:raise ValueError('Both unchanged scientific source pins required')
        for name,expected in plan['scientific_source_identities'].items():
            if identity(Path(plan['source_root'])/name)!=expected:raise ValueError('Scientific implementation changed: '+name)
        result['runtime']=runtime_inventory(plan)
        result['states']=inspect_states(plan)
        result['common_features']=dict(gsm8k=inspect_common_gsm8k(plan,inputs),mathqa=inspect_common_mathqa(plan,inputs))
        geometry={}
        for dataset,sub in plan['datasets'].items():
            cfg=read(Path(sub['origin_export'])/'config.json')
            measured=dict(hidden_size=cfg['hidden_size'],captured_layers=len(cfg['dflash_config']['target_layer_ids']))
            if measured!=sub['geometry']:raise ValueError('Sealed feature geometry differs: '+dataset)
            geometry[dataset]=dict(measured,next_prompt_lengths=[len(r['prompt_ids']) for r in inputs[dataset]['next_inputs']['requests']],
                recovery_bytes=Path(sub['origin_state']).stat().st_size,
                export_bytes=Path(sub['origin_export'],'model.safetensors').stat().st_size)
        measured=storage_plan(geometry,retain_final_exports=plan['storage']['retain_final_exports'])
        if measured!=plan['storage'] or measured['allocation_cap_bytes']!=ALLOCATION_BYTES or measured['reserve_bytes']!=RESERVE_BYTES:
            raise ValueError('Actual storage geometry differs from complete phase admission')
        free=shutil.disk_usage(root).free
        result['storage']=dict(measured,actual_free_bytes=free)
        if measured['status']!='passed' or free<measured['required_initial_free_bytes']:
            raise RuntimeError('Complete peak allocation and shared-server reserve do not fit')
        if (type(plan['compiler_baseline_allocated_bytes'])is not int or plan['compiler_baseline_allocated_bytes']<0
                or plan['compiler_growth_reserved_bytes']!=measured['compiler_bytes']):
            raise ValueError('Compiler baseline or growth reservation differs')
        result['allocation']=allocation_roots(plan)
        result.update(status='passed',cuda_initialized=False,
            parallel_qualification_admitted=result['runtime']['host']['parallel_qualification_admitted'])
    except BaseException as error:
        result.update(status='failed',error=repr(error),error_path=getattr(error,'filename',None));raise
    finally:
        result['wall_s']=time.perf_counter()-began
        write_json(root/'CPU_PREFLIGHT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    args=parser.parse_args();print(json.dumps(inspect(args.plan)))
