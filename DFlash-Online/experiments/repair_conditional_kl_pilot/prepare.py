"""CPU-only authentication, ordered subset sealing and conservative pilot admission."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.contract import validate_manifest, digest
from experiments.opencode_native_comparison.evaluation import check_reference
from experiments.opencode_native_comparison.serving_gate import qualification_slots
from experiments.osd_components.prepare import verify_native_runtime
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from .configuration import build_config
from .contract import ARMS, BASE_COMMIT, LIMITS, SERVING, validate_inputs
from .resources import capacity, boundary, read


def prepare(source_run, root, execution_commit, source_root):
    import torch
    import transformers
    old, root = Path(source_run), Path(root)
    root.mkdir(parents=True, exist_ok=False)
    seal = root/'sealed'; seal.mkdir()
    prior = read(old/'sealed/PLAN_BUDGET_RESUME_001.json')
    if prior['execution_commit'] != BASE_COMMIT:
        raise ValueError('Require the actual PR56 scientific execution reference')
    source_inputs = old/'sealed/INPUTS.json'
    if identity(source_inputs) != prior['input_identities'][str(source_inputs)]:
        raise ValueError('Authenticated PR56 input manifest is unavailable or changed')
    inputs = read(source_inputs)
    validate_manifest(inputs)
    original = old/'sealed/ORIGINAL_CONFIG.yaml'
    if identity(original) != prior['input_identities'][str(original)]:
        raise ValueError('Authenticated native recipe changed')
    source_canonicals = old/'precheck/CANONICALS.json'
    references = read(source_canonicals)
    completion = read(old/'precheck/AR_F_PRECHECK.json')
    if completion['status'] != 'passed' or completion['canonical_digest'] != digest(references):
        raise ValueError('Frozen-F canonical authentication failed')
    selected = {key:inputs[key][:count] for key,count in (
        ('train_requests',128),('evaluation_requests',128),
        ('qualification_requests',2),('warmup_requests',2))}
    validate_inputs(selected)
    canonicals = {r['request_id']:references[r['request_id']]['F'] for r in selected['evaluation_requests']}
    warmup_path = old/'qualification_policy_001/WARMUP_CANONICALS.json'
    warmup = read(warmup_path)
    warmup = {r['request_id']:warmup[r['request_id']]['F'] for r in selected['warmup_requests']}
    for request in selected['evaluation_requests'] + selected['warmup_requests']:
        reference = (canonicals if request in selected['evaluation_requests'] else warmup)[request['request_id']]
        check_reference(request, reference, prior['canonical_provenance'], method='F')
    qualification = {}
    for slot in qualification_slots(inputs):
        if slot['card']=='A' and slot['method']=='F' and slot['phase']=='reserved':
            if slot['request_id'] in {r['request_id'] for r in selected['qualification_requests']}:
                path=old/'qualification_policy_001/A/records'/(slot['slot_id']+'.json')
                record=read(path)
                qualification[slot['request_id']]=record
    if len(qualification)!=2:
        raise ValueError('The two reserved qualification references are missing')
    write_json(seal/'INPUTS.json', selected)
    write_json(seal/'CANONICALS.json', canonicals)
    write_json(seal/'WARMUP_CANONICALS.json', warmup)
    write_json(seal/'QUALIFICATION_REFERENCES.json', qualification)
    shutil.copy2(original, seal/'ORIGINAL_CONFIG.yaml')
    for name in ('MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json'):
        path=old/'sealed'/name
        if identity(path)!=prior['input_identities'][str(path)]:
            raise ValueError('Native pin changed: '+name)
        shutil.copy2(path,seal/name)
    checked=[]
    for entry in prior['authenticated_files']:
        if entry['role'] not in ('released_draft','released_target'):
            continue
        if identity(entry['path'])!=entry['expected_identity']:
            raise ValueError('Released model artifact changed: '+entry['path'])
        checked.append(entry)
    native_runtime=verify_native_runtime(prior['worker_python'],read(seal/'RUNTIME_PIN.json'))
    training_environment=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)
    if training_environment!=read(seal/'TRAINING_PIN.json')['environment'] or torch.cuda.is_initialized():
        raise ValueError('Training environment changed or CPU preparation initialized CUDA')
    configs={}
    for arm in ARMS:
        path=seal/(arm+'.yaml')
        path.write_text(build_config(original.read_text(),arm=arm,
            output_dir=root/'arms'/arm/'trainer',maximum_events=1))
        configs[arm]=str(path)
    production_storage=capacity(selected['train_requests'])
    feature=production_storage['active_feature_bound_bytes']
    recovery=production_storage['recovery_bound_bytes']
    weights=production_storage['export_bytes']
    overlap=production_storage['replacement_overlap_bytes']
    no_offload_peak=int((2*feature+2*recovery+2*weights+overlap)*1.05)+4_000_000_000
    # The actual offload probe cannot fit the three-hour limit. The handoff
    # explicitly permits bounded qualification without a production launch.
    storage=dict(projected_peak_bytes=25_000_000_000,
        qualification_only=True, qualification_components=dict(recovery_copies=2, export_copies=2,
            max_feature_rows=4*4096, feature_bytes_per_row=49164,
            total_tensor_bound_bytes=2*recovery+2*weights+4*4096*49164,
            margin_fraction=.05,source_compiler_logs_allowance_bytes=4_000_000_000),
        production_without_offload_peak_bytes=no_offload_peak,
        production_offload_plan=production_storage)
    forecast=dict(version='prospectively_sealed_one_event',
        retained_source='PR56 C: event1 update589.38s/feedback154.95s; endpointF first128max139.822s/pass',
        expected_elapsed_s=1200, conservative_elapsed_s=1800, conservative_card_s=3600,
        report_reserved_s=300, scope='bounded qualification only; production blocked',
        observed_transfer_bytes=Path(source_canonicals).stat().st_size, observed_transfer_s=128.89,
        maximum_elapsed_s=10800, maximum_card_s=18000,
        two_event_rejected='Conservative retained-feature plus atomic-recovery peak exceeds current free minus400GB; no preexisting retirement allowed')
    plan=dict(schema='repair_conditional_kl_pilot_v1',root=str(root),source_root=str(source_root),
        source_manifest=str(Path(source_root).parent/'SOURCE.json'),
        base_commit=BASE_COMMIT,execution_commit=execution_commit,maximum_events=1,
        production_admitted=False,execution_mode='qualification_only',
        arms=list(ARMS),serving=SERVING,limits=LIMITS,configs=configs,
        target=prior['target'],draft=prior['draft'],worker_python=prior['worker_python'],
        training_python=prior['training_python'],devices=prior['devices'],
        original_config=str(seal/'ORIGINAL_CONFIG.yaml'),manifest=str(seal/'INPUTS.json'),
        canonical_provenance=prior['canonical_provenance'],storage=storage,forecast=forecast,
        allocation_roots=[str(root),str(Path(source_root).parent)],compiler_baseline=compiler_snapshot(),
        expected_qualification_work=dict(feedback_generation=4,evaluation_generation=4,
            teacher_preparation=4,response_forward=4,response_backward=4,adam=2,
            startup_response_forward=4,startup_response_backward=4,startup_adam=0),
        unexecuted_production_work=dict(feedback_generation=256,scored_generation=768,warmup_generation=24,
            response_forward=512,response_backward=512,adam=256,teacher_preparation=256),
        source_identities={str(p):identity(p) for p in (source_inputs,source_canonicals,warmup_path,original)},
        model_identities=checked,canonical_policy='Every speculative output matches existing frozen-F canonical',
        source_subset='First128train and first128evaluation in retained ordered PR56 manifest',
        production_blocker=dict(required_free_bytes=no_offload_peak+400_000_000_000,
            actual_free_bytes=shutil.disk_usage(root).free,
            reason='Maximum-length one-event storage exceeds reserve; observed offload cannot fit3h'),
        offload_required='BASE prepared features hash-verified on Fenrir before PAIR training; first8witnesses remain local',
        auxiliary_coefficient='1-mu, no tuned coefficient',bootstrap_seed=20261124)
    plan['sealed_files']={str(p):identity(p) for p in seal.iterdir() if p.is_file()}
    plan['sealed_files'][plan['source_manifest']]=identity(plan['source_manifest'])
    plan['binding']=digest(plan)
    write_json(seal/'PLAN.json',plan)
    result=dict(status='passed',native_runtime=native_runtime,training_environment=training_environment,
        model_artifacts_checked=len(checked),input_validation=validate_inputs(selected),
        admission=boundary(plan),forecasts=forecast,model_work=0)
    write_json(root/'CPU_PREFLIGHT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-run','root','execution-commit','source-root'):
        parser.add_argument('--'+key,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())),indent=2))
