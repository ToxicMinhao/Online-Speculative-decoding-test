"""Seal actual artifact paths, published source and fixed execution budgets."""
import argparse
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from experiments.lost_repair.prepare import source_manifest, identity, read
from .analysis import ANALYSIS
from .evaluation import evaluation_schedule
from .panel import verify_panel


def verify_canonical_runtime(binding, original):
    """Reused answers must belong to the exact planned verifier realization."""
    versions = dict(python=binding['python_version'], torch=binding['torch'],
                    transformers=binding['transformers'])
    if (binding['serving'] != original['serving']
            or binding['native_runtime_sha256'] != original['runtime_contract']['native_runtime_sha256']
            or versions != original['versions']['serving']):
        raise ValueError('Canonical runtime differs from the authenticated serving plan')
    for name, expected in binding['target_files'].items():
        path = str(Path(original['target_path']) / name)
        if original['pinned_files'].get(path) != expected:
            raise ValueError('Canonical target identity differs from the serving plan: ' + name)


def prepare(task, prior, output, remote_root, remote_code):
    task,output,remote_root,remote_code=map(Path,(task,output,remote_root,remote_code))
    if output.exists():raise FileExistsError('Execution package already exists')
    if not remote_root.is_relative_to('/data/sources/minhao/runs') or not remote_code.is_relative_to('/data/sources/minhao'):
        raise ValueError('Remote paths outside approved roots')
    code=Path(__file__).resolve().parents[2]
    source=source_manifest(code)
    source['preparation_source_path']=source['source_path'];source['source_path']=str(remote_code)
    inputs=read(task/'input_bundle_001/INPUTS.json')
    panel=inputs['panel']
    verify_panel(panel)
    if not 0<len(panel['requests'])<=256 or panel['outcomes_used'] or not panel['input_only_selection']:
        raise ValueError('Invalid pre-outcome panel')
    old=read(prior/'campaign_001/PLAN.json')
    verify_canonical_runtime(inputs['target_runtime_binding'], old)
    sealed=output/'sealed';sealed.mkdir(parents=True)
    write_json(output/'SOURCE.json',source)
    write_json(sealed/'PANEL.json',panel)
    write_json(sealed/'REUSED_CANONICALS.json',inputs['references'])
    write_json(sealed/'CANONICAL_REUSE_AUDIT.json',inputs['canonical_reuse_audit'])
    write_json(sealed/'TIMING_SCHEDULE.json',evaluation_schedule(panel['requests']))
    shutil.copyfile(task/'preparation/CANDIDATE_SLOTS.json',sealed/'CANDIDATE_SLOTS.json')
    reports=code/'reports/dflash_lost_repair_20261114'
    for name in ('ALL_STARTS.csv','NATIVE_WEIGHTS.csv','IDENTITIES.json'):
        shutil.copyfile(reports/name,sealed/name)
    shutil.copyfile(task/'HANDOFF.txt',sealed/'HANDOFF.txt')
    write_json(sealed/'INPUT_PROVENANCE.json',dict(
        canonical_sources=inputs['source_files'],panel_identity=inputs['panel_identity'],
        canonical_binding=inputs['canonical_binding'],target_runtime_binding=inputs['target_runtime_binding']))
    old_inputs=read(prior/'campaign_001/training/INPUTS.json')['inputs']
    pinned={r['path']:{k:r[k] for k in ('bytes','sha256')} for r in old_inputs}
    pinned.update({str(remote_root/path.relative_to(output)):identity(path)
                   for path in output.rglob('*') if path.is_file()})
    schedule='/data/sources/minhao/runs/dflash_lost_repair_20261114/execution_001/campaign_001/training/SCHEDULE.json'
    pinned[schedule]=dict(bytes=2583042,sha256='6ac18ca2c16ded078f831ab90b72b58333403e5bece9553943de491e5193a1d0')
    feature_manifest='/data/sources/minhao/runs/dflash_lost_repair_20261114/execution_001/campaign_001/census/FEATURES.json'
    origin=old['recovery_path']
    if pinned[origin]['sha256']!='15ac28e1235851a15158e789af1480fd2d82535903b1dec5d992ec843c9ae3a1':
        raise ValueError('Origin recovery differs from handoff')
    plan=dict(schema='recent_repair_retention_event7_v1',root=str(remote_root),
        code_path=str(remote_code),execution_commit=source['commit'],source_manifest=str(remote_root/'SOURCE.json'),
        original_plan=old,input_identities=pinned,origin_state=origin,
        origin_export=old['checkpoints']['K6']['export'],origin_config=old['config_path'],
        configs={arm:old['config_path'] for arm in ('NATIVE','RECENT','CONTROL')},
        feature_manifest=feature_manifest,schedule=schedule,
        candidate_slots=str(remote_root/'sealed/CANDIDATE_SLOTS.json'),
        all_starts=str(remote_root/'sealed/ALL_STARTS.csv'),native_weights=str(remote_root/'sealed/NATIVE_WEIGHTS.csv'),
        panel_manifest=str(remote_root/'sealed/PANEL.json'),panel_size=len(panel['requests']),
        path_request_ids=panel['path_request_ids'],timing_schedule=str(remote_root/'sealed/TIMING_SCHEDULE.json'),
        reused_references=str(remote_root/'sealed/REUSED_CANONICALS.json'),
        missing_reference_ids=inputs['missing_reference_ids'],canonical_binding=inputs['canonical_binding'],
        target_path=old['target_path'],worker_python=old['worker_python'],training_python=old['training_python'],
        checkpoints={arm:old['checkpoints'][arm] for arm in ('K5','K6')},
        serving=old['serving'],analysis=ANALYSIS,serving_gpu=0,training_gpu=0,
        serving_gpu_identity='GPU-9986735a-3e68-d1bd-5998-9634ef24341e',
        qualification_root=str(remote_root/'qualification'),
        storage=dict(maximum_additional_bytes=128*2**30,reserve_bytes=8*2**30),
        host_memory_required_bytes=80*2**30,
        phase_timeout_s=14400,campaign_timeout_s=21600,fixed_start_seconds_limit=14400,
        production_budget=dict(trajectories=3,adam_steps=384,response_presentations=768),
        qualification_budget=dict(temporary_adam_steps=3,response_forwards=14,response_backwards=14),
        first_firing_budget=dict(max_temporary_adam_steps=4,max_response_forwards=8,max_response_backwards=8),
        reference_preparation_budget=dict(response_forwards=256,backwards=0,adam_steps=0),
        serving_budget=dict(timed_completions=6*len(panel['requests']),parent_walks=len(panel['path_request_ids']),
                            missing_canonicals=len(inputs['missing_reference_ids'])),
        estimated_gpu_hours=[2,3],parallel_training=False,automatic_retry=False)
    write_json(sealed/'PLAN.json',plan)
    write_json(sealed/'SEAL.json',dict(status='sealed',execution_commit=source['commit'],
        files={str(remote_root/path.relative_to(output)):identity(path)
               for path in output.rglob('*') if path.is_file()}))
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('task','prior','output','remote-root','remote-code'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    plan=prepare(args.task,args.prior,args.output,args.remote_root,args.remote_code)
    print(plan['execution_commit'],plan['panel_size'],plan['serving_budget'])
