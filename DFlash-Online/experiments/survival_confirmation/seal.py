"""Seal actual confirmation paths and frozen exports before any generation."""
import argparse
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.survival_head.data import read, digest
from experiments.survival_head.contract import ARMS, ANALYSIS, SERVING
from .evaluation import evaluation_schedule


def seal(root, prepared, pilot, source_manifest):
    root,prepared,pilot=map(lambda p:Path(p).resolve(),(root,prepared,pilot))
    source=read(source_manifest); old=read(pilot/'sealed/PLAN.json')
    prior_manifest=read(pilot/'final_report/INPUT_MANIFEST.json')
    prior_summary=read(pilot/'final_report/SUMMARY.json')
    if (prior_summary['status']!='complete' or prior_summary['execution_source']['commit']!='7f89719956f0256c0916a583d7ff615954cc66f9'
            or prior_manifest['missing'] or prior_manifest['parse_errors']):
        raise ValueError('Require completed retained pilot evidence')
    for relative in ('sealed/PLAN.json','sealed/MODEL_PIN.json','sealed/RUNTIME_PIN.json'):
        path=pilot/relative
        if identity(path)!=prior_manifest['files'].get(str(path)):
            raise ValueError('Pilot input differs from completed evidence: '+relative)
    code=Path(__file__).resolve().parents[2]
    if (source['commit'] == '' or source['dirty'] is not False
            or source['source_path'] != str(code)):
        raise ValueError('Require actual clean published source')
    for name,expected in source['files'].items():
        if identity(code/name)!=expected: raise ValueError('Staged source mismatch: '+name)
    from .data import verify_panel
    panel=read(prepared/'PANEL.json'); verify_panel(panel); n=len(panel['requests'])
    if n<136 or n>768 or n%8: raise ValueError('Panel does not meet sealed confirmation budget')
    if (pilot/'EXIT_CODE').read_text().strip()!='0': raise ValueError('Original pilot not complete')
    root.mkdir(parents=True,exist_ok=True); folder=root/'sealed';folder.mkdir(exist_ok=False)
    for name in ('PANEL.json','EXCLUSION_AUDIT.json','EXPOSURE.json','REUSED_CANONICAL.json',
                 'CANONICAL_REUSE_AUDIT.json','PILOT_RECOMPUTATION.json','COST_ESTIMATE.json'):
        shutil.copy2(prepared/name,folder/name)
    for name in ('MODEL_PIN.json','RUNTIME_PIN.json'):
        shutil.copy2(pilot/'sealed'/name,folder/name)
    for name in ('INPUT_MANIFEST.json','SUMMARY.json'):
        shutil.copy2(pilot/'final_report'/name,folder/('PILOT_'+name))
    checkpoints={}
    for arm in ARMS:
        path=pilot/'campaign_001/gsm8k/event_005'/arm/'RESULT.json'; result=read(path)
        if identity(path)!=prior_manifest['files'].get(str(path)):
            raise ValueError('Retained pilot result bytes changed: '+arm)
        export=Path(result['checkpoint_path'])
        checkpoints[arm]=dict(checkpoint_path=str(export),weights_identity=result['weights_identity'],
            result_path=str(path),result_identity=identity(path),
            export_files={p.name:identity(p) for p in export.iterdir() if p.is_file()},
            training_seed=17,adam_step=640)
    refs=read(folder/'REUSED_CANONICAL.json'); reuse=read(folder/'CANONICAL_REUSE_AUDIT.json')
    binding=reuse['binding_digest']; pinned=reuse['target_runtime_binding']
    model=read(folder/'MODEL_PIN.json'); runtime=read(folder/'RUNTIME_PIN.json')
    if (pinned['target_files']!=model['target']['files']
            or any(pinned[k]!=runtime[k] for k in ('python_version','torch','transformers'))
            or any(runtime['native_runtime_sha256'].get(k)!=v for k,v in pinned['native_runtime_sha256'].items())):
        raise ValueError('Reused canonical target/runtime differs from frozen pilot')
    if (pinned['serving']['block_size']!=16 or pinned['serving']['max_new_tokens']!=2048
            or pinned['serving']['context_limit']!=4096 or pinned['serving']['concurrency']!=1
            or pinned['serving']['precision']!='bfloat16' or pinned['serving']['temperature']!=0
            or pinned['serving']['attention_backend']!='fa3' or pinned['serving']['reasoning']!='off'
            or pinned['serving']['cache_policy']!=SERVING['cache_policy']):
        raise ValueError('Reused canonical serving contract differs')
    if any(r['binding_digest']!=binding for r in refs.values()): raise ValueError('Mixed reference bindings')
    missing=[r['request_id'] for r in panel['requests'] if r['request_id'] not in refs]
    from experiments.adaflash_c1.storage import RAW_BYTES_PER_GENERATION
    raw_budget=(6*n+len(missing))*RAW_BYTES_PER_GENERATION
    storage=dict(reserve_bytes=8*2**30,maximum_additional_bytes=raw_budget+3*2**30,
        raw_bytes_per_generation=RAW_BYTES_PER_GENERATION,
        details=dict(raw_records_and_native_sidecars=raw_budget,source_and_retrieval_staging=1*2**30,
                     runtime_cache_and_reports=2*2**30))
    plan=dict(schema='survival_confirmation_v1',root=str(root),panel_size=n,
        execution_source={k:source[k] for k in ('commit','dirty','repository','source_path')},
        source_manifest=str(Path(source_manifest).resolve()),pilot_execution_commit='7f89719956f0256c0916a583d7ff615954cc66f9',
        checkpoints=checkpoints,panel_manifest=str(folder/'PANEL.json'),
        exposure_inventory=str(folder/'EXPOSURE.json'),reused_references=str(folder/'REUSED_CANONICAL.json'),
        references=str(root/'references/CANONICAL.json'),canonical_binding=binding,missing_reference_ids=missing,
        target=old['target'],worker_python=old['worker_python'],controller_python=old['training_python'],
        serving_gpu=old['serving_gpu'],serving_gpu_identity=old['serving_gpu_identity'],
        serving=SERVING,analysis=ANALYSIS,automatic_retry=False,storage=storage,
        budget=dict(optimizer_steps=0,training_presentations=0,feedback_generations=0,
                    timed_completions=6*n,reference_generations=len(missing),extra_generation_warmups=0),
        evaluation_schedule=evaluation_schedule(panel['requests'],sealed_n=n))
    plan['binding']=digest(plan)
    write_json(folder/'PLAN.json',plan)
    files={str(p):identity(p) for p in folder.iterdir() if p.is_file()}
    files[str(Path(source_manifest).resolve())]=identity(source_manifest)
    write_json(folder/'SEAL.json',dict(binding=plan['binding'],files=files))
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','prepared','pilot','source-manifest'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();p=seal(a.root,a.prepared,a.pilot,a.source_manifest)
    print(dict(panel_size=p['panel_size'],budget=p['budget'],binding=p['binding']))
