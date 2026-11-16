"""Build a private version2 execution package from already audited inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .preflight import identity
from .references import resolve_missing_requests
from .resources import storage_plan, LIMITS
from .state import arm_lr


def build(workspace, destination, *, remote_root, source_root, execution_commit):
    import yaml
    workspace, destination = Path(workspace), Path(destination)
    research = workspace/'research'
    task = research/'dflash_persistent_prefix_withholding_20261115'
    data = task/'v2_data/sealed_001'
    references = task/'v2_reference'
    old = json.loads((research/'dflash_native_update_size_20261115/delivery/execution_003/raw/sealed/PLAN.json').read_text())
    inspection = json.loads((references/'DEPENDENCY_INSPECTION.json').read_text())
    destination.mkdir(parents=True,exist_ok=False)
    sealed = destination/'sealed'
    sealed.mkdir()
    for name in ('PANEL.json','WARMUPS.json','EVENT8_INPUTS.json','EXPECTED_ANCESTRY.json',
                 'EXPOSURE.json','EXCLUSIONS.json','DATA_RECEIPT.json','SEAL.json'):
        shutil.copyfile(data/name,sealed/name)
    for name in ('REUSED_REFERENCES.json','WARMUP_REFERENCES.json','MISSING_REFERENCES.json',
                 'REUSE_RECEIPT.json','REUSE_PROVENANCE.json','INSTALLED_SOURCE_IDENTITIES.json'):
        shutil.copyfile(references/name,sealed/name)
    resolve_missing_requests(json.loads((sealed/'PANEL.json').read_text()),
        json.loads((sealed/'MISSING_REFERENCES.json').read_text()),
        json.loads((sealed/'REUSED_REFERENCES.json').read_text()))
    shutil.copyfile(references/'RUNTIME_READINESS.md',sealed/'RUNTIME_READINESS.md')
    shutil.copyfile(task/'v2_reviews/HISTORICAL_COST_BASIS.json', sealed/'HISTORICAL_COST_BASIS.json')
    prior_raw = research/'dflash_native_update_size_20261115/delivery/execution_003/raw'
    shutil.copyfile(prior_raw/'PAPER_KEEPERS_DRAFT.json', sealed/'HISTORICAL_PAPER_KEEPERS.json')
    shutil.copyfile(research/'dflash_native_update_size_20261115/storage_closeout/STORAGE_BEFORE_AFTER.json',
                    sealed/'PR48_ALREADY_COMPLETED_CLEANUP.json')
    (destination/'references').mkdir()
    shutil.copyfile(references/'REUSED_REFERENCES.json',destination/'references/CANONICALS.json')
    shutil.copyfile(task/'HANDOFF_V2.txt',sealed/'HANDOFF_V2.txt')
    shutil.copyfile(task/'v2_data/ANCESTRY_RECEIPT.json',sealed/'ANCESTRY_RECEIPT.json')
    shutil.copyfile(research/'dflash_native_update_size_20261115/delivery/execution_003/raw/training/FINAL_NATIVE_PARITY.json',sealed/'N7_PARITY.json')
    original = yaml.safe_load((references/'ORIGINAL_CONFIG.yaml').read_text())
    configs={}
    for arm in ('N','E','S'):
        cfg = dict(original)
        cfg['training'] = dict(original['training'],learning_rate=arm_lr(arm))
        cfg.update(run_id='persistent-prefix-v2-'+arm,
                   output_dir=remote_root+'/training/'+arm)
        path=sealed/('config_'+arm+'.yaml')
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        configs[arm]=remote_root+'/sealed/'+path.name
    inputs=json.loads((sealed/'EVENT8_INPUTS.json').read_text())['requests']
    origin_cfg=inspection['dependencies']['origin_export']['config.json']['content']
    budget=storage_plan([len(x['prompt_ids']) for x in inputs],
        weight_bytes=old['checkpoints']['K6']['weights_identity']['bytes'],
        recovery_bytes=max(inspection['dependencies'][k]['identity']['bytes'] for k in ('origin_state','historical_native_state')),
        hidden_size=origin_cfg['hidden_size'],captured_layers=len(origin_cfg['dflash_config']['target_layer_ids']),
        prior_allocation_bytes=0)
    fields = dict(root=remote_root,source_root=source_root,execution_commit=execution_commit,
        source_manifest=remote_root+'/SOURCE.json',training_python=old['training_python'],
        worker_python=old['worker_python'],target=old['target_path'],configs=configs,
        origin_state=old['origin_state'],origin_export=old['origin_export'],origin_config=old['origin_config'],
        native_n7_state=old['historical_native_state'],native_n7_export=old['historical_native_export'],
        event7_feature_paths=old['feature_paths'],event7_schedule=old['schedule'],
        runtime_contract=old['runtime_contract'],storage=budget,prior_allocation_bytes=0,
        limits=LIMITS,automatic_retry=False,scope='DEVELOPMENT_FEASIBILITY_REUSED_DATA',
        devices={'A':'GPU-9986735a-3e68-d1bd-5998-9634ef24341e','B':'GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1'})
    files=dict(data_panel='PANEL.json',warmups='WARMUPS.json',event8_inputs='EVENT8_INPUTS.json',
               expected_ancestry='EXPECTED_ANCESTRY.json',warmup_references='WARMUP_REFERENCES.json',
               reused_references='REUSED_REFERENCES.json',missing_references='MISSING_REFERENCES.json',
               n7_parity_receipt='N7_PARITY.json',installed_runtime_identities='INSTALLED_SOURCE_IDENTITIES.json')
    fields.update({key:remote_root+'/sealed/'+name for key,name in files.items()})
    fields['references']=remote_root+'/references/CANONICALS.json'
    fields['historical_cost_basis'] = remote_root+'/sealed/HISTORICAL_COST_BASIS.json'
    fields['allocation_roots'] = [remote_root, str(Path(source_root).parent)]
    fields['compiler_growth_reserved_bytes'] = budget['compiler_bytes']
    fields['compiler_root'] = '/data/sources/minhao/.cache'
    fields['qualification_receipts'] = {
        'A': remote_root+'/qualification/A/STATUS.json',
        'B': remote_root+'/qualification/B/STATUS.json',
        'cross_card': remote_root+'/qualification/CROSS_CARD.json',
        'serving': remote_root+'/serving_qualification/CROSS_CARD.json',
    }
    fields['cpu_gate'] = remote_root+'/CPU_PREFLIGHT.json'
    fields['reference_provenance'] = remote_root+'/sealed/REFERENCE_PROVENANCE_GATE.json'
    fields['resource_admission'] = remote_root+'/RESOURCE_ADMISSION.json'
    provenance = dict(status='passed', execution_commit=execution_commit,
        scope='Source/retained-reference authentication; native qualification remains required',
        source_review=remote_root+'/sealed/RUNTIME_READINESS.md',
        reference_reuse_receipt=remote_root+'/sealed/REUSE_RECEIPT.json',
        reference_provenance=remote_root+'/sealed/REUSE_PROVENANCE.json',
        new_reference_route='fixed_original_K6_native_greedy_target_verifier',
        no_target_only_AR_claim=True)
    (sealed/'REFERENCE_PROVENANCE_GATE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    required={old[k] for k in ('origin_state','origin_config','schedule','historical_native_state')}
    required.update(old['feature_paths'])
    for path in old['live_input_identities']:
        prefixes = [old[key] for key in ('origin_export','historical_native_export','target_path')]
        prefixes.append(original['model']['draft_checkpoint_path'])
        if any(path.startswith(prefix+'/') for prefix in prefixes):
            required.add(path)
    fields['input_identities']={path:old['live_input_identities'][path] for path in sorted(required)}
    for path in sealed.iterdir():
        if path.is_file():fields['input_identities'][remote_root+'/sealed/'+path.name]=identity(path)
    # Bind the scientific preparation independently of the code publication hash.
    binding_fields={key:fields[key] for key in ('devices','limits','configs','origin_state','native_n7_state','input_identities')}
    fields['binding']=hashlib.sha256(json.dumps(binding_fields,sort_keys=True).encode()).hexdigest()
    (sealed/'PLAN.json').write_text(json.dumps(fields,indent=2)+'\n')
    return fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace');parser.add_argument('destination')
    parser.add_argument('--remote-root',required=True);parser.add_argument('--source-root',required=True)
    parser.add_argument('--execution-commit',required=True)
    args=parser.parse_args()
    result=build(args.workspace,args.destination,remote_root=args.remote_root,
                 source_root=args.source_root,execution_commit=args.execution_commit)
    print(json.dumps(dict(status='prepared',binding=result['binding'],storage=result['storage'])))


if __name__=='__main__':
    main()
