"""Seal actual CODE paths and identities after input selection, before model work."""
import argparse
from pathlib import Path

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from .configuration import build_config
from .contract import (ARMS, METHODS, SCHEMA, SERVING, RESOURCE_LIMITS,
                       PRODUCTION_WORK, QUALIFICATION_CEILINGS, digest, validate_manifest)
from .qualification import qualification_slots as training_slots
from .serving_gate import qualification_slots as serving_slots
from .runtime import read
from .schedule import evaluation_schedule, schedule_summary


def prepare(root, source_root, commit, storage_path):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    destination = root/'sealed/PLAN.json'
    if destination.exists() or (root/'CLOCK.json').exists():
        raise FileExistsError('Never replace a sealed or started execution plan')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Require the complete published execution commit')
    source = read(root/'SOURCE.json')
    if source['commit']!=commit or source['dirty']:
        raise ValueError('Stage the clean published source before sealing its plan')
    sealed = root/'sealed'
    inputs = read(sealed/'INPUTS.json')
    validate_manifest(inputs)
    model_pin = read(sealed/'MODEL_PIN.json')
    foundation = Path('/data/sources/minhao/models/qwen3_dflash_foundation_20261022')
    configs = {}
    for arm in ARMS:
        path = sealed/f'config_{arm}.yaml'
        path.write_text(build_config((sealed/'ORIGINAL_CONFIG.yaml').read_text(), arm=arm,
            output_dir=root/'campaign_001/arms'/arm/'trainer'))
        configs[arm] = str(path)
    initial = Path('/data/sources/minhao/runs/dflash_full_k_gsm8k_20261031/production_002/campaign_001/arms/K/trainer/initial_state.pt')
    authenticated = [dict(path=str(initial),role='historical_released_initial_witness',
                           expected_identity=identity(initial))]
    for role in ('target','draft'):
        for name,pin in model_pin[role]['files'].items():
            path = foundation/role/name
            if identity(path)!=pin:
                raise ValueError('Released model/tokenizer identity changed: '+str(path))
            authenticated.append(dict(path=str(path),role='released_'+role,expected_identity=pin))
    qualification = dict(serving=serving_slots(inputs), training=training_slots(),
        publication=dict(generations=4,optimizer_steps=0),
        exact_totals=dict(generation=64,adam=17,response_forward=40,response_backward=40,
                          teacher_preparation=16,training_initialization=3))
    write_json(sealed/'QUALIFICATION_PLAN.json',qualification)
    schedule=evaluation_schedule(inputs['evaluation_requests'])
    write_json(sealed/'SCHEDULE.json',dict(activations=schedule,balance=schedule_summary(schedule)))
    pair = sorted(inputs['qualification_requests'],key=lambda r:(-len(r['prompt_ids']),r['request_index']))[:4]
    storage_plan = read(storage_path)
    if storage_plan['status']!='admitted':
        raise ValueError('Complete storage forecast has not been admitted')
    plan = dict(schema=SCHEMA,root=str(root),source_root=str(source_root),
        source_manifest=str(root/'SOURCE.json'),execution_commit=commit,
        execution_source=dict(commit=commit,dirty=False),arms=list(ARMS),methods=list(METHODS),
        seed=17,events=58,serving=SERVING,limits=RESOURCE_LIMITS,
        production_work=PRODUCTION_WORK,qualification_work=QUALIFICATION_CEILINGS,
        exact_qualification_work=qualification['exact_totals'],
        manifest=str(sealed/'INPUTS.json'),data_seal=str(sealed/'DATA_SEAL.json'),
        original_config=str(sealed/'ORIGINAL_CONFIG.yaml'),configs=configs,
        runtime_pin=str(sealed/'RUNTIME_PIN.json'),training_pin=str(sealed/'TRAINING_PIN.json'),
        model_pin=str(sealed/'MODEL_PIN.json'),initial_witness=str(initial),
        target=str(foundation/'target'),draft=str(foundation/'draft'),
        checkpoints=dict(F=str(foundation/'draft')),
        session='opencode_native_comparison_20261120',
        training_python='/data/conda_envs/minhao/dflash_specforge_20261022/bin/python',
        worker_python='/data/conda_envs/minhao/adaflash_author/bin/python',
        zstd='/data/conda_envs/minhao/dflash_specforge_20261022/bin/zstd',
        devices=dict(A='GPU-9986735a-3e68-d1bd-5998-9634ef24341e',
                     B='GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1'),
        qualification_pair_ids=[r['request_id'] for r in pair],
        authenticated_files=authenticated,
        canonical_provenance=dict(target_identity=model_pin['target'],
            tokenizer_identity={name:model_pin['target']['files'][name] for name in
                ('tokenizer.json','tokenizer_config.json','merges.txt','vocab.json')},
            runtime_identity=identity(sealed/'RUNTIME_PIN.json')),
        allocation_roots=[str(root),str(source_root.parent)],
        compiler_baseline=compiler_snapshot(),storage=storage_plan,
        automatic_retry=False,checkpoint_selection=False)
    plan['input_identities']={str(path):identity(path) for path in sorted(sealed.iterdir())
                              if path.is_file() and path!=destination}
    plan['binding']=digest(plan)
    write_json(destination,plan)
    write_json(root/'KEEP_MANIFEST.json',dict(status='sealed_before_model_work',
        historical_dependencies=authenticated,historical_retirement_authorized=False,
        inputs=plan['input_identities'],source_manifest=plan['source_manifest'],
        keep_new=['initial witnesses','qualification/failure evidence','sealed input and canonical records',
                  'exact compact request/anchor/RNG/update/trace records','source/environment pins'],
        final_recoveries=[str(root/'campaign_001/arms'/arm/'trainer/events'/f'event_{e:03d}/training_state.pt')
                          for arm in ARMS for e in (57,58)],
        final_exports=[str(root/'campaign_001/arms'/arm/'trainer/events/event_058/export') for arm in ARMS]))
    write_json(root/'RETIRE_ALLOWLIST.json',dict(status='policy_sealed_exact_entries_required_before_unlink',
        exclusive_study_root=str(root),historical_retirement_authorized=False,
        allowed_operations=['verified completed-window features','older than two complete recoveries',
            'redundant serving exports backed by complete state','verified compact archive loose duplicates',
            'disposable qualification tensors after complete qualification evidence'],
        per_operation_receipts_required=['resolved paths','ownership','bytes','hashes','retained dependencies',
                                         'lost capabilities','allocated before/after']))
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source-root','commit','storage-path'):
        parser.add_argument('--'+name,required=True)
    prepare(**vars(parser.parse_args()))
