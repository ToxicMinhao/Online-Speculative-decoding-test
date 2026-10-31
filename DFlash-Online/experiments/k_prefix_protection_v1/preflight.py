"""Read-only original-artifact audit and capacity checks before any new model work."""
import argparse
import gc
import json
from pathlib import Path
import time

from online_feedback.storage import write_json
from experiments.ok_coding import report as old_report
from experiments.ok_coding.contract import training_contract
from experiments.ok_coding.preflight import verify_seal
from experiments.ok_events_5_8.campaign import verify_pool
from experiments.ok_events_5_8.storage import identity
from .contract import ORIGINAL_COMMIT, SEEDS, validate_plan


def read(path):
    return json.loads(Path(path).read_text())


def inventory(original, output):
    """Bind original data/state bytes, natural schedules, and native validity on CPU."""
    import torch
    from experiments.ok_coding.training import load_snapshot
    original, output = Path(original), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', original_run=str(original), model_work=0)
    started = time.perf_counter()
    try:
        plan = read(original/'sealed/PLAN.json')
        if plan['execution_source']['commit'] != ORIGINAL_COMMIT:
            raise ValueError('Not the completed original coding execution')
        receipt['original_source'] = verify_seal(original)
        training, errors = old_report.audit_training(original, plan)
        stored = read(original/'final_report/TRAINING_AUDIT.json')
        if errors or stored['errors'] or len(training) != 16:
            raise ValueError('Original numerical/training audit is incomplete: '+str(errors))
        receipt['training_audit'] = {'events':len(training), 'errors':errors}
        receipt['native_audit'] = {}
        for seed,event in ((17,2),(17,4),(29,2),(29,4)):
            rows, invalid = old_report.audit_cell(original, plan, seed, event)
            receipt['native_audit'][f'{seed}/{event}'] = {'requests':len(rows),'errors':invalid}
            if invalid or len(rows) != 128:
                raise ValueError('Original native validity failed: '+str(invalid[:10]))
        files = {}
        for name in ('TRAINING_AUDIT.json','TRAINING_COSTS.csv','SUMMARY.json','PAIRED_REQUESTS.csv','ALL_REPEATS.csv','REPORT.md'):
            p=original/'final_report'/name
            files[str(p)] = identity(p)
        for p in (original/'campaign_001/COSTS.json', original/'sealed/PLAN.json'):
            files[str(p)]=identity(p)
        pools, schedules, initial, endpoints, contracts, metadata = {},{},{},{},{},[]
        for seed in SEEDS:
            key=str(seed)
            base=original/'campaign_001'/f'seed_{seed}'
            contracts[key] = training_contract(plan, seed)
            initial[key] = str(base/'K/initial_state.pt')
            endpoints[key] = {'state':str(base/'K/events/event_004/training_state.pt'),
                              'export':str(base/'K/events/event_004/export')}
            for event, state, export in ((0,initial[key],plan['draft']),
                                        (4,endpoints[key]['state'],endpoints[key]['export'])):
                decoded, check = load_snapshot(state, export, arm='K', seed=seed,
                    campaign_contract=contracts[key], config_text=Path(plan['configs'][key]).read_text(),
                    completed_event=event)
                receipt.setdefault('states',{})[f'{seed}/{event}'] = check
                del decoded
                gc.collect()
                files[state] = identity(state)
                for p in Path(export).glob('*'):
                    if p.is_file(): files[str(p)]=identity(p)
            pools[key], schedules[key] = {},{}
            for event in range(1,5):
                poolpath=base/f'event_{event:03d}/POOL.json'
                pool=read(poolpath)
                verify_pool(pool)
                files[str(poolpath)] = identity(poolpath)
                for path, expected in pool['identities'].items():
                    files[path] = expected
                for path,expected in pool['retained_source_identities'].items():
                    if identity(path) != expected:
                        raise ValueError('Native/terminal source identity changed: '+path)
                    files[path] = expected
                schedule=base/'O/events'/f'event_{event:03d}/RESULT.json'
                result=read(schedule)
                files[str(schedule)] = identity(schedule)
                k_receipt=base/'K/events'/f'event_{event:03d}/RESULT.json'
                files[str(k_receipt)] = identity(k_receipt)
                pools[key][str(event)] = {'path':str(poolpath),'identity':files[str(poolpath)]}
                schedules[key][str(event)] = {'path':str(schedule),'identity':files[str(schedule)]}
                for ordinal, feature in enumerate(result['features']):
                    path=Path(feature['path'])
                    small=read(path.with_suffix('.json'))
                    if (len(small['input_ids']) != feature['length']
                            or small['record_id'] != feature['record_id']
                            or small['prompt_length'] != feature['prompt_length']):
                        raise ValueError('Prepared metadata and training receipt differ')
                    files[str(path.with_suffix('.json'))] = identity(path.with_suffix('.json'))
                    metadata.append(dict(seed=seed,event=event,ordinal=ordinal,
                        path=str(path),record_id=feature['record_id'],length=feature['length']))
        ordered=sorted(metadata,key=lambda r:(r['seed'],r['event'],r['ordinal']))
        longest=min(ordered,key=lambda r:(-r['length'],r['seed'],r['event'],r['ordinal']))
        earliest=next(r for r in ordered if r['path'] != longest['path'])
        receipt.update(status='passed', files=files, original_configs=plan['configs'],
            original_campaign_contracts=contracts, original_initial_states=initial,
            original_k4=endpoints, pools=pools, original_schedules=schedules,
            qualification={'state':endpoints['17']['state'],'export':endpoints['17']['export'],
                'original_config':plan['configs']['17'],
                'records':[earliest['path'],longest['path']], 'record_metadata':[earliest,longest]},
            original_data_binding=plan['data_binding'], target=plan['target'], draft=plan['draft'])
        if torch.cuda.is_initialized():
            raise ValueError('CPU inventory unexpectedly initialized CUDA')
    except BaseException as error:
        receipt.update(status='failed',error=repr(error))
        raise
    finally:
        receipt['wall_s']=time.perf_counter()-started
        write_json(output/'ORIGINAL_ARTIFACT_AUDIT.json',receipt)
    return receipt


def check_configs(plan):
    """The new arms may change only declared identity and K_HALF learning rate."""
    from copy import deepcopy
    import yaml
    from .training import validate_config_text
    receipts = {}
    for seed, configs in plan['configs'].items():
        original = yaml.safe_load(Path(plan['original_configs'][seed]).read_text())
        for arm, path in configs.items():
            text = Path(path).read_text()
            validate_config_text(text, int(seed), arm)
            current = yaml.safe_load(text)
            expected = deepcopy(original)
            for name in ('run_id', 'output_dir'):
                expected[name] = current[name]
            expected['training']['learning_rate'] = 1.5e-4 if arm == 'K_HALF' else 3e-4
            if current != expected:
                raise ValueError('Unexpected configuration change: '+seed+'/'+arm)
            if (current['model']['draft_checkpoint_path'] != plan['draft']
                    or current['model']['target_model_path'] != plan['target']):
                raise ValueError('Model paths do not match sealed pins')
            receipts[seed+'/'+arm] = identity(path)
    return receipts


def verify(root, *, postflight=False, receipt_path=None, phase=None):
    from experiments.kw_v1.preflight import hardware_admission
    from experiments.osd_components.prepare import verify_files, verify_native_runtime
    from .storage import admit
    from .training import validate_config_text
    import platform
    import torch
    import transformers
    root=Path(root)
    receipt={'status':'running','model_work':0,'postflight':postflight,
             'phase':phase or ('postflight' if postflight else 'preflight')}
    started=time.perf_counter()
    try:
        receipt['seal']=verify_seal(root)
        plan=read(root/'sealed/PLAN.json')
        receipt['plan']=validate_plan(plan)
        receipt['configs']=check_configs(plan)
        for path,expected in read(root/'sealed/ORIGINAL_ARTIFACT_AUDIT.json')['files'].items():
            if identity(path) != expected:
                raise ValueError('Reused original artifact changed: '+path)
        pins=read(root/'sealed/MODEL_PIN.json')
        for model in ('target','draft'):
            verify_files(plan[model],pins[model]['files'],exact_files=True)
        receipt['native_runtime']=verify_native_runtime(plan['worker_python'],read(root/'sealed/RUNTIME_PIN.json'))
        actual=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)
        if actual != read(root/'sealed/TRAINING_PIN.json')['environment'] or torch.cuda.is_initialized():
            raise ValueError('CPU runtime admission mismatch')
        receipt['training_runtime']=actual
        if not postflight:
            receipt['hardware']=hardware_admission()
            receipt['storage']=admit(root)
        receipt['status']='passed'
        return receipt
    except BaseException as error:
        receipt.update(status='failed',error=repr(error))
        raise
    finally:
        receipt['wall_s']=time.perf_counter()-started
        write_json(Path(receipt_path) if receipt_path is not None else
                   root/('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'),receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--inventory-output')
    parser.add_argument('--postflight',action='store_true')
    args=parser.parse_args()
    if args.inventory_output:
        result=inventory(args.root,args.inventory_output)
    else:
        result=verify(args.root,postflight=args.postflight)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','error','wall_s')}))

if __name__=='__main__':
    main()
