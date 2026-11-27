"""Four disposable same-state comparisons to the frozen PR65 integration.

The historical parent is a numerical fixture only. It cannot be supplied to the
production constructor, and no qualification update is recorded as a new event.
"""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import time
import traceback

import torch

from experiments.continuation_repair_pilot import qualification as reference
from experiments.continuation_repair_pilot.configuration import PARENT_BINDING
from experiments.frontier_preservation_pilot.fork import validate_parent_snapshot
from experiments.frontier_preservation_pilot.allocation_qualify import vector_identity
from experiments.frontier_preservation_pilot.allocation_integration import _endpoint_identity
from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.storage import identity
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.repair_conditional_kl_pilot import training as parent_training
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json
from specforge.training.strategies.base import StepContext

from .training import FullContinuationTrainer, CONTRACT_KEY, compare_initial_witnesses


def install_work_accounting(trainer,ledger):
    """Count only actual original-response calls, preserving recomputation separately."""
    from .runtime import WorkLedger
    startup=WorkLedger(ledger.root,ledger.task,'startup')
    for owner,name,kind in ((trainer.author_strategy,'forward_loss','response_forward'),
                            (trainer.backend,'backward','response_backward')):
        original=getattr(owner,name)
        def counted(*args,_original=original,_kind=kind,**kwargs):
            active=startup if trainer._active_event=='disposable_startup' else ledger
            return active.call({_kind:1},'disposable_'+_kind,_original,*args,**kwargs)
        setattr(owner,name,counted)


@contextmanager
def accounted_reference(ledger):
    """Add receipt-only hooks to the frozen PR65 class in this serial process."""
    original=reference.WarmContinuationTrainer
    class CountedReference(original):
        def _build_runtime(self):
            super()._build_runtime()
            install_work_accounting(self,ledger)
    reference.WarmContinuationTrainer=CountedReference
    try:yield
    finally:reference.WarmContinuationTrainer=original


def reference_plan(plan):
    """Rebind only output/run locators; the reference numerical code is frozen."""
    return dict(root=plan['root'],binding=plan['binding'],
        configs=plan['qualification_configs'],parent_plan=plan['qualification_parent_plan'],
        retained_event=plan['qualification_retained_event'],feature_manifest=plan['qualification_feature_manifest'])


def scientific_case(case):
    value=reference.scientific_case(case)
    for row in value['rows']:
        row['response_observation'].pop('effective_components',None)
    return value


def load_parent(trainer,plan):
    parent=json.loads(Path(plan['qualification_parent_plan']).read_text())
    recovery=Path(parent['root'])/'arms/PAIR/trainer/training_state.pt'
    state=torch.load(recovery,map_location='cpu',weights_only=False,mmap=True)
    contract=deepcopy(trainer.contract);del contract[CONTRACT_KEY]
    contract[parent_training.CONTRACT_KEY]=parent_training.experiment_contract('PAIR',PARENT_BINDING,1)
    checked=validate_parent_snapshot(state,parent_identity=identity(recovery),
        parent_config_text=Path(parent['configs']['PAIR']).read_text(),
        original_config_text=Path(parent['original_config']).read_text(),expected_contract=contract,
        expected_recipe=trainer.backend.optimizer.recipe,expected_cuda_rng_states=1)
    trainer.import_qualification_parent(state,checked)
    return state,checked


def run_integration_case(plan,arm,output,ledger,initial_comparator=None):
    """One actual PR65 two-response update through the new honest wrapper."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    trainer=None;parent_state=None;before_runtime=None
    case=dict(status='running',arm=arm,fixture=0,rows=[],coefficient_checks=[],
        schedule_advance_work=dict(response_forwards=0,backwards=0,optimizer_steps=0))
    started=time.perf_counter()
    try:
        load_started=time.perf_counter()
        trainer=FullContinuationTrainer(plan['configs'][arm],output/'trainer',arm=arm,binding=plan['binding'],
            original_config_text=Path(plan['original_config']).read_text(),qualification_only=True,adam_ledger=ledger)
        install_work_accounting(trainer,ledger)
        case['fresh_initialization']=trainer.save_initial(initial_comparator)
        parent_state,parent_checked=load_parent(trainer,plan)
        case['parent_authentication']=parent_checked
        case['loading_s']=time.perf_counter()-load_started
        features=json.loads(Path(plan['qualification_feature_manifest']).read_text())
        event=json.loads(Path(plan['qualification_retained_event']).read_text())
        case['startup']=parent_training.discarded_startup(trainer,[row['path'] for row in features],output/'startup')
        before_runtime=runtime_state(trainer.bundle.model)
        case['initial_state']=_endpoint_identity(trainer)
        trainer._active_event='qualification_window';(trainer.events_root/trainer._active_event).mkdir()
        trainer.bundle.model.begin_event()
        schedule=PresentationSchedule(event,[row['record_id'] for row in features],2)
        trainer._presentation_schedule=schedule
        order=torch.randperm(len(features)).tolist()
        if [features[i]['record_id'] for i in order]!=[row['record_id'] for row in event['records'][:128]]:
            raise ValueError('Natural qualification shuffle differs from the authenticated fixture')
        reference.install_preservation_capture(trainer,case)
        extension=trainer.bundle.model.continuation_repair
        if extension is not None:
            def probe(payload):
                checked=reference.coefficient_check(payload)
                checked['preservation_coefficients']=vector_identity(['coefficients','prefix_mask'],
                    [payload['preservation_coefficients'],payload['preservation_mask']])
                case['coefficient_checks'].append(checked)
                write_json(output/'RESULT.json',case)
                if checked['status']!='passed':raise ValueError('Unchanged FP64 coefficient qualification failed')
            extension.coefficient_probe=probe
        def capture_gradient(current):
            named=list(current.bundle.draft_model.named_parameters())
            case['gradient']=vector_identity([n for n,_ in named],[p.grad for _,p in named])
            write_json(output/'RESULT.json',case)
        trainer.step_observer=capture_gradient
        for ordinal in range(2):
            feature=features[order[ordinal]];schedule.begin_presentation(0,feature['record_id'])
            raw=torch.load(feature['path'],map_location='cpu',weights_only=True)
            trainer._anchor_observations.clear()
            result=trainer.core.train_step(trainer._batch(raw),StepContext(global_step=trainer.global_step,
                total_steps=trainer.horizon,collect_detailed_metrics=True))
            schedule.finish_presentation();metrics=result.materialize_metrics()
            trainer.presentations+=1;trainer.global_step+=int(result.optimizer_stepped)
            row=dict(record_id=feature['record_id'],pass_index=0,optimizer_step_after=trainer.global_step,
                optimizer_stepped=result.optimizer_stepped,metrics=metrics,
                sampled_anchors=deepcopy(trainer._anchor_observations),response_observation=deepcopy(trainer.response_records[-1]))
            if row['sampled_anchors']!=event['records'][ordinal]['sampled_anchors']:
                raise ValueError('Natural anchors differ from the authenticated numerical fixture')
            case['rows'].append(row);write_json(output/'RESULT.json',case)
        trainer._presentation_schedule=None
        if extension is not None:extension.coefficient_probe=None
        trainer.step_observer=None
        case.update(gradient_norm=trainer.window_records[-1]['gradient_norm'],endpoint=_endpoint_identity(trainer),
            actual_work=dict(trainer.work),objective_costs=trainer.bundle.model.end_event(),
            counters=[trainer.global_step,trainer.presentations])
        if case['counters']!=[129,258]:raise ValueError('Disposable update budget changed')
        snapshot=output/'disposable_state.pt'
        case['serialization']=trainer.save_qualification(snapshot)
        snapshot_identity=identity(snapshot)
        runtime=runtime_state(trainer.bundle.model)
        # Restoring the actual serialized endpoint is deliberately not a fabricated event002.
        case['serialization']['restore']=trainer.restore_qualification(snapshot,runtime=runtime)
        if _endpoint_identity(trainer)!=case['endpoint']:raise ValueError('Disposable endpoint serialization changed scientific state')
        from specforge.export.to_hf import export_to_hf
        config=Path(trainer.cfg.model.draft_model_config)
        if config.is_dir():config/='config.json'
        export=output/'export'
        with preserve_rng():
            export_to_hf(str(snapshot),str(config),str(export),embedding_source=trainer.cfg.model.target_model_path,
                embedding_key=trainer.cfg.model.embedding_key)
            _export_equal(torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True),export)
        case['export']=dict(path=str(export),identity=identity(export/'model.safetensors'),exact=True,
                            kind='disposable_one_step_numerical_fixture_not_production')
        # Keep the compact serialized identity and export needed by serving qualification.
        # This new temporary full tensor has no production recovery role.
        case['serialization'].update(identity=snapshot_identity,retired_after_exact_readback=True,
            retained_export=str(export),lost_capability='Disposable fixture optimizer/RNG restore only')
        write_json(output/'SERIALIZATION.json',case['serialization']);snapshot.unlink()
        case['status']='completed'
    except BaseException as error:
        case.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        try:
            if trainer is not None:
                case['actual_work']=dict(trainer.work)
                try:
                    if parent_state is not None and before_runtime is not None:
                        trainer.import_qualification_parent(parent_state,case['parent_authentication'])
                        restore_runtime(trainer.bundle.model,before_runtime)
                        case['restored_state']=_endpoint_identity(trainer)
                        case['restoration_exact']=case['restored_state']==case['initial_state']
                        if not case['restoration_exact']:raise ValueError('Disposable fixture did not restore its own origin')
                except BaseException as error:
                    already_failed=case['status']=='failed'
                    case.update(status='failed',restoration_error=repr(error))
                    if not already_failed:raise
                finally:
                    if case['status']!='completed':trainer.failed=True
                    try:trainer.close()
                    except BaseException as error:
                        already_failed=case['status']=='failed'
                        case.update(status='failed',close_error=repr(error))
                        if not already_failed:raise
        finally:
            case['wall_s']=time.perf_counter()-started;write_json(output/'RESULT.json',case)
    return case


def qualify_training(plan,ledger,output=None):
    root=Path(output) if output is not None else Path(plan['root'])/'qualification'
    root.mkdir(parents=True,exist_ok=True)
    pin_retained_graph_runtime(root)
    result=dict(status='running',binding=plan['binding'],execution_commit=plan['execution_commit'],cases=[],
                planned_disposable_adam=4,maximum_disposable_adam=8,production_updates=0)
    try:
        refs={};new={}
        for arm in ('BASE','TAIL'):
            old_output=root/f'reference_{arm}';old_output.mkdir()
            with accounted_reference(ledger):
                refs[arm]=reference.run_case(reference_plan(plan),arm,0,old_output,ledger)
            result['cases'].append(dict(arm=arm,implementation='frozen_PR65',path=str(old_output/'RESULT.json')))
            write_json(root/'TRAINING_QUALIFICATION.json',result)
        for arm in ('BASE','TAIL'):
            initial=root/'integration_BASE/trainer/INITIAL_WITNESS.pt' if arm=='TAIL' else None
            new[arm]=run_integration_case(plan,arm,root/f'integration_{arm}',ledger,initial)
            mismatch=differences(scientific_case(refs[arm]),scientific_case(new[arm]))
            if mismatch:raise ValueError('Full-run integration differs from frozen PR65: '+repr(mismatch[:12]))
            if refs[arm]['actual_work']!=new[arm]['actual_work']:
                raise ValueError('New wrapper changed native model/gradient operation counts')
            if refs[arm]['preservation_coefficients']!=new[arm]['preservation_coefficients']:
                raise ValueError('New wrapper changed actual C coefficients')
            result['cases'].append(dict(arm=arm,implementation='fullrun',path=str(root/f'integration_{arm}/RESULT.json'),
                scientific_exact=True,native_work_exact=True,serialization_exact=True))
            write_json(root/'TRAINING_QUALIFICATION.json',result)
        if new['BASE']['preservation_coefficients']!=new['TAIL']['preservation_coefficients']:
            raise ValueError('TAIL changed shared-state preservation coefficients')
        if len(new['TAIL']['coefficient_checks'])!=2 or any(c['status']!='passed' for c in new['TAIL']['coefficient_checks']):
            raise ValueError('TAIL coefficient oracle coverage incomplete')
        result.update(status='passed',fresh_initializations_exact=True,BASE_TAIL_PR65_integration_exact=True,
            exports={arm:new[arm]['export'] for arm in ('BASE','TAIL')},
            main_initialization='new released fresh seed17 runtime; fixture states never reused')
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:write_json(root/'TRAINING_QUALIFICATION.json',result)
    return result


def run_training_qualification(plan,output,ledger):
    return qualify_training(plan,ledger,output)


def retire_disposable_states(plan,result):
    """Retire only passed new qualification exports after their serving gate."""
    root=Path(plan['root'])/'qualification'
    serving=json.loads((root/'SERVING.json').read_text())
    if result.get('status')!='passed' or serving.get('status') not in ('passed','completed'):
        raise ValueError('Both qualification phases must pass before temporary export retirement')
    destination=root/'DISPOSABLE_RETIREMENT.json'
    if destination.exists():
        receipt=json.loads(destination.read_text())
        if receipt['status']!='completed' or any(Path(row['path']).exists() for row in receipt['retired']):
            raise ValueError('Incomplete or reappeared disposable retirement')
        return receipt
    candidates=[]
    for arm,export in result['exports'].items():
        path=Path(export['path'])/'model.safetensors'
        path.resolve().relative_to(root.resolve())
        if (arm not in ('BASE','TAIL') or path.parent.parent.name!='integration_'+arm or
            export['kind']!='disposable_one_step_numerical_fixture_not_production' or
            not export['exact'] or identity(path)!=export['identity']):
            raise ValueError('Unverified or foreign disposable export retirement')
        case=json.loads((path.parent.parent/'RESULT.json').read_text())
        if case.get('status')!='completed' or not case['restoration_exact'] or case['export']!=export:
            raise ValueError('Disposable scientific evidence is incomplete')
        candidates.append(dict(path=str(path),identity=export['identity'],arm=arm,
            retained_receipt=str(path.parent.parent/'RESULT.json'),
            reason='Passed disposable numerical and serving qualification; not a production endpoint'))
    if {row['arm'] for row in candidates}!={'BASE','TAIL'}:raise ValueError('Disposable arm membership differs')
    receipt=dict(status='retiring',retired=[],candidates=candidates,
        lost_capability='Direct serving of the disposable qualification endpoints; compact evidence and export companions retained')
    write_json(destination,receipt)
    for row in candidates:
        Path(row['path']).unlink();receipt['retired'].append(row);write_json(destination,receipt)
    receipt['status']='completed';write_json(destination,receipt)
    return receipt
