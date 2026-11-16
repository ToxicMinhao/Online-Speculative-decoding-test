"""Five disposable native Adam applications; exact own-state replay, no tuning."""
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from .runtime import WorkLedger


def qualify(plan):
    import torch
    from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.persistent_prefix_withholding.qualification import scientific_identity, tensor_identity, _assert_equal
    from experiments.recent_repair_retention.training import schedule_adapter
    from online_feedback.presentation_schedule import PresentationSchedule
    from online_feedback.training import SpecForgeEventTrainer
    from online_feedback.rng import preserve_rng
    from specforge.export.to_hf import export_to_hf
    from .training import CandidateTrainer, fork_state
    from .runtime import read

    base=Path(plan['root'])/'qualification/training'
    base.mkdir(parents=True,exist_ok=False)
    ledger=WorkLedger(plan['root'],'qualification_training')
    report=dict(status='running',cases=[],execution_commit=plan['execution_commit'],planned_adam=5,
        startup_separate=True,started_epoch=time.time())
    trainer=None
    failure=None
    write_json(base/'STATUS.json',report)
    try:
        pin_retained_graph_runtime(base)
        loading_started=time.perf_counter()
        trainer=ledger.call({'training_initialization':1},'load',CandidateTrainer,
            plan['configs']['C'],base/'trainer',binding=plan['binding'],
            origin_state=plan['origin_state'],origin_export=plan['origin_export'],
            original_config_text=Path(plan['origin_config']).read_text(),completed_event=6,
            schedule=read(plan['event7_schedule']),qualification_only=True)
        report['loading_s']=time.perf_counter()-loading_started
        ledger.wrap(trainer.strategy,'forward_loss','response_forward')
        ledger.wrap(trainer.backend,'backward','response_backward')
        ledger.wrap(trainer.backend.optimizer.optimizer,'step','adam')
        report['startup']=discarded_startup(trainer,plan['event7_feature_paths'],base/'startup')
        report['startup_s']=report['startup']['wall_s']
        source=torch.load(plan['origin_state'],map_location='cpu',weights_only=False,mmap=True)
        origin=fork_state(source,binding=plan['binding'],config_text=trainer.config_text,run_id=trainer.cfg.run_id)
        expected=scientific_identity(trainer)
        runtime=runtime_state(trainer.bundle.model)
        features=SpecForgeEventTrainer._prepare_event(trainer,'qualification_features',plan['event7_feature_paths'],2)

        def reset(cap,label,*,draw_order=True):
            trainer.backend.optimizer.zero_grad()
            trainer.core._ratio_totals.clear();trainer.core._sum_totals.clear()
            trainer.qualification_allocation(cap)
            trainer._load_scientific_state(origin)
            trainer.response_records.clear();trainer.window_records.clear();trainer._anchor_observations.clear()
            trainer._active_event=label
            (trainer.events_root/label).mkdir(exist_ok=True)
            trainer._presentation_schedule=(PresentationSchedule(schedule_adapter(trainer.schedule,features),
                [x['record_id'] for x in features],2,features=features) if draw_order else None)
            restore_runtime(trainer.bundle.model,runtime)
            trainer.bundle.model.begin_event()
            _assert_equal(scientific_identity(trainer),expected,'Exact initial scientific restoration')
            _assert_equal(runtime_state(trainer.bundle.model),runtime,'Initial mode/buffer/cache restoration')
            return torch.randperm(len(features)).tolist() if draw_order else None

        def run_pair(label,pair,ordinal):
            row=dict(case=label,status='attempted',pair=[x['record_id'] for x in pair])
            report['cases'].append(row);write_json(base/'STATUS.json',report)
            observation_s=0.
            block_observation_s=0.
            in_original_forward=False
            row['block_inputs']=[]
            def observe(current):
                nonlocal observation_s
                began=time.perf_counter()
                row['preclip']={n:tensor_identity(p.grad) for n,p in current.bundle.draft_model.named_parameters() if p.requires_grad}
                observation_s+=time.perf_counter()-began
            def observe_block(proposals,labels,weights,selection):
                nonlocal block_observation_s
                if not in_original_forward:
                    return
                began=time.perf_counter()
                row['block_inputs'].append({name:tensor_identity(value) for name,value in (
                    ('proposals',proposals),('labels',labels),('weights',weights),('selection',selection))})
                block_observation_s+=time.perf_counter()-began
            model=trainer.bundle.model
            had_observer=hasattr(model,'block_observer')
            prior_observer=getattr(model,'block_observer',None)
            if prior_observer is not None:
                raise ValueError('Qualification requires an unused block observation hook')
            forward=trainer.strategy.forward_loss
            def original_forward(*args,**kwargs):
                nonlocal in_original_forward
                in_original_forward=True
                try:
                    return forward(*args,**kwargs)
                finally:
                    in_original_forward=False
            trainer.strategy.forward_loss=original_forward
            model.block_observer=observe_block
            trainer.step_observer=observe
            started=time.perf_counter()
            try:
                rows=trainer.run_pair(pair,0,ordinal)
            finally:
                trainer.step_observer=None
                trainer.strategy.forward_loss=forward
                if had_observer:
                    model.block_observer=prior_observer
                else:
                    delattr(model,'block_observer')
            pair_wall_s=time.perf_counter()-started
            if not row['block_inputs']:
                raise ValueError('Qualification captured no original-forward block inputs')
            # Additional coefficient diagnostics cannot alter native comparisons.
            def native_metrics(values):return {k:v for k,v in values.items() if not k.startswith('prefix_allocation_')}
            row.update(status='completed',wall_s=pair_wall_s,preclip_observer_s=observation_s,
                block_observer_s=block_observation_s,
                pair_wall_excluding_preclip_observer_s=pair_wall_s-observation_s,
                pair_wall_excluding_observers_s=pair_wall_s-observation_s-block_observation_s,forward=[dict(
                record_id=x['record_id'],anchors=x['sampled_anchors'],
                loss=x['response_observation']['loss'],
                sums=native_metrics(x['response_observation']['sum_metrics']),
                ratios=x['response_observation']['ratio_metrics'],metrics=native_metrics(x['metrics'])) for x in rows],
                allocation=[x['response_observation']['prefix_allocation'] for x in rows],
                endpoint=scientific_identity(trainer))
            write_json(base/'STATUS.json',report)
            return row

        baseline=None
        for label,cap in (('native',None),('cap_zero',0.),('active',.25)):
            order=reset(cap,label);pair=[features[i] for i in order[:2]]
            row=run_pair(label,pair,0)
            if baseline is None:baseline=row
            elif cap==0:
                for key in ('forward','preclip','endpoint','block_inputs'):_assert_equal(row[key],baseline[key],'Native/cap0 '+key)
                row['native_equivalence']='exact'
            else:
                _assert_equal(row['block_inputs'],baseline['block_inputs'],'Active original proposals/labels/weights/selection')
                for actual,reference in zip(row['forward'],baseline['forward']):
                    keep=lambda values:{k:v for k,v in values.items() if k!='main_loss_numerator'}
                    _assert_equal(keep(actual['sums']),keep(reference['sums']),'Active native first-error/support scalars')
                row['unchanged_native_inputs_and_first_error_terms']=True
                movement=sum(x['changed_positions'] for x in row['allocation'])
                changed=[n for n,v in row['endpoint']['weights'].items() if v!=baseline['endpoint']['weights'][n]]
                row.update(changed_coefficient_occurrences=movement,changed_parameter_tensors=len(changed))
                if movement<=0 or not changed:raise ValueError('Qualification applied no actual candidate intervention')
            write_json(base/'STATUS.json',report)
        # This actual post-first-pair state is the only disposable saved model.
        snapshot=base/'active_state.pt'
        report['snapshot']=trainer.save_qualification(snapshot)
        saved_runtime=runtime_state(trainer.bundle.model)
        export=base/'active_export'
        config=Path(trainer.cfg.model.draft_model_config)
        if config.is_dir():config/= 'config.json'
        with preserve_rng():
            export_to_hf(str(snapshot),str(config),str(export),
                embedding_source=trainer.cfg.model.target_model_path,embedding_key=trainer.cfg.model.embedding_key)
        report['export']=str(export)
        pair=[features[i] for i in order[2:4]]
        continued=run_pair('continued_next_pair',pair,2)
        report['restore']=trainer.restore_qualification(snapshot,runtime=saved_runtime)
        replay=run_pair('reloaded_next_pair',pair,2)
        for key in ('forward','preclip','endpoint','allocation','block_inputs'):
            _assert_equal(replay[key],continued[key],'Serialized own-state next update '+key)
        reset(.25,'restored_origin',draw_order=False)
        _assert_equal(scientific_identity(trainer),expected,'Final original-state restoration')
        report.update(status='passed',optimizer_applications=5,response_presentations=10,
            startup_forward_backward_each=2,exact_native_cap0=True,exact_own_resume=True,
            actual_intervention=True,origin_restored=True,physical_work=dict(trainer.work))
    except BaseException as error:
        failure=error
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        if trainer is not None:trainer.failed=True
        raise
    finally:
        if trainer is not None:
            trainer.step_observer=None
            report['peak_gpu_allocated_bytes']=int(torch.cuda.max_memory_allocated())
            report['peak_gpu_reserved_bytes']=int(torch.cuda.max_memory_reserved())
            report['physical_work']=dict(trainer.work)
        report['wall_s']=time.time()-report['started_epoch']
        write_json(base/'STATUS.json',report)
        if trainer is not None:
            try:
                trainer.close()
            except BaseException as error:
                report['cleanup_error']=repr(error)
                if failure is None:
                    report.update(status='failed',error=repr(error))
                write_json(base/'STATUS.json',report)
                if failure is None:
                    raise
                failure.add_note('Qualification cleanup also failed: '+repr(error))
    return report
