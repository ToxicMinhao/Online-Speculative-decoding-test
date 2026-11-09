"""Four live R1 cases; reuse passed proofs without repeating their model work."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time
import traceback

import torch

from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime, persist_comparison
from experiments.prefix_ranking.qualify import override
from online_feedback.storage import write_json
from .qualify import (cleanup, physical_work, restore_origin, run_case, runtime_state,
                      scientific_state, set_observation_directory, warmup, check_serving)
from .training import FunctionalGeometryTrainer
from .work import QualificationWork

REVISION = 'norm_feasible_bf16_r1'
LIMITS = dict(adam=4, ordinary_forward=8, ordinary_backward=8,
              additional_forward=32, additional_backward=16, serving_generation=4)


def capture_reference(trainer, features, retained_rows, work, baseline, root):
    """Capture warmed ordinary gradients, without applying another Adam update.

    The real native endpoint is checked against the retained ordinary weight
    export during each live case. We do not reconstruct a retired optimizer.
    """
    from specforge.training.strategies.base import StepContext
    root.mkdir()
    restore_origin(trainer, baseline, root)
    set_observation_directory(trainer, 'warmed_reference')
    before = scientific_state(trainer)
    trainer.bundle.model.begin_event()
    gradients, rows = {}, []
    forward, backward = trainer.author_strategy.forward_loss, trainer.backend.backward
    boundaries = 0
    receipt = dict(status='running', adam=0)
    started = time.perf_counter()
    work.context = dict(dataset=trainer.dataset, case='warmed_reference')

    def counted_forward(*args, **kwargs):
        with work.operation('additional_forward', purpose='warmed_reference'):
            return forward(*args, **kwargs)

    def counted_backward(*args, **kwargs):
        with work.operation('additional_backward', purpose='warmed_reference'):
            return backward(*args, **kwargs)

    def capture_without_adam():
        nonlocal boundaries
        boundaries += 1
        gradients.update({n:None if p.grad is None else p.grad.detach().cpu().clone()
                          for n,p in trainer.bundle.draft_model.named_parameters()})
        norm = torch.nn.utils.clip_grad_norm_(trainer.backend.optimizer.model_params,
            trainer.backend.optimizer.max_grad_norm, error_if_nonfinite=True)
        trainer.backend.optimizer.zero_grad()
        return norm.detach()

    failure = None
    try:
        with override(trainer.author_strategy, 'forward_loss', counted_forward), \
             override(trainer.backend, 'backward', counted_backward), \
             override(trainer.backend.optimizer, 'step', capture_without_adam):
            for path in features:
                raw = torch.load(path, map_location='cpu', weights_only=True)
                trainer._anchor_observations.clear()
                result = trainer.core.train_step(trainer._batch(raw), StepContext(
                    global_step=512, total_steps=trainer.horizon, collect_detailed_metrics=True))
                rows.append(dict(record_id=raw['record_id'],
                    sampled_anchors=deepcopy(trainer._anchor_observations),
                    metrics=result.materialize_metrics(), optimizer_stepped=result.optimizer_stepped))
                trainer.presentations += 1
                write_json(root/'PRESENTATIONS.json', rows)
                del raw, result
        if boundaries != 1 or trainer.global_step != 512:
            raise ValueError('Reference capture must intercept exactly one boundary without Adam')
        persist_comparison(root/'RETAINED_ANCHORS_LOSSES.json', rows, retained_rows)
        after = scientific_state(trainer)
        for key in ('weights', 'optimizer', 'gradients', 'global_step'):
            persist_comparison(root/('UNCHANGED_'+key.upper()+'.json'), after[key], before[key])
        receipt['physical_work'] = physical_work(trainer)
        del after
        restore_origin(trainer, baseline, root)
        persist_comparison(root/'EXACT_RESTORATION.json', scientific_state(trainer), before)
        receipt.update(status='passed', additional_forwards=2, additional_backwards=2,
            optimizer_state_reference='first live native Adam result; never reconstructed from retired payload')
        return dict(gradients=gradients, rows=rows)
    except BaseException as error:
        failure=error;receipt.update(status='failed', error=repr(error));raise
    finally:
        receipt['wall_s']=time.perf_counter()-started
        cleanup(lambda:write_json(root/'RESULT.json',receipt), failure,receipt,'reference receipt')


def reused_oracle(_hook, root):
    """No model computation: proofs are authenticated in the sealed preflight."""
    return dict(status='reused', new_forwards=0, new_backwards=0,
                reason='unchanged formula, same-layout, restoration and maximum-resource evidence')


def activity(rows):
    fsg, scalar = [], []
    for dataset, data in rows.items():
        for case in data['cases']:
            step=case['hook']['steps'][0]
            committed=step['committed']
            entry=dict(dataset=dataset,case=case['case'],decision=step['decision'],
                nontrivial_accepted=bool(step['decision']['accepted']
                    and not committed['identical_native'] and not committed['identical_prestate']),
                final_candidate_check=step.get('final_candidate_check'),
                scalar_matching_exercised=step.get('scalar_matching_exercised',False))
            (fsg if case['case']=='FSG' else scalar).append(entry)
    fsg_active=any(x['nontrivial_accepted'] for x in fsg)
    scalar_checked=any(x['scalar_matching_exercised'] and
        x['final_candidate_check'] in ('actual_bf16_replay','exact_native_risk_reuse') for x in scalar)
    return dict(fsg=fsg,scalar=scalar,nontrivial_fsg=fsg_active,
                scalar_matching_checked=scalar_checked,mechanism_gate=fsg_active and scalar_checked)


def run(root):
    root=Path(root);plan=json.loads((root/'sealed/PLAN.json').read_text())
    if (plan.get('scope')!='bf16_realization_r1_qualification'
            or plan.get('realization_revision')!=REVISION
            or plan.get('supplementary_limits')!=LIMITS):
        raise ValueError('Require the sealed supplementary R1 contract and separate work budget')
    output=root/'qualification';output.mkdir(exist_ok=False)
    work=QualificationWork(root/'QUALIFICATION_WORK.jsonl',limits=LIMITS)
    if work.summary():
        raise ValueError('No automatic qualification retry')
    prep=json.loads((root/'qualification_feedback/PREPARATION.json').read_text())
    if prep['status']!='completed':raise ValueError('Original qualification feedback unavailable')
    summary=dict(status='running',execution_commit=plan['execution_source']['commit'],
        revision=REVISION,limits=LIMITS,datasets={},main_pilot_started=False,automatic_retry=False,
        prior_work='preserved separately; no historical counters reset or reclassified')
    started=time.perf_counter();trainer=None;failure=None
    try:
        pin_retained_graph_runtime(output)
        for dataset,item in plan['datasets'].items():
            directory=output/dataset;directory.mkdir()
            packets=prep['datasets'][dataset]
            if len(packets)!=2:raise ValueError('Exactly the retained two-response window is required')
            for row in packets:
                if identity(row['feature_path'])!=row['feature_identity']:
                    raise ValueError('Retained feature identity changed')
            features=[row['feature_path'] for row in packets]
            torch.cuda.reset_peak_memory_stats()
            trainer=FunctionalGeometryTrainer(item['configs']['KPLUS'],directory/'trainer',
                arm='KPLUS',dataset=dataset,binding=plan['binding'],
                original_config_text=Path(plan['original_config']).read_text(),
                origin_state=item['origin_state'],origin_export=item['origin_export'],
                realization_revision=REVISION)
            baseline=runtime_state(trainer.bundle.model)
            work.context=dict(dataset=dataset,case='discarded_startup')
            startup=directory/'startup';startup.mkdir()
            warmup(trainer,features,work,baseline,startup)
            ordinary=plan['supplementary_references'][dataset]
            reference=capture_reference(trainer,features,
                json.loads(Path(ordinary['presentations']).read_text()),
                work,baseline,directory/'ordinary_reference')
            cases=[];summary['datasets'][dataset]=dict(cases=cases)
            for label in ('FSG','SCALAR'):
                _,case=run_case(trainer,features,label,1.,work,baseline,directory/label,reference,
                    oracle_runner=reused_oracle,realization_revision=REVISION,
                    reference_export=ordinary['export'])
                cases.append(case);summary['work']=work.summary()
                write_json(output/'QUALIFICATION.json',summary)
            restore_origin(trainer,baseline,directory)
            trainer.close();trainer=None
            del reference,baseline
            gc.collect();torch.cuda.empty_cache()
            summary['datasets'][dataset]['serving']=check_serving(
                plan,dataset,cases,packets[0],work,directory/'serving')
        counts=work.summary()
        for kind,expected in (('adam',4),('ordinary_forward',8),('ordinary_backward',8),('serving_generation',4)):
            if (counts.get(kind+'_attempted',0)!=expected or counts.get(kind+'_completed',0)!=expected
                    or counts.get(kind+'_failed',0)):
                raise ValueError('Incomplete supplementary work: '+kind)
        summary.update(status='passed',activity=activity(summary['datasets']))
        summary['next_scope']=('conditional_pilot_requires_native_bindings_and_storage'
            if summary['activity']['mechanism_gate'] else 'stop_r1_without_additional_cases')
    except BaseException as error:
        failure=error;summary.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if trainer is not None:cleanup(trainer.close,failure,summary,'trainer close')
        summary.update(wall_s=time.perf_counter()-started,work=work.summary())
        cleanup(lambda:write_json(output/'QUALIFICATION.json',summary),failure,summary,'qualification receipt')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');args=p.parse_args()
    result=run(args.root)
    print(json.dumps({k:result[k] for k in ('status','wall_s','work','activity','next_scope')}))
