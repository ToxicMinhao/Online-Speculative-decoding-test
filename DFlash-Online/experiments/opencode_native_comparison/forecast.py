"""Resource forecasts from qualification and the unscored complete CODE panel."""
from pathlib import Path
import math
import statistics
import time

from online_feedback.storage import write_json
from .contract import qualification_root, ARMS
from .runtime import read, forecast


def measured_seconds(value,label):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Invalid measured duration: '+label)
    return value


def measurements(plan):
    from .qualification import CASES
    root=Path(plan['root'])
    native={'F':[],'AR':[]}
    for card in ('A','B'):
        folder=root/'precheck'/card
        work=read(folder/'WORK.json')
        for entry in work.values():
            if entry['status']!='valid':
                raise ValueError('Forecast only after the complete AR/F precheck passes')
            record=read(entry['record_path'])
            measured_seconds(record['metadata']['request_latency_s'],'native request')
            native[entry['method']].append(record)
    if any(len(rows)!=1024 for rows in native.values()):
        raise ValueError('Use the whole unscored panel, not only short examples')
    load=[]
    for path in (root/'precheck').rglob('LIFECYCLE.json'):
        lifecycle=read(path)
        if lifecycle['status']!='loaded':
            raise ValueError('Forecast requires completed engine initialization')
        load.append(measured_seconds(lifecycle['loading_s'],'engine initialization'))
    if len(load)!=4:
        raise ValueError('Forecast requires both AR/F initialization receipts on both cards')
    teacher=[]
    features=read(qualification_root(plan)/'A/FEATURES.json')
    if len(features)!=16:
        raise ValueError('Forecast requires all16 retained teacher-preparation receipts')
    for value in features.values():
        receipt=value['receipt']
        if receipt['status']!='prepared':
            raise ValueError('Forecast requires completed teacher preparation')
        teacher.append(sum(measured_seconds(receipt[key],key)
            for key in ('teacher_preparation_s','feature_persistence_s')))
    training={}
    for arm in ARMS:
        detail=read(qualification_root(plan)/arm/'DETAILS.json')
        # The synthetic full-capacity pair is deliberately included. Its
        # uniform teacher is a numerical/resource fixture, not scientific data.
        cases=detail['cases']
        if (detail['status']!='passed' or [row['case'] for row in cases]!=list(CASES[arm])
                or any(row['status']!='completed' or row['checks']!='passed' for row in cases)):
            raise ValueError('Forecast requires every bounded training control')
        training[arm]=max(measured_seconds(row['observer_adjusted_training_s'],arm+' pair') for row in cases)
    return dict(native=native,loading=max(load),teacher=max(teacher),training=training)


def remaining(plan, *, completed_arms=()):
    """Conservative planning estimate; never a promise or an efficacy gate.

    Before the first trajectory, retain the historical ten-hour/arm floor and
    the larger estimate from the actual CODE panel plus full-context controls.
    After an arm, also retain its observed total rather than assuming later
    arms are cheaper. Persistence/retirement are explicitly reserved.
    """
    root=Path(plan['root'])
    if tuple(completed_arms)!=ARMS[:len(completed_arms)]:
        raise ValueError('Completed arms must be the exact fixed-order prefix')
    measured=measurements(plan)
    mean=lambda method:statistics.mean(r['metadata']['request_latency_s'] for r in measured['native'][method])
    observed=[]
    for arm in completed_arms:
        status=read(root/'campaign_001/arms'/arm/'STATUS.json')
        if status['status']!='completed':
            raise ValueError('Resource forecast cannot call an incomplete arm complete')
        observed.append(measured_seconds(status['worker_wall_s'],'completed trajectory'))
    estimates={}
    for arm in ARMS:
        if arm in completed_arms:
            continue
        components=dict(feedback_s=7424*mean('F'),teacher_preparation_s=7424*measured['teacher'],
            native_training_s=7424*measured['training'][arm],
            persistence_validation_retirement_s=58*120,initialization_s=2*measured['loading'])
        estimates[arm]=dict(components=components,raw_s=max(sum(components.values()),36000,*observed))
    # Four speculative endpoints use the measured F rate as a prediction only.
    # No improvement is assumed. Switch allowance includes all120 activations,
    # even though some are draft-only publications, plus the240 warmups.
    evaluation=(3*1024*(4*mean('F')+mean('AR'))
                +120*measured['loading']+240*max(mean('F'),mean('AR')))
    analysis=1800.
    training=sum(value['raw_s'] for value in estimates.values())
    result=dict(created_epoch=time.time(),completed_arms=list(completed_arms),
        per_arm=estimates,final_evaluation_s=evaluation,audit_packaging_s=analysis,
        assumptions=['same sealed input distribution; future outputs and costs remain uncertain',
          'no adapted serving speedup assumed','full-capacity synthetic training included',
          'two occupied cards per online trajectory; one during scored evaluation',
          'two minutes/event for checkpoint persistence, authentication and retirement',
          'historical ten-hour floor per arm; completed arm times form an additional floor'],
        lengths={method:dict(mean=statistics.mean(len(r['response_ids']) for r in rows),
                    maximum=max(len(r['response_ids']) for r in rows),
                    capped=sum(r['finish_reason']=='length' for r in rows))
                 for method,rows in measured['native'].items()},
        remaining_elapsed_s=training+evaluation+analysis,
        remaining_card_s=2*training+evaluation)
    name=f'AFTER_{len(completed_arms)}_ARMS.json'
    write_json(root/'forecasts'/name,result)
    try:
        result['admission']=forecast(plan,result['remaining_elapsed_s'],result['remaining_card_s'])
        result['status']='admitted'
    except BaseException as error:
        result.update(status='blocked',error=repr(error))
        raise
    finally:
        write_json(root/'forecasts'/name,result)
    return result


def at_training_boundary(plan,arm,event,worker_elapsed_s):
    """Refresh time admission at safe boundaries; no model efficacy enters it."""
    index=ARMS.index(arm)
    root=Path(plan['root'])
    prior=read(root/'forecasts'/f'AFTER_{index}_ARMS.json')
    full_arm=max(prior['per_arm'][arm]['raw_s'],worker_elapsed_s*58/event)
    own_remaining=max(0.,full_arm-worker_elapsed_s)
    later=sum(max(value['raw_s'],full_arm) for key,value in prior['per_arm'].items() if key!=arm)
    evaluation=prior['final_evaluation_s']
    result=dict(arm=arm,event=event,worker_elapsed_s=worker_elapsed_s,
        projected_full_arm_s=full_arm,remaining_training_s=own_remaining+later,
        final_evaluation_s=evaluation,audit_packaging_s=prior['audit_packaging_s'])
    destination=root/'campaign_001/arms'/arm/'forecasts'/f'event_{event:03d}.json'
    write_json(destination,result)
    try:
        result['admission']=forecast(plan,own_remaining+later+evaluation+prior['audit_packaging_s'],
                                     2*(own_remaining+later)+evaluation)
        result['status']='admitted'
    except BaseException as error:
        result.update(status='blocked',error=repr(error))
        raise
    finally:
        write_json(destination,result)
    return result
