"""Pinned qualification profile and causal updates; benchmark replays never learn."""
import copy
import math
import numpy as np

BINS=(8192,16384,32768)


def profile_from_native(raw, observations):
    paths={int(p['modules']):p for p in raw['paths']}
    if set(paths)!={0,1,2} or not observations: raise ValueError('Incomplete native cost qualification')
    if any(len(paths[k]['repeat_s'])!=20 for k in paths): raise ValueError('Expected 20 profile repeats')
    increments={str(k):max(0.,float(paths[k]['mean_s'])-float(paths[0]['mean_s'])) for k in paths}
    # Paired gain claims never use this model. The model estimates a cycle from
    # causal native forward observations excluding prefill plus measured head cost.
    reference=float(np.mean([max(o['host_forward_s'],o['device_elapsed_s']) for o in observations]))
    if not math.isfinite(reference) or reference<=0: raise ValueError('Invalid cycle observations')
    measured=next(e for e in BINS if raw['last_canonical_context_length']<=e)
    result=dict(schema='bank_v02_cost_profile_v1',head_increment_s=increments,bins={},
        source='qualification native forwards + controlled 0/1/2 head paths',
        scope='cycle proxy; prefill excluded; unmeasured bins share initial observed profile',
        raw_profile=copy.deepcopy(raw), initial_observations=copy.deepcopy(observations))
    for edge in BINS:
        result['bins'][str(edge)]=dict(cycle_s={str(k):reference+increments[str(k)] for k in paths},
            base_s=reference,observations=0,measured_context_bin=measured,
            extrapolated=edge!=measured)
    return result


def observe(profile, observations, active_modules):
    """Only the real canonical response calls this, after its paired scores."""
    if not profile or not observations:return
    for row in observations:
        edge=next((b for b in BINS if row['context_length']<=b),BINS[-1])
        entry=profile['bins'][str(edge)]
        cost=max(float(row['host_forward_s']),float(row['device_elapsed_s']))
        baseline=max(1e-9,cost-profile['head_increment_s'][str(active_modules)])
        count=entry['observations']
        entry['base_s']=(entry['base_s']*count+baseline)/(count+1)
        entry['observations']=count+1;entry['extrapolated']=False
        entry['cycle_s']={str(k):entry['base_s']+profile['head_increment_s'][str(k)] for k in (0,1,2)}
